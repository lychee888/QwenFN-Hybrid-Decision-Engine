"""WIDE1-R2 blocker 2: the requested deadline must bound the WHOLE wide acquisition.

Every test here is real loopback HTTP against a purpose-built slow engine double: it can delay
``/tokenize``, delay or drip the completion response, withhold the capture record, or publish the
record late. That is how the reviewer reproduced the defect (100 ms requested -> HTTP 200 after
2 to 3 s), so the regression is measured the same way: wall-clock elapsed plus HTTP status plus
the engine-side call counts.

No GPU, no real engine, no inference. The doubles are explicitly synthetic.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import pathlib
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from qfn_decision import api, backend, features, protocol, wide

ROOT = pathlib.Path(os.environ.get("QFN_WIDE_TEST_ROOT", "") or pathlib.Path(__file__).resolve().parents[1])
CATALOG_PATH = ROOT / "configs" / "wide-label-catalog.json"
CATALOG = wide.WideCatalog.from_file(CATALOG_PATH)
UUID32 = "0123456789abcdef0123456789abcdef"


def _wide_block(**over) -> dict:
    block = {
        "enabled": True,
        "catalog_path": str(CATALOG_PATH),
        "catalog_file_sha256": wide.PINNED_CATALOG_FILE_SHA256,
        "catalog_digest": wide.PINNED_CATALOG_DIGEST,
        "tokenizer_revision": wide.PINNED_CATALOG_TOKENIZER_REVISION,
        "template_digest": wide.WIDE_TEMPLATE_CONTENT_SHA256,
        "max_options": 256,
    }
    block.update(over)
    return block


# =====================================================================================
# slow engine double
# =====================================================================================

class SlowEngine:
    """A synthetic engine whose /tokenize and /v1/completions can be made slow on demand."""

    def __init__(
        self,
        capture_dir: pathlib.Path,
        *,
        tokenize_delay: float = 0.0,
        completion_delay: float = 0.0,
        drip_chunks: int = 0,
        drip_delay: float = 0.0,
        write_record: bool = True,
        record_delay: float = 0.0,
        slow_state_marker: str | None = None,
        slow_state_delay: float = 0.0,
    ) -> None:
        self.capture_dir = pathlib.Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.tokenize_delay = float(tokenize_delay)
        self.completion_delay = float(completion_delay)
        self.drip_chunks = int(drip_chunks)
        self.drip_delay = float(drip_delay)
        self.write_record = bool(write_record)
        self.record_delay = float(record_delay)
        self.slow_state_marker = slow_state_marker
        self.slow_state_delay = float(slow_state_delay)
        self.tokenize_calls = 0
        self.completion_calls = 0
        self.model = "slow-fixture-model"
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._late: list[threading.Thread] = []

    # -- lifecycle
    @property
    def port(self) -> int:
        assert self._httpd is not None
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "SlowEngine":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                return

            def _body(self) -> bytes:
                length = self.headers.get("Content-Length")
                return self.rfile.read(int(length)) if length else b""

            def _json(self, payload, status=200) -> None:
                raw = json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # the client aborted at its deadline: an expected outcome in these tests
                    pass

            def do_POST(self):  # noqa: N802
                path = self.path.split("?")[0]
                raw = self._body()
                if path == "/tokenize":
                    return self._tokenize(raw)
                if path == "/v1/completions":
                    return self._completion(raw)
                if path == "/health":
                    return self._json({"status": "ok"})
                return self._json({"error": {"message": "no route"}}, status=404)

            def do_GET(self):  # noqa: N802
                return self._json({"status": "ok"})

            def _tokenize(self, raw: bytes) -> None:
                doc = json.loads(raw or b"{}")
                with owner._lock:
                    owner.tokenize_calls += 1
                delay = owner.tokenize_delay
                if delay:
                    time.sleep(delay)
                ids = backend._fixture_tokenize_messages(
                    doc.get("messages") or [], add_generation_prompt=True)
                if owner.slow_state_marker:
                    text = json.dumps(doc.get("messages") or doc.get("prompt") or "")
                    if owner.slow_state_marker in text:
                        # the marker id travels with the prompt so /v1/completions can recognise it
                        ids = [424242, *ids]
                self._json({"count": len(ids), "tokens": ids, "max_model_len": 900000,
                            "token_strs": [f"<{i}>" for i in ids]})

            def _completion(self, raw: bytes) -> None:
                doc = json.loads(raw or b"{}")
                with owner._lock:
                    owner.completion_calls += 1
                ids = [int(v) for v in (doc.get("prompt") or [])]
                request_id = doc.get("request_id") or ""
                delay = owner.completion_delay
                if owner.slow_state_marker:
                    marker_id = 424242
                    if marker_id in ids:
                        delay = owner.slow_state_delay
                if owner.write_record:
                    # the capture file exists BEFORE the response is delayed, which is what makes a
                    # late/preexisting record test meaningful
                    owner._publish_record(request_id, ids)
                if delay:
                    time.sleep(delay)
                payload = {
                    "id": f"cmpl-{request_id}",
                    "object": "text_completion",
                    "model": owner.model,
                    "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": 1,
                              "total_tokens": len(ids) + 1},
                }
                if owner.drip_chunks:
                    self._drip(payload)
                    return
                self._json(payload)

            def _drip(self, payload: dict) -> None:
                raw = json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    chunk = max(1, len(raw) // max(1, owner.drip_chunks))
                    for i in range(0, len(raw), chunk):
                        self.wfile.write(raw[i:i + chunk])
                        self.wfile.flush()
                        time.sleep(owner.drip_delay)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        httpd.daemon_threads = True
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="slow-engine", daemon=True)
        self._thread.start()
        return self

    def _publish_record(self, caller_request_id: str, ids: list[int]) -> None:
        """Write the wide capture record this request would have produced (synthetic)."""
        try:
            n_options, request_uuid = wide.wide_request_identity_from_caller_id(caller_request_id)
        except wide.WideError:
            return
        engine_id = backend.engine_request_id_for(caller_request_id, 0)
        prompt_hash = features.token_sequence_sha256(ids)
        base = backend.make_fixture_record(
            engine_request_id=engine_id,
            prompt_ids=ids,
            vector_fp32=np.linspace(-0.5, 0.5, 2560, dtype=np.float32),
            slot_logits=np.linspace(-1.0, 1.0, protocol.NUM_SLOTS, dtype=np.float32),
            prompt_hash=prompt_hash,
        )
        logits = np.linspace(-2.0, 2.0, CATALOG.n_labels, dtype=np.float32)
        logits[3] = 12.0
        blob = wide.add_wide_extension(
            backend.fixture_only_npz_bytes(base),
            catalog=CATALOG,
            template_digest=wide.WIDE_TEMPLATE_CONTENT_SHA256,
            wide_request_uuid=request_uuid,
            prompt_token_sha256=prompt_hash,
            prompt_len=len(ids),
            n_options=n_options,
            logits256=logits,
        )
        name = backend.expected_record_file_name(engine_id, prompt_hash, 0)
        target = self.capture_dir / name

        def write_later() -> None:
            time.sleep(self.record_delay)
            try:
                target.write_bytes(blob)
            except OSError:
                pass

        if self.record_delay > 0:
            thread = threading.Thread(target=write_later, daemon=True, name="late-record")
            self._late.append(thread)
            thread.start()
        else:
            target.write_bytes(blob)

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd, self._thread = None, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=10.0)

    def __enter__(self) -> "SlowEngine":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# =====================================================================================
# gateway scaffolding
# =====================================================================================

def _config(tmp_path: pathlib.Path, engine_server: SlowEngine, **over) -> backend.ApiConfig:
    cfg = json.loads((ROOT / "configs" / "decision_api.json").read_text(encoding="utf-8"))
    cfg["engine"]["base_url"] = engine_server.base_url
    cfg["engine"]["model"] = engine_server.model
    cfg["engine"]["health_ping"] = False
    cfg["capture"]["dir"] = str(engine_server.capture_dir)
    cfg["capture"]["dir_allowlist"] = [str(engine_server.capture_dir)]
    cfg["provider"] = "engine"
    cfg["wide_choice"] = _wide_block()
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return backend.ApiConfig.from_dict(cfg)


@contextlib.contextmanager
def gateway(tmp_path, engine_server: SlowEngine, **over):
    srv = api.GatewayServer(_config(tmp_path, engine_server, **over), port=0, log_sink=[].append)
    srv.start()
    try:
        yield srv, engine_server
    finally:
        srv.stop()


def _options(n: int) -> list[dict]:
    return [{"key": f"k{i}", "description": f"description {i}"} for i in range(n)]


def _post(port: int, path: str, body: dict, timeout: float = 60.0) -> tuple[int, dict, float]:
    started = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
    finally:
        conn.close()
    elapsed = time.monotonic() - started
    try:
        doc = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        doc = {"_raw": raw[:200].decode("utf-8", "replace")}
    return status, doc, elapsed


def _decide(n: int = 17, **over) -> dict:
    body = {"kind": "choice", "state": "state text", "question": "which?", "options": _options(n)}
    body.update(over)
    return body


# =====================================================================================
# RED 1: slow tokenizer must not produce a success after the deadline
# =====================================================================================

def test_slow_tokenizer_respects_the_request_deadline(tmp_path):
    engine = SlowEngine(tmp_path / "cap", tokenize_delay=2.0).start()
    try:
        with gateway(tmp_path, engine) as (srv, eng):
            status, doc, elapsed = _post(srv.port, "/v1/decide", _decide(17, deadline_ms=100))
    finally:
        engine.stop()
    assert status == 504, (status, doc, elapsed)
    assert doc["error"]["code"] == "deadline_exceeded"
    assert elapsed < 0.75, f"deadline not honoured: {elapsed:.3f}s"
    assert eng.completion_calls == 0, "no completion may be sent once tokenization exhausted the budget"


# =====================================================================================
# the rest of the deadline contract
# =====================================================================================

def test_slow_completion_respects_the_request_deadline(tmp_path):
    engine = SlowEngine(tmp_path / "cap", completion_delay=3.0).start()
    try:
        with gateway(tmp_path, engine) as (srv, eng):
            status, doc, elapsed = _post(srv.port, "/v1/decide", _decide(17, deadline_ms=100))
    finally:
        engine.stop()
    assert status == 504, (status, doc, elapsed)
    assert doc["error"]["code"] == "deadline_exceeded"
    assert "probabilities" not in doc
    assert elapsed < 0.75, f"deadline not honoured: {elapsed:.3f}s"
    assert eng.tokenize_calls == 1 and eng.completion_calls == 1


def test_dripping_response_cannot_extend_the_absolute_deadline(tmp_path):
    # headers arrive immediately, then 40 small body chunks 100 ms apart: a per-recv socket
    # timeout is reset each time, so only an absolute deadline bounds this.
    engine = SlowEngine(tmp_path / "cap", drip_chunks=40, drip_delay=0.1).start()
    try:
        with gateway(tmp_path, engine) as (srv, eng):
            status, doc, elapsed = _post(srv.port, "/v1/decide", _decide(17, deadline_ms=150))
    finally:
        engine.stop()
    assert status == 504, (status, doc, elapsed)
    assert doc["error"]["code"] == "deadline_exceeded"
    assert elapsed < 1.0, f"a dripping body extended the deadline: {elapsed:.3f}s"


def test_a_timely_wide_request_is_unchanged(tmp_path):
    engine = SlowEngine(tmp_path / "cap").start()
    try:
        with gateway(tmp_path, engine) as (srv, eng):
            status, doc, elapsed = _post(srv.port, "/v1/decide", _decide(17, deadline_ms=5000))
    finally:
        engine.stop()
    assert status == 200, (status, doc, elapsed)
    assert doc["schema"] == "qfn.decide.wide/1"
    assert len(doc["probabilities"]) == 17
    assert abs(sum(doc["probabilities"].values()) - 1.0) < 1e-9
    assert doc["selected"] == "k3"
    assert elapsed < 2.0


def test_concurrent_long_and_short_budgets_on_one_provider_do_not_interfere(tmp_path):
    # marked requests ("SLOWSTATE") get a slow completion; the marker travels as token id 424242
    engine = SlowEngine(tmp_path / "cap", slow_state_marker="SLOWSTATE",
                        slow_state_delay=1.2).start()
    results = {}
    try:
        with gateway(tmp_path, engine) as (srv, eng):
            baseline_threads = threading.active_count()

            def call(label, body):
                results[label] = _post(srv.port, "/v1/decide", body)

            slow_long = threading.Thread(
                target=call, name="long-budget",
                args=("long", _decide(17, state="SLOWSTATE long", deadline_ms=8000)))
            slow_short = threading.Thread(
                target=call, name="short-budget",
                args=("short", _decide(17, state="SLOWSTATE short", deadline_ms=120)))
            fast = threading.Thread(
                target=call, name="fast", args=("fast", _decide(17, deadline_ms=5000)))
            slow_long.start()
            slow_short.start()
            fast.start()
            for t in (slow_long, slow_short, fast):
                t.join(30.0)

            status_long, doc_long, elapsed_long = results["long"]
            status_short, doc_short, elapsed_short = results["short"]
            status_fast, doc_fast, _ = results["fast"]
            permit_value = srv.semaphore._value

        assert status_long == 200, (status_long, doc_long, elapsed_long)
        assert elapsed_long >= 1.0, "the long request really did wait for the slow engine"
        assert status_short == 504, (status_short, doc_short, elapsed_short)
        assert elapsed_short < 0.75, f"short budget not honoured: {elapsed_short:.3f}s"
        assert status_fast == 200 and len(doc_fast["probabilities"]) == 17
        # the short request must not have cancelled the long one, and vice versa
        assert doc_long["schema"] == "qfn.decide.wide/1"
        # permits are back: nothing leaked while the slow worker was unwinding
        assert permit_value == 8, permit_value
        # no watchdog/timer thread survives the requests
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not [t for t in threading.enumerate() if isinstance(t, threading.Timer)]:
                break
            time.sleep(0.02)
        assert not [t for t in threading.enumerate() if isinstance(t, threading.Timer)], (
            "a deadline watchdog leaked")
        assert threading.active_count() <= baseline_threads + 1
    finally:
        engine.stop()


def test_budget_is_shared_across_phases_and_the_capture_poll(tmp_path):
    # 300 ms of completion + a missing record under a 500 ms budget: the poll must only get the
    # ~200 ms that is left, so the whole request ends at ~0.5 s. If the poll got a FRESH 500 ms
    # (or the completion were unbounded) this would take ~0.8 s or more.
    engine = SlowEngine(tmp_path / "cap", completion_delay=0.3, write_record=False).start()
    try:
        with gateway(tmp_path, engine) as (srv, eng):
            status, doc, elapsed = _post(srv.port, "/v1/decide", _decide(17, deadline_ms=500))
    finally:
        engine.stop()
    assert status == 504, (status, doc, elapsed)
    assert doc["error"]["code"] == "deadline_exceeded"
    assert elapsed < 0.65, f"one shared budget expected (~0.5 s), got {elapsed:.3f}s"
    assert eng.completion_calls == 1


def test_a_late_or_preexisting_record_is_never_accepted_after_the_deadline(tmp_path):
    # the record is written BEFORE the slow response returns, so the capture file exists while the
    # request is already past its deadline: it must still be a 504, never a 200.
    engine = SlowEngine(tmp_path / "cap", completion_delay=0.8, write_record=True).start()
    try:
        with gateway(tmp_path, engine) as (srv, eng):
            status, doc, elapsed = _post(srv.port, "/v1/decide", _decide(17, deadline_ms=100))
            records = list(pathlib.Path(eng.capture_dir).glob("*.npz"))
    finally:
        engine.stop()
    assert status == 504, (status, doc, elapsed)
    assert "probabilities" not in doc
    assert elapsed < 0.75
    assert records, "the synthetic record really was on disk before the deadline check"


def test_socket_timeout_shorter_than_the_budget_stays_a_transport_error(tmp_path):
    engine = SlowEngine(tmp_path / "cap", completion_delay=2.0).start()
    try:
        with gateway(tmp_path, engine, engine={"socket_timeout_s": 0.1}) as (srv, eng):
            status, doc, elapsed = _post(srv.port, "/v1/decide", _decide(17, deadline_ms=5000))
    finally:
        engine.stop()
    assert status == 502, (status, doc, elapsed)
    assert doc["error"]["code"] == "backend_error"
    assert "deadline" not in json.dumps(doc["error"]).lower()
    assert elapsed < 1.5, elapsed
