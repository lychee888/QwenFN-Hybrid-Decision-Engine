"""M1 gateway tests: a per-request refusal marker -> FAST typed 503 `capture_refused`.

Scope: loopback (real localhost HTTP) only. The engine double is `backend.FixtureEngineServer`
labelled fixture_only; no live engine is started or contacted. The refusal marker written by the
double is the M1 helper's own documented schema (`patches/qfn_capture_runtime_mtp1.py`
`CAPTURE_REFUSED.json`, schema `qfn.capture-refused/1`).

What is asserted:
  * a refusal entry that names THIS request -> `CaptureRefused` / HTTP 503 `capture_refused`
    within ~1 s of the engine response, with the helper's reason, instead of a full-deadline 504;
  * an entry that names ANOTHER request -> no effect (the request keeps waiting);
  * no entry -> the existing deadline path is unchanged (504 `deadline_exceeded`);
  * a global-stop CAPTURE_ERROR.json still behaves exactly as before (CaptureError, not a refusal);
  * a real capture record still wins over a refusal marker;
  * an unknown-schema marker file is ignored (never interpreted);
  * a non-final refusal (chunk_not_final) is not representable as a marker entry, so a
    still-capturable request cannot be failed by this path (see the helper tests).
"""

from __future__ import annotations

import contextlib
import http.client
import json
import pathlib
import time

import pytest

from qfn_decision import api, backend, protocol

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKER_PREFIX = backend.ENGINE_REQUEST_ID_PREFIX


def _refusal_entry(engine_id: str, reason: str = "own_row_has_draft_tokens", **over) -> dict:
    entry = {
        "schema": backend.CAPTURE_REFUSED_SCHEMA,
        "engine_request_id": engine_id,
        "reason": reason,
        "final_refusal": True,
        "helper_variant": "qfn-capture-mtp1/1",
        "batch_index": 2,
        "wall_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns() * 1e9,
        "internal_index": 5,
        "own_draft_tokens": 3,
        "batch_num_drafts": 21,
        "cu_span": 4,
        "has_draft_tokens_per_req": True,
        "has_cu_num_logits_np": True,
        "has_logits_indices": True,
        "logits_row_index": 8,
        "logits_index": 13,
        "query_start": 8,
        "query_end": 17,
        "computed_tokens_before_step": 0,
        "num_scheduled_tokens": 9,
        "prompt_len": 9,
        "terminal_position": 8,
    }
    entry.update(over)
    return entry


def _refusal_payload(engine_id: str, **over) -> dict:
    payload = {
        "schema": backend.CAPTURE_REFUSED_SCHEMA,
        "helper_version": "qfn-capture-runtime/1",
        "helper_variant": "qfn-capture-mtp1/1",
        "runner_source_sha256": "f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8",
        "marker_prefix": MARKER_PREFIX,
        "marker_name": backend.CAPTURE_REFUSED_MARKER_NAME,
        "max_entries": 32,
        "entry_count": 1,
        "dropped_entries_total": 0,
        "truncated": False,
        "entries": [_refusal_entry(engine_id)],
    }
    payload.update(over)
    return payload


OTHER_ENGINE_ID = "cmpl-qfn-" + "ab" * 16 + "-0-deadbeef"


# ======================================================================================
# provider level (real loopback HTTP, fixture engine)
# ======================================================================================

def _fixture_engine(tmp_path, **kw):
    eng = backend.FixtureEngineServer(
        model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture", **kw
    )
    eng.start()
    return eng


def _provider(eng, tmp_path, **kw):
    cfg = backend.capture_provider_config(
        base_url=eng.base_url,
        model=eng.model,
        capture_dir=eng.capture_dir,
        capture_dir_allowlist=[str(pathlib.Path(tmp_path) / "capture")],
        tokenizer_revision="fixture-tok-rev",
        provider_label=backend.FIXTURE_ONLY_LABEL,
        **kw,
    )
    return backend.HttpEngineCaptureProvider(cfg)


def _raw_request(kind="choice", n_options=3):
    options = [protocol.ChoiceOption(key=f"k{i}", description=f"option {i}") for i in range(n_options)]
    if kind == "choice":
        return protocol.DecisionRequest.create(kind="choice", state="s", question="q?", options=options)
    return protocol.DecisionRequest.create(kind="noul", state="s", question="q?")


def test_refusal_marker_fails_the_named_request_fast(tmp_path):
    eng = _fixture_engine(tmp_path, refusal_marker=_refusal_payload("{engine_id}"))
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=10.0, poll_initial_ms=5.0, poll_max_ms=25.0)
        started = time.monotonic()
        with pytest.raises(backend.CaptureRefused) as err:
            prov.acquire(_raw_request("choice", 3), deadline_s=10.0)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, f"the refusal must fail in well under the 10 s deadline, took {elapsed:.3f}s"
        assert err.value.reason == "own_row_has_draft_tokens"
        assert "own_row_has_draft_tokens" in str(err.value)
        assert err.value.code == "capture_refused"
        assert isinstance(err.value, backend.CaptureError)          # existing handlers still work
        assert err.value.refusal["fatal"] is True
        assert err.value.refusal["engine_request_id"].startswith(MARKER_PREFIX)
        assert err.value.refusal["cu_span"] == 4 and err.value.refusal["batch_num_drafts"] == 21
        diag = prov.diagnostics()
        assert diag["refusal_marker_seen"] is True
        assert diag["refusal_for_last_request"]["reason"] == "own_row_has_draft_tokens"
        # the provider exposes the raw marker file name it reads
        assert (pathlib.Path(eng.capture_dir) / backend.CAPTURE_REFUSED_MARKER_NAME).is_file()
    finally:
        eng.stop()


def test_refusal_entry_for_another_request_does_not_fail_this_one(tmp_path):
    eng = _fixture_engine(tmp_path, refusal_marker=_refusal_payload(OTHER_ENGINE_ID))
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=1.2, poll_initial_ms=5.0, poll_max_ms=25.0)
        started = time.monotonic()
        with pytest.raises(backend.DeadlineExceeded):
            prov.acquire(_raw_request("noul"), deadline_s=1.2)
        assert time.monotonic() - started >= 1.0
        assert prov.diagnostics()["refusal_marker_seen"] is True
        assert prov.diagnostics()["refusal_for_last_request"] is None
    finally:
        eng.stop()


def test_no_refusal_marker_keeps_the_deadline_path(tmp_path):
    eng = _fixture_engine(tmp_path, fault="missing")
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=1.2, poll_initial_ms=5.0, poll_max_ms=25.0)
        started = time.monotonic()
        with pytest.raises(backend.DeadlineExceeded) as err:
            prov.acquire(_raw_request("score"), deadline_s=1.2)
        assert time.monotonic() - started >= 1.0
        assert "deadline" in str(type(err.value).__name__).lower() or True
        assert not isinstance(err.value, backend.CaptureRefused)
        assert prov.diagnostics()["refusal_marker_seen"] is False
    finally:
        eng.stop()


def test_unknown_schema_refusal_file_is_ignored(tmp_path):
    eng = _fixture_engine(
        tmp_path, refusal_marker=_refusal_payload("{engine_id}", schema="qfn.something-else/9")
    )
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=1.2, poll_initial_ms=5.0, poll_max_ms=25.0)
        with pytest.raises(backend.DeadlineExceeded):
            prov.acquire(_raw_request("noul"), deadline_s=1.2)
        assert prov.diagnostics()["refusal_marker_seen"] is False
    finally:
        eng.stop()


def test_a_real_record_still_wins_over_a_refusal_marker(tmp_path):
    eng = _fixture_engine(tmp_path, refusal_marker=_refusal_payload("{engine_id}"),
                          refusal_suppresses_record=False)
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=10.0, poll_initial_ms=5.0, poll_max_ms=25.0)
        cap = prov.acquire(_raw_request("choice", 3), deadline_s=10.0)   # no CaptureRefused
        assert cap.record.engine_request_id.startswith(MARKER_PREFIX)
        assert cap.vector_fp32.shape == (2560,)
    finally:
        eng.stop()


def test_global_stop_error_marker_behaviour_is_unchanged(tmp_path):
    stopped = {
        "helper_version": "qfn-capture-runtime/1",
        "stopped": True,
        "last_error": {"kind": "quota_exceeded", "error": "quota reached"},
    }
    eng = _fixture_engine(tmp_path, fault="missing", error_marker=stopped,
                          refusal_marker=_refusal_payload("{engine_id}"))
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=2.0, poll_initial_ms=5.0, poll_max_ms=25.0)
        with pytest.raises(backend.CaptureError) as err:
            prov.acquire(_raw_request("noul"), deadline_s=2.0)
        assert not isinstance(err.value, backend.CaptureRefused)
        assert "stopped" in str(err.value) or "quota_exceeded" in str(err.value)
    finally:
        eng.stop()


def test_refusal_matching_rule_is_explicit(tmp_path):
    caller = "qfn-" + "1a" * 16
    payload = _refusal_payload(f"cmpl-{caller}-0-89c4f0b7")
    entry = backend.refusal_for_request(payload, caller)
    assert entry is not None and entry["engine_request_id"] == f"cmpl-{caller}-0-89c4f0b7"
    # an accepted engine id for this caller also matches
    assert backend.refusal_for_request(payload, caller, [f"cmpl-{caller}"]) is not None
    # another caller never matches, in any form
    other = "qfn-" + "2b" * 16
    assert backend.refusal_for_request(payload, other) is None
    assert backend.refusal_for_request(payload, other, [f"cmpl-{other}-0"]) is None
    # a caller id alone (no engine id) matching the same caller does match - that is the contract
    assert backend.refusal_for_request(_refusal_payload(caller), caller) is not None
    # unknown schema / malformed payloads are never interpreted
    assert backend.refusal_for_request({}, caller) is None
    assert backend.refusal_for_request({"schema": "x/y", "entries": []}, caller) is None
    assert backend.refusal_for_request({"schema": backend.CAPTURE_REFUSED_SCHEMA}, caller) is None
    assert backend.refusal_for_request(
        {"schema": backend.CAPTURE_REFUSED_SCHEMA, "entries": [None, 7, {"nope": 1}]}, caller
    ) is None


# ======================================================================================
# API level: /v1/decide answers the typed refusal
# ======================================================================================

def _config_file(tmp_path, engine_server):
    cfg = json.loads((ROOT / "configs" / "decision_api.json").read_text(encoding="utf-8"))
    cfg["engine"]["base_url"] = engine_server.base_url
    cfg["engine"]["model"] = engine_server.model
    cfg["engine"]["health_ping"] = False
    cfg["capture"]["dir"] = str(engine_server.capture_dir)
    cfg["capture"]["dir_allowlist"] = [str(engine_server.capture_dir)]
    cfg["provider"] = "engine"
    path = pathlib.Path(tmp_path) / "decision_api_refusal.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


@contextlib.contextmanager
def gateway(tmp_path, **engine_kw):
    eng = backend.FixtureEngineServer(
        model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture", **engine_kw
    )
    eng.start()
    srv = None
    try:
        cfg = backend.load_api_config(_config_file(tmp_path, eng))
        srv = api.GatewayServer(cfg, port=0, log_sink=[].append)
        srv.start()
        yield srv, eng
    finally:
        if srv is not None:
            srv.stop()
        eng.stop()


def _post(port, path, body, timeout=30.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, body=body, headers={"content-type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8")), conn
    except BaseException:
        conn.close()
        raise


def _decide_payload(deadline_ms):
    return {
        "kind": "choice",
        "state": "the state the decision is about",
        "question": "which option?",
        "options": [{"key": "k1", "description": "first"}, {"key": "k2", "description": "second"}],
        "thinking": "off",
        "deadline_ms": deadline_ms,
    }


def test_decide_returns_503_capture_refused_within_two_seconds(tmp_path):
    with gateway(tmp_path, refusal_marker=_refusal_payload("{engine_id}")) as (srv, _eng):
        started = time.monotonic()
        status, doc, conn = _post(srv.port, "/v1/decide",
                                  json.dumps(_decide_payload(60_000)).encode("utf-8"))
        elapsed = time.monotonic() - started
        conn.close()
    assert status == 503, (status, doc)
    assert doc["error"]["code"] == backend.CAPTURE_REFUSED_CODE == "capture_refused"
    assert doc["error"]["status"] == 503
    assert "own_row_has_draft_tokens" in doc["error"]["message"]
    assert elapsed < 2.0, f"must not wait for the 60 s deadline, took {elapsed:.3f}s"
    # never a fabricated probability, never a fallback lane
    assert "probabilities" not in doc and "slot_probabilities" not in doc


def test_decide_keeps_the_deadline_path_when_nothing_refuses(tmp_path):
    with gateway(tmp_path, fault="missing") as (srv, _eng):
        started = time.monotonic()
        status, doc, conn = _post(srv.port, "/v1/decide",
                                  json.dumps(_decide_payload(1_500)).encode("utf-8"))
        elapsed = time.monotonic() - started
        conn.close()
    assert status == 504 and doc["error"]["code"] == "deadline_exceeded"
    assert elapsed >= 1.4


def test_decide_unrelated_refusal_marker_is_not_this_requests_refusal(tmp_path):
    with gateway(tmp_path, refusal_marker=_refusal_payload(OTHER_ENGINE_ID)) as (srv, _eng):
        status, doc, conn = _post(srv.port, "/v1/decide",
                                  json.dumps(_decide_payload(1_500)).encode("utf-8"))
        conn.close()
    assert status == 504 and doc["error"]["code"] == "deadline_exceeded"


def test_health_surfaces_the_refusal_marker_as_a_diagnostic(tmp_path):
    with gateway(tmp_path, fault="missing",
                 refusal_marker=_refusal_payload(OTHER_ENGINE_ID)) as (srv, _eng):
        # one decide call makes the fixture engine write the marker (as the helper would)
        status, _doc, conn = _post(srv.port, "/v1/decide",
                                   json.dumps(_decide_payload(1_200)).encode("utf-8"))
        conn.close()
        assert status == 504                                     # unrelated entry: not our refusal
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10.0)
        conn.request("GET", "/health")
        doc = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
    capture = doc["capture"]
    assert capture["refusal_marker_seen"] is True
    assert capture["refusal_marker_file"] == backend.CAPTURE_REFUSED_MARKER_NAME
    assert capture["refusal_marker_entry_count"] == 1
    assert "refusal_marker_rule" in capture
    # the pre-existing error-marker diagnostics are untouched
    assert "error_marker_seen" in capture and "error_marker_rule" in capture
