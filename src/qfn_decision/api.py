"""S6 same-engine typed gateway: bounded HTTP service for typed S1 decisions + S2 proxying.

Routes
------
* ``POST /v1/decide``            typed S1 readout (B0 or H1) over the shared S1 protocol.
* ``POST /v1/chat/completions``  ordinary generation, forwarded unchanged to the SAME engine.
* ``POST /v1/completions``       ordinary generation, forwarded unchanged to the SAME engine.
* ``GET  /health``               readiness only; it never claims a validated live decision path.

Design rules enforced here
--------------------------
* Strict, bounded JSON parsing: exact ``Content-Length`` required, chunked bodies refused, unknown
  top-level/sub-fields refused, duplicate JSON keys refused, ``NaN``/``Infinity`` literals refused,
  oversized bodies answered ``413``.
* Bounded work: a fixed worker pool plus a non-blocking inflight semaphore; a request that cannot
  get a slot is answered ``429`` instead of growing threads.
* Status mapping: ``400``/``411``/``413`` parse problems, ``422`` validation/arm problems and a
  template surface that has no form for the requested kind (``template_kind_unsupported``),
  ``429`` inflight bound, ``502`` backend/transport/capture failure, ``503`` backend unavailable,
  ``504`` capture deadline. A failure never returns partial probabilities.
* Ordinary generation keeps the caller's JSON body, status, headers and streaming chunks exactly as
  the engine produced them; hop-by-hop headers are never forwarded and ``Content-Length`` is set
  from the actual bytes, never double-counted. Engine-side caps stay engine-owned.
* Prompt text, generated text and credential values are never logged or returned.

Scope: the B0 arm with the ``qwen_chat_letters-v1`` template has run against the real engine through
:mod:`qfn_decision.hybrid`. Other arms, templates and providers are covered by fixture tests only.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import itertools
import json
import os
import pathlib
import queue
import select
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

from . import backend, decision_adapter, jev_compat, protocol, templates, wide

__all__ = [
    "RequestError", "parse_json_object", "build_decide_request", "build_wide_decide_request",
    "GatewayServer", "GatewayRequestHandler", "main", "DECIDE_SCHEMA", "PROXY_ROUTES",
    "DECISION_ROUTES",
]

DECIDE_SCHEMA = "qfn.decide/1"
PROXY_ROUTES = ("/v1/chat/completions", "/v1/completions")
# typed decision routes: the native envelope and the Jev-compatible translation of it
DECISION_ROUTES = ("/v1/decide", jev_compat.SYSTEMONE_PATH)

ALLOWED_DECIDE_FIELDS = {
    "kind", "state", "question", "options", "score_rubrics", "rubrics", "thinking", "arm",
    "seed", "deadline_ms",
}
ALLOWED_OPTION_FIELDS = {"key", "description"}
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade",
}
MAX_DEADLINE_MS = 3_600_000
STREAM_POLL_S = 0.2
SERVE_POLL_S = 0.05          # serve_forever's own poll interval (its signature default is 0.5s)
STREAM_QUEUE_MAXSIZE = 8     # bounded reader->connection event queue (backpressure, never unbounded)
STREAM_READ_SIZE = 4096      # bounded engine read per decoded chunk
STREAM_JOIN_S = 0.6          # bounded join window while cancelling a reader (3 poll intervals)


class RequestError(Exception):
    """An HTTP-level rejection. ``status`` is the response code, ``code`` a stable label.

    ``extra`` carries optional, non-secret detail into the error body (a field path, the limit that
    applied, the named question). It is empty for every pre-existing error path, so those bodies are
    byte-identical to what they were before.
    """

    def __init__(self, status: int, code: str, message: str, extra: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra: dict[str, Any] = dict(extra or {})


# ======================================================================================
# strict bounded JSON parsing
# ======================================================================================

class _DuplicateKey(ValueError):
    pass


class _NonFiniteLiteral(ValueError):
    pass


def _no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise _NonFiniteLiteral(name)


def parse_json_object(raw: bytes) -> dict:
    """Parse a request body as a JSON object, refusing duplicates and non-finite literals."""
    if not isinstance(raw, (bytes, bytearray)):
        raise RequestError(400, "invalid_body", "request body must be bytes")
    if not raw.strip():
        raise RequestError(400, "empty_body", "request body must be a non-empty JSON object")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestError(400, "invalid_utf8", f"request body is not valid UTF-8: {exc}") from exc
    try:
        doc = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except _DuplicateKey as exc:
        raise RequestError(400, "duplicate_key", f"duplicate JSON key {exc} is not allowed") from exc
    except _NonFiniteLiteral as exc:
        raise RequestError(
            400, "nonfinite_number", f"the JSON literal {exc} is not allowed; use finite numbers only"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RequestError(400, "invalid_json", f"request body is not valid JSON: {exc.msg}") from exc
    if not isinstance(doc, dict):
        raise RequestError(400, "not_an_object", "request body must be a JSON object")
    return doc


def _require_int_field(payload: Mapping[str, Any], name: str, minimum: int, maximum: int) -> int | None:
    if name not in payload:
        return None
    value = payload[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(400, "bad_field_type", f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum or value > maximum:
        raise RequestError(400, "bad_field_value", f"{name} must be in {minimum}..{maximum}, got {value}")
    return int(value)


def build_decide_request(payload: Mapping[str, Any], config: backend.ApiConfig) -> tuple[protocol.DecisionRequest, str, int | None, int | None]:
    """Validate the decide envelope and build the authoritative protocol request.

    Returns ``(request, arm, seed, deadline_ms)``. ``protocol.DecisionRequest.create`` stays the
    single source of truth for semantic validation, so the endpoint cannot drift from the shared
    protocol module.
    """
    if not isinstance(payload, Mapping):
        raise RequestError(400, "not_an_object", "request body must be a JSON object")
    unknown = sorted(set(payload) - ALLOWED_DECIDE_FIELDS)
    if unknown:
        raise RequestError(400, "unknown_field", f"unknown request field(s): {unknown}")

    thinking = payload.get("thinking", "off")
    if not isinstance(thinking, str):
        raise RequestError(400, "bad_field_type", f"thinking must be a string, got {type(thinking).__name__}")
    if thinking != "off":
        if thinking == "on":
            raise RequestError(
                422, "thinking_on_unsupported",
                "thinking='on' is not supported: it would need a real S2 escalation path, which this "
                "gateway deliberately does not implement",
            )
        raise RequestError(422, "thinking_unsupported", f"thinking must be 'off', got {thinking!r}")

    arm = payload.get("arm", config.default_arm)
    if not isinstance(arm, str) or not arm:
        raise RequestError(400, "bad_field_type", f"arm must be a non-empty string, got {arm!r}")
    if arm == decision_adapter.ARM_NAME:
        # The experimental L1O arm is opt-in: it is never reachable through a config that does not
        # declare it, and a declared-but-disabled adapter says so explicitly instead of being
        # reported as an unknown arm.
        spec = getattr(config, "decision_adapter", None)
        if spec is None:
            raise RequestError(422, "arm_unsupported", f"arm must be B0 or H1, got {arm!r}")
        if not spec.enabled:
            raise RequestError(
                422, "adapter_disabled",
                "arm 'L1O' is the opt-in EXPERIMENTAL adapter arm and this gateway declares it but "
                "has it disabled (decision_adapter.enabled=false); it is never enabled implicitly "
                f"(configured alias {spec.model_alias!r})",
            )
    elif arm not in ("B0", "H1"):
        raise RequestError(422, "arm_unsupported", f"arm must be B0 or H1, got {arm!r}")

    seed = _require_int_field(payload, "seed", 0, 2**31 - 1)
    deadline_ms = _require_int_field(payload, "deadline_ms", 1, MAX_DEADLINE_MS)

    if "score_rubrics" in payload and "rubrics" in payload:
        raise RequestError(
            400, "ambiguous_field",
            "supply either score_rubrics or its documented alias rubrics, not both",
        )
    rubrics = payload.get("score_rubrics", payload.get("rubrics"))

    options_raw = payload.get("options")
    options: list[protocol.ChoiceOption] | None = None
    if options_raw is not None:
        if not isinstance(options_raw, list):
            raise RequestError(422, "protocol_error", "options must be a JSON array of {key, description} objects")
        options = []
        for i, item in enumerate(options_raw):
            if not isinstance(item, dict):
                raise RequestError(422, "protocol_error", f"options[{i}] must be an object with key and description")
            extra = sorted(set(item) - ALLOWED_OPTION_FIELDS)
            if extra:
                raise RequestError(400, "unknown_field", f"unknown field(s) in options[{i}]: {extra}")
            options.append(protocol.ChoiceOption(key=item.get("key"), description=item.get("description")))

    if rubrics is not None and not isinstance(rubrics, (list, tuple)):
        raise RequestError(422, "protocol_error", "score_rubrics must be a JSON array of six ordered descriptions")

    try:
        request = protocol.DecisionRequest.create(
            kind=payload.get("kind"),
            state=payload.get("state"),
            question=payload.get("question"),
            options=options,
            score_rubrics=list(rubrics) if rubrics is not None else None,
            limits=config.limits,
        )
    except protocol.ProtocolError as exc:
        raise RequestError(422, "protocol_error", f"invalid typed decision request: {exc}") from exc
    return request, arm, seed, deadline_ms


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ======================================================================================
# wide-label request routing (opt-in; a payload with >=17 options is never legacy)
# ======================================================================================

def wide_payload_is_wide(payload: Mapping[str, Any], config: backend.ApiConfig) -> bool:
    """True only when this payload belongs to the wide lane.

    The lane must be enabled explicitly and the payload must be a choice carrying at least 17
    options. Everything else (including every <=16 payload and a disabled lane) keeps the legacy
    path, including its exact legacy error messages.
    """
    wc = getattr(config, "wide_choice", None)
    if wc is None or not wc.enabled:
        return False
    if not isinstance(payload, Mapping) or payload.get("kind") != protocol.KIND_CHOICE:
        return False
    options = payload.get("options")
    return isinstance(options, list) and len(options) >= protocol.MAX_CHOICE_OPTIONS + 1


def build_wide_decide_request(
    payload: Mapping[str, Any], config: backend.ApiConfig
) -> tuple[wide.WideChoiceRequest, str, int | None, int | None]:
    """Validate a wide decide envelope into the wide request type (never a 24-slot request)."""
    if not isinstance(payload, Mapping):
        raise RequestError(400, "not_an_object", "request body must be a JSON object")
    unknown = sorted(set(payload) - ALLOWED_DECIDE_FIELDS)
    if unknown:
        raise RequestError(400, "unknown_field", f"unknown request field(s): {unknown}")

    thinking = payload.get("thinking", "off")
    if not isinstance(thinking, str):
        raise RequestError(400, "bad_field_type", f"thinking must be a string, got {type(thinking).__name__}")
    if thinking != "off":
        if thinking == "on":
            raise RequestError(
                422, "thinking_on_unsupported",
                "thinking='on' is not supported: it would need a real S2 escalation path, which this "
                "gateway deliberately does not implement",
            )
        raise RequestError(422, "thinking_unsupported", f"thinking must be 'off', got {thinking!r}")

    arm = payload.get("arm", config.default_arm)
    if not isinstance(arm, str) or not arm:
        raise RequestError(400, "bad_field_type", f"arm must be a non-empty string, got {arm!r}")
    if arm not in ("B0",):
        # the wide lane is a raw-logit readout: no trained head, no adapter, no native score
        raise RequestError(
            422, "arm_unsupported",
            f"arm must be B0 for a wide-label request, got {arm!r} (no trained arm or adapter "
            "supports 17..256 options)",
        )

    seed = _require_int_field(payload, "seed", 0, 2**31 - 1)
    deadline_ms = _require_int_field(payload, "deadline_ms", 1, MAX_DEADLINE_MS)

    if "score_rubrics" in payload and "rubrics" in payload:
        raise RequestError(400, "ambiguous_field",
                           "supply either score_rubrics or its documented alias rubrics, not both")
    if payload.get("score_rubrics") is not None or payload.get("rubrics") is not None:
        raise RequestError(
            422, "protocol_error",
            "a wide-label request is a choice only; it must not carry score_rubrics "
            "(there is no wide score surface)",
        )

    options_raw = payload.get("options")
    if not isinstance(options_raw, list):
        raise RequestError(422, "protocol_error",
                           "options must be a JSON array of {key, description} objects")
    wc = config.wide_choice
    n = len(options_raw)
    limit = int(wc.max_options) if wc is not None else wide.MAX_WIDE_CHOICE_OPTIONS
    if n > wide.MAX_WIDE_CHOICE_OPTIONS:
        raise RequestError(
            422, "choice_too_many_options",
            f"{n} options exceed the supported maximum of {wide.MAX_WIDE_CHOICE_OPTIONS}",
        )
    if n > limit:
        raise RequestError(
            422, "choice_too_many_options",
            f"{n} options exceed this gateway's configured wide maximum ({limit})",
        )
    options: list[protocol.ChoiceOption] = []
    for i, item in enumerate(options_raw):
        if not isinstance(item, dict):
            raise RequestError(422, "protocol_error",
                               f"options[{i}] must be an object with key and description")
        extra = sorted(set(item) - ALLOWED_OPTION_FIELDS)
        if extra:
            raise RequestError(400, "unknown_field", f"unknown field(s) in options[{i}]: {extra}")
        options.append(protocol.ChoiceOption(key=item.get("key"), description=item.get("description")))

    try:
        request = wide.WideChoiceRequest.create(
            payload.get("state"), payload.get("question"), options, limits=config.limits
        )
    except wide.WideError as exc:
        raise RequestError(422, "protocol_error", f"invalid wide decision request: {exc}") from exc
    return request, arm, seed, deadline_ms


# ======================================================================================
# bounded server: fixed worker pool + non-blocking inflight semaphore
# ======================================================================================

class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a *fixed* worker pool, so thread count cannot grow."""

    daemon_threads = True
    allow_reuse_address = True
    poll_interval = 0.05                      # snappy shutdown without busy-waiting

    def __init__(self, address, handler, *, max_workers: int, socket_timeout_s: float) -> None:
        self._work: "queue.Queue[Any]" = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._live: set[Any] = set()
        self._live_lock = threading.Lock()
        self._max_workers = max(1, int(max_workers))
        self._socket_timeout_s = float(socket_timeout_s)
        # Explicit shutdown signal: a proxied stream cannot rely on noticing that the framework
        # closed its client socket (on this platform a locally closed socket is not reported
        # readable by select), so in-flight streams poll this instead.
        self._stopping = threading.Event()
        super().__init__(address, handler)
        for i in range(self._max_workers):
            thread = threading.Thread(target=self._worker_loop, name=f"qfn-gateway-worker-{i}", daemon=True)
            thread.start()
            self._workers.append(thread)

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def begin_stop(self) -> None:
        """Tell every in-flight proxied stream to wind down now (idempotent)."""
        self._stopping.set()

    def server_bind(self) -> None:
        # skip socketserver's reverse-DNS lookup (socket.getfqdn), which can stall ~1s on Windows
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port

    def process_request(self, request, client_address) -> None:  # noqa: D102 - see BaseServer
        try:
            request.settimeout(self._socket_timeout_s)
        except OSError:
            pass
        with self._live_lock:
            self._live.add(request)
        self._work.put((request, client_address))

    def _worker_loop(self) -> None:
        while True:
            item = self._work.get()
            if item is None:
                return
            request, client_address = item
            try:
                self.finish_request(request, client_address)
            except Exception:  # noqa: BLE001 - a broken connection must not kill the worker
                self.handle_error(request, client_address)
            finally:
                with self._live_lock:
                    self._live.discard(request)
                self.shutdown_request(request)

    def server_close(self) -> None:
        self._stopping.set()
        for _ in self._workers:
            self._work.put(None)
        super().server_close()
        # close connections still parked on a keep-alive read so the workers can exit promptly
        with self._live_lock:
            live = list(self._live)
            self._live.clear()
        for sock in live:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        for thread in self._workers:
            thread.join(timeout=2.0)


class GatewayRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qfn-decision-gateway/0.1"
    sys_version = ""

    def setup(self) -> None:  # type: ignore[override]
        super().setup()
        self._body_fully_read = False

    # -- plumbing ----------------------------------------------------------------------
    @property
    def app(self) -> "GatewayServer":
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # never log raw request lines
        return

    def log_error(self, fmt: str, *args: Any) -> None:
        self.app.log(f"http-error {self.command} {self.path} {fmt % args}")

    def _send_bytes(self, status: int, raw: bytes, content_type: str, extra: Mapping[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, raw, "application/json; charset=utf-8")

    def _send_error(self, error: RequestError) -> None:
        body = {"error": {"code": error.code, "message": error.message, "status": error.status}}
        body["error"].update(error.extra)
        self._send_json(error.status, body)

    def _read_body(self, limit: int) -> bytes:
        self._body_fully_read = False
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding:
            codings = [c.strip().lower() for c in transfer_encoding.split(",") if c.strip()]
            # 'identity' is a no-op; anything else (notably chunked) is unsupported and must be
            # answered rather than guessed at, otherwise the body length is unknown.
            if [c for c in codings if c != "identity"]:
                raise RequestError(
                    400, "transfer_encoding_unsupported",
                    "chunked or otherwise transfer-encoded request bodies are not supported; send a "
                    "Content-Length and a plain body",
                )
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise RequestError(411, "missing_content_length", "Content-Length is required")
        try:
            length = int(length_header)
        except (TypeError, ValueError) as exc:
            raise RequestError(400, "bad_content_length", f"Content-Length {length_header!r} is not an integer") from exc
        if length < 0:
            raise RequestError(400, "bad_content_length", f"Content-Length {length} is negative")
        if length > limit:
            # Read and discard what the client already sent (bounded) before answering, so a
            # well-behaved client still sees this 413 instead of a connection reset.
            drain_cap = max(limit * 4, 65536)
            if length > drain_cap:
                raise RequestError(
                    413, "body_too_large",
                    f"request body of {length} bytes exceeds the {limit}-byte limit and is too large "
                    "to drain; the connection will be closed",
                )
            self._discard(length)
            raise RequestError(
                413, "body_too_large", f"request body of {length} bytes exceeds the {limit}-byte limit"
            )
        body = self.rfile.read(length) if length else b""
        if len(body) != length:
            raise RequestError(400, "truncated_body", "request body ended before Content-Length bytes arrived")
        self._body_fully_read = True
        return body

    def _discard(self, length: int) -> bool:
        """Read and drop exactly ``length`` request-body bytes (bounded by the caller)."""
        remaining = length
        try:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    return False
                remaining -= len(chunk)
        except (OSError, ValueError):
            return False
        self._body_fully_read = True
        return True

    def _acquire_slot(self) -> bool:
        return self.app.semaphore.acquire(blocking=False)

    def _release_slot(self) -> None:
        try:
            self.app.semaphore.release()
        except ValueError:  # pragma: no cover - defensive
            pass

    def _drain_body(self, limit: int) -> bool:
        """Best-effort read of the request body so the connection can stay reusable.

        Used only on the cheap rejection paths (unknown route, inflight bound): without it, closing
        the connection while the client is still sending a body can abort that client's request.
        """
        if self.headers.get("Transfer-Encoding"):
            return False
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return False
        try:
            remaining = int(length_header)
        except (TypeError, ValueError):
            return False
        if remaining < 0 or remaining > limit:
            return False
        try:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    return False
                remaining -= len(chunk)
        except (OSError, ValueError):
            return False
        return True

    # -- GET ---------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/v1/health"):
            self._send_json(200, self.app.health())
            return
        if path in DECISION_ROUTES or path in PROXY_ROUTES:
            self._send_json(405, {"error": {
                "code": "method_not_allowed",
                "message": f"{path} only accepts POST",
                "status": 405,
            }})
            return
        self._send_error(RequestError(404, "not_found", f"no route for GET {path}"))

    # -- POST --------------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/v1/health"):  # readiness probes are often POSTed by monitors
            self._send_json(200, self.app.health())
            return
        if path not in DECISION_ROUTES and path not in PROXY_ROUTES:
            if not self._drain_body(self.app.config.max_proxy_body_bytes):
                self.close_connection = True
            self._send_error(RequestError(404, "not_found", f"no route for POST {path}"))
            return
        if not self._acquire_slot():
            if not self._drain_body(self.app.config.max_body_bytes):
                self.close_connection = True
            self._send_error(RequestError(
                429, "too_many_inflight",
                f"the gateway is at its configured inflight bound "
                f"({self.app.config.concurrency_max_inflight}); retry shortly",
            ))
            return
        started = time.monotonic()
        self._systemone_slot_transferred = False
        try:
            if path == "/v1/decide":
                self._handle_decide(started)
            elif path == jev_compat.SYSTEMONE_PATH:
                self._handle_systemone(started)
            else:
                self._handle_proxy(path, started)
        except RequestError as exc:
            self.app.log(f"{self.command} {path} -> {exc.status} {exc.code}")
            if not getattr(self, "_body_fully_read", False):
                self.close_connection = True   # an unconsumed body would desynchronise keep-alive
            self._send_error(exc)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            self.app.log(f"{self.command} {path} -> client_disconnect {type(exc).__name__}")
            self.close_connection = True
        finally:
            if not self._systemone_slot_transferred:
                self._release_slot()

    # -- /v1/decide ---------------------------------------------------------------------
    def _handle_decide(self, started: float) -> None:
        config = self.app.config
        body = self._read_body(config.max_body_bytes)
        payload = parse_json_object(body)
        wide_request = None
        if wide_payload_is_wide(payload, config):
            wide_request, arm, seed, deadline_ms = build_wide_decide_request(payload, config)
            request = None
        else:
            request, arm, seed, deadline_ms = build_decide_request(payload, config)
        deadline_s = None if deadline_ms is None else deadline_ms / 1000.0
        try:
            if wide_request is not None:
                outcome = self.app.backend.decide_wide(
                    wide_request, arm=arm, seed=seed, deadline_s=deadline_s)
            else:
                outcome = self.app.backend.decide(request, arm=arm, seed=seed, deadline_s=deadline_s)
        except decision_adapter.L1OUnsupportedPrimitive as exc:
            # the lettered surface has no form for this primitive: refused, never coerced
            raise RequestError(422, "primitive_unsupported_on_l1o", str(exc)) from exc
        except backend.ArmUnavailable as exc:
            raise RequestError(422, "arm_unavailable", str(exc)) from exc
        except templates.TemplateError as exc:
            # a restricted surface (e.g. qwen_chat_letters-v1) has no verbatim form for this kind:
            # refused before any engine call, never silently re-rendered as another template
            raise RequestError(422, "template_kind_unsupported", str(exc)) from exc
        except backend.DeadlineExceeded as exc:
            raise RequestError(504, "deadline_exceeded", str(exc)) from exc
        except backend.CaptureRefused as exc:
            # M1: the engine helper refused THIS request's capture for good (final refusal). Typed,
            # fast, and never a fallback lane: no probability is produced for a refused capture.
            reason = exc.reason if exc.reason else "capture_refused"
            raise RequestError(
                503, backend.CAPTURE_REFUSED_CODE, f"capture refused by the engine helper: reason={reason}; {exc}"
            ) from exc
        except backend.BackendUnavailable as exc:
            raise RequestError(503, "backend_unavailable", str(exc)) from exc
        except backend.ConfigError as exc:
            raise RequestError(500, "config_error", str(exc)) from exc
        except wide.WideError as exc:
            # a wide capture that cannot be proven genuine never decays into probabilities
            raise RequestError(502, "backend_error", f"wide capture rejected: {exc}") from exc
        except (backend.TransportError, backend.CaptureError, backend.BackendError) as exc:
            raise RequestError(502, "backend_error", f"{type(exc).__name__}: {exc}") from exc
        if wide_request is not None:
            payload_out = backend.wide_outcome_payload(outcome)
        else:
            payload_out = backend.outcome_payload(outcome)
        total_ms = (time.monotonic() - started) * 1000.0
        if wide_request is not None:
            self.app.log(
                "POST /v1/decide -> 200 wide=wide-labels "
                f"arm={outcome.arm} kind=choice n_options={wide_request.n_options} "
                f"state_sha256={_hash(wide_request.state)} question_sha256={_hash(wide_request.question)} "
                f"template={wide.WIDE_TEMPLATE_ID}/{wide.WIDE_TEMPLATE_VERSION} "
                f"prompt_token_sha256={outcome.captured.capture.prompt_token_sha256[:16]} "
                f"engine_request_id={outcome.captured.engine_request_id} "
                f"transport={outcome.captured.usage.get('completion_finish_reason') or 'n/a'} "
                f"provider={outcome.provider_block['label']} total_ms={total_ms:.1f}"
            )
        else:
            self.app.log(
                "POST /v1/decide -> 200 "
                f"arm={outcome.arm} kind={outcome.result.kind} n_options={len(outcome.result.caller_keys)} "
                f"state_sha256={_hash(request.state)} question_sha256={_hash(request.question)} "
                f"template={config.template_id}/{config.template_version} "
                f"prompt_token_sha256={outcome.tokenizer_block['prompt_token_sha256'][:16]} "
                f"engine_request_id={outcome.engine_block['engine_request_id']} "
                f"transport={outcome.transport} provider={outcome.provider_block['label']} "
                f"total_ms={total_ms:.1f}"
            )
        self._send_json(200, payload_out)

    # -- /v1/systemone (Jev-compatible translation of the same typed path) ---------------
    def _handle_systemone(self, started: float) -> None:
        """Jev's request/response shape over the existing typed decision path (arm B0 by default).

        All schema validation and translation lives in ``qfn_decision.jev_compat``; this method only
        moves bytes, maps the translation layer's errors onto the gateway's error envelope, and logs
        hashes/counts (never prompt text).
        """
        config = self.app.config
        body = self._read_body(config.max_body_bytes)
        payload = parse_json_object(body)
        try:
            doc = jev_compat.handle_systemone(
                payload, config, self.app.backend, semaphore=self.app.semaphore,
                on_slot_transfer=lambda: setattr(self, "_systemone_slot_transferred", True),
            )
        except jev_compat.JevError as exc:
            raise RequestError(exc.status, exc.code, exc.message, extra=exc.detail()) from exc
        total_ms = (time.monotonic() - started) * 1000.0
        self.app.log(
            f"POST {jev_compat.SYSTEMONE_PATH} -> 200 model={doc['model']} "
            f"questions={len(doc['answers'])} input_tokens={doc['usage']['input_tokens']} "
            f"arm={config.default_arm} template={config.template_id}/{config.template_version} "
            f"total_ms={total_ms:.1f}"
        )
        self._send_json(200, doc)

    # -- ordinary generation proxy ------------------------------------------------------
    def _guard_ordinary_adapter_alias(self, body: bytes, path: str, spec: Any) -> None:
        """Opted-in mode only: refuse an ordinary body that selects the adapter alias.

        Runs BEFORE the upstream connection is opened, so a refused request produces no engine
        traffic at all (streaming included, because the decision is taken on the body alone). The
        base path is untouched: with the adapter disabled/absent this method is never called and
        the caller's bytes are forwarded verbatim.
        """
        try:
            doc = parse_json_object(body)
        except RequestError as exc:
            raise RequestError(
                400, "adapter_mode_body_unclassifiable",
                f"the experimental L1O adapter is enabled, so an ordinary {path} body must be a "
                f"JSON object that can be classified before it is forwarded; this body is not "
                f"({exc.code}: {exc.message})",
            ) from exc
        model = doc.get("model")
        if not isinstance(model, str) or not model:
            raise RequestError(
                422, "adapter_mode_model_missing",
                f"the experimental L1O adapter is enabled, so an ordinary {path} body must name the "
                f"model it selects so the request can be classified; got model={model!r}",
            )
        if model == spec.model_alias:
            raise RequestError(
                422, "adapter_typed_only",
                f"model {model!r} is the experimental L1O adapter alias and L1O is available on the "
                f"TYPED route only (POST /v1/decide with arm='L1O'); refusing to forward this "
                f"ordinary {path} request instead of silently serving it with the adapter",
            )

    def _handle_proxy(self, path: str, started: float) -> None:
        config = self.app.config
        body = self._read_body(config.max_proxy_body_bytes)
        adapter_spec = getattr(config, "decision_adapter", None)
        if adapter_spec is not None and adapter_spec.enabled:
            # opted-in experimental mode: classify first, fail closed on anything ambiguous
            self._guard_ordinary_adapter_alias(body, path, adapter_spec)
        parsed = urllib.parse.urlsplit(config.engine_base_url)
        headers: dict[str, str] = {}
        forwarded_names: list[str] = []
        for name in config.forward_headers:
            value = self.headers.get(name)
            if value is not None:
                headers[name] = value
                forwarded_names.append(name)
        for name, env_name in config.forward_headers_from_env.items():
            value = os.environ.get(env_name)
            if value:
                headers[name] = value
                forwarded_names.append(name)
        # our own Content-Length for exactly these bytes; the inbound one is never forwarded
        headers["content-length"] = str(len(body))
        headers.pop("transfer-encoding", None)

        # One bounded reader thread per admitted proxy request owns the engine connection: the
        # connect, the request, the response header wait and the HTTPResponse decode. The
        # connection thread only relays already-decoded bytes, so the short downstream poll
        # interval can never poison the buffered decoder (Python cannot reuse a socket file that
        # has experienced a timeout) and a stalled header wait stays cancellable.
        #
        # ``deadline`` is absolute: measured once from the monotonic request start and never
        # reset per chunk, so an engine that keeps trickling data cannot stream forever.
        deadline = started + config.request_timeout_s
        events: queue.Queue = queue.Queue(maxsize=STREAM_QUEUE_MAXSIZE)
        stop = threading.Event()
        handles = _EngineHandles()
        scheme = parsed.scheme or "http"
        reader = threading.Thread(
            target=_read_engine_response,
            kwargs={
                "scheme": scheme,
                "host": parsed.hostname or "",
                "port": parsed.port or (443 if scheme == "https" else 80),
                "method": "POST",
                "path": path,
                "body": body,
                "headers": headers,
                "deadline": deadline,
                "read_timeout_s": config.timeout_socket_s,
                "events": events,
                "stop": stop,
                "handles": handles,
            },
            name=f"qfn-stream-{next(_READER_SEQ)}",
            daemon=True,
        )
        reader.start()
        try:
            self._consume_engine_response(
                path=path, started=started, in_bytes=len(body), forwarded=forwarded_names,
                deadline=deadline, events=events,
            )
        finally:
            self._teardown_engine_reader(stop=stop, handles=handles, reader=reader, events=events)

    def _downstream_closed(self) -> bool:
        """True when the client has gone away (EOF/hang-up detected without consuming body bytes)."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
        except (OSError, ValueError):
            return True
        if not readable:
            return False
        try:
            data = self.connection.recv(1, socket.MSG_PEEK)
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True
        return data == b""

    def _send_stream_headers(self, status: int, passthrough: Mapping[str, str]) -> None:
        """Send the engine's own status and forwarded headers, then chunked framing (unchanged set)."""
        self.send_response(status)
        for key, value in passthrough.items():
            self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _finish_downstream_chunked(self, path: str, chunks: int, byte_count: int) -> bool:
        """Write the terminating chunk. Called exactly once and only for a genuine clean EOF."""
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True
            self.app.log(
                f"proxy {path} -> client_disconnect chunks={chunks} bytes={byte_count}; closing the "
                "upstream engine connection"
            )
            return False
        self.close_connection = True
        return True

    def _write_downstream_chunk(self, chunk: bytes, path: str, chunks: int, byte_count: int) -> bool:
        """Write one chunked frame; a dead client is reported once and ends the relay (bounded write)."""
        try:
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True
            self.app.log(
                f"proxy {path} -> client_disconnect chunks={chunks} bytes={byte_count}; closing the "
                "upstream engine connection"
            )
            return False

    def _gateway_stopping(self) -> bool:
        """True once the server is winding down.

        A proxied stream must not rely on noticing that the framework closed its client socket:
        on this platform a locally closed socket is not reported readable, so shutdown would
        otherwise be invisible to an in-flight stream.
        """
        return bool(getattr(self.server, "stopping", False))

    def _consume_engine_response(self, *, path: str, started: float, in_bytes: int,
                                 forwarded: Sequence[str], deadline: float,
                                 events: "queue.Queue") -> None:
        """Relay the reader's decoded engine response while polling the client, deadline and shutdown.

        The reader's events are honoured strictly and distinctly:
        ``HEADERS -> DATA* -> (EOF | ERROR)``.
        Only a genuine clean ``EOF`` writes the terminating HTTP chunk. An upstream ``ERROR``
        (truncated, malformed or timed-out engine body) and the absolute request deadline close the
        downstream response *without* one, so the client sees a truncated stream instead of a
        silently normal-looking completion. Nothing here logs request text, forwarded header values,
        exception messages, prompts, tokens or credentials: failures travel as metadata only.
        """
        status = 0
        stream = False
        headers_sent = False
        passthrough: Mapping[str, str] = {}
        parts: list[bytes] = []
        chunks = 0
        byte_count = 0
        while True:
            if self._gateway_stopping():
                self.close_connection = True
                self.app.log(
                    f"proxy {path} -> gateway_shutdown chunks={chunks} bytes={byte_count}; closing the "
                    "upstream engine connection"
                )
                return
            if time.monotonic() >= deadline:
                if not headers_sent:
                    self.close_connection = True
                    raise RequestError(
                        504, "deadline_exceeded",
                        f"the ordinary {path} request exceeded its engine deadline before the engine "
                        "produced a response",
                    )
                self.close_connection = True
                self.app.log(
                    f"proxy {path} -> deadline_exceeded after_headers chunks={chunks} out_bytes={byte_count} "
                    f"ms={(time.monotonic() - started) * 1000:.1f}; closing the downstream stream without a "
                    "terminating chunk"
                )
                return
            if self._downstream_closed():
                self.close_connection = True
                self.app.log(
                    f"proxy {path} -> client_disconnect chunks={chunks} bytes={byte_count}; closing the "
                    "upstream engine connection"
                )
                return
            try:
                event = events.get(timeout=STREAM_POLL_S)
            except queue.Empty:
                continue
            tag = event[0]
            if tag == "HEADERS":
                status = int(event[1])
                passthrough = event[2]
                stream = "text/event-stream" in str(passthrough.get("Content-Type") or "").lower()
                if stream:
                    self._send_stream_headers(status, passthrough)
                    headers_sent = True
                continue
            if tag == "DATA":
                chunk = event[1]
                if not stream:
                    parts.append(chunk)          # non-stream bodies are byte-exact, as before
                    continue
                if not self._write_downstream_chunk(chunk, path, chunks, byte_count):
                    return
                chunks += 1
                byte_count += len(chunk)
                continue
            if tag == "EOF":
                if stream:
                    if not self._finish_downstream_chunked(path, chunks, byte_count):
                        return
                    self.app.log(
                        f"proxy {path} -> {status} stream=true chunks={chunks} out_bytes={byte_count} "
                        f"forwarded_headers={list(forwarded)} ms={(time.monotonic() - started) * 1000:.1f}"
                    )
                    return
                data = b"".join(parts)
                # the engine's own status and body are preserved byte for byte
                self._send_bytes(status, data, str(passthrough.get("Content-Type") or "application/json"), passthrough)
                self.app.log(
                    f"proxy {path} -> {status} stream=false in_bytes={in_bytes} out_bytes={len(data)} "
                    f"forwarded_headers={list(forwarded)} ms={(time.monotonic() - started) * 1000:.1f}"
                )
                return
            # tag == "ERROR": the engine failed or truncated; nothing is claimed as a clean end
            if headers_sent:
                self.close_connection = True
                self.app.log(
                    f"proxy {path} -> engine_stream_{event[1]} after_headers chunks={chunks} "
                    f"out_bytes={byte_count} ({event[2]}); closing the downstream stream without a "
                    "terminating chunk"
                )
                return
            raise RequestError(
                502, "backend_error",
                f"engine {path} request failed: {event[1]} ({event[2]})",
            )

    def _teardown_engine_reader(self, *, stop: threading.Event, handles: "_EngineHandles",
                                reader: threading.Thread, events: "queue.Queue") -> None:
        """Cancel, wake, join and close the engine reader exactly once (idempotent, bounded).

        Order matters: signal cancellation, shut down the *snapshotted* upstream socket (never the
        possibly-detached ``resp.sock``: an engine answer with ``Connection: close`` closes and
        forgets it). The shutdown is what tells the engine to cancel the generation; an engine that
        honours it closes its side, the reader's blocked read then ends at EOF and the join
        completes in milliseconds. Draining the queue while joining means a full queue can never
        strand the producer, and the response/connection are closed only *after* the reader has
        exited: closing the buffered response while the reader is still inside it can block on the
        very read the reader owns, and closing the descriptor under a blocked read is worse.
        """
        stop.set()
        _sock, conn, resp = handles.snapshot()
        limit = time.monotonic() + STREAM_JOIN_S
        while True:
            sock = handles.socket()          # re-read: the reader may connect after the first look
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            while True:
                try:
                    events.get_nowait()
                except queue.Empty:
                    break
            if not reader.is_alive() or time.monotonic() >= limit:
                break
            reader.join(timeout=STREAM_POLL_S)
        if not reader.is_alive():
            for closeable in (resp, conn):
                if closeable is not None:
                    try:
                        closeable.close()
                    except Exception:  # noqa: BLE001
                        pass
            return
        # The reader is still parked in a read on the engine socket. On this platform a blocked
        # read cannot be woken from the outside (shutdown/close from another thread do not abort
        # it); it ends when the engine closes its side or when the configured engine socket timeout
        # fires, whichever comes first. It holds no request slot, it never writes downstream, and
        # it closes its own connection when it wakes, so nothing is stranded and the count stays
        # bounded by the engine timeout, not by traffic. Reported, never hidden.
        self.app.log(
            f"proxy reader {reader.name} still parked on a read after {STREAM_JOIN_S:g}s; it is bounded "
            "by the configured engine socket timeout, holds no gateway slot and streams nothing further"
        )


_READER_SEQ = itertools.count(1)


class _EngineHandles:
    """Reader -> request-thread handoff of the engine socket, connection and response.

    The engine socket is snapshotted *before* the response decoder can adopt and later detach it
    (an engine answer with ``Connection: close`` makes ``HTTPResponse`` close and forget
    ``conn.sock``), so cancellation never has to rely on a possibly-``None`` ``resp.sock``.
    """

    __slots__ = ("_lock", "_sock", "_conn", "_resp")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sock: Any = None
        self._conn: Any = None
        self._resp: Any = None

    def publish(self, *, sock: Any = None, conn: Any = None, resp: Any = None) -> None:
        with self._lock:
            if sock is not None:
                self._sock = sock
            if conn is not None:
                self._conn = conn
            if resp is not None:
                self._resp = resp

    def socket(self) -> Any:
        with self._lock:
            return self._sock

    def snapshot(self) -> tuple[Any, Any, Any]:
        with self._lock:
            return self._sock, self._conn, self._resp


def _put_event(events: "queue.Queue", stop: threading.Event, event: tuple) -> bool:
    """Bounded, cancellable enqueue: never blocks a reader longer than one poll interval."""
    while not stop.is_set():
        try:
            events.put(event, timeout=STREAM_POLL_S)
            return True
        except queue.Full:
            continue
    return False


def _engine_passthrough(resp: Any) -> dict[str, str]:
    """The engine headers the gateway forwards verbatim (unchanged set and order)."""
    ctype = resp.getheader("Content-Type") or ""
    passthrough: dict[str, str] = {"Content-Type": ctype or "application/json"}
    for name in ("Cache-Control", "OpenAI-Processing-Ms", "X-Request-Id"):
        value = resp.getheader(name)
        if value:
            passthrough[name] = value
    return passthrough


def _read_engine_response(
    *,
    scheme: str,
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes,
    headers: Mapping[str, str],
    deadline: float,
    read_timeout_s: float,
    events: "queue.Queue",
    stop: threading.Event,
    handles: _EngineHandles,
) -> None:
    """One bounded reader thread: owns the engine connect, request, header wait and body decode.

    Events are distinct and ordered ``HEADERS -> DATA* -> (EOF | ERROR)``, with exactly one
    terminal event. The configured engine socket timeout is set once and never changed mid-stream,
    so the buffered decoder is never poisoned by a short poll timeout; the connect itself may never
    outlive the request's remaining budget. Nothing here logs request text, forwarded values or
    exception messages: a failure travels as a stable reason label plus the exception *type* name.
    """
    conn: Any = None
    resp: Any = None
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _put_event(events, stop, ("ERROR", "deadline", "RequestDeadlineExceeded"))
            return
        timeout = min(read_timeout_s, remaining)
        if scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        handles.publish(conn=conn)
        conn.request(method, path, body=body, headers=dict(headers))
        sock = getattr(conn, "sock", None)
        handles.publish(sock=sock)
        if sock is not None:
            # the configured engine socket timeout, set before the decoder adopts the socket
            sock.settimeout(read_timeout_s)
        resp = conn.getresponse()
        handles.publish(resp=resp)
    except (socket.timeout, TimeoutError) as exc:
        _put_event(events, stop, ("ERROR", "timeout", type(exc).__name__))
        return
    except (OSError, http.client.HTTPException, ValueError) as exc:
        _put_event(events, stop, ("ERROR", "connect", type(exc).__name__))
        return
    try:
        if not _put_event(events, stop, ("HEADERS", resp.status, _engine_passthrough(resp))):
            return
        read = resp.read1 if hasattr(resp, "read1") else resp.read
        while not stop.is_set():
            chunk = read(STREAM_READ_SIZE)
            if not chunk:
                _put_event(events, stop, ("EOF",))    # genuine clean end of the engine body
                return
            if not _put_event(events, stop, ("DATA", chunk)):
                return
    except (socket.timeout, TimeoutError) as exc:
        _put_event(events, stop, ("ERROR", "timeout", type(exc).__name__))
    except http.client.HTTPException as exc:
        # IncompleteRead and any other protocol-level truncation: never reported as a clean EOF
        _put_event(events, stop, ("ERROR", "truncated", type(exc).__name__))
    except (OSError, ValueError) as exc:
        _put_event(events, stop, ("ERROR", "read", type(exc).__name__))
    finally:
        # the reader owns its connection: it closes it even when the request thread gave up, so a
        # reader can never leave a socket behind (closing twice from either side is harmless)
        for closeable in (resp, conn):
            if closeable is not None:
                try:
                    closeable.close()
                except Exception:  # noqa: BLE001
                    pass


class GatewayServer:
    """Owns the HTTP server, the decision backend and the bounded concurrency machinery."""

    def __init__(
        self,
        config: backend.ApiConfig,
        *,
        backend_instance: backend.DecisionBackend | None = None,
        host: str | None = None,
        port: int | None = None,
        log_sink: Any = None,
    ) -> None:
        self.config = config
        self.host = host or config.host
        self.port = int(port if port is not None else config.port)
        self.backend = backend_instance if backend_instance is not None else backend.DecisionBackend(config)
        self.semaphore = threading.BoundedSemaphore(max(1, config.concurrency_max_inflight))
        self._log_sink = log_sink
        self._logs: list[str] = []
        self._httpd: _BoundedThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- logging (redacted by construction: callers pass hashes/metrics only) ----------
    def log(self, line: str) -> None:
        entry = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}"
        self._logs.append(entry)
        if self._log_sink is not None:
            self._log_sink(entry)
            return
        if self.config.log_file:
            try:
                pathlib.Path(self.config.log_file).parent.mkdir(parents=True, exist_ok=True)
                with open(self.config.log_file, "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(entry + "\n")
                return
            except OSError:
                pass
        print(entry, file=sys.stdout, flush=True)

    def logs(self) -> tuple[str, ...]:
        return tuple(self._logs)

    # -- lifecycle ----------------------------------------------------------------------
    @property
    def max_workers(self) -> int:
        return self.config.concurrency_max_inflight + 4

    def start(self) -> "GatewayServer":
        httpd = _BoundedThreadingHTTPServer(
            (self.host, self.port),
            GatewayRequestHandler,
            max_workers=self.max_workers,
            socket_timeout_s=self.config.timeout_socket_s,
        )
        httpd.app = self  # type: ignore[attr-defined]
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": SERVE_POLL_S}, name="qfn-gateway", daemon=True
        )
        self._thread.start()
        self.log(
            f"gateway listening on http://{self.host}:{self.port} "
            f"provider={getattr(self.backend.provider, 'label', 'unknown')} "
            f"max_inflight={self.config.concurrency_max_inflight} workers={self.max_workers}"
        )
        return self

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd, self._thread = None, None
        if httpd is not None:
            httpd.begin_stop()
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=10.0)

    def serve_forever(self) -> None:  # pragma: no cover - exercised by the CLI, not the tests
        if self._httpd is None:
            raise backend.BackendError("gateway is not started")
        self._httpd.serve_forever(poll_interval=SERVE_POLL_S)

    def __enter__(self) -> "GatewayServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- views -------------------------------------------------------------------------
    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def health(self) -> dict:
        doc = self.backend.health()
        doc["bind"]["port"] = self.port          # the port actually bound (tests use ephemeral ports)
        doc["bind"]["host"] = self.host
        return doc

    def print_curl(self) -> str:
        base = self.url
        decide_body = json.dumps({
            "kind": "noul",
            "state": "<the state the decision is about>",
            "question": "<one question>",
            "thinking": "off",
            "arm": self.config.default_arm,
        }, indent=2)
        choice_body = json.dumps({
            "kind": "choice",
            "state": "<the state the decision is about>",
            "question": "<one question>",
            "options": [{"key": "k1", "description": "first option"}, {"key": "k2", "description": "second option"}],
            "thinking": "off",
        }, indent=2)
        return "\n".join([
            f"# health (readiness only; never a live-success claim)",
            f"curl -sS {base}/health",
            f"",
            f"# typed decision: noul (P(true) only, no generated tokens returned)",
            f"curl -sS -X POST {base}/v1/decide -H 'Content-Type: application/json' -d '{decide_body}'",
            f"",
            f"# typed decision: choice with caller keys and 2..16 options",
            f"curl -sS -X POST {base}/v1/decide -H 'Content-Type: application/json' -d '{choice_body}'",
            f"",
            f"# ordinary generation (forwarded unchanged to the SAME engine at {self.config.engine_base_url})",
            f"curl -sS -X POST {base}/v1/chat/completions -H 'Content-Type: application/json' "
            f"-d '{{\"model\":\"{self.config.engine_model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'",
            f"",
            f"# streaming generation (SSE chunks are relayed verbatim)",
            f"curl -N -sS -X POST {base}/v1/completions -H 'Content-Type: application/json' "
            f"-d '{{\"model\":\"{self.config.engine_model}\",\"prompt\":\"hi\",\"stream\":true,\"max_tokens\":8}}'",
            f"",
            f"# These commands print real responses; nothing here is a canned result.",
        ])

    def summary(self) -> str:
        health = self.health()
        arms = ", ".join(
            f"{a['arm']}={'available' if a['available'] else 'unavailable'}" for a in health["arms"]
        )
        adapter = health.get("decision_adapter")
        adapter_lines: list[str] = []
        if adapter is not None:
            adapter_lines = [
                f"exp. L1O arm     : enabled={adapter['enabled']} status={adapter['status']} "
                f"alias={adapter['model_alias']!r} available={adapter['available']} "
                f"runtime_verified={adapter['runtime_verified']}",
                f"                   checkpoint={adapter['checkpoint']['sha256_expected']} "
                f"(local sha256_verified={adapter['checkpoint']['sha256_verified']}) typed_only=true",
            ]
        lines = [
            f"config file      : {getattr(self.config, 'config_path', '(in-memory)')}",
            f"bind             : {self.host}:{self.port} (loopback_only={health['bind']['loopback_only']})",
            f"engine           : {self.config.engine_base_url} model={self.config.engine_model}",
            f"provider         : {health['provider']['label']} ({health['provider']['kind']})",
            f"template         : {self.config.template_id}/{self.config.template_version} (frozen)",
            f"tokenizer rev    : {self.config.tokenizer_revision}",
            f"capture dir      : {self.config.capture_dir} (allowlisted={health['capture']['dir_allowlisted']}, "
            f"exists={bool(self.config.capture_dir) and pathlib.Path(self.config.capture_dir).is_dir()})",
            f"arms             : {arms}",
            f"max_inflight     : {self.config.concurrency_max_inflight} (workers={self.max_workers})",
            f"redaction        : prompts/log text redacted={self.config.redact_prompts}",
            *adapter_lines,
            "status           : ok",
            f"readiness        : engine_ping={health['readiness']['engine_ping']} "
            f"live_engine_validated={health['readiness']['live_engine_validated']}",
            f"note             : {health['readiness']['note']}",
        ]
        return "\n".join(lines)


# ======================================================================================
# CLI
# ======================================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qfn_decision.api",
        description=(
            "S6 same-engine typed decision gateway: POST /v1/decide plus ordinary "
            "/v1/chat/completions and /v1/completions forwarded to the same vLLM engine."
        ),
    )
    parser.add_argument("--config", default=None, help=f"gateway config JSON (default: {backend.CONFIG_PATH})")
    parser.add_argument("--host", default=None, help="bind host (default from config: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default from config: 8890)")
    parser.add_argument("--provider", choices=("engine", "fixture"), default=None,
                        help="feature provider; 'fixture' is a labelled synthetic provider")
    parser.add_argument("--demo", action="store_true",
                        help="FIXTURE ONLY: start an internal fixture engine and point the gateway at it")
    parser.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    parser.add_argument("--print-curl", action="store_true", help="print smoke curl commands and exit")
    parser.add_argument("--check", action="store_true", help="validate config/provider/arms and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = backend.load_api_config(args.config)
    except backend.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    demo_engine: backend.FixtureEngineServer | None = None
    try:
        if args.demo:
            import tempfile
            capture_dir = pathlib.Path(tempfile.mkdtemp(prefix="qfn-demo-capture-"))
            demo_engine = backend.FixtureEngineServer(model=config.engine_model, capture_dir=capture_dir)
            demo_engine.start()
            raw = config.as_dict()
            raw["engine"]["base_url"] = demo_engine.base_url
            raw["capture"]["dir"] = str(capture_dir)
            raw["capture"]["dir_allowlist"] = [str(capture_dir)]
            raw["provider"] = "engine"
            config = backend.ApiConfig.from_dict(raw)
            print("FIXTURE ONLY: --demo started an internal synthetic engine; no live vLLM engine is involved.")
        if args.provider:
            raw = config.as_dict()
            raw["provider"] = args.provider
            config = backend.ApiConfig.from_dict(raw)

        if args.print_config:
            print(json.dumps(config.as_dict(), indent=2))
            return 0

        if args.print_curl:
            server = GatewayServer(config, host=args.host, port=args.port)
            print(server.print_curl())
            return 0

        if args.check:
            server = GatewayServer(config, host=args.host, port=args.port)
            print(server.summary())
            return 0

        server = GatewayServer(config, host=args.host, port=args.port)
        server.start()
        print(server.summary())
        print(f"gateway url      : {server.url}")
        print("")
        print(server.print_curl())
        print("")
        print("Startup succeeded. Startup does not call the engine; send a request to confirm the live path.")
        print("Press Ctrl-C to stop.")
        server.serve_forever()
        return 0
    except backend.BackendError as exc:
        print(f"startup error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 0
    finally:
        if 'server' in locals() and server is not None:
            server.stop()
        if demo_engine is not None:
            demo_engine.stop()


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
