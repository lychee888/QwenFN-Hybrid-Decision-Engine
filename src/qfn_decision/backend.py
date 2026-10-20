"""S6 typed-gateway backends: feature acquisition, head arms, config, fixture engine.

What this module is
-------------------
The gateway in :mod:`qfn_decision.api` exposes ``POST /v1/decide`` and forwards ordinary
``/v1/chat/completions`` + ``/v1/completions`` to the *same* vLLM engine. This module owns
everything between the HTTP layer and the shared S1 modules (``protocol`` / ``templates`` /
``features`` / ``head``):

* the request-id / capture-file-name contract of the engine-side helper
  (``patches/qfn_capture_runtime.py``), so the gateway never guesses an identifier,
* a bounded, fail-closed HTTP capture provider (real ``/tokenize`` -> real
  ``/v1/completions`` with explicit token ids -> bounded poll of the capture directory),
* the two readout arms: ``B0`` (masked normalisation of the *raw* captured 24-slot logits)
  and ``H1`` (the shared FP32 :class:`~qfn_decision.head.DecisionHead` applied to the exactly
  widened BF16 feature of the same engine),
* config loading for ``configs/decision_api.json``.

Honesty scope
-------------
* **Live scope.** The B0 arm with the ``qwen_chat_letters-v1`` template has run against the real
  engine through :mod:`qfn_decision.hybrid`; other arms and templates are fixture-tested only.
* The feature transport may schedule **one discarded internal completion token**; that cost is
  reported as ``internal_generated_tokens`` and as ``transport='one_discarded_token_capture'``.
  Zero returned text is **not** a native zero-sampler engine implementation.
* No returned reasoning text and no generated verbalizer text ever leaves the service.
* B0/H1 parity is **declared, not measured** (see ``configs/numeric_tolerances.json``).
* :class:`FixtureEngineServer` and :class:`FixtureFeatureProvider` are labelled
  ``fixture_only``: they are deterministic synthetic doubles used by tests and the offline
  demo. They must never be presented as, or pointed at as, the live engine.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import http.client
import io
import json
import os
import pathlib
import re
import socket
import socketserver
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from . import decision_adapter, features, head as head_mod, protocol, templates, wide

__all__ = [
    # errors
    "BackendError",
    "TransportError",
    "CaptureError",
    "CaptureRefused",
    "DeadlineExceeded",
    "BackendUnavailable",
    "ArmUnavailable",
    "ConfigError",
    # constants
    "TRANSPORT_ONE_DISCARDED_TOKEN",
    "CALLER_REQUEST_ID_PREFIX",
    "ENGINE_REQUEST_ID_PREFIX",
    "CACHE_EVIDENCE_UNVERIFIED",
    "TYPED_OUTPUT_TOKENS",
    "FIXTURE_ONLY_LABEL",
    "READINESS_NOTE",
    "CONFIG_SCHEMA",
    "CONFIG_PATH",
    "CAPTURE_REFUSED_MARKER_NAME",
    "CAPTURE_REFUSED_SCHEMA",
    "CAPTURE_REFUSED_CODE",
    # id / file-name contract
    "new_caller_request_id",
    "marked_request_id_ok",
    "engine_request_id_for",
    "resolve_engine_request_id",
    "engine_id_pattern",
    "engine_id_matches",
    "expected_record_file_name",
    "resolve_record_files",
    "discover_record_files",
    "read_error_marker",
    "error_marker_is_fatal",
    "annotate_error_marker",
    "read_refusal_marker",
    "refusal_for_request",
    "annotate_refusal",
    "decode_record_bytes",
    "fixture_only_npz_bytes",
    "validate_record",
    # capture
    "CapturedFeature",
    "CaptureProviderConfig",
    "capture_provider_config",
    "HttpEngineCaptureProvider",
    "FixtureEngineServer",
    "FixtureFeatureProvider",
    "FeatureProvider",
    # arms
    "ArmReadout",
    "B0Arm",
    "H1Arm",
    "L1OArm",
    # config + orchestration
    "ApiConfig",
    "default_api_config_dict",
    "load_api_config",
    "DecisionBackend",
    "DecisionOutcome",
    "outcome_payload",
    "REPO_ROOT",
]

TRANSPORT_ONE_DISCARDED_TOKEN = "one_discarded_token_capture"
CALLER_REQUEST_ID_PREFIX = "qfn-"                 # what the gateway puts in the request body
ENGINE_REQUEST_ID_PREFIX = "cmpl-qfn-"            # what the engine/OpenAI layer actually uses
CACHE_EVIDENCE_UNVERIFIED = "unverified"
TYPED_OUTPUT_TOKENS = 0
FIXTURE_ONLY_LABEL = "fixture_only"
CONFIG_SCHEMA = "qfn.decision-api/1"
CONFIG_PATH = "configs/decision_api.json"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# The experimental typed arm lives in its own module; re-exported so callers can reach it as
# ``backend.L1OArm`` alongside B0Arm/H1Arm (it is *not* a fitted head).
L1OArm = decision_adapter.L1OArm
MAX_DUP_DISAMBIGUATION = 9
ALLOWED_KEY_CHARS = re.compile(r"^[A-Za-z0-9._-]+$")
KEY_MAX_LEN = 160

READINESS_NOTE = (
    "Live-validated scope: release B0 with qwen_chat_letters-v1 and qwen_chat_wide-v1, "
        "served through qfn_decision.hybrid on a two-node GB10 deployment in October 2026. "
        "This health response does not independently verify the current engine identity. "
        "Other arms and providers are outside that live acceptance scope."
)

ENGINE_ID_RE = re.compile(r"^cmpl-qfn-[0-9a-f]{32}-[0-9]+$")
# PM live finding 2026-10-10 (real capture candidate, image fc120ece):
#   * the HTTP response `id` is `cmpl-<caller>` with NO prompt-index suffix;
#   * the engine-internal id the capture helper sees is `cmpl-<caller>-<index>-<8 lowercase hex>`
#     (e.g. cmpl-qfn-06b6d07d543447bfa1811ccab1844a10-0-89c4f0b7), and both the record file name and
#     the record's engine_request_id field carry that suffixed id.
# Accepted id forms (nothing else, and never another request):
#   cmpl-<caller>  |  cmpl-<caller>-<index>  |  cmpl-<caller>-<index>-<8hex>
ENGINE_ID_SUFFIX_HEX_LEN = 8
ENGINE_MARKER_PREFIX = "cmpl-"
CAPTURE_ERROR_MARKER_NAME = "CAPTURE_ERROR.json"
# M1: the engine helper's per-request REFUSAL marker (patches/qfn_capture_runtime_mtp1.py). Unlike
# CAPTURE_ERROR.json (a helper-level diagnostic that only fails a request when it names that
# request or reports a global stop), an entry here is a deliberate, final per-request refusal:
# the marked prompt's terminal token was computed by the engine step and the helper refused the
# capture fail-closed. A request named by an entry must fail FAST (503 capture_refused) instead of
# waiting out its deadline; a request that is not named is unaffected.
CAPTURE_REFUSED_MARKER_NAME = "CAPTURE_REFUSED.json"
CAPTURE_REFUSED_SCHEMA = "qfn.capture-refused/1"
CAPTURE_REFUSED_CODE = "capture_refused"
MAX_REFUSAL_ENTRIES_SCANNED = 128
# Marker kinds that mean "no capture is possible at all" (the helper stopped). Anything else in the
# marker is a helper-level or per-skip diagnostic and must NOT fail every request.
GLOBAL_STOP_MARKER_KINDS = frozenset({
    "quota_exceeded",
    "capture_stopped",
    "capture_disabled",
    "helper_disabled",
    "no_torch",
    "no_req_states",
})
MARKER_REQUEST_ID_KEYS = ("request_id", "engine_request_id", "req_id", "caller_request_id", "id")


# ======================================================================================
# errors
# ======================================================================================

class BackendError(Exception):
    """Base class for every failure this module reports."""


class ConfigError(BackendError):
    """The gateway configuration is invalid or unsafe."""


class TransportError(BackendError):
    """The engine could not be reached, or answered with an unusable response."""


class CaptureError(BackendError):
    """A capture record was absent-but-present, malformed, stale or mismatched."""


class CaptureRefused(CaptureError):
    """The engine helper refused *this* marked request's capture, for good (M1).

    Raised only from a refusal-marker entry that identifies this caller's request. It is a
    :class:`CaptureError` so every existing handler keeps working, and it carries the helper's
    typed ``reason`` so the gateway can answer with HTTP 503 ``capture_refused`` and the reason
    inside ~1 s of the engine response instead of a full-deadline 504.
    """

    code = CAPTURE_REFUSED_CODE

    def __init__(self, message: str, *, reason: str | None = None, refusal: dict | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.refusal = refusal


class DeadlineExceeded(BackendError):
    """The bounded capture deadline elapsed before a record appeared."""


class BackendUnavailable(BackendError):
    """No feature provider is configured, so the typed endpoint cannot serve."""


class ArmUnavailable(BackendError):
    """The requested readout arm is not available (never silently substituted)."""


# ======================================================================================
# request-id and capture-file-name contract (mirrors patches/qfn_capture_runtime.py)
# ======================================================================================

def new_caller_request_id() -> str:
    """A server-generated, allowlisted request id. A caller can never supply one."""
    return CALLER_REQUEST_ID_PREFIX + uuid.uuid4().hex


def marked_request_id_ok(request_id: Any) -> bool:
    if not isinstance(request_id, str) or not request_id.startswith(CALLER_REQUEST_ID_PREFIX):
        return False
    rest = request_id[len(CALLER_REQUEST_ID_PREFIX):]
    if not rest or len(rest) > KEY_MAX_LEN:
        return False
    return bool(re.fullmatch(r"[0-9a-zA-Z._-]+", rest))


def engine_request_id_for(caller_request_id: str, prompt_index: int = 0) -> str:
    """The engine-side id: the OpenAI layer prepends ``cmpl-`` and appends ``-<prompt_index>``."""
    if not marked_request_id_ok(caller_request_id):
        raise BackendError(
            f"refusing to build an engine request id from {caller_request_id!r}: it is not a "
            f"server-generated {CALLER_REQUEST_ID_PREFIX}<hex> id"
        )
    if not isinstance(prompt_index, int) or isinstance(prompt_index, bool) or prompt_index < 0:
        raise BackendError(f"prompt_index must be a non-negative int, got {prompt_index!r}")
    return f"cmpl-{caller_request_id}-{prompt_index}"


def engine_id_pattern(caller_request_id: str, prompt_index: int | None = None) -> "re.Pattern[str]":
    """Compiled pattern for the id forms the live engine can legitimately produce.

    ``prompt_index=None`` accepts any index (the HTTP response omits the index entirely);
    ``prompt_index=0`` accepts only the index this single-prompt request actually used. The random
    8-hex suffix is accepted only *after* an index, mirroring the engine's own
    ``cmpl-<caller>-<index>-<8hex>`` form.
    """
    if not marked_request_id_ok(caller_request_id):
        raise BackendError(
            f"refusing to build an engine id pattern from {caller_request_id!r}: it is not a "
            f"server-generated {CALLER_REQUEST_ID_PREFIX}<hex> id"
        )
    index = r"[0-9]+" if prompt_index is None else str(int(prompt_index))
    return re.compile(
        rf"^{re.escape(ENGINE_MARKER_PREFIX + caller_request_id)}"
        rf"(-{index}(-[0-9a-f]{{{ENGINE_ID_SUFFIX_HEX_LEN}}})?)?$"
    )


def engine_id_matches(engine_id: Any, caller_request_id: str, prompt_index: int | None = None) -> bool:
    """True only for the accepted id forms of *this* caller (never another request or prefix)."""
    if not isinstance(engine_id, str) or not engine_id.startswith(ENGINE_REQUEST_ID_PREFIX):
        return False
    try:
        pattern = engine_id_pattern(caller_request_id, prompt_index)
    except BackendError:
        return False
    return bool(pattern.match(engine_id))


def resolve_engine_request_id(engine_response_id: Any, caller_request_id: str, prompt_index: int = 0) -> str:
    """Validate the engine's own response id against the accepted live forms.

    Accepts the live ``cmpl-<caller>`` form (no index), the documented ``cmpl-<caller>-<index>``
    form and the engine-internal ``cmpl-<caller>-<index>-<8hex>`` form. Any other id (another
    request, another prefix, a malformed suffix) is refused; an absent id falls back to the
    documented per-prompt rule.
    """
    expected = engine_request_id_for(caller_request_id, prompt_index)
    if engine_response_id in (None, ""):
        return expected
    if not engine_id_matches(engine_response_id, caller_request_id, prompt_index=None):
        raise CaptureError(
            f"engine returned an id outside the accepted forms for {caller_request_id!r}: "
            f"{engine_response_id!r}; refusing to guess which record belongs to this prompt"
        )
    return str(engine_response_id)


def discover_record_files(
    capture_dir: str,
    caller_request_id: str,
    prompt_hash: str,
    prompt_index: int = 0,
    marker_prefix: str = ENGINE_REQUEST_ID_PREFIX,
) -> list[tuple[pathlib.Path, str]]:
    """Find the ONE capture record for this caller + exact prompt hash (never a guess).

    Accepted file name: ``<engine id>__<prompt_hash[:16]>__<dup>.npz`` where the engine id is one of
    the live forms (``cmpl-<caller>`` / ``cmpl-<caller>-<index>`` / ``cmpl-<caller>-<index>-<8hex>``)
    for *this* caller. More than one candidate is an error. Symlinks, paths resolving outside the
    capture dir, a relative dir and a traversal-ish caller id are all refused.
    """
    real = _ensure_plain_dir(capture_dir)
    if not isinstance(caller_request_id, str) or not marked_request_id_ok(caller_request_id):
        raise CaptureError(
            f"caller request id {caller_request_id!r} is not a server-generated "
            f"{CALLER_REQUEST_ID_PREFIX}<hex> id"
        )
    _require_safe_token(caller_request_id, what="caller request id", exc=CaptureError)
    if not isinstance(prompt_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", prompt_hash):
        raise CaptureError(f"prompt_hash must be a 64-char hex digest, got {prompt_hash!r}")
    try:
        pattern = engine_id_pattern(caller_request_id, prompt_index)
    except BackendError as exc:
        raise CaptureError(str(exc)) from exc

    root = pathlib.Path(real)
    if not root.is_dir():
        return []                                    # the helper may not have created it yet
    hash16 = prompt_hash[:16]
    name_re = re.compile(
        rf"^(?P<id>[A-Za-z0-9._-]{{1,{KEY_MAX_LEN}}})__(?P<hash>[0-9a-fA-F]{{16}})"
        rf"__(?P<dup>[0-{MAX_DUP_DISAMBIGUATION}])\.npz$"
    )
    found: list[tuple[pathlib.Path, str]] = []
    for entry in sorted(root.iterdir()):
        match = name_re.match(entry.name)
        if match is None or match.group("hash") != hash16:
            continue
        engine_id = match.group("id")
        if not engine_id.startswith(marker_prefix) or not pattern.match(engine_id):
            continue
        if entry.is_symlink():
            raise CaptureError(f"capture record {entry.name!r} is a symlink; refusing to follow it")
        if not entry.is_file():
            continue
        resolved = os.path.realpath(str(entry))
        if not resolved.startswith(real + os.sep):
            raise CaptureError(f"capture record {entry.name!r} resolves outside the capture directory")
        found.append((entry, engine_id))
    if len(found) > 1:
        raise CaptureError(
            f"ambiguous capture records for {caller_request_id} / {hash16}: "
            f"{[p.name for p, _ in found]}; refusing to guess which one belongs to this prompt"
        )
    return found


def read_error_marker(capture_dir: str) -> dict | None:
    """Read the helper's CAPTURE_ERROR.json marker, if present and readable (never fatal by itself)."""
    path = pathlib.Path(capture_dir) / CAPTURE_ERROR_MARKER_NAME
    try:
        if path.is_symlink() or not path.is_file():
            return None
    except OSError:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "kind": "unreadable_error_marker",
            "error": f"{type(exc).__name__}: {exc}",
            "fatal": False,
            "marker_present": True,
        }
    if not isinstance(payload, dict):
        return {"kind": "unreadable_error_marker", "error": "marker is not a JSON object", "fatal": False,
                "marker_present": True}
    return payload


def error_marker_is_fatal(
    payload: Any, caller_request_id: str, engine_ids: Sequence[str] = ()
) -> tuple[bool, str | None]:
    """Does this marker mean *this* request cannot be captured?

    Fatal only when the marker identifies this request (one of its id fields is this caller's id, one
    of the accepted engine ids, or an accepted id form of this caller) or reports a global stop
    (``stopped: true`` or a global-stop kind such as ``quota_exceeded``). Helper-level errors and
    skip counters are diagnostics: they are surfaced and the request keeps polling.
    """
    if not isinstance(payload, dict):
        return False, None
    if payload.get("stopped") is True:
        return True, "the engine helper reports stopped=true, so no further captures will be written"
    last = payload.get("last_error") if isinstance(payload.get("last_error"), dict) else {}
    kind = last.get("kind") or payload.get("kind") or payload.get("reason")
    if isinstance(kind, str) and kind in GLOBAL_STOP_MARKER_KINDS:
        detail = last.get("error") or payload.get("error") or ""
        return True, f"global capture stop ({kind}): {detail}".strip()
    accepted = {str(v) for v in (engine_ids or ())}
    for scope in (last, payload):
        for key in MARKER_REQUEST_ID_KEYS:
            value = scope.get(key)
            if not isinstance(value, str) or not value:
                continue
            if value == caller_request_id or value in accepted or engine_id_matches(value, caller_request_id):
                return True, f"the error marker identifies this request ({key}={value})"
    return False, None


def read_refusal_marker(capture_dir: str) -> dict | None:
    """Read the helper's per-request refusal marker, if present and of the known schema.

    Never fatal by itself: an unreadable or unknown-schema file is ignored here (and surfaced in
    diagnostics), because guessing at a foreign format must not fail a request.
    """
    path = pathlib.Path(capture_dir) / CAPTURE_REFUSED_MARKER_NAME
    try:
        if path.is_symlink() or not path.is_file():
            return None
    except OSError:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != CAPTURE_REFUSED_SCHEMA:
        return None
    if not isinstance(payload.get("entries"), list):
        return None
    return payload


def refusal_for_request(
    payload: Any,
    caller_request_id: str,
    engine_ids: Sequence[str] = (),
    *,
    max_entries: int = MAX_REFUSAL_ENTRIES_SCANNED,
) -> dict | None:
    """The refusal entry that names *this* request, or None.

    The id rule is exactly the error marker's rule: an entry fails a request only when one of its
    id fields equals this caller's id, equals one of the accepted engine ids for this request, or
    matches an accepted engine id FORM of this caller. Another request's entry never matches, and
    a caller id can never be supplied by the client (it is server-generated), so a stale entry
    cannot fail a later request.
    """
    if not isinstance(payload, dict) or payload.get("schema") != CAPTURE_REFUSED_SCHEMA:
        return None
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return None
    accepted = {str(v) for v in (engine_ids or ())}
    for entry in entries[: max(int(max_entries), 1)]:
        if not isinstance(entry, dict):
            continue
        for key in MARKER_REQUEST_ID_KEYS:
            value = entry.get(key)
            if not isinstance(value, str) or not value:
                continue
            if value == caller_request_id or value in accepted or engine_id_matches(value, caller_request_id):
                return entry
    return None


def annotate_refusal(payload: Any, caller_request_id: str, engine_ids: Sequence[str] = ()) -> dict | None:
    """The refusal entry as disclosed to the caller (plus the fatality verdict for that request)."""
    entry = refusal_for_request(payload, caller_request_id, engine_ids)
    if entry is None:
        return None
    return {
        "marker_present": True,
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
        "helper_variant": entry.get("helper_variant"),
        "engine_request_id": entry.get("engine_request_id"),
        "reason": entry.get("reason"),
        "final_refusal": entry.get("final_refusal"),
        "batch_index": entry.get("batch_index"),
        "own_draft_tokens": entry.get("own_draft_tokens"),
        "batch_num_drafts": entry.get("batch_num_drafts"),
        "cu_span": entry.get("cu_span"),
        "logits_row_index": entry.get("logits_row_index"),
        "logits_index": entry.get("logits_index"),
        "query_start": entry.get("query_start"),
        "query_end": entry.get("query_end"),
        "computed_tokens_before_step": entry.get("computed_tokens_before_step"),
        "num_scheduled_tokens": entry.get("num_scheduled_tokens"),
        "prompt_len": entry.get("prompt_len"),
        "terminal_position": entry.get("terminal_position"),
        "wall_ns": entry.get("wall_ns"),
        "fatal": True,
        "reason_detail": (
            f"the engine helper refused this request's capture (reason={entry.get('reason')!r}); "
            "no capture record will be written, so the request cannot be answered"
        ),
    }


def annotate_error_marker(
    payload: dict, caller_request_id: str, engine_ids: Sequence[str] = ()
) -> dict:
    """Marker as disclosed to the caller: the helper's own fields plus the fatality verdict."""
    fatal, why = error_marker_is_fatal(payload, caller_request_id, engine_ids)
    last = payload.get("last_error") if isinstance(payload.get("last_error"), dict) else {}
    return {
        "marker_present": True,
        "kind": last.get("kind") or payload.get("kind"),
        "error": last.get("error") or payload.get("error"),
        "wall_ns": last.get("wall_ns"),
        "helper_version": payload.get("helper_version"),
        "stopped": payload.get("stopped"),
        "skips": payload.get("skips"),
        "error_count": payload.get("error_count"),
        "records_written": payload.get("records_written"),
        "fatal": fatal,
        "reason": why,
    }


def _require_safe_token(value: Any, *, what: str, exc: type[Exception]) -> str:
    """Path-hazard check for ONE opaque id / basename component (delegates to features).

    Interior dots are legitimate in opaque request/example ids and in on-disk basenames;
    separators, NUL, the exact tokens '.'/'..', a leading/trailing '..' and drive/UNC prefixes
    are refused. This is the same rule the extraction client and the record format use, so the
    gateway and the client agree on what a record id is.
    """
    try:
        return features.reject_unsafe_path_token(value, what=what)
    except features.FeatureError as err:
        raise exc(str(err)) from err


def _safe_engine_key(engine_request_id: str, marker_prefix: str = ENGINE_REQUEST_ID_PREFIX) -> str:
    if not isinstance(engine_request_id, str) or not engine_request_id:
        raise BackendError("engine_request_id must be a non-empty string")
    _require_safe_token(engine_request_id, what="engine request id", exc=BackendError)
    if not engine_request_id.startswith(marker_prefix):
        raise BackendError(
            f"engine request id {engine_request_id!r} does not start with the allowed marker "
            f"{marker_prefix!r}; the gateway will not look for a record it did not request"
        )
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in engine_request_id)
    if len(safe) > KEY_MAX_LEN or not ALLOWED_KEY_CHARS.match(safe):
        raise BackendError(f"refusing over-long or non-allowlisted record key {safe!r}")
    return safe


def expected_record_file_name(
    engine_request_id: str,
    prompt_hash: str,
    dup: int = 0,
    marker_prefix: str = ENGINE_REQUEST_ID_PREFIX,
) -> str:
    """``<safe engine id>__<prompt_hash[:16]>__<dup>.npz`` - the engine helper's own rule."""
    safe = _safe_engine_key(engine_request_id, marker_prefix)
    if not isinstance(prompt_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", prompt_hash):
        raise BackendError(f"prompt_hash must be a 64-char hex digest, got {prompt_hash!r}")
    if not isinstance(dup, int) or isinstance(dup, bool) or not (0 <= dup <= MAX_DUP_DISAMBIGUATION):
        raise BackendError(f"dup must be an int in 0..{MAX_DUP_DISAMBIGUATION}, got {dup!r}")
    return f"{safe}__{prompt_hash[:16]}__{dup}.npz"


def _ensure_plain_dir(dir_path: str | os.PathLike[str]) -> str:
    """Absolute, non-escaping directory path (a symlinked/relative target is refused)."""
    try:
        path_str = os.fspath(dir_path)
    except TypeError as exc:
        raise CaptureError(f"capture directory must be a path, got {type(dir_path).__name__}") from exc
    if not isinstance(path_str, str) or not path_str:
        raise CaptureError("capture directory must be a non-empty string")
    p = pathlib.Path(path_str)
    if not p.is_absolute():
        raise CaptureError(f"capture directory must be absolute, got {path_str!r}")
    if any(part == ".." for part in p.parts):
        raise CaptureError(f"capture directory must not contain '..': {path_str!r}")
    return os.path.realpath(path_str)


def resolve_record_files(
    capture_dir: str,
    engine_request_id: str,
    prompt_hash: str,
    marker_prefix: str = ENGINE_REQUEST_ID_PREFIX,
) -> list[pathlib.Path]:
    """Existing record files for this request, with symlinks and escapes refused."""
    real = _ensure_plain_dir(capture_dir)
    _require_safe_token(engine_request_id, what="engine request id", exc=CaptureError)
    found: list[pathlib.Path] = []
    for dup in range(MAX_DUP_DISAMBIGUATION + 1):
        try:
            name = expected_record_file_name(engine_request_id, prompt_hash, dup, marker_prefix)
        except BackendError as exc:
            raise CaptureError(str(exc)) from exc
        candidate = pathlib.Path(real) / name
        if candidate.is_symlink():
            raise CaptureError(f"capture record {name!r} is a symlink; refusing to follow it")
        if not candidate.is_file():
            continue
        resolved = os.path.realpath(str(candidate))
        if not resolved.startswith(real + os.sep):
            raise CaptureError(f"capture record {name!r} resolves outside the capture directory")
        found.append(candidate)
    return found


# ======================================================================================
# record decoding / validation
# ======================================================================================

def _npz_arrays(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {name: z[name].copy() for name in z.files}


def _npz_encode(arrays: Mapping[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def _rewrite_meta(blob: bytes, *, drop: Sequence[str] = (), add: Mapping[str, Any] | None = None) -> bytes:
    arrays = _npz_arrays(blob)
    if "meta_json" not in arrays:
        raise CaptureError("record is missing the meta_json array")
    meta = json.loads(bytes(np.asarray(arrays["meta_json"], dtype=np.uint8)).decode("utf-8"))
    for key in drop:
        meta.pop(key, None)
    if add:
        meta.update(dict(add))
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"), dtype=np.uint8
    )
    return _npz_encode(arrays)


def decode_record_bytes(blob: bytes, *, tolerate_unknown_meta_fields: bool = False) -> tuple[features.FeatureRecord, tuple[str, ...]]:
    """Decode a capture record through the shared ``features`` reader.

    Unknown meta fields are a *hard* failure by default: the engine helper writes at least one
    field (``raw_vector_dtype``) that the shared reader does not know, so a silent drop would
    hide a real schema divergence. ``tolerate_unknown_meta_fields=True`` is the explicit
    operator escape hatch; the dropped field names are returned and reported.

    A record that carries the wide-label capture extension is refused here unconditionally: the
    legacy 24-slot reader must never consume (or silently strip) a wide record, and the operator
    hatch is deliberately not extended to it.
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise CaptureError(f"record payload must be bytes, got {type(blob).__name__}")
    if record_has_wide_extension(bytes(blob)):
        raise CaptureError(
            "capture record carries a wide-label extension (wide_* metadata); the legacy 24-slot "
            "reader refuses it - use the wide capture reader"
        )
    try:
        return features.decode_record_npz(bytes(blob)), ()
    except features.FeatureError as exc:
        message = str(exc)
        unknown = _unknown_meta_fields(message)
        if tolerate_unknown_meta_fields and unknown and not any(k.startswith("wide_") for k in unknown):
            try:
                rec = features.decode_record_npz(_rewrite_meta(bytes(blob), drop=unknown))
            except features.FeatureError as inner:  # pragma: no cover - defensive
                raise CaptureError(f"feature record rejected: {inner}") from inner
            return rec, tuple(sorted(unknown))
        raise CaptureError(f"feature record rejected: {message}") from exc
    except Exception as exc:  # noqa: BLE001 - any decode failure is a corrupt record
        raise CaptureError(f"corrupt feature record: {type(exc).__name__}: {exc}") from exc


def record_has_wide_extension(blob: bytes) -> bool:
    """True when this npz payload carries the wide-label capture extension.

    Reads only the ``meta_json`` array; a corrupt payload returns False so the caller's normal
    corrupt-record path reports it.
    """
    try:
        arrays = _npz_arrays(bytes(blob))
        if "meta_json" not in arrays:
            return False
        meta = json.loads(bytes(np.asarray(arrays["meta_json"], dtype=np.uint8)).decode("utf-8"))
    except Exception:  # noqa: BLE001 - corruption is reported by the decode path
        return False
    if not isinstance(meta, dict):
        return False
    return any(isinstance(k, str) and k.startswith("wide_") for k in meta)


def _unknown_meta_fields(message: str) -> tuple[str, ...]:
    marker = "record meta has unknown fields:"
    if marker not in message:
        return ()
    tail = message.split(marker, 1)[1].strip()
    try:
        parsed = ast.literal_eval(tail)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return ()
    if isinstance(parsed, (list, tuple)) and all(isinstance(v, str) for v in parsed):
        return tuple(parsed)
    return ()


def fixture_only_npz_bytes(rec: features.FeatureRecord, extra_meta: Mapping[str, Any] | None = None) -> bytes:
    """FIXTURE-ONLY: encode a record, optionally with extra meta keys (engine-schema doubles)."""
    blob = features.encode_record_npz(rec)
    if not extra_meta:
        return blob
    return _rewrite_meta(blob, add=extra_meta)


def validate_record(
    rec: Any,
    *,
    expected_engine_request_id: str | None = None,
    expected_prompt_token_sha256: str | None = None,
    expected_prompt_len: int | None = None,
    not_before_wall_ns: int | None = None,
    require_terminal_is_last: bool = True,
    active_slot_indices: Sequence[int] | None = None,
) -> None:
    """Fail-closed validation of a capture record against what this request asked for.

    ``FeatureRecord`` only bounds-checks ``terminal_position``; the decision boundary requires
    the terminal vector to be the *last* prompt token, and every identity field to match.

    ``active_slot_indices`` (when supplied) restricts the finiteness requirement of the raw
    24-slot logits to the slots this request can actually read: inactive slots are masked before
    normalisation and can never carry mass, so a placeholding ``-inf`` there is harmless, while a
    non-finite *active* logit would silently corrupt the decision.
    """
    if not isinstance(rec, features.FeatureRecord):
        raise CaptureError(f"expected a FeatureRecord, got {type(rec).__name__}")
    if expected_engine_request_id is not None and rec.engine_request_id != expected_engine_request_id:
        raise CaptureError(
            f"record engine_request_id {rec.engine_request_id!r} does not match the request "
            f"{expected_engine_request_id!r}"
        )
    if expected_prompt_token_sha256 is not None and rec.prompt_token_sha256 != expected_prompt_token_sha256:
        raise CaptureError(
            "record prompt_token_sha256 does not match the token ids this request sent "
            f"({rec.prompt_token_sha256[:12]}... vs {expected_prompt_token_sha256[:12]}...)"
        )
    if expected_prompt_len is not None and rec.prompt_len != expected_prompt_len:
        raise CaptureError(f"record prompt_len {rec.prompt_len} != requested {expected_prompt_len}")
    if require_terminal_is_last and rec.terminal_position != rec.prompt_len - 1:
        raise CaptureError(
            f"record terminal_position {rec.terminal_position} is not the last prompt token "
            f"(prompt_len={rec.prompt_len}); the decision boundary is not this prompt's last position"
        )
    if not rec.logits_captured:
        raise CaptureError(
            f"record has no captured raw logits ({rec.logits_absent_reason!r}); the B0 readout "
            "cannot be derived from raw captured logits"
        )
    vec = rec.decode_vector_fp32()
    if vec.shape != (head_mod.FEATURE_WIDTH,) or vec.dtype != np.float32:
        raise CaptureError(f"record vector is {vec.dtype}{vec.shape}, expected float32 (2560,)")
    if not np.isfinite(vec).all():
        raise CaptureError("record feature vector contains non-finite values")
    if len(rec.verbalizer_ids) != protocol.NUM_SLOTS or len(set(rec.verbalizer_ids)) != protocol.NUM_SLOTS:
        raise CaptureError("record verbalizer_ids are not 24 unique slot rows")
    logits = np.asarray(rec.verbalizer_logits_fp32, dtype=np.float32)
    if logits.shape != (protocol.NUM_SLOTS,):
        raise CaptureError("record verbalizer logits are not 24 values")
    read_slots = (
        [int(i) for i in active_slot_indices]
        if active_slot_indices is not None
        else list(range(protocol.NUM_SLOTS))
    )
    bad = [i for i in read_slots if i < 0 or i >= protocol.NUM_SLOTS or not np.isfinite(logits[i])]
    if bad:
        raise CaptureError(f"record raw logits are non-finite in readable slot(s) {bad}")
    if active_slot_indices is None and not np.isfinite(logits).all():
        raise CaptureError("record raw logits contain non-finite values")
    if not_before_wall_ns is not None and int(rec.capture_wall_ns) < int(not_before_wall_ns):
        raise CaptureError(
            f"capture_wall_ns {rec.capture_wall_ns} predates this request ({not_before_wall_ns}); "
            "the record is stale, not evidence for this prompt"
        )


# ======================================================================================
# captured feature
# ======================================================================================

@dataclasses.dataclass
class CapturedFeature:
    """One verified feature capture, plus everything the response must disclose about it."""

    record: features.FeatureRecord
    vector_fp32: np.ndarray
    slot_logits_fp32: np.ndarray
    caller_request_id: str
    engine_request_id: str
    engine_response_id: str
    usage: dict
    timing_ms: dict
    transport: str
    provider_label: str
    caller_prompt_token_sha256: str
    cache_raw: dict = dataclasses.field(default_factory=dict)
    cache_evidence: str = CACHE_EVIDENCE_UNVERIFIED
    completion_text_empty: bool = True
    completion_finish_reason: str | None = None
    dropped_meta_fields: tuple[str, ...] = ()
    provider_kind: str = "engine"
    error_marker: dict | None = None
    # The model the engine's own response named for this request (None when the response carried
    # no model field). Only a configured ``expected_served_model`` makes it a hard requirement.
    served_model: str | None = None

    def cache_block(self) -> dict:
        raw = self.record.cache_hit_tokens
        return {
            "source_field": "cache_hit_tokens",
            "value": None if raw is None else int(raw),
            "num_scheduled_tokens": int(self.record.num_scheduled_tokens),
            "record_declared_provenance": getattr(self.record, "cache_hit_provenance", None),
            "record_declared_source": getattr(self.record, "cache_hit_source", None),
            "record_status_reason": getattr(self.record, "cache_hit_status_reason", None),
            "computed_tokens_before_step": getattr(self.record, "computed_tokens_before_step", None),
            "cache_evidence": self.cache_evidence,
            "note": (
                "The S3 record field cache_hit_tokens may count previously computed tokens rather "
                "than genuine reused KV. It is labelled unverified until PM live instrumentation "
                "establishes the actual reused count; no genuine prefill reuse is claimed here, and "
                "a 'measured' declaration in the record is not treated as PM-validated evidence."
            ),
        }


# ======================================================================================
# HTTP capture provider
# ======================================================================================

@dataclasses.dataclass
class CaptureProviderConfig:
    base_url: str
    model: str
    capture_dir: str
    capture_dir_allowlist: Sequence[str] = ()
    tokenizer_revision: str | None = None
    template_id: str = templates.TEMPLATE_AUTOTRUST_RAW
    template_version: str | None = None
    chat_template_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    add_special_tokens: bool = False
    socket_timeout_s: float = 30.0
    request_timeout_s: float = 120.0
    capture_deadline_s: float = 30.0
    poll_initial_ms: float = 5.0
    poll_max_ms: float = 100.0
    poll_backoff: float = 1.5
    record_max_bytes: int = 8 * 1024 * 1024
    max_completion_tokens: int = 1
    tolerate_unknown_meta_fields: bool = False
    marker_prefix: str = ENGINE_REQUEST_ID_PREFIX
    caller_request_id_prefix: str = CALLER_REQUEST_ID_PREFIX
    provider_label: str = "engine"
    provider_kind: str = "engine"
    # When set, the engine's own completion response must name exactly this model, and the observed
    # value is reported as ``CapturedFeature.served_model``. This is what prevents a base-model
    # response from being presented as adapter-on: a requested body model, a cache key or a config
    # value is not served identity.
    expected_served_model: str | None = None
    # Opt-in wide-label lane (None/disabled => no wide capture, no wide prompt surface).
    wide_choice: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.startswith(("http://", "https://")):
            raise ConfigError(f"engine base_url must be an http(s) URL, got {self.base_url!r}")
        if not isinstance(self.model, str) or not self.model:
            raise ConfigError("engine model must be a non-empty string")
        try:
            raw_dir = os.fspath(self.capture_dir)
        except TypeError as exc:
            raise ConfigError(f"capture dir must be a path, got {type(self.capture_dir).__name__}") from exc
        if not pathlib.Path(raw_dir).is_absolute() or any(part == ".." for part in pathlib.Path(raw_dir).parts):
            raise ConfigError(f"capture dir must be an absolute path without '..', got {raw_dir!r}")
        try:
            self.capture_dir = _ensure_plain_dir(raw_dir)
        except CaptureError as exc:
            raise ConfigError(str(exc)) from exc
        allow = []
        for entry in self.capture_dir_allowlist or ():
            if not isinstance(entry, str) or not entry:
                raise ConfigError("capture dir_allowlist entries must be non-empty strings")
            p = pathlib.Path(entry)
            if not p.is_absolute() or any(part == ".." for part in p.parts):
                raise ConfigError(f"capture dir_allowlist entry must be an absolute path: {entry!r}")
            allow.append(os.path.realpath(str(p)))
        if allow:
            inside = any(self.capture_dir == a or self.capture_dir.startswith(a + os.sep) for a in allow)
            if not inside:
                raise ConfigError(
                    f"capture dir {self.capture_dir!r} is not inside the configured allowlist "
                    f"{allow!r}; the operator must add it explicitly"
                )
        if self.template_id not in templates.TEMPLATES:
            raise ConfigError(f"unknown template id {self.template_id!r}")
        if self.template_version is not None:
            known = templates.TEMPLATES[self.template_id].version
            if known != self.template_version:
                raise ConfigError(
                    f"frozen template {self.template_id!r} is version {known!r} but the config "
                    f"declares {self.template_version!r}; refusing to reinterpret the surface"
                )
        if not (isinstance(self.max_completion_tokens, int) and self.max_completion_tokens == 1):
            raise ConfigError(
                "this transport schedules exactly one discarded internal completion token "
                f"(got max_completion_tokens={self.max_completion_tokens!r})"
            )
        for name in ("socket_timeout_s", "request_timeout_s", "capture_deadline_s", "poll_initial_ms", "poll_max_ms"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not (value > 0) or not np.isfinite(value):
                raise ConfigError(f"{name} must be a positive finite number, got {value!r}")
        if self.poll_max_ms < self.poll_initial_ms:
            raise ConfigError("poll_max_ms must be >= poll_initial_ms")
        if not isinstance(self.poll_backoff, (int, float)) or self.poll_backoff < 1.0:
            raise ConfigError("poll_backoff must be >= 1.0")
        if not isinstance(self.record_max_bytes, int) or isinstance(self.record_max_bytes, bool) or self.record_max_bytes <= 0:
            raise ConfigError("record_max_bytes must be a positive int")
        if self.expected_served_model is not None and (
            not isinstance(self.expected_served_model, str) or not self.expected_served_model
        ):
            raise ConfigError(
                "expected_served_model must be null or a non-empty string (the alias the engine's "
                f"own response must name), got {self.expected_served_model!r}"
            )
        self.chat_template_kwargs = dict(self.chat_template_kwargs or {})
        if self.wide_choice is not None and not isinstance(self.wide_choice, wide.WideChoiceConfig):
            raise ConfigError(
                "wide_choice must be a validated wide.WideChoiceConfig (or absent); refusing an "
                f"unvalidated wide configuration ({type(self.wide_choice).__name__})"
            )

    @property
    def split(self) -> urllib.parse.SplitResult:
        return urllib.parse.urlsplit(self.base_url)

    @property
    def host(self) -> str:
        return self.split.hostname or "127.0.0.1"

    @property
    def port(self) -> int:
        return self.split.port or (443 if self.split.scheme == "https" else 80)


def capture_provider_config(**kwargs: Any) -> CaptureProviderConfig:
    """Build a validated provider config (raises :class:`ConfigError` on unsafe input)."""
    known = {f.name for f in dataclasses.fields(CaptureProviderConfig)}
    unknown = set(kwargs) - known
    if unknown:
        raise ConfigError(f"unknown capture provider option(s): {sorted(unknown)}")
    return CaptureProviderConfig(**kwargs)


class HttpEngineCaptureProvider:
    """Acquire features from the same engine over real HTTP, bounded and fail-closed.

    Steps (all against the engine's own endpoints, no fabricated ids):

    1. render the frozen surface for the request: the raw template (``render``) for
       ``autotrust_raw_bare-v1``, or the frozen messages (``render_messages``) for a chat
       template such as ``qwen_chat_letters-v1``,
    2. ``POST /tokenize`` -> the exact token ids. Raw mode sends the prompt text with
       ``add_special_tokens=false``; chat mode sends ``messages`` + ``add_generation_prompt=true``
       + the configured ``chat_template_kwargs`` and no ``prompt`` at all, so the engine's chat
       template is the only renderer of the surface,
    3. ``POST /v1/completions`` with ``prompt=[ids]`` and a server-generated
       ``request_id`` so the engine-side helper writes a capture record,
    4. poll the operator-configured capture directory (bounded, backoff, size-capped,
       symlink/escape-denying) and verify the record against this exact request.

    In both modes the request identity is the hash of the exact ids the engine returned, and the
    completion/capture/readout path is identical.
    """

    label = "engine"
    kind = "engine"

    def __init__(self, config: CaptureProviderConfig) -> None:
        self.config = config
        self.last_error_marker: dict | None = None
        # M1: last per-request refusal disclosed to the caller (diagnostic only; never a verdict
        # for a different request).
        self.last_refusal: dict | None = None

    # -- HTTP plumbing -----------------------------------------------------------------
    def _connection(self) -> http.client.HTTPConnection:
        return self._connection_with_timeout(self.config.socket_timeout_s)

    def _connection_with_timeout(self, timeout: float) -> http.client.HTTPConnection:
        """A fresh request-local connection with an explicit socket timeout (no shared state)."""
        cfg = self.config
        if cfg.split.scheme == "https":
            return http.client.HTTPSConnection(cfg.host, cfg.port, timeout=timeout)
        return http.client.HTTPConnection(cfg.host, cfg.port, timeout=timeout)

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> tuple[int, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            conn = self._connection()
        except OSError as exc:  # pragma: no cover - constructor rarely fails
            raise TransportError(f"cannot open a connection to the engine: {exc}") from exc
        try:
            conn.request(
                "POST", path, body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            raw = resp.read()
            status = resp.status
        except (OSError, http.client.HTTPException) as exc:
            raise TransportError(f"engine {path} request failed: {type(exc).__name__}: {exc}") from exc
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        text = raw.decode("utf-8", errors="replace") if raw else ""
        if status < 200 or status >= 300:
            raise TransportError(
                f"engine {path} returned HTTP {status}: {text[:400]}"
            )
        try:
            return status, json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise TransportError(f"engine {path} returned non-JSON body: {text[:200]!r}") from exc

    # -- steps -------------------------------------------------------------------------
    def _tokenize_payload(self, *, text: str | None = None, messages: Sequence[Mapping[str, Any]] | None = None) -> dict:
        """The exact ``/tokenize`` body for one representation.

        Raw mode is unchanged (``prompt`` + ``add_special_tokens``, and ``chat_template_kwargs``
        only when configured). Chat mode sends ``messages`` + ``add_generation_prompt`` +
        ``chat_template_kwargs`` and NEVER a ``prompt`` key, so the engine's own chat template is
        the only thing that renders the surface (PM live fact, reports/pm/jevbench/token-surface-*).
        """
        if messages is not None:
            return {
                "model": self.config.model,
                "messages": [dict(m) for m in messages],
                "add_generation_prompt": True,
                "chat_template_kwargs": dict(self.config.chat_template_kwargs),
            }
        payload = {
            "model": self.config.model,
            "prompt": text,
            "add_special_tokens": bool(self.config.add_special_tokens),
        }
        if self.config.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.config.chat_template_kwargs)
        return payload

    def _post_tokenize(self, payload: Mapping[str, Any]) -> list[int]:
        _, doc = self._post_json("/tokenize", payload)
        return self.validate_tokenize_doc(doc, source="/tokenize")

    def validate_tokenize_doc(self, doc: Any, *, source: str = "/tokenize") -> list[int]:
        """Validate a ``/tokenize`` body and return its exact token ids (shared by both lanes)."""
        if not isinstance(doc, dict):
            raise TransportError(f"{source} returned {type(doc).__name__}, expected an object")
        ids = doc.get("tokens", doc.get("input_ids"))
        if "count" not in doc or not isinstance(ids, list) or not ids:
            raise TransportError(f"{source} response is unusable: {sorted(doc)}")
        if int(doc["count"]) != len(ids):
            raise TransportError(
                f"{source} count {doc['count']} disagrees with the {len(ids)} returned token ids"
            )
        out: list[int] = []
        for i, token in enumerate(ids):
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise TransportError(f"{source} token[{i}] is not a non-negative int: {token!r}")
            out.append(int(token))
        return out

    def _tokenize(self, text: str) -> list[int]:
        return self._post_tokenize(self._tokenize_payload(text=text))

    def _tokenize_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        return self._post_tokenize(self._tokenize_payload(messages=messages))

    def _completion_body(self, ids: Sequence[int], caller_request_id: str, seed: int | None) -> dict:
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little")
        return {
            "model": self.config.model,
            "prompt": [int(v) for v in ids],          # explicit ids: no chat templating a second time
            "max_tokens": 1,                           # one discarded internal token
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": seed,
            "stream": False,
            "add_special_tokens": False,
            "logprobs": None,
            "echo": False,
            "request_id": caller_request_id,
        }

    def _wait_for_record(
        self,
        caller_request_id: str,
        prompt_hash: str,
        deadline_s: float,
        not_before_wall_ns: int,
        active_slot_indices: Sequence[int],
        accepted_engine_ids: Sequence[str] = (),
        wide_context: Mapping[str, Any] | None = None,
    ) -> tuple[bytes, pathlib.Path, str, float, dict | None]:
        """Poll the capture dir for the ONE record of this caller + prompt, bounded by ``deadline_s``.

        A CAPTURE_ERROR.json marker fails this request only when it identifies *this* request or
        reports a global stop; otherwise it is recorded as a diagnostic and polling continues (PM
        live fact 3: helper-level errors such as row_export_failed and skip counters are not about a
        given request).
        """
        cfg = self.config
        start = time.monotonic()
        delay = cfg.poll_initial_ms / 1000.0
        marker_for_request: dict | None = None
        while True:
            elapsed = time.monotonic() - start
            if elapsed > deadline_s:
                detail = ""
                if marker_for_request is not None:
                    detail = (
                        f"; a non-fatal capture marker was present "
                        f"(kind={marker_for_request.get('kind')!r})"
                    )
                raise DeadlineExceeded(
                    f"no capture record for caller id {caller_request_id!r} after {deadline_s:.3f}s "
                    f"(capture dir {cfg.capture_dir}){detail}"
                )
            marker = read_error_marker(cfg.capture_dir)
            if marker is not None:
                annotated = annotate_error_marker(marker, caller_request_id, accepted_engine_ids)
                marker_for_request = annotated
                self.last_error_marker = annotated
                if annotated["fatal"]:
                    raise CaptureError(
                        f"engine refused the capture: {annotated.get('kind')!r} "
                        f"({annotated.get('reason')}); {annotated.get('error') or ''}".strip()
                    )
            for path, engine_id in discover_record_files(
                cfg.capture_dir, caller_request_id, prompt_hash, 0, cfg.marker_prefix
            ):
                size = path.stat().st_size
                if size > cfg.record_max_bytes:
                    raise CaptureError(
                        f"capture record {path.name!r} is {size} bytes, over the {cfg.record_max_bytes}-byte "
                        "cap; refusing to read it"
                    )
                if size <= 0:
                    continue
                blob = path.read_bytes()
                if wide_context is not None:
                    # Wide lane: a dedicated strict reader. A missing extension is a hard failure
                    # (never an empty/placeholder probability vector), and the legacy decoder is
                    # not used at all.
                    if not wide.is_wide_engine_id(engine_id):
                        raise CaptureError(
                            f"wide request produced a non-wide engine id {engine_id!r}; refusing"
                        )
                    wide.decode_wide_capture(
                        blob,
                        catalog=wide_context["catalog"],
                        expected_request_uuid=wide_context["request_uuid"],
                        expected_engine_request_id=engine_id,
                        expected_prompt_token_sha256=prompt_hash,
                        expected_prompt_len=int(wide_context["prompt_len"]),
                        n_options=int(wide_context["n_options"]),
                        not_before_wall_ns=not_before_wall_ns,
                    )
                    return blob, path, engine_id, elapsed, marker_for_request
                rec, dropped = decode_record_bytes(
                    blob, tolerate_unknown_meta_fields=cfg.tolerate_unknown_meta_fields
                )
                validate_record(
                    rec,
                    expected_engine_request_id=engine_id,
                    expected_prompt_token_sha256=prompt_hash,
                    not_before_wall_ns=not_before_wall_ns,
                    active_slot_indices=active_slot_indices,
                )
                return blob, path, engine_id, elapsed, marker_for_request
            # M1: no record this poll. If the helper has refused THIS request for good (final
            # refusal), fail fast with the typed refusal instead of waiting out the deadline.
            refusal = read_refusal_marker(cfg.capture_dir)
            if refusal is not None:
                annotated_refusal = annotate_refusal(refusal, caller_request_id, accepted_engine_ids)
                if annotated_refusal is not None:
                    self.last_refusal = annotated_refusal
                    raise CaptureRefused(
                        f"engine refused the capture: reason={annotated_refusal.get('reason')!r} "
                        f"(engine_request_id={annotated_refusal.get('engine_request_id')!r}, "
                        f"batch_index={annotated_refusal.get('batch_index')}, "
                        f"own_draft_tokens={annotated_refusal.get('own_draft_tokens')}, "
                        f"cu_span={annotated_refusal.get('cu_span')}, "
                        f"batch_num_drafts={annotated_refusal.get('batch_num_drafts')}); "
                        "no capture record will be written",
                        reason=annotated_refusal.get("reason"),
                        refusal=annotated_refusal,
                    )
            time.sleep(max(delay, 0.0005))
            delay = min(delay * cfg.poll_backoff, cfg.poll_max_ms / 1000.0)

    # -- diagnostics -------------------------------------------------------------------
    def diagnostics(self) -> dict:
        """Last capture-error marker seen (for /health); never a per-request verdict."""
        marker = self.last_error_marker
        refusal_file = read_refusal_marker(self.config.capture_dir) if self.config.capture_dir else None
        return {
            "error_marker_seen": marker is not None,
            "error_marker": marker,
            "error_marker_rule": (
                "a marker fails a request only when it identifies that request or reports a global "
                "stop (stopped=true or a global-stop kind such as quota_exceeded); helper-level "
                "errors and skip counters are diagnostics and the request keeps polling"
            ),
            "refusal_marker_seen": refusal_file is not None,
            "refusal_marker_file": CAPTURE_REFUSED_MARKER_NAME,
            "refusal_marker_schema": (refusal_file or {}).get("schema") if refusal_file else None,
            "refusal_marker_entry_count": (
                len(refusal_file.get("entries") or []) if refusal_file is not None else None
            ),
            "refusal_marker_truncated": (refusal_file or {}).get("truncated") if refusal_file else None,
            "refusal_for_last_request": self.last_refusal,
            "refusal_marker_rule": (
                "an entry fails ONLY the request it names (its engine_request_id / caller id, or an "
                "accepted id form of that caller) with HTTP 503 capture_refused; every other "
                "request and the global-stop behaviour are unchanged. Entries are written only for "
                "a FINAL refusal: the engine step computed the marked prompt's terminal token and "
                "the helper refused fail-closed, so no record will ever appear"
            ),
        }

    # -- public ------------------------------------------------------------------------
    def acquire_wide(
        self,
        request: wide.WideChoiceRequest,
        catalog: wide.WideCatalog,
        *,
        deadline_s: float | None = None,
        seed: int | None = None,
    ) -> CapturedWideFeature:
        """Acquire a wide (17..256 option) capture over the same HTTP transport."""
        return _http_acquire_wide(self, request, catalog, deadline_s=deadline_s, seed=seed)

    def acquire(self, request: protocol.DecisionRequest, *, deadline_s: float | None = None, seed: int | None = None) -> CapturedFeature:
        if not isinstance(request, protocol.DecisionRequest):
            raise BackendError(f"acquire() expects a DecisionRequest, got {type(request).__name__}")
        cfg = self.config
        deadline = float(deadline_s) if deadline_s is not None else float(cfg.capture_deadline_s)
        if not np.isfinite(deadline) or deadline <= 0:
            raise BackendError(f"deadline_s must be positive and finite, got {deadline!r}")
        not_before = time.time_ns()

        t0 = time.monotonic()
        if templates.is_chat_template(cfg.template_id):
            # chat representation: the engine's own chat template renders the surface from the
            # frozen messages; the ids it returns are the request identity, exactly as in raw mode
            ids = self._tokenize_messages(templates.render_messages(request, cfg.template_id))
        else:
            text = templates.render(request, cfg.template_id)
            ids = self._tokenize(text)
        t1 = time.monotonic()
        prompt_hash = features.token_sequence_sha256(ids)

        caller_request_id = new_caller_request_id()
        body = self._completion_body(ids, caller_request_id, seed)
        t2 = time.monotonic()
        _, doc = self._post_json("/v1/completions", body)
        t3 = time.monotonic()
        if not isinstance(doc, dict):
            raise TransportError(f"/v1/completions returned {type(doc).__name__}, expected an object")
        observed_model = doc.get("model")
        if cfg.expected_served_model is not None:
            # served identity: only the engine's own response names the model that answered. A
            # requested body model, a cache key or a config value is NOT evidence of adapter-on.
            if not isinstance(observed_model, str) or not observed_model:
                raise CaptureError(
                    f"the engine's /v1/completions response carries no served model "
                    f"(model={observed_model!r}) but this arm requires the adapter alias "
                    f"{cfg.expected_served_model!r}; refusing to treat the response as adapter-on"
                )
            if observed_model != cfg.expected_served_model:
                raise CaptureError(
                    f"engine served model {observed_model!r} does not equal the requested adapter "
                    f"alias {cfg.expected_served_model!r}; refusing the response (a base-model "
                    "response is not adapter-on evidence)"
                )
        served_model = observed_model if isinstance(observed_model, str) else None
        engine_response_id = doc.get("id")
        # accepted live forms: `cmpl-<caller>` (no index), `cmpl-<caller>-<index>`,
        # `cmpl-<caller>-<index>-<8hex>`; anything else is refused
        accepted_response_id = resolve_engine_request_id(engine_response_id, caller_request_id, 0)
        usage = doc.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens != len(ids):
            raise CaptureError(
                f"engine usage.prompt_tokens {prompt_tokens!r} does not equal the {len(ids)} token ids "
                "this request sent; the capture would not be request-aligned"
            )
        choices = doc.get("choices") or []
        empty = True
        finish_reason = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            text_out = choices[0].get("text")
            empty = not (isinstance(text_out, str) and text_out != "")
            finish_reason = choices[0].get("finish_reason")
            text_out = None  # generated text is never returned or stored

        blob, path, engine_request_id, waited, marker = self._wait_for_record(
            caller_request_id,
            prompt_hash,
            max(deadline - (t2 - t0), 1e-3),
            not_before,
            request.active_slot_indices(),
            accepted_engine_ids=(accepted_response_id,),
        )
        rec, dropped = decode_record_bytes(blob, tolerate_unknown_meta_fields=cfg.tolerate_unknown_meta_fields)
        validate_record(
            rec,
            expected_engine_request_id=engine_request_id,
            expected_prompt_token_sha256=prompt_hash,
            expected_prompt_len=len(ids),
            not_before_wall_ns=not_before,
            active_slot_indices=request.active_slot_indices(),
        )
        t4 = time.monotonic()

        vector = rec.decode_vector_fp32()
        slot_logits = np.asarray(rec.verbalizer_logits_fp32, dtype=np.float32).copy()
        t5 = time.monotonic()

        return CapturedFeature(
            record=rec,
            vector_fp32=vector,
            slot_logits_fp32=slot_logits,
            caller_request_id=caller_request_id,
            engine_request_id=engine_request_id,
            engine_response_id=str(engine_response_id or accepted_response_id),
            usage={
                "prompt_tokens": len(ids),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
            timing_ms={
                "tokenize": (t1 - t0) * 1000.0,
                "request": (t3 - t2) * 1000.0,
                "capture_wait": waited * 1000.0,
                "readout": (t5 - t4) * 1000.0,
                "total": (t5 - t0) * 1000.0,
            },
            transport=TRANSPORT_ONE_DISCARDED_TOKEN,
            provider_label=cfg.provider_label,
            provider_kind=cfg.provider_kind,
            caller_prompt_token_sha256=prompt_hash,
            cache_raw={
                "cache_hit_tokens": None if rec.cache_hit_tokens is None else int(rec.cache_hit_tokens),
                "num_scheduled_tokens": int(rec.num_scheduled_tokens),
            },
            cache_evidence=CACHE_EVIDENCE_UNVERIFIED,
            completion_text_empty=bool(empty),
            completion_finish_reason=None if finish_reason is None else str(finish_reason),
            dropped_meta_fields=dropped,
            error_marker=marker,
            served_model=served_model,
        )


class FeatureProvider(Protocol):
    """Anything that can turn a typed request into a verified :class:`CapturedFeature`."""

    label: str

    def acquire(self, request: protocol.DecisionRequest, *, deadline_s: float | None = None, seed: int | None = None) -> CapturedFeature:
        ...


# ======================================================================================
# fixture engine (FIXTURE-ONLY synthetic feature provider)
# ======================================================================================

def _fixture_tokenize(text: str) -> list[int]:
    """Deterministic stand-in tokenizer: one id per UTF-8 byte."""
    return [1000 + (b % 512) for b in text.encode("utf-8")]


def _fixture_tokenize_messages(messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = True) -> list[int]:
    """Deterministic stand-in for a chat template: role markers + content, one id per byte.

    The synthetic fixture engine has no jinja template, so it renders the messages form
    deterministically instead of the raw prompt text. FIXTURE ONLY: this is plumbing for tests and
    is never evidence about the real tokenizer or the real chat template.
    """
    parts = [f"<|{m.get('role')}|>\n{m.get('content')}\n" for m in messages]
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return _fixture_tokenize("".join(parts))


def make_fixture_record(
    *,
    engine_request_id: str,
    prompt_ids: Sequence[int],
    vector_fp32: np.ndarray,
    slot_logits: np.ndarray,
    prompt_hash: str | None = None,
    capture_wall_ns: int | None = None,
    example_id: str = "fixture-example",
    **over: Any,
) -> features.FeatureRecord:
    """FIXTURE-ONLY: build a feature record with the same shape the engine helper writes."""
    prompt_hash = prompt_hash or features.token_sequence_sha256(list(prompt_ids))
    vec = np.asarray(vector_fp32, dtype=np.float32).reshape(-1)
    if vec.shape != (head_mod.FEATURE_WIDTH,):
        raise BackendError(f"fixture vector must be 2560 wide, got {vec.shape}")
    bits = features.fp32_to_bf16_bits(vec)
    wide = features.bf16_bits_to_fp32(bits)   # the writer stores BF16 bits; FP32 must agree exactly
    logits = np.asarray(slot_logits, dtype=np.float32).reshape(-1)
    if logits.shape != (protocol.NUM_SLOTS,):
        raise BackendError(f"fixture slot logits must be 24 wide, got {logits.shape}")
    prompt_len = len(prompt_ids)
    meta: dict[str, Any] = dict(
        example_id=example_id,
        engine_request_id=engine_request_id,
        prompt_token_sha256=prompt_hash,
        prompt_tokens_count=prompt_len,
        prompt_len=prompt_len,
        terminal_position=prompt_len - 1,
        cache_hit_tokens=0,
        num_scheduled_tokens=prompt_len,
        query_start=0,
        query_end=prompt_len,
        logits_row_index=0,
        batch_index=0,
        internal_index=0,
        vector_bf16=bits,
        vector_fp32=wide,
        verbalizer_ids=list(range(1000, 1000 + protocol.NUM_SLOTS)),
        verbalizer_logits_fp32=logits,
        runtime_argmax_token_id=1000,
        runtime_max_logit_fp32=float(np.max(logits)) if np.isfinite(logits).all() else 0.0,
        head_scale=1.0,
        head_soft_cap=None,
        head_org_vocab_size=248320,
        head_dtype="bfloat16",
        lm_head_tp_size=2,
        tp_rank=0,
        world_size=2,
        cudagraph_mode="FULL_DECODE_ONLY",
        kernel_mode="prefill",
        runner_source_sha256="f" * 64,
        capture_wall_ns=int(capture_wall_ns if capture_wall_ns is not None else time.time_ns()),
        logits_captured=True,
        logits_absent_reason=None,
        vector_source="fixture_only:synthetic",
        helper_version="fixture-only/1",
        engine_build_version="fixture-only",
    )
    meta.update(over)
    # The shared reader gates are deliberately strict (a decoded record outside this hatch can
    # never carry a non-terminal boundary or non-finite numbers). This builder is FIXTURE-ONLY and
    # exists precisely to *represent* invalid records so the gateway's fail-closed paths can be
    # exercised; the hatch does not weaken the reading side, which still refuses them.
    with features.fixture_records_allowed():
        return features.FeatureRecord(**meta)


def _fixture_vector(seed_text: str, seed: int, vector_source: str, fixed: np.ndarray | None) -> np.ndarray:
    if fixed is not None:
        return np.asarray(fixed, dtype=np.float32).reshape(-1)
    digest = int(features.token_sequence_sha256([int(seed) % 100000])[:8], 16)
    rng = np.random.default_rng(digest)
    return rng.standard_normal(head_mod.FEATURE_WIDTH).astype(np.float32) * 0.05


class _LocalFastThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the reverse-DNS lookup on bind (a ~1s Windows stall)."""

    daemon_threads = True
    poll_interval = 0.05

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port


class FixtureEngineServer:
    """FIXTURE-ONLY real localhost HTTP server that imitates the engine's client surface.

    It is a synthetic feature provider: ``/tokenize``, ``/v1/completions`` (writing a synthetic
    record into ``capture_dir``), ``/v1/chat/completions`` (JSON echo or SSE) and ``/v1/models``.
    ``fault`` injects the failure modes the gateway must survive. It is NOT the vLLM engine and
    no live claim may be derived from it.
    """

    def __init__(
        self,
        *,
        model: str = "fixture-model",
        capture_dir: str | os.PathLike[str],
        host: str = "127.0.0.1",
        port: int = 0,
        fault: str | None = None,
        seed: int = 12345,
        vector_source: str = "seeded",
        fixed_vector: np.ndarray | None = None,
        tokenizer_revision: str = "fixture-tok-rev",
        extra_meta: Mapping[str, Any] | None = None,
        stream_events: Sequence[str] | None = None,
        stream_delay_s: float = 0.0,
        empty_completion_text: bool = False,
        chat_error: tuple[int, dict] | None = None,
        record_id_mode: str = "suffixed",
        response_id_mode: str = "base",
        error_marker: Mapping[str, Any] | None = None,
        refusal_marker: Mapping[str, Any] | None = None,
        refusal_suppresses_record: bool = True,
        wide_choice: "wide.WideChoiceConfig | None" = None,
        wide_logits: Any = None,
    ) -> None:
        self.model = model
        self.capture_dir = pathlib.Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.requested_port = port
        self.fault = fault
        self.seed = seed
        self.vector_source = vector_source
        self.fixed_vector = fixed_vector
        self.tokenizer_revision = tokenizer_revision
        # PM live facts (2026-10-10): the HTTP response id is `cmpl-<caller>` (no index) while the
        # engine-internal record id is `cmpl-<caller>-<index>-<8hex>`. These defaults mirror that.
        if record_id_mode not in ("plain", "index", "suffixed"):
            raise BackendError(f"record_id_mode must be plain|index|suffixed, got {record_id_mode!r}")
        if response_id_mode not in ("base", "index"):
            raise BackendError(f"response_id_mode must be base|index, got {response_id_mode!r}")
        self.record_id_mode = record_id_mode
        self.response_id_mode = response_id_mode
        self.error_marker = dict(error_marker) if error_marker else None
        # M1: a fixture-level per-request refusal (the M1 helper's CAPTURE_REFUSED.json). `{caller}`
        # and `{engine_id}` placeholders are substituted with this request's ids.
        self.refusal_marker = dict(refusal_marker) if refusal_marker else None
        self.refusal_suppresses_record = bool(refusal_suppresses_record)
        self.extra_meta = dict(extra_meta or {})
        self.wide_choice = wide_choice
        self.wide_logits = wide_logits
        self.stream_events = list(stream_events) if stream_events is not None else [
            'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        self.stream_delay_s = float(stream_delay_s)
        self.empty_completion_text = empty_completion_text
        self.chat_error = chat_error

        self.tokenize_bodies: list[dict] = []
        self.completion_bodies: list[dict] = []
        self.chat_bodies: list[bytes] = []
        self.last_chat_headers: dict[str, str] = {}
        self.client_disconnects = 0
        self.stream_completed_send = False
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._connections: set[Any] = set()

    # -- lifecycle ---------------------------------------------------------------------
    @property
    def port(self) -> int:
        if self._httpd is None:
            raise BackendError("fixture engine is not started")
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "FixtureEngineServer":
        handler_owner = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # keep pytest output clean
                return

            @property
            def fx(self) -> "FixtureEngineServer":
                return handler_owner

            def setup(self) -> None:
                super().setup()
                with self.fx._lock:
                    self.fx._connections.add(self.connection)

            def finish(self) -> None:
                try:
                    with self.fx._lock:
                        self.fx._connections.discard(self.connection)
                finally:
                    super().finish()

            # -- helpers
            def _body(self) -> bytes:
                length = self.headers.get("Content-Length")
                if length is None:
                    return b""
                try:
                    n = int(length)
                except ValueError:
                    return b""
                return self.rfile.read(n) if n > 0 else b""

            def _json(self, status: int, payload: Any) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _error(self, status: int, message: str) -> None:
                self._json(status, {"error": {"message": message, "type": "FixtureEngineError"}})

            # -- routes
            def do_GET(self) -> None:  # noqa: N802
                path = urllib.parse.urlsplit(self.path).path
                if path in ("/health", "/v1/health"):
                    self._json(200, {"status": "ok", "fixture_only": True, "model": self.fx.model})
                elif path == "/v1/models":
                    self._json(200, {"object": "list", "data": [{
                        "id": self.fx.model, "object": "model", "root": self.fx.model,
                        "revision": self.fx.tokenizer_revision, "max_model_len": 900000,
                    }]})
                else:
                    self._error(404, f"fixture engine has no route {path}")

            def do_POST(self) -> None:  # noqa: N802
                path = urllib.parse.urlsplit(self.path).path
                raw = self._body()
                if path == "/tokenize":
                    return self._tokenize(raw)
                if path == "/detokenize":
                    try:
                        doc = json.loads(raw or b"{}")
                        ids = doc.get("tokens") or []
                        return self._json(200, {"prompt": ",".join(str(i) for i in ids)})
                    except json.JSONDecodeError:
                        return self._error(400, "fixture detokenize requires JSON")
                if path == "/v1/completions":
                    return self._completions(raw)
                if path == "/v1/chat/completions":
                    return self._chat(raw)
                return self._error(404, f"fixture engine has no route {path}")

            def _tokenize(self, raw: bytes) -> None:
                try:
                    doc = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    return self._error(400, "fixture tokenize requires JSON")
                with self.fx._lock:
                    self.fx.tokenize_bodies.append(doc)
                if isinstance(doc.get("messages"), list):
                    # chat representation: the fixture has no jinja template, so it renders the
                    # messages deterministically (see _fixture_tokenize_messages)
                    ids = _fixture_tokenize_messages(
                        doc["messages"], add_generation_prompt=bool(doc.get("add_generation_prompt", False))
                    )
                else:
                    prompt = doc.get("prompt")
                    if isinstance(prompt, list):
                        ids = [int(v) for v in prompt]
                    else:
                        ids = _fixture_tokenize(str(prompt or ""))
                if self.fx.fault == "bad_tokenize":
                    return self._json(200, {"count": len(ids) + 1, "tokens": ids})
                self._json(200, {
                    "count": len(ids), "max_model_len": 900000, "tokens": ids,
                    "token_strs": [f"<{i}>" for i in ids],
                })

            def _completions(self, raw: bytes) -> None:
                try:
                    doc = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    return self._error(400, "fixture completions requires JSON")
                with self.fx._lock:
                    self.fx.completion_bodies.append(doc)
                if self.fx.fault == "engine_500":
                    return self._error(500, "fixture engine internal error")
                if self.fx.fault == "engine_400":
                    return self._error(400, "fixture engine rejected the request")
                request_id = doc.get("request_id")
                if not isinstance(request_id, str) or not request_id.startswith(CALLER_REQUEST_ID_PREFIX):
                    return self._json(200, {
                        "id": "cmpl-fixture-plain-0", "object": "text_completion",
                        "model": self.fx.model, "received": doc,
                        "choices": [{"text": "fixture", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
                    })
                base_engine_id = engine_request_id_for(request_id, 0)
                ids = doc.get("prompt") or []
                if not isinstance(ids, list) or not ids:
                    return self._error(400, "capture requests must carry explicit token ids")
                ids = [int(v) for v in ids]
                record_engine_id = self.fx.record_engine_id(base_engine_id, request_id)
                self.fx._write_capture(record_engine_id, ids, caller_request_id=request_id)
                completion_tokens = 1
                usage_prompt = len(ids) + 1 if self.fx.fault == "usage_mismatch" else len(ids)
                text = "" if self.fx.empty_completion_text else "x"
                # live fact 1: the HTTP response id carries NO prompt-index suffix
                response_id = base_engine_id if self.fx.response_id_mode == "index" else f"cmpl-{request_id}"
                self._json(200, {
                    "id": response_id, "object": "text_completion", "model": self.fx.model,
                    "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": usage_prompt,
                        "completion_tokens": completion_tokens,
                        "total_tokens": usage_prompt + completion_tokens,
                    },
                })

            def _chat(self, raw: bytes) -> None:
                with self.fx._lock:
                    self.fx.chat_bodies.append(raw)
                    self.fx.last_chat_headers = {k.lower(): v for k, v in self.headers.items()}
                if self.fx.chat_error is not None:
                    status, payload = self.fx.chat_error
                    return self._json(status, payload)
                try:
                    doc = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    doc = None
                if isinstance(doc, dict) and doc.get("stream"):
                    return self._stream()
                if doc is None:
                    return self._error(400, "fixture chat requires JSON")
                self._json(200, {"id": "cmpl-fixture-chat-0", "object": "chat.completion",
                                 "model": self.fx.model, "received": doc})

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.fx.stream_completed_send = False
                try:
                    for event in self.fx.stream_events:
                        data = event.encode("utf-8")
                        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                        self.wfile.flush()
                        if self.fx.stream_delay_s:
                            time.sleep(self.fx.stream_delay_s)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    self.fx.stream_completed_send = True
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    with self.fx._lock:
                        self.fx.client_disconnects += 1

        httpd = _LocalFastThreadingHTTPServer((self.host, self.requested_port), _Handler)
        httpd.daemon_threads = True
        httpd.poll_interval = 0.05
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, name="fixture-engine", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd, self._thread = None, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
            # ThreadingHTTPServer's handler threads park on keep-alive reads; close them so the
            # fixture stops promptly instead of waiting for the socket timeout.
            with self._lock:
                if self._connections:
                    children = list(self._connections)
                    self._connections.clear()
                else:
                    children = []
            for sock in children:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
        if thread is not None:
            thread.join(timeout=10.0)

    def __enter__(self) -> "FixtureEngineServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- record writing -----------------------------------------------------------------
    def record_engine_id(self, base_engine_id: str, caller_request_id: str) -> str:
        """The engine-internal id the record is keyed by (live forms, deterministic for fixtures)."""
        if self.record_id_mode == "plain":
            return f"cmpl-{caller_request_id}"
        if self.record_id_mode == "index":
            return base_engine_id
        suffix = hashlib.sha256(caller_request_id.encode("ascii")).hexdigest()[:ENGINE_ID_SUFFIX_HEX_LEN]
        return f"{base_engine_id}-{suffix}"

    def _write_error_marker(self, caller_request_id: str | None = None) -> None:
        """Write CAPTURE_ERROR.json exactly as the helper does (fixture-level errors included)."""
        if not self.error_marker:
            return
        payload = json.loads(json.dumps(self.error_marker))
        if caller_request_id:
            text = json.dumps(payload)
            payload = json.loads(text.replace("{caller}", caller_request_id))
        path = self.capture_dir / CAPTURE_ERROR_MARKER_NAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_refusal_marker(self, engine_id: str, caller_request_id: str | None = None) -> None:
        """Write CAPTURE_REFUSED.json exactly as the M1 helper does (fixture-level refusals)."""
        if not self.refusal_marker:
            return
        text = json.dumps(self.refusal_marker)
        text = text.replace("{engine_id}", engine_id)
        text = text.replace("{caller}", "" if caller_request_id is None else caller_request_id)
        payload = json.loads(text)
        path = self.capture_dir / CAPTURE_REFUSED_MARKER_NAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _maybe_extend_wide(self, blob: bytes, caller_request_id: str | None) -> bytes:
        """FIXTURE-ONLY: attach the wide extension for a gateway-minted wide caller id."""
        if not caller_request_id or not caller_request_id.startswith(wide.WIDE_CALLER_PREFIX):
            return blob
        if self.wide_choice is None or not self.wide_choice.enabled:
            # emulate an engine without the wide lane: the record is written WITHOUT the extension
            return blob
        catalog = self.wide_choice.require_catalog()
        n_options, request_uuid = wide.wide_request_identity_from_caller_id(caller_request_id)
        logits = self.wide_logits
        if logits is None:
            logits = np.linspace(-2.0, 2.0, catalog.n_labels, dtype=np.float32)
        meta = json.loads(bytes(np.asarray(_npz_arrays(blob)["meta_json"], dtype=np.uint8)).decode("utf-8"))
        return wide.add_wide_extension(
            blob,
            catalog=catalog,
            template_digest=wide.WIDE_TEMPLATE_CONTENT_SHA256,
            wide_request_uuid=request_uuid,
            prompt_token_sha256=meta["prompt_token_sha256"],
            prompt_len=int(meta["prompt_len"]),
            n_options=n_options,
            logits256=logits,
        )

    def _write_capture(self, engine_id: str, ids: list[int], caller_request_id: str | None = None) -> None:
        fault = self.fault
        prompt_hash = features.token_sequence_sha256(ids)
        self._write_error_marker(caller_request_id)
        if self.refusal_marker:
            # the engine served the request but the helper refused the capture (M1 live case)
            self._write_refusal_marker(engine_id, caller_request_id)
            if self.refusal_suppresses_record:
                return
        if fault == "missing":
            return
        name = expected_record_file_name(engine_id, prompt_hash, 0)
        path = self.capture_dir / name
        if fault == "two_records":
            # two candidate records for the same request + prompt hash: the client must refuse
            rec = self._record(engine_id, ids, prompt_hash)
            blob = self._maybe_extend_wide(
                fixture_only_npz_bytes(rec, self.extra_meta or None), caller_request_id)
            path.write_bytes(blob)
            if caller_request_id:
                other = expected_record_file_name(
                    f"cmpl-{caller_request_id}", prompt_hash, 0
                )
                (self.capture_dir / other).write_bytes(blob)
            return
        if fault == "symlink":
            target = self.capture_dir / "symlink-target.npz"
            rec = self._record(engine_id, ids, prompt_hash)
            blob = self._maybe_extend_wide(
                fixture_only_npz_bytes(rec, self.extra_meta or None), caller_request_id)
            target.write_bytes(blob)
            try:
                if path.exists():
                    path.unlink()
                os.symlink(str(target), str(path))
            except (OSError, NotImplementedError):
                path.write_bytes(blob)
            return
        if fault == "corrupt_npz":
            path.write_bytes(b"PK\x03\x04 this is not a zip file at all")
            return
        if fault == "oversize":
            path.write_bytes(b"\x00" * (4 * 1024 * 1024))
            return
        rec = self._record(engine_id, ids, prompt_hash)
        blob = self._maybe_extend_wide(fixture_only_npz_bytes(rec, self.extra_meta or None), caller_request_id)
        path.write_bytes(blob)
        (pathlib.Path(str(path) + ".json")).write_text(json.dumps(rec.meta(), sort_keys=True), encoding="utf-8")

    def _record(self, engine_id: str, ids: list[int], prompt_hash: str) -> features.FeatureRecord:
        fault = self.fault
        vec = _fixture_vector("", self.seed, self.vector_source, self.fixed_vector)
        logits = np.linspace(-1.0, 1.0, protocol.NUM_SLOTS, dtype=np.float32)
        over: dict[str, Any] = {}
        if fault == "wrong_hash":
            over["prompt_token_sha256"] = features.token_sequence_sha256(ids[:-1] or ids)
        if fault == "bad_terminal":
            over["terminal_position"] = 0
        if fault == "stale":
            over["capture_wall_ns"] = 1
        if fault == "no_logits":
            over["logits_captured"] = False
            over["logits_absent_reason"] = "verbalizer_ids_unresolved"
        if fault == "nan_logits":
            logits = np.full(protocol.NUM_SLOTS, np.nan, dtype=np.float32)
        engine_for_record = engine_id
        if fault == "wrong_engine_id":
            engine_for_record = "cmpl-qfn-0000-0"
        return make_fixture_record(
            engine_request_id=engine_for_record,
            prompt_ids=ids,
            vector_fp32=vec,
            slot_logits=logits,
            prompt_hash=prompt_hash,
            **over,
        )


class FixtureFeatureProvider:
    """FIXTURE-ONLY provider: deterministic synthetic capture read back through the real path."""

    label = FIXTURE_ONLY_LABEL
    kind = "fixture"

    def __init__(self, *, seed: int = 12345, capture_dir: str | os.PathLike[str] | None = None,
                 template_id: str = templates.TEMPLATE_AUTOTRUST_RAW, fixed_vector: np.ndarray | None = None) -> None:
        self.seed = int(seed)
        self.fixed_vector = fixed_vector
        base = pathlib.Path(capture_dir) if capture_dir else pathlib.Path(os.path.abspath(".")) / "fixture-capture"
        base.mkdir(parents=True, exist_ok=True)
        self.capture_dir = base
        self.config = CaptureProviderConfig(
            base_url="http://fixture.invalid:1",
            model="fixture-model",
            capture_dir=str(base),
            tokenizer_revision="fixture-tok-rev",
            template_id=template_id,
            provider_label=FIXTURE_ONLY_LABEL,
            provider_kind="fixture",
        )
        self.tokenizer_revision = "fixture-tok-rev"
        self.wide_logits = None

    def acquire_wide(
        self,
        request: wide.WideChoiceRequest,
        catalog: wide.WideCatalog,
        *,
        deadline_s: float | None = None,
        seed: int | None = None,
    ) -> CapturedWideFeature:
        """FIXTURE-ONLY wide acquisition through the shared capture reader."""
        return _fixture_acquire_wide(self, request, catalog, deadline_s=deadline_s, seed=seed)

    def acquire(self, request: protocol.DecisionRequest, *, deadline_s: float | None = None, seed: int | None = None) -> CapturedFeature:
        t0 = time.monotonic()
        if templates.is_chat_template(self.config.template_id):
            msgs = templates.render_messages(request, self.config.template_id)
            ids = _fixture_tokenize_messages(msgs, add_generation_prompt=True)
            seed_text = json.dumps(msgs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            seed_text = templates.render(request, self.config.template_id)
            ids = _fixture_tokenize(seed_text)
        prompt_hash = features.token_sequence_sha256(ids)
        caller_request_id = new_caller_request_id()
        engine_id = engine_request_id_for(caller_request_id, 0)
        vec = _fixture_vector(seed_text, self.seed if seed is None else int(seed), "seeded", self.fixed_vector)
        logits = np.linspace(-1.0, 1.0, protocol.NUM_SLOTS, dtype=np.float32)
        rec = make_fixture_record(
            engine_request_id=engine_id, prompt_ids=ids, vector_fp32=vec, slot_logits=logits,
            prompt_hash=prompt_hash,
        )
        name = expected_record_file_name(engine_id, prompt_hash, 0)
        path = self.capture_dir / name
        path.write_bytes(fixture_only_npz_bytes(rec))
        t1 = time.monotonic()
        blob = path.read_bytes()
        rec2, dropped = decode_record_bytes(blob)
        validate_record(
            rec2,
            expected_engine_request_id=engine_id,
            expected_prompt_token_sha256=prompt_hash,
            expected_prompt_len=len(ids),
            active_slot_indices=request.active_slot_indices(),
        )
        t2 = time.monotonic()
        return CapturedFeature(
            record=rec2,
            vector_fp32=rec2.decode_vector_fp32(),
            slot_logits_fp32=np.asarray(rec2.verbalizer_logits_fp32, dtype=np.float32).copy(),
            caller_request_id=caller_request_id,
            engine_request_id=engine_id,
            engine_response_id=engine_id,
            usage={"prompt_tokens": len(ids), "completion_tokens": 1, "total_tokens": len(ids) + 1},
            timing_ms={
                "tokenize": 0.0,
                "request": (t1 - t0) * 1000.0,
                "capture_wait": 0.0,
                "readout": (t2 - t1) * 1000.0,
                "total": (t2 - t0) * 1000.0,
            },
            transport=TRANSPORT_ONE_DISCARDED_TOKEN,
            provider_label=FIXTURE_ONLY_LABEL,
            provider_kind="fixture",
            caller_prompt_token_sha256=prompt_hash,
            cache_raw={"cache_hit_tokens": 0, "num_scheduled_tokens": len(ids)},
            cache_evidence=CACHE_EVIDENCE_UNVERIFIED,
            completion_text_empty=True,
            completion_finish_reason="fixture",
            dropped_meta_fields=dropped,
        )


# ======================================================================================
# readout arms
# ======================================================================================

@dataclasses.dataclass
class ArmReadout:
    result: protocol.DecisionResult
    block: dict


def _check_probabilities(result: protocol.DecisionResult) -> None:
    probs = result.option_probabilities
    if not probs or not all(isinstance(v, float) and np.isfinite(v) for v in probs):
        raise BackendError("readout produced non-finite probabilities")
    total = float(sum(probs))
    if abs(total - 1.0) > 1e-5:
        raise BackendError(f"readout probabilities sum to {total!r}, not 1")
    if any(p < 0.0 for p in probs):
        raise BackendError("readout produced a negative probability")


class B0Arm:
    """Base arm: masked normalisation of the *raw* captured 24-slot logits (the effective rows)."""

    name = "B0"

    @property
    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def read(self, request: protocol.DecisionRequest, captured: CapturedFeature) -> ArmReadout:
        logits = np.asarray(captured.slot_logits_fp32, dtype=np.float64).reshape(-1)
        if logits.shape != (protocol.NUM_SLOTS,):
            raise BackendError(f"captured slot logits must be 24 wide, got {logits.shape}")
        active = request.active_slot_indices()
        active_logits = logits[active]
        if not np.isfinite(active_logits).all() or active_logits.size == 0:
            raise BackendError(
                "the captured raw logits are not finite in the slot(s) this request can read: "
                f"{[int(i) for i in active]}"
            )
        # Inactive slots are masked before normalisation and can never carry mass, so a
        # placeholder (e.g. -inf) there must not abort the readout. Substituting the minimum
        # active logit keeps the shared protocol path for the active slots; the substitution is
        # unobservable because the result zeroes every inactive slot.
        if not np.isfinite(logits).all():
            logits = np.where(np.isfinite(logits), logits, float(np.min(active_logits)))
        try:
            result = protocol.DecisionResult.from_slot_logits(request, [float(v) for v in logits])
        except protocol.ProtocolError as exc:
            raise BackendError(f"B0 readout failed: {exc}") from exc
        _check_probabilities(result)
        return ArmReadout(result, {
            "arm": "B0",
            "lane": "raw_captured_logits",
            "note": "masked softmax over the active slots of the raw engine-captured 24-slot logits",
            "head_artifact": None,
            "trained": False,
            "calibration_status": "not_applicable",
            "calibration_status_overall": "not_applicable",
            "calibration_primitive": {
                "primitive": request.kind,
                "status": "not_applicable",
                "calibrated": False,
                "reason": "B0 is a masked normalisation of the engine's own captured logits, not a fitted head",
            },
            "provenance": None,
        })


class H1Arm:
    """Trained-readout arm: the shared FP32 head on the exactly widened BF16 feature."""

    name = "H1"

    def __init__(
        self,
        *,
        artifact_path: str | None,
        verify_checksum: bool = True,
        template_id: str,
        template_version: str | None = None,
        tokenizer_revision: str | None = None,
        feature_base: str | None = None,
        feature_sha256: str | None = None,
    ) -> None:
        self._head: head_mod.DecisionHead | None = None
        self._reason: str | None = None
        self._artifact: dict | None = None
        self.artifact_path = artifact_path
        if artifact_path is None:
            self._reason = "h1 head artifact is not configured (head.h1.artifact_path is null)"
            return
        if tokenizer_revision is None:
            self._reason = (
                "tokenizer_revision_not_configured: without the serving tokenizer revision the head "
                "artifact cannot be proven to match the live tokenizer"
            )
            return
        try:
            loaded = head_mod.DecisionHead.load(artifact_path, verify_checksum=verify_checksum)
        except Exception as exc:  # noqa: BLE001 - any load problem is a fail-closed condition
            self._reason = f"h1 head artifact load failed: {type(exc).__name__}: {exc}"
            return
        try:
            loaded.validate_against(
                feature_base=feature_base,
                feature_width=head_mod.FEATURE_WIDTH,
                template_id=template_id,
                template_version=template_version,
                tokenizer_revision=tokenizer_revision,
                feature_sha256=feature_sha256,
            )
        except head_mod.HeadError as exc:
            self._reason = f"h1 head provenance mismatch: {exc}"
            return
        self._head = loaded
        try:
            meta = head_mod.DecisionHead.load_metadata(artifact_path)
            self._artifact = {
                "path": str(artifact_path),
                "data_sha256": meta.get("data_sha256"),
                "parameters": meta.get("parameters"),
                "schema": meta.get("schema"),
                "format_version": meta.get("format_version"),
                "provenance_fingerprint": loaded.provenance_fingerprint(),
                "parameter_fingerprint": loaded.parameter_fingerprint(),
            }
        except Exception:  # noqa: BLE001 - descriptive only
            self._artifact = {"path": str(artifact_path)}

    @classmethod
    def from_config_dict(cls, payload: Mapping[str, Any], **kwargs: Any) -> "H1Arm":
        if not isinstance(payload, Mapping):
            raise ConfigError(f"head.h1 must be a mapping, got {type(payload).__name__}")
        unknown = set(payload) - {"artifact_path", "verify_checksum", "note"}
        if unknown:
            raise ConfigError(f"unknown head.h1 option(s): {sorted(unknown)}")
        return cls(artifact_path=payload.get("artifact_path"), verify_checksum=bool(payload.get("verify_checksum", True)), **kwargs)

    @property
    def available(self) -> bool:
        return self._head is not None

    def unavailable_reason(self) -> str | None:
        return self._reason

    def read(self, request: protocol.DecisionRequest, captured: CapturedFeature) -> ArmReadout:
        if self._head is None:
            raise ArmUnavailable(self._reason or "h1 arm unavailable")
        # The head's 24 rows ARE the verbalizer token ids of the captured slots. A record written
        # under different rows than the head was initialized from would silently mix two bases.
        head_rows = tuple(int(r) for r in self._head.provenance.rows)
        record_rows = tuple(int(v) for v in captured.record.verbalizer_ids)
        if head_rows != record_rows:
            raise BackendError(
                "head rows do not match the captured verbalizer rows: the H1 artifact was "
                f"initialized from rows {head_rows[:4]}... but this capture carries rows "
                f"{record_rows[:4]}...; refusing to mix a head with a different capture"
            )
        vector = np.asarray(captured.vector_fp32, dtype=np.float32).reshape(1, -1)
        if vector.shape != (1, head_mod.FEATURE_WIDTH):
            raise BackendError(f"captured vector must be 2560 wide, got {vector.shape}")
        try:
            # the shared head module owns the typed mapping (masked, temperature-scaled softmax ->
            # protocol.DecisionResult); the gateway never re-implements it
            result = head_mod.decision_result(self._head, vector, request)
        except head_mod.HeadError as exc:
            raise BackendError(f"H1 readout failed: {exc}") from exc
        _check_probabilities(result)
        head = self._head
        primitive_calibration = head.primitive_calibration_status(request.kind)
        return ArmReadout(result, {
            "arm": "H1",
            "lane": "fp32_linear_on_bf16_widened_feature",
            "note": (
                "FP32 linear readout on the exactly widened BF16 feature of the same engine, mapped "
                "through head.decision_result; the file-transport readout is a slow path and its "
                "timings include I/O"
            ),
            "head_artifact": self._artifact,
            "trained": bool(head.trained),
            "training_record": head.training,
            # per-primitive honesty: the overall status may be "calibrated" because *another*
            # primitive was fitted, so it is never reported as this kind's claim
            "calibration_status": primitive_calibration["status"],
            "calibration_status_overall": head.calibration_status(),
            "calibration_primitive": primitive_calibration,
            "calibration": head.calibration.describe(),
            "calibration_summary": head.metadata_calibration_summary(),
            "provenance": head.provenance.as_dict(),
            "provenance_fingerprint": head.provenance_fingerprint(),
            "parameter_fingerprint": head.parameter_fingerprint(),
            "head_scale": captured.record.head_scale,
            "head_soft_cap": captured.record.head_soft_cap,
        })


# ======================================================================================
# config
# ======================================================================================

_TOP_KEYS = {
    "schema", "service_version", "stage", "host", "port", "provider", "engine", "template",
    "tokenizer", "head", "limits", "concurrency", "timeouts", "capture", "logging", "parity", "fixture",
    "decision_adapter",
    # additive: the opt-in wide-label lane (absent => the legacy gateway is unchanged)
    "wide_choice",
}
_ENGINE_KEYS = {
    "base_url", "model", "socket_timeout_s", "request_timeout_s", "max_proxy_body_bytes",
    "health_ping", "forward_headers", "forward_headers_from_env",
}
_TEMPLATE_KEYS = {"template_id", "version", "require_version_match", "chat_template_kwargs"}
_TOKENIZER_KEYS = {"revision", "add_special_tokens"}
_HEAD_KEYS = {"default_arm", "h1"}
_LIMIT_KEYS = {
    "max_body_bytes", "max_options", "max_state_chars", "max_question_chars",
    "max_option_description_chars", "max_key_chars", "max_total_chars",
}
_CONCURRENCY_KEYS = {"max_inflight"}
_TIMEOUT_KEYS = {"request_s", "socket_s", "capture_deadline_s", "poll_initial_ms", "poll_max_ms", "poll_backoff"}
_CAPTURE_KEYS = {
    "dir", "dir_allowlist", "record_max_bytes", "transport", "max_completion_tokens",
    "tolerate_unknown_meta_fields",
}
_LOGGING_KEYS = {"redact_prompts", "log_file"}
_FIXTURE_KEYS = {"seed", "vector_source", "record"}
_PRIMITIVE_LIMITS = ("max_state_chars", "max_question_chars", "max_option_description_chars", "max_key_chars", "max_total_chars")


def _check_keys(payload: Mapping[str, Any], allowed: set[str], where: str) -> None:
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{where} must be a JSON object, got {type(payload).__name__}")
    unknown = sorted(set(payload) - allowed - {"note"})
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {unknown}")


def _require(mapping: Mapping[str, Any], key: str, where: str, default: Any = dataclasses._MISSING_TYPE) -> Any:
    if key in mapping:
        return mapping[key]
    if default is not dataclasses._MISSING_TYPE:
        return default
    raise ConfigError(f"{where} is missing the required key {key!r}")


def _as_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a bool, got {value!r}")
    return value


def _as_int(value: Any, where: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{where} must be an int, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{where} must be <= {maximum}, got {value!r}")
    return value


def _as_number(value: Any, where: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ConfigError(f"{where} must be a finite number, got {value!r}")
    if minimum is not None and float(value) < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, got {value!r}")
    return float(value)


def _as_str(value: Any, where: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} must be a non-empty string, got {value!r}")
    return value


class ApiConfig:
    """Validated gateway configuration. Unknown keys anywhere are rejected, not ignored."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        _check_keys(raw, _TOP_KEYS, "config")
        self.raw = json.loads(json.dumps(raw))  # detached, JSON-only copy
        schema = _require(raw, "schema", "config")
        if schema != CONFIG_SCHEMA:
            raise ConfigError(f"unsupported config schema {schema!r}; this build reads {CONFIG_SCHEMA!r}")
        self.schema = schema
        self.service_version = _as_str(raw.get("service_version", "s6-gateway/0.0.0"), "service_version")
        self.stage = str(raw.get("stage", ""))
        self.host = _as_str(raw.get("host", "127.0.0.1"), "host")
        self.port = _as_int(raw.get("port", 8890), "port", 1, 65535)
        self.provider = _as_str(raw.get("provider", "engine"), "provider")
        if self.provider not in ("engine", "fixture"):
            raise ConfigError(f"provider must be 'engine' or 'fixture', got {self.provider!r}")

        engine = _require(raw, "engine", "config")
        _check_keys(engine, _ENGINE_KEYS, "engine")
        self.engine_base_url = _as_str(_require(engine, "base_url", "engine"), "engine.base_url")
        if not self.engine_base_url.startswith(("http://", "https://")):
            raise ConfigError(f"engine.base_url must be http(s), got {self.engine_base_url!r}")
        self.engine_model = _as_str(_require(engine, "model", "engine"), "engine.model")
        self.socket_timeout_s = _as_number(engine.get("socket_timeout_s", 30.0), "engine.socket_timeout_s", 0.001)
        self.request_timeout_s = _as_number(engine.get("request_timeout_s", 120.0), "engine.request_timeout_s", 0.001)
        self.max_proxy_body_bytes = _as_int(engine.get("max_proxy_body_bytes", 8 * 1024 * 1024), "engine.max_proxy_body_bytes", 1)
        self.health_ping = _as_bool(bool(engine.get("health_ping", False)), "engine.health_ping")
        fwd = engine.get("forward_headers", ["content-type", "accept"])
        if not isinstance(fwd, list) or not all(isinstance(v, str) for v in fwd):
            raise ConfigError("engine.forward_headers must be a list of header names")
        self.forward_headers = [v.lower() for v in fwd]
        envmap = engine.get("forward_headers_from_env", {})
        if not isinstance(envmap, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in envmap.items()):
            raise ConfigError("engine.forward_headers_from_env must map header -> environment variable")
        bad = sorted(set(k.lower() for k in envmap) - {"authorization", "x-api-key"})
        if bad:
            raise ConfigError(f"forward_headers_from_env may only carry credential headers, got {bad}")
        self.forward_headers_from_env = {k.lower(): v for k, v in envmap.items()}

        template = _require(raw, "template", "config")
        _check_keys(template, _TEMPLATE_KEYS, "template")
        self.template_id = _as_str(_require(template, "template_id", "template"), "template.template_id")
        self.template_version = _as_str(_require(template, "version", "template"), "template.version")
        self.require_version_match = _as_bool(bool(template.get("require_version_match", True)), "template.require_version_match")
        if not self.require_version_match:
            raise ConfigError(
                "template.require_version_match cannot be disabled in this stage: the frozen template "
                "version is mandatory so a run cannot silently reinterpret the prompt surface"
            )
        kwargs = template.get("chat_template_kwargs", {})
        if not isinstance(kwargs, dict):
            raise ConfigError("template.chat_template_kwargs must be an object")
        self.chat_template_kwargs = dict(kwargs)
        if self.template_id not in templates.TEMPLATES:
            raise ConfigError(f"unknown template id {self.template_id!r}")
        frozen = templates.TEMPLATES[self.template_id].version
        if self.require_version_match and frozen != self.template_version:
            raise ConfigError(
                f"frozen template {self.template_id!r} is version {frozen!r} but the config declares "
                f"{self.template_version!r}; refusing to reinterpret the prompt surface"
            )

        tokenizer = _require(raw, "tokenizer", "config")
        _check_keys(tokenizer, _TOKENIZER_KEYS, "tokenizer")
        self.tokenizer_revision = _as_str(tokenizer.get("revision"), "tokenizer.revision", allow_none=True)
        self.add_special_tokens = _as_bool(bool(tokenizer.get("add_special_tokens", False)), "tokenizer.add_special_tokens")

        # Optional, disabled-by-default wide-label lane (17..256 options). Absent => nothing changes.
        self.wide_choice: wide.WideChoiceConfig | None = None
        if "wide_choice" in raw:
            try:
                self.wide_choice = wide.WideChoiceConfig.from_dict(
                    raw["wide_choice"], gateway_tokenizer_revision=self.tokenizer_revision
                )
            except wide.WideError as exc:
                raise ConfigError(f"wide_choice: {exc}") from exc

        # Optional, disabled-by-default experimental typed arm (L1O). Absent => nothing changes.
        raw_adapter = raw.get("decision_adapter")
        self.decision_adapter: decision_adapter.AdapterSpec | None = None
        if raw_adapter is not None:
            try:
                self.decision_adapter = decision_adapter.AdapterSpec.from_config(
                    raw_adapter,
                    engine_base_url=self.engine_base_url,
                    engine_model=self.engine_model,
                    tokenizer_revision=self.tokenizer_revision,
                    repo_root=REPO_ROOT,
                )
            except decision_adapter.L1OConfigError as exc:
                raise ConfigError(str(exc)) from exc

        head = _require(raw, "head", "config")
        _check_keys(head, _HEAD_KEYS, "head")
        self.default_arm = _as_str(head.get("default_arm", "B0"), "head.default_arm")
        if self.default_arm == decision_adapter.ARM_NAME:
            if self.decision_adapter is None or not self.decision_adapter.enabled:
                raise ConfigError(
                    "head.default_arm 'L1O' requires decision_adapter.enabled=true: the experimental "
                    "adapter arm is disabled by default and is never a silent default"
                )
        elif self.default_arm not in ("B0", "H1"):
            raise ConfigError(f"head.default_arm must be B0, H1 or L1O, got {self.default_arm!r}")
        self.h1 = dict(_require(head, "h1", "head"))
        _check_keys(self.h1, {"artifact_path", "verify_checksum"}, "head.h1")
        artifact = self.h1.get("artifact_path")
        if artifact is not None and (not isinstance(artifact, str) or not artifact):
            raise ConfigError(f"head.h1.artifact_path must be null or a non-empty string, got {artifact!r}")
        if "verify_checksum" in self.h1:
            _as_bool(self.h1["verify_checksum"], "head.h1.verify_checksum")

        limits = _require(raw, "limits", "config")
        _check_keys(limits, _LIMIT_KEYS, "limits")
        self.max_body_bytes = _as_int(limits.get("max_body_bytes", 262144), "limits.max_body_bytes", 1)
        self.max_options = _as_int(limits.get("max_options", protocol.MAX_CHOICE_OPTIONS), "limits.max_options",
                                   protocol.MIN_CHOICE_OPTIONS, protocol.MAX_CHOICE_OPTIONS)
        protocol_defaults = protocol.ProtocolLimits()
        limit_kwargs = {}
        for name in _PRIMITIVE_LIMITS:
            limit_kwargs[name] = _as_int(limits.get(name, getattr(protocol_defaults, name)), f"limits.{name}", 1)
        self.limits = protocol.ProtocolLimits(**limit_kwargs)
        # the operator may tighten the per-field caps, but the model-facing bound that matters for
        # a single typed question is the total char budget and each field's own cap
        if self.limits.max_total_chars > protocol_defaults.max_total_chars:
            raise ConfigError("limits.max_total_chars may not exceed the protocol's declared bound")
        self.limit_values = dict(limit_kwargs)

        conc = _require(raw, "concurrency", "config")
        _check_keys(conc, _CONCURRENCY_KEYS, "concurrency")
        self.concurrency_max_inflight = _as_int(conc.get("max_inflight", 8), "concurrency.max_inflight", 1, 4096)

        timeouts = _require(raw, "timeouts", "config")
        _check_keys(timeouts, _TIMEOUT_KEYS, "timeouts")
        self.capture_deadline_s = _as_number(timeouts.get("capture_deadline_s", 30.0), "timeouts.capture_deadline_s", 0.001)
        self.timeout_request_s = _as_number(timeouts.get("request_s", 120.0), "timeouts.request_s", 0.001)
        self.timeout_socket_s = _as_number(timeouts.get("socket_s", self.socket_timeout_s), "timeouts.socket_s", 0.001)
        self.poll_initial_ms = _as_number(timeouts.get("poll_initial_ms", 5.0), "timeouts.poll_initial_ms", 0.001)
        self.poll_max_ms = _as_number(timeouts.get("poll_max_ms", 100.0), "timeouts.poll_max_ms", 0.001)
        self.poll_backoff = _as_number(timeouts.get("poll_backoff", 1.5), "timeouts.poll_backoff", 1.0)
        if self.poll_max_ms < self.poll_initial_ms:
            raise ConfigError("timeouts.poll_max_ms must be >= timeouts.poll_initial_ms")

        capture = _require(raw, "capture", "config")
        _check_keys(capture, _CAPTURE_KEYS, "capture")
        self.capture_dir = _as_str(capture.get("dir"), "capture.dir", allow_none=True)
        allow = capture.get("dir_allowlist", [])
        if not isinstance(allow, list) or not all(isinstance(v, str) for v in allow):
            raise ConfigError("capture.dir_allowlist must be a list of absolute paths")
        self.capture_dir_allowlist = list(allow)
        if self.capture_dir is not None and not pathlib.Path(self.capture_dir).is_absolute():
            raise ConfigError(f"capture.dir must be an absolute path, got {self.capture_dir!r}")
        if self.capture_dir is not None and self.capture_dir_allowlist:
            real = os.path.realpath(self.capture_dir)
            allowed = [os.path.realpath(a) for a in self.capture_dir_allowlist]
            if not any(real == a or real.startswith(a + os.sep) for a in allowed):
                raise ConfigError(
                    f"capture.dir {self.capture_dir!r} is not inside capture.dir_allowlist {self.capture_dir_allowlist!r}"
                )
        self.record_max_bytes = _as_int(capture.get("record_max_bytes", 8 * 1024 * 1024), "capture.record_max_bytes", 1)
        self.capture_transport = _as_str(capture.get("transport", TRANSPORT_ONE_DISCARDED_TOKEN), "capture.transport")
        if self.capture_transport != TRANSPORT_ONE_DISCARDED_TOKEN:
            raise ConfigError(
                f"capture.transport {self.capture_transport!r} is not implemented; this build implements "
                f"{TRANSPORT_ONE_DISCARDED_TOKEN!r} only"
            )
        self.max_completion_tokens = _as_int(capture.get("max_completion_tokens", 1), "capture.max_completion_tokens", 1, 1)
        self.tolerate_unknown_meta_fields = _as_bool(
            bool(capture.get("tolerate_unknown_meta_fields", False)), "capture.tolerate_unknown_meta_fields"
        )

        logging_cfg = _require(raw, "logging", "config")
        _check_keys(logging_cfg, _LOGGING_KEYS, "logging")
        self.redact_prompts = _as_bool(bool(logging_cfg.get("redact_prompts", True)), "logging.redact_prompts")
        if not self.redact_prompts:
            raise ConfigError("logging.redact_prompts cannot be disabled in this stage")
        self.log_file = _as_str(logging_cfg.get("log_file"), "logging.log_file", allow_none=True)

        parity = raw.get("parity", {})
        if not isinstance(parity, dict):
            raise ConfigError("parity must be an object")
        self.parity = dict(parity)

        fixture = raw.get("fixture", {})
        _check_keys(fixture, _FIXTURE_KEYS, "fixture")
        self.fixture = {**dict(fixture), "seed": _as_int(fixture.get("seed", 12345), "fixture.seed", 0)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ApiConfig":
        if not isinstance(payload, Mapping):
            raise ConfigError(f"config must be a JSON object, got {type(payload).__name__}")
        return cls(payload)

    def as_dict(self) -> dict:
        return json.loads(json.dumps(self.raw))

    # -- derived -----------------------------------------------------------------------
    def capture_provider_config(self) -> CaptureProviderConfig:
        if self.capture_dir is None:
            raise ConfigError(
                "capture.dir is null: the operator must configure the engine capture directory before "
                "the typed endpoint can read features"
            )
        return capture_provider_config(
            base_url=self.engine_base_url,
            model=self.engine_model,
            capture_dir=self.capture_dir,
            capture_dir_allowlist=self.capture_dir_allowlist,
            tokenizer_revision=self.tokenizer_revision,
            template_id=self.template_id,
            template_version=self.template_version,
            chat_template_kwargs=self.chat_template_kwargs,
            add_special_tokens=self.add_special_tokens,
            socket_timeout_s=self.socket_timeout_s,
            request_timeout_s=self.request_timeout_s,
            capture_deadline_s=self.capture_deadline_s,
            poll_initial_ms=self.poll_initial_ms,
            poll_max_ms=self.poll_max_ms,
            poll_backoff=self.poll_backoff,
            record_max_bytes=self.record_max_bytes,
            max_completion_tokens=self.max_completion_tokens,
            tolerate_unknown_meta_fields=self.tolerate_unknown_meta_fields,
            wide_choice=self.wide_choice,
        )

    def head_arm(self) -> H1Arm:
        return H1Arm.from_config_dict(
            self.h1,
            template_id=self.template_id,
            template_version=self.template_version,
            tokenizer_revision=self.tokenizer_revision,
        )


def default_api_config_dict() -> dict:
    """Load the bundled defaults, independently of a checkout or caller's cwd."""
    from importlib.resources import files
    return json.loads(files(__package__).joinpath("default_api.json").read_text(encoding="utf-8"))


def load_api_config(path: str | os.PathLike[str] | None = None) -> ApiConfig:
    target = pathlib.Path(path) if path is not None else pathlib.Path(default_api_config_path())
    if not target.is_file():
        raise ConfigError(f"config file not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file is not valid JSON: {exc}") from exc
    cfg = ApiConfig.from_dict(payload)
    cfg.config_path = str(target)  # type: ignore[attr-defined]
    return cfg


def default_api_config_path() -> str:
    for candidate in (
        pathlib.Path(os.getcwd()) / CONFIG_PATH,
        pathlib.Path(__file__).resolve().parents[2] / CONFIG_PATH,
    ):
        if candidate.is_file():
            return str(candidate)
    return CONFIG_PATH


# ======================================================================================
# decision backend
# ======================================================================================

@dataclasses.dataclass
class DecisionOutcome:
    result: protocol.DecisionResult
    arm: str
    arm_block: dict
    captured: CapturedFeature
    template_block: dict
    engine_block: dict
    tokenizer_block: dict
    timing: dict
    usage: dict
    cache: dict
    transport: str
    uses_s2: bool
    provider_block: dict
    parity: dict
    capture_block: dict = dataclasses.field(default_factory=dict)
    seed: int | None = None
    # Status of the experimental decision adapter for THIS response (None when the config
    # declares no decision_adapter block at all, so default responses are unchanged).
    adapter_block: dict | None = None


def _jsonable(value: Any, where: str) -> Any:
    """Recursively convert to JSON types, refusing NaN/Infinity anywhere in a response."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, where) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not np.isfinite(f):
            raise BackendError(f"refusing to serialise a non-finite value at {where}: {value!r}")
        return f
    raise BackendError(f"refusing to serialise {type(value).__name__} at {where}")


def outcome_payload(outcome: DecisionOutcome) -> dict:
    """The JSON body of a successful ``/v1/decide`` response (no prompt or generated text)."""
    payload = dict(outcome.result.as_payload())
    payload["schema"] = "qfn.decide/1"
    payload["slot_probabilities"] = [float(v) for v in outcome.result.slot_probabilities]
    payload["typed_output_tokens"] = TYPED_OUTPUT_TOKENS
    payload["internal_generated_tokens"] = int(outcome.usage["internal_generated_tokens"])
    payload["uses_s2"] = bool(outcome.uses_s2)
    payload["transport"] = outcome.transport
    payload["transport_note"] = (
        "the feature transport may schedule one discarded internal completion token; zero returned "
        "text is NOT a native zero-sampler engine implementation"
    )
    payload["arm"] = outcome.arm_block
    payload["template"] = outcome.template_block
    payload["tokenizer"] = outcome.tokenizer_block
    payload["engine"] = outcome.engine_block
    payload["timing"] = outcome.timing
    payload["usage"] = outcome.usage
    payload["cache"] = outcome.cache
    payload["capture"] = outcome.capture_block or {
        "engine_id_forms": "cmpl-<caller> | cmpl-<caller>-<index> | cmpl-<caller>-<index>-<8hex>",
        "error_marker_seen": False,
        "error_marker": None,
    }
    payload["provider"] = outcome.provider_block
    payload["parity"] = outcome.parity
    payload["seed"] = outcome.seed
    if outcome.adapter_block is not None:
        payload["decision_adapter"] = outcome.adapter_block
    return _jsonable(payload, "decide")


class DecisionBackend:
    """Provider + arms + config: turns a validated request into a serialisable outcome."""

    def __init__(
        self,
        config: ApiConfig,
        *,
        provider: FeatureProvider | None = None,
        arms: Mapping[str, Any] | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.config = config
        self.clock = clock
        if arms is None:
            self.arms: dict[str, Any] = {"B0": B0Arm(), "H1": config.head_arm()}
        else:
            self.arms = dict(arms)
        if "B0" not in self.arms or "H1" not in self.arms:
            raise ConfigError("the arm registry must provide B0 and H1")
        self.provider = provider if provider is not None else self._build_provider()

        # The experimental L1O arm is additive: with no decision_adapter block (or a disabled one)
        # nothing below changes the arm registry, the providers or the responses.
        self.adapter_spec: decision_adapter.AdapterSpec | None = getattr(config, "decision_adapter", None)
        self.adapter_provider: FeatureProvider | None = None
        # hashed once per process, not on every /health call
        self._adapter_checks: dict | None = (
            self.adapter_spec.cpu_checks() if self.adapter_spec is not None else None
        )
        if self.adapter_spec is not None and self.adapter_spec.enabled:
            if provider is not None:
                # An injected provider serves every arm: that is a test seam only (the real path
                # below builds the adapter a provider of its own, so B0/H1 keep the base alias).
                self.adapter_provider = provider
            else:
                self.adapter_provider = self._build_adapter_provider(self.adapter_spec)
            self.arms.setdefault(
                decision_adapter.ARM_NAME,
                decision_adapter.L1OArm(self.adapter_spec, provider=self.adapter_provider),
            )
        # Request-scoped provider selection: each arm names its own provider, and nothing here ever
        # rewrites config.model / config.provider while a request is in flight.
        self.arm_providers: dict[str, FeatureProvider] = {
            name: (self.adapter_provider if name == decision_adapter.ARM_NAME else self.provider)
            for name in self.arms
        }

    # -- provider ----------------------------------------------------------------------
    def _build_provider(self) -> FeatureProvider:
        if self.config.provider == "fixture":
            return FixtureFeatureProvider(
                seed=int(self.config.fixture.get("seed", 12345)),
                capture_dir=self.config.capture_dir or None,
            )
        return HttpEngineCaptureProvider(self.config.capture_provider_config())

    def _build_adapter_provider(self, spec: decision_adapter.AdapterSpec) -> FeatureProvider:
        """The L1O arm's OWN provider: a NEW config derived from the gateway's declared one.

        Everything the operator declared about the transport budgets and the capture-reader policy
        is inherited from ``config.capture_provider_config()`` (a fresh object, never the base
        provider's instance): socket/request/capture deadlines, poll spacing and backoff, record size
        cap, unknown-meta tolerance and the transport token count. ONLY the model alias, base url,
        capture directory/allowlist, tokenizer/template surface and expected served model are
        overridden per the adapter spec. ``config.model`` / ``config.provider`` are never modified.
        """
        if self.config.provider == "fixture":
            return FixtureFeatureProvider(
                seed=int(self.config.fixture.get("seed", 12345)),
                capture_dir=spec.capture_dir or None,
                template_id=spec.template_id,
            )
        declared = self.config.capture_provider_config()
        override = spec.capture_provider_kwargs()
        adapter_config = dataclasses.replace(
            declared,
            base_url=override["base_url"],
            model=override["model"],
            capture_dir=override["capture_dir"],
            capture_dir_allowlist=tuple(override["capture_dir_allowlist"]),
            tokenizer_revision=override["tokenizer_revision"],
            template_id=override["template_id"],
            template_version=override["template_version"],
            chat_template_kwargs=dict(override["chat_template_kwargs"]),
            add_special_tokens=override["add_special_tokens"],
            expected_served_model=override["expected_served_model"],
        )
        return HttpEngineCaptureProvider(adapter_config)

    def provider_for(self, arm: str) -> FeatureProvider:
        """The provider this arm reads through (request-scoped: never a shared mutable alias)."""
        return self.arm_providers.get(arm, self.provider)

    # -- views -------------------------------------------------------------------------
    def available_arms(self) -> dict[str, bool]:
        return {name: bool(arm.available) for name, arm in self.arms.items()}

    def unavailable_reason(self, arm: str) -> str | None:
        return self.arms[arm].unavailable_reason() if arm in self.arms else f"unknown arm {arm!r}"

    def _health_arm_names(self) -> tuple[str, ...]:
        """The arms /health lists: B0/H1 always, L1O only when the adapter is enabled."""
        names = ["B0", "H1"]
        if self.adapter_spec is not None and self.adapter_spec.enabled:
            names.append(decision_adapter.ARM_NAME)
        return tuple(names)

    def adapter_status_block(self, *, used: bool) -> dict | None:
        """The experimental-adapter status disclosed on a typed response (None when unconfigured)."""
        spec = self.adapter_spec
        if spec is None:
            return None
        arm = self.arms.get(decision_adapter.ARM_NAME)
        available = bool(arm.available) if arm is not None else False
        reason = arm.unavailable_reason() if arm is not None else spec.declared_status
        if not spec.enabled:
            reason = (
                "the experimental L1O adapter is present but disabled "
                "(decision_adapter.enabled=false)"
            )
        return spec.request_block(available=available, reason=reason, used=used)

    def adapter_health_block(self) -> dict | None:
        spec = self.adapter_spec
        if spec is None:
            return None
        arm = self.arms.get(decision_adapter.ARM_NAME)
        available = bool(arm.available) if arm is not None else False
        if not spec.enabled:
            reason = (
                "the experimental L1O adapter is present but disabled "
                "(decision_adapter.enabled=false)"
            )
        elif arm is not None:
            reason = arm.unavailable_reason()
        else:
            reason = "the L1O arm was not registered"
        checks = arm.checks() if arm is not None else (self._adapter_checks or spec.cpu_checks())
        return spec.health_block(available=available, reason=reason, checks=checks)

    def health(self) -> dict:
        ping = "not_checked"
        detail = None
        if self.config.health_ping:
            try:
                conn = http.client.HTTPConnection(
                    urllib.parse.urlsplit(self.config.engine_base_url).hostname or "127.0.0.1",
                    urllib.parse.urlsplit(self.config.engine_base_url).port or 80,
                    timeout=min(self.config.socket_timeout_s, 5.0),
                )
                conn.request("GET", "/health")
                resp = conn.getresponse()
                resp.read()
                ping = "ok" if resp.status == 200 else f"error: HTTP {resp.status}"
                conn.close()
            except Exception as exc:  # noqa: BLE001 - a ping failure is reported, not raised
                ping = f"error: {type(exc).__name__}"
                detail = str(exc)
        arms = []
        for name in self._health_arm_names():
            arm = self.arms[name]
            entry = {
                "arm": name,
                "available": bool(arm.available),
                "lane": {"B0": "raw_captured_logits",
                         "H1": "fp32_linear_on_bf16_widened_feature"}.get(
                             name, decision_adapter.L1OArm.lane),
            }
            if name == decision_adapter.ARM_NAME:
                entry["experimental"] = True
                entry["runtime_verified"] = False
                entry["calibration_status"] = "not_applicable"
                entry["note"] = (
                    "opt-in experimental typed arm: the served LoRA plus the masked raw effective "
                    "verbalizer rows of the frozen lettered surface; NOT a fitted FP32 head"
                )
            if not arm.available:
                entry["reason"] = arm.unavailable_reason()
                entry["calibration_status"] = "unavailable"
            elif name == "B0":
                entry["calibration_status"] = "not_applicable"
                entry["head_artifact"] = None
            elif name == "H1":
                head = arm._head
                entry["calibration_overall"] = head.calibration_status()
                entry["calibration_per_primitive"] = {
                    kind: head.primitive_calibration_status(kind) for kind in head_mod.PRIMITIVE_KEYS
                }
                entry["calibrated_primitives"] = head.calibrated_primitives()
                entry["trained"] = bool(head.trained)
                entry["head_artifact"] = arm._artifact
            arms.append(entry)
        h1_available = bool(self.arms["H1"].available)
        calibration_overall = self.arms["H1"]._head.calibration_status() if h1_available else "unavailable"
        calibration_per_primitive = (
            {kind: self.arms["H1"]._head.primitive_calibration_status(kind) for kind in head_mod.PRIMITIVE_KEYS}
            if h1_available
            else {}
        )
        diagnostics_fn = getattr(self.provider, "diagnostics", None)
        provider_diagnostics = (
            diagnostics_fn() if callable(diagnostics_fn)
            else {"error_marker_seen": False, "error_marker": None}
        )
        ready = bool(self.arms["B0"].available)
        doc = {
            "schema": "qfn.decide.health/1",
            "status": "ready" if ready else "degraded",
            "service_version": self.config.service_version,
            "stage": self.config.stage,
            "bind": {
                "host": self.config.host,
                "port": self.config.port,
                "lan_exposure_warning": (
                    "binding to a non-loopback interface exposes this gateway UNAUTHENTICATED, exactly "
                    "like the engine's own OpenAI API. It is not published to the internet."
                ),
                "loopback_only": self.config.host in ("127.0.0.1", "localhost", "::1"),
            },
            "provider": {
                "label": getattr(self.provider, "label", "unknown"),
                "fixture": getattr(self.provider, "label", "") == FIXTURE_ONLY_LABEL,
                "kind": getattr(self.provider, "kind", "engine"),
            },
            "engine": {
                "base_url": self.config.engine_base_url,
                "model": self.config.engine_model,
                "ping": ping,
                "ping_detail": detail,
            },
            "protocol": {
                "schema": "qfn.protocol/1",
                "kinds": list(protocol.KINDS),
                "num_slots": protocol.NUM_SLOTS,
                "choice_min_options": protocol.MIN_CHOICE_OPTIONS,
                "choice_max_options": self.config.max_options,
                "score_levels": protocol.SCORE_LEVEL_COUNT,
                "typed_output_tokens": TYPED_OUTPUT_TOKENS,
                "transport": self.config.capture_transport,
                "jev_confidence": None,
            },
            "arms": arms,
            "calibration_overall": calibration_overall,
            "calibration_per_primitive": calibration_per_primitive,
            "calibration_note": (
                "calibration_overall may be 'calibrated' while a given primitive is not; the "
                "per-primitive block is the authoritative claim for a request's kind"
            ),
            "template": {
                "template_id": self.config.template_id,
                "version": self.config.template_version,
                "frozen": True,
            },
            "tokenizer": {
                "revision": self.config.tokenizer_revision,
                "add_special_tokens": self.config.add_special_tokens,
            },
            "capture": {
                "dir_configured": self.config.capture_dir is not None,
                "dir_allowlisted": bool(self.config.capture_dir_allowlist),
                "record_max_bytes": self.config.record_max_bytes,
                "transport": self.config.capture_transport,
                "tolerate_unknown_meta_fields": self.config.tolerate_unknown_meta_fields,
                "engine_id_forms": "cmpl-<caller> | cmpl-<caller>-<index> | cmpl-<caller>-<index>-<8hex>",
                "error_marker_seen": bool(provider_diagnostics.get("error_marker_seen")),
                "error_marker": provider_diagnostics.get("error_marker"),
                "error_marker_rule": provider_diagnostics.get("error_marker_rule", (
                    "a marker fails a request only when it identifies that request or reports a "
                    "global stop (stopped=true / quota_exceeded); otherwise it is a diagnostic"
                )),
                "refusal_marker_seen": bool(provider_diagnostics.get("refusal_marker_seen")),
                "refusal_marker_file": CAPTURE_REFUSED_MARKER_NAME,
                "refusal_marker_entry_count": provider_diagnostics.get("refusal_marker_entry_count"),
                "refusal_marker_rule": provider_diagnostics.get("refusal_marker_rule", (
                    "a refusal entry fails ONLY the request it names, with HTTP 503 "
                    "capture_refused; other requests and the global-stop behaviour are unchanged"
                )),
            },
            "max_inflight": self.config.concurrency_max_inflight,
            "parity": self.config.parity,
            "readiness": {
                "provider_configured": True,
                "engine_ping": ping,
                "live_engine_validated": False,
                "note": READINESS_NOTE,
            },
        }
        adapter_block = self.adapter_health_block()
        if adapter_block is not None:
            doc["decision_adapter"] = adapter_block
            doc["readiness"]["adapter_runtime_verified"] = False
            doc["readiness"]["adapter_note"] = (
                "the experimental L1O adapter status is disclosed above; its loaded-checkpoint "
                "identity is NOT verified by this process and stays unverified until the PM reads "
                "the loaded adapter back off the serving engine"
            )
        if self.config.wide_choice is not None:
            doc["wide_choice"] = self.config.wide_choice.health_block()
            doc["readiness"]["wide_labels_runtime_verified"] = False
            doc["readiness"]["wide_labels_note"] = (
                "the wide-label lane is live-validated on the production deployment for serving behavior "
                "(typed calls, per-record audits, refusals); response probabilities are normalized over "
                "the active option logits and are not guaranteed calibrated confidence"
            )
        return doc

    # -- the typed readout -------------------------------------------------------------
    def decide(
        self,
        request: protocol.DecisionRequest,
        *,
        arm: str | None = None,
        seed: int | None = None,
        deadline_s: float | None = None,
    ) -> DecisionOutcome:
        if not isinstance(request, protocol.DecisionRequest):
            raise BackendError(f"decide() expects a DecisionRequest, got {type(request).__name__}")
        arm_name = arm or self.config.default_arm
        if arm_name not in self.arms:
            raise BackendError(f"unsupported arm {arm_name!r}; available arms: {sorted(self.arms)}")
        chosen = self.arms[arm_name]
        if not chosen.available:
            raise ArmUnavailable(f"arm {arm_name} is unavailable: {chosen.unavailable_reason()}")
        deadline = float(deadline_s) if deadline_s is not None else float(self.config.capture_deadline_s)

        # Request-scoped routing: an L1O request is captured through the adapter's OWN provider and
        # on the surface the adapter actually saw (a converted lettered choice for a noul question).
        # Nothing here rewrites config.model / config.provider, so B0/H1 keep the base alias.
        wire_request = request
        adapter_surface: dict | None = None
        provider = self.provider_for(arm_name)
        if isinstance(chosen, decision_adapter.L1OArm):
            try:
                wire_request = chosen.wire_request(request)
                captured = provider.acquire(wire_request, deadline_s=deadline, seed=seed)
                t0 = self.clock()
                adapter_readout = chosen.read(request, wire_request, captured)
                readout = ArmReadout(adapter_readout.result, adapter_readout.block)
                adapter_surface = chosen.surface_blocks(request, wire_request, captured)
                t1 = self.clock()
            except decision_adapter.L1OArmUnavailable as exc:
                raise ArmUnavailable(f"arm {arm_name} is unavailable: {exc}") from exc
            except decision_adapter.L1OReadoutError as exc:
                raise BackendError(f"L1O readout failed: {exc}") from exc
        else:
            captured = provider.acquire(request, deadline_s=deadline, seed=seed)
            t0 = self.clock()
            readout = chosen.read(request, captured)
            t1 = self.clock()

        timing = {
            "tokenize_ms": float(captured.timing_ms.get("tokenize", 0.0)),
            "request_ms": float(captured.timing_ms.get("request", 0.0)),
            "capture_wait_ms": float(captured.timing_ms.get("capture_wait", 0.0)),
            "readout_ms": (t1 - t0) * 1000.0,
            "total_ms": float(captured.timing_ms.get("total", 0.0)) + (t1 - t0) * 1000.0,
            "note": (
                "capture_wait is polling the engine's capture directory, so a local-filesystem "
                "readout path makes the CPU FP32 head readout a slow path and includes I/O time"
            ),
        }
        usage = {
            "prompt_tokens": int(captured.usage.get("prompt_tokens", 0)),
            "completion_tokens": int(captured.usage.get("completion_tokens", 0)),
            "total_tokens": int(captured.usage.get("total_tokens", 0)),
            "internal_generated_tokens": int(captured.usage.get("completion_tokens", 0)),
            "typed_output_tokens": TYPED_OUTPUT_TOKENS,
            "completion_text_returned": False,
            "completion_text_empty": bool(captured.completion_text_empty),
            "completion_finish_reason": captured.completion_finish_reason,
            "note": (
                "one internal completion token may be scheduled to make the engine run the capture "
                "path; the count is the engine's actual usage. An empty visible text with a valid "
                "capture is a success (EOS may have been sampled), not a capture failure."
            ),
        }
        engine_block = {
            "base_url": self.config.engine_base_url,
            "model": self.config.engine_model,
            "engine_request_id": captured.engine_request_id,
            "engine_response_id": captured.engine_response_id,
            "caller_request_id": captured.caller_request_id,
            "engine_build_version": captured.record.engine_build_version,
            "helper_version": captured.record.helper_version,
            "runner_source_sha256": captured.record.runner_source_sha256,
            "logits_captured": bool(captured.record.logits_captured),
            "logits_absent_reason": captured.record.logits_absent_reason,
            "head_scale": captured.record.head_scale,
            "head_soft_cap": captured.record.head_soft_cap,
            "head_dtype": captured.record.head_dtype,
            "head_org_vocab_size": captured.record.head_org_vocab_size,
            "lm_head_tp_size": captured.record.lm_head_tp_size,
            "tp_rank": captured.record.tp_rank,
            "world_size": captured.record.world_size,
            "prompt_len": captured.record.prompt_len,
            "terminal_position": captured.record.terminal_position,
            "kernel_mode": captured.record.kernel_mode,
            "cudagraph_mode": captured.record.cudagraph_mode,
            "vector_source": captured.record.vector_source,
            "token_hash_from_engine_state": bool(captured.record.token_hash_from_engine_state),
        }
        tokenizer_block = {
            "revision": self.config.tokenizer_revision,
            "add_special_tokens": bool(self.config.add_special_tokens),
            "prompt_tokens_count": int(captured.record.prompt_tokens_count),
            "prompt_token_sha256": captured.record.prompt_token_sha256,
            "token_hash_schema": features.TOKEN_HASH_SCHEMA,
            "dropped_meta_fields": list(captured.dropped_meta_fields),
        }
        template_block = {
            "template_id": self.config.template_id,
            "version": self.config.template_version,
            "frozen": True,
            "template_sha256": templates.template_sha256(self.config.template_id, request),
        }
        provider_block = {
            "label": captured.provider_label,
            "fixture": captured.provider_label == FIXTURE_ONLY_LABEL,
            "kind": captured.provider_kind,
            "transport": (
                "http_engine_capture" if captured.provider_kind == "engine" else "in_process_fixture"
            ),
            "note": (
                "label identifies the configured feature source: 'engine' is the real engine capture "
                "path, 'fixture_only' is a deterministic synthetic provider used by reports/s6 "
                "fixtures and the offline demo. provider mode never implies live validation."
            ),
        }
        capture_block = {
            "engine_id_forms": "cmpl-<caller> | cmpl-<caller>-<index> | cmpl-<caller>-<index>-<8hex>",
            "caller_request_id": captured.caller_request_id,
            "engine_response_id": captured.engine_response_id,
            "engine_request_id": captured.engine_request_id,
            "record_file_name": captured.record.file_stem() + ".npz",
            "dropped_meta_fields": list(captured.dropped_meta_fields),
            "error_marker_seen": captured.error_marker is not None,
            "error_marker": captured.error_marker,
            "error_marker_note": (
                "a capture marker fails a request only when it identifies that request or reports a "
                "global stop; helper-level errors and skip counters are diagnostics only"
            ),
        }
        if adapter_surface is not None:
            # The identity that is reported is the surface this arm ACTUALLY captured (the frozen
            # lettered template, tokenizer and adapter instance), while the probabilities stay in
            # the public contract of the original request.
            template_block = adapter_surface["template"]
            engine_block.update(adapter_surface["engine"])
            tokenizer_block["revision"] = (
                self.adapter_spec.tokenizer_revision if self.adapter_spec is not None else None
            )
            tokenizer_block["surface"] = "letters"
            provider_block = dict(provider_block)
            provider_block["adapter_alias"] = adapter_surface["capture"]["adapter_alias"]
            provider_block["adapter_lane"] = True
            provider_block["note"] = (
                provider_block["note"] + " This request used the experimental L1O adapter lane: the "
                "label still identifies the feature source (the engine capture path), and the "
                "served alias is reported separately as provider.adapter_alias."
            )
            capture_block.update(adapter_surface["capture"])
        used_adapter = isinstance(chosen, decision_adapter.L1OArm)
        return DecisionOutcome(
            result=readout.result,
            arm=arm_name,
            arm_block=readout.block,
            captured=captured,
            template_block=template_block,
            engine_block=engine_block,
            tokenizer_block=tokenizer_block,
            timing=timing,
            usage=usage,
            cache=captured.cache_block(),
            transport=captured.transport,
            uses_s2=False,
            provider_block=provider_block,
            parity=self.config.parity,
            capture_block=capture_block,
            seed=seed,
            adapter_block=self.adapter_status_block(used=used_adapter),
        )

    # -- the wide-label readout (17..256 options; separate type and schema) -------------
    def decide_wide(
        self,
        request: wide.WideChoiceRequest,
        *,
        arm: str | None = None,
        seed: int | None = None,
        deadline_s: float | None = None,
    ) -> WideOutcome:
        """Read out a wide choice request. Never touches the 24-slot vector or a trained arm."""
        if not isinstance(request, wide.WideChoiceRequest):
            raise BackendError(f"decide_wide() expects a WideChoiceRequest, got {type(request).__name__}")
        wc = getattr(self.config, "wide_choice", None)
        if wc is None or not wc.enabled:
            raise ConfigError(
                "the wide-label lane is not configured on this gateway (wide_choice absent or "
                "disabled); it is never enabled implicitly"
            )
        catalog = wc.require_catalog()
        if request.n_options > wc.max_options:
            raise BackendError(
                f"this wide request carries {request.n_options} options; the configured wide "
                f"maximum is {wc.max_options}"
            )
        arm_name = arm or self.config.default_arm
        if arm_name != "B0":
            raise ArmUnavailable(
                f"arm {arm_name!r} is not available for wide labels: the wide lane is a raw "
                "logit readout only (no trained head, no adapter, no native score)"
            )
        if self.config.default_arm != "B0":
            raise ArmUnavailable(
                f"the configured default arm is {self.config.default_arm!r}; the wide lane is "
                "only available on B0"
            )
        provider = self.provider_for(arm_name)
        acquire = getattr(provider, "acquire_wide", None)
        if not callable(acquire):
            raise BackendUnavailable(
                f"{type(provider).__name__} does not implement wide-label acquisition"
            )
        deadline = float(deadline_s) if deadline_s is not None else float(self.config.capture_deadline_s)
        captured = acquire(request, catalog, deadline_s=deadline, seed=seed)

        t0 = self.clock()
        result = wide.wide_choice_result(captured.capture.logits, request, catalog)
        t1 = self.clock()

        record = captured.capture.record
        template_block = wide.wide_template_identity(request, catalog)
        template_block = {
            **template_block,
            "prompt_token_sha256": captured.capture.prompt_token_sha256,
            "prompt_tokens_count": int(captured.capture.prompt_len),
            "note": (
                "the wide request uses its OWN frozen chat surface (one or two letters); the legacy "
                "qwen_chat_letters-v1 surface is never reused for a wide request"
            ),
        }
        engine_block = {
            "base_url": self.config.engine_base_url,
            "model": self.config.engine_model,
            "engine_request_id": captured.engine_request_id,
            "engine_response_id": captured.engine_response_id,
            "caller_request_id": captured.caller_request_id,
            "served_model": captured.served_model,
            "engine_build_version": record.engine_build_version,
            "helper_version": record.helper_version,
            "runner_source_sha256": record.runner_source_sha256,
            "logits_captured": bool(record.logits_captured),
            "head_scale": record.head_scale,
            "head_soft_cap": record.head_soft_cap,
            "head_dtype": record.head_dtype,
            "head_org_vocab_size": record.head_org_vocab_size,
            "lm_head_tp_size": record.lm_head_tp_size,
            "tp_rank": int(record.tp_rank),
            "world_size": record.world_size,
            "prompt_len": captured.capture.prompt_len,
            "terminal_position": captured.capture.terminal_position,
            "kernel_mode": record.kernel_mode,
            "cudagraph_mode": record.cudagraph_mode,
            "vector_source": record.vector_source,
            "token_hash_from_engine_state": bool(record.token_hash_from_engine_state),
        }
        tokenizer_block = {
            "revision": wc.tokenizer_revision,
            "add_special_tokens": bool(self.config.add_special_tokens),
            "prompt_tokens_count": int(captured.capture.prompt_len),
            "prompt_token_sha256": captured.capture.prompt_token_sha256,
            "token_hash_schema": features.TOKEN_HASH_SCHEMA,
            "surface": "wide_letters",
        }
        capture_block = {
            "schema": wide.WIDE_CAPTURE_SCHEMA,
            "wide_capture_uuid": captured.capture.request_uuid,
            "engine_id_forms": "cmpl-qfn-wide-<n>-<32hex>[-<index>[-<8hex>]]",
            "caller_request_id": captured.caller_request_id,
            "engine_request_id": captured.engine_request_id,
            "engine_response_id": captured.engine_response_id,
            "record_file_name": record.file_stem() + ".npz",
            "prompt_token_sha256": captured.capture.prompt_token_sha256,
            "prompt_len": captured.capture.prompt_len,
            "terminal_position": captured.capture.terminal_position,
            "tp_rank": captured.capture.tp_rank,
            "error_marker_seen": captured.error_marker is not None,
            "error_marker": captured.error_marker,
            "wide": {
                "schema": wide.WIDE_CAPTURE_SCHEMA,
                "catalog_schema": wide.CATALOG_SCHEMA,
                "catalog_digest": captured.capture.catalog_digest,
                "template_id": wide.WIDE_TEMPLATE_ID,
                "template_version": wide.WIDE_TEMPLATE_VERSION,
                "template_digest": captured.capture.template_digest,
                "label_count": captured.capture.label_count,
                "n_options": captured.capture.n_options,
                "request_uuid": captured.capture.request_uuid,
                "label_rows_sha256": _wide_id_sha256(captured.capture.wide_ids),
                "raw_logits_sha256": _wide_logits_sha256(captured.capture.logits),
            },
            "note": (
                "all 256 catalog label logits were captured from the same validated terminal raw "
                "logit row; only the first n labels enter the softmax"
            ),
        }
        provider_block = {
            "label": getattr(provider, "label", "unknown"),
            "fixture": getattr(provider, "label", "") == FIXTURE_ONLY_LABEL,
            "kind": getattr(provider, "kind", "engine"),
            "transport": (
                "http_engine_capture" if getattr(provider, "kind", "engine") == "engine"
                else "in_process_fixture"
            ),
            "wide_lane": True,
            "note": (
                "provider identifies the configured feature source; provider mode never implies "
                "live validation. This request used the wide-label lane."
            ),
        }
        readout_ms = (t1 - t0) * 1000.0
        timing = {
            "tokenize_ms": float(captured.timing_ms.get("tokenize", 0.0)),
            "request_ms": float(captured.timing_ms.get("request", 0.0)),
            "capture_wait_ms": float(captured.timing_ms.get("capture_wait", 0.0)),
            "readout_ms": readout_ms,
            "total_ms": float(captured.timing_ms.get("total", 0.0)) + readout_ms,
            "note": (
                "capture_wait is polling the engine's capture directory; the wide readout itself is "
                "a CPU softmax over n raw logits"
            ),
        }
        usage = {
            "prompt_tokens": int(captured.usage.get("prompt_tokens", 0)),
            "completion_tokens": int(captured.usage.get("completion_tokens", 0)),
            "total_tokens": int(captured.usage.get("total_tokens", 0)),
            "internal_generated_tokens": int(captured.usage.get("completion_tokens", 0)),
            "typed_output_tokens": TYPED_OUTPUT_TOKENS,
            "completion_text_returned": False,
            "completion_text_empty": True,
            "completion_finish_reason": captured.usage.get("completion_finish_reason"),
            "note": (
                "one internal completion token may be scheduled to make the engine run the capture "
                "path; the count is the engine's actual usage"
            ),
        }
        return WideOutcome(
            request=request,
            catalog=catalog,
            result=result,
            captured=captured,
            arm=arm_name,
            template_block=template_block,
            engine_block=engine_block,
            tokenizer_block=tokenizer_block,
            capture_block=capture_block,
            timing=timing,
            usage=usage,
            provider_block=provider_block,
            parity=self.config.parity,
            seed=seed,
        )


# ======================================================================================
# wide-label lane (separate from the 24-slot path; additive)
# ======================================================================================

@dataclasses.dataclass
class CapturedWideFeature:
    """One verified wide capture plus the transport facts the response must disclose."""

    capture: wide.WideCapture
    caller_request_id: str
    engine_request_id: str
    engine_response_id: str
    prompt_ids_count: int
    usage: dict
    timing_ms: dict
    served_model: str | None = None
    error_marker: dict | None = None


@dataclasses.dataclass
class WideOutcome:
    """The wide readout + every block the wide response reports."""

    request: wide.WideChoiceRequest
    catalog: wide.WideCatalog
    result: wide.WideResult
    captured: CapturedWideFeature
    arm: str
    template_block: dict
    engine_block: dict
    tokenizer_block: dict
    capture_block: dict
    timing: dict
    usage: dict
    provider_block: dict
    parity: dict
    seed: int | None = None


def _wide_id_sha256(ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes(order="C")).hexdigest()


def _wide_logits_sha256(logits: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(logits, dtype=np.float32).tobytes(order="C")
    ).hexdigest()


def wide_outcome_payload(outcome: WideOutcome) -> dict:
    """The JSON body of a wide ``/v1/decide`` response.

    Deliberately a separate schema with no 24-slot vector: the caller gets exactly ``n`` keyed
    probabilities, the wide adaptation/provenance block and the raw capture identity.
    """
    payload = dict(outcome.result.as_payload())
    payload["typed_output_tokens"] = TYPED_OUTPUT_TOKENS
    payload["internal_generated_tokens"] = int(outcome.usage["internal_generated_tokens"])
    payload["uses_s2"] = False
    payload["transport"] = TRANSPORT_ONE_DISCARDED_TOKEN
    payload["transport_note"] = (
        "the feature transport may schedule one discarded internal completion token; zero returned "
        "text is NOT a native zero-sampler engine implementation"
    )
    payload["arm"] = {
        "arm": outcome.arm,
        "lane": "raw_captured_label_logits",
        "trained": False,
        "calibration_status": "not_applicable",
        "note": (
            "wide-label lane: the raw engine logits at the catalog label rows, normalised over the "
            "n active labels; no trained head and no calibration is involved"
        ),
    }
    payload["template"] = outcome.template_block
    payload["tokenizer"] = outcome.tokenizer_block
    payload["engine"] = outcome.engine_block
    payload["capture"] = outcome.capture_block
    payload["timing"] = outcome.timing
    payload["usage"] = outcome.usage
    payload["provider"] = outcome.provider_block
    payload["parity"] = outcome.parity
    payload["seed"] = outcome.seed
    return _jsonable(payload, "decide_wide")


def _wide_catalog_from_config(config: Any) -> wide.WideCatalog:
    wc = getattr(config, "wide_choice", None)
    if wc is None or not wc.enabled:
        raise BackendUnavailable(
            "the wide-label lane is not enabled in this provider's configuration"
        )
    return wc.require_catalog()


class _SocketDeadlineWatchdog:
    """Shut down ONE request's socket when its absolute deadline expires.

    A receive timeout is reset by every byte the peer sends, so a slowly dripping response can hold
    a request open far past its budget. This watchdog only owns the request's own socket and only
    shuts it down, which interrupts a blocking ``recv``. It performs no I/O and issues no requests,
    so it can never extend a request or act after the gateway has answered.
    """

    def __init__(self, sock: Any, deadline: float) -> None:
        self._sock = sock
        self._fired = threading.Event()
        self._timer = threading.Timer(max(float(deadline) - time.monotonic(), 0.0), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self._fired.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    @property
    def fired(self) -> bool:
        return self._fired.is_set()

    def cancel(self) -> None:
        self._timer.cancel()


def _wide_deadline_phase(deadline: float, phase: str, budget: float) -> None:
    if time.monotonic() >= float(deadline):
        raise DeadlineExceeded(
            f"wide request deadline expired during {phase} (request budget {float(budget):.3f}s)"
        )


def _wide_post_json(
    provider: "HttpEngineCaptureProvider",
    path: str,
    payload: Mapping[str, Any],
    *,
    deadline: float,
    budget: float,
) -> tuple[int, Any]:
    """POST one wide-lane request under an ABSOLUTE deadline (never the shared default timeout).

    The per-phase socket timeout is clamped to the time actually left, and a request-local watchdog
    shuts the socket down at the absolute deadline so a dripping peer cannot extend it. A receive
    timeout that fires while there is still budget left stays a transport error.
    """
    cfg = provider.config
    _wide_deadline_phase(deadline, f"the request before {path}", budget)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    remaining = float(deadline) - time.monotonic()
    configured_timeout = float(cfg.socket_timeout_s)
    timeout = min(configured_timeout, max(remaining, 1e-3))
    # True when the request budget, not the configured socket timeout, is the binding constraint.
    deadline_binding = timeout < configured_timeout - 1e-9
    try:
        conn = provider._connection_with_timeout(timeout)
    except OSError as exc:  # pragma: no cover - constructor rarely fails
        raise TransportError(f"cannot open a connection to the engine: {exc}") from exc
    watchdog: _SocketDeadlineWatchdog | None = None
    try:
        conn.connect()
        watchdog = _SocketDeadlineWatchdog(conn.sock, deadline)
        conn.request(
            "POST", path, body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
    except (OSError, http.client.HTTPException) as exc:
        deadline_hit = time.monotonic() >= float(deadline) or (
            deadline_binding and isinstance(exc, socket.timeout)
        )
        if deadline_hit:
            raise DeadlineExceeded(
                f"wide request deadline expired during engine {path} "
                f"(request budget {float(budget):.3f}s)"
            ) from exc
        raise TransportError(f"engine {path} request failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if watchdog is not None:
            watchdog.cancel()
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    # A response that only lands at/after the deadline is NOT a success.
    _wide_deadline_phase(deadline, f"engine {path}", budget)
    text = raw.decode("utf-8", errors="replace") if raw else ""
    if status < 200 or status >= 300:
        raise TransportError(f"engine {path} returned HTTP {status}: {text[:400]}")
    try:
        return status, json.loads(text) if text else {}
    except json.JSONDecodeError as exc:
        raise TransportError(f"engine {path} returned non-JSON body: {text[:200]!r}") from exc


def _wide_tokenize_messages(
    provider: "HttpEngineCaptureProvider",
    messages: Sequence[Mapping[str, Any]],
    *,
    deadline: float,
    budget: float,
) -> list[int]:
    """Tokenize the wide prompt under the same absolute budget (no default-timeout call)."""
    payload = provider._tokenize_payload(messages=messages)
    _, doc = _wide_post_json(provider, "/tokenize", payload, deadline=deadline, budget=budget)
    return provider.validate_tokenize_doc(doc, source="/tokenize")


def _http_acquire_wide(
    provider: "HttpEngineCaptureProvider",
    request: wide.WideChoiceRequest,
    catalog: wide.WideCatalog,
    *,
    deadline_s: float | None = None,
    seed: int | None = None,
) -> CapturedWideFeature:
    """Wide acquisition over the engine's own HTTP surface, bounded by ONE absolute deadline.

    ``deadline_s`` is a duration for THIS request (default: the configured capture deadline). It is
    converted once into an absolute monotonic deadline that spans tokenization, the completion
    request and its response body, the capture poll and the readout. The provider, its config and
    the shared socket timeout are never mutated, so concurrent requests keep their own budgets.
    """
    if not isinstance(request, wide.WideChoiceRequest):
        raise BackendError(f"acquire_wide() expects a WideChoiceRequest, got {type(request).__name__}")
    if not isinstance(catalog, wide.WideCatalog):
        raise BackendError(f"expected a WideCatalog, got {type(catalog).__name__}")
    cfg = provider.config
    configured = _wide_catalog_from_config(cfg)
    if configured.digest != catalog.digest:
        raise BackendError("the catalog supplied to acquire_wide() is not the configured wide catalog")
    budget = float(deadline_s) if deadline_s is not None else float(cfg.capture_deadline_s)
    if not np.isfinite(budget) or budget <= 0:
        raise BackendError(f"deadline_s must be positive and finite, got {budget!r}")
    not_before = time.time_ns()
    n = request.n_options

    t0 = time.monotonic()
    # ONE absolute budget for the whole acquisition. Every phase below checks it, and the transport
    # clamps its socket timeout to the time actually left.
    deadline = t0 + budget
    messages = wide.wide_chat_messages(request, catalog)
    ids = _wide_tokenize_messages(provider, messages, deadline=deadline, budget=budget)
    prompt_hash = wide.wide_prompt_token_sha256(ids, expected_count=len(ids))
    _wide_deadline_phase(deadline, "tokenization", budget)
    t1 = time.monotonic()

    caller_request_id = wide.new_wide_caller_request_id(n)
    body = provider._completion_body(ids, caller_request_id, seed)
    t2 = time.monotonic()
    # no completion is sent once the budget is already gone
    _wide_deadline_phase(deadline, "the completion request", budget)
    _, doc = _wide_post_json(provider, "/v1/completions", body, deadline=deadline, budget=budget)
    t3 = time.monotonic()
    if not isinstance(doc, dict):
        raise TransportError(f"/v1/completions returned {type(doc).__name__}, expected an object")
    observed_model = doc.get("model")
    if cfg.expected_served_model is not None:
        if not isinstance(observed_model, str) or not observed_model:
            raise CaptureError(
                "the engine's /v1/completions response carries no served model but this arm "
                f"requires {cfg.expected_served_model!r}"
            )
        if observed_model != cfg.expected_served_model:
            raise CaptureError(
                f"engine served model {observed_model!r} does not equal the requested alias "
                f"{cfg.expected_served_model!r}"
            )
    engine_response_id = doc.get("id")
    accepted_response_id = resolve_engine_request_id(engine_response_id, caller_request_id, 0)
    usage = doc.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens != len(ids):
        raise CaptureError(
            f"engine usage.prompt_tokens {prompt_tokens!r} does not equal the {len(ids)} token ids "
            "this wide request sent"
        )
    choices = doc.get("choices") or []
    finish_reason = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")

    expected_engine_id = engine_request_id_for(caller_request_id, 0)
    request_uuid = wide.wide_request_uuid_from_engine_id(expected_engine_id)
    blob, path, engine_request_id, waited, marker = provider._wait_for_record(
        caller_request_id,
        prompt_hash,
        max(deadline - time.monotonic(), 1e-3),
        not_before,
        list(range(n)),
        accepted_engine_ids=(accepted_response_id,),
        wide_context={
            "catalog": catalog,
            "request_uuid": request_uuid,
            "n_options": n,
            "prompt_len": len(ids),
        },
    )
    # A record that only appears at/after the deadline is stale for THIS request: never 200.
    _wide_deadline_phase(deadline, "the capture wait", budget)
    t4 = time.monotonic()
    capture = wide.decode_wide_capture(
        blob,
        catalog=catalog,
        expected_request_uuid=request_uuid,
        expected_engine_request_id=engine_request_id,
        expected_prompt_token_sha256=prompt_hash,
        expected_prompt_len=len(ids),
        n_options=n,
        not_before_wall_ns=not_before,
    )
    _wide_deadline_phase(deadline, "the readout", budget)
    t5 = time.monotonic()
    return CapturedWideFeature(
        capture=capture,
        caller_request_id=caller_request_id,
        engine_request_id=engine_request_id,
        engine_response_id=str(engine_response_id or accepted_response_id),
        prompt_ids_count=len(ids),
        usage={
            "prompt_tokens": len(ids),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "completion_finish_reason": None if finish_reason is None else str(finish_reason),
        },
        timing_ms={
            "tokenize": (t1 - t0) * 1000.0,
            "request": (t3 - t2) * 1000.0,
            "capture_wait": waited * 1000.0,
            "readout": (t5 - t4) * 1000.0,
            "total": (t5 - t0) * 1000.0,
        },
        served_model=observed_model if isinstance(observed_model, str) else None,
        error_marker=marker,
    )


def _fixture_acquire_wide(
    provider: "FixtureFeatureProvider",
    request: wide.WideChoiceRequest,
    catalog: wide.WideCatalog,
    *,
    deadline_s: float | None = None,
    seed: int | None = None,
) -> CapturedWideFeature:
    """FIXTURE-ONLY wide acquisition: same record shape, same reader, no engine."""
    if not isinstance(request, wide.WideChoiceRequest):
        raise BackendError(f"acquire_wide() expects a WideChoiceRequest, got {type(request).__name__}")
    if not isinstance(catalog, wide.WideCatalog):
        raise BackendError(f"expected a WideCatalog, got {type(catalog).__name__}")
    t0 = time.monotonic()
    messages = wide.wide_chat_messages(request, catalog)
    ids = _fixture_tokenize_messages(messages, add_generation_prompt=True)
    prompt_hash = wide.wide_prompt_token_sha256(ids, expected_count=len(ids))
    not_before = time.time_ns()
    n = request.n_options
    caller_request_id = wide.new_wide_caller_request_id(n)
    engine_id = engine_request_id_for(caller_request_id, 0)
    vec = _fixture_vector("wide", provider.seed if seed is None else int(seed), "seeded", provider.fixed_vector)
    base = make_fixture_record(
        engine_request_id=engine_id,
        prompt_ids=ids,
        vector_fp32=vec,
        slot_logits=np.linspace(-1.0, 1.0, protocol.NUM_SLOTS, dtype=np.float32),
        prompt_hash=prompt_hash,
        capture_wall_ns=not_before,
    )
    logits = getattr(provider, "wide_logits", None)
    if logits is None:
        logits = np.linspace(-2.0, 2.0, catalog.n_labels, dtype=np.float32)
    blob = wide.add_wide_extension(
        fixture_only_npz_bytes(base),
        catalog=catalog,
        template_digest=wide.WIDE_TEMPLATE_CONTENT_SHA256,
        wide_request_uuid=wide.wide_request_uuid_from_engine_id(engine_id),
        prompt_token_sha256=prompt_hash,
        prompt_len=len(ids),
        n_options=n,
        logits256=logits,
    )
    name = expected_record_file_name(engine_id, prompt_hash, 0)
    (provider.capture_dir / name).write_bytes(blob)
    t1 = time.monotonic()
    capture = wide.decode_wide_capture(
        blob,
        catalog=catalog,
        expected_request_uuid=wide.wide_request_uuid_from_engine_id(engine_id),
        expected_engine_request_id=engine_id,
        expected_prompt_token_sha256=prompt_hash,
        expected_prompt_len=len(ids),
        n_options=n,
        not_before_wall_ns=not_before,
    )
    t2 = time.monotonic()
    return CapturedWideFeature(
        capture=capture,
        caller_request_id=caller_request_id,
        engine_request_id=engine_id,
        engine_response_id=engine_id,
        prompt_ids_count=len(ids),
        usage={
            "prompt_tokens": len(ids),
            "completion_tokens": 1,
            "total_tokens": len(ids) + 1,
            "completion_finish_reason": "fixture",
        },
        timing_ms={
            "tokenize": 0.0,
            "request": (t1 - t0) * 1000.0,
            "capture_wait": 0.0,
            "readout": (t2 - t1) * 1000.0,
            "total": (t2 - t0) * 1000.0,
        },
        served_model=provider.config.model,
    )
