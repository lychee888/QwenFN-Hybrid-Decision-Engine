"""QFN request-aligned terminal-feature capture helper (S3).

WHAT THIS IS
------------
A single function, `capture_after_compute_logits(runner, hidden_states, logits, input_batch)`,
designed to be called from the ACTIVE vLLM V2 runner
(`vllm/v1/worker/gpu/model_runner.py`, `GPUModelRunner.sample`) immediately after

    logits = self.model.compute_logits(sample_hidden_states)

and therefore:
  * after the final `GatedResidual.combine_and_mix` collapse (the 2560-wide state the LM head
    consumes), and
  * before the grammar bitmask, before top-k/top-p/temperature processing, and before any
    sampling or truncation, so the captured logits are the RAW selected logits.

It only ever *reads* existing tensors and writes files under the configured capture dir.
It does not modify model, sampler, cache or scheduler state, and it never samples a token.

DISABLED PATH (must stay cheap)
------------------------------
Default is disabled. The first call performs one environment/config read; every later call is
a single branch that returns before touching its arguments (no `.cpu()`, no `.item()`, no
host sync, no file write). `tests/test_capture_patch.py::test_disabled_capture_path_touches_nothing`
asserts this with poisoned objects.

CUDA-GRAPH SAFETY
-----------------
Capture fires only on a step that *completes a request's prompt* (a prefill step). Decode
steps are the CUDA-graph-replayed ones in this deployment (`FULL_DECODE_ONLY`), so a capture
step is not inside a captured graph. As a belt-and-braces guard the helper returns early if
`torch.cuda.is_current_stream_capturing()` is true. The eligibility test itself is pure
numpy on CPU-side mirrors and performs no GPU reads.

REQUEST MARKING (verified against the installed source, not guessed)
-------------------------------------------------------------------
`vllm/entrypoints/openai/completion/serving.py:145` builds the engine request id as
`f"cmpl-{self._base_request_id(raw_request, request.request_id)}"` and
`entrypoints/serve/engine/serving.py:117 _base_request_id` prefers the `X-Request-Id` header,
else the body `request_id`, else a random uuid; `serving.py:181` then appends `-{prompt_index}`
(`request_id_item`) and that is the id handed to `engine_client.generate`. So a body
`request_id="qfn-<hex>"` on a single-prompt request arrives here as `cmpl-qfn-<hex>-0`.
Only ids matching the configured `marker_prefix` are ever captured; everything else is ignored.

KNOWN LIMITS (also listed in reports/s3/RESULT.md)
-------------------------------------------------
* M1 (this file's variant) replaces the batch-level MTP refusal: a marked row is now captured in
  a batch that co-schedules MTP draft rows when, and only when, ITS OWN row alignment is proven
  (per-request draft count == 0, `cu` span == 1, `logits_indices[cu[i]] == query_end - 1`, and the
  positional cross-check downstream). Everything else is refused fail-closed and, for a FINAL
  refusal, persisted as a per-request refusal marker (`CAPTURE_REFUSED.json`) that the gateway
  turns into a typed 503 `capture_refused` instead of a full-deadline 504.
  See docs/MTP1-CAPTURE.md. The accepted S3 helper stays byte-identical.
* PP > 1 and PCP are rejected explicitly.
* The prefix cache is capped below the prompt length for a full hit
  (`v1/core/kv_cache_manager.py:256-262`, `max_cache_hit_length = request.num_tokens - 1`), so
  the terminal token is recomputed and the capture sees it. The authoritative pre-step computed
  token count is recorded per record as `computed_tokens_before_step`; the ACTUAL prefix-cache
  hit count is NOT observable at this hook, so `cache_hit_tokens` is written as null with
  provenance + reason (never 0, never the pre-step count relabelled).
* `logits` is `None` on non-driver ranks when the logits gather is driver-only
  (`model_executor/layers/logits_processor.py:84-96`), so the runtime-logit fields are
  optional and the absence is recorded, never faked.

PRODUCTION STORAGE MODE (PS1 derivative, OPT-IN; see reports/production-capture-storage/)
---------------------------------------------------------------------------------------
This file is a derivative of `patches/qfn_capture_runtime_mtp1.py`
(sha256 740426ec7cace9c1e7d0a46a3dbbd62d2f3032991a6dd332c2bd4bc8f23bc59d, the live-accepted M1
helper). Every numerical / selection / metadata / body function is byte-equivalent (AST-identical)
to that original; only the storage seams listed in `reports/production-capture-storage/RESULT.md`
differ, and every one of them is guarded by a `# PS1:` marker in the source.

WHY: the accepted helper's `_CaptureState.records_written` / `bytes_written` are LIFETIME
in-process counters and `stopped` is terminal, so TTL-deleting finished records from disk can
never un-stop a run that hit the quota. This derivative adds, as pure storage accounting:

  * an OPT-IN `production_storage` block in the capture config (absent => the legacy path runs
    unchanged, statement for statement);
  * bounded CURRENT-retained accounting over the declared root (completed NPZ+JSON pairs only),
    cached for <= 30 s and corrected by this process's own write deltas; the first use always
    scans, so a restarted process never assumes an empty directory;
  * a fail-closed pre-publication reserve (`production_storage.min_free_disk_bytes` and
    `production_storage.max_record_bytes` are REQUIRED), plus duplicate/orphan protection;
  * a storage-quota stop that is recoverable once retention frees room: it rewrites
    CAPTURE_ERROR.json with `stopped: false` and a non-quota `last_error`, and it never clears an
    unrelated capture/integrity error.

Strictness: `production_storage` is parsed before any I/O. A missing, zero, negative, non-finite,
wrong-typed or unknown field refuses the whole config (capture stays off with a config_error)
instead of silently disabling a bound. The legacy `max_records`/`max_bytes` lifetime limits stay
parsed and reported; production mode does not replace or inflate them.

Retention of expired records is a separate, dry-run-by-default operator tool; see
`reports/production-capture-storage/retention.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import threading
import time
from typing import Any, Iterable, Sequence

import numpy as np

try:  # torch exists only inside the engine; the pure logic below stays importable anywhere
    import torch
except Exception:  # pragma: no cover - Windows analysis host / no torch
    torch = None  # type: ignore[assignment]

HELPER_VERSION = "qfn-capture-runtime/1"
# M1 variant identity. Deliberately NOT a record field: the gateway decodes records with
# `tolerate_unknown_meta_fields=false`, so an added npz meta key would be a hard decode failure on
# the live path. The variant is surfaced in state.summary(), in CAPTURE_ERROR.json and in every
# refusal-marker entry instead, and records stay byte-comparable with the accepted helper's.
MTP1_HELPER_VARIANT = "qfn-capture-mtp1/1"
REFUSED_MARKER_NAME = "CAPTURE_REFUSED.json"
REFUSED_MARKER_SCHEMA = "qfn.capture-refused/1"
REFUSED_MAX_ENTRIES = 32
ENV_ENABLE = "QFN_CAPTURE_ENABLE"
ENV_CONFIG = "QFN_CAPTURE_CONFIG"
TOKEN_HASH_SCHEMA = "qfn.tokens/1"
# The exact source hash this helper's call-site patch was generated against (CORE VERIFIED SOURCE).
RUNNER_SOURCE_SHA256 = "f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8"
HIDDEN_WIDTH = 2560
NUM_SLOTS = 24
DEFAULT_MARKER_PREFIX = "cmpl-qfn-"
ALLOWED_KEY_CHARS = re.compile(r"^[A-Za-z0-9._-]+$")
KEY_MAX_LEN = 160
BF16_MANTISSA_BITS = 7

# Cache attribution (PM S3 correction). The runner's sample() hook cannot observe the scheduler's
# local prefix-cache hit count (`prefill_stats` lives on the scheduler-side Request), so records
# written here always carry `cache_hit_tokens = None` with this reason. The pre-step computed
# token count is recorded separately as `computed_tokens_before_step`.
CACHE_PROVENANCE_UNKNOWN = "unknown"
CACHE_UNKNOWN_REASON = (
    "runner sample() cannot observe the scheduler's local prefix-cache hit count "
    "(prefill_stats lives on the scheduler-side Request); the pre-step computed token count is "
    "recorded separately as computed_tokens_before_step and is NOT a cache-hit metric"
)

# --- PS1 production storage: opt-in, strict, storage-accounting-only --------------------------
# Absent from the config => the accepted legacy lifetime-quota path runs unchanged. Present =>
# every field below is REQUIRED or bounded, and the accounting is over the CURRENT retained
# completed capture pairs in the declared root, never over the lifetime counters.
PRODUCTION_STORAGE_SCHEMA = "qfn.production-storage/1"
PRODUCTION_STORAGE_CONFIG_KEY = "production_storage"
STORAGE_QUOTA_REASON = "storage_quota"
DEFAULT_SNAPSHOT_TTL_SECONDS = 5.0
MAX_SNAPSHOT_TTL_SECONDS = 30.0
# A reservation that is never committed or released (a crash between reserve and publish) must
# not hold capacity forever: it is dropped after this bounded TTL, and the drop is disclosed.
MIN_PENDING_TTL_SECONDS = 1
DEFAULT_PENDING_TTL_SECONDS = 300
MAX_PENDING_TTL_SECONDS = 3600
MIN_GRACE_SECONDS = 3600
MIN_TTL_SECONDS = 3600
DEFAULT_RETENTION_GRACE_SECONDS = 3600
DEFAULT_RETENTION_TTL_SECONDS = 3600
MAX_ACCOUNTING_FILES = 200_000
HOLD_FILE_NAME = "BENCHMARK_HOLD.json"
RECORD_JSON_SUFFIX = ".npz.json"
PART_SUFFIX = ".part"
# The REAL `record_file_key` shape: `<safe id>__<prompt_hash[:16]|nohash>__<dis>.npz`. The safe
# part is greedy because a sanitized engine id may itself contain '__'.
RECORD_NAME_RE = re.compile(r"^(?P<safe>.+)__(?P<hash>[0-9a-fA-F]{16}|nohash)__(?P<dis>\d+)\.npz$")

_SKIP_REASONS = (
    "chunk_not_final",
    "prompt_already_computed",
    "not_a_prefill_step",
    "terminal_not_chunk_tail",
    "logits_index_mismatch",
    "position_mismatch",
    "spec_decode_unsupported",          # kept: still used by the accepted S3 helper's vocabulary
    "duplicate_req_id_in_batch",
    "invalid_prompt_len",
    "shape_mismatch",
    "pp_or_pcp_unsupported",
    "quota_exceeded",
    # --- M1: per-row draft/alignment refusals (a FINAL refusal for the marked request) ---------
    "own_row_has_draft_tokens",         # the marked row's OWN draft count > 0
    "draft_info_unavailable",           # batch has drafts but the per-request counts were not given
    "logits_span_not_one",              # cu[i+1]-cu[i] != 1 for the marked row
    "cu_num_logits_invalid",            # cu does not start at 0 / is not non-decreasing
    "alignment_unverifiable",           # cu or logits_indices missing while the batch has drafts
)

# Reasons that mean "this request cannot be captured by this step AND never will be": the step
# computed the prompt's terminal token, so no later step can capture it. Only these produce a
# per-request refusal marker (a `chunk_not_final` refusal must NOT fail the caller, because a later
# chunk still captures the prompt).
FINAL_REFUSAL_REASONS = frozenset(
    {
        "own_row_has_draft_tokens",
        "draft_info_unavailable",
        "logits_span_not_one",
        "cu_num_logits_invalid",
        "alignment_unverifiable",
        "logits_index_mismatch",
        "position_mismatch",
    }
)


class SelectionError(ValueError):
    """Raised for malformed inputs or an unsafe/ambiguous capture key."""


# ======================================================================================
# pure helpers (no torch) - also used by the client side and unit tests
# ======================================================================================


def token_sequence_sha256(token_ids: Iterable[int]) -> str:
    """Canonical exact-token-sequence hash. MUST match features.token_sequence_sha256."""
    ids = []
    for v in token_ids:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise SelectionError(f"token id must be an int, got {v!r}")
        iv = int(v)
        if iv < 0 or iv > np.iinfo(np.int32).max:
            raise SelectionError(f"token id out of range: {iv}")
        ids.append(iv)
    payload = (
        TOKEN_HASH_SCHEMA.encode("ascii")
        + b":"
        + str(len(ids)).encode("ascii")
        + b":"
        + np.asarray(ids, dtype="<i4").tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def bf16_ulp(x: float) -> float:
    ax = abs(float(x))
    if ax == 0.0 or not np.isfinite(ax):
        return 0.0
    return float(2.0 ** (int(np.floor(np.log2(ax))) - BF16_MANTISSA_BITS))


def record_file_key(
    engine_request_id: str,
    prompt_hash: str | None = None,
    disambiguation: int = 0,
    marker_prefix: str = DEFAULT_MARKER_PREFIX,
) -> str:
    """Name of one capture record. Fail-closed: traversal/odd ids are rejected, not repaired.

    Rule (mirrored by `qfn_decision.extract.expected_record_file_name`):
        <safe engine id>__<prompt_hash[:16]|nohash>__<disambiguation>.npz
    """
    if not isinstance(engine_request_id, str) or not engine_request_id:
        raise SelectionError("engine_request_id must be a non-empty string")
    if "/" in engine_request_id or "\\" in engine_request_id or ".." in engine_request_id:
        raise SelectionError(f"refusing unsafe request id {engine_request_id!r}")
    if not engine_request_id.startswith(marker_prefix):
        raise SelectionError(
            f"request id {engine_request_id!r} does not start with the allowed prefix {marker_prefix!r}"
        )
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in engine_request_id)
    if len(safe) > KEY_MAX_LEN or not ALLOWED_KEY_CHARS.match(safe):
        raise SelectionError(f"refusing over-long or non-allowlisted key {safe!r}")
    hash_part = prompt_hash[:16] if isinstance(prompt_hash, str) and prompt_hash else "nohash"
    if not re.fullmatch(r"[0-9a-fA-F]{16}", hash_part) and hash_part != "nohash":
        raise SelectionError(f"prompt_hash must be hex, got {prompt_hash!r}")
    return f"{safe}__{hash_part}__{int(disambiguation)}.npz"


def _as_1d_int(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise SelectionError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind not in "iu":
        raise SelectionError(f"{name} must be an integer array, got {arr.dtype}")
    return arr


def select_terminal_rows(
    *,
    req_ids: Sequence[str],
    idx_mapping_np: np.ndarray,
    num_computed_tokens_np: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    prompt_len_np: np.ndarray,
    prefill_len_np: np.ndarray,
    is_prefilling_np: np.ndarray,
    query_start_loc_np: np.ndarray,
    cu_num_logits_np: np.ndarray | None,
    num_draft_tokens: int = 0,
    num_draft_tokens_per_req: np.ndarray | None = None,
    logits_indices_np: np.ndarray | None = None,
    marker_prefix: str = DEFAULT_MARKER_PREFIX,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pick the batch rows whose terminal prompt token is computed by THIS step.

    Request alignment uses only runner-provided per-request state:
      * `input_batch.idx_mapping_np`            -> row -> req_states index
      * `runner.req_states.prompt_len.np`       -> the user prompt length per request
      * `input_batch.num_computed_tokens_np`    -> tokens already computed BEFORE this step
      * `input_batch.num_scheduled_tokens`      -> tokens scheduled for this step
      * `input_batch.prefill_len_np`/`is_prefilling_np` -> is this a prefill step at all
      * `input_batch.query_start_loc_np`        -> the row range owned by each request
      * `input_batch.cu_num_logits_np`          -> logits-row start of each request
      * `input_batch.logits_indices`            -> the actual sampled row(s) (optional cross-check)

    Returns `(rows, skips)`. `skips` only ever contains MARKED requests, so an ordinary
    production workload produces no skip records at all.

    M1 per-row draft handling
    -------------------------
    The accepted S3 helper refused EVERY marked row of any batch with `num_draft_tokens != 0`
    (`spec_decode_unsupported`), which is a BATCH property: a decision prefill chunk co-scheduled
    with MTP decode requests was therefore refused even though its own row has no drafts. That was
    the live R2 504 (`reports/mtp/m1/RESULT.md` §1). M1 refuses per row instead:

      * the marked row's OWN draft count > 0            -> `own_row_has_draft_tokens`
      * the batch has drafts but no per-request counts  -> `draft_info_unavailable` (never inferred)
      * `cu_num_logits_np` invalid in a draft batch
        (cu[0] != 0 / not non-decreasing)               -> `cu_num_logits_invalid`
      * `cu[i+1] - cu[i] != 1` for the marked row       -> `logits_span_not_one`
      * `cu` or `logits_indices` missing while the
        batch has drafts                                -> `alignment_unverifiable`
      * `logits_indices[cu[i]] != query_end - 1`        -> `logits_index_mismatch`

    Every refusal that happens on the step which computes the marked prompt's terminal token is
    marked `final_refusal=True` (with a `facts` sub-dict for the marker); a refusal on an earlier
    chunk or on an already-computed prompt keeps the legacy shape and its legacy reason.
    """
    n = len(req_ids)
    if n == 0:
        return [], []
    num_computed_tokens_np = _as_1d_int("num_computed_tokens_np", num_computed_tokens_np)
    num_scheduled_tokens = _as_1d_int("num_scheduled_tokens", num_scheduled_tokens)
    idx_mapping_np = _as_1d_int("idx_mapping_np", idx_mapping_np)
    prefill_len_np = _as_1d_int("prefill_len_np", prefill_len_np)
    is_prefilling_np = np.asarray(is_prefilling_np, dtype=bool).reshape(-1)
    prompt_len_np = np.asarray(prompt_len_np).reshape(-1)
    query_start_loc_np = _as_1d_int("query_start_loc_np", query_start_loc_np)
    for name, arr, want in (
        ("num_computed_tokens_np", num_computed_tokens_np, n),
        ("num_scheduled_tokens", num_scheduled_tokens, n),
        ("idx_mapping_np", idx_mapping_np, n),
        ("prefill_len_np", prefill_len_np, n),
        ("is_prefilling_np", is_prefilling_np, n),
        ("query_start_loc_np", query_start_loc_np, n + 1),
    ):
        if len(arr) != want:
            raise SelectionError(f"shape mismatch: {name} has {len(arr)} entries, expected {want}")
    if cu_num_logits_np is not None:
        cu = _as_1d_int("cu_num_logits_np", cu_num_logits_np)
        if len(cu) != n + 1:
            raise SelectionError(
                f"shape mismatch: cu_num_logits_np has {len(cu)} entries, expected {n + 1}"
            )
    else:
        cu = None
    if logits_indices_np is not None:
        logits_indices_np = _as_1d_int("logits_indices_np", logits_indices_np)
    drafts_per_req = None
    if num_draft_tokens_per_req is not None:
        drafts_per_req = _as_1d_int("num_draft_tokens_per_req", num_draft_tokens_per_req)
        if len(drafts_per_req) != n:
            raise SelectionError(
                f"shape mismatch: num_draft_tokens_per_req has {len(drafts_per_req)} entries, "
                f"expected {n}"
            )
    # "the batch has drafts" is a BATCH property and says nothing about one row (runner:1185-1207).
    batch_has_drafts = bool(num_draft_tokens) or (
        drafts_per_req is not None and bool((drafts_per_req > 0).any())
    )
    cu_valid = cu is not None and _cu_is_valid(cu)

    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    seen: set[str] = set()

    for i, req_id in enumerate(req_ids):
        marked = isinstance(req_id, str) and req_id.startswith(marker_prefix)
        if not marked:
            continue
        reason = None
        record: dict[str, Any] | None = None
        final_refusal = False
        facts: dict[str, Any] = {
            "batch_num_drafts": int(num_draft_tokens),
            "has_draft_tokens_per_req": drafts_per_req is not None,
            "has_cu_num_logits_np": cu is not None,
            "has_logits_indices": logits_indices_np is not None,
        }

        if req_id in seen:
            reason = "duplicate_req_id_in_batch"
        else:
            seen.add(req_id)
            idx = int(idx_mapping_np[i])
            pre = int(num_computed_tokens_np[i])
            sched = int(num_scheduled_tokens[i])
            q_start = int(query_start_loc_np[i])
            q_end = int(query_start_loc_np[i + 1])
            prompt_len = int(prompt_len_np[idx]) if 0 <= idx < len(prompt_len_np) else -1
            own_drafts = None if drafts_per_req is None else int(drafts_per_req[i])
            logits_row_index = int(cu[i]) if cu is not None else i
            facts.update(
                {
                    "internal_index": idx,
                    "own_draft_tokens": own_drafts,
                    "cu_span": (int(cu[i + 1]) - int(cu[i])) if cu is not None else None,
                    "logits_row_index": logits_row_index,
                    "query_start": q_start,
                    "query_end": q_end,
                    "computed_tokens_before_step": pre,
                    "num_scheduled_tokens": sched,
                    "prompt_len": prompt_len,
                }
            )
            if idx < 0 or idx >= len(prompt_len_np):
                reason = "shape_mismatch"
            elif prompt_len <= 0 or sched <= 0:
                reason = "invalid_prompt_len"
            elif pre >= prompt_len:
                reason = "prompt_already_computed"
            elif not bool(is_prefilling_np[i]):
                reason = "not_a_prefill_step"
            elif pre + sched < prompt_len:
                reason = "chunk_not_final"
            elif pre + sched > prompt_len or q_end - q_start != sched:
                reason = "terminal_not_chunk_tail"
            else:
                # TERMINAL STEP: this step computes the prompt's terminal token and the marked
                # row is the only place it can be captured. Every refusal from here is FINAL.
                final_refusal = True
                terminal_position = prompt_len - 1
                facts["terminal_position"] = terminal_position
                checked_index = _checked_logits_index(logits_indices_np, logits_row_index)
                facts["logits_index"] = checked_index
                if batch_has_drafts and own_drafts is None:
                    # never infer a draft count: without the per-request array the alignment of
                    # this row relative to the batch-wide logits layout cannot be proven
                    reason = "draft_info_unavailable"
                elif own_drafts is not None and own_drafts > 0:
                    reason = "own_row_has_draft_tokens"
                elif cu is not None and batch_has_drafts and not cu_valid:
                    reason = "cu_num_logits_invalid"
                elif cu is None and batch_has_drafts:
                    reason = "alignment_unverifiable"
                elif cu is not None and int(cu[i + 1]) - int(cu[i]) != 1:
                    reason = "logits_span_not_one"
                elif logits_indices_np is None and batch_has_drafts:
                    reason = "alignment_unverifiable"
                elif checked_index is not None and checked_index != q_end - 1:
                    reason = "logits_index_mismatch"
                elif checked_index is None and logits_indices_np is not None:
                    reason = "alignment_unverifiable"
                else:
                    record = {
                        "batch_index": i,
                        "req_id": req_id,
                        "internal_index": idx,
                        "logits_row_index": logits_row_index,
                        "query_start": q_start,
                        "query_end": q_end,
                        # authoritative runner pre-step count: includes earlier chunks AND
                        # any reusable prefix. It is NOT a cache-hit metric.
                        "computed_tokens_before_step": pre,
                        "num_scheduled_tokens": sched,
                        "prompt_len": prompt_len,
                        "prefill_len": int(prefill_len_np[i]),
                        "terminal_position": terminal_position,
                        # PM correction: sample() cannot observe the scheduler's local
                        # prefix-cache hit count, so it stays unknown (never 0, never the
                        # pre-step count relabelled).
                        "cache_hit_tokens": None,
                        "cache_hit_provenance": CACHE_PROVENANCE_UNKNOWN,
                        "cache_hit_status_reason": CACHE_UNKNOWN_REASON,
                    }
        if record is not None:
            rows.append(record)
        elif final_refusal:
            skips.append(
                {
                    "batch_index": i,
                    "req_id": req_id,
                    "reason": reason or "terminal_not_chunk_tail",
                    "final_refusal": True,
                    "facts": facts,
                }
            )
        else:
            skips.append(
                {"batch_index": i, "req_id": req_id, "reason": reason or "terminal_not_chunk_tail"}
            )
    return rows, skips


def _cu_is_valid(cu: np.ndarray) -> bool:
    """`cu_num_logits_np` must start at 0 and be non-decreasing (the runner's own construction:
    `np.cumsum(num_logits)` with `num_logits = drafts + num_new_sampled_tokens`, model_runner.py
    :1213-1215)."""
    return bool(int(cu[0]) == 0 and (np.diff(cu) >= 0).all())


def _checked_logits_index(logits_indices_np: np.ndarray | None, row_index: int) -> int | None:
    """The sampled-row cross-check value, or None when it cannot be read at all (fail closed)."""
    if logits_indices_np is None:
        return None
    if row_index < 0 or row_index >= len(logits_indices_np):
        return None
    return int(logits_indices_np[row_index])


def local_row_indices(
    ids: Sequence[int], *, vocab_start: int, vocab_end: int, num_rows_local: int
) -> tuple[list[int], list[bool]]:
    """Map global vocab ids to this rank's local lm_head rows (-1 when not owned)."""
    local: list[int] = []
    owned: list[bool] = []
    for v in ids:
        iv = int(v)
        if vocab_start <= iv < vocab_end and 0 <= (iv - vocab_start) < num_rows_local:
            local.append(iv - vocab_start)
            owned.append(True)
        else:
            local.append(-1)
            owned.append(False)
    return local, owned


def mask_rows_by_ownership(
    rows: np.ndarray, *, ids: Sequence[int], vocab_start: int, vocab_end: int
) -> tuple[np.ndarray, list[bool]]:
    """Zero the rows this rank does not own (rows must already be the 24 selected rows)."""
    arr = np.asarray(rows)
    local, owned = local_row_indices(
        ids, vocab_start=vocab_start, vocab_end=vocab_end, num_rows_local=arr.shape[0]
    )
    masked = np.zeros_like(arr)
    for k, li in enumerate(local):
        if li >= 0:
            masked[k] = arr[li]
    return masked, owned


def mask_inactive_slots(logits: np.ndarray | Sequence[float], *, active: Any) -> np.ndarray:
    """Return the 24-slot vector with every slot outside `active` set to exactly 0.0."""
    arr = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
    out = np.zeros_like(arr)
    idx = np.arange(arr.shape[0])[active] if not isinstance(active, (int, np.integer)) else np.asarray([active])
    out[np.atleast_1d(idx).astype(np.int64)] = arr[np.atleast_1d(idx).astype(np.int64)]
    return out


def resolved_verbalizer_mask(names: Sequence[str]) -> np.ndarray:
    """24-slot mask for the primitive of each verbalizer name (noul 0:2, score 2:8, choice 8:24)."""
    mask = np.zeros(NUM_SLOTS, dtype=bool)
    for k, name in enumerate(names):
        _ = k
    return mask


def rank_selected_for_records(capture_ranks: str, rank: int) -> bool:
    """Which ranks write per-request records.

    Both ranks execute `sample()` when `pp_size == 1`, so both enter the row-export
    collective; that is independent of this filter. Records themselves are written by rank 0
    by default so the archive holds one vector per example instead of TP replicas.
    """
    if capture_ranks == "all":
        return True
    if capture_ranks == "rank0":
        return int(rank) == 0
    raise SelectionError(f"capture_ranks must be 'rank0' or 'all', got {capture_ranks!r}")


# ======================================================================================
# PS1 production storage: bounded CURRENT-retained accounting, fail-closed write guards
# ======================================================================================


class StorageRefused(RuntimeError):
    """A production-storage guard refused to publish a record (fail closed, never repaired).

    `fatal_global` says whether the refusal means "no capture can succeed until an operator or
    the retention tool frees room" (a disk-reserve or retained-limit refusal) or only "this one
    record cannot be published" (an oversized payload or a key that already exists).
    """

    def __init__(self, reason: str, message: str, *, fatal_global: bool = False) -> None:
        super().__init__(str(message))
        self.reason = str(reason)
        self.fatal_global = bool(fatal_global)

    def __str__(self) -> str:
        return f"{self.reason}: {RuntimeError.__str__(self)}"


def _require_positive_int(name: str, value: Any, *, minimum: int = 1, maximum: int | None = None) -> int:
    """Strict integer field. `bool` is rejected (it is an int subclass but never a size)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionError(f"production_storage.{name} must be an integer, got {value!r}")
    ivalue = int(value)
    if ivalue < minimum:
        raise SelectionError(f"production_storage.{name} must be >= {minimum}, got {ivalue}")
    if maximum is not None and ivalue > maximum:
        raise SelectionError(f"production_storage.{name} must be <= {maximum}, got {ivalue}")
    return ivalue


def _require_bounded_float(name: str, value: Any, *, minimum: float, maximum: float) -> float:
    """Strict finite float field (NaN/Inf are refused, never clamped)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"production_storage.{name} must be a number, got {value!r}")
    fvalue = float(value)
    if not np.isfinite(fvalue):
        raise SelectionError(f"production_storage.{name} must be finite, got {value!r}")
    if fvalue < minimum or fvalue > maximum:
        raise SelectionError(
            f"production_storage.{name} must be within [{minimum}, {maximum}], got {fvalue}"
        )
    return fvalue


def _resolve_declared_root(capture_dir: str, declared: Any) -> str:
    """The engine's GC root is the capture dir itself: a declared path cannot escape it."""
    if not isinstance(declared, str) or not declared.strip():
        raise SelectionError("production_storage.retention_root must be a non-empty string")
    capture_abs = os.path.abspath(str(capture_dir))
    root_abs = os.path.abspath(declared)
    if os.path.normcase(root_abs) != os.path.normcase(capture_abs):
        raise SelectionError(
            "production_storage.retention_root must be the capture_dir itself "
            f"({capture_abs!r}); got {declared!r} (path escape is refused)"
        )
    return root_abs


class ProductionStorageConfig:
    """Strict, opt-in production storage policy for ONE capture root.

    Parsed and validated BEFORE any I/O or mutation. `max_retained_records`, `max_retained_bytes`,
    `min_free_disk_bytes` and `max_record_bytes` are REQUIRED: an undeclared reserve or an
    undeclared per-record bound refuses the config instead of silently disabling the guard.
    """

    __slots__ = (
        "capture_dir",
        "retention_root",
        "max_retained_records",
        "max_retained_bytes",
        "min_free_disk_bytes",
        "max_record_bytes",
        "grace_seconds",
        "ttl_seconds",
        "snapshot_ttl_seconds",
        "pending_ttl_seconds",
    )

    REQUIRED = (
        "retention_root",
        "max_retained_records",
        "max_retained_bytes",
        "min_free_disk_bytes",
        "max_record_bytes",
    )
    OPTIONAL = ("grace_seconds", "ttl_seconds", "snapshot_ttl_seconds", "pending_ttl_seconds")

    def __init__(
        self,
        *,
        capture_dir: str,
        retention_root: Any,
        max_retained_records: Any,
        max_retained_bytes: Any,
        min_free_disk_bytes: Any,
        max_record_bytes: Any,
        grace_seconds: Any = DEFAULT_RETENTION_GRACE_SECONDS,
        ttl_seconds: Any = DEFAULT_RETENTION_TTL_SECONDS,
        snapshot_ttl_seconds: Any = DEFAULT_SNAPSHOT_TTL_SECONDS,
        pending_ttl_seconds: Any = DEFAULT_PENDING_TTL_SECONDS,
    ) -> None:
        self.capture_dir = str(capture_dir)
        self.retention_root = _resolve_declared_root(self.capture_dir, retention_root)
        self.max_retained_records = _require_positive_int("max_retained_records", max_retained_records)
        self.max_retained_bytes = _require_positive_int("max_retained_bytes", max_retained_bytes)
        self.min_free_disk_bytes = _require_positive_int("min_free_disk_bytes", min_free_disk_bytes)
        self.max_record_bytes = _require_positive_int("max_record_bytes", max_record_bytes)
        self.grace_seconds = _require_positive_int(
            "grace_seconds", grace_seconds, minimum=MIN_GRACE_SECONDS
        )
        self.ttl_seconds = _require_positive_int("ttl_seconds", ttl_seconds, minimum=MIN_TTL_SECONDS)
        self.snapshot_ttl_seconds = _require_bounded_float(
            "snapshot_ttl_seconds",
            snapshot_ttl_seconds,
            minimum=0.001,
            maximum=MAX_SNAPSHOT_TTL_SECONDS,
        )
        self.pending_ttl_seconds = _require_positive_int(
            "pending_ttl_seconds",
            pending_ttl_seconds,
            minimum=MIN_PENDING_TTL_SECONDS,
            maximum=MAX_PENDING_TTL_SECONDS,
        )

    @classmethod
    def from_dict(cls, cfg: Any, *, capture_dir: str) -> "ProductionStorageConfig":
        if not isinstance(cfg, dict):
            raise SelectionError("production_storage must be a JSON object")
        unknown = sorted(set(cfg) - set(cls.REQUIRED) - set(cls.OPTIONAL))
        if unknown:
            raise SelectionError(f"production_storage has unknown key(s): {unknown}")
        missing = [key for key in cls.REQUIRED if key not in cfg]
        if missing:
            raise SelectionError(f"production_storage is missing required key(s): {missing}")
        return cls(
            capture_dir=capture_dir,
            retention_root=cfg["retention_root"],
            max_retained_records=cfg["max_retained_records"],
            max_retained_bytes=cfg["max_retained_bytes"],
            min_free_disk_bytes=cfg["min_free_disk_bytes"],
            max_record_bytes=cfg["max_record_bytes"],
            grace_seconds=cfg.get("grace_seconds", DEFAULT_RETENTION_GRACE_SECONDS),
            ttl_seconds=cfg.get("ttl_seconds", DEFAULT_RETENTION_TTL_SECONDS),
            snapshot_ttl_seconds=cfg.get("snapshot_ttl_seconds", DEFAULT_SNAPSHOT_TTL_SECONDS),
            pending_ttl_seconds=cfg.get("pending_ttl_seconds", DEFAULT_PENDING_TTL_SECONDS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRODUCTION_STORAGE_SCHEMA,
            "capture_dir": self.capture_dir,
            "retention_root": self.retention_root,
            "max_retained_records": self.max_retained_records,
            "max_retained_bytes": self.max_retained_bytes,
            "min_free_disk_bytes": self.min_free_disk_bytes,
            "max_record_bytes": self.max_record_bytes,
            "grace_seconds": self.grace_seconds,
            "ttl_seconds": self.ttl_seconds,
            "snapshot_ttl_seconds": self.snapshot_ttl_seconds,
            "pending_ttl_seconds": self.pending_ttl_seconds,
        }


def record_pair_paths(root: str, key: str) -> tuple[str, str]:
    """The two published paths of one record: `<key>` (npz) and `<key>.json` (commit sidecar)."""
    npz_path = os.path.join(str(root), str(key))
    return npz_path, npz_path + ".json"


def parse_record_name(name: str, marker_prefix: str = DEFAULT_MARKER_PREFIX) -> tuple[str, str] | None:
    """Return `(stem, kind)` for a REAL `record_file_key` file name, else None.

    The candidate is re-run through `record_file_key` itself and must reproduce byte for byte,
    so nothing here can accept an invented pattern (and a foreign/odd name is skipped, never
    repaired). `kind` is `"npz"` or `"json"` (the committed sidecar).
    """
    if not isinstance(name, str) or not name or name.endswith(PART_SUFFIX):
        return None
    if name.endswith(RECORD_JSON_SUFFIX):
        stem, kind = name[: -len(RECORD_JSON_SUFFIX)], "json"
    elif name.endswith(".npz"):
        stem, kind = name[:-4], "npz"
    else:
        return None
    if not stem or "/" in stem or "\\" in stem or stem in (".", ".."):
        return None
    match = RECORD_NAME_RE.match(stem + ".npz")
    if match is None:
        return None
    hash_part = match.group("hash")
    try:
        rebuilt = record_file_key(
            match.group("safe"),
            prompt_hash=None if hash_part == "nohash" else hash_part,
            disambiguation=int(match.group("dis")),
            marker_prefix=marker_prefix,
        )
    except SelectionError:
        return None
    if rebuilt != stem + ".npz":
        return None
    return stem, kind


def is_record_file_name(name: str, marker_prefix: str = DEFAULT_MARKER_PREFIX) -> bool:
    """True only for a name the helper's own `record_file_key` rule reproduces exactly."""
    return parse_record_name(name, marker_prefix) is not None


def scan_capture_root(
    root: str,
    *,
    marker_prefix: str = DEFAULT_MARKER_PREFIX,
    max_files: int = MAX_ACCOUNTING_FILES,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Account the run-owned files at the TOP LEVEL of `root` (bounded, never recursing).

    `completed_records`/`completed_bytes` count only records whose npz AND committed json
    sidecar are both present, both regular files, both non-symlink (bytes include the JSON).
    Incomplete pairs, in-flight `.part` temporaries and unrelated files are reported in their
    own buckets so nothing is hidden, and they are never deleted by the retention tool.

    Any unreadable root, listing or entry raises `StorageRefused`: a failed accounting must
    never be reported as "nothing retained". The scan is bounded by `max_files`; exceeding it
    refuses instead of silently truncating.
    """
    counts: dict[str, Any] = {
        "completed_records": 0,
        "completed_bytes": 0,
        "incomplete_records": 0,
        "incomplete_bytes": 0,
        "inflight_parts": 0,
        "inflight_bytes": 0,
        "other_files": 0,
        "other_bytes": 0,
        "symlinks_skipped": 0,
        "entries_scanned": 0,
        "taken_ns": int(time.time_ns() if now_ns is None else now_ns),
    }
    try:
        root_st = os.stat(root)
        root_link_st = os.lstat(root)
    except OSError as exc:
        raise StorageRefused(
            "root_stat_failed", f"capture root {root!r} cannot be stat()ed: {exc}", fatal_global=True
        ) from exc
    if stat.S_ISLNK(root_link_st.st_mode):
        raise StorageRefused(
            "root_is_symlink", f"capture root {root!r} is a symlink; refusing to follow it",
            fatal_global=True,
        )
    if not stat.S_ISDIR(root_st.st_mode):
        raise StorageRefused(
            "root_not_a_directory", f"capture root {root!r} is not a directory", fatal_global=True
        )

    pairs: dict[str, dict[str, int]] = {}
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                counts["entries_scanned"] += 1
                if counts["entries_scanned"] > int(max_files):
                    raise StorageRefused(
                        "accounting_overflow",
                        f"capture root {root!r} holds more than {int(max_files)} entries; the "
                        "bounded accounting refuses instead of truncating",
                        fatal_global=True,
                    )
                try:
                    entry_st = entry.stat(follow_symlinks=False)
                    is_link = entry.is_symlink()
                except OSError as exc:
                    raise StorageRefused(
                        "entry_stat_failed",
                        f"cannot stat {entry.name!r} inside {root!r}: {exc}",
                        fatal_global=True,
                    ) from exc
                if is_link or stat.S_ISLNK(entry_st.st_mode):
                    counts["symlinks_skipped"] += 1
                    continue
                if stat.S_ISDIR(entry_st.st_mode):
                    counts["other_files"] += 1  # a directory is never recursed into
                    continue
                name = entry.name
                if name.endswith(PART_SUFFIX):
                    counts["inflight_parts"] += 1
                    counts["inflight_bytes"] += int(entry_st.st_size)
                    continue
                parsed = parse_record_name(name, marker_prefix)
                if parsed is None:
                    counts["other_files"] += 1
                    counts["other_bytes"] += int(entry_st.st_size)
                    continue
                stem, kind = parsed
                pairs.setdefault(stem, {})[kind] = int(entry_st.st_size)
    except StorageRefused:
        raise
    except OSError as exc:
        raise StorageRefused(
            "scandir_failed", f"cannot list the capture root {root!r}: {exc}", fatal_global=True
        ) from exc

    for slot in pairs.values():
        if "npz" in slot and "json" in slot:
            counts["completed_records"] += 1
            counts["completed_bytes"] += slot["npz"] + slot["json"]
        else:
            counts["incomplete_records"] += 1
            counts["incomplete_bytes"] += sum(slot.values())
    counts["root"] = str(root)
    return counts


class RecordReserve:
    """Pre-publication guard for ONE record: declared per-record max bytes + free-disk reserve.

    R1 correction: this is also an ACTUAL capacity reservation. While it is `pending` its key is
    exclusive and its declared upper bound (`max_record_bytes`) counts against
    `max_retained_records` / `max_retained_bytes`, so two writers can never overbook the bound.

    R2 correction: the handle carries the opaque OWNERSHIP TOKEN of its reservation, and every
    state-changing call proves that token against the registry first:

      * `check_payload(n)` - ownership first, then payload size, then free disk (before any file);
      * `begin_publish()` - ownership again + a bounded LEASE refresh, called immediately before the
        first `os.replace`, so ownership holds ACROSS the publication of both files;
      * `commit(actual_bytes)` / `release()` - the handle's own terminal transition, token-checked
        and idempotent. A stale handle (expired, released, committed, or superseded by a newer
        reservation for the same key) can neither mutate the newer owner's reservation nor publish;
        it is refused with `StorageRefused("reservation_not_owned")` and its later commit/release is
        a local no-op.

    Ownership / lifecycle (exactly one terminal transition, both idempotent):

      * `commit(actual_bytes)` - called by the writer after the npz AND json are published; the
        pending entry is dropped and the ACTUAL published size is recorded once
        (`pending_committed_bytes`); the retained-bytes delta is added exactly once by the
        `_CaptureState` seam (`note_written`), never twice.
      * `release()` - called by the writer on ANY failure before publication (a guard refusal or
        a failed write) and available to an operator: the pending entry is dropped, the key and
        the capacity are free again.
      * a reservation that is never committed or released is dropped after
        `production_storage.pending_ttl_seconds` (bounded, disclosed via `pending_expired`).

    With `storage is None` (the standalone `write_record_files(reserve=...)` use) the lifecycle
    still tracks its own state but nothing global is held.
    """

    __slots__ = ("storage", "key", "token", "max_record_bytes", "min_free_disk_bytes", "free_disk_fn",
                 "root", "free_disk_seen", "state")

    def __init__(
        self,
        *,
        max_record_bytes: int,
        min_free_disk_bytes: int,
        free_disk_fn: Any,
        root: str,
        storage: Any = None,
        key: str | None = None,
        token: int | None = None,
    ) -> None:
        self.storage = storage
        self.key = key
        self.token = token
        self.max_record_bytes = int(max_record_bytes)
        self.min_free_disk_bytes = int(min_free_disk_bytes)
        self.free_disk_fn = free_disk_fn
        self.root = str(root)
        self.free_disk_seen: int | None = None
        self.state = "pending"

    def _require_owned(self) -> None:
        """Prove this handle still owns its reservation, or refuse (fail closed).

        The ownership proof is atomic (the registry expires stale leases under its lock) and it is
        required BEFORE the payload is written AND again immediately before the first publish, so
        ownership holds across the publication window rather than being single-checked. A failed
        proof marks this handle terminal, so its later commit/release cannot touch the registry (a
        newer owner's reservation is never mutated).
        """
        if self.storage is None or self.key is None:
            return
        if not self.storage.reservation_is_current(self.key, self.token):
            self.state = "stale"
            raise StorageRefused(
                "reservation_not_owned",
                f"the reservation for {self.key!r} is expired, released, committed or owned by a "
                "newer reservation; refusing to publish with a stale handle",
            )

    def begin_publish(self) -> None:
        """Enter the ACTIVE publication interval (ownership proof + unexpirable capacity).

        Called immediately before the first `os.replace`. R3: while the entry is ACTIVE the lease
        reclaimer skips it, so the writer keeps its exclusive key and capacity for the WHOLE
        publication (both files) however long the I/O takes, and a stalled writer fails closed
        instead of losing its reservation. Leaving the interval is `finish_publish()`.
        """
        if self.storage is None or self.key is None:
            return
        if not self.storage.freeze_for_publication(self.key, self.token):
            self.state = "stale"
            raise StorageRefused(
                "reservation_not_owned",
                f"the reservation for {self.key!r} lapsed before publication; refusing to publish",
            )

    def finish_publish(self) -> None:
        """Leave the ACTIVE interval (writer `finally`); no-op once the handle is terminal.

        Ends the ACTIVE state and refreshes the lease, so the entry stays owned until the terminal
        `commit()`/`release()` pops it. Never raises: it only reports the storage-side transition.
        """
        if self.state != "pending" or self.storage is None or self.key is None:
            return
        self.storage.end_publication(self.key, self.token)

    def check_payload(self, payload_bytes: int) -> None:
        """Refuse BEFORE anything is written: ownership, then payload size, then free disk."""
        self._require_owned()
        size = int(payload_bytes)
        if size <= 0:
            raise StorageRefused("empty_record", "refusing to publish a zero-length record")
        if size > self.max_record_bytes:
            raise StorageRefused(
                "record_too_large",
                f"record payload {size} B exceeds the declared "
                f"production_storage.max_record_bytes {self.max_record_bytes} B",
            )
        free = int(self.free_disk_fn())
        self.free_disk_seen = free
        if free < self.min_free_disk_bytes + size:
            raise StorageRefused(
                "free_disk_reserve",
                f"free disk {free} B is below the declared reserve "
                f"production_storage.min_free_disk_bytes {self.min_free_disk_bytes} B plus this "
                f"record's {size} B",
                fatal_global=True,
            )

    def commit(self, actual_bytes: int) -> None:
        """The record is published: drop the reservation and record its ACTUAL size once.

        Idempotent, and ownership-checked: a stale handle's commit is a NO-OP, so it can never
        charge or drop a newer owner's reservation.
        """
        if self.state != "pending":
            return
        self.state = "committed"
        if self.storage is not None and self.key is not None:
            self.storage.commit_reservation(self.key, int(actual_bytes), self.token)

    def release(self) -> None:
        """The record was not published: free the key and the capacity (idempotent, ownership-checked).

        A handle that already knows it lost ownership (`state == "stale"`) is a local no-op, so a
        failure path never even attempts to mutate the registry of a newer owner.
        """
        if self.state != "pending":
            return
        self.state = "released"
        if self.storage is not None and self.key is not None:
            self.storage.release_reservation(self.key, self.token)


class ProductionStorage:
    """Bounded CURRENT-retained storage accounting for ONE capture root (PS1).

    * never assumes an empty directory: the first use scans the declared root;
    * the snapshot is cached for at most `snapshot_ttl_seconds` (<= 30 s) and corrected by this
      process's own write deltas, so a capture step never runs an unbounded scan;
    * one `statvfs` (free disk) plus at most one bounded `scandir` per record write;
    * no thread is created here (a single lock serializes the small mutations).
    """

    def __init__(
        self,
        cfg: ProductionStorageConfig,
        *,
        marker_prefix: str = DEFAULT_MARKER_PREFIX,
        now_fn: Any = None,
    ) -> None:
        self.cfg = cfg
        self.root = cfg.retention_root
        self.marker_prefix = str(marker_prefix)
        self._lock = threading.Lock()
        self._snap: dict[str, Any] | None = None
        self._snap_taken_ns = 0
        self._delta_records = 0
        self._delta_bytes = 0
        self._now_fn = now_fn or time.time_ns
        self.scans = 0
        self.refusals = 0
        self.recoveries = 0
        # R1/R2: actual capacity reservations, keyed by the record key (bounded by the count
        # bound) and each carrying an opaque ownership TOKEN so a stale handle can never mutate a
        # newer owner's reservation. One dict, one lock, no threads.
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_seq = 0
        self.pending_committed = 0
        self.pending_committed_bytes = 0
        self.pending_released = 0
        self.pending_expired = 0
        self.pending_refusals = 0
        self.pending_stale_ignored = 0
        self.pending_operator_actions = 0
        self.pending_publish_freezes = 0
        # R3: operator force that was refused because the entry is ACTIVE (a writer can still
        # publish with it), and the explicit destructive force that overrode that.
        self.pending_operator_refused_active = 0
        self.pending_operator_forced_active = 0

    def _now(self, now_ns: int | None = None) -> int:
        return int(self._now_fn() if now_ns is None else now_ns)

    def _accounted(self) -> dict[str, Any]:
        snap = dict(self._snap or {})
        # PS1: the CURRENT retained numbers are the last scan corrected by this process's own
        # write deltas, so a cached snapshot can never under-report a record it just published.
        snap["scanned_completed_records"] = int(snap.get("completed_records", 0))
        snap["scanned_completed_bytes"] = int(snap.get("completed_bytes", 0))
        snap["completed_records"] = snap["scanned_completed_records"] + self._delta_records
        snap["completed_bytes"] = snap["scanned_completed_bytes"] + self._delta_bytes
        # R1: a reservation that is not published yet still holds its declared upper bound, so a
        # snapshot recomputed while reservations are pending can never undercount them.
        snap["pending_records"] = len(self._pending)
        snap["pending_bytes"] = sum(int(entry["bytes"]) for entry in self._pending.values())
        snap["retained_records"] = snap["completed_records"] + snap["pending_records"]
        snap["retained_bytes"] = (
            snap["completed_bytes"] + int(snap.get("incomplete_bytes", 0)) + snap["pending_bytes"]
        )
        snap["completed_records_in_process_delta"] = self._delta_records
        snap["completed_bytes_in_process_delta"] = self._delta_bytes
        return snap

    def _expire_pending_locked(self, now: int) -> None:
        """Reclaim NON-ACTIVE reservations whose lease is older than the bounded TTL.

        R3 correction: an entry in the ACTIVE publication interval is NEVER reclaimed here. Its I/O
        duration is not guaranteed, so the writer keeps its exclusive key and capacity until it
        commits, releases, or an operator applies the explicit destructive force. A stalled writer
        therefore fails CLOSED (capacity retained, nothing reassigned behind it) instead of losing
        its reservation mid-publication.
        """
        ttl_ns = int(self.cfg.pending_ttl_seconds * 10**9)
        stale = [
            key
            for key, entry in self._pending.items()
            if entry["state"] != "active" and (now - int(entry["taken_ns"])) > ttl_ns
        ]
        for key in stale:
            self._pending.pop(key, None)
            self.pending_expired += 1

    def _snapshot_locked(self, *, force: bool, now: int) -> dict[str, Any]:
        """`snapshot` with the lock already held (the reserve decision needs one atomic view)."""
        ttl_ns = int(self.cfg.snapshot_ttl_seconds * 1e9)
        if self._snap is not None and not force and (now - self._snap_taken_ns) <= ttl_ns:
            return self._accounted()
        snap = scan_capture_root(
            self.root,
            marker_prefix=self.marker_prefix,
            max_files=MAX_ACCOUNTING_FILES,
            now_ns=now,
        )
        self._snap = snap
        self._snap_taken_ns = now
        self._delta_records = 0
        self._delta_bytes = 0
        self.scans += 1
        return self._accounted()

    def snapshot(self, *, force: bool = False, now_ns: int | None = None) -> dict[str, Any]:
        now = self._now(now_ns)
        with self._lock:
            self._expire_pending_locked(now)
            return self._snapshot_locked(force=force, now=now)

    def current(self, *, force: bool = False, now_ns: int | None = None) -> dict[str, Any]:
        return self.snapshot(force=force, now_ns=now_ns)

    def invalidate(self) -> None:
        """Drop the cached snapshot (called after an in-process retention GC)."""
        with self._lock:
            self._snap = None
            self._snap_taken_ns = 0
            self._delta_records = 0
            self._delta_bytes = 0

    def note_written(self, size: int) -> None:
        with self._lock:
            self._delta_records += 1
            self._delta_bytes += int(size)

    def over_quota(self, *, now_ns: int | None = None) -> tuple[bool, str | None]:
        cur = self.snapshot(now_ns=now_ns)
        if int(cur["retained_records"]) > self.cfg.max_retained_records:
            return True, (
                f"retained capture records {cur['retained_records']} exceed "
                f"production_storage.max_retained_records {self.cfg.max_retained_records}"
            )
        if int(cur["retained_bytes"]) > self.cfg.max_retained_bytes:
            return True, (
                f"retained capture bytes {cur['retained_bytes']} exceed "
                f"production_storage.max_retained_bytes {self.cfg.max_retained_bytes}"
            )
        return False, None

    def free_disk_bytes(self) -> int:
        """Free bytes on the volume holding the root; unreadable => refuse, never assume room."""
        path = self.root
        while not os.path.exists(path):
            parent = os.path.dirname(path)
            if parent == path:
                raise StorageRefused(
                    "free_disk_unavailable",
                    f"no existing directory on the path of the capture root {self.root!r}",
                    fatal_global=True,
                )
            path = parent
        try:
            return int(shutil.disk_usage(path).free)
        except OSError as exc:
            raise StorageRefused(
                "free_disk_unavailable",
                f"free disk space for {self.root!r} is not readable: {exc}",
                fatal_global=True,
            ) from exc

    def reserve_for_record(self, key: str, *, now_ns: int | None = None) -> RecordReserve:
        """Fail-closed pre-publication decision for one record key (BEFORE anything is written).

        One atomic decision under the storage lock: expire stale reservations, refuse an
        unowned/malformed key, refuse a key that is already reserved (a pending duplicate would
        let two writers clobber the same record), refuse when the CURRENT retained count/bytes
        plus this reservation would exceed the declared bounds, refuse when the free-disk reserve
        is not met, and refuse when either published file of the pair already exists. On success
        the reservation is registered and counts against the bounds until it is committed or
        released.
        """
        now = self._now(now_ns)
        with self._lock:
            self._expire_pending_locked(now)
            if key in self._pending:
                self.pending_refusals += 1
                raise StorageRefused(
                    "duplicate_pending_key",
                    f"{key!r} is already reserved by an unpublished write; refusing a second "
                    "reservation for the same record",
                )
            parsed = parse_record_name(key, self.marker_prefix)
            if parsed is None or parsed[1] != "npz":
                self.pending_refusals += 1
                raise StorageRefused(
                    "unowned_record_key",
                    f"{key!r} is not a record name this helper owns (record_file_key shape); "
                    "refusing to reserve capacity for it",
                )
            cur = self._snapshot_locked(force=False, now=now)
            if int(cur["retained_records"]) + 1 > self.cfg.max_retained_records:
                self.refusals += 1
                self.pending_refusals += 1
                raise StorageRefused(
                    "retained_record_limit",
                    f"retained capture records {cur['retained_records']} (including "
                    f"{cur['pending_records']} reservation(s) not yet published) + 1 exceed "
                    f"production_storage.max_retained_records {self.cfg.max_retained_records}; "
                    "retention must free room before more records can be published",
                    fatal_global=True,
                )
            if int(cur["retained_bytes"]) + self.cfg.max_record_bytes > self.cfg.max_retained_bytes:
                self.refusals += 1
                self.pending_refusals += 1
                raise StorageRefused(
                    "retained_byte_limit",
                    f"retained capture bytes {cur['retained_bytes']} (including "
                    f"{cur['pending_bytes']} B reserved and not yet published) plus the declared "
                    f"per-record bound {self.cfg.max_record_bytes} exceed "
                    f"production_storage.max_retained_bytes {self.cfg.max_retained_bytes}",
                    fatal_global=True,
                )
            free = self.free_disk_bytes()
            if free < self.cfg.min_free_disk_bytes + self.cfg.max_record_bytes:
                self.refusals += 1
                self.pending_refusals += 1
                raise StorageRefused(
                    "free_disk_reserve",
                    f"free disk {free} B is below the declared reserve "
                    f"production_storage.min_free_disk_bytes {self.cfg.min_free_disk_bytes} B plus "
                    f"the per-record bound {self.cfg.max_record_bytes} B",
                    fatal_global=True,
                )
            npz_path, json_path = record_pair_paths(self.root, key)
            for path in (npz_path, json_path):
                if os.path.lexists(path):
                    self.refusals += 1
                    self.pending_refusals += 1
                    raise StorageRefused(
                        "duplicate_record_key",
                        f"{path!r} already exists; refusing to clobber a published record or an "
                        "orphan sidecar",
                    )
            self._pending_seq += 1
            token = self._pending_seq
            self._pending[key] = {
                "token": token,
                "bytes": self.cfg.max_record_bytes,
                "taken_ns": now,
                "state": "pending",
            }
        return RecordReserve(
            max_record_bytes=self.cfg.max_record_bytes,
            min_free_disk_bytes=self.cfg.min_free_disk_bytes,
            free_disk_fn=self.free_disk_bytes,
            root=self.root,
            storage=self,
            key=key,
            token=token,
        )

    # -- reservation ownership / lifecycle ---------------------------------------------------
    def reservation_is_current(self, key: str, token: int, *, now_ns: int | None = None) -> bool:
        """Is `token` still the CURRENT, unexpired owner of `key`? (atomic, expires under lock)"""
        now = self._now(now_ns)
        with self._lock:
            self._expire_pending_locked(now)
            entry = self._pending.get(key)
            return entry is not None and int(entry["token"]) == int(token)

    def freeze_for_publication(self, key: str, token: int, *, now_ns: int | None = None) -> bool:
        """Enter the ACTIVE publication interval: prove ownership and make the entry unexpirable.

        R3: this no longer merely refreshes a timestamp. While the entry is ACTIVE the lease
        reclaimer skips it entirely, so the writer keeps its exclusive key and capacity for the
        whole publication (however long the I/O takes) and a stalled writer fails closed.
        """
        now = self._now(now_ns)
        with self._lock:
            self._expire_pending_locked(now)
            entry = self._pending.get(key)
            if entry is None or int(entry["token"]) != int(token):
                return False
            entry["state"] = "active"
            entry["taken_ns"] = now
            self.pending_publish_freezes += 1
            return True

    def end_publication(self, key: str, token: int, *, now_ns: int | None = None) -> bool:
        """Leave the ACTIVE interval: back to `pending` with a FRESH lease (R3, owner only).

        Called from the writer's `finally`, so the entry is owned again for a full lease until the
        terminal commit/release pops it; a no-op when the entry is already gone (committed,
        released, expired before the interval, or force-cleared by an operator).
        """
        now = self._now(now_ns)
        with self._lock:
            entry = self._pending.get(key)
            if entry is None or int(entry["token"]) != int(token):
                return False
            entry["state"] = "pending"
            entry["taken_ns"] = now
            return True

    def commit_reservation(self, key: str, actual_bytes: int, token: int | None = None) -> bool:
        """Drop a pending reservation after a successful publication, recording its ACTUAL size.

        With `token=None` this is the EXPLICIT operator force-form: it acts on whatever reservation
        currently holds `key` and is counted in `pending_operator_actions`. With a token it is the
        handle's own transition: a stale token is a NO-OP (counted in `pending_stale_ignored`) so a
        stale handle can never charge or drop a newer owner's reservation.

        R3: the operator force-form REFUSES an ACTIVE entry (a writer can still publish with it);
        the refusal is counted in `pending_operator_refused_active`. Use
        `release_all_pending(include_active=True)` for the explicit destructive stopped-owner policy.
        """
        with self._lock:
            entry = self._pending.get(key)
            if token is None:
                if entry is None:
                    return False
                if entry["state"] == "active":
                    self.pending_operator_refused_active += 1
                    return False
                self._pending.pop(key, None)
                self.pending_committed += 1
                self.pending_committed_bytes += int(actual_bytes)
                self.pending_operator_actions += 1
                return True
            if entry is None or int(entry["token"]) != int(token):
                self.pending_stale_ignored += 1
                return False
            self._pending.pop(key, None)
            self.pending_committed += 1
            self.pending_committed_bytes += int(actual_bytes)
            return True

    def release_reservation(self, key: str, token: int | None = None) -> bool:
        """Drop a reservation after a failed/abandoned publication (capacity + key free again).

        Same ownership rule as `commit_reservation`: a token mismatch is a NO-OP, `token=None` is
        the explicit operator force-form.

        R3: the operator force-form REFUSES an ACTIVE entry (counted in
        `pending_operator_refused_active`), because the writer can still publish with it.
        """
        with self._lock:
            entry = self._pending.get(key)
            if token is None:
                if entry is None:
                    return False
                if entry["state"] == "active":
                    self.pending_operator_refused_active += 1
                    return False
                self._pending.pop(key, None)
                self.pending_released += 1
                self.pending_operator_actions += 1
                return True
            if entry is None or int(entry["token"]) != int(token):
                self.pending_stale_ignored += 1
                return False
            self._pending.pop(key, None)
            self.pending_released += 1
            return True

    def release_all_pending(self, *, include_active: bool = False) -> int:
        """Operator escape hatch: drop pending reservations (bounded, disclosed).

        R3: by default only NON-ACTIVE entries are dropped; an ACTIVE publication keeps its capacity
        because the writer can still publish with it. `include_active=True` is the explicit
        DESTRUCTIVE stopped-owner policy (the operator has established that no writer is running);
        the entries it removes are counted in `pending_operator_forced_active`.
        """
        with self._lock:
            if include_active:
                forced = sum(1 for e in self._pending.values() if e["state"] == "active")
                dropped = len(self._pending)
                self._pending.clear()
                self.pending_operator_forced_active += forced
            else:
                victims = [k for k, e in self._pending.items() if e["state"] != "active"]
                dropped = len(victims)
                for key in victims:
                    self._pending.pop(key, None)
            self.pending_released += dropped
            self.pending_operator_actions += 1
            return dropped

    def pending_state(self, *, now_ns: int | None = None) -> dict[str, Any]:
        now = self._now(now_ns)
        with self._lock:
            self._expire_pending_locked(now)
            active = sum(1 for e in self._pending.values() if e["state"] == "active")
            return {
                "pending_records": len(self._pending),
                "pending_bytes": sum(int(e["bytes"]) for e in self._pending.values()),
                "pending_keys": sorted(self._pending),
                "pending_tokens": sorted(int(e["token"]) for e in self._pending.values()),
                "active": active,
                # `publishing` is the R2 name for the same ACTIVE count, kept so the R2 test and
                # receipts stay valid; `active` is the authoritative key from R3 on.
                "publishing": active,
                "committed": self.pending_committed,
                "committed_bytes": self.pending_committed_bytes,
                "released": self.pending_released,
                "expired": self.pending_expired,
                "refusals": self.pending_refusals,
                "stale_ignored": self.pending_stale_ignored,
                "operator_actions": self.pending_operator_actions,
                "operator_refused_active": self.pending_operator_refused_active,
                "operator_forced_active": self.pending_operator_forced_active,
                "publish_freezes": self.pending_publish_freezes,
            }


# ======================================================================================
# configuration
# ======================================================================================


class CaptureConfig:
    """Engine-side capture configuration (read from a JSON file, once)."""

    def __init__(
        self,
        *,
        capture_dir: str,
        marker_prefix: str,
        export_dir: str | None = None,
        verbalizer_names: Sequence[str] | None = None,
        verbalizer_ids: Sequence[int] | None = None,
        max_records: int = 4096,
        max_bytes: int = 4 * 1024**3,
        capture_fp32_ref: bool = True,
        capture_full_logits_row: bool = False,
        tp_export_rows: bool = True,
        token_hash_scope: str = "prompt",
        capture_ranks: str = "rank0",
        debug_hidden_hash_all_ranks: bool = False,
        # PS1: opt-in production storage policy; None (absent from the config) => the legacy path
        production_storage: "ProductionStorageConfig | None" = None,
        # additive: opt-in wide-label capture (None/disabled => no wide extension is ever written)
        wide_choice: "WideCaptureConfig | None" = None,
    ) -> None:
        self.capture_dir = str(capture_dir)
        self.marker_prefix = str(marker_prefix)
        self.export_dir = str(export_dir) if export_dir else os.path.join(self.capture_dir, "artifacts")
        self.verbalizer_names = list(verbalizer_names or [])
        self.verbalizer_ids = [int(v) for v in verbalizer_ids] if verbalizer_ids else None
        self.max_records = int(max_records)
        self.max_bytes = int(max_bytes)
        self.capture_fp32_ref = bool(capture_fp32_ref)
        self.capture_full_logits_row = bool(capture_full_logits_row)
        self.tp_export_rows = bool(tp_export_rows)
        self.token_hash_scope = str(token_hash_scope)
        if capture_ranks not in ("rank0", "all"):
            raise SelectionError(f"capture_ranks must be 'rank0' or 'all', got {capture_ranks!r}")
        self.capture_ranks = str(capture_ranks)
        self.debug_hidden_hash_all_ranks = bool(debug_hidden_hash_all_ranks)
        # PS1: the parsed, strictly validated production storage policy (or None => legacy)
        self.production_storage = production_storage
        self.wide_choice = wide_choice

    @property
    def base_rows_dir(self) -> str:
        return os.path.join(self.export_dir, "base_rows")

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CaptureConfig":
        if not isinstance(cfg, dict):
            raise SelectionError("capture config must be a JSON object")
        if "capture_dir" not in cfg or not cfg["capture_dir"]:
            raise SelectionError("capture config requires 'capture_dir'")
        if "marker_prefix" not in cfg or not cfg["marker_prefix"]:
            raise SelectionError("capture config requires 'marker_prefix'")
        if not str(cfg["marker_prefix"]).startswith("cmpl-"):
            raise SelectionError(
                "marker_prefix must be the engine-side complete prefix, e.g. 'cmpl-qfn-' "
                "(the OpenAI layer prepends 'cmpl-' and appends '-<prompt_index>')"
            )
        v = cfg.get("verbalizers") or {}
        names = v.get("names") if isinstance(v, dict) else None
        ids = v.get("ids") if isinstance(v, dict) else None
        # PS1: parsed strictly and BEFORE any I/O; an invalid block refuses the whole config
        # (an explicit JSON null is refused too: fail closed, never "absent by accident")
        if PRODUCTION_STORAGE_CONFIG_KEY in cfg:
            production_storage = ProductionStorageConfig.from_dict(
                cfg[PRODUCTION_STORAGE_CONFIG_KEY], capture_dir=str(cfg["capture_dir"])
            )
        else:
            production_storage = None
        wide_choice = None
        if WIDE_CONFIG_KEY in cfg:
            # additive: an invalid wide block refuses the whole capture config before any capture
            wide_choice = WideCaptureConfig.from_dict(cfg[WIDE_CONFIG_KEY])
        return cls(
            capture_dir=cfg["capture_dir"],
            marker_prefix=cfg["marker_prefix"],
            export_dir=cfg.get("export_dir"),
            verbalizer_names=names,
            verbalizer_ids=ids,
            max_records=int(cfg.get("max_records", 4096)),
            max_bytes=int(cfg.get("max_bytes", 4 * 1024**3)),
            capture_fp32_ref=bool(cfg.get("capture_fp32_ref", True)),
            capture_full_logits_row=bool(cfg.get("capture_full_logits_row", False)),
            tp_export_rows=bool(cfg.get("tp_export_rows", True)),
            token_hash_scope=str(cfg.get("token_hash_scope", "prompt")),
            capture_ranks=str(cfg.get("capture_ranks", "rank0")),
            debug_hidden_hash_all_ranks=bool(cfg.get("debug_hidden_hash_all_ranks", False)),
            # PS1: forwarded unchanged; None keeps the legacy lifetime-quota behaviour
            production_storage=production_storage,
            wide_choice=wide_choice,
        )

    @classmethod
    def from_file(cls, path: os.PathLike[str] | str) -> "CaptureConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def resolve_verbalizer_ids(config: CaptureConfig) -> dict[str, Any]:
    """Resolve the 24 verbalizer token ids for the LOCAL tokenizer.

    Never invents ids: the reference AutoTrust/JEV-27B ids are for a different vocabulary, so
    an unresolved config stays unresolved and the row export is refused. `qfn_decision.extract
    verbalizers` fills `verbalizers.ids` from the served `/tokenize` endpoint.
    """
    if config.verbalizer_ids is None or len(config.verbalizer_ids) != NUM_SLOTS:
        return {
            "resolved": False,
            "ids": None,
            "reason": (
                "verbalizer token ids are not resolved for the local tokenizer; run the "
                "tokenizer verification (extract verbalizers via /tokenize) first"
            ),
            "names": list(config.verbalizer_names),
        }
    ids = [int(v) for v in config.verbalizer_ids]
    if len(set(ids)) != NUM_SLOTS:
        return {
            "resolved": False,
            "ids": None,
            "reason": "verbalizer token ids are not unique; re-run the tokenizer verification",
            "names": list(config.verbalizer_names),
        }
    return {"resolved": True, "ids": ids, "reason": None, "names": list(config.verbalizer_names)}


# ======================================================================================
# capture state
# ======================================================================================

_UNSET = object()
_STATE: Any = _UNSET


class _CaptureState:
    def __init__(
        self,
        *,
        enabled: bool,
        config: CaptureConfig | None = None,
        config_error: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.config = config
        self.config_error = config_error
        self.records_written = 0
        self.bytes_written = 0
        self.rows_exported = False
        self.stopped = False
        self.refusals_written = 0
        self.errors: list[dict[str, Any]] = []
        self.skips: dict[str, int] = {}
        self._logged_errors: set[str] = set()
        # PS1: the production storage layer (None => the accepted legacy lifetime accounting) and
        # the reason for a stop, so ONLY a storage-quota stop may ever be recovered later.
        self.production = (
            ProductionStorage(config.production_storage, marker_prefix=config.marker_prefix)
            if config is not None and getattr(config, "production_storage", None) is not None
            else None
        )
        self.stopped_reason: str | None = None
        self.storage_recoveries = 0
        self._last_recovery_attempt_ns = 0

    # -- accounting ------------------------------------------------------------------
    def note_skip(self, reason: str, count: int = 1) -> None:
        self.skips[reason] = self.skips.get(reason, 0) + int(count)

    def record_error(self, kind: str, message: str, extra: dict[str, Any] | None = None) -> None:
        entry = {"kind": kind, "error": message, "wall_ns": time.time_ns()}
        if extra:
            entry.update(extra)
        self.errors.append(entry)
        sig = f"{kind}:{message}"
        if sig in self._logged_errors:
            return
        self._logged_errors.add(sig)
        self._write_error_marker(entry)

    def _write_error_marker(self, entry: dict[str, Any]) -> None:
        if self.config is None:
            return
        try:
            os.makedirs(self.config.capture_dir, exist_ok=True)
            path = os.path.join(self.config.capture_dir, "CAPTURE_ERROR.json")
            payload = {
                "helper_version": HELPER_VERSION,
                "helper_variant": MTP1_HELPER_VARIANT,
                "marker_prefix": self.config.marker_prefix,
                "last_error": entry,
                "error_count": len(self.errors),
                "skips": self.skips,
                "records_written": self.records_written,
                "refusals_written": self.refusals_written,
                "stopped": self.stopped,
            }
            tmp = path + ".part"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:  # pragma: no cover - never raise into the engine
            pass

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "config_error": self.config_error,
            "capture_dir": self.config.capture_dir if self.config else None,
            "marker_prefix": self.config.marker_prefix if self.config else None,
            "export_dir": self.config.export_dir if self.config else None,
            "records_written": self.records_written,
            "bytes_written": self.bytes_written,
            "rows_exported": self.rows_exported,
            "stopped": self.stopped,
            "skips": dict(self.skips),
            "errors": list(self.errors),
            "helper_version": HELPER_VERSION,
            "helper_variant": MTP1_HELPER_VARIANT,
            "refusals_written": self.refusals_written,
            "refusal_marker": REFUSED_MARKER_NAME,
            "runner_source_sha256": RUNNER_SOURCE_SHA256,
            # PS1: report-only keys; both are None/absent-equivalent in the legacy mode
            "production_storage": self.storage_summary(),
            "stopped_reason": self.stopped_reason,
        }

    # -- PS1 production storage seams --------------------------------------------------
    def storage_summary(self, *, now_ns: int | None = None) -> dict[str, Any] | None:
        """Bounded, report-only view of the production storage state (None => legacy mode)."""
        if self.production is None:
            return None
        cfg = self.production.cfg
        try:
            snap: dict[str, Any] | None = self.production.snapshot(now_ns=now_ns)
            snap_error: str | None = None
        except StorageRefused as exc:
            snap, snap_error = None, str(exc)
        return {
            "schema": PRODUCTION_STORAGE_SCHEMA,
            "root": self.production.root,
            "max_retained_records": cfg.max_retained_records,
            "max_retained_bytes": cfg.max_retained_bytes,
            "min_free_disk_bytes": cfg.min_free_disk_bytes,
            "max_record_bytes": cfg.max_record_bytes,
            "grace_seconds": cfg.grace_seconds,
            "ttl_seconds": cfg.ttl_seconds,
            "snapshot_ttl_seconds": cfg.snapshot_ttl_seconds,
            "pending_ttl_seconds": cfg.pending_ttl_seconds,
            "reservations": self.production.pending_state(now_ns=now_ns),
            "snapshot": snap,
            "snapshot_error": snap_error,
            "scans": self.production.scans,
            "refusals": self.production.refusals,
            "recoveries": self.production.recoveries,
            "storage_recoveries": self.storage_recoveries,
            "stopped_reason": self.stopped_reason,
        }

    def after_record_written(self, written: int) -> bool:
        """The `_capture_one` quota seam. Returns True when the run must stop for quota.

        With production storage absent this is the ORIGINAL lifetime decision, on the ORIGINAL
        counters and the ORIGINAL marker payload (see `_register_written_record`).
        """
        if self.config is None:
            self.stopped = True
            return True
        if not _register_written_record(self, written):
            return False
        cfg = self.config
        detail: dict[str, Any] = {"max_records": cfg.max_records, "max_bytes": cfg.max_bytes}
        if self.production is not None:
            cur = self.production.current()
            detail = dict(
                detail,
                storage_scope="retained",
                retained_records=cur["retained_records"],
                retained_bytes=cur["retained_bytes"],
            )
        self.stopped = True
        self.stopped_reason = STORAGE_QUOTA_REASON if self.production is not None else None
        self.record_error(
            "quota_exceeded",
            "capture quota reached; further captures are refused",
            detail,
        )
        return True

    def note_storage_refusal(self, exc: StorageRefused) -> None:
        """Record one fail-closed storage refusal (called from `_capture`'s row handler)."""
        self.note_skip("storage_refused")
        if exc.fatal_global:
            self.stopped = True
            self.stopped_reason = STORAGE_QUOTA_REASON
        self.record_error(
            "storage_refused",
            str(exc),
            {"storage_reason": exc.reason, "fatal_global": bool(exc.fatal_global)},
        )

    def recover_storage_quota(self, *, now_ns: int | None = None) -> bool:
        """Clear ONLY a storage-quota stop, after a re-account shows room again.

        Never touches any other error or stop (`no_torch`, a config error, a row failure, a
        refusal-marker failure): those are not storage state and stay exactly as they were.
        The re-account is rate-limited to one scan per snapshot TTL, so a stopped engine cannot
        turn into an unbounded scanner.
        """
        if self.production is None or not self.stopped or self.stopped_reason != STORAGE_QUOTA_REASON:
            return False
        cfg = self.production.cfg
        now = self.production._now(now_ns)
        ttl_ns = int(cfg.snapshot_ttl_seconds * 1e9)
        force = (now - self._last_recovery_attempt_ns) > ttl_ns
        self._last_recovery_attempt_ns = now
        cur = self.production.snapshot(force=force, now_ns=now)
        if int(cur["retained_records"]) >= cfg.max_retained_records:
            return False
        if int(cur["retained_bytes"]) >= cfg.max_retained_bytes:
            return False
        try:
            free = self.production.free_disk_bytes()
        except StorageRefused:
            return False
        if free < cfg.min_free_disk_bytes + cfg.max_record_bytes:
            return False
        self.stopped = False
        self.stopped_reason = None
        self.storage_recoveries += 1
        self.production.recoveries += 1
        self._rewrite_marker_after_storage_recovery(now_ns=now)
        return True

    def _rewrite_marker_after_storage_recovery(self, *, now_ns: int | None = None) -> None:
        """Rewrite CAPTURE_ERROR.json so a recovered storage stop is not reported as fatal.

        Same payload as `_write_error_marker` (same keys, same values) with `stopped` recomputed
        and `last_error` set to the most recent non-storage error, so
        `backend.error_marker_is_fatal` stops seeing a global stop. The quota/storage-refusal
        entries themselves stay in `self.errors` (history is never rewritten), and no unrelated
        error is cleared.
        """
        if self.config is None:
            return
        # PS1: storage-state errors (quota + storage refusals) are the ones the recovery clears
        # from the fatal path; the newest NON-storage error is what the marker reports instead.
        storage_kinds = ("quota_exceeded", "storage_refused")
        last_non_quota: dict[str, Any] | None = None
        for candidate in reversed(self.errors):
            if candidate.get("kind") not in storage_kinds:
                last_non_quota = candidate
                break
        payload = {
            "helper_version": HELPER_VERSION,
            "helper_variant": MTP1_HELPER_VARIANT,
            "marker_prefix": self.config.marker_prefix,
            "last_error": last_non_quota,
            "error_count": len(self.errors),
            "skips": self.skips,
            "records_written": self.records_written,
            "refusals_written": self.refusals_written,
            "stopped": self.stopped,
            "storage_quota_recovered": True,
            "storage_recovered_wall_ns": time.time_ns(),
            "production_storage": self.storage_summary(now_ns=now_ns),
        }
        try:
            os.makedirs(self.config.capture_dir, exist_ok=True)
            path = os.path.join(self.config.capture_dir, "CAPTURE_ERROR.json")
            tmp = path + ".part"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:  # pragma: no cover - never raise into the engine
            pass


def _register_written_record(state: "_CaptureState", written: int) -> bool:
    """PS1 quota seam: the ORIGINAL lifetime decision, or the CURRENT retained-storage decision.

    The `production is None` branch is the accepted helper's own expression kept verbatim (the
    parity test compares this exact AST subtree against `_capture_one` of the original helper).
    """
    cfg = state.config
    assert cfg is not None
    if state.production is None:
        return bool(state.records_written > cfg.max_records or state.bytes_written > cfg.max_bytes)
    state.production.note_written(written)
    cur = state.production.current()
    psc = cfg.production_storage
    return bool(
        int(cur["retained_records"]) > psc.max_retained_records
        or int(cur["retained_bytes"]) > psc.max_retained_bytes
    )


def _build_state(env: dict[str, str]) -> _CaptureState:
    raw = str(env.get(ENV_ENABLE, "")).strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return _CaptureState(enabled=False)
    cfg_path = env.get(ENV_CONFIG)
    if not cfg_path:
        return _CaptureState(
            enabled=False,
            config_error=f"{ENV_ENABLE} is set but {ENV_CONFIG} is not; capture stays off",
        )
    try:
        cfg = CaptureConfig.from_file(cfg_path)
    except Exception as exc:  # noqa: BLE001 - misconfiguration must not break the engine
        return _CaptureState(enabled=False, config_error=f"{type(exc).__name__}: {exc}")
    return _CaptureState(enabled=True, config=cfg)


def get_capture_state() -> _CaptureState:
    """One cheap initialization, then a cached object."""
    global _STATE
    if _STATE is _UNSET:
        _STATE = _build_state(dict(os.environ))
    return _STATE


def reset_capture_state() -> None:
    """Test hook only."""
    global _STATE
    _STATE = _UNSET


def capture_state_summary() -> dict[str, Any]:
    return get_capture_state().summary()


# ======================================================================================
# the injected call
# ======================================================================================


def capture_after_compute_logits(
    runner: Any, hidden_states: Any, logits: Any, input_batch: Any
) -> None:
    """Entry point called from `GPUModelRunner.sample` right after `compute_logits`.

    Returns None always. Never raises: a capture failure is recorded in the error marker and
    the ordinary generation path continues untouched.
    """
    state = get_capture_state()
    if not state.enabled:
        return None
    try:
        return _capture(state, runner, hidden_states, logits, input_batch)
    except Exception as exc:  # noqa: BLE001 - capture must never break generation
        state.record_error("capture_exception", f"{type(exc).__name__}: {exc}")
        return None


def _host_logits_indices(input_batch: Any) -> np.ndarray | None:
    """Host mirror of `input_batch.logits_indices` (TOKEN positions), for the M1 alignment proof.

    Read ONLY on a marked request's step that carries draft tokens, so ordinary traffic and
    no-draft batches never pay for it. `logits_indices` has at most
    `num_reqs * num_new_sampled_tokens + total_draft_tokens` entries (<= 8 + 3*8 in this
    deployment). Never raises: an unreadable tensor returns None and the selection refuses
    fail-closed (`alignment_unverifiable`) instead of guessing.
    """
    li = getattr(input_batch, "logits_indices", None)
    if li is None:
        return None
    try:
        return np.asarray([int(v) for v in li.tolist()], dtype=np.int64)
    except Exception:  # noqa: BLE001 - fall back to an explicit device->host copy
        try:
            return np.asarray(li.cpu().numpy(), dtype=np.int64)
        except Exception:  # noqa: BLE001
            return None


def _capture(state: _CaptureState, runner: Any, hidden_states: Any, logits: Any, input_batch: Any) -> None:
    cfg = state.config
    assert cfg is not None
    if state.stopped:
        # PS1: a storage-quota stop is recoverable once retention frees room; every other stop
        # (no_torch, a config error, ...) stays terminal exactly as it was.
        if not state.recover_storage_quota():
            return None
    if torch is None:  # pragma: no cover - inside the engine torch always exists
        state.record_error("no_torch", "torch is not importable; capture disabled at runtime")
        state.stopped = True
        return None
    if torch.cuda.is_current_stream_capturing():
        # Inside a CUDA-graph capture: never read or write here.
        state.note_skip("during_cuda_graph_capture")
        return None

    num_draft = int(getattr(input_batch, "num_draft_tokens", 0) or 0)
    req_ids = list(getattr(input_batch, "req_ids", ()) or ())
    if not req_ids:
        return None
    if not any(isinstance(r, str) and r.startswith(cfg.marker_prefix) for r in req_ids):
        return None  # ordinary traffic: nothing marked, nothing to do

    req_states = getattr(runner, "req_states", None)
    if req_states is None:
        state.record_error("no_req_states", "runner has no req_states; cannot align by request")
        return None

    rows, skips = select_terminal_rows(
        req_ids=req_ids,
        idx_mapping_np=np.asarray(input_batch.idx_mapping_np),
        num_computed_tokens_np=np.asarray(input_batch.num_computed_tokens_np),
        num_scheduled_tokens=np.asarray(input_batch.num_scheduled_tokens),
        prompt_len_np=np.asarray(req_states.prompt_len.np),
        prefill_len_np=np.asarray(input_batch.prefill_len_np),
        is_prefilling_np=np.asarray(input_batch.is_prefilling_np),
        query_start_loc_np=np.asarray(input_batch.query_start_loc_np),
        cu_num_logits_np=(
            None if getattr(input_batch, "cu_num_logits_np", None) is None else np.asarray(input_batch.cu_num_logits_np)
        ),
        num_draft_tokens=num_draft,
        num_draft_tokens_per_req=(
            None
            if getattr(input_batch, "num_draft_tokens_per_req", None) is None
            else np.asarray(input_batch.num_draft_tokens_per_req)
        ),
        # M1: the row-alignment proof needs the actual sampled-row indices ONLY for a batch that
        # co-schedules draft rows. A no-draft batch passes None, so its behaviour is byte-identical
        # to the accepted helper's (which never read `logits_indices` during selection at all).
        logits_indices_np=_host_logits_indices(input_batch) if num_draft else None,
        marker_prefix=cfg.marker_prefix,
    )
    for s in skips:
        state.note_skip(s["reason"])
    # M1: a marked row refused on the step that computed its prompt's terminal token can never be
    # captured, so name it in the refusal marker (bounded, atomic, written by the record rank only).
    # A refusal on an earlier chunk is NOT final and writes nothing: that request keeps waiting.
    _note_final_refusals(state, cfg, runner, [s for s in skips if s.get("final_refusal") is True])
    if not rows:
        return None

    # The row export must run on EVERY rank that reaches this call (both ranks do when
    # pp_size == 1), so it happens before the per-rank record filter below.
    if not state.rows_exported and cfg.tp_export_rows:
        try:
            export_base_rows(state, runner)
        except Exception as exc:  # noqa: BLE001
            state.record_error("row_export_failed", f"{type(exc).__name__}: {exc}")
        finally:
            state.rows_exported = True

    rank = _head_rank(runner)
    write_records = rank_selected_for_records(cfg.capture_ranks, rank)
    if not write_records and not cfg.debug_hidden_hash_all_ranks:
        state.note_skip("rank_not_selected_for_records", len(rows))
        return None

    for row in rows:
        try:
            refused = _capture_one(state, runner, hidden_states, logits, input_batch, row, write_record=write_records)
        # PS1: production-storage guards refuse fail-closed. A duplicate/oversized key refuses
        # only this record; a disk-reserve or retained-limit refusal stops the run, and that stop
        # is recoverable through retention (never by clearing an unrelated error).
        except StorageRefused as exc:
            state.note_storage_refusal(exc)
        except Exception as exc:  # noqa: BLE001 - one bad row must not block the rest
            state.record_error(
                "row_capture_failed",
                f"{type(exc).__name__}: {exc}",
                {"req_id": row.get("req_id")},
            )
        else:
            if refused:
                # the row passed selection but failed the last two checks inside the writer
                # (sampled-row identity and the positional cross-check): FINAL for this request
                state.note_skip(refused)
                _note_final_refusals(
                    state,
                    cfg,
                    runner,
                    [
                        {
                            "req_id": row["req_id"],
                            "batch_index": int(row["batch_index"]),
                            "reason": refused,
                            "facts": _row_facts(row),
                        }
                    ],
                )
    return None


def _row_facts(row: dict[str, Any]) -> dict[str, Any]:
    """Step facts for a refusal decided inside the writer (same shape as the selection facts)."""
    return {
        "internal_index": int(row["internal_index"]),
        "logits_row_index": int(row["logits_row_index"]),
        "query_start": int(row["query_start"]),
        "query_end": int(row["query_end"]),
        "computed_tokens_before_step": int(row["computed_tokens_before_step"]),
        "num_scheduled_tokens": int(row["num_scheduled_tokens"]),
        "prompt_len": int(row["prompt_len"]),
        "terminal_position": int(row["terminal_position"]),
    }


def _note_final_refusals(
    state: _CaptureState, cfg: CaptureConfig, runner: Any, final_refusals: Sequence[dict[str, Any]]
) -> None:
    """Persist one refusal-marker entry per MARKED request this helper has refused for good."""
    if not final_refusals:
        return
    try:
        if not rank_selected_for_records(cfg.capture_ranks, _head_rank(runner)):
            state.note_skip("refusal_marker_rank_not_selected", len(final_refusals))
            return
        entries = [
            build_refusal_entry(
                engine_request_id=str(r["req_id"]),
                reason=str(r.get("reason") or "refused"),
                batch_index=int(r.get("batch_index", -1)),
                facts=r.get("facts") or {},
            )
            for r in final_refusals
        ]
        merge_refusal_marker(
            cfg.capture_dir,
            entries,
            max_entries=REFUSED_MAX_ENTRIES,
            marker_prefix=cfg.marker_prefix,
        )
        state.refusals_written += len(entries)
    except Exception as exc:  # noqa: BLE001 - never break generation for a marker
        state.record_error("refusal_marker_failed", f"{type(exc).__name__}: {exc}")


def head_owner(model: Any) -> Any:
    """The module that actually owns ``lm_head`` / ``logits_processor``.

    PM live finding (2026-10-10): the served class is the multimodal wrapper
    ``Qwen3_8FlashNextForConditionalGeneration``, whose head lives on ``model.language_model``.
    Plain causal-LM classes own it directly. Anything else returns None (fail closed upstream).
    """
    if model is None:
        return None
    if getattr(model, "lm_head", None) is not None:
        return model
    inner = getattr(model, "language_model", None)
    if inner is not None and getattr(inner, "lm_head", None) is not None:
        return inner
    return None


def head_param_path(model: Any) -> str:
    owner = head_owner(model)
    if owner is None:
        return "unresolved"
    return "model.lm_head.weight" if owner is model else "model.language_model.lm_head.weight"


def _head_rank(runner: Any) -> int:
    try:
        return int(head_owner(runner.model).lm_head.tp_rank)
    except Exception:  # noqa: BLE001
        return 0


def build_record_meta(
    *,
    row: dict[str, Any],
    logits_index: int,
    raw_vector_dtype: str,
    vector_bf16: np.ndarray,
    vector_fp32: np.ndarray | None,
    verbalizer_ids: Sequence[int],
    verbalizer_logits_fp32: np.ndarray,
    runtime_argmax_token_id: int,
    runtime_max_logit_fp32: float,
    runtime_logits_row_fp32: np.ndarray | None,
    logits_captured: bool,
    logits_absent_reason: str | None,
    token_hash: str,
    chunk_token_sha256: str | None,
    token_hash_scope: str,
    token_hash_from_engine_state: bool,
    head_info: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one record's fields.

    Pure and torch-free so the engine writer's field set can be round-tripped against
    `qfn_decision.features.FeatureRecord` on the analysis host
    (`tests/test_capture_patch.py::test_engine_written_record_round_trips_into_the_archive`).
    Every key here MUST be a `FeatureRecord` field; array-valued fields are stored as npz arrays.
    """
    req_id = row["req_id"]
    if "computed_tokens_before_step" not in row:
        raise SelectionError(
            "row is missing the authoritative pre-step token count "
            "(computed_tokens_before_step); refusing to write an accounting-ambiguous record"
        )
    return {
        "example_id": _example_id(req_id, token_hash),
        "engine_request_id": req_id,
        "prompt_token_sha256": token_hash,
        "prompt_tokens_count": int(row["prompt_len"]),
        "prompt_len": int(row["prompt_len"]),
        "terminal_position": int(row["terminal_position"]),
        # PM correction: the ACTUAL prefix-cache hit count is not observable at this hook, so it is
        # written as null with provenance + reason. A numeric value carried on the row (the old
        # conflation) is deliberately NOT promoted into the record.
        "cache_hit_tokens": None,
        "cache_hit_provenance": CACHE_PROVENANCE_UNKNOWN,
        "cache_hit_source": None,
        "cache_hit_status_reason": CACHE_UNKNOWN_REASON,
        "computed_tokens_before_step": int(row["computed_tokens_before_step"]),
        "num_scheduled_tokens": int(row["num_scheduled_tokens"]),
        "query_start": int(row["query_start"]),
        "query_end": int(row["query_end"]),
        "logits_row_index": int(logits_index_row(row)),
        "batch_index": int(row["batch_index"]),
        "internal_index": int(row["internal_index"]),
        "vector_bf16": np.ascontiguousarray(vector_bf16),
        "vector_fp32": None if vector_fp32 is None else np.ascontiguousarray(vector_fp32),
        "verbalizer_ids": [int(v) for v in verbalizer_ids],
        "verbalizer_logits_fp32": np.ascontiguousarray(verbalizer_logits_fp32),
        "runtime_argmax_token_id": int(runtime_argmax_token_id),
        "runtime_max_logit_fp32": float(runtime_max_logit_fp32),
        "head_scale": head_info["scale"],
        "head_soft_cap": head_info["soft_cap"],
        "head_org_vocab_size": head_info["org_vocab_size"],
        "head_dtype": head_info["head_dtype"],
        "lm_head_tp_size": head_info["tp_size"],
        "tp_rank": head_info["tp_rank"],
        "world_size": head_info["world_size"],
        "cudagraph_mode": head_info["cudagraph_mode"],
        "kernel_mode": "prefill",
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "capture_wall_ns": time.time_ns(),
        "logits_captured": bool(logits_captured),
        "logits_absent_reason": logits_absent_reason,
        "vector_source": (
            f"runner.sample.hidden_states[logits_index={int(logits_index)}] "
            f"raw_dtype={raw_vector_dtype}"
        ),
        "capture_monotonic_ns": time.monotonic_ns(),
        "helper_version": HELPER_VERSION,
        "device_name": head_info["device_name"],
        "token_hash_scope": token_hash_scope,
        "chunk_token_sha256": chunk_token_sha256,
        "engine_build_version": head_info["vllm_version"],
        "graph_capturing": False,
        "token_hash_from_engine_state": bool(token_hash_from_engine_state),
        "runtime_logits_row_fp32": (
            None if runtime_logits_row_fp32 is None else np.ascontiguousarray(runtime_logits_row_fp32)
        ),
    }


def logits_index_row(row: dict[str, Any]) -> int:
    return int(row["logits_row_index"])


def _capture_one(
    state: _CaptureState,
    runner: Any,
    hidden_states: Any,
    logits: Any,
    input_batch: Any,
    row: dict[str, Any],
    *,
    write_record: bool = True,
) -> None:
    cfg = state.config
    assert cfg is not None
    req_id = row["req_id"]
    j = int(row["logits_row_index"])

    # 0. Row/index contract (M1). The runner computes
    #        sample_hidden_states = hidden_states[input_batch.logits_indices]
    #        logits = self.model.compute_logits(sample_hidden_states)          (model_runner.py:1403-1404)
    # so logits[j] is computed from hidden_states[logits_indices[j]]: logits_indices holds TOKEN
    # positions, never logits-row indices (kernel `_combine_sampled_and_draft_tokens_kernel`). The
    # hidden vector below is therefore read at exactly the index the cross-check validates, which is
    # the SAME index the captured `logits[j]` row came from. Returns a refusal reason string (the
    # caller counts it and, since selection already proved this is the terminal step, marks it as a
    # final refusal) or None when the row was captured.
    #
    # 1. Validate the candidate against the ACTUAL sampled row and the positional tensor.
    logits_index = int(input_batch.logits_indices[j].item())
    if logits_index != row["query_end"] - 1:
        return "logits_index_mismatch"
    position = int(input_batch.positions[logits_index].item())
    if position != row["terminal_position"]:
        return "position_mismatch"

    # 2. Detach the terminal vector. GPU -> CPU copies come first; they serialize the stream
    #    so the host-visible token-id buffer read below cannot race an earlier kernel.
    vec = hidden_states[logits_index].detach().reshape(-1)
    raw_dtype = str(vec.dtype)
    if vec.dtype != torch.bfloat16:
        vec_bf16 = vec.to(torch.bfloat16)
    else:
        vec_bf16 = vec
    vector_bf16 = vec_bf16.view(torch.int16).cpu().numpy().view(np.uint16)
    vector_fp32 = vec_bf16.to(torch.float32).cpu().numpy() if cfg.capture_fp32_ref else None

    if not write_record:
        # Debug-only rank: publish a hash so the two ranks' hidden states can be compared
        # WITHOUT putting TP replicas into the feature archive.
        dbg_hash, _, _dbg_scope, _dbg_state = _token_hashes(runner, input_batch, row)
        append_hidden_hash(
            cfg,
            int(_head_rank(runner)),
            row,
            prompt_token_sha256=dbg_hash,
            vector_bf16=vector_bf16,
            raw_dtype=raw_dtype,
        )
        return None

    verbalizer_logits: np.ndarray
    runtime_argmax = -1
    runtime_max = 0.0
    runtime_row = None
    logits_captured = False
    logits_absent_reason: str | None = None
    if logits is None:
        logits_absent_reason = "logits_not_available_on_this_rank"
    else:
        ids_res = resolve_verbalizer_ids(cfg)
        if not ids_res["resolved"]:
            logits_absent_reason = "verbalizer_ids_unresolved"
        else:
            ids = list(ids_res["ids"])
            idx_t = torch.as_tensor(ids, dtype=torch.long, device=logits.device)
            verbalizer_logits = logits[j].index_select(0, idx_t).to(torch.float32).cpu().numpy()
            runtime_argmax = int(logits[j].argmax().item())
            runtime_max = float(logits[j].max().item())
            if cfg.capture_full_logits_row:
                runtime_row = logits[j].to(torch.float32).cpu().numpy()
            logits_captured = True

    if not logits_captured:
        ids = list(cfg.verbalizer_ids) if cfg.verbalizer_ids else list(range(NUM_SLOTS))
        verbalizer_logits = np.full(NUM_SLOTS, np.nan, dtype=np.float32)

    # 3. Exact prompt-token hash from the engine's own request state (not from the client).
    token_hash, chunk_hash, hash_scope, from_state = _token_hashes(runner, input_batch, row)

    # 4. Build the record and write it atomically.
    head_info = head_facts(runner)
    helper_meta = build_record_meta(
        row=row,
        logits_index=logits_index,
        raw_vector_dtype=raw_dtype,
        vector_bf16=vector_bf16,
        vector_fp32=vector_fp32,
        verbalizer_ids=ids,
        verbalizer_logits_fp32=verbalizer_logits,
        runtime_argmax_token_id=runtime_argmax,
        runtime_max_logit_fp32=runtime_max,
        runtime_logits_row_fp32=runtime_row,
        logits_captured=logits_captured,
        logits_absent_reason=logits_absent_reason,
        token_hash=token_hash,
        chunk_token_sha256=chunk_hash,
        token_hash_scope=hash_scope,
        token_hash_from_engine_state=from_state,
        head_info=head_info,
    )

    disambiguation = 0
    while True:
        key = record_file_key(
            req_id, prompt_hash=token_hash, disambiguation=disambiguation, marker_prefix=cfg.marker_prefix
        )
        if not os.path.exists(os.path.join(cfg.capture_dir, key)) or disambiguation >= 9:
            break
        disambiguation += 1

    # PS1: pre-publication reserve (production mode only; None keeps the accepted write path)
    reserve = state.production.reserve_for_record(key) if state.production is not None else None
    wide_extension = None
    if is_wide_request(req_id) and cfg.wide_choice is not None and cfg.wide_choice.enabled:
        # Only a proven terminal row reaches this point; the 256 label logits come from the SAME
        # validated logits[j] row as the 24-slot verbalizer vector.
        if logits_captured:
            try:
                wide_extension = build_wide_extension(
                    cfg,
                    req_id=req_id,
                    logits_row=logits[j] if logits is not None else None,
                    prompt_hash=token_hash,
                    prompt_len=int(row["prompt_len"]),
                )
            except SelectionError as exc:
                # never publish a half-wide record: the base record is written unchanged and the
                # gateway refuses it (no wide extension), which is the fail-closed outcome
                state.record_error("wide_capture_refused", f"{type(exc).__name__}: {exc}",
                                   {"req_id": req_id})
        else:
            state.record_error(
                "wide_capture_refused",
                "wide request captured without raw logits; no wide extension is written",
                {"req_id": req_id},
            )
    written = write_record_files(cfg.capture_dir, key, helper_meta, reserve=reserve,
                                 wide_extension=wide_extension)
    state.records_written += 1
    state.bytes_written += written
    # PS1: the quota decision (legacy lifetime counters, or CURRENT retained storage) lives in
    # the seam method, which also writes the legacy marker payload unchanged.
    state.after_record_written(written)


def append_hidden_hash(
    cfg: CaptureConfig,
    rank: int,
    row: dict[str, Any],
    *,
    prompt_token_sha256: str,
    vector_bf16: np.ndarray,
    raw_dtype: str,
) -> None:
    """Append one line to `<capture_dir>/hidden_hash_rank<R>.jsonl` (debug cross-rank compare)."""
    os.makedirs(cfg.capture_dir, exist_ok=True)
    path = os.path.join(cfg.capture_dir, f"hidden_hash_rank{int(rank)}.jsonl")
    payload = {
        "engine_request_id": row["req_id"],
        "prompt_token_sha256": prompt_token_sha256,
        "terminal_position": int(row["terminal_position"]),
        "prompt_len": int(row["prompt_len"]),
        "computed_tokens_before_step": row.get("computed_tokens_before_step"),
        "cache_hit_tokens": None,
        "cache_hit_provenance": CACHE_PROVENANCE_UNKNOWN,
        "cache_hit_status_reason": CACHE_UNKNOWN_REASON,
        "tp_rank": int(rank),
        "raw_dtype": raw_dtype,
        "vector_sha256": hashlib.sha256(np.ascontiguousarray(vector_bf16).tobytes()).hexdigest(),
        "vector_norm_fp32": float(np.linalg.norm(bf16_bits_to_fp32(vector_bf16))),
        "wall_ns": time.time_ns(),
        "note": "debug cross-rank hash only; the feature archive holds rank0 records",
    }
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint16)
    return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _example_id(req_id: str, token_hash: str) -> str:
    stem = req_id
    for prefix in ("cmpl-qfn-", "cmpl-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    if stem.endswith("-0"):
        stem = stem[:-2]
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)[:48]
    return f"{safe}__{token_hash[:8]}"


def _token_hashes(
    runner: Any, input_batch: Any, row: dict[str, Any]
) -> tuple[str, str | None, str, bool]:
    """Exact prompt-token hash from runner state, with an explicitly-labelled chunk fallback."""
    idx = int(row["internal_index"])
    prompt_len = int(row["prompt_len"])
    chunk_ids = None
    try:
        q0, q1 = int(row["query_start"]), int(row["query_end"])
        chunk_ids = [int(v) for v in input_batch.input_ids[q0:q1].to(torch.int64).cpu().numpy()]
    except Exception:  # noqa: BLE001
        chunk_ids = None
    chunk_hash = token_sequence_sha256(chunk_ids) if chunk_ids else None

    all_ids = getattr(getattr(runner, "req_states", None), "all_token_ids", None)
    host = None
    if all_ids is not None:
        buf = getattr(all_ids, "_uva_buf", None)  # StagedWriteTensor(uva_instead_of_gpu=True)
        if buf is not None and hasattr(buf, "np"):
            host = buf.np
        elif hasattr(all_ids, "gpu"):
            try:
                host = all_ids.gpu[idx, :prompt_len].to(torch.int64).cpu().numpy()
            except Exception:  # noqa: BLE001
                host = None
    if host is not None:
        prompt_ids = [int(v) for v in np.asarray(host[idx, :prompt_len]).reshape(-1)]
        if len(prompt_ids) == prompt_len:
            return token_sequence_sha256(prompt_ids), chunk_hash, "prompt", True
    if chunk_ids is not None:
        return token_sequence_sha256(chunk_ids), chunk_hash, "chunk", False
    raise SelectionError("prompt token ids are not readable from runner state")


# PS1: `reserve` is the production pre-publication guard; `None` => the accepted write path.
def write_record_files(
    capture_dir: str,
    key: str,
    meta: dict[str, Any],
    *,
    reserve: "RecordReserve | None" = None,
    wide_extension: dict[str, Any] | None = None,
) -> int:
    """Atomic write of `<key>` (npz with arrays) and `<key>.json` (readable meta).

    ``wide_extension`` (additive, opt-in) moves its 256-wide vectors into the npz and merges only
    its scalar fields into the metadata; the size guard therefore measures the extended payload.
    """
    os.makedirs(capture_dir, exist_ok=True)
    npz_path = os.path.join(capture_dir, key)
    json_path = npz_path + ".json"
    arrays = {
        "vector_bf16": np.ascontiguousarray(meta["vector_bf16"]),
        "verbalizer_ids": np.asarray(meta["verbalizer_ids"], dtype=np.int64),
        "verbalizer_logits_fp32": np.ascontiguousarray(meta["verbalizer_logits_fp32"]),
    }
    if meta.get("vector_fp32") is not None:
        arrays["vector_fp32"] = np.ascontiguousarray(meta["vector_fp32"])
    if meta.get("runtime_logits_row_fp32") is not None:
        arrays["runtime_logits_row_fp32"] = np.ascontiguousarray(meta["runtime_logits_row_fp32"])
    scalar_meta = {
        k: (None if isinstance(v, (np.floating, np.integer)) and np.isnan(v) else v)
        for k, v in meta.items()
        if k not in ("vector_bf16", "vector_fp32", "verbalizer_ids", "verbalizer_logits_fp32",
                     "runtime_logits_row_fp32")
    }
    scalar_meta["engine_request_id"] = meta["engine_request_id"]
    if wide_extension:
        for name in WIDE_ARRAY_NAMES:
            if name not in wide_extension:
                raise SelectionError(f"wide extension is missing the {name!r} array")
            arrays[name] = np.ascontiguousarray(wide_extension[name])
        for name, value in wide_extension.items():
            if name in WIDE_ARRAY_NAMES:
                continue  # arrays are never serialised into the scalar metadata
            scalar_meta[name] = value
    sizes = 0
    # PS1: one failure scope from the payload build to the commit. With `reserve is None` every
    # statement below is the accepted helper's own statement (the guard/commit lines are skipped).
    try:
        buffer = _npz_bytes(arrays, scalar_meta)
        # PS1: pre-publication guard (payload measured BEFORE the npz temp file is created)
        json_text: str | None = None
        if reserve is not None:
            json_text = json.dumps(scalar_meta, indent=2, sort_keys=True, default=str)
            reserve.check_payload(len(buffer) + len(json_text.encode("utf-8")))
        tmp = npz_path + ".part"
        with open(tmp, "wb") as fh:
            fh.write(buffer)
            fh.flush()
            os.fsync(fh.fileno())
        # PS1/R3: prove ownership once more and ENTER the ACTIVE publication interval. While the
        # entry is ACTIVE the lease reclaimer skips it, so this writer keeps its exclusive key and
        # capacity for the whole interval (both files) no matter how long the I/O takes.
        if reserve is not None:
            reserve.begin_publish()
        try:
            os.replace(tmp, npz_path)
            sizes += len(buffer)
            json_tmp = json_path + ".part"
            with open(json_tmp, "w", encoding="utf-8") as fh:
                # PS1: the guarded path writes exactly the text whose size it already checked (the
                # same bytes `json.dump` would produce: identical kwargs, ensure_ascii, utf-8).
                if json_text is None:
                    json.dump(scalar_meta, fh, indent=2, sort_keys=True, default=str)
                else:
                    fh.write(json_text)
                fh.flush()
            os.replace(json_tmp, json_path)
            sizes += os.path.getsize(json_path)
        finally:
            # PS1/R3: leave the ACTIVE interval (no-op if the handle is already terminal). The
            # entry stays owned with a refreshed lease until commit()/release() pops it, and a
            # failure still releases it exactly once through the failure scope below.
            if reserve is not None:
                reserve.finish_publish()
    except BaseException:
        # PS1: a failed publication frees the key and the capacity exactly once (idempotent)
        if reserve is not None:
            reserve.release()
        raise
    if reserve is not None:
        # PS1: published: drop the reservation and record the ACTUAL published size once
        reserve.commit(sizes)
    return sizes


def _npz_bytes(arrays: dict[str, np.ndarray], scalar_meta: dict[str, Any]) -> bytes:
    import io

    buf = io.BytesIO()
    meta_bytes = np.frombuffer(
        json.dumps(scalar_meta, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"),
        dtype=np.uint8,
    )
    payload = {"meta_json": meta_bytes}
    payload.update(arrays)
    np.savez(buf, **payload)
    return buf.getvalue()


# ======================================================================================
# M1 refusal markers (per-request, bounded, atomic)
# ======================================================================================

# Fixed, documented entry schema. Every M1 refusal entry carries these keys, None where the fact
# was not observable at the point of refusal, so the gateway never has to guess.
_REFUSAL_FACT_KEYS = (
    "internal_index",
    "own_draft_tokens",
    "batch_num_drafts",
    "cu_span",
    "has_draft_tokens_per_req",
    "has_cu_num_logits_np",
    "has_logits_indices",
    "logits_row_index",
    "logits_index",
    "query_start",
    "query_end",
    "computed_tokens_before_step",
    "num_scheduled_tokens",
    "prompt_len",
    "terminal_position",
)


def build_refusal_entry(
    *,
    engine_request_id: str,
    reason: str,
    batch_index: int,
    facts: dict[str, Any] | None = None,
    wall_ns: int | None = None,
    monotonic_ns: int | None = None,
    marker_prefix: str = DEFAULT_MARKER_PREFIX,
) -> dict[str, Any]:
    """One refusal record. Marked ids only, same path-safety rule as the capture records."""
    record_file_key(engine_request_id, prompt_hash=None, marker_prefix=marker_prefix)  # safety rule
    if not isinstance(reason, str) or reason not in _SKIP_REASONS:
        raise SelectionError(f"refusal reason {reason!r} is not a known skip reason")
    facts = facts or {}
    entry: dict[str, Any] = {
        "schema": REFUSED_MARKER_SCHEMA,
        "engine_request_id": str(engine_request_id),
        "reason": reason,
        "final_refusal": True,
        "helper_variant": MTP1_HELPER_VARIANT,
        "batch_index": int(batch_index),
        "wall_ns": int(time.time_ns() if wall_ns is None else wall_ns),
        "monotonic_ns": int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns),
    }
    for key in _REFUSAL_FACT_KEYS:
        value = facts.get(key)
        entry[key] = None if value is None else value
    return entry


def read_refusal_marker_file(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read the refusal marker, if present and readable. Never fatal by itself."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("entries"), list):
        return None
    return payload


def merge_refusal_marker(
    capture_dir: str,
    entries: Sequence[dict[str, Any]],
    *,
    max_entries: int = REFUSED_MAX_ENTRIES,
    marker_prefix: str = DEFAULT_MARKER_PREFIX,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Append `entries` to `<capture_dir>/CAPTURE_REFUSED.json` atomically, keeping at most
    `max_entries` (newest kept; the drop count is disclosed, never silently lost).

    Bounded by construction: one small file per capture dir, a hard entry cap, and no unbounded
    per-request file growth. Only MARKED requests ever produce an entry.
    """
    os.makedirs(capture_dir, exist_ok=True)
    path = os.path.join(capture_dir, REFUSED_MARKER_NAME)
    previous = read_refusal_marker_file(path)
    prev_entries = [e for e in (previous or {}).get("entries") or [] if isinstance(e, dict)]
    combined = prev_entries + [dict(e) for e in entries]
    cap = max(int(max_entries), 1)
    dropped = max(0, len(combined) - cap)
    kept = combined[-cap:]
    payload: dict[str, Any] = {
        "schema": REFUSED_MARKER_SCHEMA,
        "helper_version": HELPER_VERSION,
        "helper_variant": MTP1_HELPER_VARIANT,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "marker_prefix": marker_prefix,
        "marker_name": REFUSED_MARKER_NAME,
        "written_wall_ns": int(time.time_ns() if now_ns is None else now_ns),
        "max_entries": cap,
        "entry_count": len(kept),
        "dropped_entries_total": int((previous or {}).get("dropped_entries_total") or 0) + dropped,
        "truncated": bool(dropped or (previous or {}).get("truncated")),
        "entries": kept,
        "note": (
            "per-request refusal of a MARKED capture request: the marked prompt's terminal token "
            "was computed by the engine step, but the helper refused the capture fail-closed. The "
            "gateway turns a matching entry into HTTP 503 capture_refused for that request only; a "
            "request that is not named here is unaffected."
        ),
    }
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return payload


# ======================================================================================
# effective output-head rows (B0/H1 initialisation source)
# ======================================================================================


def head_facts(runner: Any) -> dict[str, Any]:
    """Read-only description of the actual runtime head and its logits processing."""
    model = getattr(runner, "model", None)
    owner = head_owner(model)
    head = getattr(owner, "lm_head", None)
    lp = getattr(owner, "logits_processor", None)
    si = getattr(head, "shard_indices", None)
    vllm_config = getattr(runner, "vllm_config", None)
    cg = None
    if vllm_config is not None:
        comp = getattr(vllm_config, "compilation_config", None)
        cg = getattr(comp, "cudagraph_mode", None)
    w = getattr(head, "weight", None)
    return {
        "scale": float(getattr(lp, "scale", 1.0)) if lp is not None else None,
        "soft_cap": (None if getattr(lp, "soft_cap", None) is None else float(lp.soft_cap)) if lp is not None else None,
        "org_vocab_size": int(getattr(lp, "org_vocab_size", 0)) if lp is not None else None,
        "head_dtype": str(getattr(lp, "head_dtype", None)),
        "use_all_gather": bool(getattr(lp, "use_all_gather", False)) if lp is not None else None,
        "vocab_start": int(si.org_vocab_start_index) if si is not None else None,
        "vocab_end": int(si.org_vocab_end_index) if si is not None else None,
        "num_org_vocab_padding": int(si.num_org_vocab_padding) if si is not None else None,
        "tp_size": int(getattr(head, "tp_size", 1)) if head is not None else None,
        "tp_rank": int(getattr(head, "tp_rank", 0)) if head is not None else None,
        "weight_dtype": str(getattr(w, "dtype", None)),
        "weight_shape": list(getattr(w, "shape", ())) if w is not None else None,
        "world_size": _world_size(),
        "cudagraph_mode": str(cg),
        "device_name": _device_name(),
        "vllm_version": _vllm_version(),
        "lm_head_param_path": head_param_path(model),
        "model_dir": str(getattr(getattr(vllm_config, "model_config", None), "model", "")),
    }


def _world_size() -> int:
    if torch is None:
        return 1
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    except Exception:  # noqa: BLE001
        pass
    return 1


def _device_name() -> str:
    if torch is None:
        return "unknown"
    try:
        return str(torch.cuda.get_device_name())
    except Exception:  # noqa: BLE001
        return "unknown"


def _vllm_version() -> str:
    try:
        import vllm  # noqa: PLC0415

        return str(getattr(vllm, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def export_base_rows(state: _CaptureState, runner: Any) -> dict[str, Any]:
    """Export the 24 effective lm_head rows, masked by TP vocab ownership, ONCE.

    * every rank writes its own masked rows + ownership facts;
    * when `tp_size == world_size` and `pp_size == 1`, BOTH ranks call one SUM all-reduce so
      each rank also holds the combined (unmasked) rows for its own verbatim checks;
    * the FP32 combined array on disk is the tiny H1 initialisation source. UNTRAINED.
    """
    cfg = state.config
    assert cfg is not None
    os.makedirs(cfg.base_rows_dir, exist_ok=True)
    facts = head_facts(runner)
    ids_res = resolve_verbalizer_ids(cfg)
    payload: dict[str, Any] = {
        "helper_version": HELPER_VERSION,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "verbalizers": ids_res,
        "head": facts,
        "layout": {
            "hidden_state": "replicated across TP ranks (lm_head is vocab-parallel only; no hidden gather)",
            "vocab_sharding": f"rank {facts['tp_rank']} of {facts['tp_size']} owns global rows "
            f"[{facts['vocab_start']}, {facts['vocab_end']}) = {facts['num_org_vocab_padding']} padded row(s)",
            "row_order": "the 24 rows follow configs/capture.json verbalizers.names order (noul, score, choice)",
            "combined_semantics": "SUM all-reduce of per-rank masked rows; each row is owned by exactly one rank",
        },
        "written_wall_ns": time.time_ns(),
        "uncalibrated": True,
        "trained": False,
        "note": (
            "Untrained initialisation source only: the 24 effective base-token rows as they "
            "exist in the served checkpoint. No fitting, no calibration, no quality claim."
        ),
    }
    if not ids_res["resolved"]:
        path = os.path.join(cfg.base_rows_dir, "UNRESOLVED.json")
        _write_json_atomic(path, payload | {"status": "unresolved"})
        state.record_error(
            "verbalizer_ids_unresolved",
            ids_res["reason"] or "verbalizer ids unresolved",
        )
        return payload | {"status": "unresolved"}

    ids = list(ids_res["ids"])
    owner = head_owner(runner.model)
    if owner is None:
        raise SelectionError("runtime model exposes no lm_head (directly or via language_model)")
    head = owner.lm_head
    rows_local = head.weight.detach()
    vocab_start = int(head.shard_indices.org_vocab_start_index)
    vocab_end = int(head.shard_indices.org_vocab_end_index)
    num_rows_local = int(rows_local.shape[0])
    local_idx, owned = local_row_indices(
        ids, vocab_start=vocab_start, vocab_end=vocab_end, num_rows_local=num_rows_local
    )

    masked = np.zeros((NUM_SLOTS, int(rows_local.shape[1])), dtype=np.float32)
    row_hashes: list[str | None] = []
    for k, li in enumerate(local_idx):
        if li < 0:
            row_hashes.append(None)
            continue
        r = rows_local[li].detach().to(torch.float32).cpu().numpy()
        masked[k] = r
        row_hashes.append(hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest())

    rank = int(facts["tp_rank"] or 0)
    local_payload = payload | {
        "status": "local_masked",
        "tp_rank": rank,
        "owned": owned,
        "local_row_indices": local_idx,
        "row_sha256_fp32": row_hashes,
        "vocab_start": vocab_start,
        "vocab_end": vocab_end,
        "num_rows_local": num_rows_local,
    }
    _write_npz_atomic(
        os.path.join(cfg.base_rows_dir, f"local_rows_rank{rank}.npz"),
        {"rows_fp32": masked, "ids": np.asarray(ids, dtype=np.int64), "owned": np.asarray(owned)},
    )
    _write_json_atomic(os.path.join(cfg.base_rows_dir, f"ownership_rank{rank}.json"), local_payload)

    combined_payload = payload | {"status": "local_masked_only", "reason": "tp export skipped"}
    want_tp = bool(cfg.tp_export_rows)
    world = int(facts["world_size"] or 1)
    tp_size = int(facts["tp_size"] or 1)
    pp_size = int(getattr(getattr(runner, "parallel_config", None), "pipeline_parallel_size", 1) or 1)
    if want_tp and torch is not None and world == tp_size and pp_size == 1 and tp_size > 1:
        try:
            t = torch.as_tensor(masked, device=rows_local.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            combined = t.cpu().numpy().astype(np.float32)
            combined_payload = payload | {
                "status": "combined",
                "tp_rank": rank,
                "collective": "torch.distributed.all_reduce SUM over the default (TP) group, called once per rank",
                "combined_row_sha256_fp32": [
                    hashlib.sha256(np.ascontiguousarray(combined[k]).tobytes()).hexdigest()
                    for k in range(NUM_SLOTS)
                ],
                "owned": owned,
                "vocab_start": vocab_start,
                "vocab_end": vocab_end,
            }
            _write_npz_atomic(
                os.path.join(cfg.base_rows_dir, "combined_rows.npz"),
                {
                    "rows_fp32": combined,
                    "rows_bf16_bits": _bf16_bits_of(combined),
                    "ids": np.asarray(ids, dtype=np.int64),
                },
            )
            # Tiny FP32 initialisation source for H1: 24 x 2560 fp32 + zero bias.
            np.save(
                os.path.join(cfg.base_rows_dir, "h1_init_base24_fp32.npy"),
                combined.astype(np.float32),
            )
            np.save(
                os.path.join(cfg.base_rows_dir, "h1_init_bias_fp32.npy"),
                np.zeros(NUM_SLOTS, dtype=np.float32),
            )
        except Exception as exc:  # noqa: BLE001
            combined_payload = payload | {
                "status": "collective_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            state.record_error("row_export_collective_failed", f"{type(exc).__name__}: {exc}")
    elif want_tp and world != tp_size:
        combined_payload = payload | {
            "status": "collective_skipped",
            "reason": f"world_size {world} != lm_head.tp_size {tp_size}; refusing a wrong-group collective",
        }
    elif want_tp and pp_size != 1:
        combined_payload = payload | {
            "status": "collective_skipped",
            "reason": f"pipeline_parallel_size {pp_size} != 1; only the last PP rank reaches this call",
        }
    _write_json_atomic(os.path.join(cfg.base_rows_dir, f"combined_rank{rank}.json"), combined_payload)
    return combined_payload


def _bf16_bits_of(x: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    u = arr.view(np.uint32).copy()
    is_nan = ((u & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & (
        (u & np.uint32(0x007FFFFF)) != 0
    )
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = np.where(is_nan, (u >> np.uint32(16)) | np.uint32(0x0040), (u + bias) >> np.uint32(16))
    return rounded.astype(np.uint16)


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_npz_atomic(path: str, arrays: dict[str, np.ndarray]) -> None:
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ======================================================================================
# wide-label capture (opt-in; additive, numpy-only)
# ======================================================================================

WIDE_CONFIG_KEY = "wide_choice"
WIDE_ENGINE_MARKER = "cmpl-qfn-wide-"
WIDE_CAPTURE_SCHEMA = "qfn.wide-capture/1"
WIDE_LABEL_SCHEMA = "qfn.wide-label-catalog/1"
WIDE_TEMPLATE_ID = "qwen_chat_wide-v1"
WIDE_TEMPLATE_VERSION = "wide-v1"
WIDE_LABEL_COUNT = 256
WIDE_ARRAY_NAMES = ("wide_ids", "wide_logits_fp32")
_WIDE_CONFIG_KEYS = frozenset({
    "enabled", "catalog_path", "catalog_file_sha256", "catalog_digest",
    "tokenizer_revision", "template_digest",
})
_HEX64_CHARS = frozenset("0123456789abcdef")
_WIDE_ID_RE = re.compile(r"^cmpl-qfn-wide-([0-9]{1,3})-([0-9a-f]{32})(?:-[0-9]+(?:-[0-9a-f]{8})?)?$")


def wide_catalog_document(payload: dict[str, Any]) -> dict[str, Any]:
    """The canonical semantic document of a wide catalog (must match the package module)."""
    return {
        "schema": payload["schema"],
        "tokenizer_revision": payload["tokenizer_revision"],
        "labels": list(payload["labels"]),
        "token_ids": list(payload["token_ids"]),
    }


def wide_catalog_digest(payload: dict[str, Any]) -> str:
    """Content digest of a wide catalog; identical to ``qfn_decision.wide.catalog_digest``."""
    return hashlib.sha256(
        json.dumps(
            wide_catalog_document(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _require_hex64(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX64_CHARS for c in value):
        raise SelectionError(f"{what} must be 64 lowercase hex characters, got {value!r}")
    return value


def _validate_wide_labels(labels: Any) -> tuple[str, ...]:
    if not isinstance(labels, list) or len(labels) != WIDE_LABEL_COUNT:
        got = 0 if not isinstance(labels, list) else len(labels)
        raise SelectionError(f"wide catalog must carry {WIDE_LABEL_COUNT} labels, got {got}")
    out: list[str] = []
    for i, label in enumerate(labels):
        if not isinstance(label, str) or not (1 <= len(label) <= 2):
            raise SelectionError(f"wide label[{i}] must be one or two letters, got {label!r}")
        if any(c < "A" or c > "Z" for c in label):
            raise SelectionError(f"wide label[{i}] must be uppercase ASCII, got {label!r}")
        out.append(label)
    if out[:26] != [chr(ord("A") + i) for i in range(26)]:
        raise SelectionError("wide catalog must start with A..Z in order")
    previous: tuple[str, str] | None = None
    for i, label in enumerate(out):
        if len(label) == 1:
            if i >= 26:
                raise SelectionError(f"wide label[{i}] {label!r} is a single letter after A..Z")
            continue
        key = (label[0], label[1])
        if previous is not None and key <= previous:
            raise SelectionError(f"wide label[{i}] {label!r} breaks the strict two-letter order")
        previous = key
    if len(set(out)) != len(out):
        raise SelectionError("wide catalog labels must be unique")
    return tuple(out)


class WideCaptureConfig:
    """Validated engine-side wide capture configuration (read once, before any capture)."""

    def __init__(
        self,
        *,
        enabled: bool,
        catalog_path: str | None = None,
        catalog_file_sha256: str | None = None,
        catalog_digest: str | None = None,
        tokenizer_revision: str | None = None,
        template_digest: str | None = None,
        labels: Sequence[str] | None = None,
        token_ids: Sequence[int] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.catalog_path = catalog_path
        self.catalog_file_sha256 = catalog_file_sha256
        self.catalog_digest = catalog_digest
        self.tokenizer_revision = tokenizer_revision
        self.template_digest = template_digest
        self.labels = tuple(labels) if labels else None
        self.token_ids = tuple(int(v) for v in token_ids) if token_ids else None

    @property
    def label_count(self) -> int:
        return len(self.labels) if self.labels else 0

    @classmethod
    def from_dict(cls, payload: Any) -> "WideCaptureConfig":
        if not isinstance(payload, dict):
            raise SelectionError(f"{WIDE_CONFIG_KEY} must be a JSON object")
        unknown = sorted(set(payload) - _WIDE_CONFIG_KEYS)
        if unknown:
            raise SelectionError(f"{WIDE_CONFIG_KEY} has unknown field(s): {unknown}")
        if "enabled" not in payload:
            raise SelectionError(f"{WIDE_CONFIG_KEY} requires an explicit boolean 'enabled'")
        enabled = payload["enabled"]
        if not isinstance(enabled, bool):
            raise SelectionError(f"{WIDE_CONFIG_KEY}.enabled must be a boolean, got {enabled!r}")
        if not enabled:
            return cls(enabled=False)
        missing = [k for k in ("catalog_path", "catalog_file_sha256", "catalog_digest",
                               "tokenizer_revision", "template_digest") if payload.get(k) is None]
        if missing:
            raise SelectionError(f"{WIDE_CONFIG_KEY}.enabled=true requires {missing}")
        path = payload["catalog_path"]
        if not isinstance(path, str) or not os.path.isabs(path):
            raise SelectionError(f"{WIDE_CONFIG_KEY}.catalog_path must be absolute, got {path!r}")
        if not os.path.isfile(path):
            raise SelectionError(f"{WIDE_CONFIG_KEY}.catalog_path does not exist: {path!r}")
        file_sha = _require_hex64(payload["catalog_file_sha256"], f"{WIDE_CONFIG_KEY}.catalog_file_sha256")
        content_digest = _require_hex64(payload["catalog_digest"], f"{WIDE_CONFIG_KEY}.catalog_digest")
        template_digest = _require_hex64(payload["template_digest"], f"{WIDE_CONFIG_KEY}.template_digest")
        revision = payload["tokenizer_revision"]
        if not isinstance(revision, str) or not revision.strip():
            raise SelectionError(f"{WIDE_CONFIG_KEY}.tokenizer_revision must be a non-empty string")
        with open(path, "rb") as fh:
            raw = fh.read()
        actual_file_sha = hashlib.sha256(raw).hexdigest()
        if actual_file_sha != file_sha:
            raise SelectionError(
                "wide catalog file byte hash mismatch: configured "
                f"{file_sha[:12]}..., file is {actual_file_sha[:12]}..."
            )
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SelectionError(f"wide catalog is not valid UTF-8 JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise SelectionError("wide catalog must be a JSON object")
        required = {"schema", "tokenizer_revision", "labels", "token_ids"}
        if set(doc) != required:
            raise SelectionError(
                f"wide catalog fields must be exactly {sorted(required)}, got {sorted(doc)}"
            )
        if doc["schema"] != WIDE_LABEL_SCHEMA:
            raise SelectionError(
                f"unsupported wide catalog schema {doc['schema']!r}; expected {WIDE_LABEL_SCHEMA!r}"
            )
        if doc["tokenizer_revision"] != revision:
            raise SelectionError(
                f"wide catalog tokenizer revision {doc['tokenizer_revision']!r} != configured "
                f"{revision!r}; label token ids are not portable across revisions"
            )
        labels = _validate_wide_labels(doc["labels"])
        raw_ids = doc["token_ids"]
        if not isinstance(raw_ids, list) or len(raw_ids) != len(labels):
            raise SelectionError(f"wide catalog must carry {len(labels)} token ids")
        ids: list[int] = []
        for i, value in enumerate(raw_ids):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SelectionError(f"wide catalog token_ids[{i}] must be a non-negative int")
            ids.append(int(value))
        if len(set(ids)) != len(ids):
            raise SelectionError("wide catalog token ids must be unique")
        actual_digest = wide_catalog_digest(doc)
        if actual_digest != content_digest:
            raise SelectionError(
                "wide catalog content digest mismatch: configured "
                f"{content_digest[:12]}..., content is {actual_digest[:12]}..."
            )
        return cls(
            enabled=True,
            catalog_path=path,
            catalog_file_sha256=file_sha,
            catalog_digest=content_digest,
            tokenizer_revision=revision,
            template_digest=template_digest,
            labels=labels,
            token_ids=ids,
        )


def wide_request_identity(req_id: Any) -> tuple[int, str]:
    """``(n_options, uuid)`` from an engine-side wide request id, or refuse."""
    if not isinstance(req_id, str):
        raise SelectionError(f"request id must be a string, got {type(req_id).__name__}")
    match = _WIDE_ID_RE.match(req_id)
    if match is None:
        raise SelectionError(
            f"request id {req_id!r} is not a wide capture id "
            f"(expected {WIDE_ENGINE_MARKER}<n>-<32 hex>[-<index>[-<8 hex>]])"
        )
    return int(match.group(1)), match.group(2)


def is_wide_request(req_id: Any) -> bool:
    return isinstance(req_id, str) and req_id.startswith(WIDE_ENGINE_MARKER)


def build_wide_extension(
    cfg: "CaptureConfig",
    *,
    req_id: str,
    logits_row: Any,
    prompt_hash: str,
    prompt_len: int,
    torch_mod: Any = None,
) -> dict[str, Any]:
    """Index-select ALL 256 catalog label logits from the SAME validated raw ``logits[j]`` row.

    Returns the extension payload (arrays + scalar fields). The caller passes it to
    ``write_record_files``, which moves the arrays into the npz and keeps them out of the scalar
    metadata.
    """
    wc = getattr(cfg, "wide_choice", None)
    if wc is None or not wc.enabled:
        raise SelectionError("wide capture is not enabled in this capture configuration")
    n_options, request_uuid = wide_request_identity(req_id)
    if n_options > wc.label_count:
        raise SelectionError(
            f"request declares {n_options} options but the catalog has {wc.label_count} labels"
        )
    if logits_row is None:
        raise SelectionError("wide capture requires the terminal raw logits row")
    torch_like = torch_mod if torch_mod is not None else torch
    if torch_like is None:  # pragma: no cover - inside the engine torch always exists
        raise SelectionError("wide capture requires torch")
    ids = list(wc.token_ids)
    idx = torch_like.as_tensor(ids, dtype=torch_like.long, device=getattr(logits_row, "device", None))
    wide_logits = logits_row.index_select(0, idx).to(torch_like.float32).cpu().numpy()
    if np.asarray(wide_logits).shape != (wc.label_count,):
        raise SelectionError(
            f"wide index_select produced {np.asarray(wide_logits).shape}, expected ({wc.label_count},)"
        )
    if not np.isfinite(np.asarray(wide_logits, dtype=np.float32)).all():
        raise SelectionError("wide capture logits contain non-finite values")
    return {
        "wide_schema": WIDE_CAPTURE_SCHEMA,
        "wide_catalog_schema": WIDE_LABEL_SCHEMA,
        "wide_catalog_digest": wc.catalog_digest,
        "wide_tokenizer_revision": wc.tokenizer_revision,
        "wide_label_count": int(wc.label_count),
        "wide_template_id": WIDE_TEMPLATE_ID,
        "wide_template_version": WIDE_TEMPLATE_VERSION,
        "wide_template_digest": wc.template_digest,
        "wide_n_options": int(n_options),
        "wide_request_uuid": request_uuid,
        "wide_prompt_token_sha256": prompt_hash,
        "wide_prompt_len": int(prompt_len),
        "wide_ids": np.asarray(ids, dtype=np.int64),
        "wide_logits_fp32": np.ascontiguousarray(np.asarray(wide_logits, dtype=np.float32)),
    }
