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
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
        }


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

    written = write_record_files(cfg.capture_dir, key, helper_meta)
    state.records_written += 1
    state.bytes_written += written
    if state.records_written > cfg.max_records or state.bytes_written > cfg.max_bytes:
        state.stopped = True
        state.record_error(
            "quota_exceeded",
            "capture quota reached; further captures are refused",
            {"max_records": cfg.max_records, "max_bytes": cfg.max_bytes},
        )


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


def write_record_files(capture_dir: str, key: str, meta: dict[str, Any]) -> int:
    """Atomic write of `<key>` (npz with arrays) and `<key>.json` (readable meta)."""
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
    sizes = 0
    buffer = _npz_bytes(arrays, scalar_meta)
    tmp = npz_path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(buffer)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, npz_path)
    sizes += len(buffer)
    json_tmp = json_path + ".part"
    with open(json_tmp, "w", encoding="utf-8") as fh:
        json.dump(scalar_meta, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
    os.replace(json_tmp, json_path)
    sizes += os.path.getsize(json_path)
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
