"""S3 feature transport format: BF16 codecs, record encoding, resumable archive, tolerances.

This module is the *format* half of the S3 feature-capture stage. It is deliberately
torch-free and GPU-free so it runs on the Windows analysis host; the engine-side helper
(`patches/qfn_capture_runtime.py`) implements the same token-hash canonicalisation and the
same record file-name rule, and `tests/test_capture_patch.py` asserts the two agree.

Honest scope: everything here is a container format and pure-numpy arithmetic. No parity,
accuracy or performance claim is made or implied by this module.

Storage contract (one record = one detached decision-boundary state):

    vector_bf16   uint16[2560]   raw BF16 *bits* exactly as the engine wrote them
    vector_fp32   float32[2560]  optional FP32 reference (must equal the BF16 round-trip)
    verbalizer_logits_fp32  float32[24]  RAW runtime logits at the verbalizer rows
    meta_json     uint8[...]     UTF-8 JSON of every scalar field

All integers are exact; hashes are exact. The only approximate numbers in a record are the
BF16 vectors and logits, which is why the tolerance classes in
`configs/numeric_tolerances.json` are split into EXACT metadata vs BF16/FP32 numeric classes.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import re
import time
import uuid
from typing import Any, Iterable, Sequence

import numpy as np

FEATURES_SCHEMA = "qfn.features/1"
ARCHIVE_SCHEMA = "qfn.features-archive/1"
TOKEN_HASH_SCHEMA = "qfn.tokens/1"
HIDDEN_WIDTH = 2560
NUM_SLOTS = 24
BF16_MANTISSA_BITS = 7  # bf16 = 1 sign + 8 exponent + 7 explicit mantissa bits

# Cache-attribution vocabulary (PM S3 correction). `cache_hit_tokens` is the number of prompt
# tokens served from the prefix cache, which the runner's sample() hook CANNOT observe
# (`prefill_stats.num_local_cached_tokens` lives on the scheduler-side Request). It is therefore
# nullable, and a value is only representable with explicit provenance.
CACHE_PROVENANCE_UNKNOWN = "unknown"
CACHE_PROVENANCE_MEASURED = "measured"
CACHE_PROVENANCE_LEGACY = "legacy_single_chunk_known"
CACHE_PROVENANCE_VALUES = (
    CACHE_PROVENANCE_UNKNOWN,
    CACHE_PROVENANCE_MEASURED,
    CACHE_PROVENANCE_LEGACY,
)
CACHE_UNKNOWN_REASON = (
    "runner sample() cannot observe the scheduler's local prefix-cache hit count "
    "(prefill_stats lives on the scheduler-side Request); the pre-step computed token count is "
    "recorded separately as computed_tokens_before_step and is NOT a cache-hit metric"
)
LEGACY_CACHE_SOURCE = "legacy_single_chunk_explicit_value"

_FIXTURE_MODE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "qfn_fixture_records_allowed", default=False
)


class FeatureError(ValueError):
    """Raised for any malformed/unsafe feature record or archive operation."""


@contextlib.contextmanager
def fixture_records_allowed():
    """TEST/FIXTURE ONLY: allow deliberately-invalid records (e.g. a nonterminal position).

    The format gate is fail-closed by default. Fixture builders that must *represent* an invalid
    record (to prove a downstream validator rejects it) opt in explicitly and only for the scope
    of the block. The engine writer and the client ingest path never use this.
    """
    token = _FIXTURE_MODE.set(True)
    try:
        yield
    finally:
        _FIXTURE_MODE.reset(token)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _require_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise FeatureError(f"{name} must be an integer (got {type(value).__name__})")
    iv = int(value)
    if minimum is not None and iv < minimum:
        raise FeatureError(f"{name} must be >= {minimum}, got {iv}")
    return iv


def _require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise FeatureError(f"{name} must be a real number (got {type(value).__name__})")
    fv = float(value)
    if not np.isfinite(fv):
        raise FeatureError(f"{name} must be finite, got {fv!r}")
    return fv


@dataclasses.dataclass
class ArchiveEntry:
    """Result of writing one record into an archive."""

    file_name: str
    sha256: str
    example_id: str
    engine_request_id: str
    prompt_token_sha256: str
    prompt_tokens_count: int
    dup_disambiguation: int
    received_wall_ns: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# BF16 codecs
# --------------------------------------------------------------------------------------


def fp32_to_bf16_bits(x: np.ndarray | Sequence[float]) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16, returned as uint16 bits (numpy has no bf16)."""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    u = arr.view(np.uint32).copy()
    # NaN must stay NaN: skip the rounding bias for exponent=all-ones with a non-zero mantissa.
    is_nan = ((u & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & (
        (u & np.uint32(0x007FFFFF)) != 0
    )
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = (u + bias) >> np.uint32(16)
    rounded = np.where(is_nan, (u >> np.uint32(16)) | np.uint32(0x0040), rounded)
    return rounded.astype(np.uint16)


def bf16_bits_to_fp32(bits: np.ndarray | Sequence[int]) -> np.ndarray:
    """Exact BF16 bit pattern -> FP32 (no rounding: BF16 is a subset of FP32)."""
    arr = np.asarray(bits, dtype=np.uint16)
    return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16_roundtrip_fp32(x: np.ndarray | Sequence[float]) -> np.ndarray:
    return bf16_bits_to_fp32(fp32_to_bf16_bits(x))


def bf16_ulp(x: float) -> float:
    """ULP of a normal BF16 value: 2**(floor(log2|x|) - 7). 0.0/NaN -> 0.0.

    BF16 keeps 8 significant bits (1 implicit + 7 explicit mantissa bits), so consecutive
    representable values inside [2**e, 2**(e+1)) are spaced 2**(e-7) apart.
    """
    ax = abs(float(x))
    if ax == 0.0 or not np.isfinite(ax):
        return 0.0
    e = int(np.floor(np.log2(ax)))
    return float(2.0 ** (e - BF16_MANTISSA_BITS))


def bf16_gemm_allowance(a: float, b: float) -> float:
    """Registered allowance for a BF16 GEMM result vs the runtime value.

    Two BF16 ULPs of max(|a|, |b|) plus a 1e-4 floor. The two ULPs bound the rounding of
    the operands/accumulation that enter and leave the head; the floor keeps the bound
    meaningful for tiny operands, where a pure ULP bound collapses to ~0 and would be an
    unreachably tight tolerance rather than a physically motivated one.
    """
    scale = max(abs(float(a)), abs(float(b)))
    return 2.0 * bf16_ulp(scale) + 1e-4


def bf16_representable(x: np.ndarray | Sequence[float]) -> bool:
    """True when every element is finite and exactly representable in BF16.

    Live finding 2026-10-10: the installed head emits BF16 VALUES, so the RAW runtime logits
    satisfy this predicate. A recorded runtime vector that does not is not a BF16 runtime
    target and must be refused rather than compared.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        return False
    return bool(np.array_equal(bf16_roundtrip_fp32(arr), arr))


def runtime_values_representable(x: np.ndarray | Sequence[float], dtype: str) -> bool:
    """Fail-closed dtype check for a RAW runtime logit vector. Unknown dtype -> FeatureError."""
    kind = str(dtype).strip().lower()
    if kind == "bfloat16":
        return bf16_representable(x)
    if kind == "float32":
        return bool(np.all(np.isfinite(np.asarray(x, dtype=np.float32))))
    raise FeatureError(f"unknown runtime dtype {dtype!r}")


# --------------------------------------------------------------------------------------
# canonical token hashing (must match the engine helper bit-for-bit)
# --------------------------------------------------------------------------------------


def _validate_token_ids(ids: Iterable[int]) -> list[int]:
    out: list[int] = []
    for v in ids:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise FeatureError(f"token id must be an int, got {v!r}")
        iv = int(v)
        if iv < 0 or iv > np.iinfo(np.int32).max:
            raise FeatureError(f"token id out of range: {iv}")
        out.append(iv)
    return out


def token_sequence_sha256(token_ids: Iterable[int]) -> str:
    """Canonical hash of an exact token-ID sequence: sha256(b"qfn.tokens/1:<n>:" + LE int32)."""
    ids = _validate_token_ids(token_ids)
    payload = (
        TOKEN_HASH_SCHEMA.encode("ascii")
        + b":"
        + str(len(ids)).encode("ascii")
        + b":"
        + np.asarray(ids, dtype="<i4").tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def args_fingerprint(obj: Any) -> str:
    """Stable sha256 of a JSON-serialisable object (used for run/args provenance)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class FeatureRecord:
    """One detached decision-boundary capture, with its runtime logits and provenance."""

    example_id: str
    engine_request_id: str
    prompt_token_sha256: str
    prompt_tokens_count: int
    prompt_len: int
    terminal_position: int
    # PM S3 correction: `cache_hit_tokens` is the ACTUAL prefix-cache reuse count, which the
    # runner hook cannot observe -> nullable, with explicit provenance. It is NOT the pre-step
    # computed count (that is `computed_tokens_before_step`, below).
    cache_hit_tokens: int | None
    num_scheduled_tokens: int
    query_start: int
    query_end: int
    logits_row_index: int
    batch_index: int
    internal_index: int
    vector_bf16: np.ndarray
    vector_fp32: np.ndarray | None
    verbalizer_ids: list[int]
    verbalizer_logits_fp32: np.ndarray
    runtime_argmax_token_id: int
    runtime_max_logit_fp32: float
    head_scale: float
    head_soft_cap: float | None
    head_org_vocab_size: int
    head_dtype: str
    lm_head_tp_size: int
    tp_rank: int
    world_size: int
    cudagraph_mode: str
    kernel_mode: str
    runner_source_sha256: str
    capture_wall_ns: int
    logits_captured: bool
    logits_absent_reason: str | None
    vector_source: str
    dup_disambiguation: int = 0
    capture_monotonic_ns: int = 0
    helper_version: str = ""
    device_name: str = ""
    token_hash_scope: str = "prompt"
    chunk_token_sha256: str | None = None
    engine_build_version: str = ""
    graph_capturing: bool = False
    token_hash_from_engine_state: bool = True
    runtime_logits_row_fp32: np.ndarray | None = None
    # --- PM S3 correction: authoritative accounting + honest cache provenance -------------
    # Pre-step computed token count from the runner (includes earlier chunks AND reusable
    # prefix). Required on new engine records; optional for legacy/fixture records.
    computed_tokens_before_step: int | None = None
    cache_hit_provenance: str = CACHE_PROVENANCE_UNKNOWN
    cache_hit_source: str | None = None
    cache_hit_status_reason: str | None = None

    def __post_init__(self) -> None:
        self.vector_bf16 = np.ascontiguousarray(np.asarray(self.vector_bf16))
        if self.vector_bf16.dtype != np.uint16:
            raise FeatureError(
                f"vector_bf16 must be uint16 BF16 bits, got {self.vector_bf16.dtype}"
            )
        if self.vector_bf16.shape != (HIDDEN_WIDTH,):
            raise FeatureError(f"vector_bf16 must have shape ({HIDDEN_WIDTH},)")

        if self.vector_fp32 is not None:
            self.vector_fp32 = np.ascontiguousarray(np.asarray(self.vector_fp32, dtype=np.float32))
            if self.vector_fp32.shape != (HIDDEN_WIDTH,):
                raise FeatureError(f"vector_fp32 must have shape ({HIDDEN_WIDTH},)")
            rt = bf16_bits_to_fp32(self.vector_bf16)
            if not np.array_equal(rt, self.vector_fp32):
                n_bad = int(np.count_nonzero(rt != self.vector_fp32))
                raise FeatureError(
                    "vector_fp32 is inconsistent with vector_bf16 "
                    f"({n_bad} of {HIDDEN_WIDTH} elements differ)"
                )

        self.verbalizer_logits_fp32 = np.ascontiguousarray(
            np.asarray(self.verbalizer_logits_fp32, dtype=np.float32)
        )
        if self.verbalizer_logits_fp32.shape != (NUM_SLOTS,):
            raise FeatureError(f"verbalizer_logits_fp32 must have shape ({NUM_SLOTS},)")
        self.verbalizer_ids = [int(v) for v in self.verbalizer_ids]
        if len(self.verbalizer_ids) != NUM_SLOTS:
            raise FeatureError(f"verbalizer_ids must have {NUM_SLOTS} entries")
        if len(set(self.verbalizer_ids)) != NUM_SLOTS:
            raise FeatureError("verbalizer_ids must be unique")

        if self.runtime_logits_row_fp32 is not None:
            self.runtime_logits_row_fp32 = np.ascontiguousarray(
                np.asarray(self.runtime_logits_row_fp32, dtype=np.float32)
            )

        if not isinstance(self.engine_request_id, str) or not self.engine_request_id:
            raise FeatureError("engine_request_id must be a non-empty string")
        reject_unsafe_path_token(self.engine_request_id, what="engine_request_id")
        if not isinstance(self.example_id, str) or not self.example_id:
            raise FeatureError("example_id must be a non-empty string")
        reject_unsafe_path_token(self.example_id, what="example_id")

        if not isinstance(self.prompt_token_sha256, str) or len(self.prompt_token_sha256) != 64:
            raise FeatureError("prompt_token_sha256 must be a 64-char hex digest")
        try:
            int(self.prompt_token_sha256, 16)
        except ValueError as exc:
            raise FeatureError("prompt_token_sha256 must be hex") from exc

        # --- numeric/accounting integrity (PM correction D): no silent coercion --------------
        self.prompt_tokens_count = _require_int("prompt_tokens_count", self.prompt_tokens_count, minimum=1)
        self.prompt_len = _require_int("prompt_len", self.prompt_len, minimum=1)
        if self.prompt_tokens_count != self.prompt_len:
            raise FeatureError(
                "prompt_tokens_count must equal prompt_len for a single-prompt capture "
                f"({self.prompt_tokens_count} != {self.prompt_len})"
            )
        self.terminal_position = _require_int("terminal_position", self.terminal_position, minimum=0)
        self.num_scheduled_tokens = _require_int(
            "num_scheduled_tokens", self.num_scheduled_tokens, minimum=1
        )
        self.query_start = _require_int("query_start", self.query_start, minimum=0)
        self.query_end = _require_int("query_end", self.query_end, minimum=1)
        self.logits_row_index = _require_int("logits_row_index", self.logits_row_index, minimum=0)
        self.batch_index = _require_int("batch_index", self.batch_index, minimum=0)
        self.internal_index = _require_int("internal_index", self.internal_index, minimum=0)
        self.tp_rank = _require_int("tp_rank", self.tp_rank, minimum=0)
        self.world_size = _require_int("world_size", self.world_size, minimum=1)
        self.lm_head_tp_size = _require_int("lm_head_tp_size", self.lm_head_tp_size, minimum=1)
        self.head_org_vocab_size = _require_int("head_org_vocab_size", self.head_org_vocab_size, minimum=1)
        self.capture_wall_ns = _require_int("capture_wall_ns", self.capture_wall_ns, minimum=0)
        self.capture_monotonic_ns = _require_int("capture_monotonic_ns", self.capture_monotonic_ns, minimum=0)
        self.dup_disambiguation = _require_int("dup_disambiguation", self.dup_disambiguation, minimum=0)
        self.runtime_argmax_token_id = _require_int(
            "runtime_argmax_token_id",
            self.runtime_argmax_token_id,
            # -1 is the documented "no logits captured" sentinel; a captured record must be >= 0
            minimum=0 if self.logits_captured else -1,
        )

        # --- the decision boundary is the LAST prompt token (PM correction 1) ---------------
        if self.terminal_position != self.prompt_len - 1 and not _FIXTURE_MODE.get():
            raise FeatureError(
                f"terminal_position {self.terminal_position} is not the last prompt token "
                f"(prompt_len={self.prompt_len}); the decision boundary is the final prompt token"
            )
        if self.query_end <= self.query_start:
            raise FeatureError("query_end must be greater than query_start")
        if self.query_end - self.query_start != self.num_scheduled_tokens:
            raise FeatureError(
                f"query length {self.query_end - self.query_start} must equal num_scheduled_tokens "
                f"{self.num_scheduled_tokens}"
            )

        # --- authoritative pre-step accounting + honest cache attribution (PM correction B) --
        if self.computed_tokens_before_step is not None:
            self.computed_tokens_before_step = _require_int(
                "computed_tokens_before_step", self.computed_tokens_before_step, minimum=0
            )
            if self.computed_tokens_before_step + self.num_scheduled_tokens != self.prompt_len:
                raise FeatureError(
                    "computed_tokens_before_step + num_scheduled_tokens must equal prompt_len at "
                    f"the terminal step ({self.computed_tokens_before_step} + "
                    f"{self.num_scheduled_tokens} != {self.prompt_len})"
                )
        self._validate_cache_attribution()

        # --- finiteness (PM correction D): never coerce an invalid number into a valid one ---
        # (the fixture hatch relaxes these gates only so a fixture can REPRESENT an invalid record;
        # decode outside the hatch always refuses it)
        if not _FIXTURE_MODE.get():
            if not np.isfinite(bf16_bits_to_fp32(self.vector_bf16)).all():
                raise FeatureError("vector_bf16 decodes to non-finite values")
            if self.vector_fp32 is not None and not np.isfinite(self.vector_fp32).all():
                raise FeatureError("vector_fp32 contains non-finite values")
            if self.logits_captured and not np.isfinite(self.verbalizer_logits_fp32).all():
                raise FeatureError(
                    "verbalizer_logits_fp32 contains non-finite values on a captured record"
                )
            if self.runtime_logits_row_fp32 is not None and not np.isfinite(
                self.runtime_logits_row_fp32
            ).all():
                raise FeatureError("runtime_logits_row_fp32 contains non-finite values")
            self.head_scale = _require_finite("head_scale", self.head_scale)
            if self.head_scale <= 0:
                raise FeatureError(f"head_scale must be > 0, got {self.head_scale}")
            if self.head_soft_cap is not None:
                self.head_soft_cap = _require_finite("head_soft_cap", self.head_soft_cap)
                if self.head_soft_cap <= 0:
                    raise FeatureError(f"head_soft_cap must be > 0, got {self.head_soft_cap}")
            self.runtime_max_logit_fp32 = _require_finite(
                "runtime_max_logit_fp32", self.runtime_max_logit_fp32
            )

        if self.tp_rank >= self.world_size:
            raise FeatureError("tp_rank must be < world_size")
        if self.logits_captured and self.logits_absent_reason:
            raise FeatureError("logits_captured=True cannot carry an absent reason")
        if not self.logits_captured and not self.logits_absent_reason:
            raise FeatureError("logits_captured=False requires logits_absent_reason")

    # -- cache attribution ---------------------------------------------------------------
    def _validate_cache_attribution(self) -> None:
        """`cache_hit_tokens` is either null with a reason, or a value with explicit provenance.

        A numeric value with an authoritative `computed_tokens_before_step` present MUST state
        provenance: pre-step computed tokens are never relabelled as cache hits.
        """
        if self.cache_hit_tokens is None:
            if self.cache_hit_provenance != CACHE_PROVENANCE_UNKNOWN:
                raise FeatureError(
                    f"a null cache_hit_tokens requires provenance {CACHE_PROVENANCE_UNKNOWN!r}, "
                    f"got {self.cache_hit_provenance!r}"
                )
            if not (isinstance(self.cache_hit_status_reason, str) and self.cache_hit_status_reason.strip()):
                raise FeatureError(
                    "a null cache_hit_tokens requires a non-empty cache_hit_status_reason "
                    "(why the prefix-cache hit count is unavailable)"
                )
            return

        self.cache_hit_tokens = _require_int("cache_hit_tokens", self.cache_hit_tokens, minimum=0)
        # Physical bound (PM final correction). The installed source recomputes the terminal
        # prompt token for every request, so a prompt of length L can never have had L or more
        # tokens served from the prefix cache. This holds for legacy and measured attribution
        # alike, and is NOT relaxed by the fixture hatch: an impossible reuse count is never a
        # legitimate fixture.
        if self.cache_hit_tokens >= self.prompt_len:
            raise FeatureError(
                f"cache_hit_tokens {self.cache_hit_tokens} is impossible for prompt_len "
                f"{self.prompt_len}: the terminal prompt token is always recomputed, so at most "
                f"{self.prompt_len - 1} prompt token(s) can come from the prefix cache"
            )
        prov = self.cache_hit_provenance
        if prov == CACHE_PROVENANCE_MEASURED:
            if not (isinstance(self.cache_hit_source, str) and self.cache_hit_source.strip()):
                raise FeatureError(
                    "cache_hit_provenance 'measured' requires a non-empty cache_hit_source"
                )
        elif prov in (CACHE_PROVENANCE_UNKNOWN, CACHE_PROVENANCE_LEGACY):
            if self.computed_tokens_before_step is not None:
                raise FeatureError(
                    f"a numeric cache_hit_tokens with an authoritative computed_tokens_before_step "
                    f"must state provenance ({CACHE_PROVENANCE_MEASURED!r} or "
                    f"{CACHE_PROVENANCE_LEGACY!r}); cache_hit_tokens must not be the pre-step "
                    "computed count relabelled"
                )
            if self.num_scheduled_tokens != self.prompt_len:
                raise FeatureError(
                    "a numeric cache_hit_tokens without an authoritative pre-step count is only "
                    "valid for the documented single-chunk legacy path"
                )
            self.cache_hit_provenance = CACHE_PROVENANCE_LEGACY
            self.cache_hit_source = self.cache_hit_source or LEGACY_CACHE_SOURCE
        else:
            raise FeatureError(
                f"unknown cache_hit_provenance {prov!r}; expected one of {CACHE_PROVENANCE_VALUES}"
            )
        if (
            self.computed_tokens_before_step is not None
            and self.cache_hit_tokens > self.computed_tokens_before_step
        ):
            raise FeatureError(
                "cache_hit_tokens cannot exceed computed_tokens_before_step "
                f"({self.cache_hit_tokens} > {self.computed_tokens_before_step})"
            )

    # -- derived ---------------------------------------------------------------------
    @property
    def vector_fp32_matches_bf16(self) -> bool | None:
        if self.vector_fp32 is None:
            return None
        return bool(np.array_equal(bf16_bits_to_fp32(self.vector_bf16), self.vector_fp32))

    def decode_vector_fp32(self) -> np.ndarray:
        return bf16_bits_to_fp32(self.vector_bf16)

    def meta(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name in (
                "vector_bf16",
                "vector_fp32",
                "runtime_logits_row_fp32",
                "verbalizer_logits_fp32",
                "verbalizer_ids",
            ):
                continue
            out[f.name] = getattr(self, f.name)
        return out

    def file_stem(self) -> str:
        return (
            f"{_safe_name(self.engine_request_id)}"
            f"__{self.prompt_token_sha256[:16]}__{self.dup_disambiguation}"
        )


def reject_unsafe_path_token(name: str, *, what: str) -> str:
    """Fail-closed guard for ONE opaque id or ONE on-disk basename component.

    Live finding 2026-10-10: the legitimate row id
    ``tic_tac_toe-v1:.OOX..X.X:OO...XX.X`` was refused by a guard that rejected any ``..``
    substring. That guard conflated an OPAQUE request/example id (never used as a path, no
    normalisation applied) with an on-disk basename. Interior dots, including ``..`` inside a
    token, are therefore accepted here.

    Rejected (path hazards only): a non-string/empty token, NUL, ``/`` or ``\\`` (separators),
    the exact tokens ``.`` and ``..``, a leading/trailing ``..`` (a path-boundary pattern), a
    drive prefix (``C:``) and a UNC/absolute prefix. Because separators are refused, a token
    can only ever be a single path component, so an interior ``..`` cannot escape anywhere.
    The caller must still map the token to a file name with the deterministic safe mapping
    (``_safe_name``) and must bind responses/records by hash rather than by the file name.
    """
    if not isinstance(name, str) or not name:
        raise FeatureError(f"{what} must be a non-empty string")
    if "\x00" in name:
        raise FeatureError(f"{what} contains a NUL byte: {name!r}")
    if "/" in name or "\\" in name:
        raise FeatureError(f"{what} contains a path separator: {name!r}")
    if name in (".", ".."):
        raise FeatureError(f"{what} must not be a path component such as '.' or '..': {name!r}")
    if name.startswith("..") or name.endswith(".."):
        raise FeatureError(f"{what} must not begin or end with a traversal sequence: {name!r}")
    if re.match(r"^[A-Za-z]:", name):
        raise FeatureError(f"{what} must not be a drive-relative or absolute path: {name!r}")
    return name


def _reject_unsafe_name(name: str, *, what: str) -> None:
    """Back-compat alias for :func:`reject_unsafe_path_token` (fail-closed, hazard-only)."""
    reject_unsafe_path_token(name, what=what)


def _safe_name(name: str) -> str:
    """Deterministic, single-component file name derived from an id (no normalisation of ids)."""
    _reject_unsafe_name(name, what="record key")
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)[:160]


# --------------------------------------------------------------------------------------
# npz encoding
# --------------------------------------------------------------------------------------


def encode_record_npz(rec: FeatureRecord) -> bytes:
    buf = io.BytesIO()
    meta_bytes = np.frombuffer(
        json.dumps(rec.meta(), sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )
    arrays: dict[str, np.ndarray] = {
        "meta_json": meta_bytes,
        "vector_bf16": np.ascontiguousarray(rec.vector_bf16),
        "verbalizer_ids": np.asarray(rec.verbalizer_ids, dtype=np.int64),
        "verbalizer_logits_fp32": rec.verbalizer_logits_fp32,
    }
    if rec.vector_fp32 is not None:
        arrays["vector_fp32"] = rec.vector_fp32
    if rec.runtime_logits_row_fp32 is not None:
        arrays["runtime_logits_row_fp32"] = rec.runtime_logits_row_fp32
    np.savez(buf, **arrays)
    return buf.getvalue()


def decode_record_npz(blob: bytes) -> FeatureRecord:
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            names = set(z.files)
            for required in ("meta_json", "vector_bf16", "verbalizer_ids", "verbalizer_logits_fp32"):
                if required not in names:
                    raise FeatureError(f"record is missing array {required!r}")
            meta_raw = bytes(np.asarray(z["meta_json"], dtype=np.uint8))
            meta = json.loads(meta_raw.decode("utf-8"))
            fields = {f.name for f in dataclasses.fields(FeatureRecord)}
            unknown = set(meta) - fields
            if unknown:
                raise FeatureError(f"record meta has unknown fields: {sorted(unknown)}")
            kwargs: dict[str, Any] = dict(meta)
            kwargs["vector_bf16"] = np.asarray(z["vector_bf16"])
            kwargs["verbalizer_ids"] = [int(v) for v in np.asarray(z["verbalizer_ids"])]
            kwargs["verbalizer_logits_fp32"] = np.asarray(z["verbalizer_logits_fp32"])
            kwargs["vector_fp32"] = np.asarray(z["vector_fp32"]) if "vector_fp32" in names else None
            kwargs["runtime_logits_row_fp32"] = (
                np.asarray(z["runtime_logits_row_fp32"])
                if "runtime_logits_row_fp32" in names
                else None
            )
        return FeatureRecord(**kwargs)
    except FeatureError:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode problem is a corrupt record
        raise FeatureError(f"corrupt record: {type(exc).__name__}: {exc}") from exc


def record_sha256(rec: FeatureRecord) -> str:
    return hashlib.sha256(encode_record_npz(rec)).hexdigest()


def _rewrite_meta_for_tests(
    blob: bytes, overrides: dict[str, Any], drop: Sequence[str] = ()
) -> bytes:
    """TEST/FIXTURE ONLY: rewrite the meta JSON inside an encoded record.

    Bypasses the strict constructor so negative fixtures can *represent* an invalid record
    (e.g. a nonterminal position) and prove that the decode gate refuses it. Never used by the
    engine writer or the client ingest path.
    """
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    meta_bytes = bytes(np.asarray(arrays.pop("meta_json"), dtype=np.uint8))
    meta = json.loads(meta_bytes.decode("utf-8"))
    for key in drop:
        meta.pop(key, None)
    meta.update(overrides)
    buf = io.BytesIO()
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"), dtype=np.uint8
    )
    np.savez(buf, **arrays)
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# cache attribution reporting (never derive efficiency from unknown values)
# --------------------------------------------------------------------------------------


def cache_attribution(records: Iterable[FeatureRecord]) -> dict[str, Any]:
    """Summarise what is actually known about prefix-cache reuse.

    PM S3 correction: the runner hook cannot observe the scheduler's prefix-cache hit count, so
    records normally carry `cache_hit_tokens = None`. This function therefore reports COUNTS and
    PROVENANCE only. It deliberately computes no cache-hit rate, reuse ratio or prefill saving:
    those cannot be derived from unknown values.
    """
    prov_counts: dict[str, int] = {}
    unknown = 0
    known = 0
    sources: set[str] = set()
    for rec in records:
        prov = str(getattr(rec, "cache_hit_provenance", CACHE_PROVENANCE_UNKNOWN))
        prov_counts[prov] = prov_counts.get(prov, 0) + 1
        if getattr(rec, "cache_hit_tokens", None) is None:
            unknown += 1
        else:
            known += 1
            src = getattr(rec, "cache_hit_source", None)
            if isinstance(src, str) and src and src != LEGACY_CACHE_SOURCE:
                sources.add(src)
    status = "unknown" if known == 0 else ("known" if unknown == 0 else "mixed")
    return {
        "schema": "qfn.cache-attribution/1",
        "status": status,
        "unknown_count": unknown,
        "known_count": known,
        "provenance_counts": dict(sorted(prov_counts.items())),
        "measured_sources": sorted(sources),
        "efficiency_metrics_available": False,
        "reason": (
            "the runner sample() hook cannot observe the scheduler's local prefix-cache hit "
            "count, so no cache-hit rate, reuse ratio or prefill saving is derived; "
            "computed_tokens_before_step is a computed-token count, not a reuse metric"
        ),
        "live_gate": "LIVE MEASUREMENT GATE: actual prefix-cache reuse attribution is unmeasured",
    }


# --------------------------------------------------------------------------------------
# archive
# --------------------------------------------------------------------------------------


class FeatureArchive:
    """A resumable, checksummed directory of feature records.

    Layout::

        <root>/
          records/<engine_request_id>__<prompt_hash16>__<dup>.npz
          index.jsonl        one JSON line per record (example_id -> engine request id)
          CHECKSUMS.sha256   sha256 of every record file (rewritten atomically)
          errors.jsonl       append-only client/engine problem log
          manifest.json      last written aggregate manifest
    """

    def __init__(self, root: os.PathLike[str] | str, run_id: str) -> None:
        self.root = pathlib.Path(root)
        self.run_id = run_id
        self.records_dir = self.root / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------------------
    def write(self, rec: FeatureRecord) -> ArchiveEntry:
        blob = encode_record_npz(rec)
        sha = hashlib.sha256(blob).hexdigest()
        file_name = rec.file_stem() + ".npz"
        _reject_unsafe_name(file_name, what="record file name")
        final = self.records_dir / file_name
        tmp = self.records_dir / (file_name + ".part")
        try:
            with open(tmp, "wb") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)
        except BaseException:
            if tmp.exists():
                tmp.unlink()
            raise
        entry = ArchiveEntry(
            file_name=file_name,
            sha256=sha,
            example_id=rec.example_id,
            engine_request_id=rec.engine_request_id,
            prompt_token_sha256=rec.prompt_token_sha256,
            prompt_tokens_count=rec.prompt_tokens_count,
            dup_disambiguation=rec.dup_disambiguation,
            received_wall_ns=time.time_ns(),
        )
        self._append_line(self.root / "index.jsonl", entry.as_dict())
        self._rewrite_checksums()
        return entry

    def log_error(self, payload: dict[str, Any]) -> None:
        self._append_line(self.root / "errors.jsonl", {"wall_ns": time.time_ns(), **payload})

    def _append_line(self, path: pathlib.Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def _rewrite_checksums(self) -> None:
        lines = []
        for p in sorted(self.records_dir.glob("*.npz")):
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
        tmp = self.root / "CHECKSUMS.sha256.part"
        tmp.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
        )
        os.replace(tmp, self.root / "CHECKSUMS.sha256")

    # -- reading ---------------------------------------------------------------------
    def index(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        path = self.root / "index.jsonl"
        if not path.is_file():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            out.setdefault(entry["example_id"], entry)
        return out

    def checksums(self) -> dict[str, str]:
        path = self.root / "CHECKSUMS.sha256"
        out: dict[str, str] = {}
        if not path.is_file():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split("  ", 1)
            out[name] = digest
        return out

    def record_files(self) -> list[pathlib.Path]:
        return sorted(self.records_dir.glob("*.npz"))

    def read_record(self, path: os.PathLike[str] | str) -> FeatureRecord:
        p = pathlib.Path(path)
        return decode_record_npz(p.read_bytes())

    def resume_scan(self) -> dict[str, Any]:
        """Rebuild the archive state from disk: what is complete, what is broken."""
        completed: dict[str, list[dict[str, Any]]] = {}
        corrupt: list[dict[str, Any]] = []
        index = self.index()
        sums = self.checksums()
        for p in self.record_files():
            try:
                blob = p.read_bytes()
                if hashlib.sha256(blob).hexdigest() != sums.get(p.name):
                    raise FeatureError("checksum mismatch against CHECKSUMS.sha256")
                rec = decode_record_npz(blob)
            except Exception as exc:  # noqa: BLE001 - any failure means corrupt
                corrupt.append({"file_name": p.name, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            completed.setdefault(rec.example_id, []).append(
                {
                    "file_name": p.name,
                    "example_id": rec.example_id,
                    "engine_request_id": rec.engine_request_id,
                    "prompt_token_sha256": rec.prompt_token_sha256,
                    "logits_captured": rec.logits_captured,
                    "index_entry_file": (index.get(rec.example_id) or {}).get("file_name"),
                }
            )
        dup_ids = sorted(k for k, v in completed.items() if len(v) > 1)
        return {
            "run_id": self.run_id,
            "completed_example_ids": sorted(completed),
            "records": {k: v[0] for k, v in completed.items()},
            "duplicate_example_ids": dup_ids,
            "duplicate_records": sum(len(v) - 1 for v in completed.values()),
            "corrupt": corrupt,
            "missing": [],
        }

    def verify(self) -> dict[str, Any]:
        sums = self.checksums()
        mismatched: list[str] = []
        missing: list[str] = []
        for p in self.record_files():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if sums.get(p.name) != digest:
                mismatched.append(p.name)
        for name in sums:
            if not (self.records_dir / name).is_file():
                missing.append(name)
        return {
            "ok": not mismatched and not missing,
            "mismatched": sorted(mismatched),
            "missing": sorted(missing),
            "record_count": len(self.record_files()),
        }

    def manifest(self) -> dict[str, Any]:
        scan = self.resume_scan()
        records = []
        decoded: list[FeatureRecord] = []
        hashes: set[str] = set()
        logits_captured = 0
        runner_sha = ""
        helper_version = ""
        for p in self.record_files():
            blob = p.read_bytes()
            try:
                rec = decode_record_npz(blob)
            except FeatureError:
                records.append(
                    {
                        "file_name": p.name,
                        "sha256": hashlib.sha256(blob).hexdigest(),
                        "corrupt": True,
                    }
                )
                continue
            decoded.append(rec)
            hashes.add(rec.prompt_token_sha256)
            logits_captured += int(rec.logits_captured)
            runner_sha = runner_sha or rec.runner_source_sha256
            helper_version = helper_version or rec.helper_version
            records.append(
                {
                    "file_name": p.name,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "example_id": rec.example_id,
                    "engine_request_id": rec.engine_request_id,
                    "prompt_token_sha256": rec.prompt_token_sha256,
                    "prompt_tokens_count": rec.prompt_tokens_count,
                    "logits_captured": rec.logits_captured,
                    "computed_tokens_before_step": rec.computed_tokens_before_step,
                    "cache_hit_tokens": rec.cache_hit_tokens,
                    "cache_hit_provenance": rec.cache_hit_provenance,
                }
            )
        return {
            "schema": ARCHIVE_SCHEMA,
            "features_schema": FEATURES_SCHEMA,
            "run_id": self.run_id,
            "generated_wall_ns": time.time_ns(),
            "record_count": len(self.record_files()),
            "example_count": len(scan["completed_example_ids"]),
            "logits_captured_count": logits_captured,
            "prompt_token_sha256_unique": len(hashes),
            "duplicate_example_ids": scan["duplicate_example_ids"],
            "corrupt": scan["corrupt"],
            "cache_attribution": cache_attribution(decoded),
            "engine_build": {
                "runner_source_sha256": runner_sha,
                "helper_version": helper_version,
                "archive_schema": ARCHIVE_SCHEMA,
            },
            "records": records,
        }

    def write_manifest(self) -> dict[str, Any]:
        man = self.manifest()
        tmp = self.root / "manifest.json.part"
        tmp.write_text(
            json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        os.replace(tmp, self.root / "manifest.json")
        return man


# --------------------------------------------------------------------------------------
# tolerances
# --------------------------------------------------------------------------------------


def load_tolerances(path: os.PathLike[str] | str) -> dict[str, Any]:
    p = pathlib.Path(path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    missing = {"status", "classes"} - set(cfg)
    if missing:
        raise FeatureError(f"tolerances config missing keys: {sorted(missing)}")
    cls = cfg["classes"]
    for name in (
        "bf16_gemm_vs_runtime_selected_logits",
        "fp32_direct_head_offline_vs_online",
        "base_restricted_probability",
        "inactive_slots",
        "masked_probability_sum_fp32",
        "metadata",
    ):
        if name not in cls:
            raise FeatureError(f"tolerances config missing class {name!r}")
    if str(cfg.get("schema", "")).endswith("/2"):
        # v2 declares the runtime-logit target dtype; a v2 file without that contract would
        # silently fall back to the v1 (mis-specified) behaviour, so it is refused.
        contract = cfg.get("runtime_contract")
        if not isinstance(contract, dict):
            raise FeatureError("numeric tolerances v2 requires a runtime_contract block")
        for key in ("dtype_map", "gate_classes_by_dtype", "fail_closed_reasons"):
            if not contract.get(key):
                raise FeatureError(f"numeric tolerances v2 runtime_contract missing {key!r}")
        for classes in list(contract["gate_classes_by_dtype"].values()) + list(
            contract.get("reference_classes_by_dtype", {}).values()
        ):
            for name in classes:
                if name not in cls:
                    raise FeatureError(
                        f"numeric tolerances v2 runtime_contract references unknown class {name!r}"
                    )
    return cfg


def new_run_id(prefix: str = "s3") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
