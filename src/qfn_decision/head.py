"""H1 decision head: FP32 linear readout over the 24 typed decision slots (stage S4).

What this module is
-------------------
One trainable object: ``W (24 x 2560)`` and ``b (24,)`` in float32, applied to a captured
2560-wide terminal prompt vector, followed by a *masked stable softmax* over the active
slots of one primitive. There is no hidden layer (that would be H2 and is not built), no
torch dependency and no pickled state: artifacts are a plain ``.npz`` of float32 arrays
plus a separate versioned JSON metadata file carrying the SHA-256 of the array data.

Contract implemented here (see docs/ARCHITECTURE-GATES.md and the S4 brief)
--------------------------------------------------------------------------
* slot layout comes from :mod:`qfn_decision.protocol`: ``noul`` 0:2, ``score`` 2:8,
  ``choice`` 8:8+n (2..16 options);
* the mask is applied **before** the softmax, so an inactive slot cannot receive mass no
  matter how large its logit is, and is returned as exactly ``0.0``;
* parameters = 24 * 2560 + 24 = 61464;
* initialization is ``W = effective lm_head rows cast to FP32``, ``bias = 0`` (no
  transplant from an incompatible checkpoint such as autotrust/JEV-27B);
* temperature is stored **per primitive**, must be strictly positive, and a calibration
  state can only be ``fitted`` when its method, source and scope are recorded. A head that
  was merely initialized is always reported ``untrained`` and ``uncalibrated``.

Honesty note: this module stores and validates provenance fields (feature base, template
id/version, tokenizer revision, row ids, pinned row hash, width). It refuses a head whose
provenance disagrees with a caller's declared feature/template/revision, but it does not
by itself prove that any feature archive is genuine.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import (
    KIND_CHOICE,
    KIND_NOUL,
    KIND_SCORE,
    KINDS,
    MAX_CHOICE_OPTIONS,
    MIN_CHOICE_OPTIONS,
    NUM_SLOTS,
    SLOT_RANGES,
    DecisionRequest,
    DecisionResult,
    ProtocolError,
)

__all__ = [
    "HeadError",
    "HEAD_SCHEMA",
    "HEAD_FORMAT_VERSION",
    "HEAD_DATA_HASH_SCHEMA",
    "FEATURE_WIDTH",
    "NUM_SLOTS",
    "PARAMETER_COUNT",
    "MIN_CHOICE_OPTIONS",
    "MAX_CHOICE_OPTIONS",
    "INCOMPATIBLE_INIT_SOURCES",
    "PRIMITIVE_KEYS",
    "bf16_to_fp32",
    "active_mask",
    "masked_softmax",
    "array_sha256",
    "data_sha256",
    "HeadProvenance",
    "CalibrationState",
    "DecisionHead",
    "decision_result",
]

# ---------------------------------------------------------------- contract constants

FEATURE_WIDTH = 2560                       # captured terminal prompt vector width
PARAMETER_COUNT = NUM_SLOTS * FEATURE_WIDTH + NUM_SLOTS   # 61464
PRIMITIVE_KEYS = (KIND_NOUL, KIND_CHOICE, KIND_SCORE)

HEAD_SCHEMA = "qfn.head/1"
HEAD_FORMAT_VERSION = 1
HEAD_DATA_HASH_SCHEMA = "qfn.head.arrays/1"

# A checkpoint from the reference fusion project uses a different backbone/slot mapping;
# initializing H1 from it would silently mix two incompatible bases.
INCOMPATIBLE_INIT_SOURCES = ("autotrust/JEV-27B", "jev-27b", "autotrust/jev-27b")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class HeadError(ValueError):
    """Raised when a head, its provenance, its calibration state or an artifact is invalid."""


# ---------------------------------------------------------------- exact BF16 widening

def bf16_to_fp32(raw: np.ndarray) -> np.ndarray:
    """Exact widening of raw bfloat16 bit patterns (uint16) to float32.

    bfloat16 is the top 16 bits of an IEEE-754 float32, so widening is lossless: the low
    16 mantissa bits are zero. This is a bit move, not a rounding operation.
    """
    arr = np.asarray(raw)
    if arr.dtype == np.uint16:
        bits = arr.astype(np.uint32) << 16
    elif arr.dtype == np.uint32:
        bits = arr << 16
    else:
        raise HeadError(f"expected raw bfloat16 bits as uint16, got dtype {arr.dtype}")
    return np.ascontiguousarray(bits).view(np.float32).copy()


# ---------------------------------------------------------------- masks and readout

def active_mask(kind: str, n_options: int | None = None) -> np.ndarray:
    """Boolean (24,) mask of the slots that belong to one primitive."""
    if kind not in KINDS:
        raise HeadError(f"unsupported kind {kind!r}; supported kinds: {', '.join(KINDS)}")
    mask = np.zeros(NUM_SLOTS, dtype=bool)
    start, end = SLOT_RANGES[kind]
    if kind == KIND_CHOICE:
        if not isinstance(n_options, int) or isinstance(n_options, bool):
            raise HeadError(
                f"choice requires an integer option count in "
                f"{MIN_CHOICE_OPTIONS}..{MAX_CHOICE_OPTIONS}, got {n_options!r}"
            )
        if not (MIN_CHOICE_OPTIONS <= n_options <= MAX_CHOICE_OPTIONS):
            raise HeadError(
                f"choice option count {n_options} is outside "
                f"{MIN_CHOICE_OPTIONS}..{MAX_CHOICE_OPTIONS}"
            )
        mask[start:start + n_options] = True
        return mask
    mask[start:end] = True
    return mask


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Stable softmax over the active slots only; inactive slots are exactly 0.0.

    The mask is applied *before* the exponentiation (inactive logits never enter the max
    or the denominator), which is what makes an inactive slot immune even to a logit of
    +1e4. Works in any float dtype; the shipped forward path uses float32.
    """
    z = np.asarray(logits)
    if z.ndim != 2:
        raise HeadError(f"logits must be 2-D (rows, {NUM_SLOTS}), got shape {z.shape}")
    if z.shape[1] != NUM_SLOTS:
        raise HeadError(f"logits must have {NUM_SLOTS} columns, got {z.shape[1]}")
    if z.dtype.kind != "f":
        raise HeadError(f"logits must be floating point, got dtype {z.dtype}")

    m = np.asarray(mask)
    if m.ndim == 1:
        if m.shape != (NUM_SLOTS,):
            raise HeadError(f"mask must have {NUM_SLOTS} entries, got shape {m.shape}")
        m = np.broadcast_to(m, z.shape)
    elif m.ndim == 2:
        if m.shape != z.shape:
            raise HeadError(f"mask shape {m.shape} does not match logits shape {z.shape}")
    else:
        raise HeadError(f"mask must be 1-D or 2-D, got {m.ndim} dimensions")
    m = m.astype(bool)

    counts = m.sum(axis=1)
    if np.any(counts < 2):
        bad = int(np.flatnonzero(counts < 2)[0])
        raise HeadError(
            f"row {bad} has {int(counts[bad])} active slot(s); a softmax needs at least 2"
        )

    neg_inf = np.array(-np.inf, dtype=z.dtype)
    masked = np.where(m, z, neg_inf)
    row_max = masked.max(axis=1, keepdims=True)
    # inactive slots are substituted with row_max before exp (so exp(0)=1, never overflow)
    # and then forced back to exactly 0.0: the mask is applied before normalisation.
    safe = np.where(m, z, row_max)
    exps = np.where(m, np.exp(safe - row_max), np.array(0.0, dtype=z.dtype))
    denom = exps.sum(axis=1, keepdims=True)
    if not np.all(denom > 0):
        raise HeadError("masked softmax produced a zero denominator")
    return exps / denom


# ---------------------------------------------------------------- array hashing

def array_sha256(array: np.ndarray) -> str:
    """Deterministic SHA-256 of an array's canonical float32/boolean bytes."""
    a = np.ascontiguousarray(np.asarray(array))
    if a.dtype == np.float64 or a.dtype == np.float16:
        raise HeadError(f"refusing to hash {a.dtype} artifact data; the artifact format is float32")
    if a.dtype.kind not in ("f", "b", "u", "i"):
        raise HeadError(f"unsupported artifact dtype {a.dtype}")
    head = json.dumps(
        {"dtype": str(a.dtype), "shape": list(a.shape)}, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    h = hashlib.sha256()
    h.update(HEAD_DATA_HASH_SCHEMA.encode("utf-8"))
    h.update(b"|")
    h.update(head)
    h.update(b"|")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def data_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Combined hash over the ordered per-array hashes (the artifact's data identity)."""
    parts = [f"{name}:{array_sha256(arrays[name])}" for name in sorted(arrays)]
    return hashlib.sha256(("|".join([HEAD_DATA_HASH_SCHEMA, *parts])).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- provenance

def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HeadError(f"{name} must be a non-empty string, got {value!r}")
    return value.strip()


@dataclass(frozen=True)
class HeadProvenance:
    """Where the head's numbers come from.

    ``rows`` are the 24 effective ``lm_head`` row indices (token ids) the head was
    initialized from; ``rows_sha256`` pins the exact row data; ``row_scaling`` records any
    scaling applied between the captured vector and this head's input.
    """

    feature_base: str
    feature_width: int
    template_id: str
    template_version: str
    tokenizer_revision: str
    rows: tuple[int, ...]
    row_scaling: float
    rows_sha256: str
    base_revision: str | None = None
    feature_sha256: str | None = None
    init_source: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.feature_base, "feature_base")
        _require_text(self.template_id, "template_id")
        _require_text(self.template_version, "template_version")
        _require_text(self.tokenizer_revision, "tokenizer_revision")
        if self.base_revision is not None:
            _require_text(self.base_revision, "base_revision")
        if self.init_source is not None:
            _require_text(self.init_source, "init_source")

        if not isinstance(self.feature_width, int) or isinstance(self.feature_width, bool):
            raise HeadError(f"feature_width must be an int, got {self.feature_width!r}")
        if self.feature_width != FEATURE_WIDTH:
            raise HeadError(
                f"provenance feature_width {self.feature_width} is not the contracted "
                f"{FEATURE_WIDTH}"
            )

        rows = tuple(self.rows)
        if len(rows) != NUM_SLOTS:
            raise HeadError(f"provenance rows must hold {NUM_SLOTS} lm_head row ids, got {len(rows)}")
        for r in rows:
            if not isinstance(r, int) or isinstance(r, bool) or r < 0:
                raise HeadError(f"row ids must be non-negative ints, got {r!r}")
        if len(set(rows)) != len(rows):
            raise HeadError("row ids must be unique: two slots cannot share one lm_head row")
        object.__setattr__(self, "rows", rows)

        if not isinstance(self.row_scaling, (int, float)) or isinstance(self.row_scaling, bool):
            raise HeadError(f"row_scaling must be a positive number, got {self.row_scaling!r}")
        scaling = float(self.row_scaling)
        if not math.isfinite(scaling) or scaling <= 0.0:
            raise HeadError(f"row_scaling must be finite and positive, got {self.row_scaling!r}")
        object.__setattr__(self, "row_scaling", scaling)

        if not isinstance(self.rows_sha256, str) or not _HEX64.match(self.rows_sha256):
            raise HeadError(f"rows_sha256 must be 64 lowercase hex chars, got {self.rows_sha256!r}")
        if self.feature_sha256 is not None and not _HEX64.match(str(self.feature_sha256)):
            raise HeadError(f"feature_sha256 must be 64 lowercase hex chars, got {self.feature_sha256!r}")

        # NOTE: an incompatible init_source is refused at initialization time
        # (DecisionHead.init_from_effective_rows / init_from_lm_head), not here: a provenance
        # record may legitimately describe a checkpoint that must never be used to seed H1.

    def as_dict(self) -> dict:
        return {
            "feature_base": self.feature_base,
            "feature_width": self.feature_width,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "tokenizer_revision": self.tokenizer_revision,
            "base_revision": self.base_revision,
            "feature_sha256": self.feature_sha256,
            "rows": list(self.rows),
            "row_scaling": self.row_scaling,
            "rows_sha256": self.rows_sha256,
            "init_source": self.init_source,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeadProvenance":
        if not isinstance(payload, Mapping):
            raise HeadError(f"provenance must be a mapping, got {type(payload).__name__}")
        known = {
            "feature_base", "feature_width", "template_id", "template_version",
            "tokenizer_revision", "base_revision", "feature_sha256", "rows", "row_scaling",
            "rows_sha256", "init_source", "extra",
        }
        missing = sorted(known - set(payload) - {"base_revision", "feature_sha256", "init_source", "extra"})
        if missing:
            raise HeadError(f"provenance record is missing fields: {missing}")
        return cls(
            feature_base=payload["feature_base"],
            feature_width=payload["feature_width"],
            template_id=payload["template_id"],
            template_version=payload["template_version"],
            tokenizer_revision=payload["tokenizer_revision"],
            rows=tuple(payload["rows"]),
            row_scaling=payload["row_scaling"],
            rows_sha256=payload["rows_sha256"],
            base_revision=payload.get("base_revision"),
            feature_sha256=payload.get("feature_sha256"),
            init_source=payload.get("init_source"),
            extra=dict(payload.get("extra") or {}),
        )


def _is_incompatible(source: Any) -> bool:
    if not isinstance(source, str):
        return False
    lowered = source.strip().lower()
    return any(marker.lower() in lowered for marker in INCOMPATIBLE_INIT_SOURCES)


# ---------------------------------------------------------------- calibration state

@dataclass(frozen=True)
class CalibrationState:
    """Per-primitive temperature plus an explicit calibration scope/source.

    ``fitted=False`` is the honest default: an initialized head is not calibrated. Setting
    ``fitted=True`` requires the fit method, the data source and the scope to be stated, so
    a calibrated tag can never be produced by initialization alone.
    """

    temperatures: Mapping[str, float]
    fitted: bool = False
    method: str | None = None
    source: str | None = None
    scope: str | None = None
    fitted_primitives: tuple[str, ...] = ()
    parameter_fingerprint: str | None = None
    provenance_fingerprint: str | None = None
    data_fingerprint: str | None = None
    per_primitive: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        temps = dict(self.temperatures or {})
        missing = [k for k in PRIMITIVE_KEYS if k not in temps]
        if missing:
            raise HeadError(f"temperature is per primitive; missing entries for {missing}")
        unknown = [k for k in temps if k not in PRIMITIVE_KEYS]
        if unknown:
            raise HeadError(f"unknown primitive(s) in temperatures: {unknown}")
        clean: dict[str, float] = {}
        for kind in PRIMITIVE_KEYS:
            v = temps[kind]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise HeadError(f"temperature[{kind}] must be a number, got {v!r}")
            t = float(v)
            if not math.isfinite(t) or t <= 0.0:
                raise HeadError(
                    f"temperature[{kind}] must be finite and strictly positive, got {v!r}"
                )
            clean[kind] = t
        object.__setattr__(self, "temperatures", clean)

        if not isinstance(self.fitted, bool):
            raise HeadError(f"fitted must be a bool, got {self.fitted!r}")

        fitted_primitives = tuple(self.fitted_primitives or ())
        if len(set(fitted_primitives)) != len(fitted_primitives):
            raise HeadError(f"fitted_primitives has duplicates: {fitted_primitives}")
        for kind in fitted_primitives:
            if kind not in PRIMITIVE_KEYS:
                raise HeadError(
                    f"unknown primitive {kind!r} in fitted_primitives; known: {list(PRIMITIVE_KEYS)}"
                )
        object.__setattr__(self, "fitted_primitives", fitted_primitives)

        per = {str(k): dict(v or {}) for k, v in dict(self.per_primitive or {}).items()}
        for kind in per:
            if kind not in PRIMITIVE_KEYS:
                raise HeadError(f"unknown primitive {kind!r} in per_primitive support")
        object.__setattr__(self, "per_primitive", per)

        if self.fitted:
            for name in ("method", "source", "scope"):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise HeadError(
                        f"a fitted calibration must record its {name}; refusing to tag a head "
                        "calibrated without a stated fit method/source/scope"
                    )
            if not fitted_primitives:
                raise HeadError(
                    "a fitted calibration must record which primitive(s) it actually fitted "
                    "(fitted_primitives); an empty list would claim calibration of nothing"
                )
            for kind in fitted_primitives:
                support = per.get(kind)
                if not isinstance(support, Mapping):
                    raise HeadError(
                        f"a fitted calibration must carry per_primitive support for {kind!r} "
                        f"(n_rows/scope/source); missing support for a fitted primitive"
                    )
                n_rows = support.get("n_rows")
                if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows < 1:
                    raise HeadError(
                        f"per_primitive support for {kind!r} must state n_rows >= 1, got {n_rows!r}; "
                        "a primitive with no fitted rows cannot be reported calibrated"
                    )
                for name in ("scope", "source"):
                    value = support.get(name)
                    if not isinstance(value, str) or not value.strip():
                        raise HeadError(
                            f"per_primitive support for {kind!r} must state its {name}"
                        )
                if support.get("calibrated", True) is not True:
                    raise HeadError(
                        f"{kind!r} is listed in fitted_primitives but its support says "
                        "calibrated=false; the two must agree"
                    )
            for kind, support in per.items():
                if kind not in fitted_primitives and support.get("calibrated") is True:
                    raise HeadError(
                        f"per_primitive support for {kind!r} claims calibrated=true but the "
                        "primitive is not listed in fitted_primitives"
                    )
            for name in ("parameter_fingerprint", "provenance_fingerprint", "data_fingerprint"):
                value = getattr(self, name)
                if not isinstance(value, str) or not _HEX64.match(value):
                    raise HeadError(
                        f"a fitted calibration must record its {name} (64 hex chars); a fit is "
                        "bound to the exact head parameters and calibration data it was fitted on, "
                        "so a parameter change or retrain invalidates it"
                    )
        else:
            if any(getattr(self, n) is not None for n in ("method", "source", "scope")):
                raise HeadError(
                    "an unfitted calibration must not carry a fit method/source/scope; "
                    "fit it, or leave the head uncalibrated"
                )
            if (fitted_primitives or self.parameter_fingerprint or self.provenance_fingerprint
                    or self.data_fingerprint):
                raise HeadError(
                    "an unfitted calibration must not carry fitted primitives or fit fingerprints"
                )

    # ------------------------------------------------------------------ constructors
    @classmethod
    def uncalibrated(cls) -> "CalibrationState":
        return cls(temperatures={k: 1.0 for k in PRIMITIVE_KEYS}, fitted=False)

    # ------------------------------------------------------------------ views
    def status(self) -> str:
        """The stored flag only. Use ``DecisionHead.calibration_status()`` for validity."""
        return "calibrated" if self.fitted else "uncalibrated"

    def as_dict(self) -> dict:
        return {
            "temperatures": dict(self.temperatures),
            "status": self.status(),
            "fitted": bool(self.fitted),
            "fit_method": self.method,
            "calibration_source": self.source,
            "calibration_scope": self.scope,
            "calibrated_primitives": list(self.fitted_primitives),
            "parameter_fingerprint": self.parameter_fingerprint,
            "provenance_fingerprint": self.provenance_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "per_primitive": {k: dict(v) for k, v in self.per_primitive.items()},
            "note": (
                "temperatures per primitive; applied to the slot logits before the masked softmax. "
                "The fit is bound to the parameter and calibration-data fingerprints, so changing "
                f"the weights invalidates it. Primitives actually fitted: {list(self.fitted_primitives)}"
                if self.fitted
                else "temperature 1.0 everywhere: this head is NOT calibrated"
            ),
        }

    def describe(self) -> dict:
        out = self.as_dict()
        if not self.fitted:
            out["note"] = (
                "uncalibrated: an initialized (or merely trained) head is not calibrated. "
                "Calibration requires a fit on held-out calibration data with a recorded method."
            )
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationState":
        if not isinstance(payload, Mapping):
            raise HeadError(f"calibration must be a mapping, got {type(payload).__name__}")
        if "temperatures" not in payload:
            raise HeadError("calibration record is missing 'temperatures'")
        fitted = payload.get("fitted", False)
        return cls(
            temperatures=payload["temperatures"],
            fitted=bool(fitted),
            method=payload.get("fit_method") or payload.get("method"),
            source=payload.get("calibration_source") or payload.get("source"),
            scope=payload.get("calibration_scope") or payload.get("scope"),
            fitted_primitives=tuple(payload.get("calibrated_primitives") or ()),
            parameter_fingerprint=payload.get("parameter_fingerprint"),
            provenance_fingerprint=payload.get("provenance_fingerprint"),
            data_fingerprint=payload.get("data_fingerprint"),
            per_primitive=payload.get("per_primitive") or {},
        )


# ---------------------------------------------------------------- the head

@dataclass(frozen=True)
class DecisionHead:
    """FP32 linear 24-slot readout. Immutable: mutations return a new head."""

    weight: np.ndarray                     # (24, 2560) float32
    bias: np.ndarray                       # (24,) float32
    provenance: HeadProvenance
    calibration: CalibrationState = field(default_factory=CalibrationState.uncalibrated)
    training: dict | None = None           # training record, None == never trained
    choice_max_mask: np.ndarray | None = None

    # ------------------------------------------------------------------ construction
    def __post_init__(self) -> None:
        w = np.asarray(self.weight)
        if w.dtype != np.float32:
            raise HeadError(
                f"head weight must be float32 (the artifact format is FP32), got {w.dtype}"
            )
        if w.ndim != 2 or w.shape != (NUM_SLOTS, FEATURE_WIDTH):
            raise HeadError(f"head weight must be ({NUM_SLOTS}, {FEATURE_WIDTH}), got {w.shape}")
        if not np.isfinite(w).all():
            raise HeadError("head weight contains non-finite values")
        object.__setattr__(self, "weight", np.ascontiguousarray(w))

        b = np.asarray(self.bias)
        if b.dtype != np.float32:
            raise HeadError(f"head bias must be float32, got {b.dtype}")
        if b.shape != (NUM_SLOTS,):
            raise HeadError(f"head bias must be ({NUM_SLOTS},), got {b.shape}")
        if not np.isfinite(b).all():
            raise HeadError("head bias contains non-finite values")
        object.__setattr__(self, "bias", np.ascontiguousarray(b))

        if not isinstance(self.provenance, HeadProvenance):
            raise HeadError(f"provenance must be a HeadProvenance, got {type(self.provenance).__name__}")
        if self.provenance.feature_width != w.shape[1]:
            raise HeadError(
                f"provenance feature_width {self.provenance.feature_width} does not match "
                f"the weight width {w.shape[1]}"
            )
        if self.calibration is None:
            object.__setattr__(self, "calibration", CalibrationState.uncalibrated())
        elif not isinstance(self.calibration, CalibrationState):
            raise HeadError("calibration must be a CalibrationState")
        if self.training is not None and not isinstance(self.training, dict):
            raise HeadError(f"training record must be a dict or None, got {type(self.training).__name__}")

        if self.choice_max_mask is not None:
            m = np.asarray(self.choice_max_mask)
            if m.dtype != np.bool_ or m.shape != (NUM_SLOTS,):
                raise HeadError(f"choice_max_mask must be a bool ({NUM_SLOTS},) array, got {m.dtype} {m.shape}")
            expected = active_mask(KIND_CHOICE, n_options=MAX_CHOICE_OPTIONS)
            if not np.array_equal(m.astype(bool), expected):
                raise HeadError(
                    "choice_max_mask must be exactly the 16-choice active mask "
                    f"(slots {SLOT_RANGES[KIND_CHOICE][0]}..{MAX_CHOICE_OPTIONS + SLOT_RANGES[KIND_CHOICE][0] - 1})"
                )
            object.__setattr__(self, "choice_max_mask", np.ascontiguousarray(m))

    # ------------------------------------------------------------------ initializers
    @classmethod
    def init_from_effective_rows(
        cls,
        rows: np.ndarray,
        provenance: HeadProvenance,
        bias: np.ndarray | None = None,
        training: dict | None = None,
    ) -> "DecisionHead":
        """``W = effective lm_head rows cast to FP32``, ``bias = 0`` by default.

        ``rows`` may be raw bfloat16 bits (uint16, widened exactly), float32, or float64
        (down-cast, which is NOT exact and is recorded by the caller's provenance).
        """
        if _is_incompatible(provenance.init_source):
            raise HeadError(
                f"refusing to initialize from incompatible checkpoint {provenance.init_source!r}: "
                "H1 must come from the serving backbone's own effective lm_head rows"
            )
        raw = np.asarray(rows)
        if raw.dtype == np.uint16 or raw.dtype == np.uint32:
            w = bf16_to_fp32(raw)
        elif raw.dtype in (np.float32, np.float64):
            w = raw.astype(np.float32)
        else:
            raise HeadError(f"unsupported row dtype {raw.dtype}; expected uint16 bf16 bits, float32 or float64")
        if w.shape != (NUM_SLOTS, FEATURE_WIDTH):
            raise HeadError(f"effective rows must be ({NUM_SLOTS}, {FEATURE_WIDTH}), got {w.shape}")
        if not np.isfinite(w).all():
            raise HeadError("effective rows contain non-finite values")
        b = np.zeros(NUM_SLOTS, dtype=np.float32) if bias is None else np.asarray(bias, dtype=np.float32)
        return cls(weight=np.ascontiguousarray(w), bias=np.ascontiguousarray(b),
                   provenance=provenance, calibration=CalibrationState.uncalibrated(),
                   training=training)

    @classmethod
    def init_from_lm_head(
        cls,
        lm_head: np.ndarray,
        row_indices: Sequence[int],
        provenance: HeadProvenance,
        bias: np.ndarray | None = None,
    ) -> "DecisionHead":
        """Select the 24 declared rows out of a full ``(vocab, 2560)`` lm_head matrix."""
        lm = np.asarray(lm_head)
        if lm.ndim != 2 or lm.shape[1] != FEATURE_WIDTH:
            raise HeadError(f"lm_head must be (vocab, {FEATURE_WIDTH}), got {lm.shape}")
        idx = tuple(int(i) for i in row_indices)
        if idx != tuple(provenance.rows):
            raise HeadError(
                "row_indices and provenance.rows must agree exactly; "
                f"got {idx[:4]}... vs {tuple(provenance.rows)[:4]}..."
            )
        if max(idx) >= lm.shape[0] or min(idx) < 0:
            raise HeadError(f"row index out of range for a lm_head with {lm.shape[0]} rows")
        selected = lm[list(idx)]
        return cls.init_from_effective_rows(rows=selected, provenance=provenance, bias=bias)

    @classmethod
    def random_control(
        cls,
        seed: int,
        provenance: HeadProvenance,
        scale: float = 0.02,
        bias: np.ndarray | None = None,
    ) -> "DecisionHead":
        """Seeded random control head (a control arm, never a decision-quality claim)."""
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise HeadError(f"seed must be a non-negative int, got {seed!r}")
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise HeadError(f"scale must be finite and positive, got {scale!r}")
        rng = np.random.default_rng(seed)
        w = (rng.standard_normal((NUM_SLOTS, FEATURE_WIDTH)) * float(scale)).astype(np.float32)
        b = np.zeros(NUM_SLOTS, dtype=np.float32) if bias is None else np.asarray(bias, dtype=np.float32)
        return cls(weight=w, bias=b, provenance=provenance, calibration=CalibrationState.uncalibrated())

    # ------------------------------------------------------------------ views
    def n_parameters(self) -> int:
        return int(self.weight.size + self.bias.size)

    @property
    def trained(self) -> bool:
        return self.training is not None

    # ------------------------------------------------------------------ identity fingerprints
    def provenance_fingerprint(self) -> str:
        """SHA-256 of the declared feature identity (source base/width, template, revisions, rows)."""
        payload = {
            "schema": "qfn.head.provenance/1",
            "feature_base": self.provenance.feature_base,
            "feature_width": int(self.provenance.feature_width),
            "template_id": self.provenance.template_id,
            "template_version": self.provenance.template_version,
            "tokenizer_revision": self.provenance.tokenizer_revision,
            "base_revision": self.provenance.base_revision,
            "rows": list(self.provenance.rows),
            "row_scaling": float(self.provenance.row_scaling),
            "rows_sha256": self.provenance.rows_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def parameter_fingerprint(self) -> str:
        """SHA-256 over the provenance fingerprint plus the exact float32 parameters.

        Used to bind a calibration fit to the parameters it was fitted on: any weight/bias change
        or any provenance change produces a different fingerprint.
        """
        h = hashlib.sha256(b"qfn.head.params/1")
        h.update(self.provenance_fingerprint().encode("ascii"))
        h.update(b"|w|")
        h.update(np.ascontiguousarray(self.weight, dtype=np.float32).tobytes(order="C"))
        h.update(b"|b|")
        h.update(np.ascontiguousarray(self.bias, dtype=np.float32).tobytes(order="C"))
        return h.hexdigest()

    def calibration_is_valid(self) -> bool:
        """True only when a fit exists and still matches these exact parameters and provenance."""
        return self.calibration_status() == "calibrated"

    def calibration_status(self) -> str:
        """Authoritative status: ``calibrated`` / ``uncalibrated`` / ``invalidated_*``."""
        cal = self.calibration
        if not cal.fitted:
            return "uncalibrated"
        if cal.provenance_fingerprint != self.provenance_fingerprint():
            return "invalidated_provenance_change"
        if cal.parameter_fingerprint != self.parameter_fingerprint():
            return "invalidated_parameter_change"
        return "calibrated"

    def primitive_calibration_status(self, kind: str) -> dict:
        """Per-primitive honesty: only primitives actually fitted report calibrated."""
        if kind not in PRIMITIVE_KEYS:
            raise HeadError(f"unknown primitive {kind!r}; known: {list(PRIMITIVE_KEYS)}")
        cal = self.calibration
        status = self.calibration_status()
        support = dict(cal.per_primitive.get(kind) or {})
        calibrated = bool(status == "calibrated" and kind in cal.fitted_primitives)
        reason = None
        if not calibrated:
            if not cal.fitted:
                reason = "no calibration fit is recorded on this head"
            elif status != "calibrated":
                reason = f"the calibration fit is invalid: {status}"
            else:
                reason = (
                    f"the fit covered only {list(cal.fitted_primitives)}; {kind} keeps temperature "
                    "1.0 and is not calibrated"
                )
        return {
            "primitive": kind,
            "status": "calibrated" if calibrated else "uncalibrated",
            "calibrated": calibrated,
            "temperature": float(cal.temperatures[kind]),
            "n_rows": support.get("n_rows", 0),
            "scope": support.get("scope"),
            "source": support.get("source"),
            "reason": reason,
        }

    def calibrated_primitives(self) -> list[str]:
        return [k for k in PRIMITIVE_KEYS if self.primitive_calibration_status(k)["calibrated"]]

    def metadata_calibration_summary(self) -> dict:
        """Primitive-aware calibration metadata (additive, for backend/gateway consumers)."""
        status = self.calibration_status()
        return {
            "status": status,
            "stored_status": self.calibration.status(),
            "calibrated": status == "calibrated",
            "calibrated_primitives": self.calibrated_primitives(),
            "per_primitive": {k: self.primitive_calibration_status(k) for k in PRIMITIVE_KEYS},
            "parameter_fingerprint": self.calibration.parameter_fingerprint,
            "data_fingerprint": self.calibration.data_fingerprint,
            "note": (
                "status is derived from the fit fingerprints: a parameter or provenance change "
                "invalidates the fit even though the stored flag stays set"
                if status == "invalidated_parameter_change"
                else "calibration status is bound to the head's parameter fingerprint"
            ),
        }

    def describe(self) -> dict:
        status = self.calibration_status()
        return {
            "schema": HEAD_SCHEMA,
            "format_version": HEAD_FORMAT_VERSION,
            "parameters": self.n_parameters(),
            "feature_width": int(self.weight.shape[1]),
            "num_slots": int(self.weight.shape[0]),
            "provenance": self.provenance.as_dict(),
            "calibration": self.calibration.as_dict(),
            "artifact_state": {
                "trained": self.trained,
                "training_record": self.training,
                "calibration_status": status,
                "primitives_trained": sorted((self.training or {}).get("per_primitive_support", {}).keys())
                if isinstance((self.training or {}).get("per_primitive_support"), dict) else [],
                "calibrated_primitives": self.calibrated_primitives(),
                "parameter_fingerprint": self.parameter_fingerprint(),
                "note": (
                    "an initialized but never trained head reports trained=false; a head with "
                    "temperature 1.0 everywhere reports calibration_status=uncalibrated. "
                    "calibration_status is recomputed from the parameter fingerprint, so a weight "
                    "change or a retrain invalidates a previous fit."
                ),
            },
        }

    # ------------------------------------------------------------------ forward
    def forward_logits(self, features: np.ndarray) -> np.ndarray:
        """FP32 slot logits ``x @ W.T + b`` for a (N, 2560) float32 batch."""
        x = np.asarray(features)
        if x.dtype != np.float32:
            raise HeadError(f"features must be float32, got {x.dtype}")
        if x.ndim != 2 or x.shape[1] != self.weight.shape[1]:
            raise HeadError(f"features must be (N, {self.weight.shape[1]}), got {x.shape}")
        if not np.isfinite(x).all():
            raise HeadError("features contain non-finite values")
        logits = x @ self.weight.T + self.bias
        if not np.isfinite(logits).all():
            raise HeadError("forward pass produced non-finite logits")
        return logits.astype(np.float32, copy=False)

    def forward_probs(self, features: np.ndarray, kind: str, n_options: int | None = None) -> np.ndarray:
        """Masked, temperature-scaled softmax probabilities for one primitive."""
        mask = active_mask(kind, n_options=n_options)
        temperature = np.float32(self.calibration.temperatures[kind])
        logits = self.forward_logits(features)
        scaled = (logits / temperature).astype(np.float32, copy=False)
        return masked_softmax(scaled, mask).astype(np.float32, copy=False)

    @staticmethod
    def describe_slots(probabilities: Sequence[float], kind: str, n_options: int | None = None) -> list[float]:
        """Active-slot probabilities in caller-key order."""
        mask = active_mask(kind, n_options=n_options)
        probs = np.asarray(probabilities)
        if probs.shape[-1] != NUM_SLOTS:
            raise HeadError(f"a slot vector must have {NUM_SLOTS} entries, got {probs.shape[-1]}")
        return [float(v) for v in probs[..., mask]]

    # ------------------------------------------------------------------ immutability helpers
    def with_calibration(self, calibration: CalibrationState) -> "DecisionHead":
        return dataclasses.replace(self, calibration=calibration)

    def with_training(self, training: dict | None) -> "DecisionHead":
        return dataclasses.replace(self, training=None if training is None else dict(training))

    def with_choice_max_mask(self, mask: np.ndarray | None) -> "DecisionHead":
        return dataclasses.replace(self, choice_max_mask=None if mask is None else np.asarray(mask))

    def with_parameters(self, weight: np.ndarray, bias: np.ndarray) -> "DecisionHead":
        return dataclasses.replace(self, weight=np.asarray(weight), bias=np.asarray(bias))

    # ------------------------------------------------------------------ validation
    def validate_against(
        self,
        feature_base: str | None = None,
        feature_width: int | None = None,
        template_id: str | None = None,
        template_version: str | None = None,
        tokenizer_revision: str | None = None,
        rows_sha256: str | None = None,
        feature_sha256: str | None = None,
    ) -> None:
        """Refuse a head whose provenance disagrees with the caller's declared inputs."""
        p = self.provenance
        checks = (
            ("feature_base", feature_base, p.feature_base),
            ("feature_width", feature_width, p.feature_width),
            ("template_id", template_id, p.template_id),
            ("template_version", template_version, p.template_version),
            ("tokenizer_revision", tokenizer_revision, p.tokenizer_revision),
            ("rows_sha256", rows_sha256, p.rows_sha256),
            ("feature_sha256", feature_sha256, p.feature_sha256),
        )
        for name, supplied, stored in checks:
            if supplied is None:
                continue
            if supplied != stored:
                raise HeadError(
                    f"{name} mismatch: head artifact records {stored!r}, caller declares {supplied!r}; "
                    "refusing to mix a head with a different feature base/template/hash/width"
                )

    # ------------------------------------------------------------------ artifact IO
    def save(self, path: str | os.PathLike, overwrite: bool = False, extra: Mapping[str, Any] | None = None) -> dict:
        """Atomically write ``<path>`` (npz arrays) plus ``<path stem>.json`` (metadata).

        Refuses to overwrite any existing member unless ``overwrite=True``.
        """
        npz_path = Path(path)
        if npz_path.suffix != ".npz":
            raise HeadError(f"artifact path must end in .npz, got {npz_path.name!r}")
        meta_path = npz_path.with_suffix(".json")
        for target in (npz_path, meta_path):
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"refusing to overwrite existing artifact {target}; pass overwrite=True explicitly"
                )

        arrays: dict[str, np.ndarray] = {"weight": self.weight, "bias": self.bias}
        if self.choice_max_mask is not None:
            arrays["choice_max_mask"] = self.choice_max_mask.astype(np.bool_, copy=False)

        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        npz_bytes = buffer.getvalue()

        meta = {
            "schema": HEAD_SCHEMA,
            "format_version": HEAD_FORMAT_VERSION,
            "data_hash_schema": HEAD_DATA_HASH_SCHEMA,
            "parameters": self.n_parameters(),
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
            "bytes": {k: int(v.nbytes) for k, v in arrays.items()},
            "array_sha256": {k: array_sha256(v) for k, v in arrays.items()},
            "data_sha256": data_sha256(arrays),
            "provenance": self.provenance.as_dict(),
            "calibration": self.calibration.as_dict(),
            "training": self.training,
            "artifact_state": self.describe()["artifact_state"],
            "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "extra": dict(extra or {}),
        }
        meta_bytes = json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")

        _atomic_write_bytes(npz_path, npz_bytes)
        _atomic_write_bytes(meta_path, meta_bytes)
        return {
            "npz": str(npz_path),
            "metadata": str(meta_path),
            "npz_bytes": len(npz_bytes),
            "metadata_bytes": len(meta_bytes),
            "data_sha256": meta["data_sha256"],
        }

    @classmethod
    def load(cls, path: str | os.PathLike, verify_checksum: bool = True) -> "DecisionHead":
        npz_path = Path(path)
        meta_path = npz_path.with_suffix(".json")
        if not npz_path.exists():
            raise HeadError(f"head artifact not found: {npz_path}")
        if not meta_path.exists():
            raise HeadError(f"head metadata JSON not found: {meta_path} (metadata is never embedded as pickle)")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise HeadError(f"head metadata JSON is unreadable: {e}") from e
        if not isinstance(meta, dict):
            raise HeadError("head metadata JSON must be an object")
        if meta.get("schema") != HEAD_SCHEMA:
            raise HeadError(f"unexpected head schema {meta.get('schema')!r}, expected {HEAD_SCHEMA!r}")
        if meta.get("format_version") != HEAD_FORMAT_VERSION:
            raise HeadError(
                f"unsupported head format_version {meta.get('format_version')!r}; "
                f"this build reads {HEAD_FORMAT_VERSION}"
            )

        with np.load(npz_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
        for name in ("weight", "bias"):
            if name not in arrays:
                raise HeadError(f"head artifact is missing the {name!r} array")

        if verify_checksum:
            declared = meta.get("data_sha256")
            if not isinstance(declared, str) or not _HEX64.match(declared):
                raise HeadError("head metadata is missing a valid data_sha256 checksum")
            actual = data_sha256({k: v for k, v in arrays.items() if k in ("weight", "bias", "choice_max_mask")})
            if actual != declared:
                raise HeadError(
                    "artifact checksum mismatch: the array data does not match the recorded "
                    f"data_sha256 ({actual[:12]}... vs {declared[:12]}...); refusing to load"
                )
            for name, digest in (meta.get("array_sha256") or {}).items():
                if name in arrays and array_sha256(arrays[name]) != digest:
                    raise HeadError(f"artifact checksum mismatch for array {name!r}")

        if meta.get("parameters") not in (None, PARAMETER_COUNT):
            raise HeadError(f"metadata parameters {meta.get('parameters')} is not {PARAMETER_COUNT}")
        for name in ("weight", "bias"):
            declared_shape = (meta.get("shapes") or {}).get(name)
            if declared_shape is not None and list(arrays[name].shape) != list(declared_shape):
                raise HeadError(
                    f"metadata shape for {name} ({declared_shape}) disagrees with the stored array "
                    f"({list(arrays[name].shape)})"
                )

        provenance = HeadProvenance.from_dict(meta["provenance"])
        calibration = CalibrationState.from_dict(meta.get("calibration") or {"temperatures": {}, "fitted": False})
        mask = arrays.get("choice_max_mask")
        return cls(
            weight=arrays["weight"],
            bias=arrays["bias"],
            provenance=provenance,
            calibration=calibration,
            training=meta.get("training"),
            choice_max_mask=None if mask is None else mask.astype(bool, copy=False),
        )

    @classmethod
    def load_metadata(cls, path: str | os.PathLike) -> dict:
        meta_path = Path(path).with_suffix(".json")
        if not meta_path.exists():
            raise HeadError(f"head metadata JSON not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- atomic write

def decision_result(head: "DecisionHead", features_row: np.ndarray, request: DecisionRequest) -> DecisionResult:
    """Typed readout for one request, through the existing protocol utilities.

    The probabilities are the head's masked, temperature-scaled softmax; ``DecisionResult``
    (from :mod:`qfn_decision.protocol`) then supplies the caller keys, the score expected level
    (``expected`` between 0 and 5), ``p_true`` for ``noul`` and ``raw_max_probability``. That
    raw maximum is deliberately *not* a calibrated "Jev confidence": ``jev_confidence`` stays
    ``None`` with ``confidence_status`` marked absent until the reference formula is verified.
    """
    if not isinstance(request, DecisionRequest):
        raise HeadError(f"expected a DecisionRequest, got {type(request).__name__}")
    row = np.asarray(features_row, dtype=np.float32)
    if row.ndim == 1:
        row = row.reshape(1, -1)
    if row.shape[0] != 1:
        raise HeadError(f"decision_result reads exactly one row, got shape {row.shape}")
    probs = head.forward_probs(row, request.kind, n_options=len(request.options) or None)
    try:
        # the protocol utilities validate plain Python numbers, so the fp32 row is widened to
        # Python floats at this boundary (values are unchanged; only the container type differs)
        return DecisionResult.from_slot_probabilities(request, [float(v) for v in probs[0]])
    except ProtocolError as e:
        raise HeadError(f"the head readout is not a valid typed result: {e}") from e


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-" + path.name + "-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
