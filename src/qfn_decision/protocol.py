"""Typed S1 decision protocol (stage S1, buildout only).

Scope of this module: the *typed* half of System 1. A request declares one kind
(`noul` / `choice` / `score`), the state it is a decision about, one question, and
the caller-facing payload for that kind. Nothing here generates text tokens; S1 is
a probability readout over 24 fixed slots.

Slot layout is taken from the reference judge config actually fetched from the
pinned AutoTrust revision (see reports/s1/sources/judge_config.json):
    noul  -> slots 0:2    (false, true)
    score -> slots 2:8    ("0".."5", six ordered levels)
    choice-> slots 8:24   (16 slots, A..P; this build supports 2..16 options)
Inactive slots must stay masked (exactly zero) everywhere downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "ProtocolError",
    "ProtocolLimits",
    "ChoiceOption",
    "DecisionRequest",
    "NUM_SLOTS",
    "SLOT_RANGES",
    "SCORE_LEVEL_COUNT",
    "MIN_CHOICE_OPTIONS",
    "MAX_CHOICE_OPTIONS",
    "CHOICE_LETTERS",
    "KINDS",
    "KIND_NOUL",
    "KIND_CHOICE",
    "KIND_SCORE",
]

KIND_NOUL = "noul"
KIND_CHOICE = "choice"
KIND_SCORE = "score"
KINDS = (KIND_NOUL, KIND_CHOICE, KIND_SCORE)

# 24-slot layout, verbatim from the pinned reference judge_config.json / decision_head.json.
NUM_SLOTS = 24
SLOT_RANGES = {KIND_NOUL: (0, 2), KIND_SCORE: (2, 8), KIND_CHOICE: (8, 24)}

SCORE_LEVEL_COUNT = 6
MIN_CHOICE_OPTIONS = 2
MAX_CHOICE_OPTIONS = 16
CHOICE_LETTERS = "ABCDEFGHIJKLMNOP"

NOUL_KEYS = ("false", "true")
SCORE_KEYS = tuple(str(i) for i in range(SCORE_LEVEL_COUNT))


class ProtocolError(ValueError):
    """Raised when a typed decision request/target/result violates the protocol."""


@dataclass(frozen=True)
class ProtocolLimits:
    """Input bounds. Values are deliberately generous but finite; oversized input
    is rejected rather than truncated, so a request never silently changes meaning."""

    max_state_chars: int = 32_768
    max_question_chars: int = 4_096
    max_option_description_chars: int = 2_048
    max_key_chars: int = 64
    max_total_chars: int = 96_000

    def __post_init__(self) -> None:
        for name in ("max_state_chars", "max_question_chars", "max_option_description_chars",
                     "max_key_chars", "max_total_chars"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ProtocolError(f"limit {name} must be a positive int, got {v!r}")


@dataclass(frozen=True)
class ChoiceOption:
    """A caller-facing option: the key is what the caller receives back, the
    description is what the model reads in the prompt. Internal slot labels (A..P)
    are positional, not semantic classes, and are never the caller key."""

    key: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str):
            raise ProtocolError(f"option key must be str, got {type(self.key).__name__}")
        if not isinstance(self.description, str):
            raise ProtocolError(f"option description must be str, got {type(self.description).__name__}")


def _require_text(value: object, field_name: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{field_name} must be str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ProtocolError(f"{field_name} must be non-empty (after stripping whitespace)")
    if len(text) > max_chars:
        raise ProtocolError(f"{field_name} is oversized: {len(text)} chars > limit {max_chars}")
    return text


@dataclass(frozen=True)
class DecisionRequest:
    """One typed decision question. Immutable; validated by :meth:`create`."""

    kind: str
    state: str
    question: str
    options: tuple[ChoiceOption, ...] = ()
    score_rubrics: tuple[str, ...] = ()
    limits: ProtocolLimits = field(default_factory=ProtocolLimits, repr=False, compare=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(
        cls,
        kind: str,
        state: str,
        question: str,
        options: Sequence[ChoiceOption] | None = None,
        score_rubrics: Sequence[str] | None = None,
        limits: ProtocolLimits | None = None,
    ) -> "DecisionRequest":
        limits = limits or ProtocolLimits()
        if kind not in KINDS:
            raise ProtocolError(f"unsupported kind {kind!r}; supported kinds: {', '.join(KINDS)}")

        state_t = _require_text(state, "state", limits.max_state_chars)
        question_t = _require_text(question, "question", limits.max_question_chars)

        opts: tuple[ChoiceOption, ...] = ()
        rubrics: tuple[str, ...] = ()

        if kind == KIND_CHOICE:
            if score_rubrics:
                raise ProtocolError("choice requests must not carry score_rubrics")
            if options is None or len(list(options)) == 0:
                raise ProtocolError("choice requests require 2..16 options with caller keys and descriptions")
            raw = list(options)
            for i, o in enumerate(raw):
                if not isinstance(o, ChoiceOption):
                    raise ProtocolError(f"option {i} must be a ChoiceOption, got {type(o).__name__}")
            if not (MIN_CHOICE_OPTIONS <= len(raw) <= MAX_CHOICE_OPTIONS):
                raise ProtocolError(
                    f"choice requests require {MIN_CHOICE_OPTIONS}..{MAX_CHOICE_OPTIONS} options, got {len(raw)}"
                )
            keys = []
            cleaned = []
            for i, o in enumerate(raw):
                key = _require_text(o.key, f"option[{i}].key", limits.max_key_chars)
                desc = _require_text(o.description, f"option[{i}].description", limits.max_option_description_chars)
                if "\n" in desc or "\r" in desc:
                    raise ProtocolError(
                        f"option[{i}].description must be single-line (the template lists one option per line)"
                    )
                keys.append(key)
                cleaned.append(ChoiceOption(key=key, description=desc))
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            if dupes:
                raise ProtocolError(f"duplicate option keys are not allowed: {dupes}")
            opts = tuple(cleaned)

        elif kind == KIND_SCORE:
            if options:
                raise ProtocolError("score requests must not carry options; score uses six ordered rubric descriptions")
            if score_rubrics is None or len(list(score_rubrics)) != SCORE_LEVEL_COUNT:
                got = 0 if score_rubrics is None else len(list(score_rubrics))
                raise ProtocolError(
                    f"score requests require exactly {SCORE_LEVEL_COUNT} ordered rubric descriptions, got {got}"
                )
            rubrics = tuple(
                _require_text(r, f"score_rubrics[{i}]", limits.max_option_description_chars)
                for i, r in enumerate(score_rubrics)
            )

        else:  # noul
            if options or score_rubrics:
                raise ProtocolError("noul requests must not carry options or score_rubrics; noul is fixed false/true")

        total = len(state_t) + len(question_t) + sum(len(o.description) for o in opts) + sum(len(r) for r in rubrics)
        if total > limits.max_total_chars:
            raise ProtocolError(f"request is oversized: {total} total chars > limit {limits.max_total_chars}")

        return cls(kind=kind, state=state_t, question=question_t, options=opts, score_rubrics=rubrics, limits=limits)

    # ------------------------------------------------------------------ slot mapping
    @property
    def slot_range(self) -> tuple[int, int]:
        return SLOT_RANGES[self.kind]

    def n_active(self) -> int:
        start, end = self.slot_range
        if self.kind == KIND_CHOICE:
            return len(self.options)
        return end - start

    def active_slot_indices(self) -> list[int]:
        start, _ = self.slot_range
        return list(range(start, start + self.n_active()))

    def caller_keys(self) -> tuple[str, ...]:
        if self.kind == KIND_CHOICE:
            return tuple(o.key for o in self.options)
        if self.kind == KIND_NOUL:
            return NOUL_KEYS
        return SCORE_KEYS

    def descriptions(self) -> tuple[str, ...]:
        """What the model reads, in slot order."""
        if self.kind == KIND_CHOICE:
            return tuple(o.description for o in self.options)
        if self.kind == KIND_SCORE:
            return self.score_rubrics
        return NOUL_KEYS

    def key_for_slot(self, slot: int) -> str:
        slots = dict(zip(self.active_slot_indices(), self.caller_keys()))
        if slot not in slots:
            raise ProtocolError(f"slot {slot} is inactive for kind {self.kind!r}")
        return slots[slot]

    def slot_for_key(self, key: str) -> int:
        keys = self.caller_keys()
        if key not in keys:
            raise ProtocolError(f"unknown caller key {key!r}")
        return self.active_slot_indices()[keys.index(key)]


# Declared tolerance for accepting a supplied target as a probability distribution.
# A target inside this tolerance is renormalised to sum exactly 1; anything outside is
# rejected instead of silently rescaled.
PROB_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class DecisionTarget:
    """A validated, exactly-normalised target distribution over the outcomes of one
    primitive: 2 for noul, 6 ordered levels for score, one per caller option for choice.

    Uniform targets are legal (genuine ambiguity and placeholder-looking teacher output
    both produce them); this type never discards them, it labels them via
    :attr:`is_uniform` and the provenance recorded in the data layer.
    """

    kind: str
    probabilities: tuple[float, ...]
    n_options: int | None = None

    @classmethod
    def create(
        cls,
        kind: str,
        probabilities: Sequence[float],
        n_options: int | None = None,
        tolerance: float = PROB_SUM_TOLERANCE,
    ) -> "DecisionTarget":
        if kind not in KINDS:
            raise ProtocolError(f"unsupported kind {kind!r}; supported kinds: {', '.join(KINDS)}")

        if kind == KIND_NOUL:
            expected = 2
        elif kind == KIND_SCORE:
            expected = SCORE_LEVEL_COUNT
        else:
            if n_options is None:
                raise ProtocolError("choice targets require n_options so length can be checked")
            if not isinstance(n_options, int) or isinstance(n_options, bool):
                raise ProtocolError(f"n_options must be int, got {type(n_options).__name__}")
            expected = n_options

        vals = list(probabilities)
        if len(vals) != expected:
            raise ProtocolError(f"{kind} target must have {expected} probabilities, got {len(vals)}")

        out: list[float] = []
        for i, v in enumerate(vals):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ProtocolError(f"target[{i}] must be a real number, got {type(v).__name__}")
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):  # NaN or +-inf
                raise ProtocolError(f"target[{i}] is non-finite: {v!r}")
            if f < 0.0:
                raise ProtocolError(f"target[{i}] is negative: {v!r}")
            out.append(f)

        total = sum(out)
        if total <= 0.0:
            raise ProtocolError("target probabilities must sum to a positive value")
        if abs(total - 1.0) > tolerance:
            raise ProtocolError(
                f"target is not a normalised distribution: sum={total!r} outside tolerance {tolerance}"
            )
        out = [v / total for v in out]  # exact renormalisation inside the declared tolerance

        return cls(kind=kind, probabilities=tuple(out), n_options=n_options)

    # ------------------------------------------------------------------ helpers
    @property
    def n_outcomes(self) -> int:
        return len(self.probabilities)

    @property
    def top_index(self) -> int:
        return max(range(self.n_outcomes), key=lambda i: self.probabilities[i])

    @property
    def is_uniform(self) -> bool:
        return len(set(self.probabilities)) == 1

    @property
    def uniform_value(self) -> float | None:
        return self.probabilities[0] if self.is_uniform else None

    @property
    def p_true(self) -> float:
        if self.kind != KIND_NOUL:
            raise ProtocolError("p_true is only defined for noul targets")
        return self.probabilities[1]

    @property
    def level_distribution(self) -> tuple[float, ...]:
        if self.kind != KIND_SCORE:
            raise ProtocolError("level_distribution is only defined for score targets")
        return self.probabilities

    @property
    def expected_level(self) -> float:
        if self.kind != KIND_SCORE:
            raise ProtocolError("expected_level is only defined for score targets")
        return sum(i * p for i, p in enumerate(self.probabilities))


# Jev's documented confidence formula is NOT implemented here. Until it is verified
# against the reference implementation, this module reports the raw maximum probability
# only and marks calibrated "confidence" as explicitly absent.
CONFIDENCE_STATUS_ABSENT = "absent_unverified_reference_formula"


@dataclass(frozen=True)
class DecisionResult:
    """The typed S1 readout for one request.

    Built from a 24-vector of per-slot scores. Inactive slots are masked *before*
    normalisation and stay exactly 0.0 in :attr:`slot_probabilities`, so no mass can
    leak between primitives. :attr:`raw_max_probability` is the maximum probability
    among the active slots; it is deliberately NOT a calibrated confidence.
    """

    kind: str
    caller_keys: tuple[str, ...]
    option_probabilities: tuple[float, ...]
    slot_probabilities: tuple[float, ...]
    selected_slot: int
    selected_key: str
    raw_max_probability: float
    p_true: float | None = None
    expected_level: float | None = None
    jev_confidence: None = None
    confidence_status: str = CONFIDENCE_STATUS_ABSENT

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_slot_probabilities(cls, request: DecisionRequest, slot_probabilities: Sequence[float]) -> "DecisionResult":
        active = _mask_and_normalise(request, slot_probabilities)
        return cls._build(request, active)

    @classmethod
    def from_slot_logits(cls, request: DecisionRequest, slot_logits: Sequence[float]) -> "DecisionResult":
        if request.kind not in KINDS:
            raise ProtocolError(f"unsupported kind {request.kind!r}")
        vals = _validate_slot_vector(slot_logits, allow_negative=True)
        idx = request.active_slot_indices()
        active_vals = [vals[i] for i in idx]
        if all(v == float("-inf") for v in active_vals):
            raise ProtocolError("all active slots are -inf; the decision is undefined")
        m = max(active_vals)
        exps = [math.exp(v - m) for v in active_vals]  # inactive slots never enter the softmax
        denom = sum(exps)
        return cls._build(request, exps, denom=denom)

    @classmethod
    def _build(cls, request: DecisionRequest, active_values: Sequence[float], denom: float | None = None) -> "DecisionResult":
        idx = request.active_slot_indices()
        keys = request.caller_keys()
        total = sum(active_values) if denom is None else denom
        probs = [v / total for v in active_values]
        slots = [0.0] * NUM_SLOTS
        for slot, p in zip(idx, probs):
            slots[slot] = p
        best = max(range(len(probs)), key=lambda i: probs[i])
        kwargs = {}
        if request.kind == KIND_NOUL:
            kwargs["p_true"] = probs[1]
        elif request.kind == KIND_SCORE:
            kwargs["expected_level"] = sum(i * p for i, p in enumerate(probs))
        return cls(
            kind=request.kind,
            caller_keys=tuple(keys),
            option_probabilities=tuple(probs),
            slot_probabilities=tuple(slots),
            selected_slot=idx[best],
            selected_key=keys[best],
            raw_max_probability=probs[best],
            **kwargs,
        )

    # ------------------------------------------------------------------ views
    def probabilities_by_key(self) -> dict[str, float]:
        return dict(zip(self.caller_keys, self.option_probabilities))

    @property
    def level_probabilities(self) -> tuple[float, ...]:
        if self.kind != KIND_SCORE:
            raise ProtocolError("level_probabilities is only defined for score results")
        return self.option_probabilities

    def as_payload(self) -> dict:
        """JSON-serialisable readout for the typed endpoint. No generated tokens."""
        body = {
            "kind": self.kind,
            "probabilities": self.probabilities_by_key(),
            "selected": self.selected_key,
            "raw_max_probability": self.raw_max_probability,
            "jev_confidence": None,
            "confidence_status": self.confidence_status,
            "output_tokens": 0,
        }
        if self.kind == KIND_NOUL:
            body["p_true"] = self.p_true
        if self.kind == KIND_SCORE:
            body["expected_level"] = self.expected_level
            body["level_probabilities"] = list(self.level_probabilities)
        return body


def _validate_slot_vector(vector: Sequence[float], allow_negative: bool) -> list[float]:
    if len(vector) != NUM_SLOTS:
        raise ProtocolError(f"slot vector must have exactly {NUM_SLOTS} entries, got {len(vector)}")
    out: list[float] = []
    for i, v in enumerate(vector):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ProtocolError(f"slot[{i}] must be a real number, got {type(v).__name__}")
        f = float(v)
        if f != f or f == float("inf") or f == float("-inf"):
            raise ProtocolError(f"slot[{i}] is non-finite: {v!r}")
        if not allow_negative and f < 0.0:
            raise ProtocolError(f"slot[{i}] is negative: {v!r}")
        out.append(f)
    return out


def _mask_and_normalise(request: DecisionRequest, slot_probabilities: Sequence[float]) -> list[float]:
    vals = _validate_slot_vector(slot_probabilities, allow_negative=False)
    active = [vals[i] for i in request.active_slot_indices()]
    total = sum(active)
    if total <= 0.0:
        raise ProtocolError(
            f"no probability mass in the active slots for kind {request.kind!r}; "
            "the inactive slots are masked and cannot carry the decision"
        )
    return active
