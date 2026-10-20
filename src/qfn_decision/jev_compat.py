"""J1: Jev-compatible ``POST /v1/systemone`` front end over the EXISTING typed decision path.

Why this module exists
----------------------
Clients written against TypeSafe's System One API (``POST https://api.typesafe.ai/v1/systemone``)
send ``{model, state, questions: {<name>: {type, instructions, criteria}}}`` and expect
``{model, answers, usage}``. This module is a *thin translation layer*: every named question
becomes exactly one internal typed decision (:mod:`qfn_decision.protocol`) which the existing
:class:`qfn_decision.backend.DecisionBackend` executes unchanged (arm ``B0`` by default). Nothing
about the decision path is re-implemented here, and nothing generates text.

Ground truth for the shapes below was read from the official docs, not guessed:
``reports/jev-compat/reference/`` holds the fetched copies (api, quickstart, confidence, models,
state, primitives) with the fetch date in ``FETCH.txt``.

What is verified and what is not
--------------------------------
* The **confidence formulas** are the documented TypeSafe ones
  (https://docs.typesafe.ai/confidence#how-confidence-is-calculated) and they were *verified*
  against all 480 recorded real Jev 1.13.0 choice responses before being adopted: the documented
  Choice formula reproduces Jev's reported ``confidence`` on 480/480 within 0.0047 (< 2 dp) and
  480/480 exactly at 2 dp (``reports/jev-compat/fit-confidence.json``).
* No real Score or Noul responses were recorded, so the Score formula is implemented from the
  documentation and only checked against the worked examples the documentation itself publishes
  (see ``tests/test_jev_compat.py``). That limit is stated in ``docs/jev-compat.md``.

Divergences from Jev (deliberate, documented, never silent)
----------------------------------------------------------
* Choice accepts at most ``16`` options (the S1 head has 16 choice slots); Jev documents 255.
* Score accepts exactly ``6`` levels (the head has six ordered score slots); Jev documents 2..10.
* Unknown requested models never fail, and the gateway never claims to be Jev: the response's
  ``model`` is the configured served id, never the requested one.
* A multi-line choice rubric is a ``422`` (the internal option line is single-line by protocol).
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from . import backend, protocol, wide

__all__ = [
    "SYSTEMONE_PATH",
    "JevError",
    "JevQuestion",
    "JevPlan",
    "translate_request",
    "serialize_structured",
    "choice_confidence",
    "score_confidence",
    "served_model_id",
    "build_answer",
    "handle_systemone",
    "CHOICE_CONFIDENCE_FORMULA",
    "SCORE_CONFIDENCE_FORMULA",
    "MAX_NOUL_CRITERIA_KEYS",
]

SYSTEMONE_PATH = "/v1/systemone"

CHOICE_CONFIDENCE_FORMULA = "(p_max - 1/n) / (1 - 1/n)"
SCORE_CONFIDENCE_FORMULA = "max(0, 1 - sum_i p_i*|i-m| / MAD_unif), m = argmax_i p_i"

TOP_FIELDS = frozenset({"model", "state", "questions"})
QUESTION_FIELDS = frozenset({"type", "instructions", "criteria"})
NOUL_CRITERIA_KEYS = ("true", "false")
MAX_NOUL_CRITERIA_KEYS = len(NOUL_CRITERIA_KEYS)

# The gateway never returns a Jev model name and never claims Jev compatibility at the model level.
JEV_MODEL_NAMES = ("jev", "typesafe")


class JevError(Exception):
    """A Jev-shaped rejection. ``status``/``code`` are mapped to the gateway's error envelope.

    ``field``/``limit``/``question`` are carried through into the JSON body so a caller is told
    which field was rejected, which bound applied, and which named question it belonged to.
    """

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        question: str | None = None,
        field: str | None = None,
        limit: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message
        self.question = question
        self.field = field
        self.limit = limit

    def detail(self) -> dict:
        out: dict[str, Any] = {}
        if self.question is not None:
            out["question"] = self.question
        if self.field is not None:
            out["field"] = self.field
        if self.limit is not None:
            out["limit"] = self.limit
        return out


def _fail(status: int, code: str, message: str, **detail: Any) -> "JevError":
    raise JevError(status, code, message, **detail)


# ======================================================================================
# deterministic serialisation of Jev's structured fields
# ======================================================================================

def serialize_structured(value: Any) -> str:
    """Serialise a structured (object/array) Jev field deterministically.

    Sorted keys, compact separators, no ASCII escaping: the same logical value always produces
    the same string, so the internal prompt surface for a given Jev request is stable.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _field_text(value: Any, *, field: str, question: str | None = None) -> str | None:
    """``string | object | array`` -> text. Returns None when the value is ``null``."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return serialize_structured(value)
    _fail(
        422, "bad_field_type",
        f"{field} must be a string, an object or an array, got {type(value).__name__}",
        question=question, field=field,
    )
    return None  # pragma: no cover - _fail always raises


def _rubric_description(key: str, rubric: Any, *, field: str, question: str) -> str:
    """The internal option description for one choice criterion.

    ``key`` when the rubric is null, empty or identical to the key; otherwise ``"<key>: <rubric>"``
    (with a structured rubric serialised deterministically).
    """
    if rubric is None:
        return key
    if isinstance(rubric, str):
        text = rubric.strip()
        if not text or text == key:
            return key
        return f"{key}: {text}"
    if isinstance(rubric, (dict, list)):
        return f"{key}: {serialize_structured(rubric)}"
    _fail(
        422, "bad_field_type",
        f"{field} must be a string, an object, an array or null, got {type(rubric).__name__}",
        question=question, field=field,
    )
    return key  # pragma: no cover - _fail always raises


def _score_level_text(value: Any, *, field: str, question: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return serialize_structured(value)
    _fail(
        422, "bad_field_type",
        f"{field} must be a string, an object or an array, got "
        f"{'null' if value is None else type(value).__name__}",
        question=question, field=field,
    )
    return ""  # pragma: no cover - _fail always raises


# ======================================================================================
# request translation
# ======================================================================================

@dataclass(frozen=True)
class JevQuestion:
    """One named Jev question and the internal typed decision it maps to.

    A wide (17..256 option) choice is carried by :attr:`wide_request`, never by a fabricated
    16-slot request: the two request types are different objects on purpose.
    """

    name: str
    requested_type: str
    request: protocol.DecisionRequest | None = None
    wide_request: wide.WideChoiceRequest | None = None

    @property
    def is_wide(self) -> bool:
        return self.wide_request is not None

    @property
    def kind(self) -> str:
        if self.wide_request is not None:
            return self.wide_request.kind
        assert self.request is not None
        return self.request.kind

    @property
    def criteria_keys(self) -> tuple[str, ...]:
        """The caller-facing keys of this question, in the request's own criteria order."""
        if self.wide_request is not None:
            return self.wide_request.caller_keys()
        assert self.request is not None
        return self.request.caller_keys()


@dataclass(frozen=True)
class JevPlan:
    """A validated systemone request: the named questions plus the requested model (if any)."""

    questions: tuple[JevQuestion, ...]
    requested_model: str | None
    state_text: str

    def names(self) -> tuple[str, ...]:
        return tuple(q.name for q in self.questions)


def _check_question_fields(question: Mapping[str, Any], name: str) -> None:
    unknown = sorted(set(question) - QUESTION_FIELDS)
    if unknown:
        _fail(
            422, "unknown_field",
            f"unknown field(s) in question {name!r}: {unknown}; allowed: {sorted(QUESTION_FIELDS)}",
            question=name, field=f"questions.{name}",
        )


def _translate_one(name: str, question: Any, state_text: str, config: backend.ApiConfig) -> JevQuestion:
    if not isinstance(question, Mapping):
        _fail(
            422, "bad_field_type",
            f"questions.{name} must be an object, got {type(question).__name__}",
            question=name, field=f"questions.{name}",
        )

    _check_question_fields(question, name)

    kind = question.get("type")
    if kind is None:
        _fail(422, "missing_field", f"questions.{name}.type is required",
              question=name, field=f"questions.{name}.type")
    if not isinstance(kind, str) or kind not in protocol.KINDS:
        _fail(
            422, "unknown_question_type",
            f"questions.{name}.type must be one of {list(protocol.KINDS)}, got {kind!r}",
            question=name, field=f"questions.{name}.type",
            limit=list(protocol.KINDS),
        )

    instructions = question.get("instructions")
    if instructions is None:
        _fail(422, "missing_field", f"questions.{name}.instructions is required",
              question=name, field=f"questions.{name}.instructions")
    question_text = _field_text(instructions, field=f"questions.{name}.instructions", question=name)

    options: list[protocol.ChoiceOption] | None = None
    rubrics: list[str] | None = None

    if kind == protocol.KIND_CHOICE:
        criteria = question.get("criteria")
        if criteria is None:
            _fail(422, "missing_field", f"questions.{name}.criteria is required for a choice",
                  question=name, field=f"questions.{name}.criteria")
        if not isinstance(criteria, Mapping):
            _fail(
                422, "bad_field_type",
                f"questions.{name}.criteria must be an object mapping option -> rubric, got "
                f"{type(criteria).__name__}",
                question=name, field=f"questions.{name}.criteria",
            )
        keys = list(criteria.keys())
        max_options = min(config.max_options, protocol.MAX_CHOICE_OPTIONS)
        wide_lane = getattr(config, "wide_choice", None)
        wide_enabled = bool(wide_lane is not None and wide_lane.enabled)
        if len(keys) > max_options:
            if not wide_enabled:
                _fail(
                    422, "choice_too_many_options",
                    f"questions.{name}.criteria has {len(keys)} options; this gateway supports at most "
                    f"{max_options} (the S1 choice head has {protocol.MAX_CHOICE_OPTIONS} slots, while "
                    "Jev documents up to 255)",
                    question=name, field=f"questions.{name}.criteria", limit=max_options,
                )
            max_wide = min(int(wide_lane.max_options), wide.MAX_WIDE_CHOICE_OPTIONS)
            if len(keys) > max_wide:
                _fail(
                    422, "choice_too_many_options",
                    f"questions.{name}.criteria has {len(keys)} options; the wide-label lane supports "
                    f"at most {max_wide}",
                    question=name, field=f"questions.{name}.criteria", limit=max_wide,
                )
        if len(keys) < protocol.MIN_CHOICE_OPTIONS:
            _fail(
                422, "choice_too_few_options",
                f"questions.{name}.criteria has {len(keys)} options; a choice needs at least "
                f"{protocol.MIN_CHOICE_OPTIONS}",
                question=name, field=f"questions.{name}.criteria", limit=protocol.MIN_CHOICE_OPTIONS,
            )
        options = [
            protocol.ChoiceOption(
                key=key,
                description=_rubric_description(
                    key, rubric,
                    field=f"questions.{name}.criteria.{key}", question=name,
                ),
            )
            for key, rubric in criteria.items()
        ]
        if len(keys) > protocol.MAX_CHOICE_OPTIONS:
            # Wide lane: a SEPARATE request type. The caller keys stay arbitrary strings and the
            # option order is the label mapping.
            try:
                wide_request = wide.WideChoiceRequest.create(
                    state_text, question_text or "", options, limits=config.limits,
                )
            except wide.WideError as exc:
                _fail(
                    422, "protocol_error",
                    f"questions.{name} cannot be expressed as a wide decision: {exc}",
                    question=name, field=f"questions.{name}",
                )
            return JevQuestion(
                name=name, requested_type=kind, request=None, wide_request=wide_request,
            )

    elif kind == protocol.KIND_SCORE:
        criteria = question.get("criteria")
        if criteria is None:
            _fail(422, "missing_field", f"questions.{name}.criteria is required for a score",
                  question=name, field=f"questions.{name}.criteria")
        if not isinstance(criteria, list):
            _fail(
                422, "bad_field_type",
                f"questions.{name}.criteria must be an ordered array of level descriptions, got "
                f"{type(criteria).__name__}",
                question=name, field=f"questions.{name}.criteria",
            )
        if len(criteria) != protocol.SCORE_LEVEL_COUNT:
            _fail(
                422, "score_level_count",
                f"questions.{name}.criteria has {len(criteria)} levels; this gateway supports "
                f"exactly {protocol.SCORE_LEVEL_COUNT} (the S1 score head has "
                f"{protocol.SCORE_LEVEL_COUNT} ordered slots, while Jev documents 2..10)",
                question=name, field=f"questions.{name}.criteria", limit=protocol.SCORE_LEVEL_COUNT,
            )
        rubrics = [
            _score_level_text(value, field=f"questions.{name}.criteria[{i}]", question=name)
            for i, value in enumerate(criteria)
        ]

    else:  # noul
        criteria = question.get("criteria")
        if criteria is not None:
            if not isinstance(criteria, Mapping):
                _fail(
                    422, "bad_field_type",
                    f"questions.{name}.criteria must be an object with optional 'true'/'false' "
                    f"descriptions, got {type(criteria).__name__}",
                    question=name, field=f"questions.{name}.criteria",
                )
            unknown = sorted(set(criteria) - set(NOUL_CRITERIA_KEYS))
            if unknown:
                _fail(
                    422, "unknown_field",
                    f"unknown key(s) in questions.{name}.criteria: {unknown}; a noul accepts "
                    f"{list(NOUL_CRITERIA_KEYS)}",
                    question=name, field=f"questions.{name}.criteria",
                    limit=list(NOUL_CRITERIA_KEYS),
                )
            parts: list[str] = []
            for key in NOUL_CRITERIA_KEYS:
                text = _field_text(
                    criteria.get(key), field=f"questions.{name}.criteria.{key}", question=name
                )
                if text is not None and text.strip():
                    parts.append(f"{'yes' if key == 'true' else 'no'} means: {text.strip()}")
            if parts:
                # the internal noul surface is fixed (false/true slots); Jev's optional criteria
                # text is folded into the single question string so no caller content is dropped
                question_text = f"{question_text} ({'; '.join(parts)})"

    try:
        request = protocol.DecisionRequest.create(
            kind=kind,
            state=state_text,
            question=question_text,
            options=options,
            score_rubrics=rubrics,
            limits=config.limits,
        )
    except protocol.ProtocolError as exc:
        _fail(
            422, "protocol_error",
            f"questions.{name} cannot be expressed as a typed decision: {exc}",
            question=name, field=f"questions.{name}",
        )
    return JevQuestion(name=name, requested_type=kind, request=request)


def translate_request(payload: Mapping[str, Any], config: backend.ApiConfig) -> JevPlan:
    """Validate a Jev ``systemone`` body and translate it into typed decision requests.

    Raises :class:`JevError` (422) for anything the internal typed path cannot represent; it never
    truncates, pads, or silently drops a caller-supplied field.
    """
    if not isinstance(payload, Mapping):
        _fail(400, "not_an_object", "request body must be a JSON object")

    unknown = sorted(set(payload) - TOP_FIELDS)
    if unknown:
        _fail(
            422, "unknown_field",
            f"unknown request field(s): {unknown}; allowed: {sorted(TOP_FIELDS)}",
            field="body", limit=sorted(TOP_FIELDS),
        )

    requested_model: str | None = None
    if "model" in payload and payload["model"] is not None:
        model = payload["model"]
        if not isinstance(model, str) or not model.strip():
            _fail(422, "bad_field_type", "model must be a non-empty string",
                  field="model")
        # every model value is accepted, including 'jev-latest' / 'jev-1.13.0': the requested name
        # is never used to pick an arm or a template, and the response never echoes it as our own id
        requested_model = model.strip()

    if "state" not in payload or payload["state"] is None:
        _fail(422, "missing_field", "state is required", field="state")
    state_text = _field_text(payload["state"], field="state")
    if state_text is None or not state_text.strip():
        _fail(422, "missing_field", "state must not be empty", field="state")

    questions = payload.get("questions")
    if questions is None:
        _fail(422, "missing_field", "questions is required", field="questions")
    if not isinstance(questions, Mapping):
        _fail(422, "bad_field_type",
              f"questions must be an object mapping name -> question, got {type(questions).__name__}",
              field="questions")
    if not questions:
        _fail(422, "missing_field", "questions must contain at least one named question",
              field="questions")

    translated: list[JevQuestion] = []
    for name, question in questions.items():
        name_text = str(name)
        if not name_text.strip():
            _fail(422, "missing_field", "a question name must be a non-empty string",
                  field="questions")
        translated.append(_translate_one(name_text, question, state_text, config))

    return JevPlan(questions=tuple(translated), requested_model=requested_model, state_text=state_text)


# ======================================================================================
# confidence (documented TypeSafe formulas, verified against the recorded Jev responses)
# ======================================================================================

def choice_confidence(probabilities: Sequence[float]) -> float:
    """``(p_max - 1/n) / (1 - 1/n)``: 0 on an even split, 1 on certainty.

    Documented at https://docs.typesafe.ai/confidence#choice; verified against 480/480 recorded
    Jev 1.13.0 choice answers (reports/jev-compat/fit-confidence.json). Never rounded here.
    """
    probs = [float(p) for p in probabilities]
    n = len(probs)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    return (max(probs) - 1.0 / n) / (1.0 - 1.0 / n)


def score_confidence(probabilities: Sequence[float]) -> float:
    """``max(0, 1 - sum_i p_i*|i-m| / MAD_unif)`` over ordered levels.

    Documented at https://docs.typesafe.ai/confidence#score. Levels are ordered, so distance from
    the most likely level matters: neighbouring levels cost less confidence than distant ones.
    """
    probs = [float(p) for p in probabilities]
    n = len(probs)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    peak = max(range(n), key=lambda i: probs[i])
    spread = sum(p * abs(i - peak) for i, p in enumerate(probs))
    even_spread = sum(abs(i - (n - 1) / 2.0) for i in range(n)) / n
    if even_spread <= 0.0:  # pragma: no cover - impossible for n > 1
        return 1.0
    return max(0.0, 1.0 - spread / even_spread)


def served_model_id(config: backend.ApiConfig, arm: str) -> str:
    """The id this gateway reports as the model that performed the evaluation.

    ``<engine.model>-<arm lowercased>`` (e.g. ``qwen3.8-flash-next-uncensored-b0``): it identifies
    the configured served id and the readout lane, and it never claims to be Jev.
    """
    return f"{config.engine_model}-{arm.lower()}"


# ======================================================================================
# response building (Jev shape)
# ======================================================================================

def _argmax_first_max(values: Sequence[float]) -> int:
    """Index of the largest value; ties resolve to the earliest index (the request's order)."""
    return max(range(len(values)), key=lambda i: values[i])


def build_answer(question: JevQuestion, outcome: backend.DecisionOutcome) -> dict:
    """The Jev answer object for one question, from the internal typed readout."""
    if question.wide_request is not None:
        return _build_wide_answer(question, outcome)
    result = outcome.result
    if result.kind != question.kind:
        raise backend.BackendError(
            f"internal kind mismatch for question {question.name!r}: expected {question.kind!r}, "
            f"got {result.kind!r}"
        )
    keys = list(result.caller_keys)
    probs = [float(p) for p in result.option_probabilities]

    if question.kind == protocol.KIND_NOUL:
        # Jev returns no confidence for a noul: the probability carries the uncertainty
        return {"type": "noul", "noul": float(result.p_true)}

    probabilities = {key: prob for key, prob in zip(keys, probs)}

    if question.kind == protocol.KIND_CHOICE:
        return {
            "type": "choice",
            "choice": keys[_argmax_first_max(probs)],
            "confidence": choice_confidence(probs),
            "probabilities": probabilities,
        }

    # score: the documented answer carries a probability-weighted level value plus its legend
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs)),
        "confidence": score_confidence(probs),
        "legend": {key: desc for key, desc in zip(keys, question.request.score_rubrics)},
        "probabilities": probabilities,
    }


def _build_wide_answer(question: JevQuestion, outcome: Any) -> dict:
    """The Jev answer for a wide choice, with explicit wide adaptation/provenance metadata.

    The answer keeps Jev's documented shape (``choice`` / ``confidence`` / ``probabilities``) and
    adds a ``wide`` block so no caller can mistake it for a 16-slot native readout.
    """
    request = question.wide_request
    assert request is not None
    result = outcome.result
    keys = list(result.caller_keys)
    if keys != list(request.caller_keys()):
        raise backend.BackendError(
            f"wide readout keys do not match question {question.name!r} (got {len(keys)} keys)"
        )
    probs = [float(p) for p in result.probabilities]
    capture = outcome.captured
    return {
        "type": "choice",
        "choice": keys[_argmax_first_max(probs)],
        "confidence": choice_confidence(probs),
        "probabilities": {key: prob for key, prob in zip(keys, probs)},
        "wide": {
            "adaptation": "wide-labels",
            "n_options": request.n_options,
            "native_choice_slots": protocol.MAX_CHOICE_OPTIONS,
            "slot_vector_used": False,
            "calibrated_confidence": False,
            "calibration_note": (
                "Jev's documented choice confidence formula is reported over the n wide options; "
                "no probability calibration is claimed for the wide-label adaptation"
            ),
            "raw_max_probability": float(result.raw_max_probability),
            "catalog_digest": result.catalog_digest,
            "label_schema": wide.CATALOG_SCHEMA,
            "selected_label": result.selected_label,
            "template_id": wide.WIDE_TEMPLATE_ID,
            "template_version": wide.WIDE_TEMPLATE_VERSION,
            "template_digest": wide.WIDE_TEMPLATE_CONTENT_SHA256,
            "capture_uuid": capture.capture.request_uuid,
            "prompt_token_sha256": capture.capture.prompt_token_sha256,
        },
    }


# ======================================================================================
# execution: bounded concurrency over the existing backend, fail closed
# ======================================================================================

_BACKEND_ERROR_MAP: tuple[tuple[type, int, str], ...] = (
    (backend.ArmUnavailable, 422, "arm_unavailable"),
    (backend.DeadlineExceeded, 504, "deadline_exceeded"),
    (backend.CaptureRefused, 503, backend.CAPTURE_REFUSED_CODE),
    (backend.BackendUnavailable, 503, "backend_unavailable"),
    (backend.ConfigError, 500, "config_error"),
    (backend.TransportError, 502, "backend_error"),
    (backend.CaptureError, 502, "backend_error"),
    (backend.BackendError, 502, "backend_error"),
)


def _translate_backend_error(exc: BaseException, name: str) -> JevError:
    """Map an internal decision failure to the Jev-shaped error for that named question."""
    from . import templates as _templates

    status, code = 502, "backend_error"
    if isinstance(exc, _templates.TemplateError):
        status, code = 422, "template_kind_unsupported"
    else:
        for exc_type, mapped_status, mapped_code in _BACKEND_ERROR_MAP:
            if isinstance(exc, exc_type):
                status, code = mapped_status, mapped_code
                break
    message = f"question {name!r}: {type(exc).__name__}: {exc}"
    return JevError(status, code, message, question=name, field=f"questions.{name}")


def _decide_one(
    backend_instance: backend.DecisionBackend,
    question: JevQuestion,
    *,
    arm: str,
    deadline_s: float | None,
) -> Any:
    if question.wide_request is not None:
        return backend_instance.decide_wide(
            question.wide_request, arm=arm, deadline_s=deadline_s,
        )
    return backend_instance.decide(question.request, arm=arm, deadline_s=deadline_s)


class _Fanout:
    """Shared state for one question fan-out: the work queue, results, and the first failure."""

    def __init__(self, questions: Sequence[JevQuestion]) -> None:
        self.questions = tuple(questions)
        self.names = tuple(q.name for q in questions)
        self.cond = threading.Condition()
        self.results: dict[str, backend.DecisionOutcome] = {}
        self.errors: dict[str, BaseException] = {}
        self.stop = threading.Event()
        self.completed = 0
        self._next = 0            # index of the next question to hand out (guarded by self.cond)

    def claim(self) -> JevQuestion | None:
        """Hand out the next question, or None when a failure stopped the fan-out / none remain."""
        with self.cond:
            if self.stop.is_set() or self._next >= len(self.questions):
                return None
            question = self.questions[self._next]
            self._next += 1
            return question

    def record_ok(self, name: str, outcome: backend.DecisionOutcome) -> None:
        with self.cond:
            self.results[name] = outcome
            self.completed += 1
            self.cond.notify_all()

    def record_error(self, name: str, exc: BaseException) -> None:
        with self.cond:
            self.errors[name] = exc
            self.completed += 1
            self.stop.set()      # no further question is started once one has failed
            self.cond.notify_all()

    def first_error_name(self) -> str:
        with self.cond:
            return next(name for name in self.names if name in self.errors)


def _worker(
    fanout: _Fanout,
    backend_instance: backend.DecisionBackend,
    arm: str,
    deadline_s: float | None,
    release_slot: bool,
    semaphore: threading.BoundedSemaphore | None,
) -> None:
    """One bounded worker: run questions until none are left, then give a borrowed slot back."""
    try:
        while True:
            question = fanout.claim()
            if question is None:
                return
            try:
                outcome = _decide_one(backend_instance, question, arm=arm, deadline_s=deadline_s)
            except Exception as exc:  # noqa: BLE001 - mapped by the caller, never swallowed
                fanout.record_error(question.name, exc)
                return
            fanout.record_ok(question.name, outcome)
    finally:
        if release_slot and semaphore is not None:
            try:
                semaphore.release()
            except ValueError:  # pragma: no cover - defensive
                pass


def _run_plan(
    plan: JevPlan,
    config: backend.ApiConfig,
    backend_instance: backend.DecisionBackend,
    semaphore: threading.BoundedSemaphore | None,
    on_slot_transfer: Callable[[], None] | None = None,
) -> tuple[str, dict[str, backend.DecisionOutcome]]:
    """Run every question, bounded by the gateway's inflight bound, fail-closed and fail-fast.

    Slot accounting: the calling HTTP request already holds one inflight slot, so the first worker
    runs on that slot; each additional worker is started only after a further NON-blocking slot has
    been taken from the same semaphore. A worker gives its borrowed slot back when it exits, so the
    number of decisions executing against the engine can never exceed ``concurrency.max_inflight``,
    even when the request has already answered and the abandoned decisions are still unwinding.

    Fail-closed: any failing question fails the whole request with that question's Jev-shaped error;
    no partial ``answers`` map is ever returned. Fail-fast: as soon as one question fails no further
    question is started, and the request answers immediately instead of waiting for the other
    questions' deadlines (a typed 503 refusal must not decay into a 30 s wait). Failures already
    recorded when the request answers are resolved in request order, so the reported question is
    deterministic when several questions fail together.
    """
    arm = config.default_arm
    deadline_s: float | None = None
    names = plan.names()
    questions = plan.questions

    if len(questions) == 1:
        try:
            outcome = _decide_one(backend_instance, questions[0], arm=arm, deadline_s=deadline_s)
        except Exception as exc:  # noqa: BLE001 - mapped below, never swallowed
            raise _translate_backend_error(exc, names[0]) from exc
        return arm, {names[0]: outcome}

    borrowed = 0
    budget = max(0, int(config.concurrency_max_inflight) - 1)
    if semaphore is not None:
        while borrowed < budget and (borrowed + 1) < len(questions):
            if not semaphore.acquire(blocking=False):
                break
            borrowed += 1
    workers = 1 + borrowed
    # Transfer the HTTP-owned primary slot to worker 0 BEFORE it starts. On an early failure,
    # worker 0 may still be running when the HTTP handler finishes; releasing its slot there
    # would admit another decision and violate max_inflight. The callback disables that release.
    transfer_primary = semaphore is not None and on_slot_transfer is not None
    if transfer_primary:
        on_slot_transfer()

    fanout = _Fanout(questions)
    threads = []
    for index in range(workers):
        thread = threading.Thread(
            target=_worker,
            args=(fanout, backend_instance, arm, deadline_s, index > 0 or transfer_primary, semaphore),
            name=f"qfn-systemone-{index}",
            daemon=True,           # an abandoned decision never blocks interpreter shutdown
        )
        try:
            thread.start()
        except Exception as exc:
            fanout.stop.set()
            # Slots for workers not started still belong to this function; running workers
            # return their own slots. Never strand permits on thread startup failure.
            if semaphore is not None:
                for pending in range(index, workers):
                    if pending > 0 or transfer_primary:
                        semaphore.release()
            raise _translate_backend_error(exc, names[0]) from exc
        threads.append(thread)

    with fanout.cond:
        while not fanout.errors and fanout.completed < len(questions):
            fanout.cond.wait(timeout=0.2)
            if not any(t.is_alive() for t in threads):
                break              # no worker left and work outstanding: reported as an error below

    if fanout.errors:
        name = fanout.first_error_name()
        raise _translate_backend_error(fanout.errors[name], name)

    missing = [name for name in names if name not in fanout.results]
    if missing:
        raise JevError(
            502, "backend_error",
            f"question {missing[0]!r}: internal fan-out ended without a decision "
            f"({len(missing)} unanswered)",
            question=missing[0], field=f"questions.{missing[0]}",
        )
    return arm, dict(fanout.results)


def handle_systemone(
    payload: Mapping[str, Any],
    config: backend.ApiConfig,
    backend_instance: backend.DecisionBackend,
    *,
    semaphore: threading.BoundedSemaphore | None = None,
    on_slot_transfer: Callable[[], None] | None = None,
) -> dict:
    """Full ``POST /v1/systemone`` handling below the HTTP layer: body -> Jev-shaped response."""
    plan = translate_request(payload, config)
    arm, outcomes = _run_plan(plan, config, backend_instance, semaphore, on_slot_transfer)

    answers: dict[str, dict] = {}
    input_tokens = 0
    for question in plan.questions:
        outcome = outcomes[question.name]
        answers[question.name] = build_answer(question, outcome)
        input_tokens += int(outcome.usage.get("prompt_tokens", 0))

    return {
        "model": served_model_id(config, arm),
        "answers": answers,
        # S1 returns no generated text: Jev's own zero-sampling claim, reported as 0 here too
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
