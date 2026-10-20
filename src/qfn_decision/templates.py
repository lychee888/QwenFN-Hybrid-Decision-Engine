"""Canonical templates, serialization and token hashing for typed S1 decisions.

Two versioned templates exist, and they are deliberately different things:

* ``autotrust_raw_bare-v1`` : the plain-text decision prompt. ``render_reference_exact``
  reproduces the reference prompt verbatim (noul/score options as bare words, choice as
  ``A) description`` lines) so the transcription can be checked against the pinned model
  card; ``render`` is the canonical variant used by this experiment, which lists the six
  ordered rubric descriptions for ``score`` (the experiment requires the rubric text in
  the prompt, not bare digits).
* ``qwen_chat_v1`` : a chat *representation* (messages). The string form must be produced
  by the real tokenizer's chat template from the serving checkpoint. This module never
  fabricates chat tokens or token ids.
* ``qwen_chat_letters-v1`` : a second chat representation, the letter-only readout surface of the
  PM-frozen prompt-format ablation (arm B0-chat), adopted verbatim from
  blockbrain-ai/swanone-recipe (MIT). It is CHOICE ONLY: noul and score have no verbatim lettered
  form here and are refused, so a noul question must be presented by the caller as a two-option
  lettered choice (see ``tools/jevbench/qfn_decide_adapter.py``, ``noul_mode=letters``).

Token ids only ever enter through :func:`token_sequence_sha256`, which takes tokenizer
output that the caller actually produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from .protocol import (
    CHOICE_LETTERS,
    KINDS,
    DecisionRequest,
    ProtocolError,
)

__all__ = [
    "SEMANTIC_SCHEMA",
    "TEMPLATE_AUTOTRUST_RAW",
    "TEMPLATE_QWEN_CHAT",
    "TEMPLATE_QWEN_CHAT_LETTERS",
    "CHAT_LETTERS_SYSTEM_PROMPT",
    "CHAT_LETTERS_FINAL_INSTRUCTION",
    "CHAT_LETTERS_OPTION_LETTERS",
    "TEMPLATES",
    "TemplateSpec",
    "TemplateError",
    "TokenizationPlan",
    "semantic_document",
    "canonical_semantic_bytes",
    "canonical_semantic_sha256",
    "render",
    "render_reference_exact",
    "template_sha256",
    "render_messages",
    "chat_letters_messages",
    "is_chat_template",
    "token_sequence_sha256",
    "feature_hash",
]

# Version of the canonical semantic document. Bump when the field set or its
# meaning changes; the hash of a document is only comparable within one version.
SEMANTIC_SCHEMA = "qfn.semantic/1"

TEMPLATE_AUTOTRUST_RAW = "autotrust_raw_bare-v1"
TEMPLATE_QWEN_CHAT = "qwen_chat_v1"
TEMPLATE_QWEN_CHAT_LETTERS = "qwen_chat_letters-v1"

REFERENCE_REPO = "autotrust/JEV-27B"
REFERENCE_REVISION = "ff9b67b9bcb2dc5a8499d540e0c4cd10b254f3a2"


class TemplateError(ValueError):
    """Raised when a template cannot be rendered for a request, or when supplied
    tokenizer output is not a valid token sequence."""


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    version: str
    token_ids_known: bool
    provenance: str
    verbatim_reference: bool = False
    # ``None`` means "every protocol kind" (the two original templates). A tuple restricts the
    # surface to those kinds and is enforced by :func:`render_messages`; it is deliberately NOT
    # part of :meth:`describe` so the emitted registry snapshot (configs/templates.json, an
    # immutable PM artifact of stage S1) keeps its exact shape.
    kinds: tuple[str, ...] | None = None

    def describe(self) -> dict:
        return {
            "template_id": self.template_id,
            "version": self.version,
            "token_ids_known": self.token_ids_known,
            "verbatim_reference": self.verbatim_reference,
            "provenance": self.provenance,
        }


TEMPLATES: dict[str, TemplateSpec] = {
    TEMPLATE_AUTOTRUST_RAW: TemplateSpec(
        template_id=TEMPLATE_AUTOTRUST_RAW,
        version="bare-v1",   # == judge_config.json slots.template_version at the pinned revision
        token_ids_known=False,
        provenance=(
            f"{REFERENCE_REPO}@{REFERENCE_REVISION}: verbatim prompt shape transcribed from the "
            "pinned card/code (f\"[kind] {kind}\\n[state] {state}\\n[question] {question}\\n[options]\\n\" "
            "+ lines + \"\\n[decision]:\"), slot map from judge_config.json. Token ids are NOT known "
            "for the local serving checkpoint."
        ),
        verbatim_reference=True,
    ),
    TEMPLATE_QWEN_CHAT: TemplateSpec(
        template_id=TEMPLATE_QWEN_CHAT,
        version="v1",
        token_ids_known=False,
        provenance=(
            "Chat representation only: messages must be rendered by the real tokenizer's chat "
            "template (chat_template.jinja of the pinned serving checkpoint). No token ids are "
            "assumed or synthesised in this repository."
        ),
        verbatim_reference=False,
    ),
    TEMPLATE_QWEN_CHAT_LETTERS: TemplateSpec(
        template_id=TEMPLATE_QWEN_CHAT_LETTERS,
        version="letters-v1",
        token_ids_known=False,
        provenance=(
            "Chat representation (messages) with a letter-only readout surface, adopted VERBATIM "
            "from blockbrain-ai/swanone-recipe (shim/typesafe_native_shim.py, MIT licence; "
            "https://github.com/blockbrain-ai/swanone-recipe): a fixed system prompt demanding the "
            "option LETTER and nothing else, then state / question / 'Options:' / 'A. <description>' "
            "lines / 'Answer with the letter of exactly one option, and nothing else:'. Not tuned on "
            "any JevBench item. CHOICE ONLY: noul is presented by the caller as a two-option lettered "
            "choice and score has no lettered form, so both are refused here. The letters map to the "
            "existing choice slots 8..23 in option order (no protocol change). Messages must still be "
            "rendered by the real tokenizer's chat template; no token ids are assumed or synthesised."
        ),
        verbatim_reference=False,
        kinds=("choice",),
    ),
}


# Verbatim literals of the qwen_chat_letters-v1 surface (PM-frozen in
# reports/jevbench/PROTOCOL-ablation-chat.md and docs/briefs/J3-chat-letters-template.md).
# The em dash in the system prompt is part of the adopted string: it is never normalised.
CHAT_LETTERS_SYSTEM_PROMPT = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question "
    "and a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else \u2014 a single character, no words, no punctuation, no explanation."
)
CHAT_LETTERS_FINAL_INSTRUCTION = "Answer with the letter of exactly one option, and nothing else:"
CHAT_LETTERS_OPTION_LETTERS = CHOICE_LETTERS


# ---------------------------------------------------------------- canonical semantic serialization

def semantic_document(request: DecisionRequest) -> dict:
    """Canonical semantic content of a typed decision request.

    This is the *meaning* of the request, independent of any template surface form.
    Two requests with the same document are the same decision.
    """
    if not isinstance(request, DecisionRequest):
        raise TemplateError(f"expected a DecisionRequest, got {type(request).__name__}")
    return {
        "schema": SEMANTIC_SCHEMA,
        "kind": request.kind,
        "state": request.state,
        "question": request.question,
        "options": [{"key": o.key, "description": o.description} for o in request.options],
        "rubrics": list(request.score_rubrics),
    }


def canonical_semantic_bytes(request: DecisionRequest) -> bytes:
    """Deterministic UTF-8 encoding: JSON, keys sorted, compact separators, non-ASCII
    preserved literally (not escaped), no trailing newline."""
    doc = semantic_document(request)
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_semantic_sha256(request: DecisionRequest) -> str:
    return hashlib.sha256(canonical_semantic_bytes(request)).hexdigest()


# ---------------------------------------------------------------- raw prompt templates

def _reference_lines(request: DecisionRequest) -> list[str]:
    """Option lines exactly as the reference prompt builder produces them."""
    if request.kind == "choice":
        return [f"{CHOICE_LETTERS[i]}) {d}" for i, d in enumerate(request.descriptions())]
    if request.kind == "score":
        return [str(i) for i in range(len(request.score_rubrics))]
    return list(request.descriptions())          # noul: false / true


def _canonical_lines(request: DecisionRequest) -> list[str]:
    """Option lines used by this experiment: identical to the reference except that
    ``score`` prints the six ordered rubric descriptions instead of bare digits."""
    if request.kind == "choice":
        return [f"{CHOICE_LETTERS[i]}) {d}" for i, d in enumerate(request.descriptions())]
    return list(request.descriptions())          # noul: false/true ; score: six rubrics


def _assemble(request: DecisionRequest, lines: Sequence[str]) -> str:
    return (
        f"[kind] {request.kind}\n"
        f"[state] {request.state}\n"
        f"[question] {request.question}\n"
        f"[options]\n" + "\n".join(lines) + "\n[decision]:"
    )


def render_reference_exact(request: DecisionRequest) -> str:
    """Byte-for-byte reproduction of the reference prompt builder for this request."""
    if request.kind not in KINDS:
        raise TemplateError(f"unsupported kind {request.kind!r}")
    return _assemble(request, _reference_lines(request))


def render(request: DecisionRequest, template_id: str = TEMPLATE_AUTOTRUST_RAW) -> str:
    """Canonical raw prompt for training and serving (one shared path)."""
    spec = _spec(template_id)
    if spec.verbatim_reference is not True:
        raise TemplateError(
            f"{template_id} is a chat representation; use render_messages() and the real tokenizer"
        )
    return _assemble(request, _canonical_lines(request))


def template_sha256(template_id: str, request: DecisionRequest) -> str:
    """Hash of the exact rendered surface for this request (prompt text for the raw
    template, canonical message JSON for the chat template)."""
    spec = _spec(template_id)
    if spec.verbatim_reference:
        payload = render(request, template_id).encode("utf-8")
    else:
        msgs = render_messages(request, template_id=template_id)
        payload = json.dumps(msgs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_messages(
    request: DecisionRequest,
    template_id: str = TEMPLATE_QWEN_CHAT,
    system_prompt: str | None = None,
) -> list[dict]:
    """Chat representation. Returns messages; the caller renders them with the actual
    tokenizer chat template. No token ids are produced or guessed here.

    ``qwen_chat_letters-v1`` carries its own frozen system prompt and a choice-only surface, so it
    refuses a caller-supplied ``system_prompt`` and every kind but ``choice``.
    """
    spec = _spec(template_id)
    if spec.verbatim_reference:
        raise TemplateError(f"{template_id} is a raw prompt template; use render()")
    if spec.kinds is not None:
        _require_supported_kind(spec, request, template_id)
        if system_prompt is not None:
            raise TemplateError(
                f"{template_id} carries its own frozen system prompt; a caller-supplied system "
                "prompt is refused rather than merged into the adopted surface"
            )
        return chat_letters_messages(request)
    if system_prompt is not None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise TemplateError("system_prompt must be a non-empty string when supplied")
    text = render(request, TEMPLATE_AUTOTRUST_RAW)
    msgs: list[dict] = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt.strip()})
    msgs.append({"role": "user", "content": text})
    return msgs


def chat_letters_messages(request: DecisionRequest) -> list[dict]:
    """The frozen ``qwen_chat_letters-v1`` messages (system + user), verbatim.

    Adopted from blockbrain-ai/swanone-recipe (shim/typesafe_native_shim.py, MIT); the option
    LETTERS map to the existing choice slots 8..23 in option order, so the caller keys, the
    readout and the protocol are unchanged.
    """
    if not isinstance(request, DecisionRequest):
        raise TemplateError(f"expected a DecisionRequest, got {type(request).__name__}")
    _require_supported_kind(TEMPLATES[TEMPLATE_QWEN_CHAT_LETTERS], request, TEMPLATE_QWEN_CHAT_LETTERS)
    lines = [
        f"{letter}. {description}"
        for letter, description in zip(CHAT_LETTERS_OPTION_LETTERS, request.descriptions())
    ]
    user = "\n".join([
        request.state.rstrip(),
        "",
        request.question.rstrip(),
        "",
        "Options:",
        *lines,
        "",
        CHAT_LETTERS_FINAL_INSTRUCTION,
    ])
    return [
        {"role": "system", "content": CHAT_LETTERS_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _require_supported_kind(spec: TemplateSpec, request: DecisionRequest, template_id: str) -> None:
    """A restricted surface refuses any kind it has no verbatim form for (fail closed)."""
    if spec.kinds is None or request.kind in spec.kinds:
        return
    raise TemplateError(
        f"template {template_id!r} supports kind(s) {list(spec.kinds)} only, but this request is "
        f"{request.kind!r}: the lettered surface has no verbatim form for it. A noul question is "
        "presented by the caller as a two-option lettered choice; score has no lettered form."
    )


def is_chat_template(template_id: str) -> bool:
    """True when the template is a chat (messages) representation rather than a raw prompt.

    The caller must tokenize such a template through the real tokenizer's chat template
    (``messages`` + ``add_generation_prompt``), never through :func:`render`.
    """
    return _spec(template_id).verbatim_reference is not True


def _spec(template_id: str) -> TemplateSpec:
    if template_id not in TEMPLATES:
        raise TemplateError(f"unknown template id {template_id!r}; known: {sorted(TEMPLATES)}")
    return TEMPLATES[template_id]


# ---------------------------------------------------------------- token hashing from real tokenizer output

def _flatten_token_ids(token_ids: Any) -> list[int]:
    obj = token_ids
    if hasattr(obj, "input_ids"):
        obj = obj.input_ids
    elif isinstance(obj, dict):
        if "input_ids" not in obj:
            raise TemplateError("mapping input must contain an 'input_ids' entry")
        obj = obj["input_ids"]

    if isinstance(obj, (str, bytes)) or obj is None or isinstance(obj, bool):
        raise TemplateError(f"expected token integers from a tokenizer, got {type(obj).__name__}")
    if not isinstance(obj, (list, tuple)):
        raise TemplateError(f"expected a sequence of token integers, got {type(obj).__name__}")

    seq = list(obj)
    if len(seq) == 1 and isinstance(seq[0], (list, tuple)):     # batch of one
        seq = list(seq[0])
    if not seq:
        raise TemplateError("token sequence is empty")

    out: list[int] = []
    for i, t in enumerate(seq):
        if isinstance(t, bool) or not isinstance(t, int):
            raise TemplateError(f"token[{i}] must be a non-negative int, got {t!r}")
        if t < 0:
            raise TemplateError(f"token[{i}] must be non-negative, got {t!r}")
        out.append(t)
    return out


def token_sequence_sha256(token_ids: Any) -> str:
    """Exact hash of a token sequence produced by the caller's real tokenizer.

    Accepts a sequence of ints, an object with ``.input_ids`` (HF BatchEncoding),
    a mapping with ``input_ids``, or a single-element batch of any of those.
    """
    ids = _flatten_token_ids(token_ids)
    return hashlib.sha256(json.dumps({"input_ids": ids}, separators=(",", ":")).encode("utf-8")).hexdigest()


def feature_hash(request: DecisionRequest, template_id: str, token_ids: Any = None) -> str:
    """Stable identity of the *feature-bearing* input: semantic content, template
    surface, option order (expressed through the rendered surface) and, when the
    caller has real tokenizer output, the exact token sequence."""
    spec = _spec(template_id)
    parts = {
        "semantic_sha256": canonical_semantic_sha256(request),
        "template_id": template_id,
        "template_version": spec.version,
        "template_sha256": template_sha256(template_id, request),
        "token_sha256": token_sequence_sha256(token_ids) if token_ids is not None else "token_ids_absent",
    }
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TokenizationPlan:
    """Declares the one shared tokenization path intended for training and serving.

    ``tokenizer_revision`` and ``tokenizer_chat_template_sha256`` stay ``None`` until the
    real values are read off the pinned serving checkpoint; nothing is assumed here.
    """

    template_id: str
    semantic_schema: str = SEMANTIC_SCHEMA
    tokenizer_revision: str | None = None
    tokenizer_chat_template_sha256: str | None = None
    extra: dict = field(default_factory=dict, compare=False)

    def describe(self) -> dict:
        spec = TEMPLATES.get(self.template_id)
        return {
            "template_id": self.template_id,
            "template_version": None if spec is None else spec.version,
            "valid": spec is not None,
            "semantic_schema": self.semantic_schema,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_chat_template_sha256": self.tokenizer_chat_template_sha256,
            "token_ids_known": False,
            "single_path": True,
            "note": (
                "Training and serving must tokenize through this one path; token ids are "
                "recorded from the actual tokenizer, never synthesised."
            ),
        }


# ProtocolError is re-exported implicitly through the protocol module; kept imported so
# callers can catch either error type from this module's API surface.
_ = (ProtocolError,)

