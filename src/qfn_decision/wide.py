"""WIDE1: additive wide-label (17..256 option) decision support.

Why this module is separate
---------------------------
The released protocol is a *fixed 24-slot* readout: ``noul`` 0:2, ``score`` 2:8 and ``choice``
8:24 (16 option slots ``A..P``). Widening that head is a different experiment. This module adds
a second, explicitly separate lane for a *single* choice readout over 17..256 caller options,
without touching a single legacy constant, template, request type, slot array or reader rule:

* the raw engine logits at 256 catalog label rows are captured (engine-side extension) and
  softmaxed over exactly the ``n`` active labels, never over the 256-row vector;
* the legacy ``<=16`` path keeps its own template (byte for byte), its 24-vector and its
  response schema;
* a wide request has its own frozen chat surface, its own request/result/record types and its
  own response schema (``qfn.decide.wide/1``).

Honest scope
------------
This build has been exercised on CPU only: catalog validation, template rendering, the capture
codec, the readout arithmetic and loopback HTTP against the synthetic fixture engine. No live
engine, quality, calibration or production claim is made here, and there is no native score or
trained-arm support for wide labels.

Catalog digest vs file hash
---------------------------
The catalog resource is pinned twice, for two different questions:

``file_sha256``   byte identity of the shipped JSON file (supply-chain pin);
``digest``        *semantic* identity of the catalog content::

    sha256(json.dumps({
        "schema": <catalog schema>,
        "tokenizer_revision": <source tokenizer revision>,
        "labels": [<ordered labels>],
        "token_ids": [<ordered token ids>],
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

A reformatted or key-reordered file keeps the same semantic digest; a changed label, order or id
never does. The engine-side helper derives the same digest independently
(``engine/qfn_capture_runtime_production.py``) and the CPU tests assert both agree, so the two
sides cannot silently drift.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import string
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from . import protocol

__all__ = [
    "WideError",
    "CATALOG_SCHEMA",
    "CATALOG_DIGEST_SCHEMA",
    "WideCatalog",
    "catalog_document",
    "catalog_digest",
    "MAX_WIDE_CHOICE_OPTIONS",
    "validate_wide_option_count",
    "WideChoiceRequest",
]

# The release keeps its own 16-option bound in ``protocol.MAX_CHOICE_OPTIONS``; this module never
# imports or mutates it.
MAX_WIDE_CHOICE_OPTIONS = 256
CATALOG_SCHEMA = "qfn.wide-label-catalog/1"
# Document version for the CONTENT digest (independent of the catalog's own schema string).
CATALOG_DIGEST_SCHEMA = "qfn.wide-catalog-content/1"
# Pinned identity of the validated, shipped catalog resource (configs/wide-label-catalog.json).
PINNED_CATALOG_FILE_SHA256 = "71a861ff84956974dd3294e20d09f86a49e323d4eff03299c30184e5517fb9bf"
PINNED_CATALOG_DIGEST = "d3aba3f02f6af23c32303e4b6617cde13da8e3c82ca7876d2c1b635450aa71d1"
PINNED_CATALOG_TOKENIZER_REVISION = "1b3c3cedc14fb3bee4294dea5e3515af74b8268a"

_HEX64 = frozenset("0123456789abcdef")
_LABEL_RE_LEN = (1, 2)


class WideError(ValueError):
    """Raised for any invalid wide-label request, catalog, capture record or readout."""


def _require_hex64(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX64 for c in value):
        raise WideError(f"{what} must be 64 lowercase hex characters, got {value!r}")
    return value


def catalog_document(payload: Mapping[str, Any]) -> dict:
    """The canonical *semantic* document of a catalog (see the module docstring)."""
    return {
        "schema": payload["schema"],
        "tokenizer_revision": payload["tokenizer_revision"],
        "labels": list(payload["labels"]),
        "token_ids": list(payload["token_ids"]),
    }


def canonical_catalog_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        catalog_document(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def catalog_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_catalog_bytes(payload)).hexdigest()


def _validate_ordered_labels(labels: Sequence[Any]) -> tuple[str, ...]:
    if not isinstance(labels, (list, tuple)) or len(labels) != MAX_WIDE_CHOICE_OPTIONS:
        got = 0 if not isinstance(labels, (list, tuple)) else len(labels)
        raise WideError(
            f"the wide catalog must carry exactly {MAX_WIDE_CHOICE_OPTIONS} labels, got {got}"
        )
    out: list[str] = []
    for i, label in enumerate(labels):
        if not isinstance(label, str):
            raise WideError(f"label[{i}] must be a string, got {type(label).__name__}")
        if len(label) not in _LABEL_RE_LEN or any(c not in string.ascii_uppercase for c in label):
            raise WideError(
                f"label[{i}] must be one or two uppercase ASCII letters, got {label!r}"
            )
        out.append(label)
    if tuple(out[:26]) != tuple(string.ascii_uppercase):
        raise WideError("the catalog must start with the 26 single-letter labels A..Z in order")
    previous: tuple[str, str] | None = None
    for i, label in enumerate(out):
        if len(label) == 1:
            if i >= 26:
                raise WideError(f"label[{i}] {label!r} is a single letter after the A..Z block")
            continue
        key = (label[0], label[1])
        if previous is not None and key <= previous:
            raise WideError(
                f"label[{i}] {label!r} breaks the strict two-letter order at {previous}"
            )
        previous = key
    if len(set(out)) != len(out):
        dupes = sorted({x for x in out if out.count(x) > 1})
        raise WideError(f"catalog labels must be unique; duplicates: {dupes}")
    return tuple(out)


@dataclass(frozen=True)
class WideCatalog:
    """A validated, immutable label catalog: ``labels[i]`` is the token id ``token_ids[i]``."""

    schema: str
    tokenizer_revision: str
    labels: tuple[str, ...]
    token_ids: tuple[int, ...]
    _digest: str = field(repr=False, compare=False)
    _document: dict = field(repr=False, compare=False, default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        tokenizer_revision: str | None = None,
        expected_digest: str | None = None,
        vocab_size: int | None = None,
    ) -> "WideCatalog":
        if not isinstance(payload, Mapping):
            raise WideError(f"catalog must be a JSON object, got {type(payload).__name__}")
        required = {"schema", "tokenizer_revision", "labels", "token_ids"}
        missing = sorted(required - set(payload))
        if missing:
            raise WideError(f"catalog is missing field(s): {missing}")
        unknown = sorted(set(payload) - required)
        if unknown:
            raise WideError(f"catalog has unknown field(s): {unknown}")
        if payload["schema"] != CATALOG_SCHEMA:
            raise WideError(
                f"unsupported catalog schema {payload['schema']!r}; this build reads {CATALOG_SCHEMA!r}"
            )
        revision = payload["tokenizer_revision"]
        if not isinstance(revision, str) or not revision.strip():
            raise WideError("catalog tokenizer_revision must be a non-empty string")
        if tokenizer_revision is not None and revision != tokenizer_revision:
            raise WideError(
                f"catalog was produced by tokenizer revision {revision!r} but the configured "
                f"serving tokenizer revision is {tokenizer_revision!r}; label token ids are "
                "not portable across revisions"
            )
        labels = _validate_ordered_labels(payload["labels"])
        raw_ids = payload["token_ids"]
        if not isinstance(raw_ids, (list, tuple)) or len(raw_ids) != len(labels):
            got = 0 if not isinstance(raw_ids, (list, tuple)) else len(raw_ids)
            raise WideError(f"catalog must carry {len(labels)} token ids, got {got}")
        ids: list[int] = []
        for i, value in enumerate(raw_ids):
            if isinstance(value, bool) or not isinstance(value, int):
                raise WideError(f"token_ids[{i}] must be an int, got {value!r}")
            if value < 0:
                raise WideError(f"token_ids[{i}] must be non-negative, got {value}")
            ids.append(int(value))
        if len(set(ids)) != len(ids):
            raise WideError("catalog token ids must be unique: two labels cannot share one row")
        if vocab_size is not None:
            bound = _validate_vocab_bound(vocab_size)
            bad = [(labels[i], ids[i]) for i in range(len(ids)) if ids[i] >= bound]
            if bad:
                raise WideError(
                    f"{len(bad)} catalog token id(s) are outside a {bound}-row vocabulary "
                    f"(first: {bad[0][0]}={bad[0][1]})"
                )
        document = catalog_document({**dict(payload), "labels": list(labels), "token_ids": ids})
        digest = hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if expected_digest is not None:
            _require_hex64(expected_digest, "expected catalog digest")
            if digest != expected_digest:
                raise WideError(
                    "catalog content digest mismatch: the validated catalog pins "
                    f"{expected_digest[:12]}..., this catalog hashes to {digest[:12]}...; "
                    "refusing to serve wide labels from a changed catalog"
                )
        return cls(
            schema=CATALOG_SCHEMA,
            tokenizer_revision=revision,
            labels=labels,
            token_ids=tuple(ids),
            _digest=digest,
            _document=document,
        )

    @classmethod
    def from_file(
        cls,
        path: str | pathlib.Path,
        *,
        tokenizer_revision: str | None = None,
        expected_digest: str | None = None,
        expected_file_sha256: str | None = None,
        vocab_size: int | None = None,
    ) -> "WideCatalog":
        target = pathlib.Path(path)
        if not target.is_file():
            raise WideError(f"wide catalog not found: {target}")
        raw = target.read_bytes()
        if expected_file_sha256 is not None:
            _require_hex64(expected_file_sha256, "expected catalog file sha256")
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected_file_sha256:
                raise WideError(
                    "catalog file byte hash mismatch: pinned "
                    f"{expected_file_sha256[:12]}..., file is {actual[:12]}..."
                )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WideError(f"catalog is not valid UTF-8 JSON: {exc}") from exc
        return cls.from_dict(
            payload,
            tokenizer_revision=tokenizer_revision,
            expected_digest=expected_digest,
            vocab_size=vocab_size,
        )

    # ------------------------------------------------------------------ views
    @property
    def digest(self) -> str:
        return self._digest

    @property
    def document(self) -> dict:
        return json.loads(json.dumps(self._document))

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    def label_for_index(self, index: int) -> str:
        if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < self.n_labels):
            raise WideError(f"label index {index!r} is outside 0..{self.n_labels - 1}")
        return self.labels[index]

    def labels_for(self, n_options: int) -> tuple[str, ...]:
        n = validate_wide_option_count(n_options)
        return self.labels[:n]

    def validate_vocab_size(self, vocab_size: int) -> None:
        bound = _validate_vocab_bound(vocab_size)
        bad = [(self.labels[i], self.token_ids[i]) for i in range(self.n_labels) if self.token_ids[i] >= bound]
        if bad:
            raise WideError(
                f"{len(bad)} catalog token id(s) are outside a {bound}-row vocabulary "
                f"(first: {bad[0][0]}={bad[0][1]})"
            )


def _validate_vocab_bound(vocab_size: Any) -> int:
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise WideError(f"vocab_size must be a positive int, got {vocab_size!r}")
    return int(vocab_size)


def validate_wide_option_count(n_options: Any) -> int:
    """A wide choice carries 17..256 options. Fewer is the legacy path; more is refused."""
    if isinstance(n_options, bool) or not isinstance(n_options, int):
        raise WideError(f"n_options must be an int, got {type(n_options).__name__}")
    if n_options < 17:
        raise WideError(
            f"{n_options} option(s) belong to the legacy <=16 path, not the wide-label lane"
        )
    if n_options > MAX_WIDE_CHOICE_OPTIONS:
        raise WideError(
            f"{n_options} options exceed the supported maximum of {MAX_WIDE_CHOICE_OPTIONS}"
        )
    return int(n_options)


# ======================================================================================
# the wide request (separate type: never a 24-slot DecisionRequest)
# ======================================================================================

WIDE_SEMANTIC_SCHEMA = "qfn.wide-semantic/1"


def _require_text(value: Any, field_name: str, max_chars: int) -> str:
    """Same rules as ``protocol._require_text`` (kept separate so the legacy module is untouched)."""
    if not isinstance(value, str):
        raise WideError(f"{field_name} must be str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise WideError(f"{field_name} must be non-empty (after stripping whitespace)")
    if len(text) > max_chars:
        raise WideError(f"{field_name} is oversized: {len(text)} chars > limit {max_chars}")
    return text


@dataclass(frozen=True)
class WideChoiceRequest:
    """One wide choice question: 17..256 ordered caller options, caller keys preserved.

    Invented for the wide lane only. It deliberately has no ``slot_range``, no slot vector and no
    score surface, so a wide readout can never be coerced into the 24-slot result type.
    """

    state: str
    question: str
    options: tuple[protocol.ChoiceOption, ...]
    limits: protocol.ProtocolLimits = field(default_factory=protocol.ProtocolLimits, repr=False, compare=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(
        cls,
        state: Any,
        question: Any,
        options: Sequence[protocol.ChoiceOption] | None,
        *,
        limits: protocol.ProtocolLimits | None = None,
        score_rubrics: Sequence[str] | None = None,
    ) -> "WideChoiceRequest":
        limits = limits or protocol.ProtocolLimits()
        if not isinstance(limits, protocol.ProtocolLimits):
            raise WideError("limits must be a protocol.ProtocolLimits instance")
        if score_rubrics:
            raise WideError(
                "a wide request is a choice only: it must not carry score_rubrics (there is no "
                "wide score surface)"
            )
        state_t = _require_text(state, "state", limits.max_state_chars)
        question_t = _require_text(question, "question", limits.max_question_chars)
        if options is None or isinstance(options, (str, bytes)) or not isinstance(options, (list, tuple)):
            raise WideError("a wide choice requires an ordered list of 17..256 {key, description} options")
        raw = list(options)
        validate_wide_option_count(len(raw))
        keys: list[str] = []
        cleaned: list[protocol.ChoiceOption] = []
        for i, option in enumerate(raw):
            if not isinstance(option, protocol.ChoiceOption):
                raise WideError(f"option[{i}] must be a ChoiceOption, got {type(option).__name__}")
            key = _require_text(option.key, f"option[{i}].key", limits.max_key_chars)
            desc = _require_text(
                option.description, f"option[{i}].description", limits.max_option_description_chars
            )
            if "\n" in desc or "\r" in desc:
                raise WideError(
                    f"option[{i}].description must be single-line (the prompt lists one option per line)"
                )
            keys.append(key)
            cleaned.append(protocol.ChoiceOption(key=key, description=desc))
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise WideError(f"duplicate option keys are not allowed: {dupes}")
        total = len(state_t) + len(question_t) + sum(len(o.description) for o in cleaned)
        if total > limits.max_total_chars:
            raise WideError(f"request is oversized: {total} total chars > limit {limits.max_total_chars}")
        return cls(state=state_t, question=question_t, options=tuple(cleaned), limits=limits)

    # ------------------------------------------------------------------ views
    @property
    def kind(self) -> str:
        return "choice"

    @property
    def n_options(self) -> int:
        return len(self.options)

    def caller_keys(self) -> tuple[str, ...]:
        return tuple(o.key for o in self.options)

    def descriptions(self) -> tuple[str, ...]:
        return tuple(o.description for o in self.options)

    def option_index_for_key(self, key: str) -> int:
        keys = self.caller_keys()
        if key not in keys:
            raise WideError(f"unknown caller key {key!r}")
        return keys.index(key)

    def semantic_document(self) -> dict:
        """Canonical semantic content: option ORDER is the label mapping, so order is identity."""
        return {
            "schema": WIDE_SEMANTIC_SCHEMA,
            "kind": self.kind,
            "state": self.state,
            "question": self.question,
            "options": [{"key": o.key, "description": o.description} for o in self.options],
        }

    def semantic_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.semantic_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()


# ======================================================================================
# the frozen wide chat surface (a SEPARATE template from qwen_chat_letters-v1)
# ======================================================================================

WIDE_TEMPLATE_ID = "qwen_chat_wide-v1"
WIDE_TEMPLATE_VERSION = "wide-v1"
WIDE_TEMPLATE_CONTENT_SCHEMA = "qfn.wide-template-content/1"
WIDE_OPTION_LINE_FORMAT = "{label}. {description}"

# The legacy lettered prompt demands "a single character", which cannot express a two-letter
# label. This surface is deliberately different and explicitly admits one OR two letters.
WIDE_CHAT_SYSTEM_PROMPT = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question "
    "and a list of labelled options, and you choose exactly one option. Every option carries a "
    "label of one or two capital letters. You reply with that option's LABEL and nothing else: "
    "the label only, one letter or two letters, no words, no punctuation, no explanation."
)
WIDE_CHAT_FINAL_INSTRUCTION = (
    "Answer with the label of exactly one option (one or two letters), and nothing else:"
)


def template_content_document() -> dict:
    """The frozen content of the wide surface, hashed to pin the prompt text itself."""
    return {
        "schema": WIDE_TEMPLATE_CONTENT_SCHEMA,
        "template_id": WIDE_TEMPLATE_ID,
        "version": WIDE_TEMPLATE_VERSION,
        "system_prompt": WIDE_CHAT_SYSTEM_PROMPT,
        "option_line_format": WIDE_OPTION_LINE_FORMAT,
        "final_instruction": WIDE_CHAT_FINAL_INSTRUCTION,
    }


def derive_template_content_digest() -> str:
    """Re-derive the prompt digest from the frozen literals (not from a stored hex string)."""
    return hashlib.sha256(
        json.dumps(
            template_content_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


# Pinned literal: editing any frozen prompt string without updating this pin fails closed.
WIDE_TEMPLATE_CONTENT_SHA256 = "ed5cfb299d0d4ad9789a7b6b267108027af9fd07e474fe9164b22fc41ad58e52"


def verify_template_digest(supplied: str) -> str:
    """Refuse a digest that is merely well-formed hex: it must equal the re-derivation."""
    _require_hex64(supplied, "wide template content digest")
    derived = derive_template_content_digest()
    if supplied != derived:
        raise WideError(
            "wide template content digest mismatch: the frozen prompt text re-derives to "
            f"{derived[:12]}..., the configuration declares {supplied[:12]}...; a changed prompt "
            "surface is refused rather than reinterpreted"
        )
    return derived


def wide_chat_messages(request: WideChoiceRequest, catalog: WideCatalog) -> list[dict]:
    """The frozen wide messages (system + user). The caller renders them with the real tokenizer."""
    if not isinstance(request, WideChoiceRequest):
        raise WideError(f"expected a WideChoiceRequest, got {type(request).__name__}")
    if not isinstance(catalog, WideCatalog):
        raise WideError(f"expected a WideCatalog, got {type(catalog).__name__}")
    labels = catalog.labels_for(request.n_options)
    lines = [
        WIDE_OPTION_LINE_FORMAT.format(label=label, description=description)
        for label, description in zip(labels, request.descriptions())
    ]
    user = "\n".join([
        request.state.rstrip(),
        "",
        request.question.rstrip(),
        "",
        "Options:",
        *lines,
        "",
        WIDE_CHAT_FINAL_INSTRUCTION,
    ])
    return [
        {"role": "system", "content": WIDE_CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def wide_messages_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise WideError("wide messages must be a non-empty sequence")
    return hashlib.sha256(
        json.dumps(list(messages), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def wide_prompt_token_sha256(token_ids: Any, *, expected_count: int) -> str:
    """Canonical token hash of the exact prompt ids the engine tokenized (count checked)."""
    from . import features

    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, (list, tuple)):
        raise WideError(f"expected a sequence of token ids, got {type(token_ids).__name__}")
    ids = list(token_ids)
    if not ids:
        raise WideError("the engine returned an empty token sequence for the wide prompt")
    if len(ids) != int(expected_count):
        raise WideError(
            f"the engine returned {len(ids)} prompt token ids but the request declares {expected_count}"
        )
    for i, value in enumerate(ids):
        if isinstance(value, bool) or not isinstance(value, int):
            raise WideError(f"prompt token[{i}] must be an int, got {value!r}")
        if value < 0:
            raise WideError(f"prompt token[{i}] must be non-negative, got {value}")
    return features.token_sequence_sha256([int(v) for v in ids])


def wide_template_identity(request: WideChoiceRequest, catalog: WideCatalog) -> dict:
    """The wide template block reported (and captured) separately from the legacy one."""
    messages = wide_chat_messages(request, catalog)
    derived = derive_template_content_digest()
    if derived != WIDE_TEMPLATE_CONTENT_SHA256:  # pragma: no cover - guarded at import use sites
        raise WideError("the frozen wide prompt text no longer matches its pinned digest")
    return {
        "template_id": WIDE_TEMPLATE_ID,
        "version": WIDE_TEMPLATE_VERSION,
        "frozen": True,
        "content_sha256": derived,
        "messages_sha256": wide_messages_sha256(messages),
        "label_schema": CATALOG_SCHEMA,
        "catalog_digest": catalog.digest,
        "n_options": request.n_options,
        "option_line_format": WIDE_OPTION_LINE_FORMAT,
    }


# ======================================================================================
# the wide readout: stable softmax over exactly the n active labels
# ======================================================================================

WIDE_DECIDE_SCHEMA = "qfn.decide.wide/1"
WIDE_CONFIDENCE_STATUS = "absent_unverified_for_wide_labels"


@dataclass(frozen=True)
class WideResult:
    """A readout over ``n`` caller options only. Deliberately not a 24-slot result."""

    caller_keys: tuple[str, ...]
    probabilities: tuple[float, ...]
    labels: tuple[str, ...]
    selected_index: int
    selected_key: str
    selected_label: str
    raw_max_probability: float
    catalog_digest: str
    n_labels: int

    @property
    def n_options(self) -> int:
        return len(self.probabilities)

    def probabilities_by_key(self) -> dict[str, float]:
        return dict(zip(self.caller_keys, self.probabilities))

    def as_payload(self) -> dict:
        return {
            "schema": WIDE_DECIDE_SCHEMA,
            "kind": "choice",
            "adaptation": {
                "mode": "wide-labels",
                "n_options": self.n_options,
                "native_choice_slots": protocol.MAX_CHOICE_OPTIONS,
                "slot_vector_used": False,
                "calibrated_confidence": False,
                "trained_arms": False,
                "readout": "stable_softmax_over_first_n_raw_label_logits",
                "note": (
                    "wide-label adaptation: the raw engine logits at the supplied label rows are "
                    "normalised over the n active labels only; the 16 native choice slots and any "
                    "trained head are not involved, and no calibration is claimed"
                ),
            },
            "probabilities": self.probabilities_by_key(),
            "selected": self.selected_key,
            "selected_label": self.selected_label,
            "raw_max_probability": self.raw_max_probability,
            "jev_confidence": None,
            "confidence_status": WIDE_CONFIDENCE_STATUS,
            "output_tokens": 0,
            "catalog": {
                "schema": CATALOG_SCHEMA,
                "digest": self.catalog_digest,
                "n_labels": self.n_labels,
                "active_labels": self.n_options,
                "labels": list(self.labels),
            },
        }


def wide_choice_result(
    logits: Any,
    request: WideChoiceRequest,
    catalog: WideCatalog,
) -> WideResult:
    """Softmax the raw logits of the FIRST ``n`` catalog labels (all 256 are captured).

    The vector must be the full catalog capture (one finite raw logit per catalog label, in catalog
    order). Only entries ``0..n-1`` enter the max/denominator, so an unused label can never carry
    mass and the caller never receives a 24-slot vector.
    """
    if not isinstance(request, WideChoiceRequest):
        raise WideError(f"expected a WideChoiceRequest, got {type(request).__name__}")
    if not isinstance(catalog, WideCatalog):
        raise WideError(f"expected a WideCatalog, got {type(catalog).__name__}")
    arr = np.asarray(logits)
    if arr.ndim != 1:
        raise WideError(f"wide logits must be one-dimensional, got shape {arr.shape}")
    if arr.shape[0] != catalog.n_labels:
        raise WideError(
            f"wide logits must cover all {catalog.n_labels} catalog labels, got {arr.shape[0]}"
        )
    if arr.dtype.kind not in ("f", "i", "u") or arr.dtype.kind == "b":
        raise WideError(f"wide logits must be real numbers, got dtype {arr.dtype}")
    values = arr.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise WideError("wide logits contain non-finite values")
    n = request.n_options
    active = values[:n]
    if np.all(np.isneginf(active)):  # unreachable after the finite check, kept explicit
        raise WideError("all active wide logits are -inf; the decision is undefined")
    peak = float(np.max(active))
    exps = np.exp(active - peak)
    denom = float(np.sum(exps))
    if not math.isfinite(denom) or denom <= 0.0:
        raise WideError("the wide softmax produced an invalid denominator")
    probs = exps / denom
    best = int(np.argmax(probs))
    keys = request.caller_keys()
    labels = catalog.labels_for(n)
    return WideResult(
        caller_keys=keys,
        probabilities=tuple(float(p) for p in probs),
        labels=tuple(labels),
        selected_index=best,
        selected_key=keys[best],
        selected_label=labels[best],
        raw_max_probability=float(probs[best]),
        catalog_digest=catalog.digest,
        n_labels=catalog.n_labels,
    )


# ======================================================================================
# explicit configuration (opt-in; absent => the legacy gateway is byte-identical)
# ======================================================================================

WIDE_CONFIG_KEYS = frozenset({
    "enabled", "catalog_path", "catalog_file_sha256", "catalog_digest",
    "tokenizer_revision", "template_digest", "max_options",
})


@dataclass(frozen=True)
class WideChoiceConfig:
    """Validated ``wide_choice`` configuration block.

    ``enabled=false`` validates only the declared types and reads no file, so an operator can
    disable the lane without shipping a catalog. ``enabled=true`` loads and pins the catalog, the
    content digest, the source tokenizer revision and the frozen prompt digest before the gateway
    opens a listener.
    """

    enabled: bool
    catalog_path: str | None = None
    catalog_file_sha256: str | None = None
    catalog_digest: str | None = None
    tokenizer_revision: str | None = None
    template_digest: str | None = None
    max_options: int = MAX_WIDE_CHOICE_OPTIONS
    catalog: WideCatalog | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        gateway_tokenizer_revision: str | None = None,
    ) -> "WideChoiceConfig":
        if not isinstance(payload, Mapping):
            raise WideError(f"wide_choice must be an object, got {type(payload).__name__}")
        unknown = sorted(set(payload) - WIDE_CONFIG_KEYS)
        if unknown:
            raise WideError(f"wide_choice has unknown field(s): {unknown}")
        if "enabled" not in payload:
            raise WideError("wide_choice requires an explicit boolean 'enabled'")
        enabled = payload["enabled"]
        if not isinstance(enabled, bool):
            raise WideError(f"wide_choice.enabled must be a boolean, got {enabled!r}")
        max_options = payload.get("max_options", MAX_WIDE_CHOICE_OPTIONS)
        if isinstance(max_options, bool) or not isinstance(max_options, int):
            raise WideError(f"wide_choice.max_options must be an int, got {max_options!r}")
        if not (17 <= max_options <= MAX_WIDE_CHOICE_OPTIONS):
            raise WideError(
                f"wide_choice.max_options must be in 17..{MAX_WIDE_CHOICE_OPTIONS}, got {max_options}"
            )
        path = payload.get("catalog_path")
        file_sha = payload.get("catalog_file_sha256")
        content_digest = payload.get("catalog_digest")
        revision = payload.get("tokenizer_revision")
        template_digest = payload.get("template_digest")
        for name, value in (("catalog_path", path), ("tokenizer_revision", revision)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise WideError(f"wide_choice.{name} must be a non-empty string, got {value!r}")
        for name, value in (("catalog_file_sha256", file_sha),
                            ("catalog_digest", content_digest),
                            ("template_digest", template_digest)):
            if value is not None:
                _require_hex64(value, f"wide_choice.{name}")

        if not enabled:
            return cls(
                enabled=False,
                catalog_path=path,
                catalog_file_sha256=file_sha,
                catalog_digest=content_digest,
                tokenizer_revision=revision,
                template_digest=template_digest,
                max_options=max_options,
                catalog=None,
            )

        missing = [n for n, v in (("catalog_path", path), ("catalog_file_sha256", file_sha),
                                  ("catalog_digest", content_digest),
                                  ("tokenizer_revision", revision),
                                  ("template_digest", template_digest)) if v is None]
        if missing:
            raise WideError(f"wide_choice.enabled=true requires {missing}")
        target = pathlib.Path(path)
        if not target.is_absolute():
            raise WideError(f"wide_choice.catalog_path must be absolute, got {path!r}")
        if gateway_tokenizer_revision is not None and gateway_tokenizer_revision != revision:
            raise WideError(
                f"wide_choice.tokenizer_revision {revision!r} disagrees with the gateway's "
                f"tokenizer.revision {gateway_tokenizer_revision!r}"
            )
        verify_template_digest(template_digest)
        catalog = WideCatalog.from_file(
            target,
            tokenizer_revision=revision,
            expected_digest=content_digest,
            expected_file_sha256=file_sha,
        )
        if max_options > catalog.n_labels:
            raise WideError(
                f"wide_choice.max_options {max_options} exceeds the {catalog.n_labels}-label catalog"
            )
        return cls(
            enabled=True,
            catalog_path=str(target),
            catalog_file_sha256=file_sha,
            catalog_digest=catalog.digest,
            tokenizer_revision=revision,
            template_digest=template_digest,
            max_options=max_options,
            catalog=catalog,
        )

    def require_catalog(self) -> WideCatalog:
        if not self.enabled or self.catalog is None:
            raise WideError(
                "the wide-label lane is not enabled: this gateway serves at most "
                f"{protocol.MAX_CHOICE_OPTIONS} choice options"
            )
        return self.catalog

    def health_block(self) -> dict:
        catalog = self.catalog
        return {
            "enabled": bool(self.enabled),
            "schema": WIDE_DECIDE_SCHEMA,
            "mode": "wide-labels",
            "n_options_min": 17,
            "n_options_max": int(self.max_options),
            "native_choice_slots": protocol.MAX_CHOICE_OPTIONS,
            "catalog": (
                None if catalog is None else {
                    "schema": catalog.schema,
                    "digest": catalog.digest,
                    "n_labels": catalog.n_labels,
                    "tokenizer_revision": catalog.tokenizer_revision,
                }
            ),
            "template": {
                "template_id": WIDE_TEMPLATE_ID,
                "version": WIDE_TEMPLATE_VERSION,
                "content_sha256": WIDE_TEMPLATE_CONTENT_SHA256,
            },
            "label_schema": CATALOG_SCHEMA,
            "calibrated_confidence_claim": False,
            "trained_arms_supported": False,
            "native_score_supported": False,
            "runtime_verified": False,
            "note": (
                "wide-label adaptation serves raw engine logits; the catalog label rows are "
                "normalised over the n active labels. Probabilities are not calibrated confidence."
            ),
        }


# ======================================================================================
# wide capture extension (NPZ arrays + an explicitly versioned scalar block)
# ======================================================================================

WIDE_CAPTURE_SCHEMA = "qfn.wide-capture/1"
WIDE_CALLER_PREFIX = "qfn-wide-"          # gateway-minted caller id in the completions body
WIDE_ENGINE_MARKER = "cmpl-qfn-wide-"     # engine-side id prefix (nested under cmpl-qfn-)
WIDE_ARRAY_NAMES = ("wide_ids", "wide_logits_fp32")
# Every scalar field of the extension. The reader requires EXACTLY this set: a missing or extra
# wide_* field is refused, never ignored.
WIDE_EXTENSION_META_FIELDS = frozenset({
    "wide_schema",
    "wide_catalog_schema",
    "wide_catalog_digest",
    "wide_tokenizer_revision",
    "wide_label_count",
    "wide_template_id",
    "wide_template_version",
    "wide_template_digest",
    "wide_n_options",
    "wide_request_uuid",
    "wide_prompt_token_sha256",
    "wide_prompt_len",
})
# Base arrays a wide record may carry (unchanged legacy names); anything else is refused.
WIDE_ALLOWED_ARRAYS = frozenset({
    "meta_json", "vector_bf16", "vector_fp32", "verbalizer_ids",
    "verbalizer_logits_fp32", "runtime_logits_row_fp32",
}) | frozenset(WIDE_ARRAY_NAMES)

_UUID32_CHARS = frozenset("0123456789abcdef")
_ENGINE_WIDE_RE = None


class WideExtensionMissing(WideError):
    """The record carries no wide extension: it is not, and must never be read as, a wide capture."""


def new_wide_caller_request_id(n_options: int) -> str:
    """Gateway-minted wide caller id: ``qfn-wide-<n>-<uuid>``. Callers never supply this id.

    The option count is part of the engine-visible id so the capture extension can assert which
    request shape it belongs to; the reader still takes ``n`` from the validated request.
    """
    import uuid as _uuid

    n = validate_wide_option_count(n_options)
    return f"{WIDE_CALLER_PREFIX}{n}-{_uuid.uuid4().hex}"


def wide_request_identity_from_engine_id(engine_id: Any) -> tuple[int, str]:
    """``(n_options, request_uuid)`` from an engine-side wide id, or refuse."""
    global _ENGINE_WIDE_RE
    if not isinstance(engine_id, str):
        raise WideError(f"engine request id must be a string, got {type(engine_id).__name__}")
    if _ENGINE_WIDE_RE is None:
        import re

        _ENGINE_WIDE_RE = re.compile(
            r"^cmpl-qfn-wide-([0-9]{1,3})-([0-9a-f]{32})(?:-[0-9]+(?:-[0-9a-f]{8})?)?$"
        )
    match = _ENGINE_WIDE_RE.match(engine_id)
    if match is None:
        raise WideError(
            f"engine request id {engine_id!r} is not a wide capture id "
            f"(expected {WIDE_ENGINE_MARKER}<n>-<32 hex>[-<index>[-<8 hex>]])"
        )
    return int(match.group(1)), match.group(2)


def wide_request_uuid_from_engine_id(engine_id: Any) -> str:
    return wide_request_identity_from_engine_id(engine_id)[1]


def wide_request_identity_from_caller_id(caller_id: Any) -> tuple[int, str]:
    """``(n_options, request_uuid)`` from the gateway-minted caller id ``qfn-wide-<n>-<uuid>``."""
    import re

    if not isinstance(caller_id, str):
        raise WideError(f"caller request id must be a string, got {type(caller_id).__name__}")
    match = re.match(r"^qfn-wide-([0-9]{1,3})-([0-9a-f]{32})$", caller_id)
    if match is None:
        raise WideError(
            f"caller request id {caller_id!r} is not a wide id (expected qfn-wide-<n>-<32 hex>)"
        )
    return int(match.group(1)), match.group(2)


def wide_n_options_from_engine_id(engine_id: Any) -> int:
    return wide_request_identity_from_engine_id(engine_id)[0]


def is_wide_engine_id(engine_id: Any) -> bool:
    try:
        wide_request_uuid_from_engine_id(engine_id)
    except WideError:
        return False
    return True


def _require_uuid32(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 32 or any(c not in _UUID32_CHARS for c in value):
        raise WideError(f"{what} must be 32 lowercase hex characters, got {value!r}")
    return value


def _require_wide_logits(logits: Any, n_labels: int) -> np.ndarray:
    arr = np.asarray(logits)
    if arr.ndim != 1 or arr.shape[0] != n_labels:
        raise WideError(f"wide logits must have shape ({n_labels},), got {arr.shape}")
    if arr.dtype.kind not in ("f", "i", "u") or arr.dtype.kind == "b":
        raise WideError(f"wide logits must be real numbers, got dtype {arr.dtype}")
    out = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    if not np.isfinite(out).all():
        raise WideError("wide logits contain non-finite values")
    return out


def _npz_load(blob: bytes) -> dict[str, np.ndarray]:
    import io

    if not isinstance(blob, (bytes, bytearray)):
        raise WideError(f"record payload must be bytes, got {type(blob).__name__}")
    try:
        with np.load(io.BytesIO(bytes(blob)), allow_pickle=False) as z:
            return {name: np.array(z[name], copy=True) for name in z.files}
    except WideError:
        raise
    except Exception as exc:  # noqa: BLE001 - any load failure is a corrupt record
        raise WideError(f"corrupt capture record: {type(exc).__name__}: {exc}") from exc


def _npz_dump(arrays: Mapping[str, np.ndarray]) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez(buf, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def _decode_meta(arrays: Mapping[str, np.ndarray]) -> dict:
    if "meta_json" not in arrays:
        raise WideError("capture record is missing the meta_json array")
    raw = bytes(np.asarray(arrays["meta_json"], dtype=np.uint8))
    try:
        meta = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WideError(f"capture metadata is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise WideError(f"capture metadata must be an object, got {type(meta).__name__}")
    return meta


def add_wide_extension(
    blob: bytes,
    *,
    catalog: WideCatalog,
    template_digest: str,
    wide_request_uuid: str,
    prompt_token_sha256: str,
    prompt_len: int,
    n_options: int,
    logits256: Any,
    ids: Any = None,
) -> bytes:
    """Attach the wide extension to an already-encoded base record.

    The 256-wide vectors become REAL npz arrays and are removed from the scalar metadata, so no
    array is ever serialised as a JSON list. The base identity fields are untouched, and the
    extension is refused unless it agrees with them.
    """
    if not isinstance(catalog, WideCatalog):
        raise WideError(f"expected a WideCatalog, got {type(catalog).__name__}")
    verify_template_digest(template_digest)
    _require_uuid32(wide_request_uuid, "wide_request_uuid")
    _require_hex64(prompt_token_sha256, "prompt_token_sha256")
    if isinstance(prompt_len, bool) or not isinstance(prompt_len, int) or prompt_len < 1:
        raise WideError(f"prompt_len must be a positive int, got {prompt_len!r}")
    n = validate_wide_option_count(n_options)
    if n > catalog.n_labels:
        raise WideError(f"n_options {n} exceeds the {catalog.n_labels}-label catalog")
    array_ids = np.asarray(catalog.token_ids if ids is None else ids)
    if array_ids.shape != (catalog.n_labels,) or array_ids.dtype.kind not in ("i", "u"):
        raise WideError(f"wide ids must be an integer array of {catalog.n_labels} catalog rows")
    if array_ids.dtype == np.bool_:
        raise WideError("wide ids must be integers, not booleans")
    if not np.array_equal(array_ids.astype(np.int64), np.asarray(catalog.token_ids, dtype=np.int64)):
        raise WideError("wide ids must be exactly the catalog token ids, in catalog order")
    logits = _require_wide_logits(logits256, catalog.n_labels)

    arrays = _npz_load(blob)
    meta = _decode_meta(arrays)
    engine_id = meta.get("engine_request_id")
    if not is_wide_engine_id(engine_id):
        raise WideError(
            f"the base record's engine_request_id {engine_id!r} is not a wide capture id; a wide "
            "extension may only be attached to a gateway-minted wide request"
        )
    if wide_request_identity_from_engine_id(engine_id) != (n, wide_request_uuid):
        raise WideError(
            "the base record's engine id does not carry this extension's option count and request uuid"
        )
    if meta.get("prompt_token_sha256") != prompt_token_sha256:
        raise WideError(
            "the wide extension prompt hash disagrees with the base record's prompt_token_sha256"
        )
    if int(meta.get("prompt_len", -1)) != prompt_len:
        raise WideError("the wide extension prompt_len disagrees with the base record's prompt_len")
    if meta.get("terminal_position") != prompt_len - 1:
        raise WideError(
            "the wide extension may only be attached to a record whose terminal position is the "
            "last prompt token"
        )
    meta.update({
        "wide_schema": WIDE_CAPTURE_SCHEMA,
        "wide_catalog_schema": CATALOG_SCHEMA,
        "wide_catalog_digest": catalog.digest,
        "wide_tokenizer_revision": catalog.tokenizer_revision,
        "wide_label_count": int(catalog.n_labels),
        "wide_template_id": WIDE_TEMPLATE_ID,
        "wide_template_version": WIDE_TEMPLATE_VERSION,
        "wide_template_digest": template_digest,
        "wide_n_options": n,
        "wide_request_uuid": wide_request_uuid,
        "wide_prompt_token_sha256": prompt_token_sha256,
        "wide_prompt_len": int(prompt_len),
    })
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"), dtype=np.uint8
    )
    arrays["wide_ids"] = np.ascontiguousarray(array_ids.astype(np.int64, copy=False))
    arrays["wide_logits_fp32"] = logits
    return _npz_dump(arrays)


def strip_wide_extension(blob: bytes) -> bytes:
    """Remove the wide extension, yielding the legacy base record (wide reader internals only)."""
    arrays = _npz_load(blob)
    meta = _decode_meta(arrays)
    for key in [k for k in meta if isinstance(k, str) and k.startswith("wide_")]:
        meta.pop(key)
    for name in WIDE_ARRAY_NAMES:
        arrays.pop(name, None)
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"), dtype=np.uint8
    )
    return _npz_dump(arrays)


@dataclass(frozen=True)
class WideCapture:
    """One validated wide capture: the base record plus the 256 ordered label logits."""

    record: Any
    wide_ids: np.ndarray
    logits: np.ndarray
    catalog_digest: str
    template_digest: str
    n_options: int
    request_uuid: str
    engine_request_id: str
    prompt_token_sha256: str
    prompt_len: int
    terminal_position: int
    tp_rank: int
    label_count: int

    def require_finite(self) -> np.ndarray:
        if not np.isfinite(self.logits).all():
            raise WideError("wide capture logits are not finite")
        return self.logits


def decode_wide_capture(
    blob: bytes,
    *,
    catalog: WideCatalog,
    expected_request_uuid: str,
    expected_engine_request_id: str,
    expected_prompt_token_sha256: str,
    expected_prompt_len: int,
    n_options: int,
    not_before_wall_ns: int | None = None,
) -> WideCapture:
    """Strict, fail-closed read of a wide capture. A missing extension never falls back."""
    import dataclasses

    from . import features

    if not isinstance(catalog, WideCatalog):
        raise WideError(f"expected a WideCatalog, got {type(catalog).__name__}")
    _require_uuid32(expected_request_uuid, "expected_request_uuid")
    _require_hex64(expected_prompt_token_sha256, "expected_prompt_token_sha256")
    n = validate_wide_option_count(n_options)
    arrays = _npz_load(blob)
    meta = _decode_meta(arrays)

    present = {key for key in meta if isinstance(key, str) and key.startswith("wide_")}
    if not present:
        raise WideExtensionMissing(
            "capture record carries no wide extension (no wide_* metadata); refusing to read it as "
            "a wide capture"
        )
    missing = sorted(WIDE_EXTENSION_META_FIELDS - present)
    unknown = sorted(present - WIDE_EXTENSION_META_FIELDS)
    if missing or unknown:
        raise WideError(
            "wide extension metadata is not the exact known schema "
            f"(missing={missing}, unknown={unknown})"
        )
    unknown_arrays = sorted(set(arrays) - WIDE_ALLOWED_ARRAYS)
    if unknown_arrays:
        raise WideError(f"wide capture carries unknown array(s): {unknown_arrays}")
    for name in WIDE_ARRAY_NAMES:
        if name not in arrays:
            raise WideError(f"wide capture is missing the {name!r} array")

    if meta["wide_schema"] != WIDE_CAPTURE_SCHEMA:
        raise WideError(
            f"unsupported wide capture schema {meta['wide_schema']!r}; "
            f"this build reads {WIDE_CAPTURE_SCHEMA!r}"
        )
    if meta["wide_catalog_schema"] != CATALOG_SCHEMA:
        raise WideError(f"unexpected wide label schema {meta['wide_catalog_schema']!r}")
    if meta["wide_tokenizer_revision"] != catalog.tokenizer_revision:
        raise WideError(
            f"wide capture was produced by tokenizer revision {meta['wide_tokenizer_revision']!r} "
            f"but the configured catalog is {catalog.tokenizer_revision!r}"
        )
    if meta["wide_catalog_digest"] != catalog.digest:
        raise WideError(
            "wide capture catalog digest does not match the configured catalog "
            f"({str(meta['wide_catalog_digest'])[:12]}... vs {catalog.digest[:12]}...)"
        )
    if meta["wide_template_id"] != WIDE_TEMPLATE_ID or meta["wide_template_version"] != WIDE_TEMPLATE_VERSION:
        raise WideError(
            f"wide capture template identity {meta['wide_template_id']!r}/"
            f"{meta['wide_template_version']!r} is not "
            f"{WIDE_TEMPLATE_ID!r}/{WIDE_TEMPLATE_VERSION!r}"
        )
    recorded_template_digest = meta["wide_template_digest"]
    if recorded_template_digest != WIDE_TEMPLATE_CONTENT_SHA256:
        raise WideError(
            "wide capture template digest does not match the pinned prompt content "
            f"({str(recorded_template_digest)[:12]}... vs {WIDE_TEMPLATE_CONTENT_SHA256[:12]}...)"
        )
    verify_template_digest(recorded_template_digest)
    if meta["wide_label_count"] != catalog.n_labels:
        raise WideError(
            f"wide capture declares {meta['wide_label_count']!r} labels; "
            f"the catalog has {catalog.n_labels}"
        )
    _require_uuid32(meta["wide_request_uuid"], "wide capture request uuid")
    if meta["wide_request_uuid"] != expected_request_uuid:
        raise WideError("wide capture request uuid does not match this request (replayed record?)")
    if int(meta["wide_n_options"]) != n:
        raise WideError(
            f"wide capture was taken for {meta['wide_n_options']!r} options but this request "
            f"carries {n}"
        )
    if meta["wide_prompt_token_sha256"] != expected_prompt_token_sha256:
        raise WideError("wide capture prompt token hash does not match the prompt this request sent")
    if meta["wide_prompt_len"] != int(expected_prompt_len):
        raise WideError(
            f"wide capture prompt_len {meta['wide_prompt_len']!r} != requested {expected_prompt_len}"
        )
    if meta.get("prompt_token_sha256") != expected_prompt_token_sha256:
        raise WideError("the base record's prompt_token_sha256 disagrees with the wide extension")
    if int(meta.get("prompt_len", -1)) != int(expected_prompt_len):
        raise WideError("the base record's prompt_len disagrees with the wide extension")

    ids = np.asarray(arrays["wide_ids"])
    if ids.dtype != np.int64:
        raise WideError(f"wide_ids must be int64, got {ids.dtype}")
    if ids.shape != (catalog.n_labels,):
        raise WideError(f"wide_ids must have shape ({catalog.n_labels},), got {ids.shape}")
    if not np.array_equal(ids, np.asarray(catalog.token_ids, dtype=np.int64)):
        raise WideError("wide_ids are not exactly the catalog token ids in catalog order")
    logits = np.asarray(arrays["wide_logits_fp32"])
    if logits.dtype != np.float32:
        raise WideError(f"wide_logits_fp32 must be float32, got {logits.dtype}")
    if logits.shape != (catalog.n_labels,):
        raise WideError(
            f"wide_logits_fp32 must have shape ({catalog.n_labels},), got {logits.shape}"
        )
    if not np.isfinite(logits).all():
        raise WideError("wide_logits_fp32 contains non-finite values")

    # --- base identity, through the shared FeatureRecord constructor -------------------
    base_arrays = {k: v for k, v in arrays.items() if k not in WIDE_ARRAY_NAMES}
    base_meta = {k: v for k, v in meta.items() if not (isinstance(k, str) and k.startswith("wide_"))}
    fields = {f.name for f in dataclasses.fields(features.FeatureRecord)}
    unknown_base = sorted(set(base_meta) - fields)
    if unknown_base:
        raise WideError(f"wide capture base metadata has unknown field(s): {unknown_base}")
    required_base = {"vector_bf16", "verbalizer_ids", "verbalizer_logits_fp32"}
    missing_base = sorted(required_base - set(base_arrays))
    if missing_base:
        raise WideError(f"wide capture base record is missing array(s): {missing_base}")
    kwargs = dict(base_meta)
    kwargs["vector_bf16"] = base_arrays.get("vector_bf16")
    kwargs["vector_fp32"] = base_arrays.get("vector_fp32")
    kwargs["runtime_logits_row_fp32"] = base_arrays.get("runtime_logits_row_fp32")
    kwargs["verbalizer_ids"] = [int(v) for v in np.asarray(base_arrays["verbalizer_ids"]).reshape(-1)]
    kwargs["verbalizer_logits_fp32"] = base_arrays["verbalizer_logits_fp32"]
    try:
        record = features.FeatureRecord(**kwargs)
    except features.FeatureError as exc:
        raise WideError(f"wide capture base record is invalid: {exc}") from exc

    if record.engine_request_id != expected_engine_request_id:
        raise WideError(
            f"wide capture engine_request_id {record.engine_request_id!r} != "
            f"{expected_engine_request_id!r}"
        )
    if wide_request_identity_from_engine_id(record.engine_request_id) != (n, expected_request_uuid):
        raise WideError(
            "the wide capture's engine id does not carry this request's option count and uuid "
            "(replayed record?)"
        )
    if record.prompt_token_sha256 != expected_prompt_token_sha256:
        raise WideError("wide capture base record prompt hash does not match this request")
    if record.prompt_len != int(expected_prompt_len):
        raise WideError("wide capture base record prompt_len does not match this request")
    if record.terminal_position != record.prompt_len - 1:
        raise WideError(
            f"wide capture terminal_position {record.terminal_position} is not the last prompt token "
            f"(prompt_len={record.prompt_len})"
        )
    if not record.logits_captured:
        raise WideError(
            f"wide capture record has no captured raw logits ({record.logits_absent_reason!r})"
        )
    if not record.token_hash_from_engine_state:
        raise WideError("wide capture prompt hash was not read from the engine's own request state")
    if int(record.tp_rank) != 0:
        raise WideError(f"wide capture was written by tp_rank {record.tp_rank}; only rank0 is read")
    if record.kernel_mode != "prefill":
        raise WideError(f"wide capture kernel_mode is {record.kernel_mode!r}, expected 'prefill'")
    if not isinstance(record.vector_source, str) or not record.vector_source:
        raise WideError("wide capture record carries no vector_source provenance")
    catalog.validate_vocab_size(int(record.head_org_vocab_size))
    if not_before_wall_ns is not None and int(record.capture_wall_ns) < int(not_before_wall_ns):
        raise WideError(
            f"wide capture wall time {record.capture_wall_ns} predates this request "
            f"({not_before_wall_ns}); the record is stale, not evidence for this prompt"
        )
    return WideCapture(
        record=record,
        wide_ids=ids.astype(np.int64, copy=False),
        logits=logits,
        catalog_digest=catalog.digest,
        template_digest=recorded_template_digest,
        n_options=n,
        request_uuid=expected_request_uuid,
        engine_request_id=record.engine_request_id,
        prompt_token_sha256=record.prompt_token_sha256,
        prompt_len=record.prompt_len,
        terminal_position=record.terminal_position,
        tp_rank=int(record.tp_rank),
        label_count=catalog.n_labels,
    )
