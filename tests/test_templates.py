"""Tests for src/qfn_decision/templates.py (canonical serialization + versioned templates).

Fixtures here are SYNTHETIC test literals except where explicitly quoted from the
pinned reference model card; they are never benchmark evidence.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest

from qfn_decision import protocol as P
from qfn_decision import templates as T


def _choice_req():
    return P.DecisionRequest.create(
        kind="choice", state="SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
        question="Supplier response for this scenario.",
        options=[P.ChoiceOption(key="issue_warning", description="issue_warning"),
                 P.ChoiceOption(key="renegotiate", description="renegotiate"),
                 P.ChoiceOption(key="dual_source", description="dual_source"),
                 P.ChoiceOption(key="maintain", description="maintain")])


def _noul_req():
    return P.DecisionRequest.create(
        kind="noul", state="Customer says the parcel arrived damaged and wants their money back.",
        question="Is the customer asking for a refund?")


def _score_req():
    return P.DecisionRequest.create(
        kind="score", state="A statistic in the draft attributes to a study that says the opposite.",
        question="Rate claim reliability for this scenario on a 0-5 scale.",
        score_rubrics=("counterfactually wrong", "mostly wrong", "mixed", "plausible", "solid", "verified"))


# ---------------------------------------------------------------- slice 1: canonical semantic serialization

def test_semantic_document_is_explicit_and_ordered():
    doc = T.semantic_document(_choice_req())
    assert doc["schema"] == T.SEMANTIC_SCHEMA
    assert doc["kind"] == "choice"
    assert doc["question"] == "Supplier response for this scenario."
    assert doc["options"] == [
        {"key": "issue_warning", "description": "issue_warning"},
        {"key": "renegotiate", "description": "renegotiate"},
        {"key": "dual_source", "description": "dual_source"},
        {"key": "maintain", "description": "maintain"},
    ]
    assert doc["rubrics"] == []


def test_score_semantic_document_carries_six_ordered_rubrics():
    doc = T.semantic_document(_score_req())
    assert doc["kind"] == "score"
    assert len(doc["rubrics"]) == 6
    assert doc["rubrics"][0] == "counterfactually wrong" and doc["rubrics"][5] == "verified"
    assert doc["options"] == []


def test_canonical_semantic_bytes_are_deterministic_utf8_json():
    req = _choice_req()
    b1 = T.canonical_semantic_bytes(req)
    b2 = T.canonical_semantic_bytes(req)
    assert isinstance(b1, bytes) and b1 == b2
    assert b'", "' not in b1 and b'": ' not in b1   # compact separators, no pretty printing
    assert T.canonical_semantic_sha256(req) == hashlib.sha256(b1).hexdigest()


def test_canonical_semantic_bytes_preserve_non_ascii_without_escapes():
    req = P.DecisionRequest.create(kind="noul", state="La precipitación es moderada.", question="¿Es cierto?")
    b = T.canonical_semantic_bytes(req)
    assert "precipitación".encode("utf-8") in b
    assert json.loads(b.decode("utf-8"))["state"] == "La precipitación es moderada."


def test_canonical_serialization_distinguishes_semantically_different_requests():
    base = P.DecisionRequest.create(kind="noul", state="s", question="q?")
    other_q = P.DecisionRequest.create(kind="noul", state="s", question="q2?")
    other_kind = P.DecisionRequest.create(kind="score", state="s", question="q?",
                                          score_rubrics=("a", "b", "c", "d", "e", "f"))
    order_a = P.DecisionRequest.create(kind="choice", state="s", question="q?",
                                       options=[P.ChoiceOption("x", "first"), P.ChoiceOption("y", "second")])
    order_b = P.DecisionRequest.create(kind="choice", state="s", question="q?",
                                       options=[P.ChoiceOption("y", "second"), P.ChoiceOption("x", "first")])
    hashes = {T.canonical_semantic_sha256(r) for r in (base, other_q, other_kind, order_a, order_b)}
    assert len(hashes) == 5   # option ORDER is semantic for a positional readout


# ---------------------------------------------------------------- slice 2: versioned raw + chat templates

def test_template_registry_versions_and_honest_token_id_status():
    assert T.TEMPLATE_AUTOTRUST_RAW in T.TEMPLATES and T.TEMPLATE_QWEN_CHAT in T.TEMPLATES
    raw = T.TEMPLATES[T.TEMPLATE_AUTOTRUST_RAW]
    chat = T.TEMPLATES[T.TEMPLATE_QWEN_CHAT]
    assert raw.version == "bare-v1"          # matches the reference judge_config template_version
    assert raw.token_ids_known is False
    assert chat.token_ids_known is False
    assert "autotrust/JEV-27B" in raw.provenance and "ff9b67b9" in raw.provenance
    assert "tokenizer" in chat.provenance.lower()
    assert raw.verbatim_reference is True and chat.verbatim_reference is False


def test_render_reference_exact_reproduces_the_reference_prompt_verbatim():
    # Quoted verbatim from the pinned reference card (autotrust/JEV-27B README, pinned revision
    # ff9b67b9bcb2dc5a8499d540e0c4cd10b254f3a2), not invented.
    expected = (
        "[kind] choice\n"
        "[state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.\n"
        "[question] Supplier response for this scenario.\n"
        "[options]\n"
        "A) issue_warning\n"
        "B) renegotiate\n"
        "C) dual_source\n"
        "D) maintain\n"
        "[decision]:"
    )
    assert T.render_reference_exact(_choice_req()) == expected


def test_render_reference_exact_formats_noul_and_score_like_the_reference():
    assert T.render_reference_exact(_noul_req()).endswith("[options]\nfalse\ntrue\n[decision]:")
    assert T.render_reference_exact(_score_req()).endswith(
        "[options]\n0\n1\n2\n3\n4\n5\n[decision]:")


def test_canonical_raw_render_lists_rubric_descriptions_for_score_one_per_line():
    text = T.render(_score_req())
    assert text.startswith("[kind] score\n[state] A statistic")
    assert "[options]\ncounterfactually wrong\nmostly wrong\nmixed\nplausible\nsolid\nverified\n[decision]:" in text


def test_canonical_raw_render_is_deterministic_and_exposes_a_template_hash():
    req = _choice_req()
    a = T.render(req)
    b = T.render(req)
    assert a == b
    assert T.template_sha256(T.TEMPLATE_AUTOTRUST_RAW, req) == hashlib.sha256(a.encode("utf-8")).hexdigest()


def test_render_rejects_unknown_template_id():
    with pytest.raises(T.TemplateError):
        T.render(_noul_req(), template_id="autotrust_raw_v0-invented")


def test_chat_template_returns_messages_and_never_token_ids():
    msgs = T.render_messages(_noul_req())
    assert isinstance(msgs, list) and msgs[0]["role"] == "user"
    text = msgs[0]["content"]
    assert text.startswith("[kind] noul") and T.TEMPLATE_QWEN_CHAT in T.TEMPLATES
    # the chat representation is not claimed to be a tokenizer result
    assert "token_ids" not in json.dumps(msgs)
    with pytest.raises(T.TemplateError):
        T.render_messages(_noul_req(), system_prompt="")


def test_chat_template_accepts_optional_system_prompt_but_defaults_to_none():
    msgs = T.render_messages(_noul_req(), system_prompt="Answer with the single key.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert T.render_messages(_noul_req())[0]["role"] == "user"


# ---------------------------------------------------------------- slice 3: token hash from supplied tokenizer output

def test_token_sequence_hash_accepts_supplied_tokenizer_output_shapes():
    ids = [151643, 872, 37558]
    expected = hashlib.sha256(json.dumps({"input_ids": ids}, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert T.token_sequence_sha256(ids) == expected
    assert T.token_sequence_sha256(tuple(ids)) == expected
    assert T.token_sequence_sha256(SimpleNamespace(input_ids=ids)) == expected
    assert T.token_sequence_sha256({"input_ids": ids}) == expected
    assert T.token_sequence_sha256([ids]) == expected                      # batch of one
    assert T.token_sequence_sha256(SimpleNamespace(input_ids=[ids])) == expected


def test_token_sequence_hash_rejects_non_token_input():
    for bad in ([1.5, 2], ["1", "2"], [-1, 2], [True, False], "1234", None, [[], []]):
        with pytest.raises(T.TemplateError):
            T.token_sequence_sha256(bad)


def test_feature_hash_changes_with_option_order_even_when_targets_are_permuted_consistently():
    order_a = P.DecisionRequest.create(kind="choice", state="s", question="q?",
                                       options=[P.ChoiceOption("x", "first"), P.ChoiceOption("y", "second")])
    order_b = P.DecisionRequest.create(kind="choice", state="s", question="q?",
                                       options=[P.ChoiceOption("y", "second"), P.ChoiceOption("x", "first")])
    ha = T.feature_hash(order_a, T.TEMPLATE_AUTOTRUST_RAW)
    hb = T.feature_hash(order_b, T.TEMPLATE_AUTOTRUST_RAW)
    assert ha != hb
    assert ha == T.feature_hash(order_a, T.TEMPLATE_AUTOTRUST_RAW)
    with pytest.raises(T.TemplateError):
        T.feature_hash(order_a, "not-a-template")


def test_feature_hash_incorporates_supplied_token_ids_when_present():
    req = _noul_req()
    without = T.feature_hash(req, T.TEMPLATE_AUTOTRUST_RAW)
    with_ids = T.feature_hash(req, T.TEMPLATE_AUTOTRUST_RAW, token_ids=[1, 2, 3])
    assert without != with_ids
    assert with_ids == T.feature_hash(req, T.TEMPLATE_AUTOTRUST_RAW, token_ids=[1, 2, 3])


def test_tokenization_plan_declares_one_shared_path_and_unknown_tokenizer():
    plan = T.TokenizationPlan(template_id=T.TEMPLATE_AUTOTRUST_RAW)
    d = plan.describe()
    assert d["template_id"] == T.TEMPLATE_AUTOTRUST_RAW
    assert d["semantic_schema"] == T.SEMANTIC_SCHEMA
    assert d["tokenizer_revision"] is None            # unknown until supplied from the serving checkpoint
    assert d["token_ids_known"] is False
    assert d["single_path"] is True
    assert T.TokenizationPlan(template_id="nope").describe()["valid"] is False


# ---------------------------------------------------------------- slice 4: qwen_chat_letters-v1 (J3)
#
# Arm B0-chat of the PM-frozen prompt-format ablation (reports/jevbench/PROTOCOL-ablation-chat.md,
# 2026-10-11): swanOne's letter-only system prompt and user layout adopted VERBATIM from
# blockbrain-ai/swanone-recipe (shim/typesafe_native_shim.py, MIT). The strings below are the
# PM-frozen literals of docs/briefs/J3-chat-letters-template.md; the em dash is part of them.

LETTERS_SYSTEM_PROMPT = "You are a calibration engine. You never answer in prose. You are given a state, a question and a numbered set of options, and you choose exactly one option. You reply with that option's LETTER and nothing else \u2014 a single character, no words, no punctuation, no explanation."
LETTERS_FINAL_INSTRUCTION = "Answer with the letter of exactly one option, and nothing else:"
LETTERS = "ABCDEFGHIJKLMNOP"


def _letters_choice(n, state="SKU AX-330 stock is low.", question="Supplier response?"):
    return P.DecisionRequest.create(
        kind="choice", state=state, question=question,
        options=[P.ChoiceOption(key=f"k{i}", description=f"option {i}") for i in range(n)])


def _letters_user_text(state, question, n):
    lines = [f"{LETTERS[i]}. option {i}" for i in range(n)]
    return "\n".join([state, "", question, "", "Options:", *lines, "", LETTERS_FINAL_INSTRUCTION])


def test_chat_letters_template_is_registered_frozen_and_choice_only():
    assert T.TEMPLATE_QWEN_CHAT_LETTERS == "qwen_chat_letters-v1"
    spec = T.TEMPLATES[T.TEMPLATE_QWEN_CHAT_LETTERS]
    assert spec.version == "letters-v1"
    assert spec.token_ids_known is False
    assert spec.verbatim_reference is False           # a chat representation, never a raw prompt
    assert spec.kinds == ("choice",)                  # noul and score are refused by this template
    assert "swanone-recipe" in spec.provenance and "MIT" in spec.provenance
    # the existing two templates keep their declared kind coverage and stay byte-identical
    assert T.TEMPLATES[T.TEMPLATE_AUTOTRUST_RAW].kinds is None
    assert T.TEMPLATES[T.TEMPLATE_QWEN_CHAT].kinds is None


def test_is_chat_template_distinguishes_raw_from_messages_representations():
    assert T.is_chat_template(T.TEMPLATE_AUTOTRUST_RAW) is False
    assert T.is_chat_template(T.TEMPLATE_QWEN_CHAT) is True
    assert T.is_chat_template(T.TEMPLATE_QWEN_CHAT_LETTERS) is True
    with pytest.raises(T.TemplateError):
        T.is_chat_template("not-a-template")


def test_chat_letters_system_message_is_verbatim_including_the_em_dash():
    msgs = T.render_messages(_letters_choice(2), template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == LETTERS_SYSTEM_PROMPT
    assert "\u2014" in msgs[0]["content"]          # em dash, part of the verbatim string
    assert " - " not in msgs[0]["content"]         # never silently normalised to a hyphen
    assert msgs[0]["content"].endswith("no punctuation, no explanation.")


def test_chat_letters_user_message_is_byte_exact_for_two_options():
    msgs = T.render_messages(_letters_choice(2), template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)
    assert msgs[1]["content"] == (
        "SKU AX-330 stock is low.\n"
        "\n"
        "Supplier response?\n"
        "\n"
        "Options:\n"
        "A. option 0\n"
        "B. option 1\n"
        "\n"
        "Answer with the letter of exactly one option, and nothing else:"
    )


@pytest.mark.parametrize("n", [2, 5, 16])
def test_chat_letters_option_lines_are_lettered_in_option_order(n):
    msgs = T.render_messages(_letters_choice(n), template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)
    assert msgs[1]["content"] == _letters_user_text("SKU AX-330 stock is low.", "Supplier response?", n)
    option_lines = [line for line in msgs[1]["content"].split("\n") if line[:2] in {f"{L}." for L in LETTERS}]
    assert option_lines == [f"{LETTERS[i]}. option {i}" for i in range(n)]
    assert msgs[1]["content"].endswith("\n\n" + LETTERS_FINAL_INSTRUCTION)


def test_chat_letters_strips_trailing_whitespace_on_state_and_question_only():
    req = _letters_choice(2, state="state text\n   ", question="which?  \t")
    msgs = T.render_messages(req, template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)
    assert msgs[1]["content"] == _letters_user_text("state text", "which?", 2)
    assert msgs[1]["content"].startswith("state text\n\nwhich?")


def test_chat_letters_keeps_the_choice_slot_map_unchanged():
    """Letters map to the existing choice slots 8..23 in option order: no protocol change."""
    assert P.SLOT_RANGES["choice"] == (8, 24)
    req = _letters_choice(16)
    assert req.active_slot_indices() == list(range(8, 24))
    assert T.render_messages(req, template_id=T.TEMPLATE_QWEN_CHAT_LETTERS) == \
        T.render_messages(req, template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)


@pytest.mark.parametrize("kind", ["noul", "score"])
def test_chat_letters_refuses_every_kind_that_is_not_choice(kind):
    req = _noul_req() if kind == "noul" else _score_req()
    with pytest.raises(T.TemplateError) as err:
        T.render_messages(req, template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)
    message = str(err.value)
    assert kind in message and T.TEMPLATE_QWEN_CHAT_LETTERS in message
    assert "choice" in message
    with pytest.raises(T.TemplateError):
        T.template_sha256(T.TEMPLATE_QWEN_CHAT_LETTERS, req)


def test_chat_letters_refuses_a_caller_supplied_system_prompt():
    """The letters surface carries its own frozen system message; it is never overridden."""
    with pytest.raises(T.TemplateError) as err:
        T.render_messages(_letters_choice(2), template_id=T.TEMPLATE_QWEN_CHAT_LETTERS,
                          system_prompt="Answer with the single key.")
    assert "system prompt" in str(err.value)
    assert T.render_messages(_letters_choice(2), template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)[0]["role"] == "system"


def test_chat_letters_is_not_renderable_as_a_raw_prompt():
    with pytest.raises(T.TemplateError):
        T.render(_letters_choice(2), template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)


def test_chat_letters_template_hash_is_over_the_canonical_messages():
    req = _letters_choice(3)
    msgs = T.render_messages(req, template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)
    payload = json.dumps(msgs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert T.template_sha256(T.TEMPLATE_QWEN_CHAT_LETTERS, req) == hashlib.sha256(payload).hexdigest()
    # option ORDER is semantic: a permuted request hashes differently
    permuted = P.DecisionRequest.create(
        kind="choice", state="SKU AX-330 stock is low.", question="Supplier response?",
        options=[P.ChoiceOption(key="k1", description="option 1"), P.ChoiceOption(key="k0", description="option 0"),
                 P.ChoiceOption(key="k2", description="option 2")])
    assert T.template_sha256(T.TEMPLATE_QWEN_CHAT_LETTERS, permuted) != T.template_sha256(T.TEMPLATE_QWEN_CHAT_LETTERS, req)


def test_chat_letters_messages_never_carry_token_ids():
    msgs = T.render_messages(_letters_choice(4), template_id=T.TEMPLATE_QWEN_CHAT_LETTERS)
    assert set(msgs[0]) == {"role", "content"} and set(msgs[1]) == {"role", "content"}
    assert "token_ids" not in json.dumps(msgs)
