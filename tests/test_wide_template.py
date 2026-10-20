"""WIDE1 slice 3: the separate frozen wide chat prompt.

The legacy lettered prompt demands ONE character, so it cannot express two-letter labels. The wide
surface is a different named/versioned template with its own pinned content digest; the legacy
template bytes must stay untouched.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from qfn_decision import protocol, templates, wide

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = wide.WideCatalog.from_file(ROOT / "configs" / "wide-label-catalog.json")


def _req(n: int) -> wide.WideChoiceRequest:
    return wide.WideChoiceRequest.create(
        "The state text.",
        "Which option?",
        [protocol.ChoiceOption(key=f"k{i}", description=f"description {i}") for i in range(n)],
    )


def test_wide_template_is_named_versioned_and_not_the_legacy_one():
    assert wide.WIDE_TEMPLATE_ID == "qwen_chat_wide-v1"
    assert wide.WIDE_TEMPLATE_VERSION == "wide-v1"
    assert wide.WIDE_TEMPLATE_ID != templates.TEMPLATE_QWEN_CHAT_LETTERS
    assert wide.WIDE_CHAT_SYSTEM_PROMPT != templates.CHAT_LETTERS_SYSTEM_PROMPT
    # the legacy system prompt is never smuggled into the wide surface
    msgs = wide.wide_chat_messages(_req(17), CATALOG)
    assert templates.CHAT_LETTERS_SYSTEM_PROMPT not in json.dumps(msgs)


def test_wide_template_demands_exactly_one_supplied_label_of_one_or_two_letters():
    msgs = wide.wide_chat_messages(_req(17), CATALOG)
    assert [m["role"] for m in msgs] == ["system", "user"]
    system = msgs[0]["content"].lower()
    assert "one" in system and "letter" in system
    assert "two letters" in system, "the prompt must admit two-letter labels"
    assert "no explanation" in system
    user = msgs[1]["content"]
    assert user.startswith("The state text.")
    assert "Which option?" in user
    assert "\nOptions:\n" in user
    assert user.rstrip().endswith(wide.WIDE_CHAT_FINAL_INSTRUCTION)
    final = wide.WIDE_CHAT_FINAL_INSTRUCTION.lower()
    assert "one" in final and "letter" in final and "nothing else" in final
    assert "single character" not in final.lower()


def test_wide_template_lists_catalog_labels_in_order():
    for n, last in ((17, "Q"), (26, "Z"), (27, "AA"), (256, "JU")):
        user = wide.wide_chat_messages(_req(n), CATALOG)[1]["content"]
        option_lines = [ln for ln in user.splitlines() if ln.split(".")[0].isalpha() and ". " in ln]
        assert len(option_lines) == n
        assert option_lines[0].startswith("A. description 0")
        assert option_lines[-1].startswith(f"{last}. description {n - 1}")
        # labels come from the catalog, never from the caller keys
        assert "k0" not in user


def test_wide_template_refuses_legacy_counts_and_mismatched_catalog():
    with pytest.raises(wide.WideError):
        wide.wide_chat_messages(_req(16), CATALOG)
    with pytest.raises(wide.WideError):
        wide.wide_chat_messages(_req(17), "not a catalog")


def test_wide_template_content_digest_is_pinned_and_re_derived():
    pinned = wide.WIDE_TEMPLATE_CONTENT_SHA256
    assert len(pinned) == 64 and pinned == pinned.lower()
    assert wide.derive_template_content_digest() == pinned
    # a supplied digest that is merely well-formed hex is refused: it must equal the re-derivation
    with pytest.raises(wide.WideError):
        wide.verify_template_digest("0" * 64)
    wide.verify_template_digest(pinned)


def test_wide_template_identity_record_is_separate_from_the_legacy_block():
    req = _req(17)
    rec = wide.wide_template_identity(req, CATALOG)
    assert rec["template_id"] == wide.WIDE_TEMPLATE_ID
    assert rec["version"] == wide.WIDE_TEMPLATE_VERSION
    assert rec["content_sha256"] == wide.WIDE_TEMPLATE_CONTENT_SHA256
    assert rec["messages_sha256"] == wide.wide_messages_sha256(wide.wide_chat_messages(req, CATALOG))
    assert rec["n_options"] == 17
    assert rec["label_schema"] == wide.CATALOG_SCHEMA
    assert rec["catalog_digest"] == CATALOG.digest
    # the legacy template block names a different template and a different digest
    legacy = _legacy_request()
    assert templates.template_sha256(templates.TEMPLATE_QWEN_CHAT_LETTERS, legacy) != rec["messages_sha256"]


def _legacy_request() -> protocol.DecisionRequest:
    return protocol.DecisionRequest.create(
        "choice", "legacy state", "legacy q",
        options=[protocol.ChoiceOption(key=f"k{i}", description=f"d{i}") for i in range(16)],
    )


def test_legacy_lettered_surface_is_byte_identical():
    legacy = _legacy_request()
    msgs = templates.render_messages(legacy, templates.TEMPLATE_QWEN_CHAT_LETTERS)
    assert msgs[0]["content"] == templates.CHAT_LETTERS_SYSTEM_PROMPT
    assert msgs[1]["content"].splitlines()[-1] == templates.CHAT_LETTERS_FINAL_INSTRUCTION
    assert "A. d0" in msgs[1]["content"] and "P. d15" in msgs[1]["content"]
    digest = templates.template_sha256(templates.TEMPLATE_QWEN_CHAT_LETTERS, legacy)
    assert isinstance(digest, str) and len(digest) == 64
    # rendering is deterministic: the legacy surface is not re-generated per call
    assert templates.render_messages(legacy, templates.TEMPLATE_QWEN_CHAT_LETTERS) == msgs


def test_wide_prompt_token_hash_refuses_bad_tokenizer_output():
    req = _req(17)
    ids = list(range(100, 100 + len(json.dumps(wide.wide_chat_messages(req, CATALOG)))))
    good = wide.wide_prompt_token_sha256(ids, expected_count=len(ids))
    assert len(good) == 64
    with pytest.raises(wide.WideError):
        wide.wide_prompt_token_sha256(ids, expected_count=len(ids) + 1)
    with pytest.raises(wide.WideError):
        wide.wide_prompt_token_sha256([], expected_count=0)
    with pytest.raises(wide.WideError):
        wide.wide_prompt_token_sha256([True, 1], expected_count=2)
    with pytest.raises(wide.WideError):
        wide.wide_prompt_token_sha256([1, -1], expected_count=2)
