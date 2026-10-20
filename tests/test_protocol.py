"""Tests for src/qfn_decision/protocol.py (typed S1 decision protocol).

All fixtures in this file are SYNTHETIC test literals, clearly labelled as tests.
They are never benchmark evidence.
"""

import json
import math
from pathlib import Path

import pytest

from qfn_decision import protocol as P


# ---------------------------------------------------------------- slice 1: request typing/validation

def test_noul_request_builds_with_false_true_keys():
    req = P.DecisionRequest.create(kind="noul", state="the parcel arrived damaged", question="is the customer asking for a refund?")
    assert req.kind == "noul"
    assert req.caller_keys() == ("false", "true")
    assert req.active_slot_indices() == [0, 1]


def test_request_rejects_empty_or_whitespace_question():
    for q in ("", "   ", "\n\t "):
        with pytest.raises(P.ProtocolError):
            P.DecisionRequest.create(kind="noul", state="s", question=q)


def test_request_rejects_empty_or_whitespace_state():
    for s in ("", "  "):
        with pytest.raises(P.ProtocolError):
            P.DecisionRequest.create(kind="noul", state=s, question="q?")


def test_request_rejects_unknown_kind():
    with pytest.raises(P.ProtocolError) as e:
        P.DecisionRequest.create(kind="multichoice", state="s", question="q?")
    assert "kind" in str(e.value)


def test_request_rejects_non_string_fields():
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="noul", state=123, question="q?")
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="noul", state="s", question=["q?"])


def test_choice_request_requires_two_to_sixteen_unique_keyed_options():
    opts = [P.ChoiceOption(key="billing", description="Payments, invoices, refunds"),
            P.ChoiceOption(key="technical", description="Bugs, outages, integrations")]
    req = P.DecisionRequest.create(kind="choice", state="duplicate payment", question="which team?", options=opts)
    assert req.caller_keys() == ("billing", "technical")
    assert req.active_slot_indices() == [8, 9]

    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="choice", state="s", question="q?", options=opts[:1])
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="choice", state="s", question="q?", options=opts + opts)
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="choice", state="s", question="q?",
                                 options=[P.ChoiceOption(key=f"k{i}", description="d") for i in range(17)])


def test_choice_request_rejects_duplicate_keys():
    opts = [P.ChoiceOption(key="dup", description="a"), P.ChoiceOption(key="dup", description="b")]
    with pytest.raises(P.ProtocolError) as e:
        P.DecisionRequest.create(kind="choice", state="s", question="q?", options=opts)
    assert "duplicate" in str(e.value).lower()


def test_choice_request_rejects_empty_key_or_description():
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="choice", state="s", question="q?",
                                 options=[P.ChoiceOption(key="", description="a"), P.ChoiceOption(key="b", description="d")])
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="choice", state="s", question="q?",
                                 options=[P.ChoiceOption(key="a", description="  "), P.ChoiceOption(key="b", description="d")])


def test_score_request_requires_exactly_six_ordered_rubrics():
    rubrics = ("useless", "poor", "weak", "fair", "good", "excellent")
    req = P.DecisionRequest.create(kind="score", state="s", question="rate it", score_rubrics=rubrics)
    assert req.caller_keys() == ("0", "1", "2", "3", "4", "5")
    assert req.active_slot_indices() == [2, 3, 4, 5, 6, 7]

    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="score", state="s", question="q?", score_rubrics=rubrics[:5])
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="score", state="s", question="q?", score_rubrics=rubrics + ("extra",))
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="score", state="s", question="q?", score_rubrics=("a", "b", "c", "d", "e", " "))


def test_request_rejects_missing_kind_specific_payload():
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="choice", state="s", question="q?")
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="score", state="s", question="q?")
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="noul", state="s", question="q?",
                                 options=[P.ChoiceOption(key="a", description="b")])


def test_request_rejects_oversized_input():
    limits = P.ProtocolLimits()
    with pytest.raises(P.ProtocolError) as e:
        P.DecisionRequest.create(kind="noul", state="x" * (limits.max_state_chars + 1), question="q?")
    assert "state" in str(e.value)
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="noul", state="s", question="q" * (limits.max_question_chars + 1))
    with pytest.raises(P.ProtocolError):
        P.DecisionRequest.create(kind="choice", state="s", question="q?",
                                 options=[P.ChoiceOption(key="a", description="d" * (limits.max_option_description_chars + 1)),
                                          P.ChoiceOption(key="b", description="d")])


def test_slot_layout_matches_reference_judge_config():
    assert P.NUM_SLOTS == 24
    assert P.SLOT_RANGES == {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}
    assert P.SCORE_LEVEL_COUNT == 6
    assert P.MIN_CHOICE_OPTIONS == 2 and P.MAX_CHOICE_OPTIONS == 16
    assert P.CHOICE_LETTERS == "ABCDEFGHIJKLMNOP"


def test_choice_option_index_maps_caller_key_without_alignment_bug():
    opts = [P.ChoiceOption(key="severity", description="s"), P.ChoiceOption(key="spread", description="d"),
            P.ChoiceOption(key="latency", description="l")]
    req = P.DecisionRequest.create(kind="choice", state="st", question="q?", options=opts)
    # slot 8 is the first choice slot
    assert req.key_for_slot(8) == "severity"
    assert req.key_for_slot(10) == "latency"
    assert req.slot_for_key("latency") == 10
    with pytest.raises(P.ProtocolError):
        req.key_for_slot(11)          # inactive slot
    with pytest.raises(P.ProtocolError):
        req.slot_for_key("nope")


# ---------------------------------------------------------------- slice 2: typed targets

def test_target_accepts_normalized_distributions_per_kind():
    noul = P.DecisionTarget.create(kind="noul", probabilities=[0.11, 0.89])
    assert noul.p_true == pytest.approx(0.89)
    assert noul.n_outcomes == 2

    score = P.DecisionTarget.create(kind="score", probabilities=[0.01, 0.01, 0.05, 0.41, 0.51, 0.01])
    assert score.expected_level == pytest.approx(0 * .01 + 1 * .01 + 2 * .05 + 3 * .41 + 4 * .51 + 5 * .01)
    assert score.level_distribution[4] == pytest.approx(0.51)

    choice = P.DecisionTarget.create(kind="choice", probabilities=[0.25, 0.13, 0.62, 0.0], n_options=4)
    assert choice.n_outcomes == 4
    assert choice.top_index == 2


def test_target_rejects_wrong_length():
    with pytest.raises(P.ProtocolError):
        P.DecisionTarget.create(kind="noul", probabilities=[1 / 3, 1 / 3, 1 / 3])
    with pytest.raises(P.ProtocolError):
        P.DecisionTarget.create(kind="score", probabilities=[0.2] * 5)
    with pytest.raises(P.ProtocolError):
        P.DecisionTarget.create(kind="choice", probabilities=[0.5, 0.5], n_options=3)
    with pytest.raises(P.ProtocolError):
        P.DecisionTarget.create(kind="choice", probabilities=[0.2] * 4)  # n_options required


def test_target_rejects_nonfinite_and_negative_probabilities():
    for bad in ([float("nan"), 0.5], [float("inf"), 0.5], [1.2, -0.2], [-1e-9, 1.0 + 1e-9]):
        with pytest.raises(P.ProtocolError):
            P.DecisionTarget.create(kind="noul", probabilities=bad)


def test_target_rejects_unnormalized_or_zero_sum():
    with pytest.raises(P.ProtocolError):
        P.DecisionTarget.create(kind="noul", probabilities=[0.4, 0.4])
    with pytest.raises(P.ProtocolError):
        P.DecisionTarget.create(kind="score", probabilities=[0.0] * 6)


def test_target_tolerates_float_rounding_within_declared_tolerance():
    probs = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1]
    probs[0] += P.PROB_SUM_TOLERANCE / 2  # still inside the declared tolerance
    tgt = P.DecisionTarget.create(kind="score", probabilities=probs)
    assert sum(tgt.probabilities) == pytest.approx(1.0, abs=1e-9)


def test_target_keeps_uniform_labels_instead_of_excluding_them():
    tgt = P.DecisionTarget.create(kind="noul", probabilities=[0.5, 0.5])
    assert tgt.is_uniform is True
    assert tgt.uniform_value == pytest.approx(0.5)
    assert tgt.p_true == pytest.approx(0.5)


def test_target_rejects_references_to_wrong_kind_helpers():
    tgt = P.DecisionTarget.create(kind="choice", probabilities=[0.5, 0.5], n_options=2)
    with pytest.raises(P.ProtocolError):
        _ = tgt.p_true
    with pytest.raises(P.ProtocolError):
        _ = tgt.expected_level


# ---------------------------------------------------------------- slice 3: typed results and slot masking

def _noul_req():
    return P.DecisionRequest.create(kind="noul", state="the parcel arrived damaged", question="refund?")


def _choice_req():
    return P.DecisionRequest.create(
        kind="choice", state="duplicate payment reported", question="which team?",
        options=[P.ChoiceOption(key="billing", description="Payments, invoices, refunds"),
                 P.ChoiceOption(key="technical", description="Bugs, outages, integrations"),
                 P.ChoiceOption(key="other", description="Anything else")])


def _score_req():
    return P.DecisionRequest.create(
        kind="score", state="the draft claim", question="rate reliability",
        score_rubrics=("useless", "poor", "weak", "fair", "good", "excellent"))


def _slots(active_start, values):
    s = [0.0] * P.NUM_SLOTS
    for i, v in enumerate(values):
        s[active_start + i] = v
    return s


def test_result_masks_inactive_slots_and_normalizes_active_slots():
    res = P.DecisionResult.from_slot_probabilities(_noul_req(), _slots(0, [0.2, 0.8]))
    assert res.slot_probabilities[:2] == pytest.approx((0.2, 0.8))
    assert all(p == 0.0 for p in res.slot_probabilities[2:])
    assert sum(res.slot_probabilities) == pytest.approx(1.0)
    assert res.p_true == pytest.approx(0.8)
    assert res.selected_key == "true"


def test_result_ignores_leakage_from_inactive_slots():
    clean = P.DecisionResult.from_slot_probabilities(_noul_req(), _slots(0, [0.2, 0.8]))
    leaked = _slots(0, [0.2, 0.8])
    for i in range(2, P.NUM_SLOTS):
        leaked[i] = 9.0e9
    leaky = P.DecisionResult.from_slot_probabilities(_noul_req(), leaked)
    assert leaky.option_probabilities == pytest.approx(clean.option_probabilities)
    assert all(p == 0.0 for p in leaky.slot_probabilities[2:])
    assert sum(leaky.slot_probabilities) == pytest.approx(1.0)


def test_result_from_slot_logits_softmaxes_active_slots_only():
    logits = [0.0] * P.NUM_SLOTS
    logits[0] = math.log(0.25)
    logits[1] = math.log(0.75)
    logits[5] = 1.0e4          # inactive slot must be ignored entirely
    res = P.DecisionResult.from_slot_logits(_noul_req(), logits)
    assert res.p_true == pytest.approx(0.75)
    assert all(p == 0.0 for p in res.slot_probabilities[2:])


def test_result_choice_returns_caller_keys_without_alignment_bug():
    res = P.DecisionResult.from_slot_probabilities(_choice_req(), _slots(8, [0.5, 0.3, 0.2]))
    assert res.caller_keys == ("billing", "technical", "other")
    assert res.option_probabilities == pytest.approx((0.5, 0.3, 0.2))
    assert res.selected_key == "billing"
    assert res.probabilities_by_key() == pytest.approx(
        {"billing": 0.5, "technical": 0.3, "other": 0.2})
    # the 13 unused choice slots stay masked
    assert all(p == 0.0 for p in res.slot_probabilities[11:])


def test_result_score_returns_expected_level_between_zero_and_five():
    res = P.DecisionResult.from_slot_probabilities(_score_req(), _slots(2, [0.01, 0.01, 0.05, 0.41, 0.51, 0.01]))
    assert 0.0 <= res.expected_level <= 5.0
    assert res.expected_level == pytest.approx(3.43)
    assert res.level_probabilities == pytest.approx((0.01, 0.01, 0.05, 0.41, 0.51, 0.01))
    assert res.selected_key == "4"
    assert all(p == 0.0 for p in res.slot_probabilities[:2] + res.slot_probabilities[8:])


def test_result_exposes_raw_max_probability_and_refuses_unverified_jev_confidence():
    res = P.DecisionResult.from_slot_probabilities(_choice_req(), _slots(8, [0.5, 0.3, 0.2]))
    assert res.raw_max_probability == pytest.approx(0.5)
    assert res.jev_confidence is None
    assert res.confidence_status == "absent_unverified_reference_formula"


def test_result_rejects_bad_slot_vectors():
    req = _noul_req()
    with pytest.raises(P.ProtocolError):
        P.DecisionResult.from_slot_probabilities(req, [0.5, 0.5])                      # wrong length
    with pytest.raises(P.ProtocolError):
        P.DecisionResult.from_slot_probabilities(req, [float("nan")] + [0.0] * 23)
    with pytest.raises(P.ProtocolError):
        P.DecisionResult.from_slot_probabilities(req, [-0.1, 0.9] + [0.0] * 22)
    with pytest.raises(P.ProtocolError):
        P.DecisionResult.from_slot_probabilities(req, [0.0] * 24)                      # all mass in inactive slots
    with pytest.raises(P.ProtocolError):
        P.DecisionResult.from_slot_logits(req, [float("inf")] + [0.0] * 23)


# ---------------------------------------------------------------- emitted configs must match the code

def test_emitted_configs_match_code_constants():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs" / "protocol.json").read_text(encoding="utf-8"))
    assert cfg["slots"]["ranges"] == {k: list(v) for k, v in P.SLOT_RANGES.items()}
    assert cfg["slots"]["num_slots"] == P.NUM_SLOTS == 24
    assert cfg["choice"]["max_options"] == P.MAX_CHOICE_OPTIONS == 16
    assert cfg["choice"]["min_options"] == P.MIN_CHOICE_OPTIONS == 2
    assert cfg["score"]["levels"] == P.SCORE_LEVEL_COUNT == 6
    assert cfg["confidence"]["jev_confidence"] is None
    assert cfg["confidence"]["status"] == P.CONFIDENCE_STATUS_ABSENT
    assert cfg["limits"]["max_state_chars"] == P.ProtocolLimits().max_state_chars

    # Training and evaluation policy is intentionally not part of this runtime export.

    tpl = json.loads((root / "configs" / "templates.json").read_text(encoding="utf-8"))
    assert tpl["templates"]["autotrust_raw_bare-v1"]["token_ids_known"] is False
    assert tpl["reference_verbalizer_ids"]["status"] == "reference_only"
    assert tpl["tokenization"]["tokenizer_revision"] is None
