"""WIDE1 slice 4: the wide readout (softmax over exactly n active labels) and its payload."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from qfn_decision import protocol, wide

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = wide.WideCatalog.from_file(ROOT / "configs" / "wide-label-catalog.json")


def _req(n: int) -> wide.WideChoiceRequest:
    return wide.WideChoiceRequest.create(
        "state", "question",
        [protocol.ChoiceOption(key=f"key{i}", description=f"d{i}") for i in range(n)],
    )


def _logits(winner: int | None = None, *, n: int = 256, base: float = 0.0) -> np.ndarray:
    arr = np.full(n, base, dtype=np.float32)
    if winner is not None:
        arr[winner] = 20.0
    return arr


def test_softmax_runs_over_exactly_n_labels_and_never_normalises_unused_ones():
    n = 17
    logits = _logits(0)        # a full 256-label raw vector, as the engine captures it
    logits[200] = 1e4          # an unused catalog label with an enormous logit must be ignored
    result = wide.wide_choice_result(logits, _req(n), CATALOG)
    assert len(result.probabilities) == n
    assert math.isclose(sum(result.probabilities), 1.0, rel_tol=0, abs_tol=1e-12)
    # a uniform base of 0.0 with one winner at 20.0: p_winner = e^20 / (e^20 + n-1)
    expected = math.exp(20.0) / (math.exp(20.0) + (n - 1))
    assert math.isclose(result.probabilities[0], expected, rel_tol=1e-9)
    assert all(p == 0.0 for p in result.probabilities[1:]) is False
    assert result.selected_index == 0


@pytest.mark.parametrize("winner,n,label", [
    (16, 256, "Q"), (25, 256, "Z"), (26, 256, "AA"), (127, 256, "EE"), (128, 256, "EF"),
    (200, 256, "HK"), (255, 256, "JU"), (16, 17, "Q"), (127, 128, "EE"),
])
def test_winning_labels_beyond_p(winner, n, label):
    result = wide.wide_choice_result(_logits(winner), _req(n), CATALOG)
    assert result.selected_index == winner
    assert result.selected_label == label
    assert result.selected_key == f"key{winner}"
    assert result.probabilities[winner] > 0.99


def test_keys_are_exactly_the_caller_keys_in_order():
    opts = [protocol.ChoiceOption(key=f"zz-{i}-é", description="d") for i in range(20)]
    opts.append(protocol.ChoiceOption(key="tail", description="d"))
    req = wide.WideChoiceRequest.create("s", "q", opts)
    result = wide.wide_choice_result(_logits(3), req, CATALOG)
    assert tuple(result.probabilities_by_key()) == req.caller_keys()
    assert list(result.probabilities_by_key()) == [f"zz-{i}-é" for i in range(20)] + ["tail"]
    payload = result.as_payload()
    assert list(payload["probabilities"]) == list(req.caller_keys())


def test_stable_softmax_does_not_overflow_and_ties_resolve_first():
    logits = np.array([1e4, 1e4, *([0.0] * 254)], dtype=np.float32)
    result = wide.wide_choice_result(logits, _req(17), CATALOG)
    assert all(math.isfinite(p) for p in result.probabilities)
    assert result.selected_index == 0
    assert math.isclose(result.probabilities[0], 0.5, abs_tol=1e-12)


def test_readout_refuses_bad_logit_vectors():
    req = _req(17)
    with pytest.raises(wide.WideError):
        wide.wide_choice_result(_logits(0, n=255), req, CATALOG)
    with pytest.raises(wide.WideError):
        wide.wide_choice_result(_logits(0, n=257), req, CATALOG)
    bad = _logits(0, n=256)
    bad[4] = np.nan
    with pytest.raises(wide.WideError):
        wide.wide_choice_result(bad, req, CATALOG)
    bad = _logits(0, n=256)
    bad[4] = np.inf
    with pytest.raises(wide.WideError):
        wide.wide_choice_result(bad, req, CATALOG)
    with pytest.raises(wide.WideError):
        wide.wide_choice_result(["x"] * 256, req, CATALOG)
    with pytest.raises(wide.WideError):
        wide.wide_choice_result(np.zeros((2, 256), dtype=np.float32), req, CATALOG)


def test_readout_refuses_all_negative_infinity_active_slice():
    logits = _logits(None, n=256)
    logits[:17] = -np.inf
    with pytest.raises(wide.WideError):
        wide.wide_choice_result(logits, _req(17), CATALOG)


def test_payload_is_a_separate_versioned_schema_with_explicit_adaptation():
    result = wide.wide_choice_result(_logits(20), _req(32), CATALOG)
    payload = result.as_payload()
    assert payload["schema"] == "qfn.decide.wide/1"
    assert payload["kind"] == "choice"
    assert "slot_probabilities" not in payload
    assert payload["adaptation"]["mode"] == "wide-labels"
    assert payload["adaptation"]["n_options"] == 32
    assert payload["adaptation"]["native_choice_slots"] == 16
    assert payload["adaptation"]["slot_vector_used"] is False
    assert payload["adaptation"]["calibrated_confidence"] is False
    assert payload["adaptation"]["trained_arms"] is False
    assert payload["confidence_status"] == wide.WIDE_CONFIDENCE_STATUS
    assert payload["jev_confidence"] is None
    assert payload["output_tokens"] == 0
    assert payload["catalog"]["digest"] == CATALOG.digest
    assert payload["catalog"]["n_labels"] == 256
    assert payload["selected"] == "key20"
    assert math.isclose(sum(payload["probabilities"].values()), 1.0, abs_tol=1e-12)
