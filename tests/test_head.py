"""Tests for src/qfn_decision/head.py (stage S4, buildout only).

Every input in this file is a SYNTHETIC FIXTURE. Nothing measured here is benchmark
evidence, model quality, or calibration evidence.

Float64 arithmetic appears only in the explicitly-labelled numerical tests at the
bottom (feature base / forward equivalence); the shipped forward path is float32.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from qfn_decision import head as H


# ---------------------------------------------------------------- fixtures

def provenance(**overrides):
    base = dict(
        feature_base="qwen3.8-flash-next-fixture@revA",
        feature_width=H.FEATURE_WIDTH,
        template_id="autotrust_raw_bare-v1",
        template_version="bare-v1",
        tokenizer_revision="tok-fixture-r1",
        base_revision="revA",
        rows=tuple(range(24)),
        row_scaling=1.0,
        rows_sha256="a" * 64,
    )
    base.update(overrides)
    return H.HeadProvenance(**base)


def weights(seed=17, scale=0.02):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((H.NUM_SLOTS, H.FEATURE_WIDTH)) * scale).astype(np.float32)


def head(seed=17, calibration=None, scale=0.02):
    return H.DecisionHead(
        weight=weights(seed, scale=scale),
        bias=np.zeros(H.NUM_SLOTS, dtype=np.float32),
        provenance=provenance(),
        calibration=calibration,
    )


def features(n=6, seed=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, H.FEATURE_WIDTH)).astype(np.float32)


# ---------------------------------------------------------------- slice 1: shape, params, masks

def test_parameter_count_is_the_contracted_61464():
    assert H.PARAMETER_COUNT == H.NUM_SLOTS * H.FEATURE_WIDTH + H.NUM_SLOTS == 61464
    assert head().n_parameters() == 61464


def test_active_masks_match_the_protocol_slot_ranges():
    noul = H.active_mask("noul")
    score = H.active_mask("score")
    assert list(np.flatnonzero(noul)) == [0, 1]
    assert list(np.flatnonzero(score)) == [2, 3, 4, 5, 6, 7]
    for n in (2, 3, 16):
        choice = H.active_mask("choice", n_options=n)
        assert list(np.flatnonzero(choice)) == list(range(8, 8 + n))
    assert choice_active_count(16) == 16


def choice_active_count(n):
    return int(H.active_mask("choice", n_options=n).sum())


def test_active_mask_rejects_out_of_contract_option_counts():
    for bad in (0, 1, 17, 255, None):
        with pytest.raises(H.HeadError):
            H.active_mask("choice", n_options=bad)
    with pytest.raises(H.HeadError):
        H.active_mask("unsupported")


def test_forward_logits_shape_dtype_and_finiteness():
    x = features(6)
    logits = head().forward_logits(x)
    assert logits.shape == (6, H.NUM_SLOTS)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()


def test_forward_rejects_wrong_width_and_nonfinite_features():
    h = head()
    with pytest.raises(H.HeadError):
        h.forward_logits(np.zeros((4, H.FEATURE_WIDTH + 1), dtype=np.float32))
    bad = features(4)
    bad[0, 0] = np.float32("inf")
    with pytest.raises(H.HeadError):
        h.forward_logits(bad)
    with pytest.raises(H.HeadError):
        h.forward_logits(np.zeros((4, H.FEATURE_WIDTH), dtype=np.float64))


def test_masked_softmax_zeros_inactive_slots_and_normalises_active_slots():
    logits = np.zeros((2, H.NUM_SLOTS), dtype=np.float32)
    logits[:, 3] = 25.0          # huge value on an INACTIVE score slot
    probs = H.masked_softmax(logits, H.active_mask("noul"))
    assert probs.shape == (2, H.NUM_SLOTS)
    assert probs[0, 3] == 0.0 and probs[1, 3] == 0.0
    assert probs[0, 0] == pytest.approx(0.5) and probs[0, 1] == pytest.approx(0.5)
    for row in probs:
        assert float(row.sum()) == pytest.approx(1.0, abs=1e-6)
        assert row[2:].sum() == 0.0


def test_masked_softmax_is_the_general_primitive_for_every_active_block():
    logits = np.array([[1.0, 2.0] + [0.0] * 22], dtype=np.float32)
    p = H.masked_softmax(logits, H.active_mask("noul"))[0]
    e0, e1 = np.exp(1.0 - 2.0), 1.0
    assert p[0] == pytest.approx(e0 / (e0 + e1), rel=1e-6)
    assert p[1] == pytest.approx(1.0 / (e0 + e1), rel=1e-6)


def test_forward_probs_keeps_inactive_slot_leakage_at_exactly_zero():
    h = head()
    x = features(3)
    probs = h.forward_probs(x, "score")
    assert probs.shape == (3, H.NUM_SLOTS)
    assert np.all(probs[:, :2] == 0.0) and np.all(probs[:, 8:] == 0.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    choice = h.forward_probs(x, "choice", n_options=3)
    assert np.all(choice[:, :8] == 0.0)
    assert np.all(choice[:, 11:] == 0.0)
    assert np.all(choice[:, 8:11] > 0.0)


def test_forward_probs_masks_before_softmax_even_with_extreme_inactive_logits():
    # an inactive slot carries a logit that would dominate any post-softmax mask
    w = np.zeros((H.NUM_SLOTS, H.FEATURE_WIDTH), dtype=np.float32)
    w[20, 0] = 1e4                     # inactive for score
    h = H.DecisionHead(weight=w, bias=np.zeros(H.NUM_SLOTS, dtype=np.float32), provenance=provenance())
    x = np.zeros((1, H.FEATURE_WIDTH), dtype=np.float32)
    x[0, 0] = 1.0
    probs = h.forward_probs(x, "score")
    assert probs[0, 20] == 0.0
    assert probs[0, 2:8].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- slice 2: initializers

def test_init_from_effective_rows_uses_rows_verbatim_and_zero_bias():
    rows = (np.arange(H.NUM_SLOTS * H.FEATURE_WIDTH, dtype=np.float32) / 1000.0).reshape(
        H.NUM_SLOTS, H.FEATURE_WIDTH
    )
    h = H.DecisionHead.init_from_effective_rows(rows=rows, provenance=provenance())
    np.testing.assert_allclose(h.weight, rows.astype(np.float32), rtol=0, atol=0)
    np.testing.assert_array_equal(h.bias, np.zeros(H.NUM_SLOTS, dtype=np.float32))
    assert h.calibration.fitted is False
    assert h.training is None


def test_init_from_effective_rows_converts_raw_bfloat16_bits_exactly():
    # bf16 -> fp32 widening is exact; the bit pattern 0x3F80 is 1.0
    raw = np.zeros((H.NUM_SLOTS, H.FEATURE_WIDTH), dtype=np.uint16)
    raw[0, 0] = 0x3F80
    raw[1, 0] = 0xC000          # -2.0
    h = H.DecisionHead.init_from_effective_rows(rows=raw, provenance=provenance())
    assert h.weight[0, 0] == 1.0 and h.weight[1, 0] == -2.0
    assert h.weight.dtype == np.float32
    np.testing.assert_allclose(H.bf16_to_fp32(np.array([0x3F80, 0xC000], dtype=np.uint16)), [1.0, -2.0])


def test_init_from_lm_head_selects_exactly_the_declared_rows():
    vocab = 64
    rng = np.random.default_rng(5)
    lm = rng.standard_normal((vocab, H.FEATURE_WIDTH)).astype(np.float32)
    idx = tuple(range(40, 64))
    h = H.DecisionHead.init_from_lm_head(lm_head=lm, row_indices=idx, provenance=provenance(rows=idx))
    np.testing.assert_allclose(h.weight, lm[list(idx)], rtol=0, atol=0)
    assert h.provenance.rows == idx


def test_init_from_lm_head_refuses_row_index_and_provenance_disagreement():
    lm = np.zeros((64, H.FEATURE_WIDTH), dtype=np.float32)
    with pytest.raises(H.HeadError, match="rows"):
        H.DecisionHead.init_from_lm_head(lm_head=lm, row_indices=tuple(range(24)),
                                        provenance=provenance(rows=tuple(range(1, 25))))
    with pytest.raises(H.HeadError):
        H.DecisionHead.init_from_lm_head(lm_head=lm, row_indices=tuple(range(60, 84)),
                                        provenance=provenance(rows=tuple(range(60, 84))))


def test_init_refuses_incompatible_reference_checkpoints():
    rows = np.zeros((H.NUM_SLOTS, H.FEATURE_WIDTH), dtype=np.float32)
    bad = provenance(init_source="autotrust/JEV-27B")
    with pytest.raises(H.HeadError, match="incompatible"):
        H.DecisionHead.init_from_effective_rows(rows=rows, provenance=bad)
    ok = provenance(init_source="qwen3.8-flash-next-uncensored/lm_head")
    assert H.DecisionHead.init_from_effective_rows(rows=rows, provenance=ok).provenance.init_source.endswith("lm_head")


def test_random_control_is_seed_reproducible_and_not_the_base_rows():
    a = H.DecisionHead.random_control(seed=17, provenance=provenance(), scale=0.01)
    b = H.DecisionHead.random_control(seed=17, provenance=provenance(), scale=0.01)
    c = H.DecisionHead.random_control(seed=18, provenance=provenance(), scale=0.01)
    np.testing.assert_array_equal(a.weight, b.weight)
    assert not np.array_equal(a.weight, c.weight)
    assert a.weight.dtype == np.float32
    assert a.n_parameters() == H.PARAMETER_COUNT


def test_constructor_validates_shape_dtype_and_finiteness():
    w = weights()
    with pytest.raises(H.HeadError):
        H.DecisionHead(weight=w[:, :-1], bias=np.zeros(H.NUM_SLOTS, np.float32), provenance=provenance())
    with pytest.raises(H.HeadError):
        H.DecisionHead(weight=w, bias=np.zeros(H.NUM_SLOTS + 1, np.float32), provenance=provenance())
    bad = w.copy()
    bad[0, 0] = np.float32("nan")
    with pytest.raises(H.HeadError):
        H.DecisionHead(weight=bad, bias=np.zeros(H.NUM_SLOTS, np.float32), provenance=provenance())
    with pytest.raises(H.HeadError):
        H.DecisionHead(weight=w.astype(np.float64), bias=np.zeros(H.NUM_SLOTS, np.float32),
                       provenance=provenance())
    with pytest.raises(H.HeadError):
        H.DecisionHead(weight=w, bias=np.zeros(H.NUM_SLOTS, np.float32),
                       provenance=provenance(feature_width=H.FEATURE_WIDTH - 1))


# ---------------------------------------------------------------- slice 3: artifact IO

def test_save_then_load_round_trips_with_checksum_and_metadata(tmp_path):
    h = head()
    h = h.with_training({"fixture_only": True, "release": "nonrelease"})
    path = tmp_path / "h1.npz"
    written = h.save(path)
    assert path.exists() and (tmp_path / "h1.json").exists()
    assert written["npz_bytes"] > 0

    meta = json.loads((tmp_path / "h1.json").read_text(encoding="utf-8"))
    assert meta["schema"] == H.HEAD_SCHEMA
    assert meta["format_version"] == H.HEAD_FORMAT_VERSION
    assert meta["parameters"] == H.PARAMETER_COUNT
    assert meta["shapes"]["weight"] == [H.NUM_SLOTS, H.FEATURE_WIDTH]
    assert meta["provenance"]["feature_base"] == h.provenance.feature_base
    assert meta["calibration"]["status"] == "uncalibrated"
    assert len(meta["data_sha256"]) == 64
    assert meta["array_sha256"]["weight"] != meta["array_sha256"]["bias"]

    back = H.DecisionHead.load(path)
    np.testing.assert_array_equal(back.weight, h.weight)
    np.testing.assert_array_equal(back.bias, h.bias)
    assert back.provenance == h.provenance
    assert back.training == {"fixture_only": True, "release": "nonrelease"}
    assert back.calibration.status() == "uncalibrated"


def test_artifact_contains_only_float32_npy_members_and_no_pickle(tmp_path):
    h = head()
    path = tmp_path / "h.npz"
    h.save(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert names and all(n.endswith(".npy") for n in names)
    with np.load(path, allow_pickle=False) as arrs:
        assert arrs["weight"].dtype == np.float32
        assert arrs["bias"].dtype == np.float32
        assert np.isfinite(arrs["weight"]).all()


def test_save_refuses_to_overwrite_an_existing_artifact_by_default(tmp_path):
    h = head()
    path = tmp_path / "h.npz"
    h.save(path)
    with pytest.raises(FileExistsError):
        h.save(path)
    h.save(path, overwrite=True)          # explicit opt-in still works


def test_load_detects_tampered_array_bytes(tmp_path):
    h = head()
    path = tmp_path / "h.npz"
    h.save(path)
    with np.load(path, allow_pickle=False) as arrs:
        data = {k: arrs[k] for k in arrs.files}
    data["weight"] = (data["weight"] + np.float32(1e-3)).astype(np.float32)
    np.savez(path, **data)
    with pytest.raises(H.HeadError, match="checksum"):
        H.DecisionHead.load(path)


def test_load_detects_metadata_mismatch_and_missing_metadata(tmp_path):
    h = head()
    path = tmp_path / "h.npz"
    h.save(path)
    meta_path = tmp_path / "h.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["data_sha256"] = "b" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(H.HeadError, match="checksum"):
        H.DecisionHead.load(path)

    h2 = head(seed=19)
    p2 = tmp_path / "h2.npz"
    h2.save(p2)
    (tmp_path / "h2.json").unlink()
    with pytest.raises(H.HeadError, match="metadata"):
        H.DecisionHead.load(p2)


def test_load_rejects_a_bad_format_version(tmp_path):
    h = head()
    path = tmp_path / "h.npz"
    h.save(path)
    meta_path = tmp_path / "h.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = 99
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(H.HeadError, match="format"):
        H.DecisionHead.load(path)


def test_optional_choice_max_mask_round_trips(tmp_path):
    h = head()
    mask = H.active_mask("choice", n_options=H.MAX_CHOICE_OPTIONS)
    h = h.with_choice_max_mask(mask)
    path = tmp_path / "h.npz"
    h.save(path)
    back = H.DecisionHead.load(path)
    np.testing.assert_array_equal(back.choice_max_mask, mask)
    assert back.choice_max_mask.dtype == bool
    with pytest.raises(H.HeadError):
        head().with_choice_max_mask(np.ones(H.NUM_SLOTS - 1, dtype=bool))


def test_save_is_atomic_and_leaves_no_temporary_files(tmp_path):
    h = head()
    h.save(tmp_path / "h.npz")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["h.json", "h.npz"]


# ---------------------------------------------------------------- slice 4: revisions and temperature

def test_validate_against_refuses_mismatched_base_template_hash_and_width():
    h = head()
    h.validate_against(feature_base=h.provenance.feature_base, feature_width=H.FEATURE_WIDTH,
                       template_id=h.provenance.template_id, template_version=h.provenance.template_version,
                       rows_sha256=h.provenance.rows_sha256)
    for kwargs in (
        {"feature_base": "some-other-base@revB"},
        {"feature_width": 2048},
        {"template_id": "qwen_chat_v1"},
        {"template_version": "bare-v2"},
        {"rows_sha256": "c" * 64},
    ):
        with pytest.raises(H.HeadError, match="mismatch"):
            h.validate_against(**kwargs)


def test_provenance_validation_rejects_incomplete_or_inconsistent_records():
    with pytest.raises(H.HeadError):
        provenance(feature_base="")
    with pytest.raises(H.HeadError):
        provenance(feature_width=2048)
    with pytest.raises(H.HeadError):
        provenance(rows=tuple(range(23)))
    with pytest.raises(H.HeadError):
        provenance(rows=(1,) * 24)
    with pytest.raises(H.HeadError):
        provenance(rows_sha256="not-a-hash")
    with pytest.raises(H.HeadError):
        provenance(tokenizer_revision=None)
    with pytest.raises(H.HeadError):
        provenance(row_scaling=0.0)


def test_temperature_is_per_primitive_positive_and_changes_only_its_primitive():
    x = features(2)
    base = head()
    t2 = H.CalibrationState(temperatures={"noul": 2.0, "choice": 1.0, "score": 1.0},
                            fitted=False, method=None, source=None, scope=None)
    warm = base.with_calibration(t2)
    p_base = base.forward_probs(x, "noul")
    p_warm = warm.forward_probs(x, "noul")
    manual = H.masked_softmax(base.forward_logits(x).astype(np.float64) / 2.0, H.active_mask("noul"))
    np.testing.assert_allclose(p_warm, manual, rtol=1e-6, atol=1e-6)
    assert not np.allclose(p_base, p_warm)
    np.testing.assert_allclose(warm.forward_probs(x, "score"), base.forward_probs(x, "score"))
    for bad in ({"noul": 0.0, "choice": 1.0, "score": 1.0}, {"noul": -1.0, "choice": 1.0, "score": 1.0},
                {"noul": float("inf"), "choice": 1.0, "score": 1.0}):
        with pytest.raises(H.HeadError):
            H.CalibrationState(temperatures=bad, fitted=False, method=None, source=None, scope=None)


def test_a_fitted_calibration_requires_method_source_and_scope():
    with pytest.raises(H.HeadError, match="method"):
        H.CalibrationState(temperatures={"noul": 1.2, "choice": 1.2, "score": 1.2},
                           fitted=True, method=None, source="calibration_split", scope="fixture")
    good = fitted_calibration(head())
    assert good.status() == "calibrated"
    assert good.as_dict()["calibration_scope"] == "calibration split only"
    assert good.as_dict()["calibrated_primitives"] == list(H.PRIMITIVE_KEYS)
    assert good.as_dict()["parameter_fingerprint"] and good.as_dict()["data_fingerprint"]


def test_an_initialised_head_is_untrained_and_uncalibrated():
    h = H.DecisionHead.init_from_effective_rows(rows=weights(), provenance=provenance())
    assert h.training is None
    d = h.calibration.describe()
    assert d["status"] == "uncalibrated" and d["fitted"] is False
    assert "not calibrated" in d["note"]
    assert h.describe()["artifact_state"]["calibration_status"] == "uncalibrated"
    assert h.describe()["artifact_state"]["trained"] is False


# ---------------------------------------------------------------- slice 5: explicit float64 references

def test_float32_forward_matches_a_float64_reference_within_fp32_tolerance():
    """Explicitly-labelled float64 reference test: the shipped forward is float32."""
    h = head()
    x = features(8, seed=11)
    logits32 = h.forward_logits(x)
    logits64 = x.astype(np.float64) @ h.weight.astype(np.float64).T + h.bias.astype(np.float64)
    np.testing.assert_allclose(logits32.astype(np.float64), logits64, rtol=1e-5, atol=1e-5)
    p32 = H.masked_softmax(logits32, H.active_mask("score"))
    p64 = H.masked_softmax(logits64, H.active_mask("score"))
    np.testing.assert_allclose(p32.astype(np.float64), p64, rtol=1e-5, atol=1e-6)


def test_probs_stay_finite_and_normalised_for_large_magnitude_features():
    h = head(scale=5.0)
    x = (features(4, seed=13) * 50.0).astype(np.float32)
    p = h.forward_probs(x, "score")
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_slot_probs_view_returns_caller_key_ordered_probabilities():
    h = head()
    x = features(1, seed=2)
    probs = h.forward_probs(x, "choice", n_options=4)
    view = h.describe_slots(probs[0], "choice", n_options=4)
    assert len(view) == 4 and all(0.0 < p < 1.0 for p in view)
    assert sum(view) == pytest.approx(1.0, abs=1e-6)


def test_typed_result_bridge_returns_caller_keys_expected_level_and_p_true_not_jev_confidence():
    """The head returns primitives through the existing typed protocol utilities."""
    from qfn_decision.protocol import (
        CONFIDENCE_STATUS_ABSENT, ChoiceOption, DecisionRequest, DecisionResult,
    )

    h = head()
    x = features(1, seed=8)[0]

    noul = DecisionRequest.create(kind="noul", state="s", question="q")
    res = H.decision_result(h, x, noul)
    assert isinstance(res, DecisionResult)
    assert res.caller_keys == ("false", "true")
    assert res.p_true is not None and 0.0 < res.p_true < 1.0
    assert res.raw_max_probability == pytest.approx(max(res.option_probabilities))
    assert res.jev_confidence is None
    assert res.confidence_status == CONFIDENCE_STATUS_ABSENT
    assert noul.active_slot_indices() == [0, 1]
    assert float(sum(res.slot_probabilities[2:])) == 0.0

    score = DecisionRequest.create(kind="score", state="s", question="q",
                                   score_rubrics=("a", "b", "c", "d", "e", "f"))
    sres = H.decision_result(h, x, score)
    assert 0.0 <= sres.expected_level <= 5.0
    assert len(sres.level_probabilities) == 6
    assert sres.caller_keys == ("0", "1", "2", "3", "4", "5")

    choice = DecisionRequest.create(
        kind="choice", state="s", question="q",
        options=[ChoiceOption(key=k, description=k) for k in ("alpha", "beta")],
    )
    cres = H.decision_result(h, x, choice)
    assert cres.caller_keys == ("alpha", "beta")
    assert set(cres.probabilities_by_key()) == {"alpha", "beta"}
    assert float(sum(cres.slot_probabilities[:8])) == 0.0

    with pytest.raises(H.HeadError):
        H.decision_result(h, x, "not-a-request")
    with pytest.raises(H.HeadError):
        H.decision_result(h, features(2, seed=9), noul)


# ---------------------------------------------------------------- slice 6: calibration binding
# (S4 correction: a fit must be bound to the exact parameters and data it was fitted on, so a
#  parameter change or a retrain can never leave a stale "calibrated" tag behind.)

def fitted_calibration(head_obj, kinds=("noul", "choice", "score"), data_fp="b" * 64):
    """A fitted CalibrationState bound to this head's parameters and a data fingerprint."""
    per = {
        kind: {"calibrated": True, "n_rows": 4, "scope": "calibration split only",
               "source": "calibration split KL over teacher targets", "temperature": 1.25}
        for kind in kinds
    }
    for kind in H.PRIMITIVE_KEYS:
        per.setdefault(kind, {"calibrated": False, "n_rows": 0, "scope": None,
                              "source": None, "temperature": 1.0})
    return H.CalibrationState(
        temperatures={k: (1.25 if k in kinds else 1.0) for k in H.PRIMITIVE_KEYS},
        fitted=True, method="bounded log-temperature grid",
        source="calibration split KL over teacher targets",
        scope="calibration split only",
        fitted_primitives=tuple(kinds),
        parameter_fingerprint=head_obj.parameter_fingerprint(),
        provenance_fingerprint=head_obj.provenance_fingerprint(),
        data_fingerprint=data_fp,
        per_primitive=per,
    )


def test_parameter_and_provenance_fingerprints_are_stable_and_sensitive():
    h = head()
    fp = h.parameter_fingerprint()
    assert isinstance(fp, str) and len(fp) == 64
    assert h.parameter_fingerprint() == fp                  # stable for the same parameters
    assert h.with_parameters(h.weight + np.float32(0.01), h.bias).parameter_fingerprint() != fp
    assert h.with_parameters(h.weight, h.bias + np.float32(0.01)).parameter_fingerprint() != fp
    other_base = H.DecisionHead(weight=h.weight, bias=h.bias,
                                provenance=provenance(feature_base="other@revB"))
    assert other_base.provenance_fingerprint() != h.provenance_fingerprint()
    assert other_base.parameter_fingerprint() != fp


def test_a_fitted_calibration_requires_fingerprints_and_primitive_support():
    h = head()
    base = dict(temperatures={k: 1.1 for k in H.PRIMITIVE_KEYS}, fitted=True,
                method="m", source="s", scope="calibration split only")
    support = {k: {"calibrated": True, "n_rows": 4, "scope": "sc", "source": "sr"}
               for k in H.PRIMITIVE_KEYS}
    fps = dict(parameter_fingerprint=h.parameter_fingerprint(),
               provenance_fingerprint=h.provenance_fingerprint(), data_fingerprint="c" * 64)
    with pytest.raises(H.HeadError, match="primitive"):
        H.CalibrationState(**base, fitted_primitives=(), per_primitive=support, **fps)
    with pytest.raises(H.HeadError, match="primitive"):
        H.CalibrationState(**base, fitted_primitives=("noul", "bogus"), per_primitive=support, **fps)
    with pytest.raises(H.HeadError, match="support"):
        H.CalibrationState(**base, fitted_primitives=("noul",), **fps)          # no per-primitive support
    with pytest.raises(H.HeadError, match="support"):
        H.CalibrationState(**base, fitted_primitives=("noul",),
                           per_primitive={"noul": {"calibrated": True, "n_rows": 0, "scope": "sc",
                                                   "source": "sr"}}, **fps)
    with pytest.raises(H.HeadError, match="fingerprint"):
        H.CalibrationState(**base, fitted_primitives=("noul",),
                           per_primitive={"noul": support["noul"]})              # not bound to parameters
    good = fitted_calibration(h)
    assert good.status() == "calibrated" and good.fitted_primitives == H.PRIMITIVE_KEYS


def test_calibration_status_is_bound_to_parameters_and_provenance():
    h = head()
    assert h.calibration_status() == "uncalibrated"
    assert h.calibration_is_valid() is False
    calibrated = h.with_calibration(fitted_calibration(h))
    assert calibrated.calibration_status() == "calibrated"
    assert calibrated.calibration_is_valid() is True

    moved = calibrated.with_parameters(calibrated.weight + np.float32(0.01), calibrated.bias)
    assert moved.calibration.fitted is True                 # the tag is not silently rewritten
    assert moved.calibration_is_valid() is False
    assert moved.calibration_status() == "invalidated_parameter_change"
    assert moved.describe()["artifact_state"]["calibration_status"] == "invalidated_parameter_change"

    rebased = dataclasses_replace_provenance(calibrated)
    assert rebased.calibration_status() == "invalidated_provenance_change"


def dataclasses_replace_provenance(h):
    import dataclasses
    return dataclasses.replace(h, provenance=provenance(feature_base="other@revB"))


def test_calibration_support_is_honest_per_primitive():
    h = head()
    partial = h.with_calibration(fitted_calibration(h, kinds=("noul",)))
    assert partial.primitive_calibration_status("noul")["status"] == "calibrated"
    for kind in ("choice", "score"):
        block = partial.primitive_calibration_status(kind)
        assert block["status"] == "uncalibrated" and block["reason"]
        assert block["calibrated"] is False
    assert partial.describe()["artifact_state"]["calibrated_primitives"] == ["noul"]
    with pytest.raises(H.HeadError):
        partial.primitive_calibration_status("nope")


def test_calibration_binding_survives_an_artifact_round_trip(tmp_path):
    h = head().with_calibration(fitted_calibration(head()))
    path = tmp_path / "cal.npz"
    h.save(path)
    back = H.DecisionHead.load(path)
    assert back.calibration.parameter_fingerprint == h.calibration.parameter_fingerprint
    assert back.calibration.data_fingerprint == h.calibration.data_fingerprint
    assert back.calibration.fitted_primitives == h.calibration.fitted_primitives
    assert back.calibration.per_primitive["noul"]["n_rows"] == 4
    assert back.calibration_status() == "calibrated"
    assert "fingerprint" in back.metadata_calibration_summary()["note"].lower()
