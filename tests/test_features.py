"""S3 tests: feature archive format, BF16 codecs, checksums, resume, corruption handling.

Scope: pure numpy/stdlib on the Windows host. No torch, no GPU. Fixtures here are
synthetic arrays and label themselves as test-only; nothing in this file is parity
evidence or a benchmark result (see reports/s3/RESULT.md).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from qfn_decision import features


# --------------------------------------------------------------------------------------
# BF16 codecs
# --------------------------------------------------------------------------------------


def test_bf16_roundtrip_is_exact_for_representable_values():
    import numpy as np2

    vals = np2.asarray([0.0, -0.0, 1.0, -1.0, 0.5, 0.0078125, 1.5, 256.0, -3.25], dtype=np.float32)
    bits = features.fp32_to_bf16_bits(vals)
    assert bits.dtype == np.uint16
    back = features.bf16_bits_to_fp32(bits)
    assert np.array_equal(back, vals.astype(np.float32))


def test_bf16_roundtrip_error_is_bounded_by_half_ulp():
    vals = np.linspace(-100.0, 100.0, 4001, dtype=np.float32)
    rt = features.bf16_roundtrip_fp32(vals)
    err = np.abs(rt - vals)
    allowed = np.array([features.bf16_ulp(float(v)) / 2 for v in vals], dtype=np.float32)
    assert np.all(err <= allowed + 1e-30)


def test_bf16_roundtrip_uses_round_to_nearest_even():
    # 1 + 2^-8 sits exactly between 1.0 and 1+2^-7 -> ties-to-even keeps 1.0
    x = np.asarray([np.float32(1.0) + np.float32(2.0 ** -8)], dtype=np.float32)
    assert float(features.bf16_roundtrip_fp32(x)[0]) == 1.0


def test_bf16_bit_helpers_match_numpy_cast():
    rng = np.random.default_rng(7)
    vals = (rng.standard_normal(1000) * 10).astype(np.float32)
    # Reference path: numpy has no native bf16, use the torch-free truncate+round trick:
    # viewing fp32 as uint32, add rounding bias then take the high 16 bits.
    u = vals.view(np.uint32).copy()
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    ref = ((u + bias) >> 16).astype(np.uint16)
    assert np.array_equal(features.fp32_to_bf16_bits(vals), ref)


def test_bf16_ulp_values():
    assert features.bf16_ulp(1.0) == pytest.approx(2.0 ** -7)
    assert features.bf16_ulp(0.5) == pytest.approx(2.0 ** -8)
    assert features.bf16_ulp(8.0) == pytest.approx(2.0 ** -4)
    assert features.bf16_ulp(-8.0) == pytest.approx(2.0 ** -4)
    assert features.bf16_ulp(0.0) == 0.0
    assert features.bf16_ulp(float("nan")) == 0.0


# --------------------------------------------------------------------------------------
# token hash canonicalisation (must agree bit-for-bit with the engine helper)
# --------------------------------------------------------------------------------------


def test_token_sequence_sha256_is_stable_and_order_sensitive():
    a = features.token_sequence_sha256([1, 2, 3])
    assert a == features.token_sequence_sha256([1, 2, 3])
    assert a != features.token_sequence_sha256([3, 2, 1])
    assert a != features.token_sequence_sha256([1, 2, 3, 4])
    assert features.token_sequence_sha256([]) != features.token_sequence_sha256([0])


def test_token_sequence_sha256_matches_explicit_definition():
    ids = [7, 8, 9]
    payload = b"qfn.tokens/1:3:" + np.asarray(ids, dtype="<i4").tobytes()
    assert features.token_sequence_sha256(ids) == hashlib.sha256(payload).hexdigest()


def test_token_sequence_sha256_rejects_bad_input():
    with pytest.raises(features.FeatureError):
        features.token_sequence_sha256([1.5])
    with pytest.raises(features.FeatureError):
        features.token_sequence_sha256([-1])
    with pytest.raises(features.FeatureError):
        features.token_sequence_sha256([2 ** 40])


# --------------------------------------------------------------------------------------
# record round-trip
# --------------------------------------------------------------------------------------


def _record(features_mod, **over):
    rng = np.random.default_rng(3)
    vec32 = (rng.standard_normal(2560).astype(np.float32) * 2.0)
    bits = features_mod.fp32_to_bf16_bits(vec32)
    # The FP32 reference is the exact fp32 view of the captured BF16 state (bf16 -> fp32
    # is lossless), which is what the engine writes; it is not the pre-quantisation fp32.
    vec32 = features_mod.bf16_bits_to_fp32(bits)
    base = dict(
        example_id="ex-0001",
        engine_request_id="cmpl-qfn-abc-0",
        prompt_token_sha256=features_mod.token_sequence_sha256([1, 2, 3, 4]),
        prompt_tokens_count=4,
        prompt_len=4,
        terminal_position=3,
        cache_hit_tokens=0,
        num_scheduled_tokens=4,
        query_start=0,
        query_end=4,
        logits_row_index=0,
        batch_index=0,
        internal_index=0,
        vector_bf16=bits,
        vector_fp32=vec32,
        verbalizer_ids=[150958, 150959] + list(range(2, 24)),
        verbalizer_logits_fp32=np.linspace(-3.0, 3.0, 24).astype(np.float32),
        runtime_argmax_token_id=1234,
        runtime_max_logit_fp32=9.5,
        head_scale=1.0,
        head_soft_cap=None,
        head_org_vocab_size=248320,
        head_dtype="bfloat16",
        lm_head_tp_size=2,
        tp_rank=0,
        world_size=2,
        cudagraph_mode="FULL_DECODE_ONLY",
        kernel_mode="prefill",
        runner_source_sha256="f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8",
        capture_wall_ns=1_700_000_000_000_000_000,
        logits_captured=True,
        logits_absent_reason=None,
        vector_source="runner.sample.hidden_states[logits_index]",
        dup_disambiguation=0,
    )
    base.update(over)
    return features_mod.FeatureRecord(**base)


def test_record_npz_roundtrip_keeps_bf16_exact():
    rec = _record(features)
    blob = features.encode_record_npz(rec)
    back = features.decode_record_npz(blob)
    assert np.array_equal(np.asarray(back.vector_bf16), np.asarray(rec.vector_bf16))
    assert back.vector_bf16.dtype == np.uint16
    assert np.allclose(np.asarray(back.vector_fp32), np.asarray(rec.vector_fp32))
    assert back.example_id == rec.example_id
    assert back.engine_request_id == rec.engine_request_id
    assert back.prompt_token_sha256 == rec.prompt_token_sha256
    assert list(np.asarray(back.verbalizer_ids)) == list(np.asarray(rec.verbalizer_ids))
    assert back.vector_fp32_matches_bf16 is True


def test_record_roundtrip_without_fp32_reference():
    rec = _record(features, vector_fp32=None)
    back = features.decode_record_npz(features.encode_record_npz(rec))
    assert back.vector_fp32 is None
    assert back.vector_fp32_matches_bf16 is None
    assert back.decode_vector_fp32().shape == (2560,)


def test_record_rejects_vector_fp32_inconsistent_with_bf16():
    rec = _record(features)
    bad = rec.vector_fp32.copy()
    bad[10] += 1.0
    with pytest.raises(features.FeatureError):
        features.FeatureRecord(**{**rec.__dict__, "vector_fp32": bad})


def test_record_rejects_wrong_width_and_dtype():
    rec = _record(features)
    with pytest.raises(features.FeatureError):
        features.FeatureRecord(**{**rec.__dict__, "vector_bf16": np.zeros(256, dtype=np.uint16)})
    with pytest.raises(features.FeatureError):
        features.FeatureRecord(**{**rec.__dict__, "vector_bf16": np.zeros(2560, dtype=np.float32)})


def test_record_requires_hash_and_ids():
    rec = _record(features)
    with pytest.raises(features.FeatureError):
        features.FeatureRecord(**{**rec.__dict__, "prompt_token_sha256": None})
    with pytest.raises(features.FeatureError):
        features.FeatureRecord(**{**rec.__dict__, "engine_request_id": ""})


def test_record_rejects_non_positive_prompt_len_and_bad_position():
    rec = _record(features)
    with pytest.raises(features.FeatureError):
        features.FeatureRecord(**{**rec.__dict__, "prompt_len": 0})
    with pytest.raises(features.FeatureError):
        features.FeatureRecord(**{**rec.__dict__, "terminal_position": 99})


def test_record_checksum_is_over_the_encoded_bytes():
    rec = _record(features)
    blob = features.encode_record_npz(rec)
    assert features.record_sha256(rec) == hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------------------
# archive: atomic writes, resume, verification, corruption
# --------------------------------------------------------------------------------------


def _archive(tmp_path):
    return features.FeatureArchive(tmp_path / "archive", run_id="s3test")


def test_archive_write_is_atomic_no_partial(tmp_path, monkeypatch):
    ar = _archive(tmp_path)
    rec = _record(features)
    entry = ar.write(rec)
    assert (ar.records_dir / entry.file_name).is_file()
    assert not list(ar.records_dir.glob("*.part"))
    assert not list(ar.records_dir.glob("*.tmp"))
    # a failed rename must not leave a half-written record visible
    real_replace = features.os.replace

    def boom(a, b):
        raise OSError("simulated crash")

    monkeypatch.setattr(features.os, "replace", boom)
    with pytest.raises(OSError):
        ar.write(_record(features, example_id="ex-0002"))
    monkeypatch.setattr(features.os, "replace", real_replace)
    assert len(list(ar.records_dir.glob("*.npz"))) == 1


def test_archive_index_maps_example_to_internal_request_id(tmp_path):
    ar = _archive(tmp_path)
    ar.write(_record(features, example_id="ex-A", engine_request_id="cmpl-qfn-aaa-0"))
    ar.write(_record(features, example_id="ex-B", engine_request_id="cmpl-qfn-bbb-0"))
    idx = ar.index()
    assert idx["ex-A"]["engine_request_id"] == "cmpl-qfn-aaa-0"
    assert idx["ex-B"]["prompt_token_sha256"] == features.token_sequence_sha256([1, 2, 3, 4])
    assert set(idx) == {"ex-A", "ex-B"}


def test_archive_resume_scan_reports_completed_and_skips_duplicates(tmp_path):
    ar = _archive(tmp_path)
    ar.write(_record(features, example_id="ex-A"))
    scan = ar.resume_scan()
    assert scan["completed_example_ids"] == ["ex-A"]
    assert scan["missing"] == []
    assert scan["corrupt"] == []
    ar.write(_record(features, example_id="ex-A", engine_request_id="cmpl-qfn-other-0"))
    scan2 = ar.resume_scan()
    assert scan2["completed_example_ids"] == ["ex-A"]
    assert scan2["duplicate_records"] == 1
    assert scan2["duplicate_example_ids"] == ["ex-A"]


def test_archive_resume_scan_detects_corrupt_record(tmp_path):
    ar = _archive(tmp_path)
    entry = ar.write(_record(features))
    path = ar.records_dir / entry.file_name
    raw = bytearray(path.read_bytes())
    raw[40:60] = b"xxxxxxxxxxxxxxxxxxxx"[0 : len(raw[40:60])]
    path.write_bytes(bytes(raw))
    scan = ar.resume_scan()
    assert scan["completed_example_ids"] == []
    assert scan["corrupt"] and scan["corrupt"][0]["file_name"] == entry.file_name


def test_archive_checksums_file_covers_every_record(tmp_path):
    ar = _archive(tmp_path)
    ar.write(_record(features, example_id="ex-A", engine_request_id="cmpl-qfn-cs-a-0"))
    ar.write(_record(features, example_id="ex-B", engine_request_id="cmpl-qfn-cs-b-0"))
    lines = (ar.root / "CHECKSUMS.sha256").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        digest, name = line.split("  ")
        assert hashlib.sha256((ar.records_dir / name).read_bytes()).hexdigest() == digest
    assert ar.verify()["ok"] is True


def test_archive_verify_flags_tampered_record(tmp_path):
    ar = _archive(tmp_path)
    entry = ar.write(_record(features))
    (ar.records_dir / entry.file_name).write_bytes(b"not an npz")
    report = ar.verify()
    assert report["ok"] is False
    assert report["mismatched"] == [entry.file_name]


def test_archive_manifest_counts_and_hashes(tmp_path):
    ar = _archive(tmp_path)
    ar.write(_record(features, example_id="ex-A", engine_request_id="cmpl-qfn-aaa-0"))
    ar.write(
        _record(
            features,
            example_id="ex-B",
            engine_request_id="cmpl-qfn-bbb-0",
            logits_captured=False,
            logits_absent_reason="logits_not_available_on_this_rank",
        )
    )
    man = ar.manifest()
    assert man["schema"] == features.ARCHIVE_SCHEMA
    assert man["run_id"] == "s3test"
    assert man["record_count"] == 2
    assert man["example_count"] == 2
    assert man["logits_captured_count"] == 1
    assert man["prompt_token_sha256_unique"] == 1  # synthetic fixture: same ids in both records
    assert len(man["records"]) == 2
    assert all(r["sha256"] for r in man["records"])
    assert man["engine_build"]["runner_source_sha256"].startswith("f08ba0f5")


def test_same_engine_request_id_and_prompt_hash_maps_to_one_file(tmp_path):
    """A repeated capture of the same (request id, prompt hash) must not create a stray file;
    a genuine second capture uses dup_disambiguation."""
    ar = _archive(tmp_path)
    first = ar.write(_record(features))
    again = ar.write(_record(features))
    assert first.file_name == again.file_name
    assert len(ar.record_files()) == 1
    second = ar.write(_record(features, dup_disambiguation=1))
    assert second.file_name != first.file_name
    assert len(ar.record_files()) == 2


def test_archive_rejects_record_name_with_path_separator(tmp_path):
    ar = _archive(tmp_path)
    with pytest.raises(features.FeatureError):
        ar.write(_record(features, engine_request_id="../escape"))


# --------------------------------------------------------------------------------------
# continuation 2026-10-10: opaque ids vs on-disk basenames (over-broad '..' guard)
# --------------------------------------------------------------------------------------

# the legitimate row id the live run refused: interior '..' inside an opaque example id
LIVE_ROW_ID = "tic_tac_toe-v1:.OOX..X.X:OO...XX.X"


def test_opaque_example_id_with_interior_dots_is_accepted_and_preserved_exactly(tmp_path):
    rec = _record(features, example_id=LIVE_ROW_ID, engine_request_id="cmpl-qfn-live-0")
    assert rec.example_id == LIVE_ROW_ID                     # no normalisation of opaque ids
    blob = features.encode_record_npz(rec)
    back = features.decode_record_npz(blob)
    assert back.example_id == LIVE_ROW_ID                    # exact round trip through npz
    ar = _archive(tmp_path)
    entry = ar.write(back)
    on_disk = json.loads((ar.root / "index.jsonl").read_text(encoding="utf-8").strip())
    assert on_disk["example_id"] == LIVE_ROW_ID              # raw identifier kept in metadata
    # the on-disk basename is derived ONLY from the engine request id and stays one component
    assert "/" not in entry.file_name and "\\" not in entry.file_name
    assert entry.file_name == features._safe_name("cmpl-qfn-live-0") + "__" + \
        rec.prompt_token_sha256[:16] + "__0.npz"
    reloaded = ar.read_record(ar.root / "records" / entry.file_name)
    assert reloaded.example_id == LIVE_ROW_ID


def test_path_hazards_are_still_rejected_for_opaque_ids_and_basenames():
    for evil in ("..", ".", "../escape", "..\\escape", "a/b", "a\\b", "/abs", "C:/drive",
                 "C:rel", "//unc/share", "a\x00b", "..trailing?", "..", "x..", "..x"):
        with pytest.raises(features.FeatureError):
            features.reject_unsafe_path_token(evil, what="example_id")
        with pytest.raises(features.FeatureError):
            _record(features, example_id=evil)
    # interior dots that are not a path boundary remain legitimate
    for ok in ("a.b", "a..b", ".hidden", "tic..tac.toe", LIVE_ROW_ID):
        assert features.reject_unsafe_path_token(ok, what="example_id") == ok


def test_safe_name_is_deterministic_and_never_escapes():
    name = features._safe_name(LIVE_ROW_ID)
    assert name == features._safe_name(LIVE_ROW_ID)
    assert "/" not in name and "\\" not in name and "\x00" not in name
    assert len(features._safe_name("x" * 400)) <= 160


def test_archive_rejects_a_traversal_engine_id_without_touching_siblings(tmp_path):
    ar = _archive(tmp_path)
    good = ar.write(_record(features, example_id="keep-me"))
    with pytest.raises(features.FeatureError):
        ar.write(_record(features, example_id="evil", engine_request_id="cmpl-qfn-.."))
    assert [p.name for p in ar.record_files()] == [good.file_name]


def test_archive_append_error_log(tmp_path):
    ar = _archive(tmp_path)
    ar.log_error({"engine_request_id": "cmpl-qfn-x-0", "reason": "quota_exceeded"})
    lines = (ar.root / "errors.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["reason"] == "quota_exceeded"


# --------------------------------------------------------------------------------------
# tolerance helpers (pre-registered design bounds)
# --------------------------------------------------------------------------------------


def test_bf16_gemm_allowance_formula():
    # 2 bf16 ULPs of max(|a|,|b|) plus the 1e-4 floor
    assert features.bf16_gemm_allowance(1.0, 1.0) == pytest.approx(2 * features.bf16_ulp(1.0) + 1e-4)
    assert features.bf16_gemm_allowance(8.0, -8.0) == pytest.approx(2 * features.bf16_ulp(8.0) + 1e-4)
    assert features.bf16_gemm_allowance(0.0, 0.0) == pytest.approx(1e-4)


def test_load_tolerances_config_matches_repo_file():
    cfg = features.load_tolerances(
        path=__import__("pathlib").Path(__file__).resolve().parents[1]
        / "configs/numeric_tolerances.json"
    )
    assert cfg["status"] == "pre_registered_design_bounds"
    assert cfg["classes"]["fp32_direct_head_offline_vs_online"]["atol"] == 1e-5
    assert cfg["classes"]["bf16_gemm_vs_runtime_selected_logits"]["floor"] == 1e-4
    assert cfg["classes"]["base_restricted_probability"]["absolute"] == 0.005
    assert cfg["classes"]["inactive_slots"]["exact_zero"] is True
    assert cfg["classes"]["masked_probability_sum_fp32"]["abs"] == 1e-6


# ======================================================================================
# S3 PM-review correction: decision boundary, truthful cache accounting, numeric metadata
# (RED first: see reports/s3/receipts/14-red-features.txt)
# ======================================================================================


def test_record_rejects_in_range_nonterminal_position():
    """The decision boundary is the LAST prompt token: an in-range but nonterminal position
    must be refused by the format, not merely bounds-checked."""
    rec = _record(features)
    assert rec.terminal_position == rec.prompt_len - 1  # 3 of a 4-token prompt
    for bad in (0, 1, 2):
        with pytest.raises(features.FeatureError) as exc:
            _record(features, terminal_position=bad)
        assert "terminal" in str(exc.value).lower()


def test_record_requires_computed_tokens_before_step_to_complete_the_prompt():
    """computed_tokens_before_step is the authoritative runner count: non-negative, and it must
    exactly complete the prompt with the scheduled chunk at the terminal step."""
    ok = _record(
        features,
        computed_tokens_before_step=0,
        num_scheduled_tokens=4,
        cache_hit_tokens=None,
        cache_hit_provenance="unknown",
        cache_hit_status_reason="not observable at the runner hook",
    )
    assert ok.computed_tokens_before_step == 0
    with pytest.raises(features.FeatureError):
        _record(features, computed_tokens_before_step=3, num_scheduled_tokens=4)  # 3+4 != 4
    with pytest.raises(features.FeatureError):
        _record(features, computed_tokens_before_step=-1, num_scheduled_tokens=5)
    with pytest.raises(features.FeatureError):
        _record(features, computed_tokens_before_step=0, num_scheduled_tokens=3)  # 0+3 != 4


def test_record_requires_query_length_to_match_scheduled_tokens():
    with pytest.raises(features.FeatureError):
        _record(features, query_start=0, query_end=3, num_scheduled_tokens=4)


def test_record_cache_hit_unknown_is_null_with_reason():
    """The runner hook cannot observe the scheduler's prefix-cache hit count: unknown must be
    null plus a status reason, never an invented 0 and never the pre-step count."""
    rec = _record(
        features,
        computed_tokens_before_step=0,
        cache_hit_tokens=None,
        cache_hit_provenance="unknown",
        cache_hit_status_reason="runner sample() cannot observe the scheduler prefix-cache hit count",
    )
    assert rec.cache_hit_tokens is None
    assert rec.cache_hit_provenance == "unknown"
    with pytest.raises(features.FeatureError):
        _record(features, computed_tokens_before_step=0, cache_hit_tokens=None, cache_hit_provenance="unknown")
    with pytest.raises(features.FeatureError):
        _record(features, cache_hit_tokens=None, cache_hit_provenance="measured", cache_hit_source="x")


def test_record_rejects_prestep_count_mislabeled_as_cache_hits():
    """8192 tokens computed by earlier chunks of a 9000-token cold prompt are NOT cache hits."""
    with pytest.raises(features.FeatureError) as exc:
        _record(
            features,
            prompt_len=9000,
            prompt_tokens_count=9000,
            terminal_position=8999,
            num_scheduled_tokens=808,
            query_start=0,
            query_end=808,
            computed_tokens_before_step=8192,
            cache_hit_tokens=8192,
            cache_hit_provenance="unknown",
        )
    assert "cache_hit" in str(exc.value)


def test_record_accepts_measured_cache_hit_with_explicit_provenance():
    rec = _record(
        features,
        computed_tokens_before_step=3200,
        num_scheduled_tokens=100,
        prompt_len=3300,
        prompt_tokens_count=3300,
        terminal_position=3299,
        query_end=100,
        cache_hit_tokens=3200,
        cache_hit_provenance="measured",
        cache_hit_source="scheduler.prefill_stats.num_local_cached_tokens",
    )
    assert rec.cache_hit_tokens == 3200
    with pytest.raises(features.FeatureError):
        _record(
            features,
            computed_tokens_before_step=3200,
            num_scheduled_tokens=100,
            prompt_len=3300,
            prompt_tokens_count=3300,
            terminal_position=3299,
            query_end=100,
            cache_hit_tokens=3200,
            cache_hit_provenance="measured",  # measured without a source is refused
        )


def test_record_legacy_single_chunk_known_cache_path_is_documented_and_narrow():
    """An explicit known cache count with no authoritative pre-step count is only valid for a
    single-chunk prompt; it is normalised to the documented legacy provenance."""
    rec = _record(features)  # cache_hit_tokens=0, num_scheduled_tokens == prompt_len, no pre-step count
    assert rec.cache_hit_tokens == 0
    assert rec.cache_hit_provenance == "legacy_single_chunk_known"
    with pytest.raises(features.FeatureError):
        _record(features, computed_tokens_before_step=0, cache_hit_tokens=0, cache_hit_provenance="unknown")
    with pytest.raises(features.FeatureError):
        _record(
            features,
            prompt_len=9000,
            prompt_tokens_count=9000,
            terminal_position=8999,
            num_scheduled_tokens=808,
            query_end=808,
            cache_hit_tokens=0,
            cache_hit_provenance="unknown",
        )


def test_record_rejects_nonfinite_vector_and_logits():
    rec = _record(features)
    bad_bits = rec.vector_bf16.copy()
    bad_bits[0] = features.fp32_to_bf16_bits(np.asarray([np.inf], dtype=np.float32))[0]
    with pytest.raises(features.FeatureError):
        _record(features, vector_bf16=bad_bits)
    with pytest.raises(features.FeatureError):
        _record(features, vector_bf16=features.fp32_to_bf16_bits(np.full(2560, np.nan, dtype=np.float32)))
    bad_logits = np.zeros(24, dtype=np.float32)
    bad_logits[7] = np.nan
    with pytest.raises(features.FeatureError):
        _record(features, verbalizer_logits_fp32=bad_logits)


def test_record_rejects_nonfinite_or_nonpositive_head_scale():
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(features.FeatureError):
            _record(features, head_scale=bad)


def test_record_rejects_bool_and_negative_accounting_indices():
    for field in ("batch_index", "internal_index", "logits_row_index", "query_start", "tp_rank"):
        with pytest.raises(features.FeatureError):
            _record(features, **{field: -1})
        with pytest.raises(features.FeatureError):
            _record(features, **{field: True})
    with pytest.raises(features.FeatureError):
        _record(features, world_size=0)
    with pytest.raises(features.FeatureError):
        _record(features, num_scheduled_tokens=0, query_end=0)
    with pytest.raises(features.FeatureError):
        _record(features, cache_hit_tokens=-1)


def test_decode_rejects_nonterminal_record_bytes():
    """The format gate applies on the decode path too, not only at construction."""
    rec = _record(features)
    blob = features.encode_record_npz(rec)
    tampered = features._rewrite_meta_for_tests(blob, {"terminal_position": 0})
    with pytest.raises(features.FeatureError):
        features.decode_record_npz(tampered)
    assert features.decode_record_npz(blob).terminal_position == 3


def test_fixture_escape_hatch_is_scoped_and_explicit():
    """Deliberately-invalid fixtures are constructible ONLY inside the named, scoped hatch."""
    rec = _record(features)
    blob = features.encode_record_npz(rec)
    tampered = features._rewrite_meta_for_tests(blob, {"terminal_position": 0})
    with pytest.raises(features.FeatureError):
        features.decode_record_npz(tampered)
    with features.fixture_records_allowed():
        assert features.decode_record_npz(tampered).terminal_position == 0
    with pytest.raises(features.FeatureError):
        features.decode_record_npz(tampered)


def test_record_meta_round_trip_preserves_accounting_fields():
    rec = _record(
        features,
        computed_tokens_before_step=8192,
        prompt_len=9000,
        prompt_tokens_count=9000,
        terminal_position=8999,
        num_scheduled_tokens=808,
        query_end=808,
        cache_hit_tokens=None,
        cache_hit_provenance="unknown",
        cache_hit_status_reason="not observable at the runner hook",
    )
    back = features.decode_record_npz(features.encode_record_npz(rec))
    assert back.computed_tokens_before_step == 8192
    assert back.cache_hit_tokens is None
    assert back.cache_hit_provenance == "unknown"
    assert back.cache_hit_status_reason == "not observable at the runner hook"
    assert back.meta()["computed_tokens_before_step"] == 8192


# ======================================================================================
# S3 final correction: a known cache count must be physically possible
# (RED first: reports/s3/final-count-bound/RED.txt)
# ======================================================================================


def test_pm_reproduction_impossible_known_cache_count_is_refused():
    """PM reproduction: a 4-token prompt cannot have had 9999 tokens reused from the prefix
    cache. The installed source recomputes the terminal token, so any non-null known count must
    be strictly less than prompt_len, for legacy and measured attribution alike."""
    import dataclasses

    rec = _record(features)
    assert rec.prompt_len == 4 and rec.cache_hit_tokens == 0
    with pytest.raises(features.FeatureError) as exc:
        dataclasses.replace(rec, cache_hit_tokens=9999, computed_tokens_before_step=None)
    assert "prompt_len" in str(exc.value) or "prompt" in str(exc.value)
    with pytest.raises(features.FeatureError):
        dataclasses.replace(
            rec,
            cache_hit_tokens=9999,
            computed_tokens_before_step=None,
            cache_hit_provenance=features.CACHE_PROVENANCE_MEASURED,
            cache_hit_source="synthetic_pm_fixture",
        )


def test_record_rejects_known_cache_count_at_or_above_prompt_len():
    """Boundary cases: prompt_len itself is impossible (the terminal token is recomputed),
    and so is anything larger."""
    for bad in (4, 5, 1000):
        with pytest.raises(features.FeatureError):
            _record(features, cache_hit_tokens=bad)  # legacy path (no pre-step count)
        with pytest.raises(features.FeatureError):
            _record(
                features,
                computed_tokens_before_step=2,
                num_scheduled_tokens=2,
                query_end=2,
                cache_hit_tokens=bad,
                cache_hit_provenance=features.CACHE_PROVENANCE_MEASURED,
                cache_hit_source="synthetic_measured",
            )
    # a multi-chunk record with a pre-step count larger than the prompt cannot exist either
    with pytest.raises(features.FeatureError):
        _record(
            features,
            prompt_len=9000,
            prompt_tokens_count=9000,
            terminal_position=8999,
            num_scheduled_tokens=808,
            query_end=808,
            computed_tokens_before_step=8999,
            cache_hit_tokens=8999,
            cache_hit_provenance=features.CACHE_PROVENANCE_MEASURED,
            cache_hit_source="synthetic_measured",
        )


def test_record_accepts_known_cache_counts_below_prompt_len():
    """Known 0 at prompt_len 1, and genuinely smaller measured counts, must still work."""
    one = _record(features, prompt_len=1, prompt_tokens_count=1, terminal_position=0,
                  num_scheduled_tokens=1, query_end=1, cache_hit_tokens=0)
    assert one.cache_hit_tokens == 0
    assert one.cache_hit_provenance == features.CACHE_PROVENANCE_LEGACY
    for good in (0, 1, 2):
        rec = _record(
            features,
            computed_tokens_before_step=2,
            num_scheduled_tokens=2,
            query_end=2,
            cache_hit_tokens=good,
            cache_hit_provenance=features.CACHE_PROVENANCE_MEASURED,
            cache_hit_source="synthetic_measured",
        )
        assert rec.cache_hit_tokens == good
    # the pre-existing relation is unchanged: a known count cannot exceed the pre-step count
    with pytest.raises(features.FeatureError):
        _record(
            features,
            computed_tokens_before_step=2,
            num_scheduled_tokens=2,
            query_end=2,
            cache_hit_tokens=3,
            cache_hit_provenance=features.CACHE_PROVENANCE_MEASURED,
            cache_hit_source="synthetic_measured",
        )


def test_impossible_cache_count_is_not_relaxed_by_the_fixture_hatch():
    """The hatch exists for deliberately-invalid fixtures (nonterminal position, non-finite
    numbers). A physically impossible reuse count is refused even there."""
    rec = _record(features)
    with pytest.raises(features.FeatureError):
        with features.fixture_records_allowed():
            _record(features, cache_hit_tokens=rec.prompt_len)


def test_unknown_cache_count_remains_null_and_is_never_replaced_by_zero():
    rec = _record(
        features,
        computed_tokens_before_step=0,
        cache_hit_tokens=None,
        cache_hit_provenance="unknown",
        cache_hit_status_reason="not observable at the runner hook",
    )
    assert rec.cache_hit_tokens is None
    assert rec.cache_hit_provenance == "unknown"
