"""PS1: the production-storage config must be strict and refuse BEFORE any I/O or mutation.

CPU-only (numpy + stdlib). Loads the derivative helper by path; no torch, no engine, no network.
"""

from __future__ import annotations

import json

import pytest

import ps1_support as S


@pytest.fixture(scope="module")
def prod():
    return S.derivative()


def _cfg(capture_dir, production=S.ABSENT, **over):
    return S.capture_config_dict(capture_dir, production=production, **over)


def test_production_block_absent_is_legacy_mode(prod, tmp_path):
    cfg = prod.CaptureConfig.from_dict(_cfg(tmp_path / "cap"))
    assert cfg.production_storage is None
    state = prod._CaptureState(enabled=True, config=cfg)
    assert state.production is None
    # the legacy lifetime quotas are the ones that apply
    assert cfg.max_records == 4096 and cfg.max_bytes == 4 * 1024**3


def test_production_block_present_is_parsed_strictly(prod, tmp_path, ):
    cap = tmp_path / "cap"
    ps = S.production_cfg(cap, max_retained_records=3, max_retained_bytes=1 << 20)
    cfg = prod.CaptureConfig.from_dict(_cfg(cap, ps))
    assert cfg.production_storage is not None
    assert cfg.production_storage.max_retained_records == 3
    assert cfg.production_storage.retention_root == str(cap)
    state = prod._CaptureState(enabled=True, config=cfg)
    assert isinstance(state.production, prod.ProductionStorage)
    assert state.production.root == str(cap)


_MISSING = object()

# Every field is individually wrong in exactly one way. `_MISSING` removes the key entirely.
BAD = {
    "missing_max_retained_records": {"max_retained_records": _MISSING},
    "missing_max_retained_bytes": {"max_retained_bytes": _MISSING},
    "missing_min_free_disk_bytes": {"min_free_disk_bytes": _MISSING},
    "missing_max_record_bytes": {"max_record_bytes": _MISSING},
    "missing_retention_root": {"retention_root": _MISSING},
    "zero_records": {"max_retained_records": 0},
    "negative_records": {"max_retained_records": -4},
    "negative_bytes": {"max_retained_bytes": -1},
    "zero_reserve": {"min_free_disk_bytes": 0},
    "zero_max_record_bytes": {"max_record_bytes": 0},
    "bool_is_not_an_int": {"max_retained_records": True},
    "string_int": {"max_retained_records": "8"},
    "float_int": {"max_retained_records": 8.0},
    "float_bytes": {"max_retained_bytes": 1.5},
    "nonfinite_nan": {"max_retained_records": float("nan")},
    "nonfinite_inf": {"max_retained_bytes": float("inf")},
    "none_value": {"max_record_bytes": None},
    "grace_too_small": {"grace_seconds": 3599},
    "ttl_too_small": {"ttl_seconds": 60},
    "grace_zero": {"grace_seconds": 0},
    "snapshot_ttl_too_large": {"snapshot_ttl_seconds": 30.5},
    "snapshot_ttl_zero": {"snapshot_ttl_seconds": 0},
    "snapshot_ttl_negative": {"snapshot_ttl_seconds": -1.0},
    "unknown_key": {"max_retained_byte": 1 << 20},
    "retention_root_relative": {"retention_root": "capture"},
    "retention_root_escape": {"retention_root": "../elsewhere"},
    "retention_root_other_dir": {"retention_root": "<<other-dir>>"},
    "retention_root_empty": {"retention_root": "   "},
}

# The whole block is the wrong TYPE (it must be a JSON object).
BAD_RAW = {"not_a_dict_list": ["x"], "not_a_dict_str": "on", "not_a_dict_int": 5, "not_a_dict_null": None}


@pytest.mark.parametrize("name", sorted(BAD) + sorted(BAD_RAW))
def test_bad_production_config_refuses_before_any_io(prod, tmp_path, name):
    cap = tmp_path / "cap"
    assert not cap.exists()
    if name in BAD_RAW:
        production = BAD_RAW[name]
    else:
        production = S.production_cfg(cap)
        for key, value in BAD[name].items():
            if value is _MISSING:
                production.pop(key, None)
            elif value == "<<other-dir>>":
                production[key] = str(tmp_path / "elsewhere")
            else:
                production[key] = value
    with pytest.raises(prod.SelectionError):
        prod.CaptureConfig.from_dict(_cfg(cap, production))
    assert not cap.exists(), "a refused config must not create or touch the capture dir"


def test_bad_production_config_disables_capture_with_a_config_error(prod, tmp_path):
    cap = tmp_path / "cap"
    ps = S.production_cfg(cap, max_retained_records=-1)
    p = tmp_path / "capture.json"
    S.write_capture_config(p, cap, production=ps)
    state = prod._build_state(
        {prod.ENV_ENABLE: "1", prod.ENV_CONFIG: str(p)}
    )
    assert state.enabled is False
    assert state.config_error and "production_storage" in state.config_error
    assert not cap.exists()


def test_good_production_config_keeps_the_legacy_limits_visible(prod, tmp_path):
    cap = tmp_path / "cap"
    ps = S.production_cfg(cap)
    cfg = prod.CaptureConfig.from_dict(_cfg(cap, ps, max_records=7, max_bytes=1234))
    # the legacy lifetime limits are still parsed and reported; production mode does not
    # silently replace them, it adds a second, retained-storage bound
    assert (cfg.max_records, cfg.max_bytes) == (7, 1234)
    assert cfg.production_storage.max_retained_records == ps["max_retained_records"]


def test_production_config_round_trips_through_json(prod, tmp_path):
    cap = tmp_path / "cap"
    ps = S.production_cfg(cap, max_retained_records=5)
    p = tmp_path / "capture.json"
    S.write_capture_config(p, cap, production=ps)
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["production_storage"]["max_retained_records"] == 5
    cfg = prod.CaptureConfig.from_file(p)
    assert cfg.production_storage.max_retained_records == 5
    assert cfg.production_storage.to_dict()["min_free_disk_bytes"] == ps["min_free_disk_bytes"]


def test_the_original_helper_ignores_the_new_block(tmp_path):
    """The serialized config stays forward-compatible: the ORIGINAL helper ignores the new key."""
    orig = S.original()
    cap = tmp_path / "cap"
    cfg = orig.CaptureConfig.from_dict(_cfg(cap, S.production_cfg(cap)))
    assert not hasattr(cfg, "production_storage")


def test_the_shipped_example_config_parses(prod):
    """The delivered example must itself be a valid strict config (and keep the legacy limits)."""
    cfg = prod.CaptureConfig.from_file(S.ROOT / "configs" / "capture.example.json")
    assert cfg.production_storage is not None
    assert cfg.production_storage.max_retained_records == 4096
    assert cfg.production_storage.pending_ttl_seconds == 300
    assert cfg.production_storage.snapshot_ttl_seconds == 5.0
    assert (cfg.max_records, cfg.max_bytes) == (4096, 4294967296)
    assert cfg.marker_prefix == "cmpl-qfn-"
