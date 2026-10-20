"""WIDE1 slice 5: the optional, explicit ``wide_choice`` gateway/capture configuration.

Absent (or disabled) must leave every legacy behaviour and every legacy API test unchanged.
Enabled must validate the catalog, the tokenizer revision, the content digest and the pinned prompt
digest BEFORE the gateway opens a listener or the helper captures anything.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

from qfn_decision import backend, configure, wide

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "wide-label-catalog.json"
CATALOG = wide.WideCatalog.from_file(CATALOG_PATH)


def _default() -> dict:
    return json.loads((ROOT / "configs" / "decision_api.json").read_text(encoding="utf-8"))


def _block(**over) -> dict:
    block = {
        "enabled": True,
        "catalog_path": str(CATALOG_PATH),
        "catalog_file_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        "catalog_digest": CATALOG.digest,
        "tokenizer_revision": CATALOG.tokenizer_revision,
        "template_digest": wide.WIDE_TEMPLATE_CONTENT_SHA256,
        "max_options": 256,
    }
    block.update(over)
    return block


def _config(tmp_path: pathlib.Path, block: dict | None, **over) -> backend.ApiConfig:
    cfg = _default()
    capture_dir = tmp_path / "capture"
    cfg["capture"]["dir"] = str(capture_dir)
    cfg["capture"]["dir_allowlist"] = [str(capture_dir)]
    if block is not None:
        cfg["wide_choice"] = block
    cfg.update(over)
    path = tmp_path / "decision_api.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return backend.load_api_config(path)


def test_absent_block_changes_nothing(tmp_path):
    cfg = _config(tmp_path, None)
    assert cfg.wide_choice is None
    assert backend.ApiConfig.from_dict(_default()).wide_choice is None
    # the provider config carries no wide config either
    assert cfg.capture_provider_config().wide_choice is None


def test_enabled_block_loads_the_validated_catalog(tmp_path):
    cfg = _config(tmp_path, _block())
    wc = cfg.wide_choice
    assert wc is not None and wc.enabled
    assert wc.catalog is not None and wc.catalog.digest == CATALOG.digest
    assert wc.max_options == 256
    assert wc.template_digest == wide.WIDE_TEMPLATE_CONTENT_SHA256
    provided = cfg.capture_provider_config()
    assert provided.wide_choice is not None and provided.wide_choice.enabled


def test_disabled_block_does_not_require_the_catalog_to_exist(tmp_path):
    cfg = _config(tmp_path, {"enabled": False, "catalog_path": "C:/does/not/exist.json"})
    wc = cfg.wide_choice
    assert wc is not None and not wc.enabled and wc.catalog is None


@pytest.mark.parametrize("mutate", [
    {"catalog_file_sha256": "0" * 64},
    {"catalog_digest": "0" * 64},
    {"tokenizer_revision": "0" * 40},
    {"template_digest": "0" * 64},
    {"template_digest": "not-hex"},
    {"catalog_path": "relative/catalog.json"},
    {"catalog_path": "C:/nope/missing.json"},
    {"max_options": 16},
    {"max_options": 257},
    {"max_options": True},
    {"max_options": 17.5},
    {"enabled": "yes"},
    {"unexpected": 1},
])
def test_enabled_block_fails_closed_on_drift(tmp_path, mutate):
    with pytest.raises(backend.ConfigError):
        _config(tmp_path, _block(**mutate))


def test_enabled_block_refuses_a_changed_catalog_even_with_recomputed_file_hash(tmp_path):
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    payload["labels"] = [*payload["labels"][:-1], "JV"]
    changed = tmp_path / "changed-catalog.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    block = _block(
        catalog_path=str(changed),
        catalog_file_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
    )
    with pytest.raises(backend.ConfigError):
        _config(tmp_path, block)


def test_block_tokenizer_revision_must_match_the_gateway_tokenizer(tmp_path):
    cfg = _default()
    cfg["tokenizer"]["revision"] = "different-revision"
    cfg["wide_choice"] = _block()
    path = tmp_path / "decision_api.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(backend.ConfigError):
        backend.load_api_config(path)


def test_config_reports_wide_status_without_claiming_live_support(tmp_path):
    server = None
    try:
        cfg = _config(tmp_path, _block())
        server = backend.DecisionBackend(cfg, provider=backend.FixtureFeatureProvider(
            capture_dir=tmp_path / "cap"))
        health = server.health()
        block = health["wide_choice"]
        assert block["enabled"] is True
        assert block["mode"] == "wide-labels"
        assert block["n_options_max"] == 256
        assert block["native_choice_slots"] == 16
        assert block["calibrated_confidence_claim"] is False
        assert block["runtime_verified"] is False
        assert block["catalog"]["digest"] == CATALOG.digest
        assert "slot_probabilities" not in json.dumps(block)
    finally:
        if server is not None:
            del server


def test_health_omits_the_wide_block_when_unconfigured(tmp_path):
    cfg = _config(tmp_path, None)
    backend_instance = backend.DecisionBackend(
        cfg, provider=backend.FixtureFeatureProvider(capture_dir=tmp_path / "cap2"))
    assert "wide_choice" not in backend_instance.health()


# ---------------------------------------------------------------- generator -> real consumer


def _engine_helper():
    """Load the ACTUAL standalone engine helper (the real consumer of the capture config)."""
    name = "qfn_r2_generated_config_helper"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "engine" / "qfn_capture_runtime_production.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_wide_capture_config_loads_in_the_actual_helper(tmp_path):
    """The generated capture block must be exactly what the engine helper accepts, unedited."""
    gateway, capture = configure.build_configs({
        "CAPTURE_DIR": str(tmp_path.resolve()),
        "WIDE_CHOICE": "1",
        "WIDE_CATALOG_PATH": str(CATALOG_PATH),
    })
    # the gateway keeps its own option cap...
    assert gateway["wide_choice"]["max_options"] == 256
    # ...while the engine block carries the shared pins and no gateway-only cap
    assert "max_options" not in capture["wide_choice"]
    for key in ("catalog_path", "catalog_file_sha256", "catalog_digest",
                "tokenizer_revision", "template_digest", "enabled"):
        assert capture["wide_choice"][key] == gateway["wide_choice"][key]

    loaded = _engine_helper().CaptureConfig.from_dict(capture)
    assert loaded.wide_choice is not None and loaded.wide_choice.enabled is True
    assert loaded.wide_choice.label_count == 256
    assert loaded.wide_choice.catalog_digest == gateway["wide_choice"]["catalog_digest"]
    assert loaded.wide_choice.token_ids[:16] == tuple(range(32, 48))
    # the gateway side of the same generator output still loads unchanged
    backend.ApiConfig.from_dict(gateway)


def test_generated_legacy_capture_config_still_loads_in_the_actual_helper(tmp_path):
    gateway, capture = configure.build_configs({"CAPTURE_DIR": str(tmp_path.resolve())})
    assert "wide_choice" not in gateway and "wide_choice" not in capture
    loaded = _engine_helper().CaptureConfig.from_dict(capture)
    assert loaded.wide_choice is None
