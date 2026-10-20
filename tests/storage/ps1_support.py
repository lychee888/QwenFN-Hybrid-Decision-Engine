"""Shared helpers for the PS1 production-storage tests (CPU only, no torch, no network).

Nothing here touches a real capture root: every test builds its own directory under pytest's
`tmp_path`. The only files loaded are this repo's own sources, read-only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
DERIVATIVE = ROOT / "engine" / "qfn_capture_runtime_production.py"
ORIGINAL = ROOT / "tests" / "reference" / "m1.py"
RETENTION = ROOT / "engine" / "retention.py"

MARKER_PREFIX = "cmpl-qfn-"
PROMPT_HASH = "0123456789abcdef"
SLOTS = 24
# `ABSENT` means "the production_storage key is not in the config at all" (legacy mode);
# an explicit `None` means the JSON key IS present and null (which must be refused).
ABSENT = object()


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def derivative():
    return load_module("qfn_capture_runtime_production", DERIVATIVE)


def original():
    return load_module("qfn_capture_runtime_mtp1_orig_for_ps1", ORIGINAL)


def retention():
    return load_module("qfn_retention_ps1", RETENTION)


def record_key(req_id: str = "cmpl-qfn-dec-0", disambiguation: int = 0) -> str:
    """The real `record_file_key` naming rule (imported lazily to stay navigation-free)."""
    mod = sys.modules.get("qfn_capture_runtime_production") or derivative()
    return mod.record_file_key(
        req_id, prompt_hash=PROMPT_HASH, disambiguation=disambiguation, marker_prefix=MARKER_PREFIX
    )


def minimal_meta(req_id: str = "cmpl-qfn-dec-0", *, hidden: int = 2560, marker: str = "ps1") -> dict[str, Any]:
    """A small but schema-shaped record meta (the writer only needs arrays + json-safe scalars)."""
    import numpy as np

    return {
        "example_id": f"{req_id}__{PROMPT_HASH[:8]}",
        "engine_request_id": req_id,
        "prompt_token_sha256": PROMPT_HASH,
        "vector_bf16": np.arange(hidden, dtype=np.uint16),
        "vector_fp32": np.arange(hidden, dtype=np.float32),
        "verbalizer_ids": list(range(SLOTS)),
        "verbalizer_logits_fp32": np.linspace(0.0, 1.0, SLOTS, dtype=np.float32),
        "helper_version": "qfn-capture-runtime/1",
        "ps1_test_marker": marker,
    }


def write_record(capture_dir, key: str, **kw) -> int:
    """Write one record with the REAL writer (torch-free), returning the bytes it accounted."""
    mod = sys.modules.get("qfn_capture_runtime_production") or derivative()
    os.makedirs(str(capture_dir), exist_ok=True)
    return mod.write_record_files(str(capture_dir), key, minimal_meta(**kw))


def production_cfg(capture_dir, **over) -> dict[str, Any]:
    cfg = {
        "retention_root": str(capture_dir),
        "max_retained_records": 8,
        "max_retained_bytes": 8 << 20,
        "min_free_disk_bytes": 1 << 20,
        "max_record_bytes": 1 << 16,
        "grace_seconds": 3600,
        "ttl_seconds": 3600,
        "snapshot_ttl_seconds": 5.0,
    }
    cfg.update(over)
    return cfg


def capture_config_dict(capture_dir, *, production=ABSENT, **over) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "capture_dir": str(capture_dir),
        "marker_prefix": MARKER_PREFIX,
        "max_records": 4096,
        "max_bytes": 4 * 1024**3,
    }
    if production is not ABSENT:
        cfg["production_storage"] = production
    cfg.update(over)
    return cfg


def write_capture_config(path, capture_dir, *, production=ABSENT, **over) -> str:
    cfg = capture_config_dict(capture_dir, production=production, **over)
    pathlib.Path(path).write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


def make_state(capture_dir, *, production=ABSENT, **over):
    """A real `_CaptureState` built through the real config loader (no torch needed)."""
    mod = sys.modules.get("qfn_capture_runtime_production") or derivative()
    cfg = mod.CaptureConfig.from_dict(capture_config_dict(capture_dir, production=production, **over))
    return mod._CaptureState(enabled=True, config=cfg)


def write_one(state, i: int, **meta_kw) -> int:
    """Exactly the four statements `_capture_one` runs after building the meta (no torch needed).

    The pre-write `reserve_for_record` call is part of the same seam, so a production refusal
    propagates as `StorageRefused` here instead of being silently swallowed.
    """
    mod = sys.modules.get("qfn_capture_runtime_production") or derivative()
    key = record_key(req_id=f"cmpl-qfn-dec-{i}")
    reserve = state.production.reserve_for_record(key) if state.production is not None else None
    written = mod.write_record_files(
        state.config.capture_dir, key, minimal_meta(f"cmpl-qfn-dec-{i}", **meta_kw), reserve=reserve
    )
    state.records_written += 1
    state.bytes_written += written
    state.after_record_written(written)
    return written


def fill_until_stopped(state, *, limit: int = 50, **meta_kw) -> int:
    """Capture records through the real writer + the real quota seam until the run stops.

    A production refusal is applied exactly as `_capture`'s row handler does
    (`state.note_storage_refusal`), so this loop exercises the real seam, not a copy of it.
    """
    mod = sys.modules.get("qfn_capture_runtime_production") or derivative()
    n = 0
    for i in range(limit):
        if state.stopped:
            break
        try:
            write_one(state, i, **meta_kw)
        except mod.StorageRefused as exc:
            state.note_storage_refusal(exc)
            break
        n += 1
    return n
