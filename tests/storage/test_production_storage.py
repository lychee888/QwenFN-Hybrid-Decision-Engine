"""PS1: production storage accounting (bounded CURRENT retained state), guards and recovery.

CPU-only (numpy + stdlib, no torch, no engine, no network, no GPU). Every root is a pytest
`tmp_path`; nothing outside it is created or deleted.

Honesty note: the helper's own capture path (`capture_after_compute_logits` -> `_capture` ->
`_capture_one`) needs torch tensors, which are NOT installed on this host. The writer, the
storage layer and the two seam methods that `_capture`/`_capture_one` call
(`note_storage_refusal`, `after_record_written`) are exercised directly here; the seam delta
inside those two functions is proven structurally by `test_derivative_parity.py`.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import stat
import threading
import time

import numpy as np
import pytest

import ps1_support as S


@pytest.fixture(scope="module")
def prod():
    return S.derivative()


@pytest.fixture(scope="module")
def ret():
    return S.retention()


def _files(root, pattern="*"):
    return sorted(p.name for p in pathlib.Path(root).glob(pattern))


def _backdate(root, ns):
    for p in pathlib.Path(root).iterdir():
        os.utime(p, ns=(ns, ns))


# ======================================================================================
# 1. accounting at process start + bounded snapshot
# ======================================================================================

def test_start_of_process_accounts_records_that_already_exist(prod, tmp_path):
    cap = tmp_path / "cap"
    sizes = [S.write_record(cap, S.record_key(f"cmpl-qfn-dec-{i}")) for i in range(3)]
    assert len(_files(cap, "*.npz")) == 3
    assert all(s > 0 for s in sizes)

    state = S.make_state(cap, production=S.production_cfg(cap))
    snap = state.production.snapshot()
    assert snap["completed_records"] == 3
    assert snap["retained_records"] == 3
    on_disk = sum(
        os.path.getsize(cap / f) for f in _files(cap) if f.endswith(".npz") or f.endswith(".npz.json")
    )
    assert snap["completed_bytes"] == on_disk
    # "actual bytes incl JSON" - the accounted storage is the npz AND its committed sidecar
    assert snap["completed_bytes"] > sum(os.path.getsize(cap / f) for f in _files(cap, "*.npz"))
    assert snap["completed_bytes"] == sum(sizes)  # the writer's own accounting agrees


def test_legacy_mode_has_no_storage_layer_and_keeps_the_lifetime_counters(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap)
    assert state.production is None
    assert state.stopped is False
    assert state.storage_summary() is None


def test_snapshot_cache_is_bounded_and_rescans_after_the_ttl(prod, tmp_path):
    cap = tmp_path / "cap"
    S.write_record(cap, S.record_key("cmpl-qfn-dec-0"))
    state = S.make_state(cap, production=S.production_cfg(cap, snapshot_ttl_seconds=5.0))
    st = state.production
    t0 = 1_000_000_000_000_000_000
    first = st.snapshot(now_ns=t0)
    assert first["completed_records"] == 1 and st.scans == 1
    S.write_record(cap, S.record_key("cmpl-qfn-dec-1"))  # written by "someone else"
    cached = st.snapshot(now_ns=t0 + 1_000_000_000)  # 1 s later: still allowed to be cached
    assert st.scans == 1 and cached["completed_records"] == 1
    rescanned = st.snapshot(now_ns=t0 + 6_000_000_000)  # past the 5 s bound: must rescan
    assert st.scans == 2 and rescanned["completed_records"] == 2
    assert st.cfg.snapshot_ttl_seconds <= prod.MAX_SNAPSHOT_TTL_SECONDS


def test_one_scan_per_run_when_nothing_is_deleted(prod, tmp_path):
    """Overhead proof: the cached snapshot means 1 bounded scan for N record writes."""
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap))
    S.fill_until_stopped(state, limit=5)
    assert state.records_written == 5
    assert state.production.scans == 1
    assert state.production.snapshot()["completed_records"] == 5


def test_in_process_writes_are_not_undercounted_by_the_cache(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap))
    st = state.production
    assert st.snapshot(now_ns=10**18)["retained_records"] == 0
    st.note_written(4_096)
    r = st.snapshot(now_ns=10**18 + 1)  # cache is fresh, the delta must still show up
    assert r["retained_records"] == 1 and r["retained_bytes"] == 4_096


def test_external_gc_is_seen_after_invalidate(prod, tmp_path):
    cap = tmp_path / "cap"
    S.write_record(cap, S.record_key("cmpl-qfn-dec-0"))
    state = S.make_state(cap, production=S.production_cfg(cap))
    st = state.production
    assert st.snapshot()["retained_records"] == 1
    for f in pathlib.Path(cap).iterdir():
        f.unlink()
    st.invalidate()
    assert st.snapshot()["retained_records"] == 0


# ======================================================================================
# 2. legacy lifetime counter vs production retained accounting
# ======================================================================================

def test_legacy_lifetime_counter_still_refuses_after_the_files_are_deleted(prod, tmp_path):
    """PROSPECTIVE RED (the defect this stage exists for): the legacy path never recounts."""
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, max_records=2, max_bytes=1 << 30)
    assert state.production is None
    S.fill_until_stopped(state, limit=10)
    assert state.records_written == 3  # the legacy `>` allows exactly one record over the cap
    assert state.stopped is True
    assert state.stopped_reason is None  # the legacy path has no reason field to clear
    assert state.errors[-1]["kind"] == "quota_exceeded"
    assert state.errors[-1]["error"] == "capture quota reached; further captures are refused"
    assert state.errors[-1]["max_records"] == 2 and state.errors[-1]["max_bytes"] == 1 << 30

    for f in pathlib.Path(cap).iterdir():  # the operator TTL-deletes everything
        f.unlink()
    assert _files(cap) == []

    again = S.write_one(state, 99)
    assert again > 0  # the (policy-free) writer still writes
    assert state.stopped is True  # ... but the LIFETIME counter keeps the run stopped
    state.records_written -= 1  # undo the direct-writer accounting of this probe
    assert state.recover_storage_quota() is False  # nothing recoverable without production mode


def test_production_mode_continues_after_retention_frees_room(prod, ret, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap, max_retained_records=2))
    assert S.fill_until_stopped(state, limit=10) == 2
    assert state.records_written == 2
    # the 3rd record was refused BEFORE publication (the fill loop applied the real seam)
    assert state.stopped is True and state.stopped_reason == prod.STORAGE_QUOTA_REASON
    assert state.errors[-1]["kind"] == "storage_refused"
    assert state.errors[-1]["storage_reason"] == "retained_record_limit"
    assert _files(cap, "*.npz") == ["cmpl-qfn-dec-0__0123456789abcdef__0.npz",
                                    "cmpl-qfn-dec-1__0123456789abcdef__0.npz"]

    _backdate(cap, time.time_ns() - 10 * 3600 * 10**9)
    plan = ret.run(ret.RetentionPolicy(root=str(cap)), execute=True, now_ns=time.time_ns())
    assert plan.removed_records == 2 and plan.errors == []
    assert _files(cap, "*.npz") == [] and _files(cap, "*.npz.json") == []

    free_before = state.records_written
    assert state.recover_storage_quota() is True
    assert state.stopped is False and state.stopped_reason is None
    assert state.records_written == free_before  # the LIFETIME counter is not reset
    assert state.storage_recoveries == 1
    assert S.write_one(state, 2) > 0  # capturing continues after the GC
    assert _files(cap, "*.npz") == ["cmpl-qfn-dec-2__0123456789abcdef__0.npz"]


# ======================================================================================
# 3. exact bounds, refuse-before-publication
# ======================================================================================

def test_exact_retained_count_bound(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap, max_retained_records=3))
    S.fill_until_stopped(state, limit=10)
    assert state.records_written == 3
    assert state.production.snapshot(force=True)["completed_records"] == 3
    with pytest.raises(prod.StorageRefused) as err:
        S.write_one(state, 3)
    assert err.value.reason == "retained_record_limit"
    assert len(_files(cap, "*.npz")) == 3  # the refused record was never published


def test_exact_retained_byte_bound(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    probe = S.write_record(cap, S.record_key("cmpl-qfn-probe-0"))
    (cap / S.record_key("cmpl-qfn-probe-0")).unlink()
    (cap / (S.record_key("cmpl-qfn-probe-0") + ".json")).unlink()
    assert probe > 0 and _files(cap) == []

    per_record = probe + 512
    ps = S.production_cfg(cap, max_retained_records=99, max_record_bytes=per_record,
                          max_retained_bytes=2 * per_record)
    state = S.make_state(cap, production=ps)
    assert S.fill_until_stopped(state, limit=10) == 2
    assert state.stopped is True
    assert state.errors[-1]["kind"] == "storage_refused"
    assert state.errors[-1]["storage_reason"] == "retained_byte_limit"
    snap = state.production.snapshot(force=True)
    assert snap["completed_records"] == 2
    assert snap["retained_bytes"] <= ps["max_retained_bytes"]


def test_per_record_max_bytes_refuses_before_publication(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    ps = S.production_cfg(cap, max_record_bytes=64, max_retained_bytes=1 << 20)
    state = S.make_state(cap, production=ps)
    reserve = state.production.reserve_for_record(S.record_key("cmpl-qfn-dec-0"))
    with pytest.raises(prod.StorageRefused) as err:
        prod.write_record_files(str(cap), S.record_key("cmpl-qfn-dec-0"),
                                S.minimal_meta("cmpl-qfn-dec-0"), reserve=reserve)
    assert err.value.reason == "record_too_large"
    assert err.value.fatal_global is False
    assert _files(cap) == []  # neither the npz nor a .part survives


def test_declared_free_disk_reserve_refuses_before_publication(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    ps = S.production_cfg(cap, min_free_disk_bytes=1 << 50)
    state = S.make_state(cap, production=ps)
    with pytest.raises(prod.StorageRefused) as err:
        state.production.reserve_for_record(S.record_key("cmpl-qfn-dec-0"))
    assert err.value.reason == "free_disk_reserve" and err.value.fatal_global is True
    assert _files(cap) == []


def test_unknown_disk_usage_refuses_instead_of_assuming_room(prod, tmp_path, monkeypatch):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap))

    def boom(_path):
        raise OSError(1, "disk usage unavailable")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    with pytest.raises(prod.StorageRefused) as err:
        state.production.reserve_for_record(S.record_key("cmpl-qfn-dec-0"))
    assert err.value.reason == "free_disk_unavailable" and err.value.fatal_global is True
    assert _files(cap) == []


def test_listing_failure_refuses_instead_of_assuming_zero(prod, tmp_path, monkeypatch):
    cap = tmp_path / "cap"
    S.write_record(cap, S.record_key("cmpl-qfn-dec-0"))
    state = S.make_state(cap, production=S.production_cfg(cap))
    real_scandir = os.scandir

    def boom(_path):
        raise OSError(13, "permission denied")

    monkeypatch.setattr(os, "scandir", boom)
    with pytest.raises(prod.StorageRefused) as err:
        state.production.snapshot(force=True)
    assert err.value.reason == "scandir_failed" and err.value.fatal_global is True
    monkeypatch.setattr(os, "scandir", real_scandir)
    # the failed scan must not have been cached as "zero retained"
    assert state.production.snapshot(force=True)["completed_records"] == 1


def test_entry_stat_failure_refuses(prod, tmp_path, monkeypatch):
    cap = tmp_path / "cap"
    cap.mkdir()

    class _Entry:
        name = "cmpl-qfn-dec-0__0123456789abcdef__0.npz"

        def is_symlink(self):
            return False

        def stat(self, follow_symlinks=True):
            raise OSError(5, "stat failed")

    class _It:
        def __enter__(self):
            return iter([_Entry()])

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(os, "scandir", lambda _p: _It())
    with pytest.raises(prod.StorageRefused) as err:
        prod.scan_capture_root(str(cap), marker_prefix=S.MARKER_PREFIX)
    assert err.value.reason == "entry_stat_failed" and err.value.fatal_global is True


def test_root_symlink_and_missing_root_refuse(prod, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted on this host")
    with pytest.raises(prod.StorageRefused) as err:
        prod.scan_capture_root(str(link), marker_prefix=S.MARKER_PREFIX)
    assert err.value.reason == "root_is_symlink"
    with pytest.raises(prod.StorageRefused) as err2:
        prod.scan_capture_root(str(tmp_path / "nope"), marker_prefix=S.MARKER_PREFIX)
    assert err2.value.reason == "root_stat_failed"


def test_symlinked_entries_are_skipped_by_the_accounting(prod, tmp_path, monkeypatch):
    """Symlink entries are counted apart and never followed (synthetic entry: see the skip above)."""
    cap = tmp_path / "cap"
    S.write_record(cap, S.record_key("cmpl-qfn-dec-0"))
    real_scandir = os.scandir

    class _Link:
        name = S.record_key("cmpl-qfn-dec-0")  # a real-looking name, but a symlink
        path = str(cap / name)

        def is_symlink(self):
            return True

        def stat(self, follow_symlinks=True):
            # a REAL stat_result (all fields present, incl. st_mtime_ns): the product must never
            # see a synthetic shape that no filesystem produces
            return os.stat(self.path)

    class _It:
        def __enter__(self):
            return iter([_Link()])

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(os, "scandir", lambda _p: _It())
    snap = prod.scan_capture_root(str(cap), marker_prefix=S.MARKER_PREFIX)
    assert snap["symlinks_skipped"] == 1
    assert snap["completed_records"] == 0 and snap["incomplete_records"] == 0
    monkeypatch.setattr(os, "scandir", real_scandir)
    assert prod.scan_capture_root(str(cap), marker_prefix=S.MARKER_PREFIX)["completed_records"] == 1


def test_accounting_overflow_refuses_rather_than_truncating(prod, tmp_path):
    cap = tmp_path / "cap"
    for i in range(3):
        S.write_record(cap, S.record_key(f"cmpl-qfn-dec-{i}"))
    with pytest.raises(prod.StorageRefused) as err:
        prod.scan_capture_root(str(cap), marker_prefix=S.MARKER_PREFIX, max_files=2)
    assert err.value.reason == "accounting_overflow"


# ======================================================================================
# 4. in-flight writes, duplicates, atomic handshake
# ======================================================================================

def test_duplicate_record_key_is_refused_never_clobbered(prod, tmp_path):
    cap = tmp_path / "cap"
    key = S.record_key("cmpl-qfn-dec-0")
    S.write_record(cap, key)
    before = (cap / key).read_bytes()
    state = S.make_state(cap, production=S.production_cfg(cap))
    with pytest.raises(prod.StorageRefused) as err:
        state.production.reserve_for_record(key)
    assert err.value.reason == "duplicate_record_key" and err.value.fatal_global is False
    assert (cap / key).read_bytes() == before  # untouched


def test_orphan_sidecar_also_blocks_a_rewrite(prod, tmp_path):
    cap = tmp_path / "cap"
    key = S.record_key("cmpl-qfn-dec-0")
    S.write_record(cap, key)
    (cap / key).unlink()  # npz gone, the committed sidecar remains
    state = S.make_state(cap, production=S.production_cfg(cap))
    with pytest.raises(prod.StorageRefused) as err:
        state.production.reserve_for_record(key)
    assert err.value.reason == "duplicate_record_key"
    assert state.production.snapshot(force=True)["incomplete_records"] == 1


def test_in_flight_part_files_are_counted_apart_and_never_block(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    (cap / (S.record_key("cmpl-qfn-dec-0") + ".part")).write_bytes(b"half a record")
    state = S.make_state(cap, production=S.production_cfg(cap))
    snap = state.production.snapshot(force=True)
    assert snap["inflight_parts"] == 1 and snap["inflight_bytes"] == len(b"half a record")
    assert snap["completed_records"] == 0 and snap["retained_records"] == 0
    assert S.write_one(state, 0) > 0  # an in-flight temp never blocks a real write


def test_part_then_replace_handshake_is_atomic_for_a_concurrent_reader(prod, tmp_path):
    """A reader must never observe a torn/partial record while it is being rewritten.

    The reader re-opens the path for each probe and validates the full npz + json from the bytes
    it got, so the published file is only ever the old or the new one, never a mixture. On
    Windows a replace can transiently fail while another handle is open, so the writer retries;
    that is a Windows file-locking artifact of the TEST, not a helper behaviour.
    """
    cap = tmp_path / "cap"
    cap.mkdir()
    key = S.record_key("cmpl-qfn-dec-0")
    npz_path, json_path = cap / key, cap / (key + ".json")
    prod.write_record_files(str(cap), key, S.minimal_meta("cmpl-qfn-dec-0", marker="v0"))
    stop = threading.Event()
    errors: list[str] = []
    reads = [0]

    def reader():
        while not stop.is_set():
            try:
                npz_bytes = npz_path.read_bytes()
                json_bytes = json_path.read_bytes()
                with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as z:
                    assert "vector_bf16" in z and z["vector_bf16"].shape == (2560,)
                assert isinstance(json.loads(json_bytes.decode("utf-8")), dict)
                reads[0] += 1
            except FileNotFoundError:
                pass
            except PermissionError:
                continue  # Windows sharing violation while the writer replaces the file
            except Exception as exc:  # a torn read lands here
                errors.append(repr(exc))
                return

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    written = 0
    last_marker = "v0"
    try:
        for i in range(40):
            for _ in range(50):  # Windows: retry while the reader holds the file open
                try:
                    prod.write_record_files(str(cap), key,
                                            S.minimal_meta("cmpl-qfn-dec-0", marker=f"v{i}"))
                    written += 1
                    last_marker = f"v{i}"
                    break
                except PermissionError:
                    time.sleep(0.001)
    finally:
        stop.set()
        thread.join(timeout=20)
    assert errors == []
    assert reads[0] > 0
    assert written >= 30, written
    assert not list(pathlib.Path(cap).glob("*.part"))
    assert json.loads((cap / (key + ".json")).read_text(encoding="utf-8"))["ps1_test_marker"] == last_marker


# ======================================================================================
# 5. stop / recovery must never clear unrelated errors
# ======================================================================================

def test_recovery_does_not_clear_unrelated_errors_and_keeps_the_lifetime_counter(prod, ret, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap, max_retained_records=1))
    state.record_error("no_torch", "torch is not importable; capture disabled at runtime")
    assert S.fill_until_stopped(state, limit=5) == 1
    with pytest.raises(prod.StorageRefused) as err:
        S.write_one(state, 1)
    state.note_storage_refusal(err.value)
    assert state.stopped is True

    _backdate(cap, time.time_ns() - 10 * 3600 * 10**9)
    ret.run(ret.RetentionPolicy(root=str(cap)), execute=True)
    assert state.recover_storage_quota() is True

    kinds = [e["kind"] for e in state.errors]
    assert kinds[0] == "no_torch" and "no_torch" in kinds  # never cleared
    payload = json.loads((pathlib.Path(cap) / "CAPTURE_ERROR.json").read_text(encoding="utf-8"))
    assert payload["stopped"] is False
    assert payload["last_error"]["kind"] == "no_torch"
    assert payload["storage_quota_recovered"] is True
    assert payload["records_written"] == state.records_written


def test_a_non_storage_stop_is_never_recoverable(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap))
    state.record_error("no_torch", "torch is not importable")
    state.stopped = True  # exactly what the original _capture does for no_torch
    assert state.recover_storage_quota() is False
    assert state.stopped is True and state.stopped_reason is None


def test_recovery_is_rate_limited_by_the_snapshot_ttl(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap, snapshot_ttl_seconds=5.0))
    state.stopped = True
    state.stopped_reason = prod.STORAGE_QUOTA_REASON
    t0 = 1_000_000_000_000_000_000
    assert state.recover_storage_quota(now_ns=t0) is True  # 0 retained -> recovers
    assert state.production.scans == 1
    state.stopped = True
    state.stopped_reason = prod.STORAGE_QUOTA_REASON
    state.recover_storage_quota(now_ns=t0 + 1_000_000_000)  # inside the TTL: no forced rescan
    assert state.production.scans == 1
    state.stopped = True
    state.stopped_reason = prod.STORAGE_QUOTA_REASON
    state.recover_storage_quota(now_ns=t0 + 20_000_000_000)  # past the TTL: rescan allowed
    assert state.production.scans == 2


def test_marker_key_set_is_a_superset_of_the_legacy_marker(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap))
    state.record_error("row_capture_failed", "boom")
    legacy = json.loads((pathlib.Path(cap) / "CAPTURE_ERROR.json").read_text(encoding="utf-8"))
    state.stopped = True
    state.stopped_reason = prod.STORAGE_QUOTA_REASON
    state.record_error("quota_exceeded", "capture quota reached; further captures are refused",
                       {"max_records": 1, "max_bytes": 1})
    assert state.recover_storage_quota() is True
    after = json.loads((pathlib.Path(cap) / "CAPTURE_ERROR.json").read_text(encoding="utf-8"))
    assert set(legacy) <= set(after)
    assert after["storage_quota_recovered"] is True


def test_storage_summary_reports_bounds_and_overhead_counters(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    state = S.make_state(cap, production=S.production_cfg(cap, max_retained_records=4))
    S.fill_until_stopped(state, limit=4)
    s = state.storage_summary()
    assert s["schema"] == prod.PRODUCTION_STORAGE_SCHEMA
    assert s["max_retained_records"] == 4 and s["max_record_bytes"] == 1 << 16
    assert s["min_free_disk_bytes"] == 1 << 20
    assert s["snapshot"]["completed_records"] == 4
    assert s["scans"] == 1 and s["refusals"] == 0 and s["recoveries"] == 0
    assert prod.MAX_SNAPSHOT_TTL_SECONDS >= s["snapshot_ttl_seconds"] > 0


# ======================================================================================
# 6. capacity reservations: actual, bounded, released exactly once (R1 correction)
# ======================================================================================

def _storage(prod, root, **over):
    cfg = prod.ProductionStorageConfig.from_dict(S.production_cfg(root, **over),
                                                 capture_dir=str(root))
    return prod.ProductionStorage(cfg)


def test_sequential_reservations_cannot_overbook(prod, tmp_path):
    """PM contract 1: a reservation that is not yet published still holds capacity."""
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap, max_retained_records=1)
    first = st.reserve_for_record(S.record_key("cmpl-qfn-pm-one"))
    assert first.state == "pending"
    with pytest.raises(prod.StorageRefused) as err:
        st.reserve_for_record(S.record_key("cmpl-qfn-pm-two"))
    assert err.value.reason == "retained_record_limit"
    snap = st.snapshot(force=True)
    assert snap["pending_records"] == 1 and snap["retained_records"] == 1
    assert not list(cap.glob("*.npz"))  # nothing was published by either reservation
    first.release()
    assert st.snapshot(force=True)["pending_records"] == 0


def test_same_key_cannot_be_reserved_twice_while_pending(prod, tmp_path):
    """PM contract 2: a pending key is exclusive, so two writers cannot both publish it."""
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap)
    key = S.record_key("cmpl-qfn-pm-one")
    st.reserve_for_record(key)
    with pytest.raises(prod.StorageRefused) as err:
        st.reserve_for_record(key)
    assert err.value.reason == "duplicate_pending_key"
    assert err.value.fatal_global is False  # one record, not a global stop
    st.release_all_pending()
    st.reserve_for_record(key)  # nothing was published, so the key is free again
    assert st.pending_state()["pending_records"] == 1


def test_release_is_idempotent_and_never_double_counts(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap, max_retained_records=1)
    reserve = st.reserve_for_record(S.record_key("cmpl-qfn-dec-0"))
    reserve.release()
    reserve.release()
    assert st.pending_released == 1
    assert reserve.state == "released"
    snap = st.snapshot(force=True)
    assert snap["pending_records"] == 0 and snap["retained_records"] == 0


def test_commit_counts_the_actual_bytes_exactly_once(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap)
    key = S.record_key("cmpl-qfn-dec-0")
    reserve = st.reserve_for_record(key)
    written = prod.write_record_files(str(cap), key, S.minimal_meta("cmpl-qfn-dec-0"),
                                      reserve=reserve)
    assert reserve.state == "committed"
    assert st.pending_committed_bytes == written  # the ACTUAL published size, recorded at commit
    st.note_written(written)                      # exactly what the _CaptureState seam does
    snap = st.snapshot(force=True)
    assert snap["pending_records"] == 0
    assert snap["retained_records"] == 1
    assert snap["retained_bytes"] == written      # counted once, not twice
    reserve.commit(written)                       # idempotent
    assert st.pending_committed_bytes == written
    assert st.snapshot(force=True)["retained_bytes"] == written


def test_failed_publication_releases_the_reservation_and_the_key(prod, tmp_path, monkeypatch):
    """An injected write failure must release capacity AND the key, exactly once."""
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap, max_retained_records=1)
    key = S.record_key("cmpl-qfn-dec-0")
    reserve = st.reserve_for_record(key)
    real_replace = os.replace

    def boom(*_a, **_k):
        raise OSError(5, "injected publication failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        prod.write_record_files(str(cap), key, S.minimal_meta("cmpl-qfn-dec-0"), reserve=reserve)
    assert reserve.state == "released"
    assert st.pending_released == 1
    assert st.snapshot(force=True)["pending_records"] == 0
    assert not list(cap.glob("*.npz"))
    monkeypatch.setattr(os, "replace", real_replace)
    again = st.reserve_for_record(key)  # capacity and the key are both free again
    assert again.state == "pending"
    assert prod.write_record_files(str(cap), key, S.minimal_meta("cmpl-qfn-dec-0"),
                                   reserve=again) > 0


def test_guard_refusal_releases_the_reservation(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap, max_record_bytes=64)
    key = S.record_key("cmpl-qfn-dec-0")
    reserve = st.reserve_for_record(key)
    with pytest.raises(prod.StorageRefused) as err:
        prod.write_record_files(str(cap), key, S.minimal_meta("cmpl-qfn-dec-0"), reserve=reserve)
    assert err.value.reason == "record_too_large"
    assert reserve.state == "released"
    assert st.pending_released == 1
    assert st.snapshot(force=True)["pending_records"] == 0
    assert not list(cap.glob("*.npz")) and not list(cap.glob("*.part"))


def test_pending_is_counted_across_a_forced_rescan(prod, tmp_path):
    """A snapshot recomputed while reservations are pending must not undercount them."""
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap, max_retained_records=2)
    state = S.make_state(cap, production=S.production_cfg(cap, max_retained_records=2))
    S.write_one(state, 0)                       # 1 published record, counted by the scan
    st.reserve_for_record(S.record_key("cmpl-qfn-pending-0"))
    snap = st.snapshot(force=True)
    assert snap["completed_records"] == 1 and snap["pending_records"] == 1
    assert snap["retained_records"] == 2
    with pytest.raises(prod.StorageRefused):
        st.reserve_for_record(S.record_key("cmpl-qfn-pending-1"))
    # deleting the published pair externally must NOT free the capacity a pending key holds:
    # one further reservation fits (1 pending + 1), a second one does not
    for p in pathlib.Path(cap).glob("*"):
        if p.suffix == ".npz" or p.name.endswith(".npz.json"):
            p.unlink()
    snap = st.snapshot(force=True)
    assert snap["completed_records"] == 0 and snap["pending_records"] == 1
    assert snap["retained_records"] == 1
    st.reserve_for_record(S.record_key("cmpl-qfn-pending-1"))
    with pytest.raises(prod.StorageRefused):
        st.reserve_for_record(S.record_key("cmpl-qfn-pending-2"))
    st.release_all_pending()
    st.reserve_for_record(S.record_key("cmpl-qfn-pending-2"))


def test_concurrent_writers_never_overbook(prod, tmp_path):
    """Real concurrent writers (test threads) must never exceed the declared bound."""
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap, max_retained_records=3)
    outcomes: list[tuple[str, str]] = []
    guard = threading.Lock()

    def worker(i: int) -> None:
        key = S.record_key(f"cmpl-qfn-w{i}")
        try:
            reserve = st.reserve_for_record(key)
        except prod.StorageRefused as exc:
            with guard:
                outcomes.append(("refused", exc.reason))
            return
        written = prod.write_record_files(str(cap), key, S.minimal_meta(f"cmpl-qfn-w{i}"),
                                          reserve=reserve)
        st.note_written(written)
        with guard:
            outcomes.append(("published", key))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    published = [o for o in outcomes if o[0] == "published"]
    refused = [o for o in outcomes if o[0] == "refused"]
    assert len(published) == 3, outcomes
    assert {r[1] for r in refused} == {"retained_record_limit"}, outcomes
    snap = st.snapshot(force=True)
    assert snap["pending_records"] == 0 and snap["retained_records"] == 3
    assert len(list(pathlib.Path(cap).glob("*.npz"))) == 3


def test_a_leaked_reservation_expires_after_the_bounded_ttl(prod, tmp_path):
    """The backstop: a reservation that is never committed/released is dropped, disclosed."""
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap, max_retained_records=1, pending_ttl_seconds=60)
    assert prod.MIN_PENDING_TTL_SECONDS >= 1
    t0 = 10**18
    st.reserve_for_record(S.record_key("cmpl-qfn-a-0"), now_ns=t0)
    assert st.snapshot(force=True, now_ns=t0)["pending_records"] == 1
    with pytest.raises(prod.StorageRefused):
        st.reserve_for_record(S.record_key("cmpl-qfn-b-0"), now_ns=t0)
    st.reserve_for_record(S.record_key("cmpl-qfn-b-0"), now_ns=t0 + 61 * 10**9)
    assert st.pending_expired == 1
    assert st.pending_state(now_ns=t0 + 61 * 10**9)["pending_records"] == 1


def test_unowned_record_keys_are_refused(prod, tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    st = _storage(prod, cap)
    for bad in ("../escape.npz", "not-a-record.npz", "cmpl-qfn-x-0__zzz__0.npz",
                "cmpl-qfn-x-0__0123456789abcdef__0.txt", "cmpl-qfn-x-0__0123456789abcdef__0.npz.json"):
        with pytest.raises(prod.StorageRefused) as err:
            st.reserve_for_record(bad)
        assert err.value.reason == "unowned_record_key", bad
    assert st.snapshot(force=True)["pending_records"] == 0


def test_the_standalone_writer_path_is_unchanged(prod, tmp_path):
    """`reserve=None` keeps the accepted writer and touches no reservation state."""
    cap = tmp_path / "cap"
    cap.mkdir()
    key = S.record_key("cmpl-qfn-dec-0")
    written = prod.write_record_files(str(cap), key, S.minimal_meta("cmpl-qfn-dec-0"))
    assert written > 0
    assert (cap / key).exists() and (cap / (key + ".json")).exists()
    assert not list(pathlib.Path(cap).glob("*.part"))
