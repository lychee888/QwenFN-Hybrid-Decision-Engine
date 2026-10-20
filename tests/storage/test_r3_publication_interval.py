"""PS1-R3: the ACTUAL publication interval must retain exclusive capacity.

CPU-only. Written BEFORE the R3 fix (must fail on the R2 snapshot).

The PM counterexample (`test_ps1_r2_publication_interval.py`) pauses the FIRST `os.replace` and
advances the injected clock past the pending lease. These cases go further: they pin the whole
interval (first replace -> json write -> second replace), the failure cleanup, the stalled-writer
policy, the operator-force policy against an ACTIVE entry, eventual capacity release, and the
absence of leaked threads/locks.
"""

from __future__ import annotations

import pathlib
import threading

import pytest

import ps1_support as S

SECOND_NS = 10**9


@pytest.fixture(scope="module")
def prod():
    return S.derivative()


def _clocked(prod, root, **over):
    root.mkdir(exist_ok=True)
    clock = [10**18]
    cfg = prod.ProductionStorageConfig.from_dict(
        S.production_cfg(root, pending_ttl_seconds=1, max_retained_records=1, **over),
        capture_dir=str(root),
    )
    return prod.ProductionStorage(cfg, now_fn=lambda: clock[0]), clock


def _pause_first_replace(prod, monkeypatch, storage, clock, *, same_key: bool, key: str):
    """Advance the clock 2 s past the 1 s lease INSIDE the first os.replace and compete."""
    real = prod.os.replace
    seen: list[bool] = []

    def paused_replace(src, dst):
        if not seen:
            seen.append(True)
            clock[0] += 2 * SECOND_NS
            competing = key if same_key else S.record_key("cmpl-qfn-r3-concurrent")
            with pytest.raises(prod.StorageRefused):
                storage.reserve_for_record(competing)
            assert storage.pending_state()["pending_records"] == 1
            assert storage.pending_state()["active"] == 1
        return real(src, dst)

    monkeypatch.setattr(prod.os, "replace", paused_replace)
    return seen


# ======================================================================================
# 1. the real interval (inside the first os.replace) keeps ownership and capacity
# ======================================================================================

@pytest.mark.parametrize("same_key", [False, True])
def test_the_actual_interval_holds_capacity_and_the_writer_still_commits(prod, tmp_path,
                                                                         monkeypatch, same_key):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-r3-actual")
    reserve = storage.reserve_for_record(key)
    seen = _pause_first_replace(prod, monkeypatch, storage, clock, same_key=same_key, key=key)
    size = prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-r3-actual"),
                                   reserve=reserve)
    monkeypatch.undo()
    assert size > 0 and seen and reserve.state == "committed"
    assert storage.pending_state()["pending_records"] == 0
    assert storage.pending_state()["active"] == 0
    assert storage.pending_committed == 1
    assert storage.pending_expired == 0          # an ACTIVE entry is never reclaimed by the lease
    assert (root / key).exists() and (root / (key + ".json")).exists()


def test_the_interval_stays_active_until_the_second_file_is_published(prod, tmp_path, monkeypatch):
    """The interval spans BOTH publishes: compete again during the json replace."""
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-r3-both")
    reserve = storage.reserve_for_record(key)
    real = prod.os.replace
    calls: list[int] = []

    def competing_replace(src, dst):
        calls.append(1)
        clock[0] += 2 * SECOND_NS
        with pytest.raises(prod.StorageRefused):
            storage.reserve_for_record(S.record_key("cmpl-qfn-r3-other"))
        assert storage.pending_state()["active"] == 1
        return real(src, dst)

    monkeypatch.setattr(prod.os, "replace", competing_replace)
    size = prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-r3-both"),
                                   reserve=reserve)
    monkeypatch.undo()
    assert len(calls) == 2 and size > 0        # both publishes happened under one ACTIVE interval
    assert reserve.state == "committed"
    assert storage.pending_state()["pending_records"] == 0


# ======================================================================================
# 2. failure cleanup and stalled writers
# ======================================================================================

def test_a_second_file_failure_releases_the_active_interval(prod, tmp_path, monkeypatch):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-r3-second-fail")
    reserve = storage.reserve_for_record(key)
    real = prod.os.replace
    calls: list[int] = []

    def failing_second_replace(src, dst):
        calls.append(1)
        if len(calls) == 2:                     # the json publish fails
            raise OSError(5, "injected second-publish failure")
        return real(src, dst)

    monkeypatch.setattr(prod.os, "replace", failing_second_replace)
    with pytest.raises(OSError):
        prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-r3-second-fail"),
                                reserve=reserve)
    monkeypatch.undo()
    assert reserve.state == "released"
    assert storage.pending_state()["pending_records"] == 0
    assert storage.pending_state()["active"] == 0
    assert storage.pending_released == 1
    # the npz was published, the sidecar was not: an orphan the writer never hides
    assert (root / key).exists() and not (root / (key + ".json")).exists()
    # the CAPACITY is free again for a fresh reservation ...
    other = storage.reserve_for_record(S.record_key("cmpl-qfn-r3-second-fail-next"))
    assert other.state == "pending"
    assert storage.release_reservation(other.key, other.token) is True
    # ... while the published orphan still protects its own key from a clobber
    with pytest.raises(prod.StorageRefused) as err:
        storage.reserve_for_record(key)
    assert err.value.reason == "duplicate_record_key"


def test_a_stalled_active_writer_keeps_the_capacity_and_blocks_others(prod, tmp_path):
    """Fail closed: an ACTIVE entry is never reclaimed or reassigned behind a stalled writer."""
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-r3-stalled")
    reserve = storage.reserve_for_record(key)
    reserve.begin_publish()
    clock[0] += 10_000 * SECOND_NS               # the writer is stalled far past the lease
    state = storage.pending_state()
    assert state["active"] == 1 and state["pending_records"] == 1
    assert storage.pending_expired == 0
    with pytest.raises(prod.StorageRefused):
        storage.reserve_for_record(S.record_key("cmpl-qfn-r3-stalled-other"))
    with pytest.raises(prod.StorageRefused):
        storage.reserve_for_record(key)
    # the stalled writer can still finish and commit its own interval
    reserve.finish_publish()
    reserve.commit(1234)
    assert storage.pending_committed_bytes == 1234
    assert storage.pending_state()["pending_records"] == 0


def test_leaving_the_interval_refreshes_the_lease(prod, tmp_path):
    """After the interval the entry is owned again for a full lease (no instant expiry window)."""
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-r3-refresh")
    reserve = storage.reserve_for_record(key)
    reserve.begin_publish()
    clock[0] += 2 * SECOND_NS
    reserve.finish_publish()
    assert storage.reservation_is_current(key, reserve.token) is True
    with pytest.raises(prod.StorageRefused):     # still exclusive right after the interval
        storage.reserve_for_record(key)
    assert storage.pending_expired == 0
    reserve.commit(7)


# ======================================================================================
# 3. operator force must not remove an ACTIVE writer's capacity
# ======================================================================================

def test_operator_force_is_refused_against_an_active_entry(prod, tmp_path):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-r3-force")
    reserve = storage.reserve_for_record(key)
    reserve.begin_publish()
    assert storage.release_reservation(key) is False        # key-only operator release
    assert storage.commit_reservation(key, 1) is False      # key-only operator commit
    assert storage.pending_operator_refused_active == 2
    assert storage.release_all_pending() == 0               # default: non-active only
    state = storage.pending_state()
    assert state["active"] == 1 and state["pending_records"] == 1
    assert state["committed_bytes"] == 0                    # nothing was charged by the operator
    assert storage.release_all_pending(include_active=True) == 1   # explicit stopped-owner force
    assert storage.pending_operator_forced_active == 1
    assert storage.pending_state()["pending_records"] == 0


def test_operator_force_still_works_on_a_non_active_entry(prod, tmp_path):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-r3-operator")
    storage.reserve_for_record(key)
    assert storage.release_reservation(key) is True
    assert storage.pending_operator_actions == 1
    assert storage.pending_operator_refused_active == 0
    assert storage.pending_state()["pending_records"] == 0


def test_the_owner_can_always_finish_its_own_interval(prod, tmp_path):
    """The active-entry guard applies to the OPERATOR form only, never to the owner's token."""
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-r3-owner")
    reserve = storage.reserve_for_record(key)
    reserve.begin_publish()
    reserve.commit(99)                                       # token form: allowed while active
    assert storage.pending_committed_bytes == 99
    assert storage.pending_state()["pending_records"] == 0


# ======================================================================================
# 4. eventual release + no leaked threads/locks
# ======================================================================================

def test_capacity_is_eventually_released_after_a_paused_interval(prod, tmp_path, monkeypatch):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-r3-release")
    reserve = storage.reserve_for_record(key)
    _pause_first_replace(prod, monkeypatch, storage, clock, same_key=False, key=key)
    prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-r3-release"), reserve=reserve)
    monkeypatch.undo()
    other = storage.reserve_for_record(S.record_key("cmpl-qfn-r3-release-next"))
    assert other.state == "pending"                          # capacity really came back
    assert storage.pending_state()["pending_records"] == 1
    storage.release_all_pending()


def test_no_threads_or_locks_are_leaked_by_an_interval(prod, tmp_path):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    before = threading.active_count()
    for i in range(3):
        key = S.record_key(f"cmpl-qfn-r3-leak-{i}")
        reserve = storage.reserve_for_record(key)
        reserve.begin_publish()
        reserve.finish_publish()
        reserve.commit(11)
    assert threading.active_count() == before
    assert storage.pending_state()["pending_records"] == 0
    assert storage._lock.locked() is False
    assert not list(pathlib.Path(root).glob("*.part"))
