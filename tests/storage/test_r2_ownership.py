"""PS1-R2: reservation OWNERSHIP across expiry/reassignment and across the publication window.

CPU-only (numpy + stdlib, no torch, no engine, no network). Every root is a pytest `tmp_path`.
These cases are written BEFORE the R2 product change: they must fail on the R1 snapshot.

The ownership rule under test:
  * a reservation is identified by an opaque per-reservation token, not just by its record key;
  * a stale handle (expired, released, committed, or superseded by a newer reservation for the
    same key) can neither release/commit the newer reservation nor publish;
  * the ownership proof holds ACROSS the publication window, and a publication paused past the
    bounded lease is refused instead of silently publishing.
"""

from __future__ import annotations

import pathlib

import pytest

import ps1_support as S

SECOND_NS = 10**9


@pytest.fixture(scope="module")
def prod():
    return S.derivative()


def _clocked(prod, root, **over):
    """A ProductionStorage on an injected clock, exactly as the PM counterexample builds it."""
    root.mkdir(exist_ok=True)
    clock = [10**18]
    cfg = prod.ProductionStorageConfig.from_dict(
        S.production_cfg(root, pending_ttl_seconds=1, max_retained_records=1, **over),
        capture_dir=str(root),
    )
    return prod.ProductionStorage(cfg, now_fn=lambda: clock[0]), clock


# ======================================================================================
# 1. stale handles must not mutate a newer owner's reservation
# ======================================================================================

def test_a_stale_release_cannot_release_the_new_owners_reservation(prod, tmp_path):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-pm-expired")
    old = storage.reserve_for_record(key)
    clock[0] += 2 * SECOND_NS                      # the old reservation expires (ttl = 1 s)
    new = storage.reserve_for_record(key)
    assert new.token != old.token
    old.release()
    assert storage.pending_state()["pending_records"] == 1, "stale owner released a newer reservation"
    assert storage.pending_stale_ignored == 1
    with pytest.raises(prod.StorageRefused):       # the new owner still holds the capacity
        storage.reserve_for_record(S.record_key("cmpl-qfn-pm-other"))
    new.release()
    assert storage.pending_state()["pending_records"] == 0


def test_a_stale_commit_cannot_charge_the_new_owners_reservation(prod, tmp_path):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-pm-stale-commit")
    old = storage.reserve_for_record(key)
    clock[0] += 2 * SECOND_NS
    new = storage.reserve_for_record(key)
    old.commit(1234)                               # stale: must not count or drop the new entry
    assert storage.pending_committed_bytes == 0
    assert storage.pending_stale_ignored == 1
    state = storage.pending_state()
    assert state["pending_records"] == 1 and state["pending_keys"] == [key]
    assert storage.snapshot(force=True)["retained_bytes"] == storage.cfg.max_record_bytes
    new.commit(4321)
    assert storage.pending_committed_bytes == 4321
    assert storage.pending_state()["pending_records"] == 0


def test_release_and_commit_are_idempotent_for_the_current_owner(prod, tmp_path):
    root = tmp_path / "cap"
    storage, _clock = _clocked(prod, root)
    reserve = storage.reserve_for_record(S.record_key("cmpl-qfn-pm-idem"))
    reserve.release()
    reserve.release()
    assert storage.pending_released == 1 and storage.pending_stale_ignored == 0
    other = storage.reserve_for_record(S.record_key("cmpl-qfn-pm-idem-2"))
    other.commit(99)
    other.commit(99)
    assert storage.pending_committed == 1 and storage.pending_committed_bytes == 99
    assert storage.pending_stale_ignored == 0


def test_the_operator_api_stays_explicit_and_is_counted(prod, tmp_path):
    """Key-only commit/release remain available as a documented OPERATOR force-action."""
    root = tmp_path / "cap"
    storage, _clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-pm-operator")
    storage.reserve_for_record(key)
    assert storage.release_reservation(key) is True       # operator form: no token
    assert storage.pending_operator_actions == 1
    assert storage.pending_state()["pending_records"] == 0
    storage.reserve_for_record(key)
    assert storage.commit_reservation(key, 7) is True
    assert storage.pending_committed_bytes == 7
    assert storage.pending_operator_actions == 2
    assert storage.release_reservation(key) is False      # nothing is held any more
    assert storage.pending_operator_actions == 2


# ======================================================================================
# 2. a stale handle must not publish (before or across the write)
# ======================================================================================

def test_an_expired_handle_cannot_publish_after_its_capacity_is_reassigned(prod, tmp_path):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-pm-expired-write")
    old = storage.reserve_for_record(key)
    clock[0] += 2 * SECOND_NS
    new = storage.reserve_for_record(S.record_key("cmpl-qfn-pm-new-write"))
    with pytest.raises(prod.StorageRefused) as err:
        prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-pm-expired-write"),
                                reserve=old)
    assert err.value.reason == "reservation_not_owned"
    assert not (root / key).exists() and not (root / (key + ".json")).exists()
    assert not list(root.glob("*.part"))
    assert storage.pending_state()["pending_records"] == 1   # only the new owner's
    assert storage.pending_stale_ignored == 0                # a refusal is not a stale mutation
    new.release()


def test_a_released_handle_cannot_publish(prod, tmp_path):
    root = tmp_path / "cap"
    storage, _clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-pm-released")
    reserve = storage.reserve_for_record(key)
    reserve.release()
    with pytest.raises(prod.StorageRefused) as err:
        prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-pm-released"),
                                reserve=reserve)
    assert err.value.reason == "reservation_not_owned"
    assert not (root / key).exists() and not list(root.glob("*.part"))


def test_a_committed_handle_cannot_publish_twice(prod, tmp_path):
    root = tmp_path / "cap"
    storage, _clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-pm-twice")
    reserve = storage.reserve_for_record(key)
    first = prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-pm-twice", marker="a"),
                                    reserve=reserve)
    assert first > 0 and reserve.state == "committed"
    before = (root / key).read_bytes()
    with pytest.raises(prod.StorageRefused) as err:
        prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-pm-twice", marker="b"),
                                reserve=reserve)
    assert err.value.reason == "reservation_not_owned"
    assert (root / key).read_bytes() == before       # nothing was clobbered


def test_ownership_is_frozen_across_the_publication_window(prod, tmp_path):
    """`begin_publish` refreshes the lease, so a normal publication cannot expire mid-write."""
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-pm-freeze")
    reserve = storage.reserve_for_record(key)
    clock[0] += 900_000_000                          # 0.9 s of a 1 s lease
    reserve.begin_publish()
    entry = storage.pending_state()
    assert entry["pending_records"] == 1 and entry["publishing"] == 1
    clock[0] += 900_000_000                          # now past the ORIGINAL lease, but refreshed
    assert storage.reservation_is_current(key, reserve.token) is True
    written = prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-pm-freeze"),
                                      reserve=reserve)
    assert written > 0 and reserve.state == "committed"
    assert storage.pending_state()["pending_records"] == 0


def test_a_pause_past_the_lease_keeps_the_active_interval_r3(prod, tmp_path):
    """R3 SUPERSEDES the R2 expectation in this case.

    R2 asserted that a publication paused past the lease must be refused; that encoded the wrong
    invariant (a merely refreshed timestamp could still be reclaimed). The correct R3 behaviour is
    the opposite: once the interval is ACTIVE it is NOT lease-bounded, so the writer keeps its
    exclusive key and capacity and may finish. The bounded lease still governs reservations that
    never enter publication (see `test_a_leaked_reservation_expires_after_the_bounded_ttl`).
    """
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-pm-pause")
    reserve = storage.reserve_for_record(key)
    reserve.begin_publish()
    clock[0] += 2 * SECOND_NS                        # far past the 1 s lease, inside the interval
    assert storage.reservation_is_current(key, reserve.token) is True
    assert storage.pending_state()["active"] == 1
    assert storage.pending_expired == 0
    with pytest.raises(prod.StorageRefused):
        storage.reserve_for_record(key)              # the key stays exclusive
    with pytest.raises(prod.StorageRefused):
        storage.reserve_for_record(S.record_key("cmpl-qfn-pm-pause-other"))
    reserve.finish_publish()
    reserve.commit(5)
    assert storage.pending_committed_bytes == 5
    assert storage.pending_state()["pending_records"] == 0


def test_a_stale_refusal_releases_nothing_of_the_new_owner(prod, tmp_path):
    root = tmp_path / "cap"
    storage, clock = _clocked(prod, root)
    key = S.record_key("cmpl-qfn-pm-isolated")
    old = storage.reserve_for_record(key)
    clock[0] += 2 * SECOND_NS
    new = storage.reserve_for_record(key)
    with pytest.raises(prod.StorageRefused):
        prod.write_record_files(str(root), key, S.minimal_meta("cmpl-qfn-pm-isolated"), reserve=old)
    old.release()                                    # the handle already knows it is stale
    assert storage.pending_state()["pending_records"] == 1
    assert storage.pending_stale_ignored == 0         # no registry mutation was even attempted
    assert storage.pending_released == 0
    new.release()
    assert storage.pending_state()["pending_records"] == 0
