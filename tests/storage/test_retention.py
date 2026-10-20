"""PS1: the retention library + CLI (expired completed NPZ+JSON pairs only, dry-run by default).

CPU-only (numpy + stdlib, no torch, no engine, no network). Every root is a pytest `tmp_path`:
nothing outside it is ever read or deleted.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys
import time

import pytest

import ps1_support as S

HOUR_NS = 3600 * 10**9


@pytest.fixture(scope="module")
def ret():
    return S.retention()


@pytest.fixture(scope="module")
def prod():
    return S.derivative()


def _pair(root, key, *, age_seconds=10 * 3600, size=1024):
    """An old, complete, committed record pair written by the REAL writer."""
    S.write_record(root, key, hidden=size)
    _age(root, key, age_seconds=age_seconds)


def _age(root, key, *, age_seconds, json_age_seconds=None):
    """Backdate a record pair (the sidecar is committed after the npz, so it can be newer)."""
    ns = time.time_ns() - int(age_seconds * 10**9)
    js = time.time_ns() - int((json_age_seconds if json_age_seconds is not None else age_seconds) * 10**9)
    for name, when in ((key, ns), (key + ".json", js)):
        p = pathlib.Path(root) / name
        if p.exists():
            os.utime(p, ns=(when, when))


def _names(root):
    return sorted(p.name for p in pathlib.Path(root).iterdir())


# ======================================================================================
# 1. policy + name validation
# ======================================================================================

@pytest.mark.parametrize(
    "kw",
    [
        {"ttl_seconds": 3599},
        {"grace_seconds": 3599},
        {"ttl_seconds": 0},
        {"grace_seconds": -1},
        {"max_delete_records": 0},
        {"max_delete_bytes": -5},
        {"max_scan_entries": 0},
        {"max_identity_bytes": 0},
        {"ttl_seconds": 3600.5},
        {"grace_seconds": True},
    ],
)
def test_policy_refuses_unbounded_or_too_short_values(ret, tmp_path, kw):
    p = ret.RetentionPolicy(root=str(tmp_path), **kw)
    with pytest.raises(ret.PolicyError):
        p.validate()
    with pytest.raises(ret.PolicyError):
        ret.run(p)


def test_policy_minimums_are_at_least_one_hour(ret, tmp_path):
    assert ret.MIN_GRACE_SECONDS >= 3600 and ret.MIN_TTL_SECONDS >= 3600
    p = ret.RetentionPolicy(root=str(tmp_path))
    p.validate()
    assert p.ttl_seconds == 3600 and p.grace_seconds == 3600
    assert p.max_delete_records > 0 and p.max_delete_bytes > 0


def test_real_record_names_are_recognized_and_invented_ones_are_not(ret, prod):
    real = S.record_key("cmpl-qfn-dec-0")
    real_upper = prod.record_file_key("cmpl-qfn-dec-0", prompt_hash="ABCDEF0123456789",
                                      disambiguation=3, marker_prefix=S.MARKER_PREFIX)
    nohash = prod.record_file_key("cmpl-qfn-dec-0", prompt_hash=None, disambiguation=0,
                                  marker_prefix=S.MARKER_PREFIX)
    for good in (real, real + ".json", real_upper, nohash):
        assert ret.is_record_file_name(good, marker_prefix=S.MARKER_PREFIX)
    bad = [
        "capture.npz",
        "a__b__0.npz",
        "a__0123456789abcdef__0.txt",
        "a__0123456789abcdef__x.npz",
        "cmpl-qfn-dec-0__0123456789abcdef__0.npz.json.part",
        "cmpl-qfn-dec-0__0123456789abcdef__0.npz.part",
        "cmpl-qfn-dec-0__0123456789abcdef__0000000.npz",
        "other-prefix-dec-0__0123456789abcdef__0.npz",
        "CAPTURE_ERROR.json",
        "BENCHMARK_HOLD.json",
        "hidden_hash_rank0.jsonl",
    ]
    for name in bad:
        assert not ret.is_record_file_name(name, marker_prefix=S.MARKER_PREFIX), name


# ======================================================================================
# 2. what may be deleted (and what never is)
# ======================================================================================

def test_dry_run_is_the_default_and_execute_is_required(ret, tmp_path):
    root = tmp_path / "root"
    old = S.record_key("cmpl-qfn-old-0")
    _pair(root, old, age_seconds=10 * 3600)
    before = _names(root)

    plan = ret.run(ret.RetentionPolicy(root=str(root)))
    assert plan.executed is False
    assert [d["name"] for d in plan.deletes] == [old]
    assert _names(root) == before  # a dry run deletes nothing

    plan2 = ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    assert plan2.executed is True
    assert plan2.removed_records == 1
    assert _names(root) == []


def test_only_an_old_committed_pair_is_deleted(ret, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    old = S.record_key("cmpl-qfn-old-0")
    recent = S.record_key("cmpl-qfn-recent-0")
    fresh_sidecar = S.record_key("cmpl-qfn-halfcommitted-0")
    orphan_npz = S.record_key("cmpl-qfn-orphan-0")
    orphan_json = S.record_key("cmpl-qfn-orphan-1")
    oldish = S.record_key("cmpl-qfn-ttl-0")

    _pair(root, old, age_seconds=10 * 3600)
    _pair(root, recent, age_seconds=10)                       # recent: must survive
    _pair(root, fresh_sidecar, age_seconds=10 * 3600)
    _age(root, fresh_sidecar, age_seconds=10 * 3600, json_age_seconds=5)  # npz old, commit fresh
    _pair(root, orphan_npz, age_seconds=10 * 3600)
    (root / (orphan_npz + ".json")).unlink()                  # orphan npz
    _pair(root, orphan_json, age_seconds=10 * 3600)
    (root / orphan_json).unlink()                             # orphan json
    _pair(root, oldish, age_seconds=10 * 3600)                # inside the TTL window
    _age(root, oldish, age_seconds=3500)
    (root / (old + ".part")).write_bytes(b"in flight")        # a stray temp
    S.write_record(root, S.record_key("cmpl-qfn-x-0"))
    (root / (S.record_key("cmpl-qfn-x-0"))).unlink()          # npz gone, sidecar is brand new
    (root / "CAPTURE_ERROR.json").write_text("{}", encoding="utf-8")
    (root / "hidden_hash_rank0.jsonl").write_text("", encoding="utf-8")
    (root / "artifacts").mkdir()
    nested = root / "artifacts" / "nested__0123456789abcdef__0.npz"   # must never be recursed into
    nested.write_bytes(b"not a real record")
    linked_ok = False
    try:
        (root / "link").symlink_to(root / "artifacts", target_is_directory=True)
        os.symlink(root / "CAPTURE_ERROR.json", root / S.record_key("cmpl-qfn-sym-0"))
        (root / (S.record_key("cmpl-qfn-sym-0") + ".json")).write_text("{}", encoding="utf-8")
        linked_ok = True
    except (OSError, NotImplementedError):
        pass

    before = set(_names(root))
    plan = ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    assert plan.errors == []
    assert [d["name"] for d in plan.deletes] == [old]
    assert plan.removed_records == 1
    survivors = before - {old, old + ".json"}
    assert set(_names(root)) == survivors
    assert not (root / old).exists() and not (root / (old + ".json")).exists()
    for keep in (recent, fresh_sidecar, orphan_npz, oldish):
        assert (root / keep).exists(), keep
    assert (root / (orphan_json + ".json")).exists(), orphan_json  # the orphan sidecar survives
    assert (root / (old + ".part")).exists()
    assert nested.exists() and (root / "artifacts").is_dir()
    if linked_ok:
        assert (root / (S.record_key("cmpl-qfn-sym-0"))).is_symlink()
    reasons = {s["reason"] for s in plan.skips}
    assert {"in_flight", "recent", "incomplete_pair", "not_a_record", "directory"} <= reasons


def test_symlinked_root_refuses(ret, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted on this host")
    with pytest.raises(ret.ScanError):
        ret.run(ret.RetentionPolicy(root=str(link)))


def test_symlinked_entries_are_skipped_by_retention(ret, tmp_path, monkeypatch):
    """A symlink that even LOOKS like a record is skipped, never followed or deleted.

    Synthetic scandir entry, so this branch is proven on hosts that cannot create symlinks too.
    """
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target.bin"
    target.write_bytes(b"outside the root")
    name = S.record_key("cmpl-qfn-sym-0")
    (root / name).write_bytes(b"")  # a real empty file to delete-if-the-bug-exists
    real_scandir = os.scandir

    class _Link:
        def __init__(self, name):
            self.name = name
            self.path = str(root / name)

        def is_symlink(self):
            return True

        def stat(self, follow_symlinks=True):
            # a REAL stat_result (all fields present, incl. st_mtime_ns): no synthetic shape
            return os.stat(self.path)

    class _It:
        def __enter__(self):
            return iter([_Link(name)])

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(os, "scandir", lambda _p: _It())
    result = ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    assert result.removed_records == 0
    assert [s["reason"] for s in result.skips] == ["symlink"]
    monkeypatch.setattr(os, "scandir", real_scandir)
    assert (root / name).exists() and target.exists()


def test_missing_root_refuses(ret, tmp_path):
    with pytest.raises(ret.ScanError):
        ret.run(ret.RetentionPolicy(root=str(tmp_path / "nope")))


def test_listing_failure_refuses(ret, tmp_path, monkeypatch):
    root = tmp_path / "root"
    _pair(root, S.record_key("cmpl-qfn-old-0"), age_seconds=10 * 3600)

    def boom(_p):
        raise OSError(13, "denied")

    monkeypatch.setattr(os, "scandir", boom)
    with pytest.raises(ret.ScanError):
        ret.run(ret.RetentionPolicy(root=str(root)))
    assert (root / S.record_key("cmpl-qfn-old-0")).exists()


def test_a_corrupt_candidate_pair_refuses_the_whole_run(ret, tmp_path):
    root = tmp_path / "root"
    old = S.record_key("cmpl-qfn-old-0")
    broken = S.record_key("cmpl-qfn-broken-0")
    _pair(root, old, age_seconds=10 * 3600)
    _pair(root, broken, age_seconds=10 * 3600)
    (root / (broken + ".json")).write_text("{not json", encoding="utf-8")
    _age(root, broken, age_seconds=10 * 3600)  # old enough to be a real candidate
    with pytest.raises(ret.ScanError):
        ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    assert (root / old).exists() and (root / (old + ".json")).exists()
    assert (root / broken).exists()


def test_a_truncated_npz_candidate_also_refuses(ret, tmp_path):
    root = tmp_path / "root"
    broken = S.record_key("cmpl-qfn-broken-0")
    _pair(root, broken, age_seconds=10 * 3600)
    (root / broken).write_bytes(b"not a zip archive")
    _age(root, broken, age_seconds=10 * 3600)
    with pytest.raises(ret.ScanError):
        ret.run(ret.RetentionPolicy(root=str(root)))
    assert (root / broken).exists()


def test_bounded_limits_truncate_the_run(ret, tmp_path):
    root = tmp_path / "root"
    for i in range(5):
        _pair(root, S.record_key(f"cmpl-qfn-dec-{i}"), age_seconds=10 * 3600)
    plan = ret.run(ret.RetentionPolicy(root=str(root), max_delete_records=2), execute=True)
    assert plan.removed_records == 2 and plan.truncated_by_budget is True
    assert len(_names(root)) == 6
    remaining = S.record_key("cmpl-qfn-dec-2")
    pair_size = (root / remaining).stat().st_size + (root / (remaining + ".json")).stat().st_size
    plan2 = ret.run(ret.RetentionPolicy(root=str(root), max_delete_bytes=pair_size), execute=True)
    assert plan2.removed_records == 1 and plan2.truncated_by_budget is True
    assert len(_names(root)) == 4


# ======================================================================================
# 3. HOLD semantics
# ======================================================================================

def _hold(root, payload):
    (pathlib.Path(root) / ret_hold_name()).write_text(json.dumps(payload), encoding="utf-8")


def ret_hold_name():
    return "BENCHMARK_HOLD.json"


def test_hold_with_an_explicit_future_expiry_blocks_deletion(ret, tmp_path):
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-bench-0")
    _pair(root, key, age_seconds=10 * 3600)
    _hold(root, {"schema": ret.HOLD_SCHEMA, "scope": "all", "reason": "benchmark in flight",
                 "expires_wall_ns": time.time_ns() + 6 * HOUR_NS})
    plan = ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    assert plan.removed_records == 0
    assert (root / key).exists() and (root / (key + ".json")).exists()
    assert plan.hold.present is True and plan.hold.blocking is True
    assert plan.hold.expired is False and plan.hold.invalid is False
    assert any(s["reason"] == "held" for s in plan.skips)


def test_expired_hold_warns_and_does_not_block(ret, tmp_path):
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-bench-0")
    _pair(root, key, age_seconds=10 * 3600)
    _hold(root, {"schema": ret.HOLD_SCHEMA, "scope": "all",
                 "expires_wall_ns": time.time_ns() - HOUR_NS})
    plan = ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    assert plan.removed_records == 1
    assert plan.hold.expired is True and plan.hold.blocking is False
    assert plan.hold.warn and "expired" in plan.hold.warn
    assert plan.audit()["hold"]["warn"]


def test_an_unbounded_or_malformed_hold_warns_at_plan_and_refuses_the_deletion(ret, tmp_path):
    """R1 corrective scope: plan warns; the DELETION step fails closed on an invalid hold."""
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-bench-0")
    _pair(root, key, age_seconds=10 * 3600)
    _hold(root, {"schema": ret.HOLD_SCHEMA, "scope": "all"})  # no expiry at all
    planned = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert planned.hold.invalid is True and planned.hold.blocking is False
    assert "no valid expires_wall_ns" in planned.hold.warn
    assert len(planned.deletes) == 1  # a plan is advisory; it is not a delete capability
    ret.execute_plan(planned)
    assert planned.removed_records == 0
    assert planned.refused_reason is not None
    assert (root / key).exists() and (root / (key + ".json")).exists()


def test_hold_blocks_only_the_named_files_when_scoped(ret, tmp_path):
    root = tmp_path / "root"
    held = S.record_key("cmpl-qfn-bench-0")
    other = S.record_key("cmpl-qfn-dec-0")
    _pair(root, held, age_seconds=10 * 3600)
    _pair(root, other, age_seconds=10 * 3600)
    _hold(root, {"schema": ret.HOLD_SCHEMA, "scope": "files", "files": [held],
                 "expires_wall_ns": time.time_ns() + 6 * HOUR_NS})
    plan = ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    assert plan.removed_records == 1
    assert (root / held).exists() and not (root / other).exists()


def test_hold_file_outside_the_root_is_refused(ret, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / ret_hold_name()).write_text("{}", encoding="utf-8")
    with pytest.raises(ret.PolicyError):
        ret.RetentionPolicy(root=str(root), hold_file=str(outside / ret_hold_name())).validate()


def test_a_hold_never_disables_the_storage_quota(prod, ret, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    state = S.make_state(root, production=S.production_cfg(root, max_retained_records=1))
    S.fill_until_stopped(state, limit=5)
    assert state.records_written == 1
    (root / ret_hold_name()).write_text(
        json.dumps({"schema": ret.HOLD_SCHEMA, "scope": "all",
                    "expires_wall_ns": time.time_ns() + 6 * HOUR_NS}),
        encoding="utf-8",
    )
    state.production.invalidate()
    with pytest.raises(prod.StorageRefused) as err:   # the bound still applies
        S.write_one(state, 1)
    assert err.value.reason == "retained_record_limit"
    snap = state.production.snapshot(force=True)
    assert snap["completed_records"] == 1  # the hold file is not a capture record
    assert snap["other_files"] >= 1


# ======================================================================================
# 4. audit + CLI
# ======================================================================================

def test_audit_reports_actual_removed_bytes_and_no_prompt_data(ret, tmp_path):
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-dec-0")
    _pair(root, key, age_seconds=10 * 3600)
    expected = (root / key).stat().st_size + (root / (key + ".json")).stat().st_size
    plan = ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    audit = plan.audit()
    assert audit["removed_records"] == 1 and audit["removed_bytes"] == expected
    assert audit["dry_run"] is False and audit["errors"] == []
    assert audit["removed"][0]["name"] == key
    assert audit["root"] == str(root)
    text = json.dumps(audit)
    for forbidden in ("vector_bf16", "verbalizer", "logits", "token_ids", "prompt_token"):
        assert forbidden not in text
    assert audit["accounting_cross_check"]["agrees"] is True


def test_cli_defaults_to_dry_run_and_needs_execute(ret, tmp_path):
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-dec-0")
    _pair(root, key, age_seconds=10 * 3600)
    rc = ret.main(["--root", str(root)])
    assert rc == 0
    assert (root / key).exists()
    rc = ret.main(["--root", str(root), "--execute"])
    assert rc == 0
    assert not (root / key).exists()


def test_cli_usage_and_refusals_exit_nonzero(ret, tmp_path):
    assert ret.main([]) == 3                        # no explicit root
    assert ret.main(["--root", str(tmp_path), "--nope"]) == 3
    assert ret.main(["--root", str(tmp_path), "--ttl-seconds", "60"]) == 2
    assert ret.main(["--root", str(tmp_path / "missing"), "--execute"]) == 2


def test_cli_from_a_foreign_cwd_deletes_only_with_execute(ret, tmp_path):
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-dec-0")
    _pair(root, key, age_seconds=10 * 3600)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    dry = subprocess.run(
        [sys.executable, str(S.RETENTION), "--root", str(root)],
        cwd=str(foreign), capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert dry.returncode == 0, dry.stderr
    assert (root / key).exists()
    assert json.loads(dry.stdout)["removed_records"] == 0

    exe = subprocess.run(
        [sys.executable, str(S.RETENTION), "--root", str(root), "--execute"],
        cwd=str(foreign), capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert exe.returncode == 0, exe.stderr
    assert not (root / key).exists()
    assert json.loads(exe.stdout)["removed_records"] == 1


# ======================================================================================
# 5. mutation-time re-verification (R1 correction: plan is not a delete capability)
# ======================================================================================

def _backdate_pair(root, key, *, age_seconds=10 * 3600):
    ns = time.time_ns() - int(age_seconds * 10**9)
    for name in (key, key + ".json"):
        os.utime(root / name, ns=(ns, ns))


def _mtime_ns(root, name):
    return os.lstat(root / name).st_mtime_ns


def test_hold_added_after_plan_blocks_the_deletion(ret, tmp_path):
    """PM contract 3: the hold is re-read at mutation time, not only at plan time."""
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-pm-0")
    _pair(root, key)
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert [d["name"] for d in result.deletes] == [key]
    _hold(root, {"schema": ret.HOLD_SCHEMA, "scope": "all",
                 "expires_wall_ns": time.time_ns() + 10 * HOUR_NS})
    ret.execute_plan(result)
    assert (root / key).exists() and (root / (key + ".json")).exists()
    assert result.removed_records == 0
    assert any(s["reason"] == "held" for s in result.skips)


def test_sidecar_replaced_same_size_different_old_mtime_is_refused(ret, tmp_path):
    """PM contract 4: size alone is not identity; a newer-or-different file must survive."""
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-pm-0")
    _pair(root, key)
    sidecar = root / (key + ".json")
    text = sidecar.read_text(encoding="utf-8")
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert len(result.deletes) == 1
    sidecar.write_text(text.replace("ps1", "NEW"), encoding="utf-8")
    assert len(text.replace("ps1", "NEW")) == len(text)  # same size, different content
    _backdate_pair(root, key, age_seconds=4000)          # still "old", but a different time
    ret.execute_plan(result)
    assert (root / key).exists() and sidecar.exists()
    assert result.removed_records == 0
    assert any(s["reason"] == "changed_since_plan" for s in result.skips)


def test_same_size_same_mtime_replacement_is_refused_by_content_identity(ret, tmp_path):
    """Even a restored mtime cannot hide a replaced sidecar: the content hash is checked."""
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-pm-0")
    _pair(root, key)
    sidecar = root / (key + ".json")
    before = os.lstat(sidecar)
    text = sidecar.read_text(encoding="utf-8")
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert len(result.deletes) == 1
    sidecar.write_text(text.replace("ps1", "NEW"), encoding="utf-8")
    os.utime(sidecar, ns=(before.st_atime_ns, before.st_mtime_ns))  # identical size AND mtime
    fresh = os.lstat(sidecar)
    assert (fresh.st_size, fresh.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    ret.execute_plan(result)
    assert (root / key).exists() and sidecar.exists()
    assert result.removed_records == 0
    assert any(s["reason"] == "changed_since_plan" for s in result.skips)


def test_root_replaced_by_a_different_directory_refuses_the_deletion(ret, tmp_path):
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-pm-0")
    _pair(root, key)
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert len(result.deletes) == 1
    for p in root.iterdir():          # replace the whole root with a fresh directory
        p.unlink()
    root.rmdir()
    root.mkdir()
    S.write_record(root, key)
    _backdate_pair(root, key)
    ret.execute_plan(result)
    assert (root / key).exists() and (root / (key + ".json")).exists()
    assert result.removed_records == 0
    assert result.refused_reason is not None or any(
        s["reason"] == "changed_since_plan" for s in result.skips
    )


def test_root_becoming_a_symlink_refuses_the_deletion_step(ret, tmp_path, monkeypatch):
    """Synthetic lstat: a root that turns into a symlink must stop the deletion step."""
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-pm-0")
    _pair(root, key)
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert len(result.deletes) == 1
    real_lstat = os.lstat
    root_abs = os.path.abspath(str(root))

    def fake_lstat(path):
        st = real_lstat(path)
        if os.path.abspath(str(path)) == root_abs:
            return os.stat_result((stat.S_IFLNK | 0o777, st.st_ino, st.st_dev, st.st_nlink,
                                   st.st_uid, st.st_gid, st.st_size, st.st_atime, st.st_mtime,
                                   st.st_ctime))
        return st

    monkeypatch.setattr(os, "lstat", fake_lstat)
    ret.execute_plan(result)
    monkeypatch.undo()
    assert (root / key).exists() and (root / (key + ".json")).exists()
    assert result.removed_records == 0
    assert result.refused_reason is not None


def test_hold_malformed_at_mutation_time_refuses_the_deletion_step(ret, tmp_path):
    """A NEW malformed hold fails CLOSED for the deletion step (corrective scope, R1)."""
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-pm-0")
    _pair(root, key)
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert len(result.deletes) == 1
    _hold(root, {"schema": ret.HOLD_SCHEMA, "scope": "all"})   # no expiry at all
    ret.execute_plan(result)
    assert (root / key).exists() and (root / (key + ".json")).exists()
    assert result.removed_records == 0
    assert result.refused_reason is not None
    assert any(e["kind"] == "hold_invalid" for e in result.errors)


def test_the_candidate_read_budget_refuses_instead_of_reading_without_bound(ret, tmp_path):
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-dec-0")
    _pair(root, key)
    with pytest.raises(ret.ScanError):
        ret.plan(ret.RetentionPolicy(root=str(root), max_identity_bytes=16))
    assert (root / key).exists() and (root / (key + ".json")).exists()


def test_a_large_candidate_backlog_is_not_walled_off_by_the_read_bound(ret, tmp_path):
    """The expensive identity read is limited to the PLANNED set, so a big backlog still plans."""
    root = tmp_path / "root"
    for i in range(4):
        _pair(root, S.record_key(f"cmpl-qfn-dec-{i}"), age_seconds=10 * 3600)
    result = ret.run(ret.RetentionPolicy(root=str(root), max_delete_records=1,
                                         max_identity_bytes=64 << 10))
    assert len(result.deletes) == 1
    assert result.truncated_by_budget is True
    assert result.errors == []


def test_plan_paths_are_rebuilt_from_the_validated_root(ret, tmp_path):
    """Plan entries are names, not paths: an unowned/tampered name refuses the step."""
    root = tmp_path / "root"
    key = S.record_key("cmpl-qfn-pm-0")
    _pair(root, key)
    outside = tmp_path / "outside__0123456789abcdef__0.npz"
    outside.write_bytes(b"must survive")
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert len(result.deletes) == 1
    assert "npz" not in result.deletes[0] and "json" not in result.deletes[0]
    result.deletes[0]["name"] = "../outside__0123456789abcdef__0.npz"
    ret.execute_plan(result)
    assert outside.exists() and (root / key).exists()
    assert result.removed_records == 0
    assert result.refused_reason is not None


def test_partial_deletion_is_reported_accurately(ret, tmp_path, monkeypatch):
    """A failure on the SECOND pair must never be reported as 'nothing was deleted'."""
    root = tmp_path / "root"
    first = S.record_key("cmpl-qfn-dec-0")
    second = S.record_key("cmpl-qfn-dec-1")
    _pair(root, first)
    _pair(root, second)
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert len(result.deletes) == 2
    real_remove = os.remove
    victim = result.deletes[1]["name"]

    def fake_remove(path, *a, **k):
        if os.path.basename(str(path)) == victim:
            raise OSError(13, "injected deletion failure")
        return real_remove(path, *a, **k)

    monkeypatch.setattr(os, "remove", fake_remove)
    ret.execute_plan(result)
    monkeypatch.undo()
    assert result.removed_records == 1
    assert result.removed_bytes > 0
    assert result.removed[0]["name"] == first
    assert (root / second).exists()
    assert any(e["kind"] == "delete_failed" for e in result.errors)
    audit = result.audit()
    assert audit["removed_records"] == 1
    assert audit["refused"] is True


def test_cli_reports_partial_deletions_accurately(ret, tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    first = S.record_key("cmpl-qfn-dec-0")
    second = S.record_key("cmpl-qfn-dec-1")
    _pair(root, first)
    _pair(root, second)
    real_remove = os.remove

    def fake_remove(path, *a, **k):
        if os.path.basename(str(path)) == second:
            raise OSError(13, "injected deletion failure")
        return real_remove(path, *a, **k)

    monkeypatch.setattr(os, "remove", fake_remove)
    rc = ret.main(["--root", str(root), "--execute"])
    monkeypatch.undo()
    audit = json.loads(capsys.readouterr().out)
    assert rc == ret.EXIT_REFUSED
    assert audit["ok"] is False
    assert audit["removed_records"] == 1        # NOT 0: one pair really was deleted
    assert audit["removed_bytes"] > 0
    assert (root / second).exists()
