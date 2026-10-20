"""PS1-R2: a PARTIAL unlink must report the bytes that really left the disk.

CPU-only. Written BEFORE the R2 product change (must fail on the R1 snapshot): in R1 the npz was
removed, the json `os.remove` failed, and `removed_bytes` stayed 0 while real bytes were gone.

Semantics pinned here:
  * `removed_bytes` counts EVERY successful unlink's real size (npz and/or json);
  * `removed_records` keeps its R1 meaning: FULLY removed pairs only;
  * the partial pair is disclosed in the audit and in the `delete_failed` error, and the file that
    could not be unlinked is left in place as a disclosed orphan (never hidden, never compensated);
  * the run stops on the error and the CLI exits 2 with those real counts.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import pytest

import ps1_support as S

HOUR_NS = 3600 * 10**9


@pytest.fixture(scope="module")
def prod():
    return S.derivative()


@pytest.fixture(scope="module")
def ret():
    return S.retention()


def _old_pair(root, key, *, age_seconds=7200):
    root.mkdir(exist_ok=True)
    S.write_record(root, key, hidden=1024)
    ns = time.time_ns() - int(age_seconds * 10**9)
    for name in (key, key + ".json"):
        os.utime(root / name, ns=(ns, ns))


def _fail_on(ret, monkeypatch, suffix: str):
    real = ret.os.remove

    def fake(path):
        if str(path).endswith(suffix):
            raise PermissionError(f"injected unlink failure for {suffix}")
        return real(path)

    monkeypatch.setattr(ret.os, "remove", fake)
    return real


def test_npz_unlinked_then_json_failure_reports_the_real_bytes(ret, tmp_path, monkeypatch):
    root = tmp_path / "cap"
    key = S.record_key("cmpl-qfn-pm-partial")
    _old_pair(root, key)
    npz_bytes = (root / key).stat().st_size
    json_bytes = (root / (key + ".json")).stat().st_size
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert len(result.deletes) == 1
    _fail_on(ret, monkeypatch, ".json")
    ret.execute_plan(result)
    monkeypatch.undo()

    assert not (root / key).exists() and (root / (key + ".json")).exists()
    assert result.refused is True
    assert result.removed_records == 0                 # no FULLY removed pair
    assert result.removed_bytes == npz_bytes           # but the real bytes that were unlinked
    assert result.partial_bytes == npz_bytes
    assert result.partial_unlinks == [{"name": key, "kind": "npz", "bytes": npz_bytes}]
    audit = result.audit()
    assert audit["removed_records"] == 0
    assert audit["removed_bytes"] == npz_bytes
    assert audit["partial_bytes"] == npz_bytes
    assert audit["partial_pairs"] == 1
    assert audit["ok"] is False and audit["refused"] is True
    assert audit["errors"][0]["kind"] == "delete_failed"
    assert audit["errors"][0]["unlinked"] == ["npz"]
    assert audit["errors"][0]["partial"] is True
    assert audit["errors"][0]["partial_bytes"] == npz_bytes
    assert "orphan" in audit["errors"][0]["note"]
    assert json_bytes > 0  # the sidecar that survived is the disclosed orphan


def test_a_json_only_unlink_is_also_counted(ret, tmp_path, monkeypatch):
    """The symmetric case: the npz fails first, so nothing was unlinked and 0 bytes is correct."""
    root = tmp_path / "cap"
    key = S.record_key("cmpl-qfn-pm-partial-npz")
    _old_pair(root, key)
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    _fail_on(ret, monkeypatch, ".npz")
    ret.execute_plan(result)
    monkeypatch.undo()
    assert (root / key).exists() and (root / (key + ".json")).exists()
    assert result.removed_records == 0 and result.removed_bytes == 0
    assert result.partial_unlinks == []
    assert result.errors[0]["unlinked"] == []
    assert result.errors[0]["partial"] is False


def test_a_successful_pair_still_counts_once(ret, tmp_path):
    """R1 semantics preserved: a fully removed pair counts 1 record and npz+json bytes."""
    root = tmp_path / "cap"
    key = S.record_key("cmpl-qfn-pm-full")
    _old_pair(root, key)
    npz_bytes = (root / key).stat().st_size
    json_bytes = (root / (key + ".json")).stat().st_size
    result = ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    assert result.removed_records == 1
    assert result.removed_bytes == npz_bytes + json_bytes
    assert result.partial_bytes == 0 and result.partial_unlinks == []
    assert not (root / key).exists() and not (root / (key + ".json")).exists()
    assert result.audit()["ok"] is True


def test_the_run_stops_at_the_partial_pair(ret, tmp_path, monkeypatch):
    """No compensating deletion: pairs after the failure are untouched."""
    root = tmp_path / "cap"
    first = S.record_key("cmpl-qfn-pm-a")
    second = S.record_key("cmpl-qfn-pm-b")
    _old_pair(root, first)
    _old_pair(root, second)
    result = ret.plan(ret.RetentionPolicy(root=str(root)))
    assert len(result.deletes) == 2
    victim = result.deletes[1]["name"]
    real = ret.os.remove

    def fake(path):
        if str(path).endswith(".json") and victim in str(path):
            raise PermissionError("injected second-pair json failure")
        return real(path)

    monkeypatch.setattr(ret.os, "remove", fake)
    ret.execute_plan(result)
    monkeypatch.undo()
    assert result.removed_records == 1                 # the first pair was fully removed
    assert result.partial_pairs == 1
    assert (root / second).exists() is False           # its npz WAS unlinked
    assert (root / (second + ".json")).exists()        # ... and its sidecar is the disclosed orphan
    assert result.refused is True
    assert result.removed_bytes > 0


def test_the_cli_reports_partial_bytes_and_exits_refused(ret, tmp_path, monkeypatch, capsys):
    root = tmp_path / "cap"
    key = S.record_key("cmpl-qfn-pm-cli-partial")
    _old_pair(root, key)
    npz_bytes = (root / key).stat().st_size
    _fail_on(ret, monkeypatch, ".json")
    rc = ret.main(["--root", str(root), "--execute"])
    monkeypatch.undo()
    audit = json.loads(capsys.readouterr().out)
    assert rc == ret.EXIT_REFUSED
    assert audit["ok"] is False
    assert audit["removed_records"] == 0
    assert audit["removed_bytes"] == npz_bytes         # never 0 after a real unlink
    assert audit["partial_bytes"] == npz_bytes
    assert audit["errors"][0]["kind"] == "delete_failed"


def test_the_orphan_is_left_in_place_and_never_hidden(ret, tmp_path, monkeypatch):
    root = tmp_path / "cap"
    key = S.record_key("cmpl-qfn-pm-orphan")
    _old_pair(root, key)
    _fail_on(ret, monkeypatch, ".json")
    ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    monkeypatch.undo()
    assert (root / (key + ".json")).exists()
    # a later pass sees an incomplete pair, skips it, and deletes nothing
    later = ret.run(ret.RetentionPolicy(root=str(root)), execute=True)
    assert later.removed_records == 0 and later.errors == []
    assert any(s["reason"] == "incomplete_pair" for s in later.skips)
    assert (root / (key + ".json")).exists()
