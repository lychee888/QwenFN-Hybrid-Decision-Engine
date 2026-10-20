"""PS1 retention: delete ONLY expired, completed capture record pairs (NPZ + committed JSON).

WHAT THIS TOOL IS
-----------------
A bounded, dry-run-by-default operator tool for the capture root of the PS1 derivative helper
(`qfn_capture_runtime_production.py`). It removes finished capture records so a long-lived
production run does not fill its disk, and nothing else:

  * it only ever looks at the TOP LEVEL of ONE declared root (never recurses into `artifacts/`,
    historical archives or any other directory);
  * it only ever considers names the helper's OWN `record_file_key` rule reproduces exactly, so
    no invented pattern can be matched and no foreign file can be touched;
  * a record is deletable only when BOTH files of the pair exist, both are regular non-symlink
    files, both are older than `max(ttl_seconds, grace_seconds) >= 3600 s`, and both are fully
    committed (the npz is a readable zip archive and the json sidecar parses);
  * symlinks, `.part` in-flight temporaries, orphans (only one file of a pair), recent records,
    directories, markers and unrelated files are skipped and disclosed, never deleted;
  * a `BENCHMARK_HOLD.json` with an explicit FUTURE `expires_wall_ns` blocks deletion so the PM
    can fetch a completed benchmark; an expired or malformed hold warns and does NOT block (an
    invalid hold must never silently disable the quota);
  * any unreadable root/entry, an oversized scan, or a corrupt candidate pair REFUSES the whole
    run: nothing is deleted. There is no "assume zero" path anywhere.

SAFETY DEFAULTS
---------------
Dry run unless `--execute` is given; `--root` must be passed explicitly; symlinked roots are
refused; deletions are bounded per run (`--max-delete-records`, `--max-delete-bytes`); the audit
JSON reports the ACTUAL removed record/byte counts. No prompt data is ever read or logged: only
file names, sizes, ages and reasons appear in the audit.

LIBRARY USE
-----------
    from retention import RetentionPolicy, run
    policy = RetentionPolicy(root="/path/capture", ttl_seconds=86400, grace_seconds=86400)
    plan = run(policy)                     # dry run: nothing is deleted
    plan = run(policy, execute=True)       # bounded, re-verified deletion
    print(json.dumps(plan.audit(), indent=2))

CLI
---
    python retention.py --root <dir> [--execute] [--ttl-seconds N] [--grace-seconds N]
        [--max-delete-records N] [--max-delete-bytes N] [--hold-file PATH]
        [--marker-prefix cmpl-qfn-] [--now-ns NS] [--audit-out PATH]

Exit codes: 0 = ran (dry run or executed), 2 = refused (policy or scan error), 3 = usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import stat
import sys
import time
import zipfile
from typing import Any, Sequence

HERE = pathlib.Path(__file__).resolve().parent
HELPER_PATH = HERE / "qfn_capture_runtime_production.py"

MIN_GRACE_SECONDS = 3600
MIN_TTL_SECONDS = 3600
DEFAULT_MAX_DELETE_RECORDS = 1000
DEFAULT_MAX_DELETE_BYTES = 1 << 30
# Upper bound on how much PLANNED record content one run may read to establish full identity
# (dev/ino/size/mtime_ns/sha256 of both files). The expensive read is limited to the pairs the run
# could actually delete, and exceeding the bound REFUSES the run instead of reading without bound.
DEFAULT_MAX_IDENTITY_BYTES = 1 << 30
MAX_SCAN_ENTRIES = 200_000
MAX_SIDECAR_BYTES = 1 << 20
HOLD_FILE_NAME = "BENCHMARK_HOLD.json"
HOLD_SCHEMA = "qfn.capture-hold/1"
AUDIT_SCHEMA = "qfn.capture-retention-audit/1"
EXIT_OK, EXIT_REFUSED, EXIT_USAGE = 0, 2, 3

SKIP_REASONS = (
    "symlink",
    "directory",
    "in_flight",
    "not_a_record",
    "incomplete_pair",
    "recent",
    "held",
    "changed_since_plan",
)


class RetentionError(RuntimeError):
    """Base class for a refused retention run (nothing was deleted)."""


class PolicyError(RetentionError):
    """The policy itself is unbounded or malformed."""


class ScanError(RetentionError):
    """The root could not be trusted (unreadable listing/entry, or a corrupt candidate pair)."""


_HELPER: Any = None


def helper() -> Any:
    """Load the derivative helper by absolute path (never via cwd), once."""
    global _HELPER
    if _HELPER is None:
        spec = importlib.util.spec_from_file_location("qfn_ps1_helper_for_retention", HELPER_PATH)
        if spec is None or spec.loader is None:  # pragma: no cover - deployment guard
            raise ScanError(f"cannot load the helper at {HELPER_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _HELPER = module
    return _HELPER


def parse_record_name(name: str, marker_prefix: str) -> tuple[str, str] | None:
    """The helper's own name rule (rebuilds the name with `record_file_key`)."""
    return helper().parse_record_name(name, marker_prefix)


def is_record_file_name(name: str, marker_prefix: str) -> bool:
    return parse_record_name(name, marker_prefix) is not None


def _positive_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyError(f"{name} must be an integer, got {value!r}")
    if int(value) < minimum:
        raise PolicyError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


class HoldState:
    """The state of the (optional) benchmark hold file. Blocking needs a FUTURE expiry."""

    __slots__ = ("present", "path", "scope", "names", "expires_ns", "expired", "invalid",
                 "blocking", "warn", "reason")

    def __init__(self, path: str) -> None:
        self.present = False
        self.path = str(path)
        self.scope: str | None = None
        self.names: frozenset[str] = frozenset()
        self.expires_ns: int | None = None
        self.expired = False
        self.invalid = False
        self.blocking = False
        self.warn: str | None = None
        self.reason: str | None = None

    def blocks(self, name: str) -> bool:
        if not self.blocking:
            return False
        if self.scope == "all":
            return True
        return name in self.names or (name + ".json") in self.names

    def audit(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "path": self.path,
            "scope": self.scope,
            "names": sorted(self.names),
            "expires_wall_ns": self.expires_ns,
            "expired": self.expired,
            "invalid": self.invalid,
            "blocking": self.blocking,
            "warn": self.warn,
            "reason": self.reason,
        }


def read_hold(root: str, policy: "RetentionPolicy", now_ns: int) -> HoldState:
    """Read `<root>/BENCHMARK_HOLD.json` (or the policy's hold file). Never raises."""
    path = policy.hold_file or os.path.join(str(root), HOLD_FILE_NAME)
    hold = HoldState(path)
    if not os.path.exists(path):
        return hold
    hold.present = True
    try:
        if os.path.islink(path) or not os.path.isfile(path):
            hold.invalid, hold.warn = True, "the hold file is not a regular file; ignoring it"
            return hold
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        hold.invalid, hold.warn = True, f"the hold file is unreadable ({exc}); ignoring it"
        return hold
    if not isinstance(payload, dict):
        hold.invalid, hold.warn = True, "the hold file is not a JSON object; ignoring it"
        return hold
    hold.reason = payload.get("reason") if isinstance(payload.get("reason"), str) else None
    scope = payload.get("scope")
    expires = payload.get("expires_wall_ns")
    if isinstance(expires, bool) or not isinstance(expires, int) or expires <= 0:
        hold.invalid = True
        hold.warn = (
            "the hold file has no valid expires_wall_ns, so it does not block deletion; a hold "
            "must carry an explicit future expiry (a hold must never silently disable the quota)"
        )
        return hold
    hold.expires_ns = int(expires)
    if scope == "files":
        names = payload.get("files")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names) or not names:
            hold.invalid, hold.warn = True, "the hold file's 'files' list is missing or malformed"
            return hold
        hold.names = frozenset(names)
        hold.scope = "files"
    elif scope == "all":
        hold.scope = "all"
    else:
        hold.invalid, hold.warn = True, f"the hold file's scope {scope!r} is not 'all' or 'files'"
        return hold
    if now_ns >= hold.expires_ns:
        hold.expired = True
        hold.warn = (
            f"the benchmark hold expired at {hold.expires_ns} ns (now {now_ns} ns); deletion is "
            "no longer blocked"
        )
        return hold
    hold.blocking = True
    return hold


class RetentionPolicy:
    """Bounded deletion policy. `validate()` refuses anything unbounded or too aggressive."""

    def __init__(
        self,
        *,
        root: str,
        ttl_seconds: int = MIN_TTL_SECONDS,
        grace_seconds: int = MIN_GRACE_SECONDS,
        max_delete_records: int = DEFAULT_MAX_DELETE_RECORDS,
        max_delete_bytes: int = DEFAULT_MAX_DELETE_BYTES,
        hold_file: str | None = None,
        marker_prefix: str = "cmpl-qfn-",
        max_scan_entries: int = MAX_SCAN_ENTRIES,
        max_identity_bytes: int = DEFAULT_MAX_IDENTITY_BYTES,
    ) -> None:
        self.root = str(root)
        self.ttl_seconds = ttl_seconds
        self.grace_seconds = grace_seconds
        self.max_delete_records = max_delete_records
        self.max_delete_bytes = max_delete_bytes
        self.hold_file = hold_file
        self.marker_prefix = str(marker_prefix)
        self.max_scan_entries = max_scan_entries
        self.max_identity_bytes = max_identity_bytes

    @property
    def min_age_seconds(self) -> int:
        return max(int(self.ttl_seconds), int(self.grace_seconds))

    def validate(self) -> "RetentionPolicy":
        if not isinstance(self.root, str) or not self.root.strip():
            raise PolicyError("root must be a non-empty path")
        _positive_int("ttl_seconds", self.ttl_seconds, MIN_TTL_SECONDS)
        _positive_int("grace_seconds", self.grace_seconds, MIN_GRACE_SECONDS)
        _positive_int("max_delete_records", self.max_delete_records, 1)
        _positive_int("max_delete_bytes", self.max_delete_bytes, 1)
        _positive_int("max_scan_entries", self.max_scan_entries, 1)
        _positive_int("max_identity_bytes", self.max_identity_bytes, 1)
        if not isinstance(self.marker_prefix, str) or not self.marker_prefix:
            raise PolicyError("marker_prefix must be a non-empty string")
        if self.hold_file is not None:
            if not isinstance(self.hold_file, str) or not self.hold_file.strip():
                raise PolicyError("hold_file must be a non-empty path")
            root_abs = os.path.normcase(os.path.abspath(self.root))
            hold_abs = os.path.normcase(os.path.abspath(self.hold_file))
            if hold_abs != root_abs and not hold_abs.startswith(root_abs + os.sep):
                raise PolicyError(
                    f"hold_file {self.hold_file!r} is outside the declared root {self.root!r}"
                )
        return self


class _Entry:
    __slots__ = ("name", "path", "kind", "size", "mtime_ns")

    def __init__(self, name: str, path: str, kind: str, size: int, mtime_ns: int) -> None:
        self.name, self.path, self.kind = name, path, kind
        self.size, self.mtime_ns = int(size), int(mtime_ns)


def _entry_mtime_ns(entry_st: Any, name: str, root: str) -> int:
    """A stat without a nanosecond mtime is not trustworthy: refuse instead of guessing."""
    mtime_ns = getattr(entry_st, "st_mtime_ns", None)
    if mtime_ns is None:
        raise ScanError(
            f"cannot read a nanosecond mtime for {name!r} in {root!r}; refusing the scan"
        )
    return int(mtime_ns)


def _file_identity(path: str, *, label: str) -> dict[str, Any]:
    """Identity of ONE file: (dev, ino, size, mtime_ns) plus the sha256 of its bytes.

    The lstat is taken BEFORE and AFTER the read and must agree, so the identity describes the
    bytes that were actually read (no torn identity), and any read/stat failure refuses.
    """
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ScanError(f"{label} {path!r} is not a regular file; refusing to trust it")
        with open(path, "rb") as fh:
            blob = fh.read()
        after = os.lstat(path)
    except ScanError:
        raise
    except OSError as exc:
        raise ScanError(f"cannot read {label} {path!r}: {exc}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ScanError(f"{label} {path!r} changed while it was being read; refusing to trust it")
    return {
        "dev": int(after.st_dev),
        "ino": int(after.st_ino),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": blob,
    }


def _root_identity(root: str) -> dict[str, Any]:
    """The declared root's own identity: a symlinked/not-a-directory root is refused."""
    try:
        st = os.lstat(root)
    except OSError as exc:
        raise ScanError(f"the declared root {root!r} cannot be stat()ed: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise ScanError(f"the declared root {root!r} is a symlink; refusing to follow it")
    if not stat.S_ISDIR(st.st_mode):
        raise ScanError(f"the declared root {root!r} is not a directory")
    return {"dev": int(st.st_dev), "ino": int(st.st_ino), "path": os.path.abspath(root)}


class Plan:
    """A fully decided deletion plan. `deletes` is empty unless a plan found candidates."""

    def __init__(self, policy: RetentionPolicy, now_ns: int) -> None:
        self.policy = policy
        self.now_ns = int(now_ns)
        self.hold: HoldState | None = None
        self.deletes: list[dict[str, Any]] = []
        self.skips: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.snapshot: dict[str, Any] = {}
        self.accounting_cross_check: dict[str, Any] = {}
        self.root_identity: dict[str, Any] = {}
        self.refused_reason: str | None = None
        self.partial_unlinks: list[dict[str, Any]] = []
        self.partial_bytes = 0
        self.truncated_by_budget = False
        self.executed = False
        self.removed_records = 0
        self.removed_bytes = 0
        self.removed: list[dict[str, Any]] = []

    @property
    def refused(self) -> bool:
        """True when this run refused (plan-time error, or a mutation-time refusal)."""
        return bool(self.errors) or self.refused_reason is not None

    @property
    def partial_pairs(self) -> int:
        """Pairs that lost at least one file but are NOT counted as removed records."""
        return len({str(item["name"]) for item in self.partial_unlinks})

    @property
    def planned_bytes(self) -> int:
        return sum(int(d["bytes"]) for d in self.deletes)

    def skip(self, name: str, reason: str) -> None:
        assert reason in SKIP_REASONS, reason
        self.skips.append({"name": name, "reason": reason})

    def audit(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for item in self.skips:
            counts[item["reason"]] = counts.get(item["reason"], 0) + 1
        return {
            "schema": AUDIT_SCHEMA,
            "tool": "reports/production-capture-storage/retention.py",
            "helper": str(HELPER_PATH.name),
            "ok": not self.refused,
            "refused": self.refused,
            "refused_reason": self.refused_reason,
            "root": self.policy.root,
            "root_identity": self.root_identity,
            "marker_prefix": self.policy.marker_prefix,
            "now_ns": self.now_ns,
            "ttl_seconds": self.policy.ttl_seconds,
            "grace_seconds": self.policy.grace_seconds,
            "min_age_seconds": self.policy.min_age_seconds,
            "max_delete_records": self.policy.max_delete_records,
            "max_delete_bytes": self.policy.max_delete_bytes,
            "max_scan_entries": self.policy.max_scan_entries,
            "max_identity_bytes": self.policy.max_identity_bytes,
            "dry_run": not self.executed,
            "executed": self.executed,
            "hold": self.hold.audit() if self.hold else None,
            "snapshot": self.snapshot,
            "accounting_cross_check": self.accounting_cross_check,
            "planned_records": len(self.deletes),
            "planned_bytes": self.planned_bytes,
            "removed_records": self.removed_records,
            "removed_bytes": self.removed_bytes,
            "removed": self.removed,
            "partial_unlinks": self.partial_unlinks,
            "partial_bytes": self.partial_bytes,
            "partial_pairs": self.partial_pairs,
            "skipped": self.skips,
            "skip_counts": counts,
            "errors": self.errors,
            "truncated_by_budget": self.truncated_by_budget,
            "note": (
                "Only expired, completed NPZ+JSON record pairs inside the declared root can be "
                "removed, and only after their identity (dev/ino/size/mtime_ns/sha256), the root "
                "identity and the hold file are re-verified at mutation time. Names, sizes and "
                "reasons only: no record content is ever read into the audit."
            ),
        }


def _scan(root: str, policy: RetentionPolicy) -> dict[str, list[_Entry]]:
    """Top-level scan of the declared root. Raises ScanError on anything untrustworthy."""
    _root_identity(root)  # refuses a symlinked or non-directory root
    pairs: dict[str, dict[str, _Entry]] = {}
    seen = 0
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                seen += 1
                if seen > int(policy.max_scan_entries):
                    raise ScanError(
                        f"the declared root {root!r} holds more than {policy.max_scan_entries} "
                        "entries; refusing to truncate the scan"
                    )
                try:
                    entry_st = entry.stat(follow_symlinks=False)
                    is_link = entry.is_symlink()
                except OSError as exc:
                    raise ScanError(f"cannot stat {entry.name!r} in {root!r}: {exc}") from exc
                mtime_ns = _entry_mtime_ns(entry_st, entry.name, root)
                if is_link or stat.S_ISLNK(entry_st.st_mode):
                    pairs.setdefault("\0skip:symlink", {})[entry.name] = _Entry(
                        entry.name, entry.path, "skip", entry_st.st_size, mtime_ns
                    )
                    continue
                if stat.S_ISDIR(entry_st.st_mode):
                    pairs.setdefault("\0skip:directory", {})[entry.name] = _Entry(
                        entry.name, entry.path, "skip", entry_st.st_size, mtime_ns
                    )
                    continue
                name = entry.name
                if name.endswith(".part"):
                    pairs.setdefault("\0skip:in_flight", {})[name] = _Entry(
                        name, entry.path, "skip", entry_st.st_size, mtime_ns
                    )
                    continue
                parsed = parse_record_name(name, policy.marker_prefix)
                if parsed is None:
                    pairs.setdefault("\0skip:not_a_record", {})[name] = _Entry(
                        name, entry.path, "skip", entry_st.st_size, mtime_ns
                    )
                    continue
                stem, kind = parsed
                pairs.setdefault(stem, {})[kind] = _Entry(
                    name, entry.path, kind, entry_st.st_size, mtime_ns
                )
    except ScanError:
        raise
    except OSError as exc:
        raise ScanError(f"cannot list the declared root {root!r}: {exc}") from exc
    return pairs


def _committed_cheap(root: str, name: str) -> bool:
    """Cheap completeness probe for ONE pair: regular non-symlink files, npz zip, parseable json.

    No content hash here: this runs for EVERY candidate so a corrupt candidate anywhere still
    refuses the run, while the expensive identity read is reserved for the planned set.
    """
    npz_path, json_path = helper().record_pair_paths(root, name)
    try:
        npz_st = os.lstat(npz_path)
        json_st = os.lstat(json_path)
        if stat.S_ISLNK(npz_st.st_mode) or stat.S_ISLNK(json_st.st_mode):
            return False
        if not stat.S_ISREG(npz_st.st_mode) or not stat.S_ISREG(json_st.st_mode):
            return False
        if npz_st.st_size <= 0 or json_st.st_size <= 0 or json_st.st_size > MAX_SIDECAR_BYTES:
            return False
        if not zipfile.is_zipfile(npz_path):
            return False
        with open(json_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict)


def _looks_committed(npz_bytes: bytes, json_bytes: bytes) -> bool:
    """Both files of a pair must be complete: a readable npz zip and a parseable json object."""
    if not npz_bytes or not json_bytes or len(json_bytes) > MAX_SIDECAR_BYTES:
        return False
    if not zipfile.is_zipfile(io.BytesIO(npz_bytes)):
        return False
    try:
        payload = json.loads(json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return isinstance(payload, dict)


def _pair_identity(root: str, name: str) -> dict[str, Any] | None:
    """Full identity of one COMPLETE pair, or None when it is not fully committed.

    The two paths are derived from the VALIDATED root plus the OWNED record name (never from a
    stored absolute path), and any unreadable/unsafe file raises ScanError.
    """
    npz_path, json_path = helper().record_pair_paths(root, name)
    npz = _file_identity(npz_path, label="record")
    sidecar = _file_identity(json_path, label="sidecar")
    npz_blob = npz.pop("bytes")
    json_blob = sidecar.pop("bytes")
    if not _looks_committed(npz_blob, json_blob):
        return None
    return {"npz": npz, "json": sidecar}


def _identity_matches(planned: dict[str, Any], fresh: dict[str, Any]) -> bool:
    """The planned identity must still hold, field by field (dev/ino/size/mtime_ns/sha256)."""
    for kind in ("npz", "json"):
        before, now = planned.get(kind) or {}, fresh.get(kind) or {}
        for field in ("dev", "ino", "size", "mtime_ns", "sha256"):
            if before.get(field) != now.get(field):
                return False
    return True


def _refuse(result: Plan, kind: str, message: str) -> None:
    """Refuse the DELETION STEP: record it and stop deleting (nothing further is touched)."""
    result.errors.append({"kind": kind, "name": None, "error": message})
    result.refused_reason = message


def plan(policy: RetentionPolicy, *, now_ns: int | None = None) -> Plan:
    """Decide what may be deleted. Read-only; raises PolicyError/ScanError instead of guessing."""
    policy.validate()
    now = int(time.time_ns() if now_ns is None else now_ns)
    result = Plan(policy, now)
    result.root_identity = _root_identity(policy.root)
    result.hold = read_hold(policy.root, policy, now)
    pairs = _scan(policy.root, policy)

    candidates: list[dict[str, Any]] = []
    for name, entries in pairs.items():
        if name.startswith("\0skip:"):
            reason = name.split(":", 1)[1]
            for entry in entries.values():
                result.skip(entry.name, reason)
            continue
        if "npz" not in entries or "json" not in entries:
            present = sorted(entries.values(), key=lambda e: e.name)
            for entry in present:
                result.skip(entry.name, "incomplete_pair")
            continue
        npz, sidecar = entries["npz"], entries["json"]
        if result.hold.blocks(npz.name):
            result.skip(npz.name, "held")
            continue
        age_ns = now - max(npz.mtime_ns, sidecar.mtime_ns)
        if age_ns < policy.min_age_seconds * 10**9:
            result.skip(npz.name, "recent")
            continue
        candidates.append(
            {
                "name": npz.name,
                "bytes": int(npz.size) + int(sidecar.size),
                "age_seconds": round(age_ns / 1e9, 3),
                "mtime_ns": max(npz.mtime_ns, sidecar.mtime_ns),
            }
        )

    candidates.sort(key=lambda c: (c["mtime_ns"], c["name"]))
    # cheap completeness probe for EVERY candidate: a corrupt candidate anywhere refuses the run
    for item in candidates:
        if not _committed_cheap(policy.root, item["name"]):
            result.errors.append(
                {"kind": "uncommitted_candidate", "name": item["name"], "reason": "not committed"}
            )
    if result.errors:
        raise ScanError(
            f"refusing to delete anything: {len(result.errors)} candidate pair(s) are not fully "
            f"committed (first: {result.errors[0]})"
        )

    kept: list[dict[str, Any]] = []
    budget_records = 0
    budget_bytes = 0
    for item in candidates:
        if budget_records + 1 > policy.max_delete_records or budget_bytes + item["bytes"] > policy.max_delete_bytes:
            result.truncated_by_budget = True
            break
        budget_records += 1
        budget_bytes += int(item["bytes"])
        kept.append(item)

    # BOUND: the full identity read (sha256 of BOTH files) is the expensive part, so it is limited
    # to the pairs this run could actually delete, and to the declared bound.
    if budget_bytes > policy.max_identity_bytes:
        raise ScanError(
            f"refusing to read {budget_bytes} B of planned record content: it exceeds "
            f"max_identity_bytes {policy.max_identity_bytes} (raise the bound deliberately)"
        )
    for item in kept:
        identity = _pair_identity(policy.root, item["name"])
        if identity is None:
            result.errors.append(
                {"kind": "uncommitted_candidate", "name": item["name"], "reason": "not committed"}
            )
            continue
        result.deletes.append(
            {
                "name": item["name"],
                "bytes": int(item["bytes"]),
                "age_seconds": item["age_seconds"],
                "identity": identity,
            }
        )

    if result.errors:
        raise ScanError(
            f"refusing to delete anything: {len(result.errors)} candidate pair(s) are not fully "
            f"committed (first: {result.errors[0]})"
        )

    try:
        snap = helper().scan_capture_root(policy.root, marker_prefix=policy.marker_prefix)
        result.snapshot = snap
        result.accounting_cross_check = {
            "helper_completed_records": int(snap.get("completed_records", 0)),
            "retention_completed_pairs": len(
                [n for n, e in pairs.items() if not n.startswith("\0skip:") and "npz" in e and "json" in e]
            ),
            "agrees": True,
        }
        result.accounting_cross_check["agrees"] = (
            result.accounting_cross_check["helper_completed_records"]
            == result.accounting_cross_check["retention_completed_pairs"]
        )
    except Exception as exc:  # noqa: BLE001 - the cross-check is disclosure, not a decision
        result.accounting_cross_check = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def execute_plan(result: Plan, *, now_ns: int | None = None) -> Plan:
    """Delete the planned pairs, re-verifying EVERYTHING immediately before each removal.

    R1 correction: a plan is NOT a delete capability. For every pair this

      1. re-validates the declared root (still a directory, still the same dev/ino, no symlink),
      2. re-reads the hold file at mutation time (a hold that appeared after the plan blocks its
         pairs; a hold that appeared and is INVALID refuses the deletion step),
      3. re-derives both paths from the validated root plus the OWNED record name (never from a
         stored absolute path), and refuses an unowned/tampered name,
      4. re-reads both files and requires the planned (dev, ino, size, mtime_ns, sha256) identity
         to be unchanged, the pair to be still fully committed, and the pair to be still old,
      5. deletes npz then json and accounts the ACTUAL bytes.

    Any doubt refuses the deletion step or skips that pair; `removed_records`/`removed_bytes`
    always describe what really happened, including after a partial deletion.
    """
    if result.errors:
        raise ScanError("refusing to execute a plan that carries scan errors")
    now = int(time.time_ns() if now_ns is None else now_ns)
    policy = result.policy
    planned_root = result.root_identity or {}
    for item in result.deletes:
        if result.refused_reason is not None:
            break
        try:
            root_now = _root_identity(policy.root)
        except ScanError as exc:
            _refuse(result, "root_invalid", str(exc))
            break
        if planned_root and (root_now["dev"], root_now["ino"]) != (
            planned_root.get("dev"), planned_root.get("ino")
        ):
            _refuse(
                result,
                "root_replaced",
                f"the declared root {policy.root!r} is not the directory that was planned; "
                "refusing the deletion step",
            )
            break
        hold = read_hold(policy.root, policy, now)
        result.hold = hold
        if hold.present and hold.invalid:
            _refuse(
                result,
                "hold_invalid",
                f"the hold file is present but invalid ({hold.warn}); refusing the deletion step",
            )
            break
        name = str(item.get("name") or "")
        parsed = parse_record_name(name, policy.marker_prefix)
        if parsed is None or parsed[1] != "npz":
            _refuse(result, "unowned_name", f"{name!r} is not a record name this tool owns")
            break
        if hold.blocks(name):
            result.skip(name, "held")
            continue
        try:
            fresh = _pair_identity(policy.root, name)
        except ScanError as exc:
            _refuse(result, "pair_unreadable", str(exc))
            break
        if fresh is None or not _identity_matches(item.get("identity") or {}, fresh):
            result.skip(name, "changed_since_plan")
            continue
        newest_ns = max(fresh["npz"]["mtime_ns"], fresh["json"]["mtime_ns"])
        if now - newest_ns < policy.min_age_seconds * 10**9:
            result.skip(name, "recent")
            continue
        npz_path, json_path = helper().record_pair_paths(policy.root, name)
        npz_bytes = int(fresh["npz"]["size"])
        json_bytes = int(fresh["json"]["size"])
        # PS1/R2: unlink the pair member by member and account EVERY successful unlink's real
        # bytes, so `removed_bytes` can never claim 0 after bytes really left the disk. A pair is
        # only counted as a REMOVED RECORD when BOTH files are gone.
        unlinked: list[tuple[str, int]] = []
        failure: OSError | None = None
        for kind, path, size in (("npz", npz_path, npz_bytes), ("json", json_path, json_bytes)):
            try:
                os.remove(path)
            except OSError as exc:
                failure = exc
                break
            unlinked.append((kind, size))
        if failure is None:
            result.removed.append(
                {"name": name, "bytes": npz_bytes + json_bytes, "age_seconds": item.get("age_seconds")}
            )
            result.removed_records += 1
            result.removed_bytes += npz_bytes + json_bytes
            continue
        partial_bytes = sum(size for _kind, size in unlinked)
        result.removed_bytes += partial_bytes
        result.partial_bytes += partial_bytes
        for kind, size in unlinked:
            result.partial_unlinks.append({"name": name, "kind": kind, "bytes": size})
        result.errors.append(
            {
                "kind": "delete_failed",
                "name": name,
                "error": f"{type(failure).__name__}: {failure}",
                "unlinked": [kind for kind, _size in unlinked],
                "partial": bool(unlinked),
                "partial_bytes": partial_bytes,
                "note": (
                    "the pair is NOT counted as a removed record; the file that could not be "
                    "unlinked is left in place as a disclosed orphan (retention skips incomplete "
                    "pairs), never hidden and never compensated by another deletion"
                ),
            }
        )
        result.refused_reason = (
            f"deletion of {name!r} failed after {len(unlinked)} unlink(s): {failure}"
        )
        break
    result.executed = True
    return result


def run(policy: RetentionPolicy, *, execute: bool = False, now_ns: int | None = None) -> Plan:
    """Plan (and, only when asked, execute) one bounded retention pass."""
    result = plan(policy, now_ns=now_ns)
    if execute:
        execute_plan(result, now_ns=now_ns)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retention.py",
        description="PS1 capture retention: delete only expired, completed NPZ+JSON record pairs.",
    )
    parser.add_argument("--root", required=True, help="the declared capture root (explicit)")
    parser.add_argument("--execute", action="store_true", help="actually delete (default: dry run)")
    parser.add_argument("--ttl-seconds", type=int, default=MIN_TTL_SECONDS)
    parser.add_argument("--grace-seconds", type=int, default=MIN_GRACE_SECONDS)
    parser.add_argument("--max-delete-records", type=int, default=DEFAULT_MAX_DELETE_RECORDS)
    parser.add_argument("--max-delete-bytes", type=int, default=DEFAULT_MAX_DELETE_BYTES)
    parser.add_argument("--hold-file", default=None)
    parser.add_argument("--marker-prefix", default="cmpl-qfn-")
    parser.add_argument("--now-ns", type=int, default=None)
    parser.add_argument("--audit-out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse exits; map it onto our exit codes
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK
    policy = RetentionPolicy(
        root=args.root,
        ttl_seconds=args.ttl_seconds,
        grace_seconds=args.grace_seconds,
        max_delete_records=args.max_delete_records,
        max_delete_bytes=args.max_delete_bytes,
        hold_file=args.hold_file,
        marker_prefix=args.marker_prefix,
    )
    try:
        result = run(policy, execute=bool(args.execute), now_ns=args.now_ns)
    except RetentionError as exc:
        # validate()/plan() refuse BEFORE any mutation, so 0 really is the deleted count here
        print(json.dumps(
            {"schema": AUDIT_SCHEMA, "ok": False, "refused": True, "nothing_deleted": True,
             "root": str(args.root), "error": f"{type(exc).__name__}: {exc}",
             "removed_records": 0, "removed_bytes": 0},
            indent=2, sort_keys=True,
        ))
        return EXIT_REFUSED
    audit = result.audit()
    text = json.dumps(audit, indent=2, sort_keys=True)
    if args.audit_out:
        out = pathlib.Path(args.audit_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    # a refusal AFTER a partial mutation still reports the deletions that really happened
    return EXIT_REFUSED if result.refused else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
