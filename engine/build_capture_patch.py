"""Source-hash / anchor guarded generator for the S3 capture patch (COPY ONLY).

It takes the pristine installed V2 model runner and writes a patched COPY. The original
source is never modified: the generator only ever reads `src` and writes `out`.

Fail-closed guards (all of them must pass before a single byte is written):
  1. `sha256(src) == expect_sha256`               - exact source identity
  2. the declared source version matches `expect_vllm_version`
  3. the module-level import anchor occurs EXACTLY once
  4. the two-line call anchor occurs EXACTLY once  (the single `sample_hidden_states = ...`
     line occurs TWICE in this source - lines 808 and 1390 - so a one-line anchor would
     silently patch the wrong site; that ambiguity is why the anchor is the two-line form)
  5. the patch marker is absent (an already-patched input is refused, not re-patched)
  6. the generated copy parses as valid Python (AST) and differs from the source in exactly
     one module statement and one statement inside `GPUModelRunner.sample`

Usage::

    python engine/build_capture_patch.py \
        --src "$VLLM_RUNNER_SOURCE" \
        --out build/model_runner.py \
        --checksums build/runner.sha256 --meta build/runner.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

MARKER = "# QFN-CAPTURE-PATCH v1"
PATCH_VERSION = 1
HELPER_MODULE = "vllm.v1.worker.gpu.qfn_capture_runtime"
HELPER_FUNC = "capture_after_compute_logits"
LOCAL_ALIAS = "_qfn_capture_after_logits"
ANCHOR_NAME = "runner-sample-compute-logits"
ANCHOR_IMPORT = "from vllm.logger import init_logger\n"
ANCHOR_CALL = (
    "        sample_hidden_states = hidden_states[input_batch.logits_indices]\n"
    "        logits = self.model.compute_logits(sample_hidden_states)\n"
)
EXPECTED_SHA256 = "f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8"
EXPECTED_SOURCE_BYTES = 91240
EXPECTED_VLLM_VERSION = "0.1.dev20073+g8e685d198"
PATCHED_FUNCTIONS = ("sample",)


class PatchError(RuntimeError):
    """Raised when any guard fails. Nothing is written when this is raised."""


def _import_block(source_sha256: str) -> str:
    return (
        f"{MARKER} source_sha256={source_sha256} anchor={ANCHOR_NAME} helper={HELPER_MODULE}\n"
        "try:\n"
        f"    from {HELPER_MODULE} import (\n"
        f"        {HELPER_FUNC} as {LOCAL_ALIAS},\n"
        "    )\n"
        "except ImportError as _qfn_import_error:  # helper not mounted: capture stays unavailable\n"
        f"    {LOCAL_ALIAS} = None\n"
        "    import sys as _qfn_sys\n"
        "\n"
        "    print(\n"
        '        f"[qfn-capture] helper import failed: {_qfn_import_error!r}",\n'
        "        file=_qfn_sys.stderr,\n"
        "    )\n"
    )


def _call_block() -> str:
    return (
        "        # QFN-CAPTURE-PATCH: capture the decision-boundary state and the RAW selected\n"
        "        # logits here (after compute_logits, before grammar/sampling processing).\n"
        "        # No-op unless the capture helper is mounted AND capture is enabled.\n"
        f"        if {LOCAL_ALIAS} is not None:\n"
        f"            {LOCAL_ALIAS}(self, hidden_states, logits, input_batch)\n"
    )


def _count_occurrences(text: str, needle: str) -> int:
    return text.count(needle)


def _statement_counts(text: str) -> dict[str, int]:
    tree = ast.parse(text)
    counts = {"module": len(tree.body)}
    for fn_name in PATCHED_FUNCTIONS:
        counts[fn_name] = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
                counts[fn_name] += len(node.body)
    return counts


def build_patch(
    *,
    src: str | pathlib.Path,
    out: str | pathlib.Path,
    expect_sha256: str | None = EXPECTED_SHA256,
    expect_vllm_version: str | None = EXPECTED_VLLM_VERSION,
    source_version: str = EXPECTED_VLLM_VERSION,
    expect_bytes: int | None = EXPECTED_SOURCE_BYTES,
    checksums_path: str | pathlib.Path | None = None,
    meta_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Generate the patched runner copy. Raises PatchError on any guard failure."""
    src_path = pathlib.Path(src)
    out_path = pathlib.Path(out)
    destinations = [out_path] + [pathlib.Path(p) for p in (checksums_path, meta_path) if p is not None]
    resolved = [p.resolve() for p in destinations]
    if src_path.resolve() in resolved or len(set(resolved)) != len(resolved):
        raise PatchError("source and all output paths must be distinct; in-place patching is refused")
    for destination in destinations:
        if destination.exists() and src_path.exists() and destination.samefile(src_path):
            raise PatchError("output aliases the source; refusing to modify input")
    if not src_path.is_file():
        raise PatchError(f"source not found: {src_path}")
    raw = src_path.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()

    if expect_sha256 is not None and source_sha256 != expect_sha256:
        raise PatchError(
            "source sha256 mismatch (fail closed): "
            f"expected {expect_sha256}, got {source_sha256} for {src_path}"
        )
    if expect_bytes is not None and len(raw) != expect_bytes:
        raise PatchError(
            f"source size mismatch (fail closed): expected {expect_bytes} bytes, got {len(raw)}"
        )
    if expect_vllm_version is not None and expect_vllm_version != source_version:
        raise PatchError(
            "installed vLLM version mismatch (fail closed): "
            f"expected {expect_vllm_version}, source declares {source_version}"
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError(f"source is not utf-8: {exc}") from exc

    if "\r\n" in text:
        raise PatchError(
            "source has CRLF line endings; the anchors are LF-exact (fail closed). "
            "Use the pristine installed-source copy."
        )

    if MARKER in text:
        raise PatchError(
            "source is already patched (marker present); refusing to re-patch. "
            "Re-generate from the pristine installed-source copy instead."
        )

    n_import = _count_occurrences(text, ANCHOR_IMPORT)
    if n_import != 1:
        raise PatchError(
            f"import anchor not unique: found {n_import} occurrences of "
            f"{ANCHOR_IMPORT.strip()!r} (expected exactly 1)"
        )
    n_call = _count_occurrences(text, ANCHOR_CALL)
    if n_call != 1:
        raise PatchError(
            f"call anchor not unique: found {n_call} occurrences of the two-line anchor "
            f"({ANCHOR_NAME}); expected exactly 1"
        )

    before = _statement_counts(text)

    patched = text.replace(ANCHOR_IMPORT, ANCHOR_IMPORT + _import_block(source_sha256), 1)
    patched = patched.replace(ANCHOR_CALL, ANCHOR_CALL + _call_block(), 1)

    # AST gate on the generated copy.
    try:
        tree = ast.parse(patched)
    except SyntaxError as exc:
        raise PatchError(f"generated copy is not valid Python: {exc}") from exc
    if not isinstance(tree, ast.Module):
        raise PatchError("generated copy did not parse to a module")

    after = _statement_counts(patched)
    if after["module"] != before["module"] + 1:
        raise PatchError(
            f"generated copy changed {after['module'] - before['module']} module statements, expected 1"
        )
    for fn_name in PATCHED_FUNCTIONS:
        if after[fn_name] != before[fn_name] + 1:
            raise PatchError(
                f"generated copy changed {after[fn_name] - before[fn_name]} statements in "
                f"{fn_name}(), expected 1"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patched, encoding="utf-8", newline="\n")
    out_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

    meta: dict[str, Any] = {
        "schema": "qfn.capture-patch/1",
        "patch_version": PATCH_VERSION,
        "marker": MARKER,
        "anchor": ANCHOR_CALL,
        "anchor_name": ANCHOR_NAME,
        "anchor_import": ANCHOR_IMPORT.strip(),
        "helper_module": HELPER_MODULE,
        "helper_call": f"{LOCAL_ALIAS}(self, hidden_states, logits, input_batch)",
        "source_path": str(src_path),
        "source_sha256": source_sha256,
        "source_bytes": len(raw),
        "out_path": str(out_path),
        "out_sha256": out_sha256,
        "out_bytes": out_path.stat().st_size,
        "expected_vllm_version": expect_vllm_version,
        "source_version": source_version,
        "module_statements_before": before["module"],
        "module_statements_after": after["module"],
        "patched_function_statement_deltas": {
            fn: after[fn] - before[fn] for fn in PATCHED_FUNCTIONS
        },
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_source_unmodified": hashlib.sha256(src_path.read_bytes()).hexdigest() == source_sha256,
        "notes": (
            "Copy only. The installed source is never written. The one-line "
            "`sample_hidden_states = hidden_states[...]` occurs twice in the pristine file "
            "(808, 1390); the two-line anchor is unique."
        ),
    }

    if checksums_path is not None:
        cp = pathlib.Path(checksums_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(f"{out_sha256}  {out_path.name}\n", encoding="utf-8", newline="\n")
    if meta_path is not None:
        mp = pathlib.Path(meta_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="generate the QFN capture patch copy (never in place)")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checksums", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--expect-sha256", default=EXPECTED_SHA256)
    ap.add_argument("--expect-vllm-version", default=EXPECTED_VLLM_VERSION)
    ap.add_argument("--source-version", default=EXPECTED_VLLM_VERSION)
    ap.add_argument(
        "--no-hash-guard",
        action="store_true",
        help="disable the source sha256 guard (only for a deliberately different input)",
    )
    args = ap.parse_args(argv)
    try:
        meta = build_patch(
            src=args.src,
            out=args.out,
            expect_sha256=None if args.no_hash_guard else args.expect_sha256,
            expect_vllm_version=args.expect_vllm_version,
            source_version=args.source_version,
            checksums_path=args.checksums,
            meta_path=args.meta,
        )
    except PatchError as exc:
        print(f"PATCH REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
