"""Opt-in EXPERIMENTAL typed L1O arm: the served LoRA read through the frozen lettered surface.

What this module is
-------------------
``L1O`` is **not** a trained FP32 decision head (``head.h1`` is untouched). It is the A2 epoch-3
LoRA adapter served by the engine under its own model alias (``l1o``) plus the *masked raw
effective verbalizer rows* that the engine-side capture helper wrote for that exact request. The
readout is the ordinary masked normalisation of the captured 24 raw logits restricted to the
slots the frozen ``qwen_chat_letters-v1`` surface actually uses.

Two surfaces, one request
-------------------------
The frozen lettered surface is CHOICE ONLY (``templates.TEMPLATE_QWEN_CHAT_LETTERS``), while the
public ``noul`` primitive lives in protocol slots 0:2. This module therefore builds an *explicit
internal choice request* for a ``noul`` question:

* option A: key ``false``, description ``false``,
* option B: key ``true``, description ``true``,

reads the returned letter probabilities from the lettered slots 8/9 (never from slots 0/1, and
never by re-rendering ``noul`` with a template that has no lettered form), and projects them back
onto the public ``noul`` contract (slots 0/1, keys ``false``/``true``). The converted request's
capture hash and template are the identity that is reported, because that IS the surface that was
captured; the outward semantics stay ``noul``.

F1-R1 surface policy (PM decision ``noul-surface-PM-decision.json``, 2026-10-13): every one of the
124 accepted A2 ``noul`` rows was rendered with descriptions EQUAL to the keys (``A. false`` /
``B. true``). The earlier ``no`` / ``yes`` wording came from a PM brief mistake, not from the
accepted evidence, and is NOT used here. The public keys, the slot projection and the tie rule are
unchanged by this correction.

Honesty scope (do not weaken)
-----------------------------
* **Declarative pins + CPU verification only.** The config carries the expected checkpoint digest,
  base identity, tokenizer/template identity and capture-helper digest. This module verifies the
  *local* artifact/helper files and the mapping, and reports exactly that. It never manufactures a
  loaded-weights certificate: ``runtime_verified`` stays ``False`` until the PM reads the loaded
  checkpoint back off the running engine.
* **No live evidence.** Nothing here has been run against a real engine. Local HTTP fixtures prove
  the contract and the model isolation, never numerical parity.
* **Never repaired.** A non-finite logit in a slot this arm reads, a missing/mismatched capture,
  a served model that is not the requested alias, or an unsupported primitive is a refusal: no
  substitution, no fallback to the base lane, no coerced readout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import re
from typing import Any, Mapping, Sequence

import numpy as np

from . import features, protocol, templates

__all__ = [
    "ARM_NAME",
    "STATUS_DISABLED",
    "STATUS_EXPERIMENTAL",
    "STATUS_UNAVAILABLE",
    "NOUL_BRIDGE_OPTIONS",
    "SUPPORTED_CAPTURE_SCHEMAS",
    "WIRE_SURFACE_POLICY",
    "AdapterError",
    "L1OConfigError",
    "L1OArmUnavailable",
    "L1OUnsupportedPrimitive",
    "L1OReadoutError",
    "AdapterSpec",
    "AdapterReadout",
    "L1OArm",
]

ARM_NAME = "L1O"
STATUS_DISABLED = "disabled"
STATUS_EXPERIMENTAL = "experimental_unvalidated"
STATUS_UNAVAILABLE = "unavailable"

# The wire surface of the noul bridge: A = key false / description false, B = key true /
# description true, on the lettered slots 8/9 of the frozen ``qwen_chat_letters-v1`` surface.
# This is the surface all 124 accepted A2 noul rows used (PM decision
# ``noul-surface-PM-decision.json``); the public noul keys/slots and the tie rule are unchanged.
NOUL_BRIDGE_OPTIONS = (("false", "false"), ("true", "true"))
WIRE_SURFACE_POLICY = (
    "F1-R1: the internal lettered choice uses A=(key false, description false) and "
    "B=(key true, description true), the exact surface of all 124 accepted frozen A2 noul rows "
    "(PM decision noul-surface-PM-decision.json); the no/yes wording was a PM brief mistake. "
    "Public keys/slots (false/true in slots 0/1) and the lettered slots 8/9 mapping are unchanged."
)
# The capture-record schema this arm is allowed to read (helper-written effective-row sidecars).
SUPPORTED_CAPTURE_SCHEMAS = ("l1v2-effective-rows/1",)

NOT_A_HEAD_NOTE = (
    "L1O is the served LoRA plus the masked raw effective verbalizer rows of the same engine: it "
    "is NOT a separately trained FP32 decision head, and head.h1 is not repurposed for it."
)
EXPECTATION_NOTE = "declarative pins + CPU file verification only"
RUNTIME_VERIFICATION_OWNER = (
    "PM (independent runtime readback of the loaded checkpoint on the serving engine)"
)
HONESTY_NOTE = (
    "EXPERIMENTAL and UNVALIDATED: opt-in, typed-only (noul/choice), disabled by default. The "
    "pins in this block are declarative expectations plus a CPU check of the local artifact and "
    "helper files; they are not production-ready evidence and nothing here is live-certified. "
    "runtime_verified stays false until the PM verifies the loaded checkpoint at runtime."
)

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_ADAPTER_KEYS = {
    "enabled", "status", "model_alias", "provider", "base_url", "template_id", "template_version",
    "chat_template_kwargs", "capture_dir", "capture_dir_allowlist", "expected_base_model",
    "tokenizer_revision", "checkpoint", "helper_path", "helper_sha256", "capture_schema",
    "expected_verbalizer_ids",
}
_CHECKPOINT_KEYS = {"artifact_path", "sha256", "format", "epoch"}
_NOTE_KEYS = {"note"}


class AdapterError(Exception):
    """Base class for every L1O adapter problem."""


class L1OConfigError(AdapterError):
    """The ``decision_adapter`` config block is malformed or unsafe (fail closed at load)."""


class L1OArmUnavailable(AdapterError):
    """The arm cannot serve (disabled, or its pins do not verify) and must never be faked."""


class L1OUnsupportedPrimitive(AdapterError):
    """The requested primitive has no lettered surface on L1O and is never coerced."""


class L1OReadoutError(AdapterError):
    """The capture/served-model identity or the logits make a truthful readout impossible."""


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise L1OConfigError(
            f"{where} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _check_keys(payload: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(payload) - allowed - _NOTE_KEYS)
    if unknown:
        raise L1OConfigError(f"unknown key(s) in {where}: {unknown}")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_expected_rows(raw: Any, *, enabled: bool) -> tuple[int, ...] | None:
    """Validate the pinned effective verbalizer row ids/order (id + order + count are all material)."""
    if raw is None:
        if enabled:
            raise L1OConfigError(
                "decision_adapter.expected_verbalizer_ids is required when the adapter is enabled: "
                "the 24 effective verbalizer row ids AND their order must be pinned, because a "
                "permuted or substituted row silently changes the meaning of every lettered slot"
            )
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise L1OConfigError(
            "decision_adapter.expected_verbalizer_ids must be a list of 24 integers (the effective "
            f"verbalizer row ids in slot order), got {type(raw).__name__}"
        )
    values = list(raw)
    if len(values) != protocol.NUM_SLOTS:
        raise L1OConfigError(
            f"decision_adapter.expected_verbalizer_ids must list exactly {protocol.NUM_SLOTS} rows in "
            f"slot order, got {len(values)}"
        )
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise L1OConfigError(
                f"decision_adapter.expected_verbalizer_ids[{index}] must be an int (a bool is not a "
                f"token id), got {value!r}"
            )
        if value < 0:
            raise L1OConfigError(
                f"decision_adapter.expected_verbalizer_ids[{index}] must be non-negative, got {value!r}"
            )
    distinct = len(set(values))
    if distinct != protocol.NUM_SLOTS:
        raise L1OConfigError(
            f"decision_adapter.expected_verbalizer_ids must be {protocol.NUM_SLOTS} DISTINCT rows "
            f"(24 unique slot rows); only {distinct} distinct value(s) were given"
        )
    return tuple(int(v) for v in values)


@dataclasses.dataclass(frozen=True)
class AdapterSpec:
    """Validated, immutable description of the experimental adapter arm.

    Every field is a *declared* expectation. :meth:`cpu_checks` additionally hashes the local
    artifact/helper files; that is a CPU-side verification of the local copies, nothing more.
    """

    enabled: bool
    declared_status: str | None
    model_alias: str | None
    provider: str
    base_url: str
    template_id: str
    template_version: str
    chat_template_kwargs: dict
    capture_dir: str | None
    capture_dir_allowlist: tuple[str, ...]
    expected_base_model: str | None
    tokenizer_revision: str | None
    checkpoint: dict
    helper_path: str | None
    helper_sha256: str | None
    capture_schema: str | None
    expected_verbalizer_ids: tuple[int, ...] | None
    engine_base_url: str
    engine_model: str
    repo_root: str

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(
        cls,
        raw: Mapping[str, Any],
        *,
        engine_base_url: str,
        engine_model: str,
        tokenizer_revision: str | None,
        repo_root: str | os.PathLike[str] | None = None,
    ) -> "AdapterSpec":
        block = _require_mapping(raw, "decision_adapter")
        _check_keys(block, _ADAPTER_KEYS, "decision_adapter")
        root = str(pathlib.Path(repo_root) if repo_root is not None else pathlib.Path.cwd())

        enabled = block.get("enabled", False)
        if not isinstance(enabled, bool):
            raise L1OConfigError(
                f"decision_adapter.enabled must be a boolean, got {type(enabled).__name__}"
            )
        declared_status = block.get("status")
        if declared_status is not None and not isinstance(declared_status, str):
            raise L1OConfigError("decision_adapter.status must be a string when supplied")

        alias = block.get("model_alias")
        if alias is not None and (not isinstance(alias, str) or not alias):
            raise L1OConfigError("decision_adapter.model_alias must be a non-empty string")
        if enabled and not alias:
            raise L1OConfigError(
                "decision_adapter.model_alias is required when the adapter is enabled: the served "
                "LoRA alias is the only thing that identifies an adapter-on request"
            )
        if alias and alias == engine_model:
            raise L1OConfigError(
                f"decision_adapter.model_alias {alias!r} must differ from the base engine model "
                f"{engine_model!r}: the adapter is served under its own alias, and an ordinary or "
                "typed request must never be silently upgraded to it"
            )

        provider = block.get("provider", "engine")
        if provider != "engine":
            raise L1OConfigError(
                f"decision_adapter.provider must be 'engine' (this arm reads the engine capture "
                f"path), got {provider!r}"
            )

        base_url = block.get("base_url")
        if base_url is None:
            base_url = engine_base_url
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise L1OConfigError(
                f"decision_adapter.base_url must be an http(s) URL (or null to inherit "
                f"engine.base_url), got {base_url!r}"
            )

        template_id = block.get("template_id")
        if template_id != templates.TEMPLATE_QWEN_CHAT_LETTERS:
            raise L1OConfigError(
                "decision_adapter.template_id must be "
                f"{templates.TEMPLATE_QWEN_CHAT_LETTERS!r}: L1O is defined on that frozen "
                f"lettered surface only, got {template_id!r}"
            )
        frozen_version = templates.TEMPLATES[templates.TEMPLATE_QWEN_CHAT_LETTERS].version
        template_version = block.get("template_version")
        if template_version != frozen_version:
            raise L1OConfigError(
                f"decision_adapter.template_version must be {frozen_version!r} (the frozen lettered "
                f"surface), got {template_version!r}"
            )
        kwargs = block.get("chat_template_kwargs", {})
        if not isinstance(kwargs, Mapping):
            raise L1OConfigError("decision_adapter.chat_template_kwargs must be an object")
        if kwargs.get("enable_thinking") is not False:
            raise L1OConfigError(
                "decision_adapter.chat_template_kwargs must set enable_thinking=false: the frozen "
                "lettered surface is a thinking=false surface"
            )

        capture_dir = block.get("capture_dir")
        allow_raw = block.get("capture_dir_allowlist", [])
        if not isinstance(allow_raw, Sequence) or isinstance(allow_raw, (str, bytes)) or \
                not all(isinstance(v, str) and v for v in allow_raw):
            raise L1OConfigError(
                "decision_adapter.capture_dir_allowlist must be a list of absolute paths"
            )
        if enabled and not capture_dir:
            raise L1OConfigError(
                "decision_adapter.capture_dir is required when the adapter is enabled: the operator "
                "must inject the runtime adapter capture directory (there is no default)"
            )
        if capture_dir is not None:
            if not isinstance(capture_dir, str) or not capture_dir:
                raise L1OConfigError("decision_adapter.capture_dir must be a non-empty string or null")
            if not pathlib.Path(capture_dir).is_absolute():
                raise L1OConfigError(
                    f"decision_adapter.capture_dir must be an absolute path, got {capture_dir!r}"
                )
            if any(part == ".." for part in pathlib.Path(capture_dir).parts):
                raise L1OConfigError(
                    f"decision_adapter.capture_dir must not contain '..': {capture_dir!r}"
                )
        allow: list[str] = []
        for entry in allow_raw:
            path = pathlib.Path(entry)
            if not path.is_absolute() or any(part == ".." for part in path.parts):
                raise L1OConfigError(
                    f"decision_adapter.capture_dir_allowlist entry must be an absolute path: {entry!r}"
                )
            allow.append(os.path.realpath(str(path)))
        if capture_dir is not None and allow:
            real = os.path.realpath(capture_dir)
            if not any(real == a or real.startswith(a + os.sep) for a in allow):
                raise L1OConfigError(
                    f"decision_adapter.capture_dir {capture_dir!r} is not inside "
                    f"capture_dir_allowlist {list(allow)!r}"
                )

        expected_base = block.get("expected_base_model")
        if expected_base is not None:
            if not isinstance(expected_base, str) or not expected_base:
                raise L1OConfigError(
                    "decision_adapter.expected_base_model must be a non-empty string or null"
                )
            if expected_base != engine_model:
                raise L1OConfigError(
                    f"decision_adapter.expected_base_model {expected_base!r} does not match the "
                    f"configured base engine model {engine_model!r}; refusing to bind the adapter to "
                    "a different base identity"
                )
        elif enabled:
            raise L1OConfigError(
                "decision_adapter.expected_base_model is required when the adapter is enabled: the "
                "arm must declare which base model identity it is bound to"
            )

        adapter_tok = block.get("tokenizer_revision")
        if adapter_tok is not None and (not isinstance(adapter_tok, str) or not adapter_tok):
            raise L1OConfigError(
                "decision_adapter.tokenizer_revision must be a non-empty string or null"
            )
        if enabled:
            if not adapter_tok:
                raise L1OConfigError(
                    "decision_adapter.tokenizer_revision is required when the adapter is enabled"
                )
            if not tokenizer_revision:
                raise L1OConfigError(
                    "config.tokenizer.revision must be set when the adapter is enabled: without the "
                    "serving tokenizer revision the lettered surface identity cannot be proven"
                )
            if adapter_tok != tokenizer_revision:
                raise L1OConfigError(
                    f"decision_adapter.tokenizer_revision {adapter_tok!r} does not match "
                    f"config.tokenizer.revision {tokenizer_revision!r}; the lettered surface "
                    "tokenizer identity cannot be proven"
                )

        checkpoint = dict(_require_mapping(block.get("checkpoint", {}), "decision_adapter.checkpoint"))
        _check_keys(checkpoint, _CHECKPOINT_KEYS, "decision_adapter.checkpoint")
        artifact = checkpoint.get("artifact_path")
        if artifact is not None and (not isinstance(artifact, str) or not artifact):
            raise L1OConfigError(
                "decision_adapter.checkpoint.artifact_path must be a non-empty string or null"
            )
        digest = checkpoint.get("sha256")
        if digest is None:
            if enabled:
                raise L1OConfigError(
                    "decision_adapter.checkpoint.sha256 is required when the adapter is enabled: an "
                    "unpinned adapter must never be served"
                )
        elif not isinstance(digest, str) or not _HEX64.match(digest):
            raise L1OConfigError(
                f"decision_adapter.checkpoint.sha256 must be a 64-hex-character sha256, got {digest!r}"
            )
        fmt = checkpoint.get("format", "safetensors")
        if not isinstance(fmt, str) or not fmt:
            raise L1OConfigError("decision_adapter.checkpoint.format must be a non-empty string")
        epoch = checkpoint.get("epoch")
        if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0):
            raise L1OConfigError("decision_adapter.checkpoint.epoch must be a non-negative int or null")

        helper_path = block.get("helper_path")
        if helper_path is not None and (not isinstance(helper_path, str) or not helper_path):
            raise L1OConfigError("decision_adapter.helper_path must be a non-empty string or null")
        helper_sha = block.get("helper_sha256")
        if helper_sha is not None and (not isinstance(helper_sha, str) or not _HEX64.match(helper_sha)):
            raise L1OConfigError(
                f"decision_adapter.helper_sha256 must be a 64-hex-character sha256, got {helper_sha!r}"
            )
        if helper_sha is None and enabled:
            raise L1OConfigError(
                "decision_adapter.helper_sha256 is required when the adapter is enabled: the capture "
                "helper that wrote the effective rows must be pinned, and an enabled arm must never "
                "serve while its pin cannot be checked"
            )
        capture_schema = block.get("capture_schema")
        if capture_schema is not None and (not isinstance(capture_schema, str) or not capture_schema):
            raise L1OConfigError("decision_adapter.capture_schema must be a non-empty string or null")
        if enabled:
            if not capture_schema:
                raise L1OConfigError(
                    "decision_adapter.capture_schema is required when the adapter is enabled: the "
                    "arm must declare which capture-record schema it reads"
                )
            if capture_schema not in SUPPORTED_CAPTURE_SCHEMAS:
                raise L1OConfigError(
                    f"decision_adapter.capture_schema {capture_schema!r} is not a supported capture "
                    f"schema {list(SUPPORTED_CAPTURE_SCHEMAS)}; refusing to read an unknown schema"
                )

        expected_rows = _parse_expected_rows(block.get("expected_verbalizer_ids"), enabled=enabled)

        return cls(
            enabled=enabled,
            declared_status=declared_status,
            model_alias=alias,
            provider=provider,
            base_url=base_url,
            template_id=template_id,
            template_version=template_version,
            chat_template_kwargs=dict(kwargs),
            capture_dir=capture_dir,
            capture_dir_allowlist=tuple(allow),
            expected_base_model=expected_base,
            tokenizer_revision=adapter_tok,
            checkpoint={"artifact_path": artifact, "sha256": digest, "format": fmt, "epoch": epoch},
            helper_path=helper_path,
            helper_sha256=helper_sha,
            capture_schema=capture_schema,
            expected_verbalizer_ids=expected_rows,
            engine_base_url=engine_base_url,
            engine_model=engine_model,
            repo_root=root,
        )

    # ------------------------------------------------------------------ derived
    def resolve(self, value: str | None) -> pathlib.Path | None:
        """Resolve a configured path against the repository root when it is relative."""
        if value is None:
            return None
        path = pathlib.Path(value)
        return path if path.is_absolute() else pathlib.Path(self.repo_root) / path

    def capture_provider_kwargs(self) -> dict:
        """The kwargs for :func:`qfn_decision.backend.capture_provider_config` (per-arm provider)."""
        return {
            "base_url": self.base_url,
            "model": self.model_alias,
            "capture_dir": self.capture_dir,
            "capture_dir_allowlist": list(self.capture_dir_allowlist),
            "tokenizer_revision": self.tokenizer_revision,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            "add_special_tokens": False,
            # the engine's own response must name THIS alias, otherwise the response is not
            # adapter-on evidence (a requested body model or cache key proves nothing)
            "expected_served_model": self.model_alias,
            "provider_label": "engine",
            "provider_kind": "engine",
        }

    def cpu_checks(self) -> dict:
        """CPU-side verification of the local artifact/helper copies (never a loaded-weights claim).

        The checkpoint pin may stay declarative (a pinned digest with no local copy is reported as
        unverified, which is the case the PM explicitly allows for the experimental adapter). The
        capture-helper pin must be locally verifiable whenever the arm is enabled.
        """
        return {
            "artifact": self._file_check(
                self.checkpoint.get("artifact_path"), self.checkpoint.get("sha256"),
                what="adapter checkpoint artifact", path_required=False),
            "helper": self._file_check(self.helper_path, self.helper_sha256,
                                       what="capture helper", path_required=self.enabled),
        }

    def _file_check(self, configured: str | None, expected: str | None, *, what: str,
                    path_required: bool) -> dict:
        """Hash a local file against its declared digest. Missing file or digest mismatch fails."""
        out: dict[str, Any] = {
            "path": configured,
            "resolved_path": None,
            "sha256_expected": expected,
            "sha256_actual": None,
            "sha256_verified": None,
            "verify_mode": "declared_only",
            "ok": True,
            "reason": None,
        }
        if expected is None:
            out["reason"] = f"no {what} pin was configured; nothing was verified locally"
            return out
        if configured is None:
            # A pinned digest with no local path can never be verified on this CPU. For the capture
            # helper that makes an ENABLED arm unavailable (it must not serve on an unverifiable
            # pin); for the checkpoint artifact the declared-only experimental case stays allowed.
            if path_required:
                out.update(
                    ok=False,
                    verify_mode="unverifiable_missing_path",
                    reason=(
                        f"no local path is configured for the {what}, so its pinned sha256 cannot be "
                        "verified on this CPU; an enabled arm must not serve on an unverifiable pin"
                    ),
                )
            else:
                out["reason"] = (
                    f"no local path is configured for the {what}; the pinned sha256 stays declarative "
                    "and is NOT CPU-verified here"
                )
            return out
        path = self.resolve(configured)
        out["resolved_path"] = str(path)
        if path is None or not path.is_file():
            out.update(ok=False, reason=f"{what} not found: {path}")
            return out
        actual = _sha256_file(path)
        out["sha256_actual"] = actual
        out["sha256_verified"] = actual == expected.lower()
        out["verify_mode"] = "cpu_local_file_sha256"
        if not out["sha256_verified"]:
            out.update(
                ok=False,
                reason=f"{what} sha256 mismatch: {path} is {actual}, the pin says {expected}",
            )
        return out

    def effective_status(self, *, available: bool) -> str:
        if not self.enabled:
            return STATUS_DISABLED
        return STATUS_EXPERIMENTAL if available else STATUS_UNAVAILABLE

    @property
    def status(self) -> str:
        """The config-level status: ``disabled`` unless the adapter is explicitly enabled.

        Availability (the artifact/helper pins verifying on this CPU) is an arm-level fact and is
        reported as :data:`STATUS_UNAVAILABLE` by the arm/health, never faked here.
        """
        return self.effective_status(available=True)

    # ------------------------------------------------------------------ report blocks
    def health_block(self, *, available: bool, reason: str | None, checks: Mapping[str, Any]) -> dict:
        return {
            "configured": True,
            "enabled": bool(self.enabled),
            "status": self.effective_status(available=available),
            "declared_status": self.declared_status,
            "experimental": True,
            "typed_only": True,
            "available": bool(available),
            "reason": reason,
            "model_alias": self.model_alias,
            "provider": {
                "label": "engine",
                "kind": "engine",
                "base_url": self.base_url,
                "model_alias": self.model_alias,
                "expected_served_model": self.model_alias,
                "inherits_engine_base_url": self.base_url == self.engine_base_url,
            },
            "base_model": {"config": self.engine_model, "expected": self.expected_base_model},
            "template": {
                "template_id": self.template_id,
                "version": self.template_version,
                "frozen": True,
                "chat_template_kwargs": dict(self.chat_template_kwargs),
            },
            "tokenizer": {"revision": self.tokenizer_revision},
            "checkpoint": {
                "artifact_path": self.checkpoint.get("artifact_path"),
                "format": self.checkpoint.get("format"),
                "epoch": self.checkpoint.get("epoch"),
                "sha256_expected": self.checkpoint.get("sha256"),
                "sha256_actual": checks["artifact"]["sha256_actual"],
                "sha256_verified": checks["artifact"]["sha256_verified"],
                "verify_mode": checks["artifact"]["verify_mode"],
            },
            "helper": {
                "path": self.helper_path,
                "sha256_expected": self.helper_sha256,
                "sha256_actual": checks["helper"]["sha256_actual"],
                "sha256_verified": checks["helper"]["sha256_verified"],
                "verify_mode": checks["helper"]["verify_mode"],
            },
            "capture": {
                "dir_configured": self.capture_dir is not None,
                "dir_allowlisted": bool(self.capture_dir_allowlist),
            },
            "capture_schema": {"schema": self.capture_schema},
            "verbalizer": {
                "ordering": [
                    {"slot": protocol.SLOT_RANGES[protocol.KIND_CHOICE][0] + offset,
                     "letter": protocol.CHOICE_LETTERS[offset],
                     "key": key, "description": description}
                    for offset, (key, description) in enumerate(NOUL_BRIDGE_OPTIONS)
                ],
                "expected_rows": list(self.expected_verbalizer_ids or []),
                "expected_rows_note": (
                    "the pinned 24 effective verbalizer row ids in slot order; every L1O readout "
                    "compares the captured record's rows against this pin and refuses on any "
                    "difference (a permuted row silently changes the meaning of the lettered slots)"
                ),
                "tie_rule": (
                    "first_argmax over the active lettered slots: an exact tie selects A/false, "
                    "matching the public first-argmax rule"
                ),
                "wire_surface_policy": WIRE_SURFACE_POLICY,
            },
            "ordinary_adapter_alias_refused": True,
            "expectation": EXPECTATION_NOTE,
            "runtime_verified": False,
            "runtime_verification_owner": RUNTIME_VERIFICATION_OWNER,
            "note": HONESTY_NOTE,
        }

    def request_block(self, *, available: bool, reason: str | None, used: bool) -> dict:
        """The typed-response status block (same truth as health, plus 'used for this request')."""
        return {
            "configured": True,
            "enabled": bool(self.enabled),
            "status": self.effective_status(available=available),
            "experimental": True,
            "typed_only": True,
            "available": bool(available),
            "reason": reason,
            "model_alias": self.model_alias,
            "used_for_this_request": bool(used),
            "runtime_verified": False,
            "expectation": EXPECTATION_NOTE,
            "note": HONESTY_NOTE,
        }


@dataclasses.dataclass
class AdapterReadout:
    """One L1O readout in the *public* contract, plus its provenance block."""

    result: protocol.DecisionResult
    block: dict


class L1OArm:
    """The experimental L1O arm: served LoRA + masked raw effective rows, typed-only.

    ``provider`` is the arm's own capture provider (a separate :class:`CaptureProviderConfig`
    instance with ``model=<alias>``, the frozen lettered template and the adapter capture
    directory). The base arm's provider is never touched or reconfigured for it.
    """

    name = ARM_NAME
    lane = "served_lora_masked_raw_effective_rows"

    def __init__(self, spec: AdapterSpec, *, provider: Any = None) -> None:
        self.spec = spec
        self.provider = provider
        self._checks = spec.cpu_checks()
        self._reason: str | None = None
        if not spec.enabled:
            self._reason = (
                "the experimental L1O adapter is present but disabled "
                "(decision_adapter.enabled=false)"
            )
        else:
            for name in ("artifact", "helper"):
                check = self._checks[name]
                if not check["ok"]:
                    self._reason = check["reason"]
                    break

    # -- availability ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._reason is None

    def unavailable_reason(self) -> str | None:
        return self._reason

    def checks(self) -> dict:
        return self._checks

    # -- request planning --------------------------------------------------------------
    def wire_request(self, request: protocol.DecisionRequest) -> protocol.DecisionRequest:
        """The request whose surface is actually captured for this arm.

        ``choice`` is already the lettered surface (identity). ``noul`` is converted explicitly
        into a two-option lettered choice; ``score`` has no lettered form and is refused.
        """
        if not isinstance(request, protocol.DecisionRequest):
            raise L1OReadoutError(f"expected a DecisionRequest, got {type(request).__name__}")
        if request.kind == protocol.KIND_CHOICE:
            return request
        if request.kind == protocol.KIND_NOUL:
            return protocol.DecisionRequest.create(
                kind=protocol.KIND_CHOICE,
                state=request.state,
                question=request.question,
                options=tuple(
                    protocol.ChoiceOption(key=key, description=description)
                    for key, description in NOUL_BRIDGE_OPTIONS
                ),
                limits=request.limits,
            )
        raise L1OUnsupportedPrimitive(
            f"L1O supports the noul and choice primitives only; {request.kind!r} has no lettered "
            "form on the frozen surface. Refusing instead of coercing it into a choice."
        )

    @staticmethod
    def converted(request: protocol.DecisionRequest, wire: protocol.DecisionRequest) -> bool:
        return request.kind != wire.kind

    # -- readout -----------------------------------------------------------------------
    def read(
        self,
        request: protocol.DecisionRequest,
        wire_request: protocol.DecisionRequest,
        captured: Any,
    ) -> AdapterReadout:
        """Project the captured lettered slots back onto ``request``'s public contract."""
        if not self.available:
            raise L1OArmUnavailable(self._reason or "L1O arm unavailable")
        alias = self.spec.model_alias
        served = getattr(captured, "served_model", None)
        if served != alias:
            raise L1OReadoutError(
                f"the engine response served model {served!r} is not the requested L1O alias "
                f"{alias!r}; refusing to present a base-model response as adapter-on"
            )
        self._require_capture_identity(captured, wire_request)

        logits = np.asarray(captured.slot_logits_fp32, dtype=np.float64).reshape(-1)
        if logits.shape != (protocol.NUM_SLOTS,):
            raise L1OReadoutError(f"captured slot logits must be 24 wide, got {logits.shape}")
        active = list(wire_request.active_slot_indices())
        if not active:
            raise L1OReadoutError("the lettered surface has no active slots for this request")
        active_logits = logits[active]
        if not np.isfinite(active_logits).all():
            raise L1OReadoutError(
                "the captured lettered logits are non-finite in the slot(s) this arm reads: "
                f"{[int(i) for i in active]}; refusing the readout (never repaired, never imputed)"
            )
        # inactive slots are masked before normalisation and can never carry mass, so a
        # placeholder (e.g. -inf or NaN padding) there must not abort the readout. Substituting the
        # minimum active logit keeps the shared protocol path; the substitution is unobservable
        # because the result zeroes every inactive slot (identical to the B0 semantics).
        if not np.isfinite(logits).all():
            logits = np.where(np.isfinite(logits), logits, float(np.min(active_logits)))

        try:
            wire_result = protocol.DecisionResult.from_slot_logits(
                wire_request, [float(v) for v in logits]
            )
        except protocol.ProtocolError as exc:
            raise L1OReadoutError(f"L1O lettered readout failed: {exc}") from exc

        result = self._project(request, wire_request, wire_result)
        expected = _check_probabilities(result)
        block = self._provenance_block(
            request, wire_request, captured, wire_result=wire_result, probabilities=expected
        )
        return AdapterReadout(result=result, block=block)

    def _project(
        self,
        request: protocol.DecisionRequest,
        wire_request: protocol.DecisionRequest,
        wire_result: protocol.DecisionResult,
    ) -> protocol.DecisionResult:
        """Map the lettered probabilities onto the public contract, in the documented order."""
        probs = [float(v) for v in wire_result.option_probabilities]
        if request.kind == protocol.KIND_NOUL:
            if wire_request.kind != protocol.KIND_CHOICE:
                raise L1OReadoutError("the noul bridge must be captured as a lettered choice")
            keys = tuple(wire_request.caller_keys())
            if keys != tuple(key for key, _ in NOUL_BRIDGE_OPTIONS):
                raise L1OReadoutError(
                    f"the converted lettered surface carries caller keys {keys!r}; the bridge "
                    f"requires {tuple(key for key, _ in NOUL_BRIDGE_OPTIONS)!r}"
                )
            if len(probs) != len(keys):
                raise L1OReadoutError("the lettered readout does not match the bridge options")
            vector = [0.0] * protocol.NUM_SLOTS
            vector[0] = probs[0]     # A (false) -> public noul slot 0
            vector[1] = probs[1]     # B (true)  -> public noul slot 1
            try:
                return protocol.DecisionResult.from_slot_probabilities(request, vector)
            except protocol.ProtocolError as exc:  # pragma: no cover - defensive
                raise L1OReadoutError(f"noul projection failed: {exc}") from exc
        if request.kind == protocol.KIND_CHOICE:
            if wire_request is request:
                # identity mapping: the lettered slots 8..n ARE this request's choice slots
                return wire_result
            if tuple(wire_request.caller_keys()) != tuple(request.caller_keys()):
                raise L1OReadoutError("the lettered readout does not match the requested options")
            vector = [0.0] * protocol.NUM_SLOTS
            start = request.active_slot_indices()[0]
            for offset, value in enumerate(probs):
                vector[start + offset] = value
            try:
                return protocol.DecisionResult.from_slot_probabilities(request, vector)
            except protocol.ProtocolError as exc:  # pragma: no cover - defensive
                raise L1OReadoutError(f"choice projection failed: {exc}") from exc
        raise L1OUnsupportedPrimitive(
            f"L1O supports noul and choice only, got {request.kind!r}"
        )

    def _require_capture_identity(self, captured: Any,
                                  wire_request: protocol.DecisionRequest) -> None:
        """Re-check EVERY capture identity at the arm boundary (F1-R1 correction 3).

        The HTTP provider checks several of these as well, but a readout that an injected provider
        could produce must not be able to skip them: the arm is the last gate before a probability
        leaves the process.
        """
        from . import backend as _backend   # deferred: backend imports this module

        caller = getattr(captured, "caller_request_id", None)
        engine_id = getattr(captured, "engine_request_id", None)
        response_id = getattr(captured, "engine_response_id", None)
        record = getattr(captured, "record", None)
        if not isinstance(caller, str) or not caller:
            raise L1OReadoutError("the capture carries no caller request id; refusing the readout")
        if not isinstance(engine_id, str) or not _backend.engine_id_matches(engine_id, caller):
            raise L1OReadoutError(
                f"the capture's engine request id {engine_id!r} does not belong to this request "
                f"{caller!r}; refusing an unrelated record"
            )
        if not isinstance(response_id, str) or not _backend.engine_id_matches(response_id, caller):
            raise L1OReadoutError(
                f"the engine response id {response_id!r} does not belong to this request {caller!r}; "
                "refusing to bind an unrelated response to this capture"
            )
        if record is None:
            raise L1OReadoutError("the capture carries no feature record; refusing the readout")
        record_engine_id = getattr(record, "engine_request_id", None)
        if record_engine_id != engine_id:
            raise L1OReadoutError(
                f"the feature record belongs to {record_engine_id!r} but the capture says "
                f"{engine_id!r}; refusing a mismatched record"
            )
        caller_hash = getattr(captured, "caller_prompt_token_sha256", None)
        record_hash = getattr(record, "prompt_token_sha256", None)
        if not isinstance(caller_hash, str) or not caller_hash or caller_hash != record_hash:
            raise L1OReadoutError(
                f"the record's prompt token hash {record_hash!r} does not equal the token hash this "
                f"request sent ({caller_hash!r}); refusing the readout"
            )
        try:
            # shared fail-closed validator: terminal position, 24 unique rows, vector width/dtype,
            # finiteness of the readable slots (all preserved, not re-implemented here)
            _backend.validate_record(
                record,
                expected_engine_request_id=engine_id,
                expected_prompt_token_sha256=caller_hash,
                active_slot_indices=wire_request.active_slot_indices(),
            )
        except Exception as exc:  # noqa: BLE001 - any record-shape problem fails the readout closed
            raise L1OReadoutError(
                f"the capture record failed its identity/shape validation: {exc}"
            ) from exc

        expected_rows = self.spec.expected_verbalizer_ids
        observed_rows = [int(v) for v in record.verbalizer_ids]
        if expected_rows is not None and tuple(observed_rows) != tuple(expected_rows):
            raise L1OReadoutError(
                "the captured verbalizer row ids/order do not match the configured "
                f"decision_adapter.expected_verbalizer_ids: expected {list(expected_rows)}, observed "
                f"{observed_rows}; a permuted or substituted row silently changes the meaning of every "
                "lettered slot, so the readout is refused"
            )

    # -- provenance --------------------------------------------------------------------
    def _provenance_block(
        self,
        request: protocol.DecisionRequest,
        wire_request: protocol.DecisionRequest,
        captured: Any,
        *,
        wire_result: protocol.DecisionResult,
        probabilities: Sequence[float],
    ) -> dict:
        spec = self.spec
        checks = self._checks
        record = captured.record
        converted = self.converted(request, wire_request)
        return {
            "arm": ARM_NAME,
            "lane": self.lane,
            "experimental": True,
            "status": STATUS_EXPERIMENTAL,
            "typology": "opt-in experimental typed arm",
            "not_a_decision_head": NOT_A_HEAD_NOTE,
            "model_alias": spec.model_alias,
            "served_model": getattr(captured, "served_model", None),
            "engine_base_url": spec.base_url,
            "base_model": {"config": spec.engine_model, "expected": spec.expected_base_model},
            "checkpoint": {
                "artifact_path": spec.checkpoint.get("artifact_path"),
                "format": spec.checkpoint.get("format"),
                "epoch": spec.checkpoint.get("epoch"),
                "sha256_expected": spec.checkpoint.get("sha256"),
                "sha256_actual": checks["artifact"]["sha256_actual"],
                "sha256_verified": checks["artifact"]["sha256_verified"],
                "verify_mode": checks["artifact"]["verify_mode"],
            },
            "helper": {
                "path": spec.helper_path,
                "sha256_expected": spec.helper_sha256,
                "sha256_actual": checks["helper"]["sha256_actual"],
                "sha256_verified": checks["helper"]["sha256_verified"],
                "verify_mode": checks["helper"]["verify_mode"],
            },
            "capture_schema": {
                "schema": spec.capture_schema,
                "features_schema": features.FEATURES_SCHEMA,
                "token_hash_schema": features.TOKEN_HASH_SCHEMA,
                "helper_version": record.helper_version,
                "engine_build_version": record.engine_build_version,
                "runner_source_sha256": record.runner_source_sha256,
                "capture_dir": str(spec.capture_dir),
                "record_file_name": record.file_stem() + ".npz",
            },
            "template": {
                "template_id": spec.template_id,
                "version": spec.template_version,
                "frozen": True,
                "chat_template_kwargs": dict(spec.chat_template_kwargs),
                "template_sha256": templates.template_sha256(spec.template_id, wire_request),
                "surface": "letters",
            },
            "tokenizer": {"revision": spec.tokenizer_revision},
            "verbalizer": {
                "ordering": [
                    {"slot": int(slot), "letter": protocol.CHOICE_LETTERS[offset],
                     "key": key, "description": description}
                    for offset, (slot, (key, description)) in enumerate(zip(
                        wire_request.active_slot_indices(), NOUL_BRIDGE_OPTIONS))
                ] if converted else [
                    {"slot": int(slot), "letter": protocol.CHOICE_LETTERS[offset],
                     "key": key, "description": description}
                    for offset, (slot, key, description) in enumerate(zip(
                        wire_request.active_slot_indices(), wire_request.caller_keys(),
                        wire_request.descriptions()))
                ],
                "tie_rule": (
                    "first_argmax over the active lettered slots: an exact tie selects A/false, "
                    "matching the public first-argmax rule"
                ),
                "effective_rows": [int(v) for v in record.verbalizer_ids],
                "expected_rows": list(self.spec.expected_verbalizer_ids or []),
                "observed_rows": [int(v) for v in record.verbalizer_ids],
                "rows_match_expected": True,
                "rows_source": "configured_pin_vs_captured_record",
                "wire_surface_policy": WIRE_SURFACE_POLICY,
                "effective_rows_note": (
                    "the masked raw effective verbalizer rows of the served checkpoint, read back "
                    "from this capture record and compared to the configured pin before any readout "
                    "(not re-derived from head.h1)"
                ),
                "masked": True,
                "read_slots": [int(i) for i in wire_request.active_slot_indices()],
                "image_max_probability": float(wire_result.raw_max_probability),
                "outward_max_probability": float(max(probabilities)),
            },
            "bridge": {
                "converted": bool(converted),
                "converted_from_kind": request.kind if converted else None,
                "outward_kind": request.kind,
                "wire_kind": wire_request.kind,
                "bridge_id": "noul_letters_bridge/1" if converted else None,
                "option_map": (
                    [
                        {"slot": int(slot), "letter": protocol.CHOICE_LETTERS[offset],
                         "key": key, "description": description}
                        for offset, (slot, (key, description)) in enumerate(
                            zip(wire_request.active_slot_indices(), NOUL_BRIDGE_OPTIONS))
                    ] if converted else None
                ),
                "projection": (
                    "lettered slot 8 -> public noul slot 0 (false), slot 9 -> slot 1 (true)"
                    if converted else "identity: the lettered choice slots are the public choice slots"
                ),
                "rollback": (
                    "the lettered slot probabilities and the internal choice request are reported "
                    "verbatim, so the projection can be re-derived and checked"
                ),
            },
            "expectation": EXPECTATION_NOTE,
            "runtime_verified": False,
            "runtime_verification_owner": RUNTIME_VERIFICATION_OWNER,
            "note": HONESTY_NOTE,
        }

    def surface_blocks(
        self,
        request: protocol.DecisionRequest,
        wire_request: protocol.DecisionRequest,
        captured: Any,
    ) -> dict:
        """The blocks that must describe the ACTUAL captured surface (not the outward contract)."""
        spec = self.spec
        converted = self.converted(request, wire_request)
        return {
            "template": {
                "template_id": spec.template_id,
                "version": spec.template_version,
                "frozen": True,
                "template_sha256": templates.template_sha256(spec.template_id, wire_request),
                "outward_kind": request.kind,
                "wire_kind": wire_request.kind,
                "converted_from_kind": request.kind if converted else None,
                "confirms_bridge_surface": "letters",
                "note": (
                    "this is the surface that was actually tokenized and captured for this arm; the "
                    "outward kind/probabilities stay in the public contract of the original request"
                ),
            },
            "engine": {
                "base_url": spec.base_url,
                "model": spec.model_alias,
                "requested_model": spec.model_alias,
                "served_model": getattr(captured, "served_model", None),
                "adapter_alias": spec.model_alias,
                "adapter_lane": True,
            },
            "capture": {
                "adapter_alias": spec.model_alias,
                "outward_kind": request.kind,
                "wire_kind": wire_request.kind,
                "converted": bool(converted),
                "bridge": "noul_letters_bridge/1" if converted else None,
                "template_sha256": templates.template_sha256(spec.template_id, wire_request),
            },
        }


def _check_probabilities(result: protocol.DecisionResult) -> tuple[float, ...]:
    """Fail closed on a non-finite / unnormalised projection (mirrors the B0/H1 guard)."""
    probs = result.option_probabilities
    if not probs or not all(isinstance(v, float) and np.isfinite(v) for v in probs):
        raise L1OReadoutError("L1O readout produced non-finite probabilities")
    total = float(sum(probs))
    if abs(total - 1.0) > 1e-5:
        raise L1OReadoutError(f"L1O readout probabilities sum to {total!r}, not 1")
    if any(p < 0.0 for p in probs):
        raise L1OReadoutError("L1O readout produced a negative probability")
    return tuple(float(v) for v in probs)
