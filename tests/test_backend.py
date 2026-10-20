"""S6 tests: feature-acquisition backends, head arms and gateway config (no live engine).

Scope and honesty rules for this file
-------------------------------------
* Everything here runs on the Windows host with ``.venv/Scripts/python.exe`` (Python 3.12,
  numpy, pytest, stdlib only). No torch, no GPU, no inference service is started or stopped.
* The HTTP interaction is *real* HTTP against a **fixture** engine that this file starts on an
  allocated localhost port in a background thread. The fixture engine is labelled
  ``fixture_only`` everywhere; it is a synthetic feature provider, not the vLLM engine, and
  nothing in this file is parity, accuracy or performance evidence. Live capture/parity stays
  NOT YET RUN until the PM integrates the real transport (see reports/s6/RESULT.md).
* No accuracy or benchmark claim is made from the unit mappings asserted below; they only
  check that the readout is the documented masked normalisation of whatever numbers arrive.
"""

from __future__ import annotations

import http.client
import hashlib
import json
import os
import pathlib
import re
import time

import numpy as np
import pytest

from qfn_decision import backend, features, head as head_mod, protocol, templates


# ======================================================================================
# helpers
# ======================================================================================

def _raw_request(kind="choice", n_options=4):
    options = [protocol.ChoiceOption(key=f"k{i}", description=f"option {i}") for i in range(n_options)]
    if kind == "choice":
        return protocol.DecisionRequest.create(
            kind="choice", state="state text", question="which?", options=options
        )
    if kind == "noul":
        return protocol.DecisionRequest.create(kind="noul", state="state text", question="is it true?")
    return protocol.DecisionRequest.create(
        kind="score",
        state="state text",
        question="how reliable?",
        score_rubrics=[f"level {i}" for i in range(6)],
    )


def _head_fixture(tmp_path, *, template_id=templates.TEMPLATE_AUTOTRUST_RAW, template_version="bare-v1",
                  tokenizer_revision="fixture-tok-rev", seed=7, bias=None, name=None):
    """A real DecisionHead artifact written by the shared head module (fixture data only)."""
    prov = head_mod.HeadProvenance(
        feature_base="fixture/effective-lm-head-rows",
        feature_width=head_mod.FEATURE_WIDTH,
        template_id=template_id,
        template_version=template_version,
        tokenizer_revision=tokenizer_revision,
        rows=tuple(range(1000, 1024)),
        row_scaling=1.0,
        rows_sha256="0" * 64,
        init_source="fixture/effective-lm-head-rows",
    )
    stem = name or f"h1_fixture_{template_version}_{tokenizer_revision}_{seed}".replace("/", "_")
    art = pathlib.Path(tmp_path) / f"{stem}.npz"
    if art.exists():
        art.unlink()
        art.with_suffix(".json").unlink(missing_ok=True)
    h = head_mod.DecisionHead.random_control(seed=seed, provenance=prov, scale=0.05, bias=bias)
    h.save(art)
    return art


def _fixture_engine(tmp_path, **kw):
    server = backend.FixtureEngineServer(model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture", **kw)
    server.start()
    return server


def _provider(engine, tmp_path, **kw):
    cfg = backend.capture_provider_config(
        base_url=engine.base_url,
        model=engine.model,
        capture_dir=engine.capture_dir,
        capture_dir_allowlist=[str(pathlib.Path(tmp_path) / "capture")],
        tokenizer_revision="fixture-tok-rev",
        # this test's data source is the synthetic fixture engine, so say so
        provider_label=backend.FIXTURE_ONLY_LABEL,
        **kw,
    )
    return backend.HttpEngineCaptureProvider(cfg)


# ======================================================================================
# request-id / record-name contract (mirrors patches/qfn_capture_runtime.py)
# ======================================================================================

def test_caller_request_id_is_server_generated_and_allowlisted():
    a = backend.new_caller_request_id()
    b = backend.new_caller_request_id()
    assert a.startswith(backend.CALLER_REQUEST_ID_PREFIX)
    assert a != b
    assert backend.marked_request_id_ok(a)
    for bad in ("", "cmpl-qfn-x-0", "qfn-../x", "qfn-a/b", "qfn-" + "z" * 500):
        assert not backend.marked_request_id_ok(bad)


def test_engine_request_id_matches_documented_suffix_rule():
    assert backend.engine_request_id_for("qfn-abc123") == "cmpl-qfn-abc123-0"
    assert backend.engine_request_id_for("qfn-abc123", prompt_index=3) == "cmpl-qfn-abc123-3"
    with pytest.raises(backend.BackendError):
        backend.engine_request_id_for("nope-1")


def test_expected_record_file_name_mirrors_engine_helper_rule():
    eng = "cmpl-qfn-0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f-pulled"
    name = backend.expected_record_file_name(eng, "a" * 64, dup=0)
    assert name == f"cmpl-qfn-0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f-pulled__{'a'*16}__0.npz"
    # the rule is the one features.FeatureRecord.file_stem() implements
    assert features._safe_name(eng) + "__" + "a" * 16 + "__0" + ".npz" == name


def test_expected_record_file_name_rejects_traversal_and_wrong_prefix():
    with pytest.raises(backend.BackendError):
        backend.expected_record_file_name("cmpl-qfn-../etc/passwd", "b" * 64)
    with pytest.raises(backend.BackendError):
        backend.expected_record_file_name("qfn-something", "b" * 64)  # engine id must carry the marker
    with pytest.raises(backend.BackendError):
        backend.expected_record_file_name("cmpl-qfn-x-0", "nothex")


def test_resolve_record_files_denies_escape_and_symlinks(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    inside = cap / f"cmpl-qfn-a__{'c'*16}__0.npz"
    inside.write_bytes(b"not a real npz")
    names = backend.resolve_record_files(str(cap), "cmpl-qfn-a", "c" * 64)
    assert [p.name for p in names] == [inside.name]
    # an escape attempt and a symlink are refused, not repaired
    with pytest.raises(backend.CaptureError):
        backend.resolve_record_files(str(cap), "cmpl-qfn-..", "c" * 64)
    link = cap / f"cmpl-qfn-b__{'d'*16}__0.npz"
    try:
        os.symlink(str(inside), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    with pytest.raises(backend.CaptureError):
        backend.resolve_record_files(str(cap), "cmpl-qfn-b", "d" * 64)


# ---- continuation 2026-10-10: opaque ids may carry interior dots --------------------------

def test_safe_engine_key_accepts_interior_dots_but_rejects_path_hazards():
    # a legitimate marker+interior-dot id keeps its exact characters (no normalisation of the
    # id itself); the derived basename is the deterministic safe mapping
    eng = "cmpl-qfn-0f0f-..0f0f-0"
    assert backend._safe_engine_key(eng) == features._safe_name(eng)
    assert backend._safe_engine_key(eng) == eng
    for evil in ("cmpl-qfn-..", "cmpl-qfn-../x", "cmpl-qfn-a/b", "cmpl-qfn-a\\b", "",
                 "cmpl-qfn-a\x00b", "C:/cmpl-qfn-a", "..cmpl-qfn-x", None):
        with pytest.raises(backend.BackendError):
            backend._safe_engine_key(evil)


def test_expected_record_file_name_keeps_interior_dots_and_binds_the_hash():
    eng = "cmpl-qfn-aa..bb-0"
    name = backend.expected_record_file_name(eng, "a" * 64, dup=0)
    assert name == f"{eng}__{'a' * 16}__0.npz"
    assert "/" not in name and "\\" not in name
    assert backend.expected_record_file_name(eng, "b" * 64, dup=1).endswith(f"__{'b' * 16}__1.npz")


def test_resolve_record_files_finds_an_interior_dot_id(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    name = backend.expected_record_file_name("cmpl-qfn-aa..bb-0", "e" * 64)
    (cap / name).write_bytes(b"not a real npz")
    found = backend.resolve_record_files(str(cap), "cmpl-qfn-aa..bb-0", "e" * 64)
    assert [p.name for p in found] == [name]


def test_discover_record_files_still_refuses_malicious_ids(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    (cap / f"cmpl-qfn-aa__{'f' * 16}__0.npz").write_bytes(b"x")
    for evil in ("cmpl-qfn-../evil", "cmpl-qfn-A..", "cmpl-qfn-a/b", "..\\evil"):
        with pytest.raises(backend.CaptureError):
            backend.discover_record_files(str(cap), evil, "f" * 64)


# ======================================================================================
# record validation
# ======================================================================================

def _record(engine_id="cmpl-qfn-1a1a-0", tok_hash=None, prompt_len=9, terminal=None, logits=None, **kw):
    tok_hash = tok_hash or features.token_sequence_sha256(list(range(prompt_len)))
    vec = np.linspace(-1.0, 1.0, 2560, dtype=np.float32)
    bits = features.fp32_to_bf16_bits(vec)
    wide = features.bf16_bits_to_fp32(bits)
    meta = dict(
        example_id="ex-1",
        engine_request_id=engine_id,
        prompt_token_sha256=tok_hash,
        prompt_tokens_count=prompt_len,
        prompt_len=prompt_len,
        terminal_position=(prompt_len - 1) if terminal is None else terminal,
        cache_hit_tokens=0,
        num_scheduled_tokens=prompt_len,
        query_start=0,
        query_end=prompt_len,
        logits_row_index=0,
        batch_index=0,
        internal_index=0,
        vector_bf16=bits,
        vector_fp32=wide,
        verbalizer_ids=list(range(1000, 1024)),
        verbalizer_logits_fp32=np.zeros(24, dtype=np.float32) if logits is None else np.asarray(logits, dtype=np.float32),
        runtime_argmax_token_id=100,
        runtime_max_logit_fp32=0.0,
        head_scale=1.0,
        head_soft_cap=None,
        head_org_vocab_size=248320,
        head_dtype="bfloat16",
        lm_head_tp_size=2,
        tp_rank=0,
        world_size=2,
        cudagraph_mode="FULL_DECODE_ONLY",
        kernel_mode="prefill",
        runner_source_sha256="f" * 64,
        capture_wall_ns=time.time_ns(),
        logits_captured=True,
        logits_absent_reason=None,
        vector_source="runner.sample.hidden_states[logits_index=0]",
        helper_version="qfn-capture-runtime/1",
    )
    meta.update(kw)
    # the shared reader gates are strict by design; this helper represents invalid records on
    # purpose (they must still be refused by the gateway, which reads outside the hatch)
    with features.fixture_records_allowed():
        return features.FeatureRecord(**meta)


def test_validate_record_accepts_a_well_formed_capture():
    rec = _record()
    backend.validate_record(
        rec,
        expected_engine_request_id=rec.engine_request_id,
        expected_prompt_token_sha256=rec.prompt_token_sha256,
        expected_prompt_len=rec.prompt_len,
    )


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (dict(terminal_position=3), "terminal"),
        (dict(logits_captured=False, logits_absent_reason="verbalizer_ids_unresolved"), "logits_captured"),
        (dict(prompt_len=9, prompt_tokens_count=8), "features_enforced"),  # FeatureRecord itself refuses
    ],
)
def test_validate_record_fails_closed(mutate, reason):
    try:
        rec = _record(**mutate)
    except features.FeatureError:
        assert reason == "features_enforced"
        return
    with pytest.raises(backend.CaptureError):
        backend.validate_record(
            rec,
            expected_engine_request_id=rec.engine_request_id,
            expected_prompt_token_sha256=rec.prompt_token_sha256,
            expected_prompt_len=rec.prompt_len,
        )


def test_validate_record_rejects_wrong_hash_id_and_nan_logits():
    rec = _record()
    with pytest.raises(backend.CaptureError):
        backend.validate_record(rec, expected_prompt_token_sha256="0" * 64, expected_prompt_len=9)
    with pytest.raises(backend.CaptureError):
        backend.validate_record(rec, expected_engine_request_id="cmpl-qfn-other-0", expected_prompt_len=9)
    nan_rec = _record(logits=np.array([np.nan] * 24, dtype=np.float32))
    with pytest.raises(backend.CaptureError):
        backend.validate_record(nan_rec, expected_prompt_len=9)


def test_validate_record_rejects_stale_and_future_wall_ns():
    rec = _record(capture_wall_ns=1)
    with pytest.raises(backend.CaptureError):
        backend.validate_record(rec, expected_prompt_len=9, not_before_wall_ns=time.time_ns())


def test_decode_record_bytes_refuses_corrupt_and_unknown_meta():
    blob = features.encode_record_npz(_record())
    rec, dropped = backend.decode_record_bytes(blob)
    assert rec.engine_request_id == "cmpl-qfn-1a1a-0" and dropped == ()

    with pytest.raises(backend.CaptureError):
        backend.decode_record_bytes(b"this is not an npz")

    extra = backend.fixture_only_npz_bytes(_record(), extra_meta={"raw_vector_dtype": "torch.bfloat16"})
    with pytest.raises(backend.CaptureError):
        backend.decode_record_bytes(extra)  # strict by default: unknown provenance is never dropped silently
    rec2, dropped2 = backend.decode_record_bytes(extra, tolerate_unknown_meta_fields=True)
    assert dropped2 == ("raw_vector_dtype",) and rec2.engine_request_id == rec.engine_request_id


# ======================================================================================
# real-HTTP capture provider against the fixture engine
# ======================================================================================

def test_http_provider_captures_a_record_end_to_end(tmp_path):
    eng = _fixture_engine(tmp_path)
    try:
        prov = _provider(eng, tmp_path)
        req = _raw_request("choice", 4)
        cap = prov.acquire(req, deadline_s=10.0)

        assert cap.provider_label == "fixture_only"
        assert cap.transport == backend.TRANSPORT_ONE_DISCARDED_TOKEN
        assert cap.vector_fp32.shape == (2560,) and cap.vector_fp32.dtype == np.float32
        assert cap.slot_logits_fp32.shape == (24,)
        assert cap.record.engine_request_id.startswith(backend.ENGINE_REQUEST_ID_PREFIX)
        # PM live facts: the HTTP response id is `cmpl-<caller>` (no index), while the record carries
        # the engine-internal `cmpl-<caller>-0-<8 hex>` id. Both must be accepted and matched.
        assert cap.engine_response_id == f"cmpl-{cap.caller_request_id}"
        assert cap.record.engine_request_id == cap.engine_request_id
        assert backend.engine_id_matches(cap.engine_request_id, cap.caller_request_id)
        assert cap.record.terminal_position == cap.record.prompt_len - 1
        assert cap.usage["prompt_tokens"] == cap.record.prompt_len
        assert cap.caller_request_id in cap.record.engine_request_id
        # honest labels
        assert cap.cache_evidence == backend.CACHE_EVIDENCE_UNVERIFIED
        assert "cache_hit_tokens" in cap.cache_raw
        assert set(cap.timing_ms) >= {"tokenize", "request", "capture_wait", "readout", "total"}
        assert cap.timing_ms["total"] >= cap.timing_ms["capture_wait"] >= 0.0
        # the client hash it verified is the one it tokenized, from the engine's own /tokenize
        assert cap.record.prompt_token_sha256 == cap.caller_prompt_token_sha256
        # no generation-token classification: the returned text is never kept
        assert not hasattr(cap, "completion_text") or getattr(cap, "completion_text", None) is None
    finally:
        eng.stop()


def test_http_provider_is_deterministic_for_the_same_prompt(tmp_path):
    eng = _fixture_engine(tmp_path)
    try:
        prov = _provider(eng, tmp_path)
        req = _raw_request("noul")
        a = prov.acquire(req, deadline_s=10.0)
        b = prov.acquire(req, deadline_s=10.0)
        assert a.caller_prompt_token_sha256 == b.caller_prompt_token_sha256
        assert a.caller_request_id != b.caller_request_id
        assert a.record.engine_request_id != b.record.engine_request_id
    finally:
        eng.stop()


@pytest.mark.parametrize(
    "fault,exc",
    [
        ("missing", backend.DeadlineExceeded),
        ("corrupt_npz", backend.CaptureError),
        ("wrong_hash", backend.CaptureError),
        ("bad_terminal", backend.CaptureError),
        ("stale", backend.CaptureError),
        ("oversize", backend.CaptureError),
        ("no_logits", backend.CaptureError),
        ("wrong_engine_id", backend.CaptureError),
        ("symlink", backend.CaptureError),
        ("engine_500", backend.TransportError),
        ("engine_400", backend.TransportError),
        ("bad_tokenize", backend.TransportError),
        ("usage_mismatch", backend.CaptureError),
    ],
)
def test_http_provider_fails_closed_on_engine_faults(tmp_path, fault, exc):
    if fault == "symlink" and not _can_symlink(tmp_path):
        pytest.skip(
            "this host cannot create symlinks (WinError 1314, no SeCreateSymbolicLinkPrivilege), so the "
            "symlink denial path cannot be exercised here; the guard below is unit-tested via a "
            "simulated escaping realpath instead"
        )
    eng = _fixture_engine(tmp_path, fault=fault)
    try:
        kw = {"record_max_bytes": 64 * 1024} if fault == "oversize" else {}
        prov = _provider(eng, tmp_path, capture_deadline_s=1.5, poll_initial_ms=5.0, poll_max_ms=25.0, **kw)
        with pytest.raises(exc):
            prov.acquire(_raw_request("choice", 3), deadline_s=2.0)
    finally:
        eng.stop()


def _can_symlink(tmp_path) -> bool:
    probe_target = pathlib.Path(tmp_path) / "probe-target.txt"
    probe_link = pathlib.Path(tmp_path) / "probe-link.txt"
    probe_target.write_text("x", encoding="utf-8")
    try:
        if probe_link.exists() or probe_link.is_symlink():
            probe_link.unlink()
        os.symlink(str(probe_target), str(probe_link))
    except (OSError, NotImplementedError):
        return False
    return probe_link.is_symlink()


def test_resolve_record_files_denies_a_record_that_resolves_outside_the_capture_dir(tmp_path, monkeypatch):
    """The escape guard is exercised on this host by simulating the link it cannot create."""
    cap = tmp_path / "capture"
    cap.mkdir()
    name = f"cmpl-qfn-c__{'e'*16}__0.npz"
    (cap / name).write_bytes(b"whatever")
    outside = str(tmp_path / "outside" / name)

    real_realpath = backend.os.path.realpath

    def fake_realpath(path, *args, **kwargs):
        resolved = real_realpath(path, *args, **kwargs)
        if os.path.basename(str(path)) == name:
            return outside
        return resolved

    monkeypatch.setattr(backend.os.path, "realpath", fake_realpath)
    with pytest.raises(backend.CaptureError) as err:
        backend.resolve_record_files(str(cap), "cmpl-qfn-c", "e" * 64)
    assert "outside the capture directory" in str(err.value)


def test_http_provider_requires_engine_socket_errors_to_surface(tmp_path):
    # nothing is listening on this port: must be a transport error, not a capture failure
    cfg = backend.capture_provider_config(
        base_url="http://127.0.0.1:1",
        model="fixture-model",
        capture_dir=str(tmp_path / "capture"),
        capture_dir_allowlist=[str(tmp_path / "capture")],
        tokenizer_revision="fixture-tok-rev",
        socket_timeout_s=1.0,
    )
    prov = backend.HttpEngineCaptureProvider(cfg)
    with pytest.raises(backend.TransportError):
        prov.acquire(_raw_request("noul"), deadline_s=2.0)


def test_http_provider_refuses_unallowlisted_capture_dirs(tmp_path):
    with pytest.raises(backend.ConfigError):
        backend.capture_provider_config(
            base_url="http://127.0.0.1:9",
            model="fixture-model",
            capture_dir=str(tmp_path / "elsewhere"),
            capture_dir_allowlist=[str(tmp_path / "capture")],
            tokenizer_revision="rev",
        )
    with pytest.raises(backend.ConfigError):
        backend.capture_provider_config(
            base_url="http://127.0.0.1:9",
            model="fixture-model",
            capture_dir="relative/capture",
            capture_dir_allowlist=[str(tmp_path / "capture")],
            tokenizer_revision="rev",
        )


def test_http_provider_sends_the_documented_completion_payload(tmp_path):
    eng = _fixture_engine(tmp_path)
    try:
        prov = _provider(eng, tmp_path)
        prov.acquire(_raw_request("score"), deadline_s=10.0)
        body = [b for b in eng.completion_bodies if b.get("request_id")][-1]
        assert body["max_tokens"] == 1
        assert body["temperature"] == 1.0 and body["top_p"] == 1.0
        assert body["top_k"] == 0 and body["min_p"] == 0.0
        assert body["presence_penalty"] == 0.0 and body["frequency_penalty"] == 0.0
        assert body["repetition_penalty"] == 1.0
        assert body["add_special_tokens"] is False
        assert body["stream"] is False
        assert isinstance(body["seed"], int)
        assert isinstance(body["prompt"], list) and all(isinstance(i, int) for i in body["prompt"])
        assert body["logprobs"] is None and body.get("echo") in (None, False)
    finally:
        eng.stop()


def test_http_provider_tokenize_request_uses_the_raw_template_and_no_special_tokens(tmp_path):
    eng = _fixture_engine(tmp_path)
    try:
        prov = _provider(eng, tmp_path)
        prov.acquire(_raw_request("choice", 2), deadline_s=10.0)
        tok = [b for b in eng.tokenize_bodies if isinstance(b.get("prompt"), str)][-1]
        assert tok["add_special_tokens"] is False
        assert tok["prompt"].startswith("[kind] choice\n[state] state text\n[question] which?")
        assert tok["prompt"].endswith("\n[decision]:")
        assert "prompt" in tok and "messages" not in tok
    finally:
        eng.stop()


# ======================================================================================
# qwen_chat_letters-v1: the chat (messages) tokenization + capture path (J3)
# ======================================================================================

LETTERS_TEMPLATE = templates.TEMPLATE_QWEN_CHAT_LETTERS
LETTERS_KWARGS = {"enable_thinking": False}


def _chat_provider(engine, tmp_path, **kw):
    """The same real-HTTP provider, configured with the chat-letters template."""
    kw.setdefault("template_id", LETTERS_TEMPLATE)
    kw.setdefault("chat_template_kwargs", dict(LETTERS_KWARGS))
    return _provider(engine, tmp_path, **kw)


def test_http_provider_tokenize_payload_is_exact_in_raw_and_chat_mode(tmp_path):
    """The raw payload stays byte-identical; the chat payload carries messages and NO prompt."""
    eng = _fixture_engine(tmp_path)
    try:
        raw_req = _raw_request("choice", 2)
        _provider(eng, tmp_path).acquire(raw_req, deadline_s=10.0)
        assert eng.tokenize_bodies[-1] == {
            "model": "fixture-model",
            "prompt": templates.render(raw_req, templates.TEMPLATE_AUTOTRUST_RAW),
            "add_special_tokens": False,
        }

        chat_req = _raw_request("choice", 2)
        _chat_provider(eng, tmp_path).acquire(chat_req, deadline_s=10.0)
        chat_tok = eng.tokenize_bodies[-1]
        assert chat_tok == {
            "model": "fixture-model",
            "messages": templates.render_messages(chat_req, LETTERS_TEMPLATE),
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        assert "prompt" not in chat_tok
        assert "add_special_tokens" not in chat_tok
    finally:
        eng.stop()


def test_http_provider_chat_mode_captures_a_record_end_to_end(tmp_path):
    """The completion/capture/readout path is unchanged: only the ids come from messages."""
    eng = _fixture_engine(tmp_path)
    try:
        prov = _chat_provider(eng, tmp_path)
        req = _raw_request("choice", 4)
        cap = prov.acquire(req, deadline_s=10.0)

        sent_ids = eng.completion_bodies[-1]["prompt"]
        assert all(isinstance(i, int) for i in sent_ids) and sent_ids
        assert eng.completion_bodies[-1]["max_tokens"] == 1
        # the request identity hash covers exactly the ids the engine was given
        assert cap.caller_prompt_token_sha256 == features.token_sequence_sha256(sent_ids)
        assert cap.record.prompt_token_sha256 == cap.caller_prompt_token_sha256
        assert cap.record.prompt_len == len(sent_ids)
        assert cap.usage["prompt_tokens"] == len(sent_ids)
        assert cap.record.terminal_position == cap.record.prompt_len - 1
        # those ids are the messages form, not the raw prompt text
        assert sent_ids != backend._fixture_tokenize(templates.render(req, templates.TEMPLATE_AUTOTRUST_RAW))
        assert sent_ids == backend._fixture_tokenize_messages(
            templates.render_messages(req, LETTERS_TEMPLATE), add_generation_prompt=True
        )
        assert cap.slot_logits_fp32.shape == (24,)
        assert cap.provider_label == backend.FIXTURE_ONLY_LABEL
    finally:
        eng.stop()


def test_http_provider_chat_mode_is_deterministic_for_the_same_messages(tmp_path):
    eng = _fixture_engine(tmp_path)
    try:
        prov = _chat_provider(eng, tmp_path)
        req = _raw_request("choice", 3)
        a = prov.acquire(req, deadline_s=10.0)
        b = prov.acquire(req, deadline_s=10.0)
        assert a.caller_prompt_token_sha256 == b.caller_prompt_token_sha256
        assert a.caller_request_id != b.caller_request_id
    finally:
        eng.stop()


@pytest.mark.parametrize("kind", ["noul", "score"])
def test_http_provider_refuses_noul_and_score_under_the_chat_letters_template(tmp_path, kind):
    """The letters surface is choice-only: the refusal happens before any engine call."""
    eng = _fixture_engine(tmp_path)
    try:
        prov = _chat_provider(eng, tmp_path)
        with pytest.raises(templates.TemplateError) as err:
            prov.acquire(_raw_request(kind), deadline_s=5.0)
        assert kind in str(err.value) and LETTERS_TEMPLATE in str(err.value)
        assert eng.tokenize_bodies == [] and eng.completion_bodies == []
    finally:
        eng.stop()


def test_fixture_feature_provider_supports_the_chat_template(tmp_path):
    prov = backend.FixtureFeatureProvider(seed=3, capture_dir=pathlib.Path(tmp_path) / "cap",
                                          template_id=LETTERS_TEMPLATE)
    req = _raw_request("choice", 3)
    cap = prov.acquire(req, deadline_s=1.0)
    expected_ids = backend._fixture_tokenize_messages(templates.render_messages(req, LETTERS_TEMPLATE),
                                                      add_generation_prompt=True)
    assert cap.record.prompt_len == len(expected_ids)
    assert cap.record.prompt_token_sha256 == features.token_sequence_sha256(expected_ids)
    assert cap.record.prompt_token_sha256 == cap.caller_prompt_token_sha256
    assert cap.record.terminal_position == cap.record.prompt_len - 1
    with pytest.raises(templates.TemplateError):
        prov.acquire(_raw_request("noul"), deadline_s=1.0)


def test_h1_arm_refuses_a_head_fitted_for_a_different_template(tmp_path):
    """H1 was fitted on raw-template features: under the chat template it must be refused."""
    art = _head_fixture(tmp_path)          # provenance: autotrust_raw_bare-v1 / bare-v1
    arm = backend.H1Arm(artifact_path=str(art), template_id=LETTERS_TEMPLATE,
                        template_version="letters-v1", tokenizer_revision="fixture-tok-rev")
    assert arm.available is False
    reason = arm.unavailable_reason()
    assert "provenance mismatch" in reason and "template_id mismatch" in reason
    assert "autotrust_raw_bare-v1" in reason
    with pytest.raises(backend.ArmUnavailable):
        arm.read(_raw_request("choice", 2), _captured_from())


def test_api_config_accepts_the_chat_letters_template_and_enforces_its_version(tmp_path):
    base = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "configs" / "decision_api.json").read_text(encoding="utf-8")
    )
    base["template"] = {"template_id": LETTERS_TEMPLATE, "version": "letters-v1",
                        "chat_template_kwargs": dict(LETTERS_KWARGS)}
    base["capture"]["dir"] = str(pathlib.Path(tmp_path) / "capture")
    base["capture"]["dir_allowlist"] = [str(pathlib.Path(tmp_path) / "capture")]
    cfg = backend.ApiConfig.from_dict(base)
    assert cfg.template_id == LETTERS_TEMPLATE
    assert cfg.template_version == "letters-v1"
    assert cfg.chat_template_kwargs == LETTERS_KWARGS
    assert cfg.capture_provider_config().chat_template_kwargs == LETTERS_KWARGS

    wrong_version = json.loads(json.dumps(base))
    wrong_version["template"]["version"] = "bare-v1"
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict(wrong_version)

    unknown = json.loads(json.dumps(base))
    unknown["template"]["template_id"] = "qwen_chat_letters-v2"
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict(unknown)


# ======================================================================================
# arms: B0 from raw captured logits, H1 from the FP32 head
# ======================================================================================

def _captured_from(engine_id="cmpl-qfn-1a1a-0", logits=None, vector=None, prompt_len=5):
    tok_hash = features.token_sequence_sha256(list(range(prompt_len)))
    vec = np.linspace(-1.0, 1.0, 2560, dtype=np.float32) if vector is None else np.asarray(vector, np.float32)
    bits = features.fp32_to_bf16_bits(vec)
    rec = _record(
        engine_id=engine_id,
        tok_hash=tok_hash,
        prompt_len=prompt_len,
        logits=np.arange(24, dtype=np.float32) if logits is None else np.asarray(logits, np.float32),
        vector_bf16=bits,
        vector_fp32=features.bf16_bits_to_fp32(bits),
    )
    return backend.CapturedFeature(
        record=rec,
        vector_fp32=rec.vector_fp32,
        slot_logits_fp32=rec.verbalizer_logits_fp32,
        caller_request_id="qfn-abc",
        engine_request_id=rec.engine_request_id,
        engine_response_id=rec.engine_request_id,
        usage={"prompt_tokens": prompt_len, "completion_tokens": 1, "total_tokens": prompt_len + 1},
        timing_ms={"tokenize": 0.1, "request": 0.2, "capture_wait": 0.3, "readout": 0.4, "total": 1.0},
        transport=backend.TRANSPORT_ONE_DISCARDED_TOKEN,
        provider_label="fixture_only",
        cache_raw={"cache_hit_tokens": 0, "num_scheduled_tokens": prompt_len},
        caller_prompt_token_sha256=tok_hash,
    )


def test_b0_arm_masks_inactive_slots_and_keeps_caller_key_order():
    arm = backend.B0Arm()
    req = _raw_request("choice", 3)
    read = arm.read(req, _captured_from())
    probs = read.result.option_probabilities
    assert len(probs) == 3
    assert read.result.caller_keys == ("k0", "k1", "k2")
    assert load_pub(read.result.as_payload()["probabilities"], "k0", "k1", "k2")
    assert abs(sum(probs) - 1.0) < 1e-6
    # inactive slots carry exactly zero mass and never receive any
    assert all(v == 0.0 for i, v in enumerate(read.result.slot_probabilities) if not (8 <= i < 11))
    # the raw logits are what came off the engine, softmaxed over the active slots only
    logits = np.arange(24, dtype=np.float64)[8:11]
    expect = np.exp(logits - logits.max())
    expect = expect / expect.sum()
    assert np.allclose(probs, expect, atol=1e-6)
    assert read.block["lane"] == "raw_captured_logits"
    assert read.block["calibration_status"] == "not_applicable"
    assert read.block["trained"] is False


def test_b0_arm_noul_and_score_readouts():
    b0 = backend.B0Arm()
    noul = b0.read(_raw_request("noul"), _captured_from()).result
    assert noul.caller_keys == ("false", "true")
    assert noul.p_true is not None and 0.0 <= noul.p_true <= 1.0
    assert noul.as_payload()["jev_confidence"] is None
    score = b0.read(_raw_request("score"), _captured_from()).result
    assert score.expected_level is not None
    assert 0.0 <= score.expected_level <= 5.0
    assert abs(sum(score.level_probabilities) - 1.0) < 1e-6


def test_b0_arm_never_returns_nan_or_partial_probabilities():
    b0 = backend.B0Arm()
    logits = np.full(24, -np.inf, dtype=np.float32)
    logits[8:10] = 0.0
    read = b0.read(_raw_request("choice", 2), _captured_from(logits=logits)).result
    assert all(np.isfinite(v) for v in read.option_probabilities)
    bad = np.full(24, -np.inf, dtype=np.float32)
    with pytest.raises(backend.BackendError):
        backend.B0Arm().read(_raw_request("choice", 2), _captured_from(logits=bad))


def test_h1_arm_is_unavailable_without_an_artifact(tmp_path):
    h1 = backend.H1Arm.from_config_dict({"artifact_path": None, "verify_checksum": True}, template_id="t", template_version="v")
    assert h1.available is False
    assert "artifact" in h1.unavailable_reason()
    with pytest.raises(backend.ArmUnavailable):
        h1.read(_raw_request("choice", 2), _captured_from())


def test_h1_arm_reads_the_bf16_widened_vector_through_the_shared_head(tmp_path):
    art = _head_fixture(tmp_path)
    h1 = backend.H1Arm.from_config_dict(
        {"artifact_path": str(art), "verify_checksum": True},
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision="fixture-tok-rev",
    )
    assert h1.available is True
    req = _raw_request("choice", 4)
    cap = _captured_from()
    read = h1.read(req, cap)
    assert read.block["arm"] == "H1"
    assert read.block["calibration_status"] == "uncalibrated"
    assert read.block["trained"] is False
    assert read.block["head_artifact"]["data_sha256"]
    assert read.block["provenance"]["template_version"] == "bare-v1"

    # independent numpy check of the documented mapping (unit mapping, not an accuracy claim)
    loaded = head_mod.DecisionHead.load(art)
    logits = cap.vector_fp32 @ loaded.weight.T + loaded.bias
    mask = head_mod.active_mask("choice", n_options=4)
    z = logits[mask]
    e = np.exp(z - z.max())
    expect = e / e.sum()
    assert np.allclose(read.result.option_probabilities, expect, atol=1e-6)
    assert all(v == 0.0 for i, v in enumerate(read.result.slot_probabilities) if not (8 <= i < 12))


def test_h1_arm_fails_closed_on_provenance_template_mismatch(tmp_path):
    art = _head_fixture(tmp_path, template_version="other-v9")
    h1 = backend.H1Arm.from_config_dict(
        {"artifact_path": str(art), "verify_checksum": True},
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision="fixture-tok-rev",
    )
    assert h1.available is False
    assert "template" in h1.unavailable_reason() or "provenance" in h1.unavailable_reason()
    with pytest.raises(backend.ArmUnavailable):
        h1.read(_raw_request("noul"), _captured_from())


def test_h1_arm_fails_closed_on_tokenizer_revision_mismatch_and_missing_file(tmp_path):
    art = _head_fixture(tmp_path, tokenizer_revision="other-rev")
    h1 = backend.H1Arm.from_config_dict(
        {"artifact_path": str(art), "verify_checksum": True},
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision="fixture-tok-rev",
    )
    assert h1.available is False and "tokenizer" in h1.unavailable_reason()

    missing = backend.H1Arm.from_config_dict(
        {"artifact_path": str(tmp_path / "absent.npz"), "verify_checksum": True},
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision="fixture-tok-rev",
    )
    assert missing.available is False

    undeclared = backend.H1Arm.from_config_dict(
        {"artifact_path": str(_head_fixture(tmp_path, tokenizer_revision="r2")), "verify_checksum": True},
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision=None,
    )
    assert undeclared.available is False and "tokenizer_revision" in undeclared.unavailable_reason()


def test_h1_arm_refuses_a_head_built_from_different_rows_than_the_capture(tmp_path):
    """A head initialized from different lm_head rows must never read a different capture."""
    prov = head_mod.HeadProvenance(
        feature_base="fixture/effective-lm-head-rows",
        feature_width=head_mod.FEATURE_WIDTH,
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision="fixture-tok-rev",
        rows=tuple(range(2000, 2024)),          # deliberately different rows
        row_scaling=1.0,
        rows_sha256="0" * 64,
        init_source="fixture/effective-lm-head-rows",
    )
    art = pathlib.Path(tmp_path) / "rows_mismatch.npz"
    head_mod.DecisionHead.random_control(seed=5, provenance=prov, scale=0.05).save(art)
    h1 = backend.H1Arm.from_config_dict(
        {"artifact_path": str(art), "verify_checksum": True},
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision="fixture-tok-rev",
    )
    assert h1.available is True
    with pytest.raises(backend.BackendError) as err:
        h1.read(_raw_request("noul"), _captured_from())
    assert "rows do not match" in str(err.value)


def test_h1_arm_and_b0_disagree_but_both_are_well_formed(tmp_path):
    art = _head_fixture(tmp_path, seed=3)
    h1 = backend.H1Arm.from_config_dict(
        {"artifact_path": str(art), "verify_checksum": True},
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision="fixture-tok-rev",
    )
    req = _raw_request("noul")
    cap = _captured_from()
    b0 = backend.B0Arm().read(req, cap).result
    h1r = h1.read(req, cap).result
    for r in (b0, h1r):
        assert abs(sum(r.option_probabilities) - 1.0) < 1e-6
        assert all(np.isfinite(v) for v in r.option_probabilities)
    # the two arms are distinct readouts; no parity claim is made from fixtures
    assert b0.option_probabilities != h1r.option_probabilities


# ======================================================================================
# decision backend + config
# ======================================================================================

class _FakeProvider:
    label = "fixture_only"

    def __init__(self, captured, calls=None):
        self._captured = captured
        self.calls = calls if calls is not None else []

    def acquire(self, request, *, deadline_s, seed=None):
        self.calls.append((request.kind, deadline_s))
        return self._captured


def test_decision_backend_returns_a_complete_honest_outcome(tmp_path):
    cfg = backend.load_api_config(_write_config(tmp_path))
    prov = _FakeProvider(_captured_from())
    be = backend.DecisionBackend(cfg, provider=prov)
    out = be.decide(_raw_request("choice", 3))

    assert out.arm == "B0"
    assert out.arm_block["lane"] == "raw_captured_logits"
    assert out.result.caller_keys == ("k0", "k1", "k2")
    assert out.transport == backend.TRANSPORT_ONE_DISCARDED_TOKEN
    assert out.usage["typed_output_tokens"] == 0
    assert out.usage["internal_generated_tokens"] == 1
    assert out.uses_s2 is False
    assert out.cache["cache_evidence"] == backend.CACHE_EVIDENCE_UNVERIFIED
    payload = backend.outcome_payload(out)
    assert payload["output_tokens"] == 0
    assert payload["jev_confidence"] is None
    assert payload["confidence_status"] == protocol.CONFIDENCE_STATUS_ABSENT
    assert payload["probabilities"] and "selected" in payload
    assert payload["engine"]["engine_request_id"].startswith(backend.ENGINE_REQUEST_ID_PREFIX)
    assert payload["template"]["template_id"] == templates.TEMPLATE_AUTOTRUST_RAW
    assert payload["template"]["version"] == "bare-v1"
    # no prompt text and no generated verbalizer text leaves the service
    blob = json.dumps(payload)
    assert "state text" not in blob and "which?" not in blob and "option 0" not in blob
    assert not (_forbidden_keys(payload) & {"completion_text", "generated_text", "reasoning", "text"})


def _forbidden_keys(obj):
    """Every dict key anywhere in a response payload."""
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found |= _forbidden_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _forbidden_keys(v)
    return found


# ---- continuation 2026-10-10: the suite must not write into the repo root -----------------

def _root_snapshot(root):
    """(size, mtime_ns, sha256) for every file directly in the repository root."""
    out = {}
    for p in sorted(root.iterdir()):
        if p.is_file():
            blob = p.read_bytes()
            out[p.name] = (len(blob), p.stat().st_mtime_ns, hashlib.sha256(blob).hexdigest())
    return out


def test_backend_suite_writes_only_into_tmp_paths_not_the_repo_root(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    before = _root_snapshot(root)
    cfg = backend.load_api_config(_write_config(tmp_path, h1_path=None))
    be = backend.DecisionBackend(cfg, provider=_FakeProvider(_captured_from()))
    assert be.available_arms()["H1"] is False
    after = _root_snapshot(root)
    assert after == before                     # nothing created, changed or removed at the root
    # a pre-existing root decision_api.json (left in place by an earlier defect) is untouched
    root_cfg = root / "decision_api.json"
    if root_cfg.exists():
        assert after["decision_api.json"] == before["decision_api.json"]
    # and the file this test generated lives under tmp_path
    assert (tmp_path / "decision_api.json").is_file()


def test_decision_backend_refuses_h1_when_the_arm_is_unavailable(tmp_path):
    # continuation 2026-10-10 hygiene fix: this used to pass pathlib.Path(os.getcwd()) and
    # wrote decision_api.json into the REPO ROOT. tmp_path keeps the suite's writes inside
    # pytest's temp dir; the pre-existing root file is left untouched.
    cfg = backend.load_api_config(_write_config(tmp_path, h1_path=None))
    be = backend.DecisionBackend(cfg, provider=_FakeProvider(_captured_from()))
    assert be.available_arms()["B0"] is True
    assert be.available_arms()["H1"] is False
    with pytest.raises(backend.ArmUnavailable):
        be.decide(_raw_request("noul"), arm="H1")
    with pytest.raises(backend.BackendError):
        be.decide(_raw_request("noul"), arm="H9")


def test_decision_backend_enforces_the_frozen_template_version(tmp_path):
    cfg = backend.load_api_config(_write_config(tmp_path))
    raw = cfg.as_dict()
    raw["template"] = {**raw["template"], "version": "not-bare-v1"}
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict(raw)          # refused at config load
    raw = cfg.as_dict()
    raw["template"] = {**raw["template"], "require_version_match": False}
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict(raw)          # the rule cannot be switched off in this stage


def test_api_config_rejects_an_unknown_template_and_unallowlisted_capture_dir(tmp_path):
    cfg = backend.load_api_config(_write_config(tmp_path))
    raw = cfg.as_dict()
    raw["template"] = {**raw["template"], "template_id": "made-up-template"}
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict(raw)
    raw = cfg.as_dict()
    raw["capture"] = {**raw["capture"], "dir": str(pathlib.Path(tmp_path) / "not-allowlisted")}
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict(raw)
    raw = cfg.as_dict()
    raw["capture"] = {**raw["capture"], "transport": "engine_native_zero_sampler"}
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict(raw)
    raw = cfg.as_dict()
    raw["logging"] = {**raw["logging"], "redact_prompts": False}
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict(raw)


def test_health_reports_readiness_not_live_success(tmp_path):
    cfg = backend.load_api_config(_write_config(tmp_path))
    be = backend.DecisionBackend(cfg, provider=_FakeProvider(_captured_from()))
    h = be.health()
    assert h["status"] in ("ready", "degraded")
    assert h["readiness"]["live_engine_validated"] is False
    assert h["readiness"]["engine_ping"] in ("not_checked", "ok", "error")
    assert h["provider"]["label"] == "fixture_only"
    assert {a["arm"] for a in h["arms"]} == {"B0", "H1"}
    assert h["template"]["template_id"] == templates.TEMPLATE_AUTOTRUST_RAW
    assert h["protocol"]["typed_output_tokens"] == 0


def _write_config(tmp_path, h1_path="__none__", tokenizer_revision=None):
    src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "decision_api.json"
    cfg = json.loads(src.read_text(encoding="utf-8"))
    cfg["capture"]["dir"] = str(pathlib.Path(tmp_path) / "capture")
    cfg["capture"]["dir_allowlist"] = [str(pathlib.Path(tmp_path) / "capture")]
    if h1_path is not None:
        cfg["head"]["h1"]["artifact_path"] = None if h1_path == "__none__" else str(h1_path)
    if tokenizer_revision is not None:
        cfg["tokenizer"]["revision"] = tokenizer_revision
    out = pathlib.Path(tmp_path) / "decision_api.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def test_api_config_defaults_load_and_are_strict(tmp_path):
    cfg = backend.load_api_config(_write_config(tmp_path))
    assert cfg.host == "127.0.0.1" and cfg.port == 8890
    assert cfg.engine_base_url.startswith("http")
    assert cfg.concurrency_max_inflight == 8
    assert cfg.capture_transport == backend.TRANSPORT_ONE_DISCARDED_TOKEN
    assert cfg.limits.max_state_chars == protocol.ProtocolLimits().max_state_chars
    assert cfg.provider == "engine"

    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict({**cfg.as_dict(), "nope": 1})
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict({**cfg.as_dict(), "port": "8890"})
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict({**cfg.as_dict(), "port": 0})
    with pytest.raises(backend.ConfigError):
        backend.ApiConfig.from_dict({**cfg.as_dict(), "concurrency": {"max_inflight": 0}})


def test_default_forward_headers_carry_no_secrets():
    cfg = backend.ApiConfig.from_dict(backend.default_api_config_dict())
    assert all("authorization" not in h.lower() for h in cfg.forward_headers)
    assert set(cfg.forward_headers_from_env) <= {"authorization", "x-api-key"}
    assert cfg.redact_prompts is True


def load_pub(obj, *keys):
    """Assert a JSON object preserves caller-key insertion order."""
    assert list(obj.keys()) == list(keys)
    return True


# ======================================================================================
# live engine contract (PM live facts, 2026-10-10): request-id forms, record discovery,
# CAPTURE_ERROR.json handling and per-primitive calibration reporting
# ======================================================================================

LIVE_HEX = "89c4f0b7"
CALLER = "qfn-" + "ab" * 16


def test_resolve_engine_request_id_accepts_the_live_response_id_forms():
    # live: the HTTP response id is `cmpl-<caller>` with NO index suffix
    assert backend.resolve_engine_request_id(f"cmpl-{CALLER}", CALLER) == f"cmpl-{CALLER}"
    # documented per-prompt form, with and without the engine's random 8-hex suffix
    assert backend.resolve_engine_request_id(f"cmpl-{CALLER}-0", CALLER) == f"cmpl-{CALLER}-0"
    assert backend.resolve_engine_request_id(f"cmpl-{CALLER}-0-{LIVE_HEX}", CALLER) == f"cmpl-{CALLER}-0-{LIVE_HEX}"
    assert backend.resolve_engine_request_id(f"cmpl-{CALLER}-3", CALLER) == f"cmpl-{CALLER}-3"
    # an absent id falls back to the documented rule
    assert backend.resolve_engine_request_id(None, CALLER) == f"cmpl-{CALLER}-0"
    assert backend.resolve_engine_request_id("", CALLER) == f"cmpl-{CALLER}-0"
    # everything else is still refused: another request, another prefix, a malformed suffix
    for bad in (
        f"cmpl-qfn-{'cd' * 16}-0",
        "cmpl-notmarked-0",
        "cmpl-",
        f"cmpl-{CALLER}-0-{LIVE_HEX.upper()}",
        f"cmpl-{CALLER}-0-89c4f0b",
        f"cmpl-{CALLER}-x",
        f"{CALLER}",
        "random-id",
        12345,
    ):
        with pytest.raises(backend.BackendError):
            backend.resolve_engine_request_id(bad, CALLER)


def test_engine_id_matches_accepts_only_the_documented_forms():
    for good in (
        f"cmpl-{CALLER}",
        f"cmpl-{CALLER}-0",
        f"cmpl-{CALLER}-0-{LIVE_HEX}",
        f"cmpl-{CALLER}-7",
        f"cmpl-{CALLER}-7-{LIVE_HEX}",
    ):
        assert backend.engine_id_matches(good, CALLER)
    for bad in (
        f"cmpl-{CALLER}-0-{LIVE_HEX.upper()}",
        f"cmpl-{CALLER}-0-{LIVE_HEX}8",
        f"cmpl-{CALLER}-0-89c4f0b",
        f"cmpl-{CALLER}--0",
        f"cmpl-{CALLER}-",
        f"cmpl-qfn-{'cd' * 16}-0",
        "cmpl-",
        CALLER,
        "",
    ):
        assert not backend.engine_id_matches(bad, CALLER)
    # when the caller declares its prompt index, another index is not this request
    assert backend.engine_id_matches(f"cmpl-{CALLER}-0-{LIVE_HEX}", CALLER, prompt_index=0)
    assert not backend.engine_id_matches(f"cmpl-{CALLER}-7", CALLER, prompt_index=0)


def test_discover_record_files_finds_every_accepted_id_form(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    prompt_hash = "c" * 64
    forms = [
        f"cmpl-{CALLER}-0-{LIVE_HEX}",   # live suffixed engine-internal id
        f"cmpl-{CALLER}-0",              # documented per-prompt id
        f"cmpl-{CALLER}",                # bare response id
    ]
    for engine_id in forms:
        for stale in cap.glob("*.npz"):
            stale.unlink()
        name = f"{engine_id}__{prompt_hash[:16]}__0.npz"
        (cap / name).write_bytes(b"record")
        found = backend.discover_record_files(str(cap), CALLER, prompt_hash)
        assert [(p.name, eid) for p, eid in found] == [(name, engine_id)]


def test_discover_record_files_ignores_other_requests_and_other_hashes(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    prompt_hash = "c" * 64
    other_caller = "qfn-" + "cd" * 16
    (cap / f"cmpl-{other_caller}-0-{LIVE_HEX}__{prompt_hash[:16]}__0.npz").write_bytes(b"other request")
    (cap / f"cmpl-{CALLER}-0-{LIVE_HEX}__{'d' * 16}__0.npz").write_bytes(b"other prompt hash")
    (cap / f"cmpl-{CALLER}-0-{LIVE_HEX}.json").write_bytes(b"not a record")
    assert backend.discover_record_files(str(cap), CALLER, prompt_hash) == []


def test_discover_record_files_treats_a_second_dup_file_as_ambiguous(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    prompt_hash = "c" * 64
    (cap / f"cmpl-{CALLER}-0-{LIVE_HEX}__{prompt_hash[:16]}__0.npz").write_bytes(b"a")
    (cap / f"cmpl-{CALLER}-0-{LIVE_HEX}__{prompt_hash[:16]}__1.npz").write_bytes(b"b")
    with pytest.raises(backend.CaptureError) as err:
        backend.discover_record_files(str(cap), CALLER, prompt_hash)
    assert "ambiguous" in str(err.value)


def test_discover_record_files_refuses_two_candidates_never_guesses(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    prompt_hash = "c" * 64
    (cap / f"cmpl-{CALLER}-0-{LIVE_HEX}__{prompt_hash[:16]}__0.npz").write_bytes(b"a")
    (cap / f"cmpl-{CALLER}-0__{prompt_hash[:16]}__0.npz").write_bytes(b"b")
    with pytest.raises(backend.CaptureError) as err:
        backend.discover_record_files(str(cap), CALLER, prompt_hash)
    assert "ambiguous" in str(err.value)


def test_discover_record_files_keeps_the_escape_and_symlink_guards(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    prompt_hash = "c" * 64
    with pytest.raises(backend.CaptureError):
        backend.discover_record_files(str(cap), "qfn-..", prompt_hash)
    with pytest.raises(backend.CaptureError):
        backend.discover_record_files("relative/capture", CALLER, prompt_hash)
    if _can_symlink(tmp_path):
        target = tmp_path / "outside.npz"
        target.write_bytes(b"record")
        link = cap / f"cmpl-{CALLER}-0-{LIVE_HEX}__{prompt_hash[:16]}__0.npz"
        os.symlink(str(target), str(link))
        with pytest.raises(backend.CaptureError):
            backend.discover_record_files(str(cap), CALLER, prompt_hash)


def _fixture_engine_live(tmp_path, **kw):
    return _fixture_engine(tmp_path, **kw)


@pytest.mark.parametrize("mode", ["plain", "index", "suffixed"])
def test_http_provider_accepts_every_record_id_mode(tmp_path, mode):
    eng = _fixture_engine(tmp_path, record_id_mode=mode)
    try:
        prov = _provider(eng, tmp_path)
        cap = prov.acquire(_raw_request("noul"), deadline_s=10.0)
        assert backend.engine_id_matches(cap.engine_request_id, cap.caller_request_id)
        assert cap.record.engine_request_id == cap.engine_request_id
        assert cap.error_marker is None
    finally:
        eng.stop()


def test_http_provider_matches_the_live_suffixed_record(tmp_path):
    eng = _fixture_engine(tmp_path)          # default now mirrors the live engine
    try:
        prov = _provider(eng, tmp_path)
        cap = prov.acquire(_raw_request("choice", 2), deadline_s=10.0)
        assert cap.engine_response_id == f"cmpl-{cap.caller_request_id}"
        assert cap.engine_request_id == cap.record.engine_request_id
        assert re.fullmatch(r"cmpl-qfn-[0-9a-f]{32}-0-[0-9a-f]{8}", cap.engine_request_id)
    finally:
        eng.stop()


def test_http_provider_refuses_an_ambiguous_capture_dir(tmp_path):
    eng = _fixture_engine(tmp_path, fault="two_records")
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=1.5, poll_initial_ms=5.0, poll_max_ms=25.0)
        with pytest.raises(backend.CaptureError) as err:
            prov.acquire(_raw_request("noul"), deadline_s=2.0)
        assert "ambiguous" in str(err.value)
    finally:
        eng.stop()


NON_FATAL_MARKER = {
    "helper_version": "qfn-capture-runtime/1",
    "marker_prefix": "cmpl-qfn-",
    "last_error": {"kind": "row_export_failed", "error": "RuntimeError: fixture", "wall_ns": 1},
    "error_count": 1,
    "skips": {"chunk_not_final": 3},
    "records_written": 1,
    "stopped": False,
}


def test_non_request_error_marker_does_not_fail_the_request(tmp_path):
    eng = _fixture_engine(tmp_path, error_marker=NON_FATAL_MARKER)
    try:
        prov = _provider(eng, tmp_path)
        cap = prov.acquire(_raw_request("noul"), deadline_s=10.0)
        assert cap.record.engine_request_id == cap.engine_request_id
        assert cap.error_marker is not None
        assert cap.error_marker["fatal"] is False
        assert cap.error_marker["kind"] == "row_export_failed"
        assert cap.error_marker["skips"] == {"chunk_not_final": 3}
        diag = prov.diagnostics()
        assert diag["error_marker_seen"] is True and diag["error_marker"]["kind"] == "row_export_failed"
    finally:
        eng.stop()


def test_error_marker_identifying_this_request_fails_only_that_request(tmp_path):
    marker = {
        "last_error": {
            "kind": "capture_skipped",
            "error": "terminal_not_chunk_tail",
            "wall_ns": 2,
            "request_id": "{caller}",
        },
        "records_written": 0,
        "stopped": False,
    }
    eng = _fixture_engine(tmp_path, error_marker=marker)
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=1.5, poll_initial_ms=5.0, poll_max_ms=25.0)
        with pytest.raises(backend.CaptureError) as err:
            prov.acquire(_raw_request("noul"), deadline_s=2.0)
        assert "identifies this request" in str(err.value) or "this request" in str(err.value)
    finally:
        eng.stop()


def test_global_stop_marker_fails_the_request(tmp_path):
    marker = {
        "last_error": {"kind": "quota_exceeded", "error": "capture quota reached", "wall_ns": 3},
        "records_written": 4096,
        "stopped": True,
    }
    eng = _fixture_engine(tmp_path, error_marker=marker, fault="missing")
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=1.5, poll_initial_ms=5.0, poll_max_ms=25.0)
        with pytest.raises(backend.CaptureError) as err:
            prov.acquire(_raw_request("noul"), deadline_s=2.0)
        assert "quota_exceeded" in str(err.value) or "stopped" in str(err.value)
    finally:
        eng.stop()


def test_non_fatal_marker_without_a_record_still_polls_until_the_deadline(tmp_path):
    eng = _fixture_engine(tmp_path, error_marker=NON_FATAL_MARKER, fault="missing")
    try:
        prov = _provider(eng, tmp_path, capture_deadline_s=1.0, poll_initial_ms=5.0, poll_max_ms=25.0)
        with pytest.raises(backend.DeadlineExceeded):
            prov.acquire(_raw_request("noul"), deadline_s=1.2)
        assert prov.diagnostics()["error_marker_seen"] is True
    finally:
        eng.stop()


def test_error_marker_classifier_is_explicit_about_the_rule():
    caller = CALLER
    base = f"cmpl-{caller}-0"
    fatal, why = backend.error_marker_is_fatal(NON_FATAL_MARKER, caller)
    assert fatal is False and why is None
    fatal, why = backend.error_marker_is_fatal(
        {"last_error": {"kind": "capture_exception", "error": "x", "engine_request_id": base}}, caller
    )
    assert fatal is True and "request" in why
    fatal, why = backend.error_marker_is_fatal({"stopped": True, "last_error": {"kind": "anything"}}, caller)
    assert fatal is True and "stopped" in why
    fatal, why = backend.error_marker_is_fatal({"last_error": {"kind": "quota_exceeded", "error": "q"}}, caller)
    assert fatal is True and "global" in why
    # a marker about a different request must not fail this one
    fatal, _ = backend.error_marker_is_fatal(
        {"last_error": {"kind": "capture_exception", "error": "x", "engine_request_id": f"cmpl-qfn-{'cd' * 16}-0"}},
        caller,
    )
    assert fatal is False


# ---------------------------------------------------------------- per-primitive calibration


def _head_fitted_for(tmp_path, primitives=("noul",), temperature=2.5, name="h1_calibrated"):
    """A fixture head whose calibration covers only ``primitives`` (fixture-only data)."""
    art = _head_fixture(tmp_path, name=name)
    head = head_mod.DecisionHead.load(art)
    support = {
        kind: {"n_rows": 32, "scope": "fixture_scope", "source": "fixture_noul_rows", "calibrated": True}
        for kind in primitives
    }
    cal = head_mod.CalibrationState(
        temperatures={k: (temperature if k in primitives else 1.0) for k in head_mod.PRIMITIVE_KEYS},
        fitted=True,
        method="fixture_isotonic",
        source="fixture_calibration_rows",
        scope="fixture_scope",
        fitted_primitives=tuple(primitives),
        parameter_fingerprint=head.parameter_fingerprint(),
        provenance_fingerprint=head.provenance_fingerprint(),
        data_fingerprint="b" * 64,
        per_primitive=support,
    )
    head.with_calibration(cal).save(art, overwrite=True)
    return art


def test_h1_reports_per_primitive_calibration_never_the_overall_claim(tmp_path):
    art = _head_fitted_for(tmp_path, ("noul",))
    h1 = backend.H1Arm.from_config_dict(
        {"artifact_path": str(art), "verify_checksum": True},
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision="fixture-tok-rev",
    )
    assert h1.available is True

    noul_block = h1.read(_raw_request("noul"), _captured_from()).block
    assert noul_block["calibration_status"] == "calibrated"
    assert noul_block["calibration_status_overall"] == "calibrated"
    assert noul_block["calibration_primitive"]["primitive"] == "noul"
    assert noul_block["calibration_primitive"]["temperature"] == 2.5

    for kind, n_options in (("choice", 3), ("score", None)):
        request = _raw_request(kind, n_options) if n_options else _raw_request(kind)
        block = h1.read(request, _captured_from()).block
        assert block["calibration_status_overall"] == "calibrated"       # overall is still reported
        assert block["calibration_status"] == "uncalibrated"            # but never as this kind's claim
        assert block["calibration_primitive"]["status"] == "uncalibrated"
        assert "noul" in block["calibration_primitive"]["reason"]
        assert block["calibration_primitive"]["temperature"] == 1.0


def test_health_reports_per_primitive_calibration(tmp_path):
    art = _head_fitted_for(tmp_path, ("noul",))
    cfg = backend.load_api_config(_write_config(tmp_path, h1_path=art, tokenizer_revision="fixture-tok-rev"))
    be = backend.DecisionBackend(cfg, provider=_FakeProvider(_captured_from()))
    doc = be.health()
    h1 = [a for a in doc["arms"] if a["arm"] == "H1"][0]
    assert h1["calibration_overall"] == "calibrated"
    assert h1["calibration_per_primitive"]["noul"]["status"] == "calibrated"
    assert h1["calibration_per_primitive"]["choice"]["status"] == "uncalibrated"
    assert h1["calibrated_primitives"] == ["noul"]
    assert doc["calibration_overall"] == "calibrated"
    assert doc["calibration_per_primitive"]["score"]["calibrated"] is False


def test_health_surfaces_a_non_fatal_capture_error_marker(tmp_path):
    class _MarkerProvider:
        label = backend.FIXTURE_ONLY_LABEL

        def acquire(self, request, *, deadline_s, seed=None):  # pragma: no cover - not called
            raise AssertionError("not used")

        def diagnostics(self):
            return {"error_marker_seen": True, "error_marker": {"kind": "row_export_failed", "fatal": False}}

    cfg = backend.load_api_config(_write_config(tmp_path))
    be = backend.DecisionBackend(cfg, provider=_MarkerProvider())
    capture = be.health()["capture"]
    assert capture["error_marker_seen"] is True
    assert capture["error_marker"]["kind"] == "row_export_failed"
