"""S6 tests: real-HTTP fixture tests for the typed gateway (`qfn_decision.api`).

Every test in this file speaks real HTTP over a loopback socket to a gateway running in a
background thread; the gateway in turn talks to a **fixture** engine (also a real localhost
HTTP server, started here, labelled ``fixture_only``). No inference service is started,
stopped or deployed, no GPU or model is touched, and no live engine exists in this file.
Live capture/parity with the real vLLM engine stays NOT YET RUN (see reports/s6/RESULT.md).
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import pathlib
import socket
import threading
import time

import numpy as np
import pytest

from qfn_decision import api, backend, features, head as head_mod, protocol, templates

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ======================================================================================
# scaffolding
# ======================================================================================

def _config_file(tmp_path, engine_server, **over):
    cfg = json.loads((ROOT / "configs" / "decision_api.json").read_text(encoding="utf-8"))
    cfg["engine"]["base_url"] = engine_server.base_url
    cfg["engine"]["model"] = engine_server.model
    cfg["engine"]["health_ping"] = False
    cfg["capture"]["dir"] = str(engine_server.capture_dir)
    cfg["capture"]["dir_allowlist"] = [str(engine_server.capture_dir)]
    cfg["provider"] = "engine"
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    path = pathlib.Path(tmp_path) / "decision_api.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


@contextlib.contextmanager
def gateway(tmp_path, engine_fault=None, **cfg_over):
    """Fixture engine + fixture-only gateway; both are stopped and joined in `finally`.

    The gateway always binds an ephemeral loopback port (port=0) so a stray server from another
    test can never steal a connection.
    """
    eng = backend.FixtureEngineServer(
        model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture", fault=engine_fault
    )
    eng.start()
    logs = []
    srv = None
    try:
        cfg = backend.load_api_config(_config_file(tmp_path, eng, **cfg_over))
        srv = api.GatewayServer(cfg, port=0, log_sink=logs.append)
        srv.start()
        yield srv, eng, logs
    finally:
        if srv is not None:
            srv.stop()
        eng.stop()


def _post(port, path, body, headers=None, timeout=15.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, dict((k.lower(), v) for k, v in resp.getheaders()), data, conn
    except BaseException:
        conn.close()
        raise


def _decide(port, payload, **kw):
    return _post(port, "/v1/decide", json.dumps(payload).encode("utf-8"), **kw)


def _raw_post(port, path, raw, headers, timeout=15.0):
    """Raw socket POST so we can send a chunked body or omit Content-Length."""
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    request = f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers) + "\r\n"
    s.sendall(request.encode("ascii") + raw)
    buf = b""
    try:
        while b"\r\n\r\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    return buf


def _noul(**_):
    return {"kind": "noul", "state": "the sky is blue", "question": "is the sky blue?", "thinking": "off"}


def _choice(n=3):
    return {
        "kind": "choice",
        "state": "you are asked to pick",
        "question": "which one?",
        "options": [{"key": f"k{i}", "description": f"option {i}"} for i in range(n)],
        "thinking": "off",
    }


def _score():
    return {
        "kind": "score",
        "state": "a claim was made",
        "question": "how reliable is it?",
        "score_rubrics": [f"rubric {i}" for i in range(6)],
        "thinking": "off",
    }


def _h1_artifact(tmp_path, tokenizer_revision="fixture-tok-rev"):
    prov = head_mod.HeadProvenance(
        feature_base="fixture/effective-lm-head-rows",
        feature_width=head_mod.FEATURE_WIDTH,
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision=tokenizer_revision,
        rows=tuple(range(1000, 1024)),
        row_scaling=1.0,
        rows_sha256="0" * 64,
        init_source="fixture/effective-lm-head-rows",
    )
    art = pathlib.Path(tmp_path) / "h1.npz"
    head_mod.DecisionHead.random_control(seed=11, provenance=prov, scale=0.05).save(art)
    return str(art)


# ======================================================================================
# /health
# ======================================================================================

def test_health_reports_readiness_without_claiming_a_live_engine(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        status, headers, body, conn = _post(srv.port, "/health", b"", timeout=10)
        conn.close()
        assert status == 200
        doc = json.loads(body)
        assert doc["schema"] == "qfn.decide.health/1"
        assert doc["status"] in ("ready", "degraded")
        assert doc["readiness"]["live_engine_validated"] is False
        assert doc["readiness"]["engine_ping"] == "not_checked"
        assert doc["provider"]["label"] == "engine"
        assert doc["engine"]["base_url"] == eng.base_url
        assert doc["engine"]["model"] == "fixture-model"
        assert doc["protocol"]["num_slots"] == 24
        assert doc["protocol"]["kinds"] == ["noul", "choice", "score"]
        assert doc["protocol"]["typed_output_tokens"] == 0
        arms = {a["arm"]: a for a in doc["arms"]}
        assert arms["B0"]["available"] is True
        assert arms["H1"]["available"] is False
        assert "artifact" in arms["H1"]["reason"]
        assert doc["template"] == {"template_id": templates.TEMPLATE_AUTOTRUST_RAW, "version": "bare-v1", "frozen": True}
        assert "Live-validated scope" in doc["readiness"]["note"]


def test_health_with_engine_ping_configured(tmp_path):
    with gateway(tmp_path, engine={"health_ping": True}) as (srv, eng, logs):
        status, _, body, conn = _post(srv.port, "/health", b"", timeout=10)
        conn.close()
        doc = json.loads(body)
        assert status == 200
        assert doc["readiness"]["engine_ping"] == "ok"
        assert doc["readiness"]["live_engine_validated"] is False  # a ping is not a validated decision path

# ======================================================================================
# /v1/decide happy paths
# ======================================================================================

def test_decide_noul_returns_typed_probabilities_only(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        status, headers, body, conn = _decide(srv.port, _noul())
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        assert doc["schema"] == "qfn.decide/1"
        assert doc["kind"] == "noul"
        assert list(doc["probabilities"].keys()) == ["false", "true"]
        assert abs(sum(doc["probabilities"].values()) - 1.0) < 1e-6
        assert doc["selected"] in ("false", "true")
        assert doc["p_true"] == doc["probabilities"]["true"]
        assert doc["output_tokens"] == 0 and doc["typed_output_tokens"] == 0
        assert doc["internal_generated_tokens"] == 1
        assert doc["uses_s2"] is False
        assert doc["transport"] == backend.TRANSPORT_ONE_DISCARDED_TOKEN
        assert doc["jev_confidence"] is None
        assert doc["confidence_status"] == protocol.CONFIDENCE_STATUS_ABSENT
        assert doc["raw_max_probability"] == max(doc["probabilities"].values())
        assert doc["arm"]["arm"] == "B0" and doc["arm"]["lane"] == "raw_captured_logits"
        assert doc["template"]["template_id"] == templates.TEMPLATE_AUTOTRUST_RAW
        assert doc["tokenizer"]["prompt_token_sha256"] and doc["tokenizer"]["add_special_tokens"] is False
        assert doc["engine"]["engine_request_id"].startswith("cmpl-qfn-")
        assert doc["engine"]["logits_captured"] is True
        assert doc["usage"]["prompt_tokens"] == doc["tokenizer"]["prompt_tokens_count"]
        assert doc["usage"]["completion_tokens"] == 1
        assert doc["cache"]["cache_evidence"] == "unverified"
        assert doc["cache"]["source_field"] == "cache_hit_tokens"
        assert doc["provider"]["label"] == "engine"
        # the prompt text and any generated text never come back
        raw = body.decode("utf-8")
        assert "the sky is blue" not in raw and "is the sky blue?" not in raw


def test_decide_choice_preserves_caller_key_order_and_masks_inactive_slots(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        status, _, body, conn = _decide(srv.port, _choice(4))
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        assert list(doc["probabilities"].keys()) == ["k0", "k1", "k2", "k3"]
        raw = body.decode("utf-8")
        assert raw.index('"k0"') < raw.index('"k1"') < raw.index('"k2"') < raw.index('"k3"')
        assert doc["selected"] in ("k0", "k1", "k2", "k3")
        assert abs(sum(doc["probabilities"].values()) - 1.0) < 1e-6
        assert doc["slot_probabilities"] is not None
        assert [i for i, v in enumerate(doc["slot_probabilities"]) if v != 0.0] == [8, 9, 10, 11]


def test_decide_choice_row_order_is_not_semantic_and_16_options_still_work(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        payload = _choice(16)
        status, _, body, conn = _decide(srv.port, payload)
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        assert len(doc["probabilities"]) == 16
        assert list(doc["probabilities"].keys()) == [f"k{i}" for i in range(16)]


def test_decide_score_returns_expected_level_in_range(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        status, _, body, conn = _decide(srv.port, _score())
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        assert list(doc["probabilities"].keys()) == ["0", "1", "2", "3", "4", "5"]
        assert len(doc["level_probabilities"]) == 6
        assert 0.0 <= doc["expected_level"] <= 5.0
        assert abs(sum(doc["level_probabilities"]) - 1.0) < 1e-6
        assert [i for i, v in enumerate(doc["slot_probabilities"]) if v != 0.0] == [2, 3, 4, 5, 6, 7]


def test_decide_rubrics_alias_is_accepted_and_conflicts_are_refused(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        payload = _score()
        payload["rubrics"] = payload.pop("score_rubrics")
        status, _, body, conn = _decide(srv.port, payload)
        conn.close()
        assert status == 200, body
        both = _score()
        both["rubrics"] = list(both["score_rubrics"])
        status, _, body, conn = _decide(srv.port, both)
        conn.close()
        assert status == 400 and json.loads(body)["error"]["code"] == "ambiguous_field"


def test_decide_h1_arm_uses_the_artifact_and_reports_calibration_status(tmp_path):
    art = _h1_artifact(tmp_path)
    with gateway(
        tmp_path,
        head={"default_arm": "B0", "h1": {"artifact_path": art, "verify_checksum": True}},
        tokenizer={"revision": "fixture-tok-rev"},
    ) as (srv, eng, logs):
        payload = _choice(3)
        payload["arm"] = "H1"
        status, _, body, conn = _decide(srv.port, payload)
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        assert doc["arm"]["arm"] == "H1"
        assert doc["arm"]["calibration_status"] == "uncalibrated"
        assert doc["arm"]["trained"] is False
        assert doc["arm"]["head_artifact"]["data_sha256"]
        assert doc["arm"]["provenance"]["template_version"] == "bare-v1"


def test_decide_h1_is_refused_when_no_artifact_is_configured(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        payload = _noul() | {"arm": "H1"}
        status, _, body, conn = _decide(srv.port, payload)
        conn.close()
        assert status == 422
        assert json.loads(body)["error"]["code"] == "arm_unavailable"
        # no silent fallback to B0
        assert "probabilities" not in json.loads(body)


def test_decide_fixture_provider_mode_is_explicitly_labelled(tmp_path):
    eng = backend.FixtureEngineServer(model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture")
    eng.start()
    srv = None
    try:
        cfg = backend.load_api_config(_config_file(tmp_path, eng, provider="fixture"))
        srv = api.GatewayServer(cfg, port=0)
        srv.start()
        status, _, body, conn = _decide(srv.port, _choice(2))
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        assert doc["provider"]["label"] == "fixture_only"
        assert doc["provider"]["fixture"] is True
    finally:
        if srv is not None:
            srv.stop()
        eng.stop()


def test_decide_deadline_ms_is_honoured_as_a_bounded_deadline(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        payload = _noul() | {"deadline_ms": 5000}
        status, _, body, conn = _decide(srv.port, payload)
        conn.close()
        assert status == 200
        doc = json.loads(body)
        assert doc["timing"]["total_ms"] < 5000


# ======================================================================================
# validation / bounded parsing
# ======================================================================================

@pytest.mark.parametrize(
    "payload,status,code",
    [
        (_noul() | {"nope": 1}, 400, "unknown_field"),
        (_noul() | {"thinking": "on"}, 422, "thinking_on_unsupported"),
        (_noul() | {"thinking": "maybe"}, 422, "thinking_unsupported"),
        (_noul() | {"arm": "H9"}, 422, "arm_unsupported"),
        (_noul() | {"seed": "abc"}, 400, "bad_field_type"),
        (_noul() | {"seed": True}, 400, "bad_field_type"),
        (_noul() | {"deadline_ms": -5}, 400, "bad_field_value"),
        (_noul() | {"kind": "classify"}, 422, "protocol_error"),
        (_noul() | {"state": ""}, 422, "protocol_error"),
        (_noul() | {"state": "x" * 40000}, 422, "protocol_error"),
        (_noul() | {"question": ["a", "b"]}, 422, "protocol_error"),
        (_noul() | {"options": [{"key": "a", "description": "b"}]}, 422, "protocol_error"),
        ({"kind": "choice", "state": "s", "question": "q", "options": [{"key": "a", "description": "x"}]}, 422, "protocol_error"),
        ({"kind": "choice", "state": "s", "question": "q", "options": [{"key": "a", "description": "x"}] * 17}, 422, "protocol_error"),
        ({"kind": "choice", "state": "s", "question": "q", "options": [{"key": "a", "description": "x"}, {"key": "a", "description": "y"}]}, 422, "protocol_error"),
        ({"kind": "choice", "state": "s", "question": "q", "options": [{"key": "a", "description": "x"}, {"key": "", "description": "y"}]}, 422, "protocol_error"),
        ({"kind": "choice", "state": "s", "question": "q", "options": [True, False]}, 422, "protocol_error"),
        ({"kind": "choice", "state": "s", "question": "q", "options": [{"key": "a", "description": "x", "extra": 1}, {"key": "b", "description": "y"}]}, 400, "unknown_field"),
        ({"kind": "score", "state": "s", "question": "q", "score_rubrics": ["a", "b"]}, 422, "protocol_error"),
        ({"kind": "score", "state": "s", "question": "q", "score_rubrics": ["a"] * 6, "options": [{"key": "k", "description": "d"}, {"key": "j", "description": "d"}]}, 422, "protocol_error"),
        ({"kind": "noul", "state": "s", "question": "q", "score_rubrics": ["a"] * 6}, 422, "protocol_error"),
        ([1, 2, 3], 400, "not_an_object"),
        ("a string", 400, "not_an_object"),
    ],
)
def test_decide_validation_matrix(tmp_path, payload, status, code):
    with gateway(tmp_path) as (srv, eng, logs):
        got_status, _, body, conn = _decide(srv.port, payload)
        conn.close()
        assert got_status == status, body
        doc = json.loads(body)
        assert doc["error"]["code"] == code
        assert "probabilities" not in doc and "selected" not in doc
        assert isinstance(doc["error"]["message"], str) and doc["error"]["message"]


def test_decide_rejects_malformed_json_duplicates_and_nonfinite_literals(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        cases = [
            (b"{not json", "invalid_json"),
            (b'{"kind":"noul","kind":"noul","state":"s","question":"q"}', "duplicate_key"),
            (b'{"kind":"noul","state":NaN,"question":"q"}', "nonfinite_number"),
            (b'{"kind":"noul","state":Infinity,"question":"q"}', "nonfinite_number"),
            (b"", "empty_body"),
            (b"[1,2,3]", "not_an_object"),
        ]
        for raw, code in cases:
            status, _, body, conn = _post(srv.port, "/v1/decide", raw)
            conn.close()
            assert status == 400, (raw, body)
            assert json.loads(body)["error"]["code"] == code, raw


def test_decide_enforces_the_body_limit(tmp_path):
    with gateway(tmp_path, limits={"max_body_bytes": 2048}) as (srv, eng, logs):
        payload = _noul() | {"state": "x" * 8192}
        status, _, body, conn = _decide(srv.port, payload)
        conn.close()
        assert status == 413
        assert json.loads(body)["error"]["code"] == "body_too_large"


def test_decide_requires_content_length_and_refuses_chunked_bodies(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        raw = json.dumps(_noul()).encode("utf-8")
        # no Content-Length at all
        buf = _raw_post(srv.port, "/v1/decide", raw, [("Content-Type", "application/json")])
        assert b" 411 " in buf.split(b"\r\n")[0], buf[:200]
        # chunked transfer-encoding is unsupported: the gateway must not guess a length
        chunked = b"4\r\n{\"ki\r\nc\r\nnd\":\"noul\"}\r\n0\r\n\r\n"
        buf = _raw_post(
            srv.port,
            "/v1/decide",
            chunked,
            [("Content-Type", "application/json"), ("Transfer-Encoding", "chunked")],
        )
        assert b" 400 " in buf.split(b"\r\n")[0], buf[:200]
        # a bogus Content-Length is a bad header, not a crash
        buf = _raw_post(srv.port, "/v1/decide", raw, [("Content-Length", "notanumber")])
        assert b" 400 " in buf.split(b"\r\n")[0], buf[:200]


def test_unknown_routes_and_methods(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        status, _, body, conn = _post(srv.port, "/v1/nope", b"{}", timeout=10)
        conn.close()
        assert status == 404 and json.loads(body)["error"]["code"] == "not_found"
        status, _, body, conn = _post(srv.port, "/v1/decide/extra", b"{}", timeout=10)
        conn.close()
        assert status == 404
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        conn.request("GET", "/v1/decide")
        resp = conn.getresponse()
        assert resp.status == 405
        resp.read()
        conn.close()


# ======================================================================================
# bounded concurrency and backend error mapping
# ======================================================================================

class _BlockingProvider:
    label = "fixture_only"

    def __init__(self, release, captured):
        self.release = release
        self.captured = captured
        self.entered = threading.Event()

    def acquire(self, request, *, deadline_s, seed=None):
        self.entered.set()
        assert self.release.wait(timeout=10)
        return self.captured


class _RaisingProvider:
    label = "fixture_only"

    def __init__(self, exc):
        self.exc = exc

    def acquire(self, request, *, deadline_s, seed=None):
        raise self.exc


def _captured(engine_id="cmpl-qfn-deadbeef-0", prompt_len=4):
    tok = features.token_sequence_sha256(list(range(prompt_len)))
    vec = np.linspace(-2.0, 2.0, 2560, dtype=np.float32)
    bits = features.fp32_to_bf16_bits(vec)
    wide = features.bf16_bits_to_fp32(bits)
    rec = features.FeatureRecord(
        example_id="ex", engine_request_id=engine_id, prompt_token_sha256=tok,
        prompt_tokens_count=prompt_len, prompt_len=prompt_len, terminal_position=prompt_len - 1,
        cache_hit_tokens=0, num_scheduled_tokens=prompt_len, query_start=0, query_end=prompt_len,
        logits_row_index=0, batch_index=0, internal_index=0, vector_bf16=bits, vector_fp32=wide,
        verbalizer_ids=list(range(24)), verbalizer_logits_fp32=np.arange(24, dtype=np.float32),
        runtime_argmax_token_id=0, runtime_max_logit_fp32=0.0, head_scale=1.0, head_soft_cap=None,
        head_org_vocab_size=248320, head_dtype="bfloat16", lm_head_tp_size=2, tp_rank=0, world_size=2,
        cudagraph_mode="FULL_DECODE_ONLY", kernel_mode="prefill", runner_source_sha256="f" * 64,
        capture_wall_ns=time.time_ns(), logits_captured=True, logits_absent_reason=None,
        vector_source="fixture", helper_version="qfn-capture-runtime/1",
    )
    return backend.CapturedFeature(
        record=rec, vector_fp32=wide, slot_logits_fp32=rec.verbalizer_logits_fp32,
        caller_request_id="qfn-caller", engine_request_id=engine_id, engine_response_id=engine_id,
        usage={"prompt_tokens": prompt_len, "completion_tokens": 1, "total_tokens": prompt_len + 1},
        timing_ms={"tokenize": 0.1, "request": 0.1, "capture_wait": 0.1, "readout": 0.1, "total": 0.4},
        transport=backend.TRANSPORT_ONE_DISCARDED_TOKEN, provider_label="fixture_only",
        cache_raw={"cache_hit_tokens": 0, "num_scheduled_tokens": prompt_len},
        caller_prompt_token_sha256=tok,
    )


def test_inflight_bound_returns_429_instead_of_growing_workers(tmp_path):
    release = threading.Event()
    eng = backend.FixtureEngineServer(model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture")
    eng.start()
    cfg = backend.load_api_config(_config_file(tmp_path, eng, concurrency={"max_inflight": 1}))
    srv = api.GatewayServer(cfg, port=0, backend_instance=backend.DecisionBackend(cfg, provider=_BlockingProvider(release, _captured())))
    srv.start()
    results = {}
    try:
        def first():
            results["first"] = _decide(srv.port, _noul())
            results["first"][3].close()

        t = threading.Thread(target=first)
        t.start()
        time.sleep(0.3)  # the first request is now parked inside the provider
        second = _decide(srv.port, _noul())
        assert second[0] == 429, second[2]
        assert json.loads(second[2])["error"]["code"] == "too_many_inflight"
        second[3].close()
        release.set()
        t.join(10)
        assert results["first"][0] == 200, results["first"][2]
        # the slot is released again
        third = _decide(srv.port, _noul())
        assert third[0] == 200
        third[3].close()
    finally:
        release.set()
        srv.stop()
        eng.stop()


@pytest.mark.parametrize(
    "exc,status,code",
    [
        (backend.TransportError("engine down"), 502, "backend_error"),
        (backend.CaptureError("bad record"), 502, "backend_error"),
        (backend.DeadlineExceeded("no record in time"), 504, "deadline_exceeded"),
        (backend.BackendUnavailable("engine not configured"), 503, "backend_unavailable"),
    ],
)
def test_backend_errors_map_to_http_status(tmp_path, exc, status, code):
    eng = backend.FixtureEngineServer(model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture")
    eng.start()
    srv = None
    try:
        cfg = backend.load_api_config(_config_file(tmp_path, eng))
        srv = api.GatewayServer(cfg, port=0, backend_instance=backend.DecisionBackend(cfg, provider=_RaisingProvider(exc)))
        srv.start()
        got, _, body, conn = _decide(srv.port, _noul())
        conn.close()
        assert got == status, body
        doc = json.loads(body)
        assert doc["error"]["code"] == code
        assert "probabilities" not in doc
    finally:
        if srv is not None:
            srv.stop()
        eng.stop()


def test_no_partial_probabilities_on_backend_failure(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        # a record whose active logits are all NaN must never produce a 200 with NaN mass
        eng.fault = "nan_logits"
        status, _, body, conn = _decide(srv.port, _choice(3))
        conn.close()
        assert status == 502, body
        doc = json.loads(body)
        assert doc["error"]["code"] == "backend_error"
        assert "probabilities" not in doc and "selected" not in doc


def test_empty_generation_with_a_valid_feature_is_still_a_success(tmp_path):
    with gateway(tmp_path, engine_fault=None) as (srv, eng, logs):
        eng.empty_completion_text = True  # EOS sampled internally, empty visible text
        status, _, body, conn = _decide(srv.port, _noul())
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        assert doc["usage"]["completion_text_empty"] is True
        assert doc["usage"]["completion_text_returned"] is False
        assert doc["usage"]["completion_tokens"] == 1  # the real usage count, not a guessed 0
        assert doc["internal_generated_tokens"] == 1


# ======================================================================================
# ordinary generation proxying
# ======================================================================================

def test_chat_completions_proxy_preserves_the_request_body_verbatim(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        body = {
            "model": "fixture-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "thinking": {"type": "enabled", "budget_tokens": 128},
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": "auto",
            "stream": False,
            "extra_unknown_engine_field": [1, 2, {"a": [True, None]}],
        }
        raw = json.dumps(body).encode("utf-8")
        status, headers, data, conn = _post(srv.port, "/v1/chat/completions", raw)
        conn.close()
        assert status == 200, data
        doc = json.loads(data)
        assert doc["received"] == body  # the engine saw the exact same JSON body
        assert headers["content-length"] == str(len(data))
        assert headers.get("transfer-encoding") != "chunked"


def test_completions_proxy_passes_engine_errors_through_unchanged(tmp_path):
    eng = backend.FixtureEngineServer(model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture")
    eng.chat_error = (400, {"error": {"message": "thinking not supported", "type": "BadRequestError"}})
    eng.start()
    srv = None
    try:
        cfg = backend.load_api_config(_config_file(tmp_path, eng))
        srv = api.GatewayServer(cfg, port=0)
        srv.start()
        raw = json.dumps({"model": "fixture-model", "messages": [], "thinking": "on"}).encode("utf-8")
        status, headers, data, conn = _post(srv.port, "/v1/chat/completions", raw)
        conn.close()
        assert status == 400
        assert json.loads(data) == {"error": {"message": "thinking not supported", "type": "BadRequestError"}}
        assert headers["content-type"].startswith("application/json")
    finally:
        if srv is not None:
            srv.stop()
        eng.stop()


def test_streaming_sse_chunks_are_forwarded_without_repacking(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        raw = json.dumps({"model": "fixture-model", "messages": [{"role": "user", "content": "hi"}], "stream": True}).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=15)
        conn.request("POST", "/v1/chat/completions", body=raw, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 200
        hdr = dict((k.lower(), v) for k, v in resp.getheaders())
        assert hdr.get("content-type", "").startswith("text/event-stream")
        assert hdr.get("transfer-encoding", "").lower() == "chunked"
        assert "content-length" not in hdr
        data = resp.read()
        conn.close()
        got = data.decode("utf-8")
        expect = "".join(eng.stream_events)
        assert got == expect
        assert got.count("data: ") == len(eng.stream_events)
        assert "[DONE]" in got


def test_client_cancellation_closes_the_upstream_engine_connection(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        eng.stream_events = [f"data: {{\"i\":{i}}}\n\n" for i in range(60)]
        eng.stream_delay_s = 0.05
        raw = json.dumps({"model": "fixture-model", "messages": [], "stream": True}).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=15)
        conn.request("POST", "/v1/chat/completions", body=raw, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read(10)  # read a little, then hang up on the gateway
        conn.close()
        deadline = time.time() + 8.0
        while time.time() < deadline and not eng.client_disconnects:
            time.sleep(0.05)
        assert eng.client_disconnects, "the gateway did not close the upstream stream after the client hung up"
        assert not eng.stream_completed_send, "the gateway kept streaming to a dead client"
        # the gateway survives and stays healthy
        status, _, body, c2 = _post(srv.port, "/health", b"", timeout=10)
        c2.close()
        assert status == 200


def test_proxy_hop_headers_are_not_forwarded_and_engine_owns_its_caps(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        raw = json.dumps({"model": "fixture-model", "messages": []}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive, X-Hop",
            "X-Hop": "should-not-arrive",
            "Keep-Alive": "timeout=5",
            "Proxy-Connection": "keep-alive",
        }
        status, _, data, conn = _post(srv.port, "/v1/chat/completions", raw, headers=headers)
        conn.close()
        assert status == 200
        seen = eng.last_chat_headers
        for hop in ("connection", "transfer-encoding", "keep-alive", "x-hop", "proxy-connection"):
            assert hop not in seen, seen
        assert seen.get("content-length") == str(len(raw))


def test_configured_env_header_is_forwarded_and_never_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("QFN_TEST_ENGINE_AUTH", "Bearer super-secret-token")
    with gateway(
        tmp_path,
        engine={"forward_headers_from_env": {"authorization": "QFN_TEST_ENGINE_AUTH"}},
    ) as (srv, eng, logs):
        raw = json.dumps({"model": "fixture-model", "messages": []}).encode("utf-8")
        status, _, data, conn = _post(srv.port, "/v1/chat/completions", raw)
        conn.close()
        assert status == 200
        assert eng.last_chat_headers.get("authorization") == "Bearer super-secret-token"
        joined = "\n".join(logs)
        assert "super-secret-token" not in joined
        assert "authorization" in joined.lower() or "header" in joined.lower()


def test_logs_redact_prompt_text_by_default(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        _decide(srv.port, _choice(3))[0]
        joined = "\n".join(logs)
        assert "you are asked to pick" not in joined
        assert "option 0" not in joined
        assert "which one?" not in joined
        assert "/v1/decide" in joined and "B0" in joined


def test_gateway_binds_loopback_only_by_default(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        assert srv.host == "127.0.0.1"
        h = srv.health()
        assert h["bind"]["lan_exposure_warning"]
        assert "unauthenticated" in h["bind"]["lan_exposure_warning"].lower()


# ======================================================================================
# CLI surface
# ======================================================================================

def test_cli_prints_config_and_curl_and_checks_the_configuration(tmp_path, capsys):
    eng = backend.FixtureEngineServer(model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture")
    eng.start()
    try:
        cfg_path = _config_file(tmp_path, eng, provider="fixture")
        rc = api.main(["--config", str(cfg_path), "--print-config"])
        out = capsys.readouterr().out
        assert rc == 0 and "127.0.0.1" in out and "8890" in out
        rc = api.main(["--config", str(cfg_path), "--print-curl"])
        out = capsys.readouterr().out
        assert rc == 0 and "curl" in out and "/v1/decide" in out and "8890" in out
        rc = api.main(["--config", str(cfg_path), "--check"])
        out = capsys.readouterr().out
        assert rc == 0 and "ok" in out.lower()
        assert "Live-validated scope" in out
    finally:
        eng.stop()


# ======================================================================================
# live engine contract over HTTP: id forms, per-primitive calibration, error marker
# ======================================================================================

def _h1_fitted_for(tmp_path, primitives=("noul",), temperature=2.5):
    """Fixture head artifact whose calibration covers only ``primitives`` (fixture-only data)."""
    prov = head_mod.HeadProvenance(
        feature_base="fixture/effective-lm-head-rows",
        feature_width=head_mod.FEATURE_WIDTH,
        template_id=templates.TEMPLATE_AUTOTRUST_RAW,
        template_version="bare-v1",
        tokenizer_revision="fixture-tok-rev",
        rows=tuple(range(1000, 1024)),
        row_scaling=1.0,
        rows_sha256="0" * 64,
        init_source="fixture/effective-lm-head-rows",
    )
    art = pathlib.Path(tmp_path) / "h1_calibrated.npz"
    if art.exists():
        art.unlink()
        art.with_suffix(".json").unlink(missing_ok=True)
    base = head_mod.DecisionHead.random_control(seed=21, provenance=prov, scale=0.05)
    cal = head_mod.CalibrationState(
        temperatures={k: (temperature if k in primitives else 1.0) for k in head_mod.PRIMITIVE_KEYS},
        fitted=True,
        method="fixture_isotonic",
        source="fixture_calibration_rows",
        scope="fixture_scope",
        fitted_primitives=tuple(primitives),
        parameter_fingerprint=base.parameter_fingerprint(),
        provenance_fingerprint=base.provenance_fingerprint(),
        data_fingerprint="b" * 64,
        per_primitive={
            k: {"n_rows": 32, "scope": "fixture_scope", "source": "fixture_rows", "calibrated": True}
            for k in primitives
        },
    )
    base.with_calibration(cal).save(art)
    return str(art)


def test_decide_reports_live_ids_and_per_primitive_calibration(tmp_path):
    art = _h1_fitted_for(tmp_path, ("noul",))
    with gateway(
        tmp_path,
        head={"default_arm": "B0", "h1": {"artifact_path": art, "verify_checksum": True}},
        tokenizer={"revision": "fixture-tok-rev"},
    ) as (srv, eng, logs):
        payload = _choice(2) | {"arm": "H1"}
        status, _, body, conn = _decide(srv.port, payload)
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        # live id forms: response id has no index suffix, record id carries index + 8-hex suffix
        assert doc["engine"]["engine_response_id"] == f"cmpl-{doc['engine']['caller_request_id']}"
        assert doc["engine"]["engine_request_id"].startswith(doc["engine"]["engine_response_id"] + "-0-")
        # only noul is fitted, so a choice readout must not claim the overall "calibrated"
        assert doc["arm"]["calibration_status_overall"] == "calibrated"
        assert doc["arm"]["calibration_status"] == "uncalibrated"
        assert doc["capture"]["error_marker_seen"] is False

        status, _, body, conn = _decide(srv.port, _noul() | {"arm": "H1"})
        conn.close()
        assert status == 200, body
        assert json.loads(body)["arm"]["calibration_status"] == "calibrated"

        status, _, body, conn = _post(srv.port, "/health", b"", timeout=10)
        conn.close()
        health = json.loads(body)
        entry = [a for a in health["arms"] if a["arm"] == "H1"][0]
        assert entry["calibration_overall"] == "calibrated"
        assert entry["calibration_per_primitive"]["noul"]["status"] == "calibrated"
        assert entry["calibration_per_primitive"]["score"]["status"] == "uncalibrated"
        assert health["capture"]["error_marker_seen"] is False


def test_non_fatal_capture_error_marker_is_disclosed_but_the_request_succeeds(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        eng.error_marker = {
            "helper_version": "qfn-capture-runtime/1",
            "marker_prefix": "cmpl-qfn-",
            "last_error": {"kind": "row_export_failed", "error": "RuntimeError: fixture", "wall_ns": 1},
            "error_count": 1,
            "skips": {"chunk_not_final": 2},
            "records_written": 1,
            "stopped": False,
        }
        status, _, body, conn = _decide(srv.port, _noul())
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        assert doc["capture"]["error_marker_seen"] is True
        assert doc["capture"]["error_marker"]["fatal"] is False
        assert doc["capture"]["error_marker"]["kind"] == "row_export_failed"

        status, _, body, conn = _post(srv.port, "/health", b"", timeout=10)
        conn.close()
        health = json.loads(body)
        assert health["capture"]["error_marker_seen"] is True
        assert health["capture"]["error_marker"]["kind"] == "row_export_failed"


def test_decide_fails_when_the_marker_identifies_this_request_or_stops_the_helper(tmp_path):
    with gateway(tmp_path) as (srv, eng, logs):
        eng.error_marker = {
            "last_error": {"kind": "quota_exceeded", "error": "capture quota reached", "wall_ns": 1},
            "records_written": 4096,
            "stopped": True,
        }
        eng.fault = "missing"
        status, _, body, conn = _decide(srv.port, _noul())
        conn.close()
        assert status == 502, body
        assert json.loads(body)["error"]["code"] == "backend_error"


# ======================================================================================
# qwen_chat_letters-v1 through the real gateway (J3)
# ======================================================================================

LETTERS_TEMPLATE_ID = templates.TEMPLATE_QWEN_CHAT_LETTERS
LETTERS_TEMPLATE_VERSION = "letters-v1"
LETTERS_KWARGS = {"enable_thinking": False}
LETTERS_TEMPLATE_CFG = {
    "template": {"template_id": LETTERS_TEMPLATE_ID, "version": LETTERS_TEMPLATE_VERSION,
                 "chat_template_kwargs": dict(LETTERS_KWARGS)}
}
EXAMPLE_CONFIG = ROOT / "configs" / "decision_api.chat_letters.example.json"


def _letters_request(payload):
    """The protocol request the gateway must have rendered for this decide payload."""
    return protocol.DecisionRequest.create(
        kind=payload["kind"], state=payload["state"], question=payload["question"],
        options=[protocol.ChoiceOption(key=o["key"], description=o["description"]) for o in payload["options"]],
    )


def test_decide_choice_under_the_chat_letters_template(tmp_path):
    with gateway(tmp_path, **LETTERS_TEMPLATE_CFG) as (srv, eng, logs):
        payload = _choice(3)
        status, _, body, conn = _decide(srv.port, payload)
        conn.close()
        assert status == 200, body
        doc = json.loads(body)

        assert doc["kind"] == "choice"
        assert list(doc["probabilities"]) == ["k0", "k1", "k2"]
        assert abs(sum(doc["probabilities"].values()) - 1.0) < 1e-6
        assert doc["typed_output_tokens"] == 0
        assert doc["template"]["template_id"] == LETTERS_TEMPLATE_ID
        assert doc["template"]["version"] == LETTERS_TEMPLATE_VERSION
        assert doc["template"]["frozen"] is True
        assert doc["tokenizer"]["prompt_tokens_count"] > 0
        # only the caller's option slots (8..10) may carry mass: no protocol change
        slots = doc["slot_probabilities"]
        assert len(slots) == 24
        assert slots[:8] == [0.0] * 8 and slots[11:] == [0.0] * 13
        assert all(v > 0.0 for v in slots[8:11])

        # the engine was asked to tokenize the FROZEN messages, not a raw prompt
        tok = eng.tokenize_bodies[-1]
        assert "prompt" not in tok
        assert tok["add_generation_prompt"] is True
        assert tok["chat_template_kwargs"] == LETTERS_KWARGS
        assert tok["messages"] == templates.render_messages(_letters_request(payload), LETTERS_TEMPLATE_ID)
        assert tok["messages"][0]["role"] == "system"
        assert tok["messages"][0]["content"].startswith("You are a calibration engine.")
        assert tok["messages"][1]["content"].endswith("Answer with the letter of exactly one option, and nothing else:")
        # the completion still carries explicit ids
        assert all(isinstance(i, int) for i in eng.completion_bodies[-1]["prompt"])
        # no prompt text is ever logged
        assert all(payload["state"] not in line for line in logs)


@pytest.mark.parametrize("payload_fn,kind", [(_noul, "noul"), (_score, "score")])
def test_decide_refuses_noul_and_score_under_the_chat_letters_template(tmp_path, payload_fn, kind):
    """The letters surface is choice-only: 422 with a clear message, never a silent fallback."""
    with gateway(tmp_path, **LETTERS_TEMPLATE_CFG) as (srv, eng, logs):
        status, _, body, conn = _decide(srv.port, payload_fn())
        conn.close()
        assert status == 422, body
        error = json.loads(body)["error"]
        assert error["code"] == "template_kind_unsupported"
        assert kind in error["message"] and LETTERS_TEMPLATE_ID in error["message"]
        assert "choice" in error["message"]
        assert "probabilities" not in body.decode("utf-8")
        # the refusal happened before any engine traffic
        assert eng.tokenize_bodies == [] and eng.completion_bodies == []


def test_decide_h1_is_refused_under_the_chat_letters_template(tmp_path):
    """No silent B0: an H1 artifact fitted on the raw template cannot serve this template."""
    art = _h1_artifact(tmp_path)
    over = dict(LETTERS_TEMPLATE_CFG)
    over["head"] = {"h1": {"artifact_path": art}}
    over["tokenizer"] = {"revision": "fixture-tok-rev"}
    with gateway(tmp_path, **over) as (srv, eng, logs):
        status, _, body, conn = _post(srv.port, "/health", b"", timeout=10)
        conn.close()
        health = json.loads(body)
        h1 = {a["arm"]: a for a in health["arms"]}["H1"]
        assert h1["available"] is False
        assert "template_id mismatch" in h1["reason"]
        assert templates.TEMPLATE_AUTOTRUST_RAW in h1["reason"]
        assert health["template"]["template_id"] == LETTERS_TEMPLATE_ID

        status, _, body, conn = _decide(srv.port, dict(_choice(3), arm="H1"))
        conn.close()
        assert status == 422, body
        error = json.loads(body)["error"]
        assert error["code"] == "arm_unavailable"
        assert "template_id mismatch" in error["message"]
        assert "probabilities" not in body.decode("utf-8")
        assert eng.tokenize_bodies == [] and eng.completion_bodies == []


def test_chat_letters_example_config_is_a_copy_of_the_default_config_but_for_the_template():
    base = json.loads((ROOT / "configs" / "decision_api.json").read_text(encoding="utf-8"))
    example = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert set(example) == set(base)
    assert {k: v for k, v in example.items() if k != "template"} == \
        {k: v for k, v in base.items() if k != "template"}
    assert example["template"]["template_id"] == LETTERS_TEMPLATE_ID
    assert example["template"]["version"] == LETTERS_TEMPLATE_VERSION
    assert example["template"]["chat_template_kwargs"] == LETTERS_KWARGS
    assert example["template"]["require_version_match"] is True

    cfg = backend.load_api_config(EXAMPLE_CONFIG)
    assert cfg.template_id == LETTERS_TEMPLATE_ID
    assert cfg.template_version == LETTERS_TEMPLATE_VERSION
    assert cfg.chat_template_kwargs == LETTERS_KWARGS
    assert cfg.require_version_match is True
    # the shipped default config is untouched by this build
    default = backend.load_api_config(ROOT / "configs" / "decision_api.json")
    assert default.template_id == templates.TEMPLATE_AUTOTRUST_RAW
    assert default.chat_template_kwargs == {}


# Measured on the dual-DGX-Spark GB10 head node (2026-10-26): the first engine request
# after a fresh weight load pays ~46 s of compile/warmup before it can answer anything.
# A socket timeout below that budget turns the first decision after every boot into a
# spurious TransportError ("engine /tokenize request failed: TimeoutError") that then
# succeeds on retry. Keep the shipped defaults above the measured cold-start budget.
COLD_START_BUDGET_S = 46.0


def test_shipped_socket_timeout_covers_the_measured_cold_start_budget():
    for path in (ROOT / "configs" / "decision_api.json", EXAMPLE_CONFIG):
        raw = json.loads(path.read_text(encoding="utf-8"))
        engine_socket = raw["engine"]["socket_timeout_s"]
        flat_socket = raw["timeouts"]["socket_s"]
        request_s = raw["engine"]["request_timeout_s"]
        assert engine_socket >= COLD_START_BUDGET_S, (path.name, engine_socket)
        assert flat_socket >= COLD_START_BUDGET_S, (path.name, flat_socket)
        assert engine_socket < request_s, (path.name, engine_socket, request_s)
        assert backend.load_api_config(path).timeout_socket_s >= COLD_START_BUDGET_S


def test_decide_choice_under_the_example_chat_letters_config_end_to_end(tmp_path):
    """The example config file itself drives a real fixture-mode gateway."""
    eng = backend.FixtureEngineServer(model="fixture-model", capture_dir=pathlib.Path(tmp_path) / "capture")
    eng.start()
    srv = None
    try:
        cfg_json = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        cfg_json["engine"]["base_url"] = eng.base_url
        cfg_json["engine"]["model"] = eng.model
        cfg_json["capture"]["dir"] = str(eng.capture_dir)
        cfg_json["capture"]["dir_allowlist"] = [str(eng.capture_dir)]
        path = pathlib.Path(tmp_path) / "example.json"
        path.write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")
        srv = api.GatewayServer(backend.load_api_config(path), port=0)
        srv.start()
        status, _, body, conn = _decide(srv.port, _choice(2))
        conn.close()
        assert status == 200, body
        doc = json.loads(body)
        assert doc["template"]["template_id"] == LETTERS_TEMPLATE_ID
        assert list(doc["probabilities"]) == ["k0", "k1"]
        assert eng.tokenize_bodies[-1]["chat_template_kwargs"] == LETTERS_KWARGS
    finally:
        if srv is not None:
            srv.stop()
        eng.stop()
