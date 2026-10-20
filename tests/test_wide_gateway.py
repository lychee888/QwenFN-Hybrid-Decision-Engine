"""WIDE1 slice 7: the wide lane over real loopback HTTP (/v1/decide and /v1/systemone).

The gateway talks to the synthetic fixture engine over a loopback socket; the fixture writes a
capture record exactly like the engine helper does (base record + wide extension). No real engine,
GPU or inference is involved, and nothing here is a live-support claim.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import pathlib
import threading
import time

import numpy as np
import pytest

from qfn_decision import api, backend, features, protocol, wide

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "wide-label-catalog.json"
CATALOG = wide.WideCatalog.from_file(CATALOG_PATH)


def _wide_block(**over) -> dict:
    block = {
        "enabled": True,
        "catalog_path": str(CATALOG_PATH),
        "catalog_file_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        "catalog_digest": CATALOG.digest,
        "tokenizer_revision": CATALOG.tokenizer_revision,
        "template_digest": wide.WIDE_TEMPLATE_CONTENT_SHA256,
        "max_options": 256,
    }
    block.update(over)
    return block


def _config_file(tmp_path: pathlib.Path, engine_server, *, wide_block=None, **over) -> pathlib.Path:
    cfg = json.loads((ROOT / "configs" / "decision_api.json").read_text(encoding="utf-8"))
    cfg["engine"]["base_url"] = engine_server.base_url
    cfg["engine"]["model"] = engine_server.model
    cfg["engine"]["health_ping"] = False
    cfg["capture"]["dir"] = str(engine_server.capture_dir)
    cfg["capture"]["dir_allowlist"] = [str(engine_server.capture_dir)]
    cfg["provider"] = "engine"
    if wide_block is not None:
        cfg["wide_choice"] = wide_block
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    path = tmp_path / "decision_api.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


@contextlib.contextmanager
def gateway(tmp_path, *, wide_block="default", wide_logits=None, provider=None, engine_fault=None,
            **cfg_over):
    block = _wide_block() if wide_block == "default" else wide_block
    eng = backend.FixtureEngineServer(
        model="fixture-model",
        capture_dir=pathlib.Path(tmp_path) / "capture",
        wide_choice=(wide.WideChoiceConfig.from_dict(block) if block else None),
        wide_logits=wide_logits,
        fault=engine_fault,
    )
    eng.start()
    srv = None
    try:
        cfg = backend.load_api_config(_config_file(tmp_path, eng, wide_block=block, **cfg_over))
        kwargs = {}
        if provider is not None:
            kwargs["backend_instance"] = backend.DecisionBackend(cfg, provider=provider)
        srv = api.GatewayServer(cfg, port=0, log_sink=[].append, **kwargs)
        srv.start()
        yield srv, eng
    finally:
        if srv is not None:
            srv.stop()
        eng.stop()


def _post(port, path, body, timeout=30.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, raw
    finally:
        conn.close()


def _options(n: int, *, key_prefix: str = "opt") -> list[dict]:
    return [{"key": f"{key_prefix}{i}", "description": f"description {i}"} for i in range(n)]


def _decide(n: int, **over) -> dict:
    body = {"kind": "choice", "state": "state text", "question": "which option?",
            "options": _options(n)}
    body.update(over)
    return body


# ---------------------------------------------------------------- provider lane


@pytest.mark.parametrize("n", [17, 26, 32, 64, 128, 256])
def test_decide_wide_returns_an_exact_distribution_over_n_keys(tmp_path, n):
    with gateway(tmp_path) as (srv, eng):
        status, raw = _post(srv.port, "/v1/decide", _decide(n))
        assert status == 200, raw[:800]
        doc = json.loads(raw)
        assert doc["schema"] == "qfn.decide.wide/1"
        assert doc["adaptation"]["mode"] == "wide-labels"
        assert doc["adaptation"]["n_options"] == n
        assert doc["adaptation"]["native_choice_slots"] == 16
        assert doc["adaptation"]["slot_vector_used"] is False
        assert doc["adaptation"]["calibrated_confidence"] is False
        assert "slot_probabilities" not in doc
        probs = doc["probabilities"]
        assert list(probs) == [f"opt{i}" for i in range(n)]
        assert len(probs) == n
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert all(v > 0 for v in probs.values())
        assert doc["selected"] in probs
        assert doc["selected"] == max(probs, key=probs.get)
        assert doc["catalog"]["digest"] == CATALOG.digest
        assert doc["catalog"]["n_labels"] == 256
        assert doc["catalog"]["active_labels"] == n
        assert doc["capture"]["wide"]["catalog_digest"] == CATALOG.digest
        assert doc["capture"]["wide"]["template_digest"] == wide.WIDE_TEMPLATE_CONTENT_SHA256
        assert doc["capture"]["wide"]["request_uuid"] == doc["capture"]["wide_capture_uuid"]
        # the engine saw the wide prompt surface and the wide request id
        assert eng.tokenize_bodies[-1]["messages"][0]["content"] == wide.WIDE_CHAT_SYSTEM_PROMPT
        assert eng.completion_bodies[-1]["request_id"].startswith(f"qfn-wide-{n}-")


@pytest.mark.parametrize("n,winner,label", [(17, 16, "Q"), (26, 25, "Z"), (32, 26, "AA"),
                                            (128, 127, "EE"), (256, 255, "JU")])
def test_winning_labels_beyond_p_through_the_gateway(tmp_path, n, winner, label):
    logits = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    logits[winner] = 30.0
    with gateway(tmp_path, wide_logits=logits) as (srv, eng):
        status, raw = _post(srv.port, "/v1/decide", _decide(n))
        assert status == 200, raw[:800]
        doc = json.loads(raw)
        assert doc["selected"] == f"opt{winner}"
        assert doc["selected_label"] == label
        assert doc["probabilities"][f"opt{winner}"] > 0.99


def test_sixteen_options_still_use_the_legacy_schema_and_slots(tmp_path):
    with gateway(tmp_path) as (srv, eng):
        status, raw = _post(srv.port, "/v1/decide", _decide(16))
        assert status == 200, raw[:800]
        doc = json.loads(raw)
        assert doc["schema"] == "qfn.decide/1"
        assert len(doc["slot_probabilities"]) == 24
        assert "adaptation" not in doc
        assert eng.completion_bodies[-1]["request_id"].startswith("qfn-")


def test_257_options_and_bad_shapes_are_refused_before_any_engine_activity(tmp_path):
    with gateway(tmp_path) as (srv, eng):
        cases = [
            (_decide(257), 422),
            ({**_decide(17), "options": _options(17) * 1, "unexpected": 1}, 400),
            ({**_decide(17), "kind": "score"}, 422),
            ({**_decide(17), "thinking": "on"}, 422),
            ({**_decide(17), "arm": "H1"}, 422),
            ({**_decide(17), "arm": "L1O"}, 422),
            ({**_decide(17), "options": [{"key": "a", "description": "d"}] * 17}, 422),
            ({**_decide(17), "options": [{"key": f"k{i}", "description": "x\ny"} for i in range(17)]}, 422),
        ]
        for body, expected in cases:
            status, raw = _post(srv.port, "/v1/decide", body)
            assert status == expected, (body.keys(), status, raw[:400])
        assert eng.tokenize_bodies == []
        assert eng.completion_bodies == []


def test_wide_lane_disabled_by_default_keeps_the_legacy_refusal(tmp_path):
    with gateway(tmp_path, wide_block=None) as (srv, eng):
        status, raw = _post(srv.port, "/v1/decide", _decide(17))
        assert status == 422
        assert json.loads(raw)["error"]["code"] == "protocol_error"
        assert eng.completion_bodies == []


def test_a_missing_wide_extension_never_yields_probabilities(tmp_path):
    # fixture engine runs WITHOUT wide support: the record has no extension
    with gateway(tmp_path, wide_block="default") as (srv, eng):
        eng.wide_choice = None
        status, raw = _post(srv.port, "/v1/decide", _decide(32))
        assert status == 502
        assert json.loads(raw)["error"]["code"] == "backend_error"
        assert "probabilities" not in json.loads(raw)


def test_wide_request_does_not_accept_a_caller_supplied_request_id(tmp_path):
    with gateway(tmp_path) as (srv, eng):
        for field in ("request_id", "engine_request_id", "x-request-id", "id"):
            status, raw = _post(srv.port, "/v1/decide", {**_decide(17), field: "cmpl-qfn-wide-1-" + "a" * 32})
            assert status == 400, (field, status)
            assert json.loads(raw)["error"]["code"] == "unknown_field"
        assert eng.completion_bodies == []


class _BlockingWideProvider(backend.FixtureFeatureProvider):
    """Blocks inside acquire_wide so the gateway's global inflight bound can be observed."""

    def __init__(self, *args, entered: threading.Event, release: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = entered
        self.release = release

    def acquire_wide(self, request, catalog, *, deadline_s=None, seed=None):
        self.entered.set()
        assert self.release.wait(20.0)
        return super().acquire_wide(request, catalog, deadline_s=deadline_s, seed=seed)


def test_inflight_bound_is_shared_between_wide_and_legacy_requests(tmp_path):
    entered, release = threading.Event(), threading.Event()
    cap_dir = pathlib.Path(tmp_path) / "blockcapture"
    provider = _BlockingWideProvider(seed=12345, capture_dir=cap_dir, entered=entered, release=release)
    with gateway(tmp_path, provider=provider, concurrency={"max_inflight": 1}) as (srv, eng):
        results = {}

        def wide_call():
            results["wide"] = _post(srv.port, "/v1/decide", _decide(17))

        thread = threading.Thread(target=wide_call, daemon=True)
        thread.start()
        assert entered.wait(10.0)
        status, raw = _post(srv.port, "/v1/decide", _decide(3))
        assert status == 429
        assert json.loads(raw)["error"]["code"] == "too_many_inflight"
        release.set()
        thread.join(20.0)
        assert results["wide"][0] == 200


def test_wide_deadline_is_honoured(tmp_path):
    with gateway(tmp_path) as (srv, eng):
        eng.wide_logits = None
        # a deadline far below the capture poll interval: the fixture writes the record immediately,
        # so use a fault that suppresses the record to exercise the deadline path.
        eng.fault = "missing"
        status, raw = _post(srv.port, "/v1/decide", _decide(17, deadline_ms=200))
        assert status == 504
        assert json.loads(raw)["error"]["code"] == "deadline_exceeded"


def test_ordinary_chat_and_streaming_are_unaffected_by_the_wide_lane(tmp_path):
    with gateway(tmp_path) as (srv, eng):
        status, raw = _post(srv.port, "/v1/chat/completions",
                            {"model": "fixture-model", "messages": [{"role": "user", "content": "hi"}]})
        assert status == 200
        status, raw = _post(srv.port, "/v1/chat/completions",
                            {"model": "fixture-model", "messages": [{"role": "user", "content": "hi"}],
                             "stream": True})
        assert status == 200
        assert b"data: [DONE]" in raw


# ---------------------------------------------------------------- /v1/systemone


def _systemone(questions: dict, state="state text") -> dict:
    return {"model": "jev-latest", "state": state, "questions": questions}


def test_systemone_wide_choice_preserves_the_jev_answer_shape(tmp_path):
    n = 26
    with gateway(tmp_path) as (srv, eng):
        body = _systemone({"pick": {"type": "choice", "instructions": "choose",
                                    "criteria": {f"k{i}": f"c{i}" for i in range(n)}}})
        status, raw = _post(srv.port, "/v1/systemone", body)
        assert status == 200, raw[:800]
        doc = json.loads(raw)
        answer = doc["answers"]["pick"]
        assert answer["type"] == "choice"
        assert list(answer["probabilities"]) == [f"k{i}" for i in range(n)]
        assert answer["choice"] == max(answer["probabilities"], key=answer["probabilities"].get)
        # Jev's documented choice confidence formula, over the n active options
        p = [answer["probabilities"][f"k{i}"] for i in range(n)]
        expected = (max(p) - 1.0 / n) / (1.0 - 1.0 / n)
        assert abs(answer["confidence"] - expected) < 1e-12
        assert answer["wide"]["adaptation"] == "wide-labels"
        assert answer["wide"]["n_options"] == n
        assert answer["wide"]["native_choice_slots"] == 16
        assert answer["wide"]["catalog_digest"] == CATALOG.digest
        assert doc["model"].endswith("-b0")


def test_systemone_mixed_wide_small_and_noul_questions(tmp_path):
    with gateway(tmp_path) as (srv, eng):
        questions = {
            "wide": {"type": "choice", "instructions": "choose",
                     "criteria": {f"w{i}": None for i in range(17)}},
            "small": {"type": "choice", "instructions": "choose",
                      "criteria": {"yes": None, "no": None}},
            "flag": {"type": "noul", "instructions": "is it true?"},
        }
        status, raw = _post(srv.port, "/v1/systemone", _systemone(questions))
        assert status == 200, raw[:800]
        answers = json.loads(raw)["answers"]
        assert set(answers) == {"wide", "small", "flag"}
        assert len(answers["wide"]["probabilities"]) == 17
        assert len(answers["small"]["probabilities"]) == 2
        assert 0.0 <= answers["flag"]["noul"] <= 1.0
        assert "wide" not in answers["small"]
        assert "wide" not in answers["flag"]


def test_systemone_refuses_too_many_options_and_bad_fields(tmp_path):
    with gateway(tmp_path) as (srv, eng):
        body = _systemone({"q": {"type": "choice", "instructions": "x",
                                 "criteria": {f"k{i}": None for i in range(257)}}})
        status, raw = _post(srv.port, "/v1/systemone", body)
        assert status == 422
        assert json.loads(raw)["error"]["code"] == "choice_too_many_options"
        body = _systemone({"q": {"type": "choice", "instructions": "x",
                                 "criteria": {f"k{i}": None for i in range(17)},
                                 "extra": 1}})
        status, raw = _post(srv.port, "/v1/systemone", body)
        assert status == 422
        assert json.loads(raw)["error"]["code"] == "unknown_field"
        assert eng.completion_bodies == []


def test_systemone_with_wide_disabled_keeps_the_legacy_bound(tmp_path):
    with gateway(tmp_path, wide_block=None) as (srv, eng):
        body = _systemone({"q": {"type": "choice", "instructions": "x",
                                 "criteria": {f"k{i}": None for i in range(17)}}})
        status, raw = _post(srv.port, "/v1/systemone", body)
        assert status == 422
        assert json.loads(raw)["error"]["code"] == "choice_too_many_options"


def test_unicode_and_custom_keys_round_trip_through_the_gateway(tmp_path):
    keys = [f"选项-{i}" for i in range(10)] + [f"key.with-punct!{i}" for i in range(8)]
    options = [{"key": k, "description": f"description {i}"} for i, k in enumerate(keys)]
    with gateway(tmp_path) as (srv, eng):
        status, raw = _post(srv.port, "/v1/decide", {
            "kind": "choice", "state": "s", "question": "q", "options": options})
        assert status == 200, raw[:600]
        doc = json.loads(raw)
        assert list(doc["probabilities"]) == keys
        assert doc["selected"] in keys


def test_tokenizer_count_mismatch_refuses_before_any_probability(tmp_path):
    with gateway(tmp_path, engine_fault="bad_tokenize") as (srv, eng):
        status, raw = _post(srv.port, "/v1/decide", _decide(17))
        assert status == 502
        assert json.loads(raw)["error"]["code"] == "backend_error"
        assert "probabilities" not in json.loads(raw)
        assert eng.completion_bodies == []


def test_extended_record_over_the_size_cap_is_refused(tmp_path):
    with gateway(tmp_path, capture={"record_max_bytes": 800}) as (srv, eng):
        status, raw = _post(srv.port, "/v1/decide", _decide(17))
        assert status == 502
        assert "probabilities" not in json.loads(raw)


def test_hybrid_route_serves_wide_alongside_the_frozen_small_readout(tmp_path, monkeypatch):
    """The production hybrid backend must serve wide + small choice + noul on one gateway."""
    import dataclasses

    from qfn_decision import configure
    from qfn_decision.hybrid import ROWS, HybridBackend

    original = backend.FixtureEngineServer._record

    def pinned_record(self, *args, **kwargs):
        return dataclasses.replace(original(self, *args, **kwargs), verbalizer_ids=tuple(ROWS))

    monkeypatch.setattr(backend.FixtureEngineServer, "_record", pinned_record)
    payload, _ = configure.build_configs({
        "CAPTURE_DIR": str(tmp_path / "captures"),
        "WIDE_CHOICE": "1",
        "WIDE_CATALOG_PATH": str(CATALOG_PATH),
    })
    cfg = backend.ApiConfig.from_dict(payload)
    assert cfg.wide_choice is not None and cfg.wide_choice.enabled
    eng = backend.FixtureEngineServer(
        model=cfg.engine_model, capture_dir=tmp_path / "captures",
        wide_choice=cfg.wide_choice).start()
    try:
        payload["engine"]["base_url"] = eng.base_url
        cfg = backend.ApiConfig.from_dict(payload)
        assert cfg.wide_choice.enabled
        with api.GatewayServer(cfg, backend_instance=HybridBackend(cfg), port=0) as srv:
            status, raw = _post(srv.port, "/v1/decide", _decide(26))
            assert status == 200, raw[:800]
            wide_doc = json.loads(raw)
            assert wide_doc["schema"] == "qfn.decide.wide/1"
            assert len(wide_doc["probabilities"]) == 26

            status, raw = _post(srv.port, "/v1/decide", {
                "kind": "noul", "state": "S", "question": "Q"})
            assert status == 200, raw[:400]
            assert set(json.loads(raw)["probabilities"]) == {"false", "true"}

            status, raw = _post(srv.port, "/v1/decide", _decide(3))
            assert status == 200, raw[:400]
            assert json.loads(raw)["schema"] == "qfn.decide/1"

            status, raw = _post(srv.port, "/v1/systemone", _systemone(
                {"wide": {"type": "choice", "instructions": "x",
                          "criteria": {f"k{i}": None for i in range(17)}},
                 "small": {"type": "choice", "instructions": "x",
                           "criteria": {"a": None, "b": None, "c": None}},
                 "flag": {"type": "noul", "instructions": "x"}}))
            assert status == 200, raw[:800]
            answers = json.loads(raw)["answers"]
            assert len(answers["wide"]["probabilities"]) == 17
            assert answers["wide"]["wide"]["adaptation"] == "wide-labels"
            assert len(answers["small"]["probabilities"]) == 3
            assert 0.0 <= answers["flag"]["noul"] <= 1.0
    finally:
        eng.stop()


def test_configure_pins_the_wide_block_only_when_opted_in(tmp_path):
    from qfn_decision import configure

    plain, capture = configure.build_configs({"CAPTURE_DIR": str(tmp_path)})
    assert "wide_choice" not in plain and "wide_choice" not in capture
    payload, capture = configure.build_configs({
        "CAPTURE_DIR": str(tmp_path / "cap"),
        "WIDE_CHOICE": "1",
        "WIDE_CATALOG_PATH": str(CATALOG_PATH),
    })
    block = payload["wide_choice"]
    assert block["catalog_digest"] == wide.PINNED_CATALOG_DIGEST
    assert block["catalog_file_sha256"] == wide.PINNED_CATALOG_FILE_SHA256
    assert block["template_digest"] == wide.WIDE_TEMPLATE_CONTENT_SHA256
    assert block["tokenizer_revision"] == wide.PINNED_CATALOG_TOKENIZER_REVISION
    # the engine block shares the pins but never the gateway-only option cap
    assert capture["wide_choice"] == {k: v for k, v in block.items() if k != "max_options"}
    assert "max_options" not in capture["wide_choice"]
    backend.ApiConfig.from_dict(payload)          # the generated config is loadable
    with pytest.raises(ValueError):
        configure.build_configs({"CAPTURE_DIR": str(tmp_path), "WIDE_CHOICE": "1"})


def test_configured_wide_max_options_is_enforced(tmp_path):
    with gateway(tmp_path, wide_block=_wide_block(max_options=32)) as (srv, eng):
        status, raw = _post(srv.port, "/v1/decide", _decide(33))
        assert status == 422
        assert json.loads(raw)["error"]["code"] == "choice_too_many_options"
        assert eng.tokenize_bodies == []
        status, raw = _post(srv.port, "/v1/decide", _decide(32))
        assert status == 200, raw[:400]
