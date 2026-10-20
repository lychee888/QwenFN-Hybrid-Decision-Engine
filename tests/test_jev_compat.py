"""J1 tests: the Jev-compatible ``POST /v1/systemone`` front end over the typed decision path.

Everything here is CPU-only and local: a **fixture** engine (``backend.FixtureEngineServer``,
labelled ``fixture_only``) plus a loopback gateway, exactly the harness the existing gateway suites
use. No live engine, no GPU, no network, no TypeSafe API call.

The real-shape tests read the frozen, recorded ground truth:
  * ``data/independent-baseline/primary-records.jsonl`` (+ ``…-shuffle-pairs.jsonl``): the 240 frozen
    decision-models-under-pressure records whose option descriptions were used as the choice criteria
    keys by the recorded Jev run,
  * ``reports/independent-baseline/runs/jev-1.13.0-dmup/ledger.jsonl``: 480 real Jev 1.13.0
    responses (240 records x 2 passes) with the probabilities remapped onto our option keys and
    ``choice``/``confidence`` kept as Jev returned them.
Those numbers are used as schema fixtures and as the confidence-formula fit, never as engine output.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import pathlib
import threading
import time

import pytest

from qfn_decision import api, backend, jev_compat, protocol

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "independent-baseline"
LEDGER = ROOT / "reports" / "independent-baseline" / "runs" / "jev-1.13.0-dmup" / "ledger.jsonl"
RECORD_FILES = ("primary-records.jsonl", "primary-shuffle-pairs.jsonl")
GATEWAY_MODEL = "fixture-model"


# ======================================================================================
# scaffolding (same shape as tests/test_gateway_capture_refusal.py)
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
    path = pathlib.Path(tmp_path) / "decision_api_jev.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


@contextlib.contextmanager
def gateway(tmp_path, engine_kw=None, **cfg_over):
    eng = backend.FixtureEngineServer(
        model=GATEWAY_MODEL, capture_dir=pathlib.Path(tmp_path) / "capture", **(engine_kw or {})
    )
    eng.start()
    srv = None
    try:
        cfg = backend.load_api_config(_config_file(tmp_path, eng, **cfg_over))
        srv = api.GatewayServer(cfg, port=0, log_sink=[].append)
        srv.start()
        yield srv, eng
    finally:
        if srv is not None:
            srv.stop()
        eng.stop()


def _post(port, path, payload=None, headers=None, raw=None, timeout=60.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = raw if raw is not None else json.dumps(payload).encode("utf-8")
    hdrs = {"content-type": "application/json"}
    hdrs.update(headers or {})
    try:
        conn.request("POST", path, body=body, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, json.loads(data.decode("utf-8"))
    finally:
        conn.close()


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30.0)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


def _systemone(port, payload, **kw):
    return _post(port, "/v1/systemone", payload, **kw)


def _choice_q(n=3, **over):
    q = {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
            "other": None,
        },
    }
    q["criteria"] = dict(list(q["criteria"].items())[:n])
    q.update(over)
    return q


def _score_q(n=6):
    return {"type": "score", "instructions": "How frustrated is the customer?",
            "criteria": [f"level {i} description" for i in range(n)]}


def _noul_q():
    return {"type": "noul", "instructions": "Does this convey urgency?"}


def _state():
    return "Help! My payouts have been failing for 3 days."


def _refreshment():
    """The documented Jev example request (api.md), as a fixture-usable smoke body."""
    return {
        "state": _state(),
        "model": "jev-latest",
        "questions": {"department": _choice_q(3), "frustration": _score_q(), "is_urgent": _noul_q()},
    }


# ------------------------------------------------------------------ synthetic internal results

def _synthetic(kind: str, probs, options=None, rubrics=None):
    """A real ``DecisionResult`` built from an explicit probability vector (no engine involved)."""
    request = protocol.DecisionRequest.create(
        kind=kind, state="s", question="q?",
        options=options, score_rubrics=rubrics,
    )
    slots = [0.0] * protocol.NUM_SLOTS
    start, _ = protocol.SLOT_RANGES[kind]
    for i, p in enumerate(probs):
        slots[start + i] = p
    result = protocol.DecisionResult.from_slot_probabilities(request, slots)
    return jev_compat.JevQuestion(name="q", requested_type=kind, request=request), result


def _outcome(result):
    return backend.DecisionOutcome(
        result=result, arm="B0", arm_block={}, captured=None, template_block={}, engine_block={},
        tokenizer_block={}, timing={}, usage={"prompt_tokens": 10}, cache={}, transport="fixture",
        uses_s2=False, provider_block={}, parity={},
    )


# ======================================================================================
# 1. shape of the three answer types (real HTTP, fixture engine)
# ======================================================================================

def test_choice_answer_has_jev_shape(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, _refreshment())
    assert status == 200, doc
    assert set(doc) == {"model", "answers", "usage"}
    assert doc["model"] == f"{GATEWAY_MODEL}-b0"
    assert "jev" not in doc["model"].lower() and "typesafe" not in doc["model"].lower()
    assert set(doc["usage"]) == {"input_tokens", "output_tokens"}
    assert doc["usage"]["output_tokens"] == 0
    assert isinstance(doc["usage"]["input_tokens"], int) and doc["usage"]["input_tokens"] > 0

    answer = doc["answers"]["department"]
    assert set(answer) == {"type", "choice", "confidence", "probabilities"}
    assert answer["type"] == "choice"
    keys = list(_choice_q(3)["criteria"])
    assert list(answer["probabilities"]) == keys, "probabilities must cover the criteria keys in order"
    probs = list(answer["probabilities"].values())
    assert abs(sum(probs) - 1.0) < 1e-9 and all(0.0 <= p <= 1.0 for p in probs)
    assert answer["choice"] == keys[probs.index(max(probs))]
    assert answer["confidence"] == jev_compat.choice_confidence(probs)


def test_noul_answer_is_a_bare_probability(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {"model": "jev-1.13.0", "state": _state(),
                                            "questions": {"is_urgent": _noul_q()}})
    assert status == 200, doc
    answer = doc["answers"]["is_urgent"]
    assert set(answer) == {"type", "noul"}, "a noul answer carries no confidence (Jev docs)"
    assert answer["type"] == "noul" and 0.0 <= answer["noul"] <= 1.0
    assert "probabilities" not in answer


def test_noul_criteria_are_accepted(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {
            "model": "jev-latest", "state": _state(),
            "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                                        "criteria": {"true": "Explicitly time-sensitive",
                                                     "false": "No urgency expressed"}}},
        })
    assert status == 200, doc
    assert set(doc["answers"]["is_urgent"]) == {"type", "noul"}


def test_score_answer_has_legend_and_keyed_probabilities(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {"model": "jev-latest", "state": _state(),
                                            "questions": {"frustration": _score_q()}})
    assert status == 200, doc
    answer = doc["answers"]["frustration"]
    assert set(answer) == {"type", "score", "confidence", "legend", "probabilities"}
    assert answer["type"] == "score"
    assert answer["legend"] == {str(i): f"level {i} description" for i in range(6)}
    assert list(answer["probabilities"]) == [str(i) for i in range(6)]
    probs = list(answer["probabilities"].values())
    assert abs(sum(probs) - 1.0) < 1e-9
    assert answer["score"] == pytest.approx(sum(i * p for i, p in enumerate(probs)), abs=1e-12)
    assert 0.0 <= answer["score"] <= 5.0
    assert answer["confidence"] == jev_compat.score_confidence(probs)


def test_multiple_questions_are_answered_batched_and_match_single_question_calls(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, combined = _systemone(srv.port, _refreshment())
        assert status == 200, combined
        assert list(combined["answers"]) == ["department", "frustration", "is_urgent"]
        singles = {}
        for name, question in _refreshment()["questions"].items():
            st, doc = _systemone(srv.port, {"model": "jev-latest", "state": _state(),
                                            "questions": {name: question}})
            assert st == 200, doc
            singles[name] = doc
    for name, doc in singles.items():
        assert combined["answers"][name] == doc["answers"][name]
    # usage.input_tokens is the sum of the prompt tokens of the questions actually asked
    assert combined["usage"]["input_tokens"] == sum(
        d["usage"]["input_tokens"] for d in singles.values()
    )


# ======================================================================================
# 2. translation rules (pure, no engine)
# ======================================================================================

def test_object_and_array_state_are_serialised_deterministically():
    cfg = backend.load_api_config(_config_path_for_offline())
    state = {"record": {"b": 2, "a": 1}, "labels": ["x", "y"], "n": 3}
    plan = jev_compat.translate_request(
        {"model": "jev-latest", "state": state, "questions": {"q": _noul_q()}}, cfg
    )
    assert plan.state_text == '{"labels":["x","y"],"n":3,"record":{"a":1,"b":2}}'
    # the same logical state always renders the same internal text, whatever the key order was
    reordered = {"n": 3, "labels": ["x", "y"], "record": {"a": 1, "b": 2}}
    assert jev_compat.translate_request(
        {"state": reordered, "questions": {"q": _noul_q()}}, cfg
    ).state_text == plan.state_text
    # a pre-serialised string is passed through verbatim
    assert jev_compat.translate_request(
        {"state": plan.state_text, "questions": {"q": _noul_q()}}, cfg
    ).state_text == plan.state_text

    as_array = ["a", {"z": 1, "y": 2}]
    plan2 = jev_compat.translate_request(
        {"state": as_array, "questions": {"q": _noul_q()}}, cfg
    )
    assert plan2.state_text == '["a",{"y":2,"z":1}]'


def test_structured_instructions_become_the_question_text():
    cfg = backend.load_api_config(_config_path_for_offline())
    instructions = {"question": "Is the resume for the same person as `potential_duplicate`?",
                    "potential_duplicate": {"name": "John Smith", "location": "Oakland"}}
    plan = jev_compat.translate_request(
        {"state": "resume text",
         "questions": {"same_person": {"type": "noul", "instructions": instructions}}}, cfg
    )
    assert plan.questions[0].request.question == (
        '{"potential_duplicate":{"location":"Oakland","name":"John Smith"},'
        '"question":"Is the resume for the same person as `potential_duplicate`?"}'
    )
    plain = jev_compat.translate_request(
        {"state": "resume text",
         "questions": {"same_person": {"type": "noul",
                                       "instructions": "Is it the same person?"}}}, cfg
    )
    assert plain.questions[0].request.question == "Is it the same person?"


def test_choice_rubric_mapping_null_equal_and_text():
    cfg = backend.load_api_config(_config_path_for_offline())
    criteria = {"billing": "Payments, invoicing, refunds", "technical": None,
                "sales": "sales", "ops": {"runbook": "restart"}}
    plan = jev_compat.translate_request(
        {"state": "s", "questions": {"department": {"type": "choice", "instructions": "Which?",
                                                    "criteria": criteria}}}, cfg
    )
    question = plan.questions[0]
    assert list(question.request.caller_keys()) == list(criteria)
    assert question.request.descriptions() == (
        "billing: Payments, invoicing, refunds",
        "technical",                      # null rubric: the key is the description
        "sales",                          # rubric equal to the key: the key is the description
        'ops: {"runbook":"restart"}',     # structured rubric: deterministic compact JSON
    )


def test_noul_criteria_are_folded_into_the_question_not_dropped():
    cfg = backend.load_api_config(_config_path_for_offline())
    plan = jev_compat.translate_request(
        {"state": "s", "questions": {"is_urgent": {
            "type": "noul", "instructions": "Does this convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}}}}, cfg
    )
    assert plan.questions[0].request.question == (
        "Does this convey urgency? (yes means: Explicitly time-sensitive; "
        "no means: No urgency expressed)"
    )


def test_choice_tie_breaks_by_criteria_order():
    keys = ["billing", "technical", "sales"]
    question, result = _synthetic(
        "choice", [0.5, 0.5, 0.0],
        options=[protocol.ChoiceOption(key=k, description=k) for k in keys],
    )
    answer = jev_compat.build_answer(question, _outcome(result))
    assert answer["choice"] == "billing", "a tie resolves to the earliest criteria entry"
    assert list(answer["probabilities"]) == keys
    assert answer["confidence"] == pytest.approx(0.25)

    question2, result2 = _synthetic(
        "choice", [0.1, 0.6, 0.3],
        options=[protocol.ChoiceOption(key=k, description=k) for k in keys],
    )
    answer2 = jev_compat.build_answer(question2, _outcome(result2))
    assert answer2["choice"] == "technical"
    assert answer2["probabilities"]["technical"] == pytest.approx(0.6)


def test_score_answer_uses_the_ordered_levels():
    rubrics = [f"L{i}" for i in range(6)]
    question, result = _synthetic("score", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], rubrics=rubrics)
    answer = jev_compat.build_answer(question, _outcome(result))
    assert answer["score"] == 2.0
    assert answer["confidence"] == 1.0
    assert answer["legend"] == {str(i): f"L{i}" for i in range(6)}


# ======================================================================================
# 3. confidence: the documented formulas, cross-checked against the published examples
# ======================================================================================

def test_choice_confidence_matches_the_documented_examples():
    # https://docs.typesafe.ai/confidence#choice
    assert jev_compat.choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4, abs=1e-12)
    assert jev_compat.choice_confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4, abs=1e-12)
    assert jev_compat.choice_confidence([1.0, 0.0]) == pytest.approx(1.0, abs=1e-12)
    assert jev_compat.choice_confidence([0.5, 0.5]) == pytest.approx(0.0, abs=1e-12)
    assert jev_compat.choice_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0, abs=1e-12)
    assert jev_compat.choice_confidence([0.64, 0.36]) == pytest.approx(0.28, abs=1e-12)


def test_score_confidence_matches_the_documented_examples():
    # https://docs.typesafe.ai/confidence#score
    assert jev_compat.score_confidence([0, 0.5, 0.5]) == pytest.approx(0.25, abs=1e-12)
    assert jev_compat.score_confidence([0.5, 0, 0.5]) == pytest.approx(0.0, abs=1e-12)
    assert jev_compat.score_confidence([0, 0.57, 0.43]) == pytest.approx(0.355, abs=1e-3)
    assert jev_compat.score_confidence([0, 0, 1, 0, 0, 0]) == pytest.approx(1.0, abs=1e-12)










# ======================================================================================
# 4. all 480 real request shapes translate and answer in the Jev shape
# ======================================================================================







# ======================================================================================
# 5. limits and unsupported input -> Jev-style 422 naming the field
# ======================================================================================

def _err(doc):
    assert set(doc) == {"error"}, doc
    return doc["error"]


@pytest.mark.parametrize("n", [0, 1])
def test_choice_under_two_options_is_422(tmp_path, n):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {
            "model": "jev-latest", "state": "s",
            "questions": {"q": {"type": "choice", "instructions": "i",
                                "criteria": {f"k{i}": None for i in range(n)}}},
        })
    error = _err(doc)
    assert status == 422 and error["code"] == "choice_too_few_options"
    assert error["field"] == "questions.q.criteria" and error["limit"] == 2
    assert error["question"] == "q"


def test_choice_over_sixteen_options_is_422(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {
            "model": "jev-latest", "state": "s",
            "questions": {"department": {"type": "choice", "instructions": "i",
                                         "criteria": {f"k{i}": None for i in range(17)}}},
        })
    error = _err(doc)
    assert status == 422 and error["code"] == "choice_too_many_options"
    assert error["field"] == "questions.department.criteria" and error["limit"] == 16
    assert "16" in error["message"]


@pytest.mark.parametrize("n", [2, 3, 5, 7, 10])
def test_score_without_exactly_six_levels_is_422(tmp_path, n):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {"model": "jev-latest", "state": "s",
                                            "questions": {"frustration": _score_q(n)}})
    error = _err(doc)
    assert status == 422 and error["code"] == "score_level_count"
    assert error["field"] == "questions.frustration.criteria" and error["limit"] == 6
    assert "6" in error["message"]


def test_unknown_question_type_is_422(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {
            "model": "jev-latest", "state": "s",
            "questions": {"q": {"type": "multichoice", "instructions": "i",
                                "criteria": {"a": None, "b": None}}},
        })
    error = _err(doc)
    assert status == 422 and error["code"] == "unknown_question_type"
    assert error["field"] == "questions.q.type"
    assert error["limit"] == ["noul", "choice", "score"]


@pytest.mark.parametrize("body,fragment", [
    ({"model": "jev-latest", "questions": {"q": _noul_q()}}, "state"),
    ({"model": "jev-latest", "state": ""}, "state"),
    ({"model": "jev-latest", "state": "s", "questions": {}}, "questions"),
    ({"model": "jev-latest", "state": "s", "questions": {"q": {"instructions": "i"}}}, "type"),
    ({"model": "jev-latest", "state": "s", "questions": {"q": {"type": "noul"}}}, "instructions"),
    ({"model": "jev-latest", "state": "s",
      "questions": {"q": {"type": "choice", "instructions": "i"}}}, "criteria"),
    ({"model": "jev-latest", "state": "s", "questions": {"q": {"type": "score", "instructions": "i"}}},
     "criteria"),
])
def test_missing_required_fields_are_422_naming_the_field(tmp_path, body, fragment):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, body)
    error = _err(doc)
    assert status == 422 and error["code"] == "missing_field"
    assert fragment in error["field"]
    assert fragment in error["message"]


@pytest.mark.parametrize("bad,fragment,code", [
    ({"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": ["a", "b"]}}},
     "criteria", "bad_field_type"),
    ({"state": "s", "questions": {"q": {"type": "score", "instructions": "i",
                                        "criteria": {str(i): None for i in range(6)}}}},
     "criteria", "bad_field_type"),
    ({"state": "s", "questions": {"q": {"type": "choice", "instructions": "i",
                                        "criteria": {"a": 7, "b": None}}}},
     "criteria.a", "bad_field_type"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i",
                                        "criteria": {"maybe": "x"}}}},
     "criteria", "unknown_field"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "criteria": "yes"}}},
     "criteria", "bad_field_type"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "extra": 1}}},
     "questions.q", "unknown_field"),
    ({"state": "s", "questions": {"q": {"type": "score", "instructions": "i",
                                        "criteria": ["a", "b", "c", "d", "e", None]}}},
     "criteria[5]", "bad_field_type"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": 12}}},
     "instructions", "bad_field_type"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}, "model": 5},
     "model", "bad_field_type"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}, "temperature": 1},
     "body", "unknown_field"),
])
def test_malformed_fields_are_422_naming_the_field(tmp_path, bad, fragment, code):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, bad)
    error = _err(doc)
    assert status == 422 and error["code"] == code, doc
    assert fragment in error["field"], doc


def test_multiline_choice_rubric_is_refused_not_rewritten(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {
            "model": "jev-latest", "state": "s",
            "questions": {"q": {"type": "choice", "instructions": "i",
                                "criteria": {"a": "line one\nline two", "b": None}}},
        })
    error = _err(doc)
    assert status == 422 and error["code"] == "protocol_error"
    assert error["field"] == "questions.q"
    assert "single-line" in error["message"]


def test_oversized_state_is_422_not_truncated(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {"model": "jev-latest", "state": "x" * 40_000,
                                            "questions": {"q": _noul_q()}})
    error = _err(doc)
    assert status == 422 and error["code"] == "protocol_error"
    assert "state" in error["message"] and "oversized" in error["message"]


# ======================================================================================
# 6. fail closed: an internal failure on any question fails the whole request
# ======================================================================================

def _refusal_entry(engine_id: str, reason: str = "own_row_has_draft_tokens", **over) -> dict:
    entry = {
        "schema": backend.CAPTURE_REFUSED_SCHEMA,
        "engine_request_id": engine_id,
        "reason": reason,
        "final_refusal": True,
        "helper_variant": "qfn-capture-mtp1/1",
        "batch_index": 2,
        "wall_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns() * 1e9,
    }
    entry.update(over)
    return entry


def _refusal_payload(engine_id: str) -> dict:
    return {
        "schema": backend.CAPTURE_REFUSED_SCHEMA,
        "helper_version": "qfn-capture-runtime/1",
        "helper_variant": "qfn-capture-mtp1/1",
        "marker_name": backend.CAPTURE_REFUSED_MARKER_NAME,
        "entry_count": 1,
        "truncated": False,
        "entries": [_refusal_entry(engine_id)],
    }


def test_internal_refusal_is_503_naming_the_question_with_no_partial_answers(tmp_path):
    with gateway(tmp_path, engine_kw={"refusal_marker": _refusal_payload("{engine_id}")}) as (srv, _eng):
        started = time.monotonic()
        status, doc = _systemone(srv.port, {
            "model": "jev-latest", "state": _state(),
            "questions": {"department": _choice_q(3), "is_urgent": _noul_q()},
        })
        elapsed = time.monotonic() - started
    error = _err(doc)
    assert status == 503, doc
    assert error["code"] == backend.CAPTURE_REFUSED_CODE == "capture_refused"
    assert error["status"] == 503
    assert error["question"] in {"department", "is_urgent"}
    assert error["field"] == f"questions.{error['question']}"
    assert "own_row_has_draft_tokens" in error["message"]
    assert elapsed < 2.0, f"a final refusal must not wait for the 30 s deadline (took {elapsed:.2f}s)"


def test_internal_deadline_is_504_naming_the_question_with_no_partial_answers(tmp_path):
    with gateway(tmp_path, engine_kw={"fault": "missing"},
                 timeouts={"capture_deadline_s": 1.0}) as (srv, _eng):
        status, doc = _systemone(srv.port, {
            "model": "jev-latest", "state": _state(),
            "questions": {"department": _choice_q(3), "is_urgent": _noul_q()},
        })
    error = _err(doc)
    assert status == 504 and error["code"] == "deadline_exceeded"
    assert error["question"] in {"department", "is_urgent"}
    assert "answers" not in doc


def test_single_question_failure_is_named_exactly(tmp_path):
    with gateway(tmp_path, engine_kw={"refusal_marker": _refusal_payload("{engine_id}")}) as (srv, _eng):
        status, doc = _systemone(srv.port, {"model": "jev-latest", "state": _state(),
                                            "questions": {"only_one": _choice_q(2)}})
    error = _err(doc)
    assert status == 503 and error["question"] == "only_one"
    assert error["field"] == "questions.only_one"


# ======================================================================================
# 7. bounded concurrency, headers, and the routes that must not change
# ======================================================================================

class _CountingBackend:
    """Wraps the real backend and records how many decisions are executing at once."""

    def __init__(self, inner):
        self.inner = inner
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0
        self.calls = 0

    def decide(self, request, **kw):
        with self.lock:
            self.now += 1
            self.calls += 1
            self.peak = max(self.peak, self.now)
        try:
            time.sleep(0.05)          # hold the slot long enough for the others to arrive
            return self.inner.decide(request, **kw)
        finally:
            with self.lock:
                self.now -= 1

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_multiple_questions_run_concurrently_but_never_exceed_max_inflight(tmp_path):
    with gateway(tmp_path, concurrency={"max_inflight": 3}) as (srv, _eng):
        counter = _CountingBackend(srv.backend)
        srv.backend = counter
        questions = {f"q{i}": _noul_q() for i in range(12)}
        status, doc = _systemone(srv.port, {"model": "jev-latest", "state": _state(),
                                            "questions": questions})
    assert status == 200, doc
    assert list(doc["answers"]) == list(questions)
    assert counter.calls == 12
    assert counter.peak <= 3, f"max_inflight=3 but {counter.peak} decisions ran at once"
    assert counter.peak > 1, "the questions were not run concurrently at all"


def test_authorization_header_is_accepted_and_ignored(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {"model": "jev-latest", "state": _state(),
                                            "questions": {"q": _noul_q()}},
                                 headers={"authorization": "Bearer not-a-real-key"})
    assert status == 200, doc


@pytest.mark.parametrize("model", ["jev-latest", "jev-1.13.0", "gpt-4o", "who-knows"])
def test_any_requested_model_is_accepted_and_never_echoed_as_ours(tmp_path, model):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {"model": model, "state": _state(),
                                            "questions": {"q": _noul_q()}})
    assert status == 200, doc
    assert doc["model"] == f"{GATEWAY_MODEL}-b0"
    assert model not in json.dumps(doc)


def test_request_without_a_model_field_is_accepted(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _systemone(srv.port, {"state": _state(), "questions": {"q": _noul_q()}})
    assert status == 200
    assert doc["model"] == f"{GATEWAY_MODEL}-b0"


def test_strict_body_parsing_is_shared_with_the_decide_route(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        assert _post(srv.port, "/v1/systemone", raw=b"{not json}")[0] == 400
        assert _post(srv.port, "/v1/systemone", raw=b"[1,2]")[0] == 400
        assert _post(srv.port, "/v1/systemone", raw=b'{"state":1,"state":2}')[0] == 400
        assert _post(srv.port, "/v1/systemone", raw=b"")[0] == 400
        assert _post(srv.port, "/v1/systemone", raw=b'{"state":NaN}')[0] == 400


def test_trailing_slash_is_the_same_route(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _post(srv.port, "/v1/systemone/", {"model": "jev-latest", "state": _state(),
                                                         "questions": {"q": _noul_q()}})
    assert status == 200, doc


def test_decide_route_is_unchanged(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _post(srv.port, "/v1/decide", {
            "kind": "choice", "state": "you are asked to pick", "question": "which one?",
            "options": [{"key": "k0", "description": "option 0"},
                        {"key": "k1", "description": "option 1"}],
            "thinking": "off",
        })
        assert status == 200, doc
        assert doc["schema"] == api.DECIDE_SCHEMA == "qfn.decide/1"
        assert set(doc["probabilities"]) == {"k0", "k1"}
        assert doc["selected"] in {"k0", "k1"}
        assert doc["output_tokens"] == 0
        # the pre-existing failure envelope is untouched: no new keys on a plain error
        status2, doc2 = _post(srv.port, "/v1/decide", {"kind": "nope"})
        assert status2 == 422 and doc2["error"]["code"] == "protocol_error"
        assert set(doc2["error"]) == {"code", "message", "status"}


def test_health_is_unchanged(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        before = srv.health()
        status, doc = _get(srv.port, "/health")
    assert status == 200
    assert doc == before
    assert doc["schema"] == "qfn.decide.health/1"
    assert doc["protocol"]["jev_confidence"] is None
    assert doc["protocol"]["choice_max_options"] == protocol.MAX_CHOICE_OPTIONS
    assert doc["protocol"]["score_levels"] == 6
    assert "systemone" not in json.dumps(doc).lower()


def test_route_methods_and_unknown_routes(tmp_path):
    with gateway(tmp_path) as (srv, _eng):
        status, doc = _get(srv.port, "/v1/systemone")
        assert status == 405 and doc["error"]["code"] == "method_not_allowed"
        assert set(doc["error"]) == {"code", "message", "status"}
        status, doc = _post(srv.port, "/v1/systemone-typo", {"state": "s"})
        assert status == 404 and doc["error"]["code"] == "not_found"


def _config_path_for_offline():
    """A config file for pure-translation tests (no server, no engine is contacted)."""
    return ROOT / "configs" / "decision_api.json"
