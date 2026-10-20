"""M1 tests: the record writer itself (torch CPU) - index agreement + refusal markers.

Scope and honesty notes
-----------------------
* torch-CPU only (`pytest.importorskip`), no GPU, no engine, no network. It exercises the REAL
  `capture_after_compute_logits` path (selection -> writer -> npz) with torch tensors, which is the
  closest thing to the engine that can run offline.
* What it proves:
  1. the captured hidden vector and the captured 24 logits are read/derived at ONE index: the token
     index `logits_indices[logits_row_index]` that the cross-check validated - NOT the logits ROW
     index (`sample_hidden_states = hidden_states[input_batch.logits_indices]`,
     reports/s3/staged/model_runner.py:1403-1404);
  2. a FINAL refusal writes a per-request refusal marker and no record; a non-final refusal
     (`chunk_not_final`) writes neither;
  3. the refusal marker is bounded and atomic (cap + drop disclosure).
* It is NOT evidence about GPU kernels, CUDA graphs, or live engine behaviour.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parents[1]
HIDDEN = 2560
SLOTS = 24
MARKED = "cmpl-qfn-dec-0"
BONUS = 1


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mtp1():
    return _load("qfn_capture_runtime_mtp1", "engine/qfn_capture_runtime_production.py")


# ======================================================================================
# a minimal stand-in for the runner/batch surface the hook reads (mirrors the real one)
# ======================================================================================

class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Head:
    def __init__(self):
        self.tp_rank = 0
        self.tp_size = 1
        self.weight = torch.zeros(8, HIDDEN, dtype=torch.float32)
        self.shard_indices = _Bag(org_vocab_start_index=0, org_vocab_end_index=8,
                                  num_org_vocab_padding=0)


class _LP:
    scale = 1.0
    soft_cap = None
    org_vocab_size = 151936
    head_dtype = torch.bfloat16
    use_all_gather = False


class _Model:
    def __init__(self):
        self.lm_head = _Head()
        self.logits_processor = _LP()


class _Runner:
    def __init__(self, prompt_ids):
        self.model = _Model()
        self.vllm_config = _Bag(
            compilation_config=_Bag(cudagraph_mode="FULL_DECODE_ONLY"),
            model_config=_Bag(model="fixture"),
        )
        self.parallel_config = _Bag(pipeline_parallel_size=1)
        n_states = int(np.asarray(prompt_ids).shape[0])
        self.req_states = _Bag(
            prompt_len=_Bag(np=np.zeros(n_states, dtype=np.int32)),
            all_token_ids=_Bag(_uva_buf=_Bag(np=prompt_ids)),
        )


class _InputBatch:
    """The subset of `InputBatch` fields the helper reads (all as torch tensors where the real one
    uses torch tensors, so `.item()` / slicing behave identically)."""

    def __init__(self, kw, hidden_rows):
        self.req_ids = kw["req_ids"]
        self.num_draft_tokens = kw["num_draft_tokens"]
        self.idx_mapping_np = kw["idx_mapping_np"]
        self.num_computed_tokens_np = kw["num_computed_tokens_np"]
        self.num_scheduled_tokens = torch.as_tensor(kw["num_scheduled_tokens"])
        self.prefill_len_np = kw["prefill_len_np"]
        self.is_prefilling_np = kw["is_prefilling_np"]
        self.query_start_loc_np = kw["query_start_loc_np"]
        self.cu_num_logits_np = kw["cu_num_logits_np"]
        self.num_draft_tokens_per_req = kw["num_draft_tokens_per_req"]
        self.logits_indices = torch.as_tensor(kw["logits_indices_np"], dtype=torch.int64)
        self.positions = torch.as_tensor(hidden_rows["positions"], dtype=torch.int64)
        self.input_ids = torch.as_tensor(hidden_rows["input_ids"], dtype=torch.int64)


def _mixed_kw(*, mark_position=2, n_decodes=7, drafts=3, mark_prompt_len=9, mark_computed=0,
              mark_scheduled=None, mark_own_drafts=0, mark_req_id=MARKED, extra_marked=0):
    """Same construction as tests/test_capture_mtp1_mixed_batch.py (the runner's own layout)."""
    entries = []
    gen = 0
    for pos in range(n_decodes + 1):
        if pos == mark_position:
            entries.append(dict(req_id=mark_req_id, prompt_len=mark_prompt_len,
                                computed=mark_computed, drafts=mark_own_drafts,
                                scheduled=(mark_scheduled if mark_scheduled is not None
                                           else mark_prompt_len - mark_computed)))
        else:
            entries.append(dict(req_id=f"cmpl-gen{gen}-0", prompt_len=5, computed=5,
                                scheduled=drafts + BONUS, drafts=drafts))
            gen += 1
    for j in range(extra_marked):
        entries.append(dict(req_id=f"cmpl-qfn-dec-extra{j}-0", prompt_len=6, computed=0,
                            scheduled=6, drafts=0))
    n = len(entries)
    scheduled = np.asarray([e["scheduled"] for e in entries], dtype=np.int32)
    own = np.asarray([e["drafts"] + BONUS for e in entries], dtype=np.int32)
    q = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(scheduled, out=q[1:])
    cu = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(own, out=cu[1:])
    li = np.empty(int(cu[-1]), dtype=np.int64)
    for i in range(n):
        for k in range(int(own[i])):
            li[int(cu[i]) + k] = int(q[i + 1]) - int(own[i]) + k
    states = list(reversed(range(n)))
    idx_mapping = np.asarray(states, dtype=np.intp)
    prompt_len_np = np.zeros(n, dtype=np.int32)
    for i, st in enumerate(states):
        prompt_len_np[st] = entries[i]["prompt_len"]
    computed = np.asarray([e["computed"] for e in entries], dtype=np.int32)
    return dict(
        req_ids=[e["req_id"] for e in entries],
        idx_mapping_np=idx_mapping,
        num_computed_tokens_np=computed,
        num_scheduled_tokens=scheduled,
        prompt_len_np=prompt_len_np,
        prefill_len_np=prompt_len_np[idx_mapping],
        is_prefilling_np=computed < prompt_len_np[idx_mapping],
        query_start_loc_np=q,
        cu_num_logits_np=cu,
        num_draft_tokens=int(own.sum() - n),
        num_draft_tokens_per_req=np.asarray([e["drafts"] for e in entries], dtype=np.int32),
        logits_indices_np=li,
        entries=entries,
    )


def _write_config(tmp_path, mtp1, **over):
    cfg = {
        "capture_dir": str(tmp_path / "captures"),
        "marker_prefix": "cmpl-qfn-",
        "verbalizers": {"names": [f"v{i}" for i in range(SLOTS)],
                        "ids": list(range(1000, 1000 + SLOTS))},
        "capture_fp32_ref": True,
    }
    cfg.update(over)
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def capture_env(tmp_path, mtp1, monkeypatch):
    """Enable the helper against a tmp capture dir; CPU torch has no CUDA stream to watch."""
    path = _write_config(tmp_path, mtp1)
    monkeypatch.setenv(mtp1.ENV_ENABLE, "1")
    monkeypatch.setenv(mtp1.ENV_CONFIG, str(path))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False, raising=False)
    mtp1.reset_capture_state()
    yield tmp_path
    mtp1.reset_capture_state()


def _tensors(kw, *, prompt_ids_fill=100):
    """hidden_states[k] is filled with the value k so the ROW INDEX it was read at is readable back;
    logits[j] is computed from hidden_states[logits_indices[j]] exactly as the runner does."""
    n_tokens = int(kw["query_start_loc_np"][-1])
    hidden = torch.zeros(n_tokens, HIDDEN, dtype=torch.float32)
    for k in range(n_tokens):
        hidden[k] = float(k)
    positions = np.zeros(n_tokens, dtype=np.int64)
    input_ids = np.full(n_tokens, int(prompt_ids_fill), dtype=np.int64)
    for i, e in enumerate(kw["entries"]):
        q0, q1 = int(kw["query_start_loc_np"][i]), int(kw["query_start_loc_np"][i + 1])
        positions[q0:q1] = e["computed"] + np.arange(q1 - q0)
        input_ids[q0:q1] = np.arange(int(prompt_ids_fill), int(prompt_ids_fill) + (q1 - q0))
    sample_rows = hidden[torch.as_tensor(kw["logits_indices_np"], dtype=torch.int64)]
    # a vocab-width logits row: the 24 verbalizer columns carry the hidden row value (so "which row
    # did this come from" is readable back), everything else is -1 so argmax is the first
    # verbalizer column, exactly as a real masked row would behave for the readout.
    logits = torch.full((sample_rows.shape[0], 1024), -1.0, dtype=torch.float32)
    logits[:, 1000:1000 + SLOTS] = sample_rows[:, :1].repeat(1, SLOTS).clone()
    return hidden, logits, {"positions": positions, "input_ids": input_ids}


def _prompt_ids_table(kw, *, fill=100):
    """A (n_states, max_prompt_len) token table for req_states.all_token_ids."""
    n_states = int(kw["idx_mapping_np"].max()) + 1
    width = max(e["prompt_len"] for e in kw["entries"])
    table = np.zeros((n_states, width), dtype=np.int64)
    for i, st in enumerate(kw["idx_mapping_np"]):
        table[int(st), : kw["entries"][i]["prompt_len"]] = np.arange(
            fill + i * 50, fill + i * 50 + kw["entries"][i]["prompt_len"]
        )
    return table


# ======================================================================================
# 1. index agreement: vector and logits come from the SAME validated token index
# ======================================================================================

def test_capture_reads_the_vector_and_the_logits_at_the_same_token_index(capture_env, mtp1):
    kw = _mixed_kw(mark_position=2)
    hidden, logits, extra = _tensors(kw)
    batch = _InputBatch(kw, extra)
    i = kw["req_ids"].index(MARKED)
    j = int(kw["cu_num_logits_np"][i])                      # the marked row's logits ROW index
    token_index = int(kw["logits_indices_np"][j])            # the TOKEN index the row maps to
    assert token_index != j, "this layout must distinguish the two index spaces"

    runner = _Runner(_prompt_ids_table(kw))
    runner.req_states.prompt_len.np[:] = kw["prompt_len_np"]
    mtp1.capture_after_compute_logits(runner, hidden, logits, batch)

    caps = sorted(p for p in (capture_env / "captures").glob("*.npz"))
    assert len(caps) == 1, f"expected exactly one record, got {[p.name for p in caps]}"
    with np.load(caps[0], allow_pickle=False) as z:
        vec = z["vector_fp32"]
        slots = z["verbalizer_logits_fp32"]
        meta = json.loads(bytes(np.asarray(z["meta_json"], dtype=np.uint8)).decode("utf-8"))
    assert meta["engine_request_id"] == MARKED
    assert meta["logits_row_index"] == j
    assert meta["terminal_position"] == meta["prompt_len"] - 1
    # the hidden vector is the row at the TOKEN index, not at the logits row index
    assert float(vec[0]) == float(token_index)
    assert float(vec[0]) != float(j)
    # ... and the 24 captured logits were derived from that SAME row (fixture: logits[j][*] == row)
    assert float(slots[0]) == float(token_index)
    assert float(slots[-1]) == float(token_index)
    assert meta["runtime_argmax_token_id"] in list(range(1000, 1000 + SLOTS))
    assert meta["logits_captured"] is True
    assert meta["prompt_len"] == kw["entries"][i]["prompt_len"]
    assert meta["computed_tokens_before_step"] == kw["entries"][i]["computed"]
    assert meta["num_scheduled_tokens"] == kw["entries"][i]["scheduled"]
    assert meta["token_hash_scope"] == "prompt" and meta["token_hash_from_engine_state"] is True
    assert meta["runner_source_sha256"] == mtp1.RUNNER_SOURCE_SHA256
    assert not (capture_env / "captures" / mtp1.REFUSED_MARKER_NAME).exists()


def test_capture_bound_to_the_prompt_tokens_the_engine_holds(capture_env, mtp1):
    from qfn_decision import features

    kw = _mixed_kw(mark_position=1)
    hidden, logits, extra = _tensors(kw)
    table = _prompt_ids_table(kw)
    runner = _Runner(table)
    runner.req_states.prompt_len.np[:] = kw["prompt_len_np"]
    mtp1.capture_after_compute_logits(runner, hidden, logits, _InputBatch(kw, extra))

    caps = list((capture_env / "captures").glob("*.npz"))
    assert len(caps) == 1
    with np.load(caps[0], allow_pickle=False) as z:
        meta = json.loads(bytes(np.asarray(z["meta_json"], dtype=np.uint8)).decode("utf-8"))
    i = kw["req_ids"].index(MARKED)
    st = int(kw["idx_mapping_np"][i])
    plen = kw["entries"][i]["prompt_len"]
    expect = features.token_sequence_sha256([int(v) for v in table[st, :plen]])
    assert meta["prompt_token_sha256"] == expect
    assert meta["prompt_len"] == plen


# ======================================================================================
# 2. refusal marker: written for a FINAL refusal, never for a non-final one
# ======================================================================================

def _refusal_marker(tmp_path):
    path = tmp_path / "captures" / "CAPTURE_REFUSED.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def test_final_refusal_writes_a_per_request_marker_and_no_record(capture_env, mtp1):
    kw = _mixed_kw(mark_position=2, mark_own_drafts=3)
    hidden, logits, extra = _tensors(kw)
    runner = _Runner(_prompt_ids_table(kw))
    runner.req_states.prompt_len.np[:] = kw["prompt_len_np"]
    mtp1.capture_after_compute_logits(runner, hidden, logits, _InputBatch(kw, extra))

    assert list((capture_env / "captures").glob("*.npz")) == []
    marker = _refusal_marker(capture_env)
    assert marker is not None
    assert marker["schema"] == mtp1.REFUSED_MARKER_SCHEMA
    assert marker["helper_variant"] == mtp1.MTP1_HELPER_VARIANT
    assert marker["entry_count"] == 1 and marker["truncated"] is False
    entry = marker["entries"][0]
    assert entry["engine_request_id"] == MARKED
    assert entry["reason"] == "own_row_has_draft_tokens"
    assert entry["final_refusal"] is True
    assert entry["own_draft_tokens"] == 3
    assert entry["cu_span"] == 4
    assert entry["batch_num_drafts"] == kw["num_draft_tokens"]
    assert entry["prompt_len"] == kw["entries"][2]["prompt_len"]
    # the flat TOKEN index the refused row maps to (logits_indices[logits_row_index]) is the first
    # of its own logits rows, i.e. query_end - cu_span; the absolute POSITION of that token is
    # prompt_len-1. Both facts are reported, neither is inferred.
    assert entry["logits_index"] == entry["query_end"] - entry["cu_span"]
    assert entry["terminal_position"] == entry["prompt_len"] - 1
    assert entry["wall_ns"] > 0 and entry["monotonic_ns"] > 0


def test_nonfinal_chunk_refusal_writes_no_marker_and_no_record(capture_env, mtp1):
    kw = _mixed_kw(mark_position=1, mark_prompt_len=10, mark_computed=0, mark_scheduled=6)
    hidden, logits, extra = _tensors(kw)
    runner = _Runner(_prompt_ids_table(kw))
    runner.req_states.prompt_len.np[:] = kw["prompt_len_np"]
    state = mtp1.get_capture_state()
    mtp1.capture_after_compute_logits(runner, hidden, logits, _InputBatch(kw, extra))

    assert list((capture_env / "captures").glob("*.npz")) == []
    assert _refusal_marker(capture_env) is None, "a still-capturable request must not be refused"
    assert state.skips.get("chunk_not_final") == 1
    assert state.refusals_written == 0


def test_a_refused_row_does_not_stop_a_second_marked_row_in_the_same_batch(capture_env, mtp1):
    # 9 rows (> max_num_seqs 8) so the batch still carries drafts while one marked row is clean
    kw = _mixed_kw(mark_position=2, n_decodes=7, mark_own_drafts=3, extra_marked=1)
    hidden, logits, extra = _tensors(kw)
    table = _prompt_ids_table(kw)
    runner = _Runner(table)
    runner.req_states.prompt_len.np[:] = kw["prompt_len_np"]
    mtp1.capture_after_compute_logits(runner, hidden, logits, _InputBatch(kw, extra))

    caps = list((capture_env / "captures").glob("*.npz"))
    assert len(caps) == 1, [p.name for p in caps]
    with np.load(caps[0], allow_pickle=False) as z:
        meta = json.loads(bytes(np.asarray(z["meta_json"], dtype=np.uint8)).decode("utf-8"))
    assert meta["engine_request_id"] == "cmpl-qfn-dec-extra0-0"
    marker = _refusal_marker(capture_env)
    assert [e["engine_request_id"] for e in marker["entries"]] == [MARKED]
    assert marker["entries"][0]["reason"] == "own_row_has_draft_tokens"


def test_refusal_marker_is_bounded_and_discloses_drops(capture_env, mtp1):
    cap = mtp1.REFUSED_MAX_ENTRIES
    assert cap == 32
    for n in range(cap + 8):
        mtp1.merge_refusal_marker(
            str(capture_env / "captures"),
            [mtp1.build_refusal_entry(engine_request_id=f"cmpl-qfn-x{n:04x}-0",
                                      reason="logits_span_not_one", batch_index=0, facts={})],
        )
    marker = _refusal_marker(capture_env)
    assert marker["entry_count"] == cap
    assert marker["max_entries"] == cap
    assert marker["truncated"] is True
    assert marker["dropped_entries_total"] == 8
    # newest kept, oldest dropped
    assert marker["entries"][-1]["engine_request_id"] == f"cmpl-qfn-x{cap + 7:04x}-0"
    assert all(e["engine_request_id"] != "cmpl-qfn-x0000-0" for e in marker["entries"])
    # the entry schema is fixed and documented
    assert set(marker["entries"][0]) >= {
        "schema", "engine_request_id", "reason", "final_refusal", "helper_variant", "batch_index",
        "wall_ns", "monotonic_ns", "own_draft_tokens", "batch_num_drafts", "cu_span",
        "has_draft_tokens_per_req", "has_cu_num_logits_np", "has_logits_indices",
        "logits_row_index", "logits_index", "query_start", "query_end",
        "computed_tokens_before_step", "num_scheduled_tokens", "prompt_len", "terminal_position",
    }


def test_refusal_entry_refuses_an_unmarked_or_unknown_reason(mtp1):
    with pytest.raises(mtp1.SelectionError):
        mtp1.build_refusal_entry(engine_request_id="cmpl-plain-0", reason="chunk_not_final",
                                 batch_index=0, facts={})
    with pytest.raises(mtp1.SelectionError):
        mtp1.build_refusal_entry(engine_request_id=MARKED, reason="nonsense",
                                 batch_index=0, facts={})
