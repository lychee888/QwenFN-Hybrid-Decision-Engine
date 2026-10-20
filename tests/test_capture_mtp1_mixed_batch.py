"""M1 tests: the M1 helper must capture a marked prefill row in a MIXED MTP3 batch.

Scope and honesty notes
-----------------------
* CPU only, numpy/stdlib only (no torch, no GPU, no engine, no network).
* The batch arrays are built here by `_mixed_layout`, which reproduces the *runner's own*
  construction, cited line by line in its docstring:
    - `reports/s3/staged/model_runner.py:1185-1216, 1234-1256, 1294-1305, 1327-1364`
      (the runner that is actually mounted; sha256 0baf739d...fca30)
    - `reports/s0/installed-source/.../vllm/v1/worker/gpu/input_batch.py`
      (`_combine_sampled_and_draft_tokens_kernel`, `combine_sampled_and_draft_tokens`)
  This proves SELECTION/ALIGNMENT LOGIC over the arrays the runner hands the hook. It is not
  evidence about GPU/torch behaviour, CUDA graphs, or any live engine path.
* The 12-13 real MTP3 records of the PM R2 window are read READ-ONLY (in-memory, from the PM
  receipt tarball) to prove the record schema stays compatible.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys
import tarfile

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
R2_RECEIPTS = ROOT / "reports/pm/mtp-20261011/remote-receipts-r2.tar.gz"

MARKER_PREFIX = "cmpl-qfn-"
MARKED = "cmpl-qfn-dec-0"
# fields of `build_record_meta` that are written as npz arrays rather than meta_json scalars
ARRAY_KEYS = {"vector_bf16", "vector_fp32", "verbalizer_ids", "verbalizer_logits_fp32",
              "runtime_logits_row_fp32"}
# num_new_sampled_tokens_per_step: model_states/interface.py:232 default 1, never overridden in
# this build (verified by grepping the installed tree for any other assignment).
BONUS = 1
MAX_NUM_SEQS = 8  # the live deployment's --max_num_seqs


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mtp1():
    """The M1 helper under test."""
    return _load("qfn_capture_runtime_mtp1", "engine/qfn_capture_runtime_production.py")


@pytest.fixture(scope="module")
def s3():
    """The accepted S3 helper (unchanged): used to reproduce the live failure."""
    return _load("qfn_capture_runtime_s3", "tests/reference/s3.py")


# ======================================================================================
# the runner's own batch construction, reproduced
# ======================================================================================

def _entry(*, req_id, prompt_len, computed, scheduled, drafts):
    return dict(req_id=req_id, prompt_len=prompt_len, computed=computed,
                scheduled=scheduled, drafts=drafts)


def _mixed_layout(
    *,
    mark_position: int = 0,
    n_decodes: int = 7,
    drafts: int = 3,
    mark_prompt_len: int = 9,
    mark_computed: int = 0,
    mark_scheduled: int | None = None,
    mark_own_drafts: int = 0,
    mark_req_id: str = MARKED,
    extra_marked: int = 0,
    state_permutation: bool = True,
):
    """Build `select_terminal_rows` kwargs exactly as the mounted runner does for a mixed batch.

    Construction, with the runner lines it mirrors:

    * per-request draft counts come from the scheduler per request, and a request WITHOUT drafts
      is a legitimate zero entry (model_runner.py:1185-1204, `len(draft_tokens.get(req_id, ()))`).
    * `num_logits_i = drafts_i + num_new_sampled_tokens_per_step` (model_runner.py:1208).
    * `cu_num_logits_np = [0] + cumsum(num_logits)` (model_runner.py:1213-1215).
    * `query_start_loc_np = [0] + cumsum(num_scheduled_tokens)` (model_runner.py:1237-1239).
    * the triton kernel places request i's logits rows at token positions
      `[query_end - num_logits_i, query_end)`, i.e.
      `logits_indices[cu[i] + k] = query_start_loc[i+1] - num_logits_i + k`
      (input_batch.py:_combine_sampled_and_draft_tokens_kernel).
    * `num_computed_tokens_np = req_states.num_computed_tokens_np[idx_mapping_np]`
      (model_runner.py:1308) and `prompt_len_np` is indexed through `idx_mapping_np`
      (the helper reads `req_states.prompt_len.np[idx]`).
    * `is_prefilling` is a property of the request state, not of the row
      (`batch_req_state.is_prefilling_np`; model_runner.py:1348).

    A decode entry is a request whose prompt is already computed (computed == prompt_len) and
    whose query is `drafts + BONUS` tokens long (the runner asserts
    `num_scheduled_tokens >= num_logits`, model_runner.py:1212).
    """
    entries: list[dict] = []
    gen = 0
    for pos in range(n_decodes + 1):
        if pos == mark_position:
            entries.append(_entry(
                req_id=mark_req_id,
                prompt_len=mark_prompt_len,
                computed=mark_computed,
                scheduled=(mark_scheduled if mark_scheduled is not None
                           else mark_prompt_len - mark_computed),
                drafts=mark_own_drafts,
            ))
        else:
            entries.append(_entry(req_id=f"cmpl-gen{gen}-0", prompt_len=5, computed=5,
                                  scheduled=drafts + BONUS, drafts=drafts))
            gen += 1
    for j in range(extra_marked):
        entries.append(_entry(req_id=f"cmpl-qfn-dec-extra{j}-0", prompt_len=6, computed=0,
                              scheduled=6, drafts=0))

    n = len(entries)
    scheduled = np.asarray([e["scheduled"] for e in entries], dtype=np.int32)
    own_logits = np.asarray([e["drafts"] + BONUS for e in entries], dtype=np.int32)
    query_start_loc = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(scheduled, out=query_start_loc[1:])
    cu = np.zeros(n + 1, dtype=np.int32)
    cu[0] = 0
    np.cumsum(own_logits, out=cu[1:])
    logits_indices = np.empty(int(cu[-1]), dtype=np.int64)
    for i in range(n):
        for k in range(int(own_logits[i])):
            logits_indices[int(cu[i]) + k] = int(query_start_loc[i + 1]) - int(own_logits[i]) + k

    # request-state indices (the runner's idx_mapping maps batch row -> req_states row and is NOT
    # the identity in general; use a permutation so a helper that ignores it fails).
    states = list(range(n))
    if state_permutation:
        states = list(reversed(states))
    idx_mapping = np.asarray(states, dtype=np.intp)
    n_states = max(states) + 1
    prompt_len_np = np.zeros(n_states, dtype=np.int32)
    prefill_len_np = np.zeros(n_states, dtype=np.int32)
    for i, st in enumerate(states):
        prompt_len_np[st] = entries[i]["prompt_len"]
        prefill_len_np[st] = entries[i]["prompt_len"]
    num_computed = np.asarray([e["computed"] for e in entries], dtype=np.int32)
    is_prefilling = num_computed < prefill_len_np[idx_mapping]

    kw = dict(
        req_ids=[e["req_id"] for e in entries],
        idx_mapping_np=idx_mapping,
        num_computed_tokens_np=num_computed,
        num_scheduled_tokens=scheduled,
        prompt_len_np=prompt_len_np,
        prefill_len_np=prefill_len_np[idx_mapping],
        is_prefilling_np=is_prefilling,
        query_start_loc_np=query_start_loc,
        cu_num_logits_np=cu,
        num_draft_tokens=int(own_logits.sum() - n),      # runner's total_num_draft_tokens
        num_draft_tokens_per_req=np.asarray([e["drafts"] for e in entries], dtype=np.int32),
        logits_indices_np=logits_indices,
        marker_prefix=MARKER_PREFIX,
    )
    return kw


def _check_layout_matches_the_runner(kw) -> None:
    """Self-check: the synthetic arrays satisfy the runner's own guarantees."""
    cu = kw["cu_num_logits_np"]
    q = kw["query_start_loc_np"]
    li = kw["logits_indices_np"]
    n = len(kw["req_ids"])
    assert cu[0] == 0 and bool((np.diff(cu) >= 1).all())
    assert len(li) == int(cu[-1])
    for i in range(n):
        own = int(cu[i + 1] - cu[i])
        assert own == int(kw["num_draft_tokens_per_req"][i]) + BONUS
        for k in range(own):
            assert int(li[int(cu[i]) + k]) == int(q[i + 1]) - own + k, "kernel position rule"
        # the bonus (last sampled) row is the tail of the request's query
        assert int(li[int(cu[i + 1]) - 1]) == int(q[i + 1]) - 1


def _marked_position(kw) -> int:
    return kw["req_ids"].index(MARKED)


# ======================================================================================
# 1. the live failure: the accepted helper refuses the marked row in a mixed batch
# ======================================================================================

@pytest.mark.parametrize("mark_position", range(MAX_NUM_SEQS))
def test_accepted_helper_refuses_the_marked_row_in_a_mixed_batch(s3, mark_position):
    """RED for the live bug (R2 504): a 0-draft marked row is refused because the BATCH has drafts.

    This is the documented live failure of `tests/reference/s3.py:267-270`
    (`spec_decode_unsupported` on `num_draft_tokens != 0`).
    """
    kw = _mixed_layout(mark_position=mark_position)
    _check_layout_matches_the_runner(kw)
    i = _marked_position(kw)
    assert int(kw["num_draft_tokens_per_req"][i]) == 0    # the marked row itself has NO drafts
    assert kw["num_draft_tokens"] > 0                     # ... but the batch does
    rows, skips = s3.select_terminal_rows(**kw)
    assert rows == []
    assert [(s["batch_index"], s["reason"]) for s in skips] == [(i, "spec_decode_unsupported")]


# ======================================================================================
# 2. GREEN: the M1 helper captures it, in every row order
# ======================================================================================

@pytest.mark.parametrize("mark_position", range(MAX_NUM_SEQS))
def test_mtp1_helper_captures_the_marked_row_in_every_mixed_row_order(mtp1, mark_position):
    kw = _mixed_layout(mark_position=mark_position)
    _check_layout_matches_the_runner(kw)
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert skips == []
    assert len(rows) == 1
    row = rows[0]
    i = _marked_position(kw)
    cu, q, li = kw["cu_num_logits_np"], kw["query_start_loc_np"], kw["logits_indices_np"]

    assert row["req_id"] == MARKED
    assert row["batch_index"] == i
    # the row's logits row is its OWN single (bonus) row inside the batch-wide cu layout
    assert row["logits_row_index"] == int(cu[i])
    assert int(cu[i + 1]) - int(cu[i]) == 1
    # the logits index the capture will read is this request's query tail == prompt terminal
    assert int(li[row["logits_row_index"]]) == int(q[i + 1]) - 1
    assert row["query_start"] == int(q[i]) and row["query_end"] == int(q[i + 1])
    assert row["terminal_position"] == row["prompt_len"] - 1
    assert row["computed_tokens_before_step"] + row["num_scheduled_tokens"] == row["prompt_len"]


@pytest.mark.parametrize("drafts", [1, 2, 3, 4, 5, 6, 7])
def test_mtp1_helper_captures_with_1_to_7_draft_decode_neighbours(mtp1, drafts):
    for mark_position in range(min(drafts, MAX_NUM_SEQS - 1) + 1):
        kw = _mixed_layout(mark_position=mark_position, n_decodes=MAX_NUM_SEQS - 1, drafts=drafts)
        _check_layout_matches_the_runner(kw)
        rows, skips = mtp1.select_terminal_rows(**kw)
        assert skips == [], (drafts, mark_position)
        assert len(rows) == 1 and rows[0]["req_id"] == MARKED
        i = _marked_position(kw)
        assert rows[0]["logits_row_index"] == int(kw["cu_num_logits_np"][i])


def test_selected_vector_index_is_the_prompt_terminal_token_slot(mtp1):
    """The index the capture reads for the hidden vector is the marked request's own query tail.

    `hidden_states[k]` is indexed by TOKEN position (`sample_hidden_states =
    hidden_states[input_batch.logits_indices]`, model_runner.py:1403), so the vector must be read
    at `logits_indices[logits_row_index]`, i.e. inside `[query_start, query_end)` and equal to
    `query_end - 1`. Reading it at the logits ROW index instead would give a different slot.
    """
    kw = _mixed_layout(mark_position=3)
    rows, _ = mtp1.select_terminal_rows(**kw)
    row = rows[0]
    li = kw["logits_indices_np"]
    token_index = int(li[row["logits_row_index"]])
    assert token_index == row["query_end"] - 1
    assert row["query_start"] <= token_index < row["query_end"]
    # and it is the LAST scheduled token of this request's final prompt chunk, i.e. the terminal
    assert row["computed_tokens_before_step"] + row["num_scheduled_tokens"] - 1 == row["terminal_position"]
    assert token_index != row["logits_row_index"]       # token index != logits row index here


# ======================================================================================
# 3. chunked decision prefill inside a mixed batch
# ======================================================================================

def test_mtp1_nonfinal_chunk_in_a_mixed_batch_is_skipped_and_not_final(mtp1):
    kw = _mixed_layout(mark_prompt_len=10, mark_computed=0, mark_scheduled=6, mark_position=2)
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert rows == []
    assert [s["reason"] for s in skips] == ["chunk_not_final"]
    assert skips[0].get("final_refusal") is not True     # the request can still be captured later


def test_mtp1_final_chunk_in_a_mixed_batch_is_captured(mtp1):
    kw = _mixed_layout(mark_prompt_len=10, mark_computed=6, mark_scheduled=4, mark_position=2)
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert skips == []
    assert len(rows) == 1
    assert rows[0]["terminal_position"] == 9
    assert int(kw["logits_indices_np"][rows[0]["logits_row_index"]]) == rows[0]["query_end"] - 1


def test_mtp1_earlier_chunk_refusal_can_be_followed_by_a_captured_final_chunk(mtp1):
    first = _mixed_layout(mark_prompt_len=10, mark_computed=0, mark_scheduled=6, mark_position=1)
    rows1, skips1 = mtp1.select_terminal_rows(**first)
    second = _mixed_layout(mark_prompt_len=10, mark_computed=6, mark_scheduled=4, mark_position=1)
    rows2, skips2 = mtp1.select_terminal_rows(**second)
    assert rows1 == [] and [s["reason"] for s in skips1] == ["chunk_not_final"]
    assert skips1[0].get("final_refusal") is not True
    assert len(rows2) == 1 and skips2 == []


# ======================================================================================
# 4. fail-closed refusals (each must refuse, and say why)
# ======================================================================================

def test_mtp1_refuses_a_marked_row_that_itself_has_drafts(mtp1):
    kw = _mixed_layout(mark_own_drafts=3, mark_position=2)
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert rows == []
    assert [s["reason"] for s in skips] == ["own_row_has_draft_tokens"]
    assert skips[0]["final_refusal"] is True


def test_mtp1_refuses_when_per_request_draft_counts_are_unavailable(mtp1):
    kw = _mixed_layout(mark_position=2)
    kw["num_draft_tokens_per_req"] = None      # batch has drafts, per-request info missing
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert rows == []
    assert [s["reason"] for s in skips] == ["draft_info_unavailable"]
    assert skips[0]["final_refusal"] is True


def test_mtp1_refuses_a_logits_span_that_is_not_one(mtp1):
    """Tampered `cu`: the marked row claims two logits rows -> refuse (fail closed)."""
    kw = _mixed_layout(mark_position=2)
    i = _marked_position(kw)
    cu = kw["cu_num_logits_np"].copy()
    cu[i + 1:] += 1                # the marked request's span becomes 2, everything after shifts
    kw["cu_num_logits_np"] = cu
    kw["logits_indices_np"] = np.concatenate([kw["logits_indices_np"], [0]])
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert rows == []
    assert [s["reason"] for s in skips] == ["logits_span_not_one"]
    assert skips[0]["final_refusal"] is True


def test_mtp1_refuses_a_non_zero_starting_or_non_monotonic_cu(mtp1):
    kw = _mixed_layout(mark_position=2)
    cu = kw["cu_num_logits_np"].copy()
    cu[0] = 1
    kw["cu_num_logits_np"] = cu
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert rows == [] and [s["reason"] for s in skips] == ["cu_num_logits_invalid"]

    kw2 = _mixed_layout(mark_position=2)
    cu2 = kw2["cu_num_logits_np"].copy()
    cu2[3], cu2[4] = cu2[4], cu2[3]
    kw2["cu_num_logits_np"] = cu2
    rows2, skips2 = mtp1.select_terminal_rows(**kw2)
    assert rows2 == [] and [s["reason"] for s in skips2] == ["cu_num_logits_invalid"]


def test_mtp1_refuses_a_tampered_logits_index(mtp1):
    kw = _mixed_layout(mark_position=2)
    i = _marked_position(kw)
    li = kw["logits_indices_np"].copy()
    li[int(kw["cu_num_logits_np"][i])] = int(li[int(kw["cu_num_logits_np"][i])]) - 1
    kw["logits_indices_np"] = li
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert rows == []
    assert [s["reason"] for s in skips] == ["logits_index_mismatch"]
    assert skips[0]["final_refusal"] is True


def test_mtp1_missing_cu_while_the_batch_has_drafts_is_refused(mtp1):
    """No `cu` -> the row alignment cannot be proven -> refuse, never infer it."""
    kw = _mixed_layout(mark_position=2)
    kw["cu_num_logits_np"] = None
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert rows == []
    assert [s["reason"] for s in skips] == ["alignment_unverifiable"]
    assert skips[0]["final_refusal"] is True


def test_mtp1_missing_logits_indices_while_the_batch_has_drafts_is_refused(mtp1):
    kw = _mixed_layout(mark_position=2)
    kw["logits_indices_np"] = None
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert rows == []
    assert [s["reason"] for s in skips] == ["alignment_unverifiable"]
    assert skips[0]["final_refusal"] is True


# ======================================================================================
# 5. two marked requests in one batch
# ======================================================================================

def test_mtp1_two_marked_rows_in_a_mixed_batch_are_both_captured(mtp1):
    kw = _mixed_layout(mark_position=1, extra_marked=1)
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert skips == []
    assert [r["req_id"] for r in rows] == [MARKED, "cmpl-qfn-dec-extra0-0"]
    cu, li = kw["cu_num_logits_np"], kw["logits_indices_np"]
    for r in rows:
        i = kw["req_ids"].index(r["req_id"])
        assert r["logits_row_index"] == int(cu[i])
        assert int(li[r["logits_row_index"]]) == r["query_end"] - 1


def test_mtp1_two_marked_rows_one_refused_one_captured(mtp1):
    kw = _mixed_layout(mark_position=1, extra_marked=1, mark_own_drafts=3)
    rows, skips = mtp1.select_terminal_rows(**kw)
    assert [r["req_id"] for r in rows] == ["cmpl-qfn-dec-extra0-0"]
    assert [(s["req_id"], s["reason"], s.get("final_refusal")) for s in skips] == [
        (MARKED, "own_row_has_draft_tokens", True)
    ]


# ======================================================================================
# 6. no-draft behaviour must be IDENTICAL to the accepted helper
# ======================================================================================

def _no_draft_batches():
    yield dict(
        req_ids=[MARKED],
        idx_mapping_np=np.asarray([0], dtype=np.intp),
        num_computed_tokens_np=np.asarray([0], dtype=np.int32),
        num_scheduled_tokens=np.asarray([5], dtype=np.int32),
        prompt_len_np=np.asarray([5], dtype=np.int32),
        prefill_len_np=np.asarray([5], dtype=np.int32),
        is_prefilling_np=np.asarray([True]),
        query_start_loc_np=np.asarray([0, 5], dtype=np.int32),
        cu_num_logits_np=np.asarray([0, 1], dtype=np.int32),
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        logits_indices_np=None,
    )
    yield dict(
        req_ids=[MARKED, "cmpl-gen0-0"],
        idx_mapping_np=np.asarray([1, 0], dtype=np.intp),
        num_computed_tokens_np=np.asarray([6, 0], dtype=np.int32),
        num_scheduled_tokens=np.asarray([4, 3], dtype=np.int32),
        prompt_len_np=np.asarray([3, 10], dtype=np.int32),
        prefill_len_np=np.asarray([10, 3], dtype=np.int32),
        is_prefilling_np=np.asarray([True, False]),
        query_start_loc_np=np.asarray([0, 4, 7], dtype=np.int32),
        cu_num_logits_np=None,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        logits_indices_np=np.asarray([3, 6], dtype=np.int64),
    )
    yield dict(
        req_ids=[MARKED],
        idx_mapping_np=np.asarray([0], dtype=np.intp),
        num_computed_tokens_np=np.asarray([0], dtype=np.int32),
        num_scheduled_tokens=np.asarray([4], dtype=np.int32),
        prompt_len_np=np.asarray([10], dtype=np.int32),
        prefill_len_np=np.asarray([10], dtype=np.int32),
        is_prefilling_np=np.asarray([True]),
        query_start_loc_np=np.asarray([0, 4], dtype=np.int32),
        cu_num_logits_np=np.asarray([0, 1], dtype=np.int32),
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        logits_indices_np=np.asarray([3], dtype=np.int64),
    )


def test_no_draft_batches_are_identical_to_the_accepted_helper(mtp1, s3):
    for kw in _no_draft_batches():
        rows_new, skips_new = mtp1.select_terminal_rows(**kw)
        rows_old, skips_old = s3.select_terminal_rows(**kw)
        assert rows_new == rows_old, kw["req_ids"]
        assert skips_new == skips_old
        for r_n, r_o in zip(rows_new, rows_old):
            assert list(r_n) == list(r_o)          # same row keys
            assert set(r_n) == set(r_o)


def test_mtp1_record_meta_is_unchanged_for_a_no_draft_row(mtp1, s3, tmp_path):
    """The written record fields must stay identical (the live gateway decodes with
    tolerate_unknown_meta_fields=false, so an added npz meta key would be a HARD failure)."""
    kw = _mixed_layout(mark_position=0, n_decodes=0)
    rows, _ = mtp1.select_terminal_rows(**kw)
    row = rows[0]
    common = dict(
        row=row,
        logits_index=8,
        raw_vector_dtype="torch.bfloat16",
        vector_bf16=np.arange(2560, dtype=np.uint16),
        vector_fp32=np.zeros(2560, dtype=np.float32),
        verbalizer_ids=list(range(1000, 1024)),
        verbalizer_logits_fp32=np.zeros(24, dtype=np.float32),
        runtime_argmax_token_id=7,
        runtime_max_logit_fp32=1.5,
        runtime_logits_row_fp32=None,
        logits_captured=True,
        logits_absent_reason=None,
        token_hash="ab" * 32,
        chunk_token_sha256=None,
        token_hash_scope="prompt",
        token_hash_from_engine_state=True,
        head_info=dict(scale=1.0, soft_cap=None, org_vocab_size=151936, head_dtype="torch.bfloat16",
                       tp_size=2, tp_rank=0, world_size=2, cudagraph_mode="FULL_DECODE_ONLY",
                       device_name="fixture", vllm_version="fixture"),
    )
    new = mtp1.build_record_meta(**common)
    old = s3.build_record_meta(**common)
    assert set(new) == set(old), "record meta key set must not change"
    shared = {k: v for k, v in new.items() if k not in ("capture_wall_ns", "capture_monotonic_ns")}
    shared_old = {k: v for k, v in old.items() if k not in ("capture_wall_ns", "capture_monotonic_ns")}
    for k in shared:
        if isinstance(shared[k], np.ndarray):
            assert np.array_equal(shared[k], shared_old[k]), k
        else:
            assert shared[k] == shared_old[k], k


# ======================================================================================
# 7. real R2 MTP3 records: schema compatibility (READ-ONLY, in memory)
# ======================================================================================







