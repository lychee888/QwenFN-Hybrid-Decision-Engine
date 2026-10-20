"""WIDE1 slice 8: the engine-side wide capture producer (numpy + an explicitly fake torch).

The real helper runs inside the vLLM process. Here it is loaded from source and driven with a
minimal fake torch/tensor API, so the ACTUAL ``_capture_one`` producer path is exercised: row
validation, ``logits[j]`` selection, the all-256 catalog index_select, the record writer (arrays
vs scalar metadata) and the storage reservation size seam.

No GPU, no engine, no inference. This is a producer-seam test, not a live-parity claim.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys

import numpy as np
import pytest

from qfn_decision import backend, features, protocol, wide

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "wide-label-catalog.json"
CATALOG = wide.WideCatalog.from_file(CATALOG_PATH)
UUID32 = "0123456789abcdef0123456789abcdef"
WIDE_ID = f"cmpl-qfn-wide-32-{UUID32}-0"
PLAIN_ID = "cmpl-qfn-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-0"

ENGINE_REL = "engine/qfn_capture_runtime_production.py"


def _engine():
    name = "qfn_capture_runtime_wide1"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / ENGINE_REL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- fake torch


class _DType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


class _FakeTensor:
    """Just enough of the torch tensor API for ``_capture_one`` (numpy underneath)."""

    def __init__(self, array, kind: str = "f32") -> None:
        self._np = np.asarray(array)
        self._kind = kind

    # -- dtype/view/device
    @property
    def dtype(self) -> _DType:
        return _DType(self._kind)

    @property
    def shape(self):
        return self._np.shape

    @property
    def device(self):
        return "cuda:0"

    def reshape(self, *shape):
        return _FakeTensor(self._np.reshape(*shape), self._kind)

    def detach(self):
        return self

    def view(self, dtype):
        if dtype.name == "int16" and self._kind == "bf16":
            return _FakeTensor(_bf16_bits(self._np).view(np.int16), "i16")
        if dtype.name in ("int16", "i16") and self._kind == "i16":
            return self
        raise TypeError(f"fake torch cannot view {self._kind} as {dtype.name}")

    def to(self, dtype):
        if isinstance(dtype, _DType):
            target = dtype.name
        else:  # .to(device)
            return self
        if target in ("float32", "f32"):
            return _FakeTensor(self._np.astype(np.float32), "f32")
        if target in ("bfloat16", "bf16"):
            return _FakeTensor(_bf16_roundtrip(self._np), "bf16")
        if target in ("int64", "long"):
            return _FakeTensor(self._np.astype(np.int64), "i64")
        raise TypeError(f"fake torch cannot convert to {target}")

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self._np)

    # -- indexing / math
    def __getitem__(self, item):
        out = self._np[item]
        if self._kind == "bf16":
            return _FakeTensor(out, "bf16")
        return _FakeTensor(out, self._kind)

    def index_select(self, dim, index):
        idx = index.numpy().astype(np.int64) if isinstance(index, _FakeTensor) else np.asarray(index)
        return _FakeTensor(np.take(self._np, idx, axis=dim), self._kind)

    def argmax(self):
        return _FakeScalar(int(np.argmax(self._np)))

    def max(self):
        return _FakeScalar(self._np.max())

    def item(self):
        return self._np.item()

    def tolist(self):
        return self._np.tolist()

    def __int__(self):
        return int(self._np)


class _FakeScalar:
    def __init__(self, value) -> None:
        self.value = value

    def item(self):
        return self.value


def _bf16_bits(array) -> np.ndarray:
    u = np.ascontiguousarray(np.asarray(array, dtype=np.float32)).view(np.uint32).copy()
    is_nan = ((u & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & ((u & np.uint32(0x7FFFFF)) != 0)
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = (u + bias) >> np.uint32(16)
    rounded = np.where(is_nan, (u >> np.uint32(16)) | np.uint32(0x0040), rounded)
    return rounded.astype(np.uint16)


def _bf16_roundtrip(array) -> np.ndarray:
    bits = _bf16_bits(array)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


class _FakeTorch:
    bfloat16 = _DType("bfloat16")
    float32 = _DType("float32")
    int16 = _DType("int16")
    int64 = _DType("int64")
    long = _DType("long")
    float64 = _DType("float64")

    class cuda:  # noqa: N801 - mirrors torch.cuda
        @staticmethod
        def is_current_stream_capturing() -> bool:
            return False

    @staticmethod
    def as_tensor(values, dtype=None, device=None):
        arr = np.asarray(values)
        kind = "i64" if (dtype is not None and dtype.name in ("int64", "long")) else "f32"
        return _FakeTensor(arr.astype(np.int64) if kind == "i64" else arr.astype(np.float32), kind)

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return _FakeTorch.as_tensor(values, dtype=dtype, device=device)


# ---------------------------------------------------------------- doubles


class _Buf:
    def __init__(self, np_array) -> None:
        self.np = np_array


class _AllTokenIds:
    def __init__(self, rows) -> None:
        self._uva_buf = _Buf(np.asarray(rows, dtype=np.int64))


class _ReqStates:
    def __init__(self, prompt_lens, token_rows) -> None:
        self.prompt_len = _Buf(np.asarray(prompt_lens, dtype=np.int64))
        self.all_token_ids = _AllTokenIds(token_rows)


class _LmHead:
    tp_rank = 0
    tp_size = 1
    shard_indices = None
    weight = _FakeTensor(np.zeros((4, 4), dtype=np.float32))


class _LogitsProcessor:
    scale = 1.0
    soft_cap = None
    org_vocab_size = 300000
    head_dtype = "bfloat16"
    use_all_gather = False


class _Model:
    def __init__(self) -> None:
        self.lm_head = _LmHead()
        self.logits_processor = _LogitsProcessor()


class _Compilation:
    cudagraph_mode = "FULL_DECODE_ONLY"


class _VllmConfig:
    class model_config:
        model = "fake-model"

    compilation_config = _Compilation()


class _Runner:
    def __init__(self, prompt_lens, token_rows) -> None:
        self.req_states = _ReqStates(prompt_lens, token_rows)
        self.model = _Model()
        self.vllm_config = _VllmConfig()


class _InputBatch:
    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


def _batch(req_ids, prompt_lens, scheduled, cu=None, logits_indices=None, drafts=None,
           batch_drafts=0, qsl=None, prefill=None, prefilling=None):
    n = len(req_ids)
    qsl = qsl if qsl is not None else np.concatenate([[0], np.cumsum(scheduled)]).astype(np.int64)
    return _InputBatch(
        req_ids=list(req_ids),
        idx_mapping_np=np.arange(n, dtype=np.int64),
        num_computed_tokens_np=np.zeros(n, dtype=np.int64),
        num_scheduled_tokens=np.asarray(scheduled, dtype=np.int64),
        prompt_len_np=np.asarray(prompt_lens, dtype=np.int64),
        prefill_len_np=np.asarray(prefill if prefill is not None else prompt_lens, dtype=np.int64),
        is_prefilling_np=np.asarray(prefilling if prefilling is not None else [True] * n, dtype=bool),
        query_start_loc_np=np.asarray(qsl, dtype=np.int64),
        cu_num_logits_np=None if cu is None else np.asarray(cu, dtype=np.int64),
        num_draft_tokens_per_req=None if drafts is None else np.asarray(drafts, dtype=np.int64),
        num_draft_tokens=batch_drafts,
        logits_indices=None if logits_indices is None else _FakeTensor(np.asarray(logits_indices), "i64"),
        positions=_FakeTensor(np.arange(int(qsl[-1]), dtype=np.int64), "i64"),
        input_ids=_FakeTensor(np.arange(int(qsl[-1]), dtype=np.int64), "i64"),
    )


# ---------------------------------------------------------------- config


def _wide_block(**over) -> dict:
    block = {
        "enabled": True,
        "catalog_path": str(CATALOG_PATH),
        "catalog_file_sha256": __import__("hashlib").sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        "catalog_digest": CATALOG.digest,
        "tokenizer_revision": CATALOG.tokenizer_revision,
        "template_digest": wide.WIDE_TEMPLATE_CONTENT_SHA256,
    }
    block.update(over)
    return block


def _capture_config(tmp_path, *, wide_block="default", **over) -> pathlib.Path:
    cfg = {
        "capture_dir": str(tmp_path / "capture"),
        "marker_prefix": "cmpl-qfn-",
        "verbalizers": {"names": [f"slot{i}" for i in range(24)],
                        "ids": list(range(1000, 1024))},
        "capture_fp32_ref": True,
        "tp_export_rows": False,
        "max_records": 100,
        "max_bytes": 100_000_000,
    }
    if wide_block == "default":
        cfg["wide_choice"] = _wide_block()
    elif wide_block is not None:
        cfg["wide_choice"] = wide_block
    cfg.update(over)
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _state(mod, path):
    cfg = mod.CaptureConfig.from_file(path)
    return mod._CaptureState(enabled=True, config=cfg)


def _logits_rows(n_rows: int = 4, vocab: int = 300000) -> np.ndarray:
    arr = np.linspace(-1.0, 1.0, n_rows * vocab, dtype=np.float32).reshape(n_rows, vocab)
    return arr


def _run_one(tmp_path, monkeypatch, *, ids=(WIDE_ID,), wide_block="default", win=7):
    mod = _engine()
    monkeypatch.setattr(mod, "torch", _FakeTorch)
    cfg_path = _capture_config(tmp_path, wide_block=wide_block)
    state = _state(mod, cfg_path)
    token_rows = np.array([[5, 6, 7, 8], [5, 6, 7, 8]], dtype=np.int64)
    runner = _Runner([4, 4], token_rows)
    hidden = _FakeTensor(np.linspace(-1.0, 1.0, 4 * 2560, dtype=np.float32).reshape(4, 2560), "bf16")
    logits_np = _logits_rows(2)
    logits_np[0, CATALOG.token_ids[win]] = 50.0
    logits = _FakeTensor(logits_np, "f32")
    batch = _batch(ids, [4] * len(ids), [4] * len(ids), cu=[0, 1] + [1] * (len(ids) - 1),
                   logits_indices=[3, 7], drafts=[0] * len(ids))
    mod._capture(state, runner, hidden, logits, batch)
    return mod, state, logits_np, pathlib.Path(state.config.capture_dir)


def _records(capture_dir: pathlib.Path):
    return sorted(capture_dir.glob("*.npz"))


# ---------------------------------------------------------------- catalog/config


def test_engine_and_package_derive_the_same_catalog_digest():
    mod = _engine()
    resolver = getattr(mod, "wide_catalog_digest", None)
    assert resolver is not None, "the engine helper must expose the canonical catalog digest"
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert resolver(payload) == CATALOG.digest == wide.catalog_digest(payload)


def test_engine_capture_config_validates_the_wide_block(tmp_path):
    mod = _engine()
    ok = mod.CaptureConfig.from_file(_capture_config(tmp_path))
    assert ok.wide_choice is not None and ok.wide_choice.enabled
    assert ok.wide_choice.catalog_digest == CATALOG.digest
    assert ok.wide_choice.token_ids[:16] == tuple(range(32, 48))
    for mutate in (
        {"catalog_file_sha256": "0" * 64},
        {"catalog_digest": "0" * 64},
        {"tokenizer_revision": "0" * 40},
        {"template_digest": "short"},
        {"catalog_path": "relative.json"},
        {"catalog_path": str(tmp_path / "missing.json")},
        {"enabled": "yes"},
        {"unexpected": 1},
    ):
        with pytest.raises(Exception):
            mod.CaptureConfig.from_file(_capture_config(tmp_path, wide_block=_wide_block(**mutate)))


def test_engine_capture_config_disabled_reads_no_catalog(tmp_path):
    mod = _engine()
    cfg = mod.CaptureConfig.from_file(_capture_config(
        tmp_path, wide_block={"enabled": False, "catalog_path": str(tmp_path / "nope.json")}))
    assert cfg.wide_choice is not None and not cfg.wide_choice.enabled
    assert cfg.wide_choice.token_ids is None


# ---------------------------------------------------------------- producer


def test_producer_writes_all_256_catalog_logits_from_logits_j(tmp_path, monkeypatch):
    mod, state, logits_np, cap = _run_one(tmp_path, monkeypatch)
    files = _records(cap)
    assert len(files) == 1
    with np.load(io.BytesIO(files[0].read_bytes()), allow_pickle=False) as z:
        ids = z["wide_ids"]
        logits = z["wide_logits_fp32"]
    assert ids.dtype == np.int64 and ids.shape == (256,)
    assert np.array_equal(ids, np.asarray(CATALOG.token_ids, dtype=np.int64))
    assert logits.dtype == np.float32 and logits.shape == (256,)
    expected = logits_np[0][np.asarray(CATALOG.token_ids, dtype=np.int64)].astype(np.float32)
    assert np.allclose(logits, expected, rtol=0, atol=0)
    meta = json.loads(bytes(np.asarray(
        np.load(io.BytesIO(files[0].read_bytes()), allow_pickle=False)["meta_json"],
        dtype=np.uint8)).decode("utf-8"))
    assert meta["wide_schema"] == wide.WIDE_CAPTURE_SCHEMA
    assert meta["wide_catalog_digest"] == CATALOG.digest
    assert meta["wide_n_options"] == 32
    assert meta["wide_request_uuid"] == UUID32
    assert meta["wide_template_digest"] == wide.WIDE_TEMPLATE_CONTENT_SHA256
    # the base 24-slot fields are still written unchanged (arrays live in the npz, not in meta)
    assert meta["logits_captured"] is True
    with np.load(io.BytesIO(files[0].read_bytes()), allow_pickle=False) as z:
        assert z["verbalizer_ids"].shape == (24,)


def test_producer_output_is_accepted_by_the_gateway_wide_reader(tmp_path, monkeypatch):
    mod, state, logits_np, cap = _run_one(tmp_path, monkeypatch, win=31)
    blob = _records(cap)[0].read_bytes()
    capture = wide.decode_wide_capture(
        blob,
        catalog=CATALOG,
        expected_request_uuid=UUID32,
        expected_engine_request_id=WIDE_ID,
        expected_prompt_token_sha256=features.token_sequence_sha256([5, 6, 7, 8]),
        expected_prompt_len=4,
        n_options=32,
        not_before_wall_ns=0,
    )
    result = wide.wide_choice_result(capture.logits, wide.WideChoiceRequest.create(
        "s", "q", [protocol.ChoiceOption(key=f"k{i}", description="d") for i in range(32)]), CATALOG)
    assert result.selected_label == "AF"   # label 31, the last ACTIVE label for n=32
    # and the legacy reader must refuse the same file outright
    with pytest.raises(backend.CaptureError):
        backend.decode_record_bytes(blob)


def test_no_wide_config_writes_a_plain_legacy_record(tmp_path, monkeypatch):
    mod, state, logits_np, cap = _run_one(tmp_path, monkeypatch, wide_block=None)
    blob = _records(cap)[0].read_bytes()
    assert not backend.record_has_wide_extension(blob)
    rec, dropped = backend.decode_record_bytes(blob)
    assert rec.engine_request_id == WIDE_ID


def test_mixed_mtp_batch_captures_only_the_wide_row_and_stays_aligned(tmp_path, monkeypatch):
    mod = _engine()
    monkeypatch.setattr(mod, "torch", _FakeTorch)
    cfg_path = _capture_config(tmp_path)
    state = _state(mod, cfg_path)
    token_rows = np.array([[5, 6, 7, 8], [5, 6, 7, 8]], dtype=np.int64)
    runner = _Runner([4, 4], token_rows)
    hidden = _FakeTensor(np.linspace(-1.0, 1.0, 8 * 2560, dtype=np.float32).reshape(8, 2560), "bf16")
    logits = _FakeTensor(_logits_rows(3), "f32")
    batch = _batch([WIDE_ID, PLAIN_ID], [4, 4], [4, 1], cu=[0, 1, 2],
                   logits_indices=[3, 4], drafts=[0, 1], batch_drafts=1)
    mod._capture(state, runner, hidden, logits, batch)
    files = _records(pathlib.Path(state.config.capture_dir))
    assert len(files) == 1, "only the marked wide row is captured; ordinary traffic is untouched"
    assert backend.record_has_wide_extension(files[0].read_bytes())


def test_draft_only_row_is_refused_and_writes_no_wide_extension(tmp_path, monkeypatch):
    mod = _engine()
    monkeypatch.setattr(mod, "torch", _FakeTorch)
    state = _state(mod, _capture_config(tmp_path))
    runner = _Runner([4], np.array([[5, 6, 7, 8]], dtype=np.int64))
    hidden = _FakeTensor(np.zeros((4, 2560), dtype=np.float32), "bf16")
    logits = _FakeTensor(_logits_rows(1), "f32")
    batch = _batch([WIDE_ID], [4], [4], cu=[0, 2], logits_indices=[3, 3], drafts=[2], batch_drafts=2)
    mod._capture(state, runner, hidden, logits, batch)
    assert _records(pathlib.Path(state.config.capture_dir)) == []
    assert state.skips.get("own_row_has_draft_tokens") == 1


def test_chunked_prefill_and_prefix_cache_replay_capture_only_the_terminal_chunk(tmp_path, monkeypatch):
    mod = _engine()
    monkeypatch.setattr(mod, "torch", _FakeTorch)
    state = _state(mod, _capture_config(tmp_path))
    runner = _Runner([8], np.array([[5, 6, 7, 8, 9, 10, 11, 12]], dtype=np.int64))
    hidden = _FakeTensor(np.zeros((4, 2560), dtype=np.float32), "bf16")
    logits = _FakeTensor(_logits_rows(1), "f32")

    # chunk 1: 4 of 8 tokens -> not final
    first = _batch([WIDE_ID], [8], [4], cu=[0, 1], logits_indices=[3], drafts=[0])
    mod._capture(state, runner, hidden, logits, first)
    assert _records(pathlib.Path(state.config.capture_dir)) == []
    assert state.skips.get("chunk_not_final") == 1

    # chunk 2: the remaining 4 tokens are computed with 4 already in the prefix cache
    second = _batch([WIDE_ID], [8], [4], cu=[0, 1], logits_indices=[3], drafts=[0],
                    qsl=[0, 4], prefill=[8], prefilling=[True])
    second.num_computed_tokens_np = np.array([4], dtype=np.int64)
    for name in ("positions", "input_ids"):
        setattr(second, name, _FakeTensor(np.arange(4, 8, dtype=np.int64), "i64"))
    mod._capture(state, runner, hidden, logits, second)
    files = _records(pathlib.Path(state.config.capture_dir))
    assert len(files) == 1
    assert backend.record_has_wide_extension(files[0].read_bytes())


def test_storage_reservation_measures_the_extended_record_and_refuses_cleanly(tmp_path, monkeypatch):
    mod = _engine()
    monkeypatch.setattr(mod, "torch", _FakeTorch)
    state = _state(mod, _capture_config(tmp_path))
    runner = _Runner([4], np.array([[5, 6, 7, 8]], dtype=np.int64))
    hidden = _FakeTensor(np.zeros((4, 2560), dtype=np.float32), "bf16")
    logits = _FakeTensor(_logits_rows(1), "f32")
    batch = _batch([WIDE_ID], [4], [4], cu=[0, 1], logits_indices=[3], drafts=[0])

    class Reserve:
        def __init__(self) -> None:
            self.checked = []
            self.committed = None
            self.released = False

        def check_payload(self, size):
            self.checked.append(int(size))

        def begin_publish(self):
            pass

        def finish_publish(self):
            pass

        def commit(self, size):
            self.committed = int(size)

        def release(self):
            self.released = True

    reserve = Reserve()
    state.production = type("P", (), {"reserve_for_record": lambda self, key: reserve})()
    state.after_record_written = lambda written: False
    mod._capture(state, runner, hidden, logits, batch)
    assert reserve.checked, "the pre-publication size guard must see the extended payload"
    # 256 int64 ids + 256 float32 logits + the wide scalar block are all inside the measured size
    base_only = 256 * 4 + 256 * 8  # a legacy-equivalent payload measure would be far smaller
    assert reserve.checked[0] > base_only
    # the guard measures the npz payload + the compact metadata; the committed value is the actual
    # published size (npz + the indented JSON sidecar), so it can only be >= the estimate
    assert reserve.committed >= reserve.checked[0]
    assert not reserve.released

    class Refusing(Reserve):
        def check_payload(self, size):
            super().check_payload(size)
            raise wide.WideError("storage refused")

    refusing = Refusing()
    state.production = type("P", (), {"reserve_for_record": lambda self, key: refusing})()
    mod._capture(state, runner, hidden, logits, batch)
    assert refusing.released, "a refused publication must release its reservation"
    assert any(e["kind"] == "row_capture_failed" for e in state.errors)
    assert len(_records(pathlib.Path(state.config.capture_dir))) == 1, (
        "the earlier successful record stays; the refused one publishes no partial file")
