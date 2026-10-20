"""WIDE1 slice 6: the wide capture extension codec.

The legacy reader rejects unknown scalar meta and silently ignores unknown arrays, so appending a
dict to ``meta_json`` cannot work: the writer must move the 256-wide vectors into the NPZ as real
arrays and out of the scalar metadata, and a dedicated reader must validate every wide field.

Everything here goes through the real ``np.savez``/``np.load`` path.
"""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import time

import numpy as np
import pytest

from qfn_decision import backend, features, protocol, wide

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = wide.WideCatalog.from_file(ROOT / "configs" / "wide-label-catalog.json")
PROMPT_IDS = list(range(1000, 1030))
UUID = "0123456789abcdef0123456789abcdef"
ENGINE_ID = f"cmpl-qfn-wide-32-{UUID}-0"


def _legacy_record(engine_id: str = ENGINE_ID, *, prompt_ids=PROMPT_IDS, **over):
    return backend.make_fixture_record(
        engine_request_id=engine_id,
        prompt_ids=list(prompt_ids),
        vector_fp32=np.linspace(-0.5, 0.5, 2560, dtype=np.float32),
        slot_logits=np.linspace(-1.0, 1.0, protocol.NUM_SLOTS, dtype=np.float32),
        **over,
    )


def _legacy_blob(**kwargs) -> bytes:
    return backend.fixture_only_npz_bytes(_legacy_record(**kwargs))


def _logits256(winner: int = 3) -> np.ndarray:
    arr = np.linspace(-2.0, 2.0, 256, dtype=np.float32)
    arr[winner] = 9.0
    return arr


def _extend(blob: bytes, *, logits=None, n_options: int = 32, **over) -> bytes:
    kwargs = dict(
        catalog=CATALOG,
        template_digest=wide.WIDE_TEMPLATE_CONTENT_SHA256,
        wide_request_uuid=UUID,
        prompt_token_sha256=features.token_sequence_sha256(PROMPT_IDS),
        prompt_len=len(PROMPT_IDS),
        n_options=n_options,
        logits256=_logits256() if logits is None else logits,
    )
    kwargs.update(over)
    return wide.add_wide_extension(blob, **kwargs)


def _arrays(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _meta(blob: bytes) -> dict:
    return json.loads(bytes(np.asarray(_arrays(blob)["meta_json"], dtype=np.uint8)).decode("utf-8"))


def _reencode(blob: bytes, *, arrays: dict | None = None, meta: dict | None = None) -> bytes:
    data = _arrays(blob)
    if meta is not None:
        data["meta_json"] = np.frombuffer(
            json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"), dtype=np.uint8
        )
    for name, value in (arrays or {}).items():
        if value is None:
            data.pop(name, None)
        else:
            data[name] = np.asarray(value)
    buf = io.BytesIO()
    np.savez(buf, **data)
    return buf.getvalue()


def _decode(blob: bytes, **over):
    kwargs = dict(
        catalog=CATALOG,
        expected_request_uuid=UUID,
        expected_engine_request_id=ENGINE_ID,
        expected_prompt_token_sha256=features.token_sequence_sha256(PROMPT_IDS),
        expected_prompt_len=len(PROMPT_IDS),
        n_options=32,
        not_before_wall_ns=0,
    )
    kwargs.update(over)
    return wide.decode_wide_capture(blob, **kwargs)


# ---------------------------------------------------------------- writer


def test_writer_moves_the_wide_vectors_into_npz_arrays_and_out_of_scalar_meta():
    blob = _extend(_legacy_blob())
    arrays = _arrays(blob)
    assert arrays["wide_ids"].dtype == np.int64 and arrays["wide_ids"].shape == (256,)
    assert arrays["wide_logits_fp32"].dtype == np.float32
    assert arrays["wide_logits_fp32"].shape == (256,)
    assert np.array_equal(arrays["wide_ids"], np.asarray(CATALOG.token_ids, dtype=np.int64))
    meta = _meta(blob)
    assert meta["wide_schema"] == wide.WIDE_CAPTURE_SCHEMA
    assert meta["wide_catalog_digest"] == CATALOG.digest
    assert meta["wide_request_uuid"] == UUID
    assert meta["wide_n_options"] == 32
    # no array serialised into the scalar metadata (no 256-element lists anywhere)
    for key, value in meta.items():
        if isinstance(value, list):
            assert len(value) <= protocol.NUM_SLOTS, f"{key} looks like an array in scalar meta"
    text = json.dumps(meta)
    assert "array(" not in text and "np.float" not in text


def test_writer_preserves_base_identity_and_is_idempotent_per_content():
    before = _legacy_blob()
    after = _extend(before)
    a, b = _meta(before), _meta(after)
    for field in ("engine_request_id", "prompt_token_sha256", "prompt_len", "terminal_position"):
        assert a[field] == b[field]
    # extending twice with identical content yields identical arrays/meta (no duplicated keys)
    twice = _extend(after)
    assert np.array_equal(_arrays(twice)["wide_logits_fp32"], _arrays(after)["wide_logits_fp32"])
    assert set(_meta(twice)) == set(_meta(after))


def test_writer_refuses_bad_inputs():
    blob = _legacy_blob()
    with pytest.raises(wide.WideError):
        _extend(blob, logits=np.zeros(255, dtype=np.float32))
    with pytest.raises(wide.WideError):
        _extend(blob, logits=np.full(256, np.nan, dtype=np.float32))
    with pytest.raises(wide.WideError):
        _extend(blob, n_options=257)
    with pytest.raises(wide.WideError):
        _extend(blob, wide_request_uuid="not-a-uuid")
    with pytest.raises(wide.WideError):
        _extend(blob, prompt_token_sha256="0" * 63)
    with pytest.raises(wide.WideError):
        _extend(b"not an npz", logits=_logits256())


# ---------------------------------------------------------------- reader: happy path


def test_roundtrip_returns_the_256_wide_logits_and_identity():
    cap = _decode(_extend(_legacy_blob()))
    assert isinstance(cap.record, features.FeatureRecord)
    assert cap.wide_ids.shape == (256,)
    assert cap.logits.dtype == np.float32 and cap.logits.shape == (256,)
    assert np.array_equal(cap.wide_ids, np.asarray(CATALOG.token_ids, dtype=np.int64))
    assert cap.catalog_digest == CATALOG.digest
    assert cap.template_digest == wide.WIDE_TEMPLATE_CONTENT_SHA256
    assert cap.request_uuid == UUID
    assert cap.n_options == 32
    assert cap.engine_request_id == ENGINE_ID
    assert cap.prompt_token_sha256 == features.token_sequence_sha256(PROMPT_IDS)
    assert cap.prompt_len == len(PROMPT_IDS)
    assert cap.terminal_position == len(PROMPT_IDS) - 1
    assert cap.tp_rank == 0
    result = wide.wide_choice_result(cap.logits, _wide_request(32), CATALOG)
    assert result.selected_index == 3
    assert len(result.probabilities) == 32


def _wide_request(n: int = 32) -> wide.WideChoiceRequest:
    return wide.WideChoiceRequest.create(
        "s", "q", [protocol.ChoiceOption(key=f"k{i}", description=f"d{i}") for i in range(n)]
    )


def test_old_records_without_a_wide_extension_are_never_read_as_wide():
    legacy = _legacy_blob()
    # the legacy reader still accepts it unchanged
    rec, dropped = backend.decode_record_bytes(legacy)
    assert rec.engine_request_id == ENGINE_ID and dropped == ()
    with pytest.raises(wide.WideExtensionMissing):
        _decode(legacy)


def test_legacy_reader_refuses_a_record_carrying_the_wide_extension(monkeypatch):
    blob = _extend(_legacy_blob())
    with pytest.raises(backend.CaptureError):
        backend.decode_record_bytes(blob)
    # the tolerate-unknown hatch must NOT strip wide fields to make it decodable as legacy
    with pytest.raises(backend.CaptureError):
        backend.decode_record_bytes(blob, tolerate_unknown_meta_fields=True)


# ---------------------------------------------------------------- reader: refusals


def test_reader_refuses_missing_or_unknown_wide_metadata():
    blob = _extend(_legacy_blob())
    meta = _meta(blob)
    for key in ("wide_schema", "wide_catalog_digest", "wide_template_digest", "wide_request_uuid"):
        broken = dict(meta)
        broken.pop(key)
        with pytest.raises(wide.WideError):
            _decode(_reencode(blob, meta=broken))
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, meta={**meta, "wide_extra": 1}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, meta={**meta, "wide_schema": "qfn.wide-capture/2"}))


def test_reader_refuses_missing_unknown_or_misshaped_arrays():
    blob = _extend(_legacy_blob())
    for name in ("wide_ids", "wide_logits_fp32"):
        with pytest.raises(wide.WideError):
            _decode(_reencode(blob, arrays={name: None}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, arrays={"wide_extra": np.zeros(4, dtype=np.float32)}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, arrays={"wide_logits_fp32": np.zeros(255, dtype=np.float32)}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, arrays={"wide_logits_fp32": np.zeros(256, dtype=np.float64)}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, arrays={"wide_ids": np.zeros(256, dtype=np.int32)}))
    bad = _logits256()
    bad[7] = np.inf
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, arrays={"wide_logits_fp32": bad}))
    bad = _logits256()
    bad[9] = np.nan
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, arrays={"wide_logits_fp32": bad}))


def test_reader_refuses_reordered_changed_or_boolean_ids():
    blob = _extend(_legacy_blob())
    ids = np.asarray(CATALOG.token_ids, dtype=np.int64).copy()
    swapped = ids.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    for bad in (swapped, np.full(256, 32, dtype=np.int64), ids.astype(np.bool_)):
        with pytest.raises(wide.WideError):
            _decode(_reencode(blob, arrays={"wide_ids": bad}))


def test_reader_refuses_catalog_and_template_drift():
    blob = _extend(_legacy_blob())
    meta = _meta(blob)
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, meta={**meta, "wide_catalog_digest": "0" * 64}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, meta={**meta, "wide_template_digest": "0" * 64}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, meta={**meta, "wide_template_id": "qwen_chat_letters-v1"}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, meta={**meta, "wide_tokenizer_revision": "0" * 40}))
    # a different catalog object (changed content) must not validate this record
    payload = json.loads((ROOT / "configs" / "wide-label-catalog.json").read_text(encoding="utf-8"))
    payload["labels"] = [*payload["labels"][:-1], "JV"]
    other = wide.WideCatalog.from_dict(payload)
    with pytest.raises(wide.WideError):
        _decode(blob, catalog=other)


def test_reader_refuses_tampered_or_replayed_request_identity():
    blob = _extend(_legacy_blob())
    with pytest.raises(wide.WideError):
        _decode(blob, expected_request_uuid="f" * 32)
    with pytest.raises(wide.WideError):
        _decode(blob, expected_engine_request_id=f"cmpl-qfn-wide-32-{'f' * 32}-0")
    with pytest.raises(wide.WideError):
        _decode(blob, expected_prompt_token_sha256="a" * 64)
    with pytest.raises(wide.WideError):
        _decode(blob, expected_prompt_len=len(PROMPT_IDS) + 1)
    with pytest.raises(wide.WideError):
        _decode(blob, n_options=31)
    meta = _meta(blob)
    with pytest.raises(wide.WideError):
        _decode(_reencode(blob, meta={**meta, "wide_request_uuid": "f" * 32}))


def test_reader_refuses_wrong_terminal_position_rank_or_missing_logits():
    meta = _meta(_extend(_legacy_blob()))
    with pytest.raises(wide.WideError):
        _decode(_reencode(_extend(_legacy_blob()), meta={**meta, "terminal_position": 0}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(_extend(_legacy_blob()), meta={**meta, "tp_rank": 1}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(_extend(_legacy_blob()), meta={
            **meta, "logits_captured": False, "logits_absent_reason": "verbalizer_ids_unresolved"}))
    with pytest.raises(wide.WideError):
        _decode(_reencode(_extend(_legacy_blob()), meta={**meta, "token_hash_from_engine_state": False}))


def test_reader_refuses_a_stale_capture():
    blob = _extend(_legacy_blob())
    now = time.time_ns()
    with pytest.raises(wide.WideError):
        _decode(blob, not_before_wall_ns=now + 10**9)
    _decode(blob, not_before_wall_ns=now - 10**9)


def test_reader_refuses_truncated_or_non_npz_payloads():
    with pytest.raises(wide.WideError):
        _decode(b"PK\x03\x04 not a zip")
    blob = _extend(_legacy_blob())
    with pytest.raises(wide.WideError):
        _decode(blob[: len(blob) // 2])


def test_a_refused_capture_never_yields_probabilities():
    with pytest.raises(wide.WideError):
        cap = _decode(_reencode(_extend(_legacy_blob()), arrays={"wide_logits_fp32": None}))
        wide.wide_choice_result(cap.logits, _wide_request(32), CATALOG)  # pragma: no cover
