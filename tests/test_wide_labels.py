"""WIDE1 slice 1: the validated 256-label catalog (RED first).

Scope: catalog parsing, order/uniqueness/vocabulary checks and the canonical *content* digest
(which is deliberately not the file-byte SHA).
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from qfn_decision import wide

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "wide-label-catalog.json"
PINNED_BYTES_SHA = "71a861ff84956974dd3294e20d09f86a49e323d4eff03299c30184e5517fb9bf"


def _catalog_dict() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_bundled_catalog_is_the_validated_resource():
    raw = CATALOG_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINNED_BYTES_SHA
    cat = wide.WideCatalog.from_file(CATALOG_PATH)
    assert cat.schema == "qfn.wide-label-catalog/1"
    assert cat.tokenizer_revision == "1b3c3cedc14fb3bee4294dea5e3515af74b8268a"
    assert cat.n_labels == 256 and len(cat.token_ids) == 256
    assert cat.labels[:16] == tuple("ABCDEFGHIJKLMNOP")
    assert cat.token_ids[:16] == tuple(range(32, 48))
    assert cat.labels[-1] == "JU"
    assert cat.label_for_index(0) == "A" and cat.label_for_index(255) == "JU"


def test_digest_is_content_canonical_not_file_bytes():
    payload = _catalog_dict()
    canonical = wide.WideCatalog.from_dict(payload)
    # same content, different file layout / key order / whitespace -> same digest
    reordered = {k: payload[k] for k in reversed(list(payload))}
    other = wide.WideCatalog.from_dict(reordered)
    assert other.digest == canonical.digest
    # file bytes differ, the semantic digest does not
    assert hashlib.sha256(json.dumps(reordered).encode()).hexdigest() != PINNED_BYTES_SHA
    # every semantic component changes the digest (a changed schema string is refused outright)
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict({**payload, "schema": "qfn.wide-label-catalog/2"})
    for mutate in (
        {"tokenizer_revision": "0" * 40},
        {"labels": [*payload["labels"][:-1], "JV"]},
        {"token_ids": [*payload["token_ids"][:-1], 4242]},
    ):
        changed = {**payload, **mutate}
        assert wide.WideCatalog.from_dict(changed).digest != canonical.digest


def test_catalog_refuses_wrong_digest_pin():
    payload = _catalog_dict()
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict(payload, expected_digest="0" * 64)


def test_catalog_refuses_bad_ids_and_labels():
    payload = _catalog_dict()

    short = {**payload, "labels": payload["labels"][:-1], "token_ids": payload["token_ids"][:-1]}
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict(short)

    dup_label = {**payload, "labels": [payload["labels"][0], *payload["labels"][:-1]]}
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict(dup_label)

    dup_id = {**payload, "token_ids": [payload["token_ids"][1], *payload["token_ids"][1:]]}
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict(dup_id)

    boolean_id = {**payload, "token_ids": [True, *payload["token_ids"][1:]]}
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict(boolean_id)

    negative_id = {**payload, "token_ids": [-1, *payload["token_ids"][1:]]}
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict(negative_id)

    lowered = {**payload, "labels": [payload["labels"][0].lower(), *payload["labels"][1:]]}
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict(lowered)


def test_catalog_refuses_out_of_vocabulary_ids_when_bound_is_known():
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict(_catalog_dict(), vocab_size=100)   # id 213977 > 100
    wide.WideCatalog.from_dict(_catalog_dict(), vocab_size=300000)
    cat = wide.WideCatalog.from_dict(_catalog_dict())
    with pytest.raises(wide.WideError):
        cat.validate_vocab_size(100)


def test_catalog_refuses_unknown_keys_and_wrong_types():
    payload = _catalog_dict()
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict({**payload, "extra": 1})
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict({**payload, "labels": "A" * 256})
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict({**payload, "token_ids": [str(v) for v in payload["token_ids"]]})
    with pytest.raises(wide.WideError):
        wide.WideCatalog.from_dict({**payload, "schema": 7})
