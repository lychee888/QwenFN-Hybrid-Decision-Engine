"""WIDE1 slice 2: the separate wide choice request (17..256 options).

The legacy ``protocol.DecisionRequest`` keeps its 2..16 bound; this is a different type so a wide
request can never be mistaken for a 24-slot one.
"""

from __future__ import annotations

import pathlib

import pytest

from qfn_decision import protocol, wide

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _opts(n: int, *, fmt: str = "option {i}") -> list[protocol.ChoiceOption]:
    return [protocol.ChoiceOption(key=f"k{i}", description=fmt.format(i=i)) for i in range(n)]


def test_wide_request_accepts_17_and_256_and_keeps_caller_keys():
    req = wide.WideChoiceRequest.create("some state", "which?", _opts(17))
    assert req.kind == "choice"
    assert req.n_options == 17
    assert req.caller_keys()[0] == "k0" and req.caller_keys()[-1] == "k16"

    big = wide.WideChoiceRequest.create("s", "q", _opts(256))
    assert big.n_options == 256
    assert len(big.descriptions()) == 256


def test_wide_request_refuses_legacy_counts_and_over_max():
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create("s", "q", _opts(16))
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create("s", "q", _opts(2))
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create("s", "q", _opts(257))
    # the *legacy* type is untouched: still 16 max, still refused at 17
    assert protocol.MAX_CHOICE_OPTIONS == 16
    with pytest.raises(protocol.ProtocolError):
        protocol.DecisionRequest.create("choice", "s", "q", options=_opts(17))


def test_wide_request_accepts_custom_and_unicode_keys_and_refuses_duplicates():
    opts = [
        protocol.ChoiceOption(key="café / 選択", description="d0"),
        protocol.ChoiceOption(key="key.with-punctuation!", description="d1"),
    ] + _opts(15)
    req = wide.WideChoiceRequest.create("state", "question", opts)
    assert req.caller_keys()[0] == "café / 選択"

    dupes = _opts(17)
    dupes[16] = protocol.ChoiceOption(key="k0", description="dup")
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create("s", "q", dupes)


@pytest.mark.parametrize("bad_n", [0, 1, 16, 257, 300])
def test_wide_request_refuses_bad_counts_before_any_engine_activity(bad_n):
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create("s", "q", _opts(bad_n))


def test_wide_request_refuses_bad_option_shapes():
    base = _opts(17)
    # ChoiceOption itself already refuses non-str fields (ProtocolError); the wide layer must
    # refuse them too, so accept either error type at this boundary.
    bad_errors = (wide.WideError, protocol.ProtocolError)
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create("s", "q", [{"key": "a", "description": "d"}] * 17)
    with pytest.raises(bad_errors):
        wide.WideChoiceRequest.create("s", "q", [protocol.ChoiceOption(key=1, description="d")] * 17)
    with pytest.raises(bad_errors):
        wide.WideChoiceRequest.create("s", "q", [protocol.ChoiceOption(key="a", description=None)] * 17)
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create(
            "s", "q", [*base[:-1], protocol.ChoiceOption(key="last", description="two\nlines")]
        )
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create(
            "s", "q", [*base[:-1], protocol.ChoiceOption(key="last", description="x" * 2049)]
        )
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create(
            "s", "q", [*base[:-1], protocol.ChoiceOption(key="last", description="   ")]
        )
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create(
            "s", "q", [*base[:-1], protocol.ChoiceOption(key="k" * 65, description="d")]
        )


def test_wide_request_refuses_empty_and_oversized_fields():
    for state, question in (("", "q"), ("s", ""), ("   ", "q"), ("s", "\n\t ")):
        with pytest.raises(wide.WideError):
            wide.WideChoiceRequest.create(state, question, _opts(17))
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create("s" * 32769, "q", _opts(17))
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create("s", "q" * 4097, _opts(17))
    # the total-char budget is the same one the legacy protocol enforces
    wide.WideChoiceRequest.create("s" * 1000, "q", _opts(17, fmt="d" * 2000))


def test_wide_request_has_no_score_or_slot_surface():
    req = wide.WideChoiceRequest.create("s", "q", _opts(26))
    assert not hasattr(req, "slot_probabilities")
    with pytest.raises(AttributeError):
        _ = req.slot_range
    with pytest.raises(wide.WideError):
        wide.WideChoiceRequest.create("s", "q", _opts(17), score_rubrics=["x"] * 6)
