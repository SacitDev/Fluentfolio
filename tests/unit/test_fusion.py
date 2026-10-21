"""Pitch fusion and its agreement labels, on hand-made tool rows.

Each expected number is worked out by hand in a comment, from st = 12 x log2(Hz / 27.5) and the fixed weights.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from fluentfolio.analyzers.measures import fusion
from fluentfolio.analyzers.measures.fusion import Source

Row = dict[str, Any]
VERSIONS = {"praat": "Praat 6.4", "swiftf0": "swift-f0 0.3", "opensmile": "opensmile 2.6"}


def row(p20: float, p50: float, p80: float, voiced_ms: int = 60_000) -> Row:
    """A tool's pitch row for one scope, as the acoustic steps write it."""
    return {
        "answers": 1, "used": ["A01"], "voiced_ms": voiced_ms, "missing": None,
        "p20_st": p20, "p50_st": p50, "p80_st": p80, "sd_st": 1.0, "median_hz": 27.5 * 2 ** (p50 / 12),
    }  # fmt: skip


def too_short() -> Row:
    """A row for a scope where no answer had enough voiced speech for prosody statistics."""
    missing = {"code": "insufficient_sample", "text": "no answer has 10 s of voiced speech or more"}
    return {"answers": 1, "used": [], "voiced_ms": 0, "missing": missing}


def st(hz: float) -> float:
    return 12 * math.log2(hz / 27.5)


@pytest.mark.parametrize(
    ("spread", "expected"),
    [(0.0, "agree"), (1.0, "agree"), (1.25, "approximate"), (3.2, "approximate"), (3.25, "unreliable"),
     (12.0, "unreliable")],
)  # fmt: skip
def test_labels_at_1_0_and_3_2_semitones(spread: float, expected: str) -> None:
    # Both limits belong to the better label: at most 1.0 st agrees, at most 3.2 st is approximate.
    assert fusion.label(spread) == expected


def test_three_tools_that_agree() -> None:
    rows = {
        "praat": row(28.0, 30.0, 32.0),
        "swiftf0": row(28.5, 30.5, 32.5),
        "opensmile": row(27.5, 29.5, 31.5),
    }
    found = fusion.fuse(rows, VERSIONS)
    # p50: 0.4 x 30 + 0.4 x 30.5 + 0.2 x 29.5 = 12 + 12.2 + 5.9 = 30.1 st; spread 30.5 - 29.5 = 1.0 st.
    assert found.statistics["p50"] == {"fused_st": 30.1, "spread_st": 1.0, "agreement": "agree"}
    # 27.5 x 2 ** (30.1 / 12) = 27.5 x 5.6896 = 156.46 Hz, shown as 156.
    assert found.median.value.value == 156.0
    assert found.median.value.unit == "Hz"
    assert (found.median.agreement, found.median.spread_st) == ("agree", 1.0)
    # p20: 11.2 + 11.4 + 5.5 = 28.1; p80: 12.8 + 13.0 + 6.3 = 32.1; range 32.1 - 28.1 = 4.0 st.
    assert found.range.value.value == 4.0
    assert found.range.value.unit == "semitones"
    assert found.range.agreement == "agree"
    assert found.median.sources == (
        Source("praat", "Praat 6.4", 0.4), Source("swiftf0", "swift-f0 0.3", 0.4),
        Source("opensmile", "opensmile 2.6", 0.2),
    )  # fmt: skip
    assert found.median.nc_input and found.range.nc_input  # openSMILE was used


def test_fusion_averages_semitones_not_hertz() -> None:
    rows = {"praat": row(30.0, st(200.0), 40.0), "swiftf0": row(30.0, st(240.0), 40.0)}
    found = fusion.fuse(rows, VERSIONS)
    # The mean of two semitone values is the geometric mean in Hz: sqrt(200 x 240) = 219.09, shown as 219.
    # Averaging in Hz would give (200 + 240) / 2 = 220.
    assert found.median.value.value == 219.0
    # The spread is 12 x log2(240 / 200) = 3.16 st, so the value is shown as approximate.
    assert (found.median.agreement, found.median.spread_st) == ("approximate", 3.16)


def test_an_octave_error_makes_the_median_unreliable() -> None:
    # SwiftF0's median is one octave (12 st) above the others; its p20 and p80 agree.
    rows = {
        "praat": row(28.0, 30.0, 32.0),
        "swiftf0": row(28.0, 42.0, 32.0),
        "opensmile": row(28.0, 30.0, 32.0),
    }
    found = fusion.fuse(rows, VERSIONS)
    assert found.median.agreement == "unreliable"
    assert found.median.value.value is None
    assert found.median.value.missing is not None
    assert found.median.value.missing.code == "tools_disagree"
    assert found.median.spread_st == 12.0
    assert found.median.nc_input  # the disagreement itself rests on openSMILE's value too
    # The range is fused separately: 32 - 28 = 4.0 st, all tools within 0 st.
    assert (found.range.value.value, found.range.agreement) == (4.0, "agree")


def test_without_opensmile_praat_and_swiftf0_weigh_half_each() -> None:
    rows = {"praat": row(28.0, 30.0, 32.0), "swiftf0": row(28.0, 31.25, 33.0)}
    found = fusion.fuse(rows, VERSIONS)
    # p50: 0.5 x 30 + 0.5 x 31.25 = 30.625 st; 27.5 x 2 ** (30.625 / 12) = 161.28 Hz; spread 1.25 st.
    assert found.median.value.value == 161.0
    assert (found.median.agreement, found.median.spread_st) == ("approximate", 1.25)
    assert [s.weight for s in found.median.sources] == [0.5, 0.5]
    assert not found.median.nc_input
    assert not found.range.nc_input
    # p20 28.0 and p80 32.5 (spreads 0 and 1.0): range 4.5 st, agree.
    assert (found.range.value.value, found.range.agreement) == (4.5, "agree")


def test_without_swiftf0_praat_weighs_two_thirds() -> None:
    rows = {"praat": row(28.0, 30.0, 32.0), "opensmile": row(28.0, 33.0, 32.0)}
    found = fusion.fuse(rows, VERSIONS)
    # p50: 0.67 x 30 + 0.33 x 33 = 20.1 + 10.89 = 30.99 st; 27.5 x 2 ** (30.99 / 12) = 164.72 Hz.
    assert found.statistics["p50"] == {"fused_st": 30.99, "spread_st": 3.0, "agreement": "approximate"}
    assert found.median.value.value == 165.0
    assert [(s.tool, s.weight) for s in found.median.sources] == [("praat", 0.67), ("opensmile", 0.33)]
    assert found.median.nc_input


def test_without_praat_swiftf0_weighs_two_thirds() -> None:
    rows = {"swiftf0": row(28.0, 30.0, 32.0), "opensmile": row(29.0, 31.0, 33.0)}
    found = fusion.fuse(rows, VERSIONS)
    # p20: 0.67 x 28 + 0.33 x 29 = 28.33; p50: 20.1 + 10.23 = 30.33; p80: 21.44 + 10.89 = 32.33.
    assert [found.statistics[s]["fused_st"] for s in ("p20", "p50", "p80")] == [28.33, 30.33, 32.33]
    assert found.statistics["range"]["fused_st"] == 4.0  # 32.33 - 28.33
    assert [(s.tool, s.weight) for s in found.range.sources] == [("swiftf0", 0.67), ("opensmile", 0.33)]


def test_one_tool_is_shown_as_not_cross_checked() -> None:
    found = fusion.fuse({"praat": row(28.0, 30.0, 32.0)}, VERSIONS)
    # 27.5 x 2 ** (30 / 12) = 27.5 x 5.6569 = 155.56 Hz, shown as 156.
    assert found.median.value.value == 156.0
    assert found.median.value.note == "not cross-checked"
    assert (found.median.agreement, found.median.spread_st) == ("single_tool", None)
    assert (found.range.value.value, found.range.agreement) == (4.0, "single_tool")
    assert found.median.sources == (Source("praat", "Praat 6.4", 1.0),)
    assert not found.median.nc_input


@pytest.mark.parametrize(("voiced_ms", "kept"), [(4999, False), (5000, True), (5001, True)])
def test_a_tool_needs_5_s_of_voiced_speech(voiced_ms: int, kept: bool) -> None:
    rows = {
        "praat": row(28.0, 30.0, 32.0, voiced_ms), "swiftf0": row(28.0, 30.0, 32.0),
        "opensmile": row(28.0, 30.0, 32.0),
    }  # fmt: skip
    found = fusion.fuse(rows, VERSIONS)
    weights = {s.tool: s.weight for s in found.median.sources}
    # Without Praat the other two take the 0.67 / 0.33 row of the weight table.
    assert weights == ({"praat": 0.4, "swiftf0": 0.4, "opensmile": 0.2} if kept else
                       {"swiftf0": 0.67, "opensmile": 0.33})  # fmt: skip
    praat = found.by_tool[0]
    assert praat["tool"] == "praat"
    assert praat["voiced_s"] == round(voiced_ms / 1000, 1)
    assert (praat["weight"], praat["dropped"]) == ((0.4, None) if kept else (None, "too_little_speech"))


def test_a_row_without_enough_speech_is_dropped() -> None:
    rows = {"praat": too_short(), "swiftf0": row(28.0, 30.0, 32.0)}
    found = fusion.fuse(rows, VERSIONS)
    assert found.median.agreement == "single_tool"
    assert found.median.sources == (Source("swiftf0", "swift-f0 0.3", 1.0),)
    assert found.by_tool[0] == {
        "tool": "praat", "version": "Praat 6.4", "voiced_s": 0.0, "weight": None,
        "dropped": "too_little_speech", "p20_st": None, "p50_st": None, "p80_st": None, "median_hz": None,
    }  # fmt: skip


def test_no_tool_with_enough_speech_gives_no_number() -> None:
    rows = {
        "praat": row(28.0, 30.0, 32.0, 4999),
        "swiftf0": too_short(),
        "opensmile": row(28.0, 30.0, 32.0, 100),
    }
    found = fusion.fuse(rows, VERSIONS)
    for fused in (found.median, found.range):
        assert fused.agreement == "too_little_speech"
        assert fused.value.value is None
        assert fused.value.missing is not None
        assert fused.value.missing.code == "too_little_speech"
        assert (fused.sources, fused.nc_input, fused.spread_st) == ((), False, None)
    assert [r["dropped"] for r in found.by_tool] == ["too_little_speech"] * 3


def test_when_no_answer_is_long_enough_the_tools_reason_is_kept() -> None:
    found = fusion.fuse({"praat": too_short(), "swiftf0": too_short()}, VERSIONS)
    for fused in (found.median, found.range):
        assert fused.agreement == "too_little_speech"
        assert fused.value.missing is not None
        assert (fused.value.missing.code, fused.value.missing.text) == (
            "insufficient_sample", "no answer has 10 s of voiced speech or more",
        )  # fmt: skip


def test_no_pitch_tool_ran() -> None:
    found = fusion.fuse({}, VERSIONS)
    assert found.median.agreement is None
    assert found.median.value.missing is not None
    assert found.median.value.missing.code == "no_value"
    assert found.by_tool == []


def test_the_range_takes_the_worse_label() -> None:
    # p20 spread 0.5 st (agree), p80 spread 2.0 st (approximate).
    rows = {"praat": row(28.0, 30.0, 32.0), "swiftf0": row(28.5, 30.0, 34.0)}
    found = fusion.fuse(rows, VERSIONS)
    # Fused p20 28.25, p80 33.0: range 4.75, shown as 4.8 (half rounds up).
    assert (found.range.value.value, found.range.agreement, found.range.spread_st) == (
        4.8,
        "approximate",
        2.0,
    )


def test_an_unreliable_p20_leaves_the_range_without_a_number() -> None:
    # p20 spread 4.0 st (unreliable), p80 spread 0 st (agree).
    rows = {"praat": row(24.0, 30.0, 32.0), "swiftf0": row(28.0, 30.0, 32.0)}
    found = fusion.fuse(rows, VERSIONS)
    assert (found.range.agreement, found.range.spread_st) == ("unreliable", 4.0)
    assert found.range.value.value is None
    assert (found.median.value.value, found.median.agreement) == (156.0, "agree")
