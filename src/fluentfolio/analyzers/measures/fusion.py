"""Pitch fusion: one median and one p20-p80 range from Praat, SwiftF0 and openSMILE, with an agreement label.

The tools' values are combined in semitones re 27.5 Hz with fixed weights; nothing here is fitted to data.
The label matters more than the fused number: it says how far apart the tools are. This module imports none
of the tools; it reads the per-scope pitch rows their steps wrote.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fluentfolio.analyzers.measures.acoustic import hertz
from fluentfolio.analyzers.measures.values import Value, absent, measured, rounded

Row = dict[str, Any]
TOOLS = ("praat", "swiftf0", "opensmile")
NC_TOOL = "opensmile"  # its non-commercial licence covers the features extracted with it
MIN_VOICED_MS = 5000  # a tool with less voiced speech in a scope is left out of that scope
AGREE_ST = 1.0  # twice the 50-cent limit the benchmark below uses for a correct pitch frame
APPROXIMATE_ST = 3.2  # 20% in frequency (1200 x log2 1.2 = 316 cents), the usual gross pitch error limit
# Hand-set, never fitted. Start: clean-speech pitch F1 in Pitch Detection Benchmark v2.1 (SwiftF0 .949,
# Praat .925, Praat's SHS tracker standing in for openSMILE's .894) gives inverse-error weights .46/.31/.22.
# SwiftF0 is capped at Praat's weight, as those numbers come from its author's benchmark, on whose corpora it
# was trained; openSMILE stays lowest, as its own tracker was not benchmarked. Rounded to tenths.
WEIGHTS = {
    frozenset(TOOLS): {"praat": 0.4, "swiftf0": 0.4, "opensmile": 0.2},
    frozenset({"praat", "swiftf0"}): {"praat": 0.5, "swiftf0": 0.5},
    frozenset({"praat", "opensmile"}): {"praat": 0.67, "opensmile": 0.33},
    frozenset({"swiftf0", "opensmile"}): {"swiftf0": 0.67, "opensmile": 0.33},
    **{frozenset({tool}): {tool: 1.0} for tool in TOOLS},
}
STATS = ("p20", "p50", "p80")
ORDER = ("agree", "approximate", "unreliable")  # from best to worst
SINGLE = "single_tool"
TOO_LITTLE = "too_little_speech"


@dataclass(frozen=True)
class Source:
    tool: str
    version: str
    weight: float


@dataclass(frozen=True)
class Fused:
    """A fused value, its agreement label, the spread between the tools in semitones, and what it used."""

    value: Value
    agreement: str | None  # None only when no pitch tool ran
    spread_st: float | None  # None unless two or more tools were compared
    sources: tuple[Source, ...]
    nc_input: bool


@dataclass(frozen=True)
class Pitch:
    """One scope's fused median (Hz) and range (semitones), each statistic's fusion and each tool's values."""

    median: Fused
    range: Fused
    statistics: dict[str, Row]  # p20, p50, p80 and range: fused_st, spread_st, agreement
    by_tool: list[Row]


def label(spread: float) -> str:
    if spread <= AGREE_ST:
        return "agree"
    return "approximate" if spread <= APPROXIMATE_ST else "unreliable"


def _value(x: float, unit: str, places: int, agreement: str, n: int) -> Value:
    if agreement == "unreliable":
        return absent("tools_disagree", "unreliable: tools disagree", unit, n=n)
    return measured(x, unit, places, n=n, note="not cross-checked" if agreement == SINGLE else None)


def _detail(tool: str, row: Row, version: str, weight: float | None) -> Row:
    found = {f"{s}_st": rounded(row[f"{s}_st"], 2) if f"{s}_st" in row else None for s in STATS}
    median_hz = rounded(row["median_hz"], 1) if "median_hz" in row else None
    dropped = None if weight is not None else TOO_LITTLE
    return {
        "tool": tool, "version": version, "voiced_s": rounded(row["voiced_ms"] / 1000, 1), "weight": weight,
        "dropped": dropped, **found, "median_hz": median_hz,
    }  # fmt: skip


def fuse(rows: Mapping[str, Row], versions: Mapping[str, str]) -> Pitch:
    """One scope's fused pitch from the rows of the tools that ran; a tool missing from `rows` did not run."""
    used = [t for t in TOOLS if t in rows and rows[t]["voiced_ms"] >= MIN_VOICED_MS]
    weights = WEIGHTS.get(frozenset(used), {})
    sources = tuple(Source(t, versions[t], weights[t]) for t in used)
    nc = NC_TOOL in used
    by_tool = [_detail(t, rows[t], versions[t], weights.get(t)) for t in TOOLS if t in rows]
    if not used:
        # When every tool left the scope empty for one reason (no answer long enough), that reason is the one.
        reasons = {(r["missing"]["code"], r["missing"]["text"]) for r in rows.values() if r["missing"]}
        if rows and len(reasons) == 1 and all(r["missing"] for r in rows.values()):
            (code, text), agreement = reasons.pop(), TOO_LITTLE
        elif rows:
            text = f"no pitch tool has {MIN_VOICED_MS // 1000} s of voiced speech in these answers"
            code, agreement = TOO_LITTLE, TOO_LITTLE
        else:
            code, text, agreement = "no_value", "no pitch tool ran", None
        empty = [Fused(absent(code, text, unit), agreement, None, (), False) for unit in ("Hz", "semitones")]
        return Pitch(empty[0], empty[1], {}, by_tool)
    single = len(used) == 1
    stats: dict[str, Row] = {}
    for s in STATS:
        values = [rows[t][f"{s}_st"] for t in used]
        spread = None if single else max(values) - min(values)
        stats[s] = {
            "fused_st": sum(weights[t] * x for t, x in zip(used, values, strict=True)), "spread_st": spread,
            "agreement": SINGLE if spread is None else label(spread),
        }  # fmt: skip
    low, high = stats["p20"], stats["p80"]
    # Linear weights make fused p80 - fused p20 equal to the weighted mean of the tools' own ranges.
    stats["range"] = {
        "fused_st": high["fused_st"] - low["fused_st"],
        "spread_st": None if single else max(low["spread_st"], high["spread_st"]),
        "agreement": SINGLE if single else max(low["agreement"], high["agreement"], key=ORDER.index),
    }
    for row in stats.values():
        row["spread_st"] = None if row["spread_st"] is None else rounded(row["spread_st"], 2)
    median, span = stats["p50"], stats["range"]
    return Pitch(
        Fused(_value(hertz(median["fused_st"]), "Hz", 0, median["agreement"], len(used)),
              median["agreement"], median["spread_st"], sources, nc),
        Fused(_value(span["fused_st"], "semitones", 1, span["agreement"], len(used)),
              span["agreement"], span["spread_st"], sources, nc),
        {s: row | {"fused_st": rounded(row["fused_st"], 2)} for s, row in stats.items()},
        by_tool,
    )  # fmt: skip
