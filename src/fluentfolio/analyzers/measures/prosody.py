"""Measures from the acoustic tools: flat stretches, phrase endings, pitch details, voice and loudness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fluentfolio.analyzers.measures.acoustic import MIN_VOICED_MS
from fluentfolio.analyzers.measures.values import Missing, Value, absent, counted, measured

Row = dict[str, Any]
ENDINGS = {"rise": "rising_endings", "fall": "falling_endings", "level": "level_endings"}
SCOPE_UNITS = {"flat_stretches": "count", **dict.fromkeys(ENDINGS.values(), "count"), "f0_sd": "semitones",
               "voiced_share": "%"}  # fmt: skip
VOICE_UNITS = {"jitter": "%", "shimmer": "%", "hnr": "dB", "f1": "Hz", "f2": "Hz", "f3": "Hz"}
LOUDNESS_UNITS = {"loudness_variation": "ratio", "loudness_peaks": "per s"}


def _missing(units: dict[str, str], why: Missing) -> dict[str, Value]:
    return {key: absent(why.code, why.text, unit) for key, unit in units.items()}


def coverage(summary: Row) -> str:
    """Which answers a scope's prosody rests on, for example "From 3 of 5 answers; 2 have under 10 s ..."."""
    used, total = len(summary["used"]), summary["answers"]
    if used == total:
        return f"From all {total} answers." if total > 1 else "From the one answer."
    short = total - used
    have = "has" if short == 1 else "have"
    return f"From {used} of {total} answers; {short} {have} under {MIN_VOICED_MS // 1000} s of voiced speech."


def scope_values(summary: Row | None, units: Sequence[Row], why: Missing | None) -> dict[str, Value]:
    """One scope's prosody from Praat's pitch row for it and the candidate's stretches between pauses.

    Only stretches in the answers the pitch row used count. Every value is missing with `why` when Praat
    gave nothing, or with the row's reason when no answer has enough voiced speech for prosody statistics.
    """
    if why is None and summary is not None and summary["missing"] is not None:
        why = Missing(summary["missing"]["code"], summary["missing"]["text"])
    if why is not None or summary is None:
        return _missing(SCOPE_UNITS, why or Missing("no_value", "Praat gave no value"))
    used = set(summary["used"])
    mine = [u for u in units if u["answer_id"] in used]
    note = coverage(summary)
    flat = [u["utterance_id"] or u["answer_id"] for u in mine if u["flat"]]
    out = {"flat_stretches": counted(len(flat), note=f"{note} In {', '.join(flat)}." if flat else note)}
    for kind, key in ENDINGS.items():
        out[key] = counted(sum(u["ending"] == kind for u in mine), note=note)
    out["f0_sd"] = measured(summary["sd_st"], "semitones", note=note)
    out["voiced_share"] = measured(100 * summary["voiced_share"], "%", note=note)
    return out


def voice_values(row: Row | None, why: Missing | None) -> dict[str, Value]:
    """One answer's voice quality from Praat, each value missing with a reason when it has none."""
    if why is None and row is not None and row["missing"] is not None and "jitter_pct" not in row:
        why = Missing(row["missing"]["code"], row["missing"]["text"])
    if why is not None or row is None:
        return _missing(VOICE_UNITS, why or Missing("no_value", "Praat gave no value"))
    found = {
        "jitter": row["jitter_pct"], "shimmer": row["shimmer_pct"], "hnr": row["hnr_db"], "f1": row["f1_hz"],
        "f2": row["f2_hz"], "f3": row["f3_hz"],
    }  # fmt: skip
    places = {"jitter": 2, "shimmer": 2, "hnr": 1, "f1": 0, "f2": 0, "f3": 0}
    stretches = "1 voiced stretch" if row["runs"] == 1 else f"{row['runs']} voiced stretches"
    note = f"Over {stretches} of 1 s or more, {row['run_ms'] / 1000:.1f} s in all."
    return {
        key: absent("undefined", "Praat could not measure it", VOICE_UNITS[key]) if x is None
        else measured(x, VOICE_UNITS[key], places[key], note=note)
        for key, x in found.items()
    }  # fmt: skip


def loudness_values(row: Row | None, why: Missing | None) -> dict[str, Value]:
    """One scope's loudness from openSMILE, over the candidate's own speech; never taken from another tool."""
    if why is None and row is not None and row["missing"] is not None:
        why = Missing(row["missing"]["code"], row["missing"]["text"])
    if why is not None or row is None:
        return _missing(LOUDNESS_UNITS, why or Missing("no_value", "openSMILE gave no value"))
    note = f"Over {row['speech_ms'] / 1000:.1f} s of the candidate's speech, pauses left out."
    return {
        "loudness_variation": measured(row["variation"], "ratio", 2, note=note),
        "loudness_peaks": measured(row["peaks_per_s"], "per s", 2, note=note),
    }
