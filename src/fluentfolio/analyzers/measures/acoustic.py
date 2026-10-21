"""Rules shared by the acoustic tools: the candidate audio's time map, semitones and the prosody rules.

Praat, SwiftF0 and openSMILE each run in their own process on the same candidate-only audio. They share
these rules, so their values cover the same answers and can be set side by side.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fluentfolio.core.files import read_json

Row = dict[str, Any]
REF_HZ = 27.5  # semitones are counted from 27.5 Hz, as in eGeMAPS, so every tool's values compare
# Prosody statistics use only answers with this much voiced speech; shorter answers are left out and
# counted in the coverage note.
MIN_VOICED_MS = 10000
# Heuristic thresholds, labelled so wherever they are shown; no study fixes them.
FLAT_MIN_MS = 2000  # a unit this long or longer can be a flat stretch
FLAT_MIN_VOICED_MS = 1000  # with at least this much voiced speech in it
FLAT_RANGE_ST = 2.0  # and its pitch range (10th to 90th percentile) under this many semitones
ENDING_MS = 300  # a unit's ending is read from its last 300 ms of voiced speech
ENDING_MIN_MS = 100  # with at least this much voiced speech in that window
ENDING_ST = 2.0  # a fitted change of at least this many semitones is a rise or a fall
VOICE_RUN_MS = 1000  # voice quality uses only voiced stretches at least this long
INSUFFICIENT = "insufficient_sample"


def switched_on(settings: Row, component: str, analyzer: str) -> bool:
    """Whether a job's settings leave a tool and one analyzer it serves on; one not listed is on."""
    components, analyzers = settings.get("components", {}), settings.get("analyzers", {})
    return bool(components.get(component, True) and analyzers.get(analyzer, True))


def candidate_audio(job: Path) -> Row | None:
    """The candidate audio step's result, or None when it did not finish."""
    path = job / "stages" / "candidate_audio.json"
    made = read_json(path) if path.is_file() else None
    return made if made is not None and made["status"] == "ok" else None


def semitones(hz: float) -> float:
    return 12 * math.log2(hz / REF_HZ)


def hertz(st: float) -> float:
    return REF_HZ * 2 ** (st / 12)


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0-100) with linear interpolation between the closest ranks, as numpy does."""
    ordered = sorted(values)
    at = (len(ordered) - 1) * q / 100
    low = math.floor(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


@dataclass(frozen=True)
class Piece:
    """One stretch of the candidate audio: its start there, its start in the recording, its length."""

    wav_ms: int
    orig_ms: int
    ms: int


def pieces(rows: Sequence[Row]) -> list[Piece]:
    return [Piece(r["wav_ms"], r["orig_ms"], r["ms"]) for r in rows]


def piece_at(wav_ms: float, found: Sequence[Piece]) -> int | None:
    """The index of the piece that holds a time of the candidate audio, or None past its end."""
    k = bisect_right([p.wav_ms for p in found], wav_ms) - 1
    if k < 0 or wav_ms >= found[k].wav_ms + found[k].ms:
        return None
    return k


def scopes(answers: Sequence[Row]) -> dict[str, list[str]]:
    """The answer IDs of "all" and of each Part that some answer belongs to; unknown Parts give the total."""
    out = {"all": [a["id"] for a in answers]}
    for part in (1, 2, 3):
        if chosen := [a["id"] for a in answers if a["part"] == part]:
            out[f"part{part}"] = chosen
    return out


def answer_at(orig_ms: int, answers: Sequence[Row]) -> str | None:
    """The answer whose time span holds a time of the recording."""
    return next((a["id"] for a in answers if a["start_ms"] <= orig_ms < a["end_ms"]), None)


@dataclass(frozen=True)
class Frame:
    """One pitch frame: its piece, its answer, and its pitch in semitones (None when unvoiced)."""

    piece: int
    answer_id: str
    wav_ms: float
    st: float | None


def frames(
    times_ms: Sequence[float], hz: Sequence[float | None], found: Sequence[Piece], answers: Sequence[Row]
) -> list[Frame]:
    """A tool's frames placed in the recording; a frame outside every piece or answer is left out."""
    out = []
    for t, f in zip(times_ms, hz, strict=True):
        k = piece_at(t, found)
        if k is None:
            continue
        answer = answer_at(round(found[k].orig_ms + t - found[k].wav_ms), answers)
        if answer is not None:
            out.append(Frame(k, answer, t, None if f is None else semitones(f)))
    return out


def enough_voice(found: Sequence[Frame], frame_ms: float, answers: Sequence[Row]) -> set[str]:
    """The answers with MIN_VOICED_MS of voiced frames or more, the only ones prosody statistics use."""
    voiced = Counter(f.answer_id for f in found if f.st is not None)
    return {a["id"] for a in answers if voiced[a["id"]] * frame_ms >= MIN_VOICED_MS}


def pitch_summary(st: Sequence[float], frame_ms: float, answers: int, used: list[str]) -> Row:
    """One scope's pitch over the answers it uses: semitone percentiles and the median in Hz, or why none."""
    row: Row = {"answers": answers, "used": used, "voiced_ms": round(len(st) * frame_ms), "missing": None}
    if not used:
        text = f"no answer has {MIN_VOICED_MS // 1000} s of voiced speech or more"
        return row | {"missing": {"code": INSUFFICIENT, "text": text}}
    mean = sum(st) / len(st)
    p50 = percentile(st, 50)
    return row | {
        "p20_st": percentile(st, 20), "p50_st": p50, "p80_st": percentile(st, 80),
        "sd_st": math.sqrt(sum((x - mean) ** 2 for x in st) / len(st)), "median_hz": hertz(p50),
    }  # fmt: skip


def pitch_scopes(found: Sequence[Frame], frame_ms: float, answers: Sequence[Row]) -> dict[str, Row]:
    """Every scope's pitch from one tool's frames, over its answers with enough voiced speech."""
    enough = enough_voice(found, frame_ms, answers)
    out = {}
    for scope, ids in scopes(answers).items():
        used = [i for i in ids if i in enough]
        chosen = set(used)
        st = [f.st for f in found if f.answer_id in chosen and f.st is not None]
        out[scope] = pitch_summary(st, frame_ms, len(ids), used)
    return out


def ending(times_ms: Sequence[float], st: Sequence[float], frame_ms: float) -> str | None:
    """A unit's final movement from its last voiced frames: "rise", "fall", "level", or None if too short.

    The change is the least-squares slope over the last ENDING_MS of voiced speech times that window's span.
    """
    last = times_ms[-1] if times_ms else 0.0
    window = [(t, s) for t, s in zip(times_ms, st, strict=True) if t >= last - ENDING_MS]
    if len(window) * frame_ms < ENDING_MIN_MS:
        return None
    mean_t = sum(t for t, _ in window) / len(window)
    mean_s = sum(s for _, s in window) / len(window)
    spread = sum((t - mean_t) ** 2 for t, _ in window)
    slope = sum((t - mean_t) * (s - mean_s) for t, s in window) / spread
    change = slope * (last - window[0][0])
    if change >= ENDING_ST:
        return "rise"
    if change <= -ENDING_ST:
        return "fall"
    return "level"


def flat(unit_ms: int, st: Sequence[float], frame_ms: float) -> bool:
    """A flat stretch: a long unit with enough voiced speech whose pitch stays inside a small range."""
    if unit_ms < FLAT_MIN_MS or len(st) * frame_ms < FLAT_MIN_VOICED_MS:
        return False
    return percentile(st, 90) - percentile(st, 10) < FLAT_RANGE_ST


def voiced_runs(found: Sequence[Frame], frame_ms: float) -> list[tuple[float, float]]:
    """Runs of consecutive voiced frames in one piece lasting VOICE_RUN_MS or more, as (start, end) ms."""
    runs: list[tuple[float, float]] = []
    group: list[Frame] = []
    for f in [*found, None]:
        voiced = f is not None and f.st is not None
        # A frame left out (outside every answer) breaks a run, as an unvoiced frame does.
        follows = voiced and bool(group) and f is not None and f.piece == group[-1].piece
        follows = follows and f is not None and f.wav_ms - group[-1].wav_ms <= 1.5 * frame_ms
        if group and not follows:
            if group[-1].wav_ms + frame_ms - group[0].wav_ms >= VOICE_RUN_MS:
                runs.append((group[0].wav_ms, group[-1].wav_ms + frame_ms))
            group = []
        if voiced and f is not None:
            group.append(f)
    return runs
