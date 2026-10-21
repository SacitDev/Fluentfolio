"""The Praat step: pitch, phrase endings, flat stretches and voice quality from each candidate's own audio.

Praat runs through Parselmouth (GPL) in a process of its own, on the WAV of the candidate audio step.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fluentfolio.analyzers.measures import acoustic
from fluentfolio.core.files import read_json

Row = dict[str, Any]
COMPONENT = "praat"
ANALYZERS = ("pitch", "voice-quality")
STEP_S = 0.01
FIRST_PASS_HZ = (60.0, 700.0)  # wide enough for any adult voice
# The second pass uses the speaker's own range (De Looze and Hirst): 0.75 of the lower quartile of the first
# pass to 1.5 of its upper quartile, which avoids the octave jumps of one fixed range for every voice.
FLOOR_FACTOR, CEILING_FACTOR = 0.75, 1.5
MAX_FORMANT_HZ = 5500.0  # Praat's own default; the formants are context only
CLEAN = "clean"  # voice quality is measured on clean audio only
NOT_CLEAN = "audio_not_clean"


def wanted(settings: Row) -> list[str]:
    """The analyzers this step serves that are switched on; none when Praat itself is switched off."""
    return [a for a in ANALYZERS if acoustic.switched_on(settings, COMPONENT, a)]


def _number(x: float, places: int) -> float | None:
    return None if math.isnan(x) else round(x, places)


def track(sound: Any) -> tuple[list[float], list[float | None], float, float]:
    """Two-pass pitch: frame times in ms, Hz per frame (None if unvoiced), and the second pass's range."""
    pitch = sound.to_pitch(time_step=STEP_S, pitch_floor=FIRST_PASS_HZ[0], pitch_ceiling=FIRST_PASS_HZ[1])
    first = pitch.selected_array["frequency"]
    floor, ceiling = FIRST_PASS_HZ
    if (first > 0).any():
        q25, q75 = np.percentile(first[first > 0], [25, 75])
        floor, ceiling = float(FLOOR_FACTOR * q25), float(CEILING_FACTOR * q75)
        pitch = sound.to_pitch(time_step=STEP_S, pitch_floor=floor, pitch_ceiling=ceiling)
    hz = [float(f) if f > 0 else None for f in pitch.selected_array["frequency"]]
    return [t * 1000 for t in pitch.xs()], hz, floor, ceiling


def _voice(sound: Any, runs: Sequence[tuple[float, float]], floor: float, ceiling: float) -> Row:
    """Jitter, shimmer, HNR and the first three formants over the voiced stretches, joined."""
    import parselmouth
    from parselmouth.praat import call

    joined = parselmouth.Sound.concatenate([sound.extract_part(from_time=a / 1000, to_time=b / 1000)
                                            for a, b in runs])  # fmt: skip
    pitch = joined.to_pitch(time_step=STEP_S, pitch_floor=floor, pitch_ceiling=ceiling)
    points = call([joined, pitch], "To PointProcess (cc)")
    # Praat's standard settings: periods of 0.1-20 ms, neighbours differing by at most 1.3 times.
    jitter = call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([joined, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    hnr = call(call(joined, "To Harmonicity (cc)", STEP_S, floor, 0.1, 1.0), "Get mean", 0, 0)
    formant = call(joined, "To Formant (burg)", 0.0, 5.0, MAX_FORMANT_HZ, 0.025, 50.0)
    values = {
        "jitter_pct": _number(100 * jitter, 3), "shimmer_pct": _number(100 * shimmer, 3),
        "hnr_db": _number(hnr, 2),
        **{f"f{k}_hz": _number(call(formant, "Get mean", k, 0, 0, "hertz"), 1) for k in (1, 2, 3)},
    }  # fmt: skip
    missing = None
    if any(v is None for v in values.values()):
        missing = {"code": "undefined", "text": "Praat could not measure every value of these stretches"}
    return {"runs": len(runs), "run_ms": round(sum(b - a for a, b in runs)), **values, "missing": missing}


def _units(found: list[acoustic.Frame], pieces: list[acoustic.Piece], utterances: list[Row]) -> list[Row]:
    """Each stretch between pauses: where it is, its final movement and whether it is flat."""
    ms = STEP_S * 1000
    out = []
    for k, p in enumerate(pieces):
        mine = [f for f in found if f.piece == k]
        if not mine:
            continue
        voiced = [f for f in mine if f.st is not None]
        st = [f.st for f in voiced if f.st is not None]
        middle = p.orig_ms + p.ms // 2
        utterance = next((u["id"] for u in utterances if u["start_ms"] <= middle <= u["end_ms"]), None)
        out.append({
            "answer_id": mine[0].answer_id, "utterance_id": utterance, "start_ms": p.orig_ms,
            "end_ms": p.orig_ms + p.ms, "voiced_ms": round(len(st) * ms),
            "ending": acoustic.ending([f.wav_ms for f in voiced], st, ms),
            "flat": acoustic.flat(p.ms, st, ms),
        })  # fmt: skip
    return out


def _candidate(
    job: Path, made: Row, answers: list[Row], utterances: list[Row], clean: bool, on: list[str]
) -> Row:
    import parselmouth

    ms = STEP_S * 1000
    if made["wav"] is None:
        return {"range_hz": None, "scopes": acoustic.pitch_scopes([], ms, answers), "units": [], "voice": {}}
    sound = parselmouth.Sound(str(job / made["wav"]))
    pieces = acoustic.pieces(read_json(job / made["map"])["pieces"])
    times, hz, floor, ceiling = track(sound)
    found = acoustic.frames(times, hz, pieces, answers)
    scopes = acoustic.pitch_scopes(found, ms, answers)
    for row in scopes.values():
        chosen = set(row["used"])
        mine = [f for f in found if f.answer_id in chosen]
        row["voiced_share"] = sum(f.st is not None for f in mine) / len(mine) if mine else None
    voice: dict[str, Row] = {}
    if "voice-quality" in on:
        for a in answers:
            runs = acoustic.voiced_runs([f for f in found if f.answer_id == a["id"]], ms)
            if not clean:
                voice[a["id"]] = {"missing": {"code": NOT_CLEAN, "text": "measured on clean audio only"}}
            elif not runs:
                text = f"no voiced stretch of {acoustic.VOICE_RUN_MS // 1000} s or more"
                voice[a["id"]] = {"missing": {"code": acoustic.INSUFFICIENT, "text": text}}
            else:
                voice[a["id"]] = _voice(sound, runs, floor, ceiling)
    return {
        "range_hz": [round(floor, 1), round(ceiling, 1)], "scopes": scopes,
        "units": _units(found, pieces, utterances), "voice": voice,
    }  # fmt: skip


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """Praat's values for every candidate, from the candidate audio."""
    settings = read_json(job / "job.json")["settings"]
    if not (on := wanted(settings)):
        return {"status": "disabled", "candidates": {}}
    made = acoustic.candidate_audio(job)
    if made is None:
        return {"status": "skipped", "reason": "the candidate audio step did not finish", "candidates": {}}
    import parselmouth

    structure = read_json(job / "stages" / "structure.json")
    clean = read_json(job / "stages" / "media.json")["quality"]["class"] == CLEAN
    out = {}
    for n, (cid, row) in enumerate(made["candidates"].items()):
        answers = [a for a in structure["answers"] if a["candidate"] == cid]
        ids = {a["id"] for a in answers}
        utterances = [u for u in structure["utterances"] if u["answer_id"] in ids]
        out[cid] = _candidate(job, row, answers, utterances, clean, on)
        progress(100 * (n + 1) // len(made["candidates"]))
    version = f"Praat {parselmouth.PRAAT_VERSION}, parselmouth {parselmouth.__version__}"
    return {
        "status": "ok", "provenance": {"tool": COMPONENT, "version": version}, "analyzers": on,
        "candidates": out,
    }  # fmt: skip
