"""The openSMILE step: loudness and a third pitch track from each candidate's own audio, after consent.

openSMILE's licence allows non-commercial use only. It runs only after the user's own consent, in a process
of its own that loads no GPL code, with its built-in eGeMAPSv02 set and no configuration file of ours.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fluentfolio.analyzers.measures import acoustic
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import data_root

Row = dict[str, Any]
COMPONENT = "opensmile"
ANALYZERS = ("pitch", "loudness")
CONSENT_MISSING = "E_OPENSMILE_CONSENT_MISSING"
SOURCES = ("ui", "cli-tty")  # the user's own act: the licence box, or typing I ACCEPT at a terminal
GPL_MODULES = ("parselmouth", "phonemizer", "espeakng_loader", "av")
POOR_AUDIO = ("meeting_codec", "noisy")  # loudness is not measured on these audio classes
RATE = 16000
FRAME_MS = 10.0  # eGeMAPS frames start every 10 ms and last 20 ms
F0 = "F0semitoneFrom27.5Hz_sma3nz"  # 0 in an unvoiced frame


def gpl_loaded() -> list[str]:
    """The GPL modules in this process; the licence split holds only while there are none."""
    return sorted(m for m in sys.modules if m.split(".")[0] in GPL_MODULES)


def consented(data: Path) -> bool:
    """Whether the data folder holds the user's own consent record for openSMILE."""
    try:
        record = read_json(data / "consent.json")["opensmile"]
        return record["source"] in SOURCES and bool(record["utc"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _pitch(smile: Any, audio: np.ndarray, pieces: list[acoustic.Piece], answers: list[Row]) -> dict[str, Row]:
    if not pieces:
        return acoustic.pitch_scopes([], FRAME_MS, answers)
    frames = smile.process_signal(audio, RATE)
    middles = [t.total_seconds() * 1000 + FRAME_MS for t in frames.index.get_level_values("start")]
    hz = [acoustic.hertz(float(st)) if st > 0 else None for st in frames[F0]]
    return acoustic.pitch_scopes(acoustic.frames(middles, hz, pieces, answers), FRAME_MS, answers)


def _loudness(
    smile: Any, audio: np.ndarray, pieces: list[acoustic.Piece], answers: list[Row], poor: bool
) -> tuple[dict[str, Row], dict[str, dict[str, float]]]:
    """Each scope's loudness values and all 88 functionals, over the pieces of the scope's answers."""
    loudness: dict[str, Row] = {}
    functionals: dict[str, dict[str, float]] = {}
    for scope, ids in acoustic.scopes(answers).items():
        chosen = set(ids)
        mine = [p for p in pieces if acoustic.answer_at(p.orig_ms + p.ms // 2, answers) in chosen]
        speech_ms = sum(p.ms for p in mine)
        row: Row = {"speech_ms": speech_ms, "missing": None}
        if not mine:
            loudness[scope] = row | {"missing": {"code": "no_speech", "text": "no speech in these answers"}}
            continue
        per_ms = RATE // 1000
        joined = np.concatenate([audio[p.wav_ms * per_ms : (p.wav_ms + p.ms) * per_ms] for p in mine])
        found = {name: float(x) for name, x in smile.process_signal(joined, RATE).iloc[0].items()}
        functionals[scope] = found
        if poor:
            text = "not measured on meeting-codec or noisy audio"
            loudness[scope] = row | {"missing": {"code": "audio_not_suited", "text": text}}
        else:
            loudness[scope] = row | {
                "variation": found["loudness_sma3_stddevNorm"], "peaks_per_s": found["loudnessPeaksPerSec"],
            }  # fmt: skip
    return loudness, functionals


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """openSMILE's values for every candidate; without the user's consent it does not start."""
    settings = read_json(job / "job.json")["settings"]
    if not (on := [a for a in ANALYZERS if acoustic.switched_on(settings, COMPONENT, a)]):
        return {"status": "disabled", "candidates": {}}
    if not consented(data_root()):
        return {"status": "disabled", "code": CONSENT_MISSING, "candidates": {}}
    made = acoustic.candidate_audio(job)
    if made is None:
        return {"status": "skipped", "reason": "the candidate audio step did not finish", "candidates": {}}
    try:
        import opensmile
    except ImportError:  # consent given, but the installer has not added the package yet
        return {"status": "unavailable", "code": "E_COMPONENT_UNAVAILABLE", "candidates": {}}
    lld = opensmile.Smile(opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.LowLevelDescriptors)
    functionals = opensmile.Smile(opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.Functionals)
    structure = read_json(job / "stages" / "structure.json")
    poor = read_json(job / "stages" / "media.json")["quality"]["class"] in POOR_AUDIO
    out = {}
    for n, (cid, row) in enumerate(made["candidates"].items()):
        answers = [a for a in structure["answers"] if a["candidate"] == cid]
        audio = sf.read(job / row["wav"], dtype="float32")[0] if row["wav"] else np.zeros(0, np.float32)
        pieces = acoustic.pieces(read_json(job / row["map"])["pieces"]) if row["map"] else []
        found: Row = {"scopes": {}, "loudness": {}, "functionals": {}}
        if "pitch" in on:
            found["scopes"] = _pitch(lld, audio, pieces, answers)
        if "loudness" in on:
            found["loudness"], found["functionals"] = _loudness(functionals, audio, pieces, answers, poor)
        out[cid] = found
        progress(100 * (n + 1) // len(made["candidates"]))
    if gpl := gpl_loaded():  # a step that shared its process with GPL code fails
        raise RuntimeError(f"GPL modules loaded in the openSMILE step: {gpl}")
    version = f"opensmile {opensmile.__version__}, eGeMAPSv02"
    return {
        "status": "ok", "provenance": {"tool": COMPONENT, "version": version}, "analyzers": on,
        "candidates": out,
    }  # fmt: skip
