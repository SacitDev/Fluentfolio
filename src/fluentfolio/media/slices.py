"""The candidate audio step: each candidate's own speech joined into one WAV, with a map to the recording.

Praat, SwiftF0 and openSMILE all read this one file, so they measure the same speech. The speech is what the
structure step kept as the candidate's own: voice-detector speech in their answers, less other voices with
300 ms around them, overlap and questions read aloud. Each piece is one stretch between pauses. Stretches
with background music are left out too, so pitch and voice values never come from a song.

Known limit: a stretch the speaker step gave to the wrong person, or overlap it missed, stays in, so the file
can hold a moment of the partner's voice; revisit if a learner reports it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.time import subtract

RATE = 16000
PER_MS = RATE // 1000
AI_AUDIO_MS = 570_000  # 9.5 min: the most audio one person's file for an AI chat may hold
# AAC adds some tens of ms of padding at the ends, so the samples stop short of the limit.
CODEC_SLACK_MS = 500
GAP_MS = 300  # silence between two pieces, so words from separate stretches do not run together
BIT_RATE = 48_000  # mono speech at 16 kHz
AI_AUDIO = {
    "candidate": "for_ai_audio.m4a",
    "person1": "for_ai_audio_person1.m4a",
    "person2": "for_ai_audio_person2.m4a",
}


def source(vad: dict[str, Any], speaker: str) -> str:
    """The input file that holds a candidate's voice: their own track, or the one mixed recording."""
    track = next((t for t in vad.get("tracks", []) if f"S{t['track']}" == speaker), None)
    return "audio16k.wav" if track is None else f"track_{track['track']}.wav"


def join(audio: np.ndarray, spans: Sequence[Sequence[int]]) -> tuple[np.ndarray, list[dict[str, int]]]:
    """The spans' samples one after another, and where each piece lies in the result and in the recording."""
    parts: list[np.ndarray] = []
    pieces: list[dict[str, int]] = []
    at = 0
    for a, b in spans:
        x = audio[a * PER_MS : b * PER_MS]
        ms = len(x) // PER_MS
        if ms == 0:
            continue
        parts.append(x[: ms * PER_MS])
        pieces.append({"wav_ms": at, "orig_ms": a, "ms": ms})
        at += ms
    return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), pieces


def _write_wav(path: Path, audio: np.ndarray) -> None:
    part = path.with_name(path.name + ".part")
    sf.write(part, audio, RATE, subtype="PCM_16", format="WAV")
    os.replace(part, path)


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """One WAV and map per candidate; a candidate with no speech of their own gets neither."""
    structure = read_json(job / "stages" / "structure.json")
    vad = read_json(job / "stages" / "vad.json")
    if not structure["candidates"]:
        return {"status": "skipped", "reason": "no candidate was found", "candidates": {}}
    music = read_json(job / "stages" / "media.json")["music_spans"]
    loaded: dict[str, np.ndarray] = {}
    out = {}
    for n, c in enumerate(structure["candidates"]):
        name = source(vad, c["speaker"])
        if name not in loaded:
            loaded[name] = sf.read(job / "input" / name, dtype="float32")[0]
        audio, pieces = join(loaded[name], subtract(structure["speech"][c["id"]], music))
        row: dict[str, Any] = {"source": name, "speech_ms": len(audio) // PER_MS, "pieces": len(pieces)}
        row |= {"wav": None, "map": None}
        if pieces:
            wav = Path("input") / f"candidate_{c['id']}.wav"
            _write_wav(job / wav, audio)
            write_json(job / wav.with_suffix(".map.json"), {"rate": RATE, "pieces": pieces})
            row |= {"wav": wav.as_posix(), "map": wav.with_suffix(".map.json").as_posix()}
        out[c["id"]] = row
        progress(100 * (n + 1) // len(structure["candidates"]))
    return {"status": "ok", "candidates": out}


def fit(pieces: Sequence[dict[str, int]], limit_ms: int) -> list[dict[str, int]]:
    """Whole pieces in time order, a gap between each two, up to the limit; each with its start in the result.

    The first piece alone longer than the limit is cut to it, so a file is never empty or over the limit.
    """
    kept: list[dict[str, int]] = []
    end = 0
    for p in pieces:
        start = end + GAP_MS if kept else 0
        if start + p["ms"] > limit_ms:
            if not kept:
                kept.append({**p, "audio_ms": 0, "ms": limit_ms})
            break
        kept.append({**p, "audio_ms": start})
        end = start + p["ms"]
    return kept


def _encode(path: Path, audio: np.ndarray) -> None:
    """Mono AAC in an .m4a, written beside the target and then renamed, so no one reads half a file."""
    import av  # loaded only here: the file is made only when the learner asks for it

    part = path.with_name(path.name + ".part")
    with av.open(str(part), "w", format="ipod") as container:
        stream = container.add_stream("aac", rate=RATE, layout="mono")
        stream.bit_rate = BIT_RATE
        frame = av.AudioFrame.from_ndarray(audio[None, :], format="flt", layout="mono")
        frame.sample_rate = RATE
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            container.mux(packet)
    os.replace(part, path)


def for_ai_audio(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """The optional audio for an AI chat: each candidate's own speech, at most 9.5 min, as AAC in .m4a.

    It is built from the candidate audio, so it holds no other voice, no overlap and no question read
    aloud. With the setting off, a file from an earlier run is removed: audio leaves only by the learner's
    choice.
    """
    folder = job / "export" / "for_ai"
    made = job / "stages" / "candidate_audio.json"
    wanted = read_json(job / "job.json")["settings"]["export"]["audio"]
    found = read_json(made)["candidates"] if wanted and made.is_file() else {}
    out = {}
    for n, (cid, row) in enumerate(found.items()):
        if row["wav"] is None:
            continue
        audio = sf.read(job / row["wav"], dtype="float32")[0]
        kept = fit(read_json(job / row["map"])["pieces"], AI_AUDIO_MS - CODEC_SLACK_MS)
        joined = np.zeros((kept[-1]["audio_ms"] + kept[-1]["ms"]) * PER_MS, dtype=np.float32)
        for p in kept:
            joined[p["audio_ms"] * PER_MS : (p["audio_ms"] + p["ms"]) * PER_MS] = audio[
                p["wav_ms"] * PER_MS : (p["wav_ms"] + p["ms"]) * PER_MS
            ]
        folder.mkdir(parents=True, exist_ok=True)
        _encode(folder / AI_AUDIO[cid], joined)
        pieces = [{k: p[k] for k in ("audio_ms", "orig_ms", "ms")} for p in kept]
        out[cid] = {"file": AI_AUDIO[cid], "duration_ms": len(joined) // PER_MS, "pieces": pieces}
        progress(100 * (n + 1) // len(found))
    # Off, or the roles changed since an earlier run: no other audio file may stay behind.
    for old in folder.glob("for_ai_audio*.m4a"):
        if old.name not in {row["file"] for row in out.values()}:
            old.unlink()
    if not wanted:
        return {"status": "disabled", "candidates": {}}
    if not out:
        return {"status": "skipped", "reason": "no candidate has speech of their own", "candidates": {}}
    return {"status": "ok", "candidates": out}
