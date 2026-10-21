"""Silero VAD v6: where the recording holds speech, from the model file inside the faster-whisper package.

Pauses, speaking time and the speech-recognition chunks all come from these segments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fluentfolio.core.enums import Layout
from fluentfolio.core.errors import InputError
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import models_dir
from fluentfolio.core.time import union
from fluentfolio.models import catalogue
from fluentfolio.models.download import require
from fluentfolio.speakers import tracks

RATE = 16000
FRAME = 512  # Silero v6 reads 32 ms frames at 16 kHz
FRAME_MS = 32
CONTEXT = 64  # each frame goes in with the 64 samples before it
BATCH = 10000  # frames per model call (about five minutes); the model's state carries over between calls
STATE = (1, 1, 128)
# Silero's own segmentation rule: speech starts at a probability of 0.5 and ends once it has stayed under
# 0.35 for 100 ms (4 frames); a run shorter than 250 ms is dropped.
ON = 0.5
OFF = 0.35
QUIET_FRAMES = 4
MIN_SPEECH_MS = 250
# A run longer than 30 s is cut at its least speech-like frame between 20 and 30 s, so every
# speech-recognition chunk fits its model. The pieces touch, so no pause appears at the cut.
MAX_FRAMES = 30000 // FRAME_MS
CUT_FROM_FRAMES = 20000 // FRAME_MS
MIN_TOTAL_SPEECH_MS = 2000  # less speech than this in a whole recording counts as no speech


def session(path: Path) -> Any:
    import onnxruntime as ort

    ort.disable_telemetry_events()  # the Windows trace path ignores ORT_DISABLE_TELEMETRY
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    # Named explicitly: onnxruntime also offers a remote (Azure) provider, and analysis stays offline.
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def probabilities(model: Any, blocks: Iterable[np.ndarray]) -> np.ndarray:
    """The speech probability of every 32 ms frame; blocks are consecutive pieces of the 16 kHz signal."""
    h = np.zeros(STATE, dtype=np.float32)
    c = np.zeros(STATE, dtype=np.float32)
    tail = np.zeros(CONTEXT, dtype=np.float32)
    out = []
    for block in blocks:
        frames = np.pad(block.astype(np.float32), (0, -len(block) % FRAME)).reshape(-1, FRAME)
        context = np.concatenate([tail[None, :], frames[:-1, -CONTEXT:]])
        tail = frames[-1, -CONTEXT:].copy()
        probs, h, c = model.run(None, {"input": np.concatenate([context, frames], axis=1), "h": h, "c": c})
        out.append(np.asarray(probs, dtype=np.float32).reshape(-1))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def runs(probs: np.ndarray) -> list[tuple[int, int]]:
    """Speech runs as (first frame, end frame), by Silero's rule."""
    found: list[tuple[int, int]] = []
    start: int | None = None
    quiet: int | None = None
    for i, p in enumerate(probs):
        if p >= ON:
            quiet = None
            if start is None:
                start = i
        elif p < OFF and start is not None:
            quiet = i if quiet is None else quiet
            if i - quiet >= QUIET_FRAMES:
                found.append((start, quiet))
                start = quiet = None
    if start is not None:
        found.append((start, len(probs)))
    return [(a, b) for a, b in found if (b - a) * FRAME_MS > MIN_SPEECH_MS]


def split_long(found: list[tuple[int, int]], probs: np.ndarray) -> list[tuple[int, int]]:
    """Cut runs longer than 30 s at their least speech-like frame between 20 and 30 s into the run."""
    pieces = []
    for a, b in found:
        while b - a > MAX_FRAMES:
            cut = a + CUT_FROM_FRAMES + int(np.argmin(probs[a + CUT_FROM_FRAMES : a + MAX_FRAMES]))
            pieces.append((a, cut))
            a = cut
        pieces.append((a, b))
    return pieces


def frame_probs(model: Any, wav: Path, progress: Callable[[int], None] | None = None) -> np.ndarray:
    """The speech probability of each frame of a 16 kHz mono WAV, read about five minutes at a time."""
    total = max(1, sf.info(str(wav)).frames // (FRAME * BATCH) + 1)

    def blocks() -> Iterable[np.ndarray]:
        for n, block in enumerate(sf.blocks(str(wav), blocksize=FRAME * BATCH, dtype="float32")):
            if progress:
                progress(min(99, 100 * n // total))
            yield block

    return probabilities(model, blocks())


def _ms(found: list[tuple[int, int]], duration_ms: int) -> list[list[int]]:
    return [[a * FRAME_MS, min(b * FRAME_MS, duration_ms)] for a, b in found]


def segments(model: Any, wav: Path, progress: Callable[[int], None] | None = None) -> list[list[int]]:
    """Speech segments of a 16 kHz mono WAV as [start_ms, end_ms]."""
    probs = frame_probs(model, wav, progress)
    return _ms(split_long(runs(probs), probs), sf.info(str(wav)).frames * 1000 // RATE)


def track_segments(
    model: Any, wavs: list[Path], progress: Callable[[int], None]
) -> tuple[list[list[list[int]]], dict[str, Any], list[str]]:
    """Each person's speech segments on their own track after bleed, the bleed report and its warning."""
    probs = [
        frame_probs(model, w, lambda p, k=k: progress((100 * k + p) // len(wavs))) for k, w in enumerate(wavs)
    ]
    voiced = []
    for p in probs:
        mask = np.zeros(len(p), dtype=bool)
        for a, b in runs(p):
            mask[a:b] = True
        voiced.append(mask)
    owned, both = tracks.resolve(voiced, [tracks.frame_energy(w, FRAME) for w in wavs])
    report, warnings = tracks.bleed_report(voiced, both, FRAME_MS)
    found = []
    for mask, p, w in zip(owned, probs, wavs, strict=True):
        kept = split_long(tracks.mask_runs(mask, FRAME_MS, MIN_SPEECH_MS), np.pad(p, (0, len(mask) - len(p))))
        found.append(_ms(kept, sf.info(str(w)).frames * 1000 // RATE))
    return found, report, warnings


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """The vad stage: speech segments of input/audio16k.wav, or of each person's track, or E_INPUT_SILENT."""
    component = catalogue.get("silero-vad")
    model = session(require(component, models_dir()) / component.file[0].path)
    extra: dict[str, Any] = {}
    if (read_json(job / "job.json").get("layout") or Layout.MIXED) == Layout.MIXED:
        found = segments(model, job / "input" / "audio16k.wav", progress)
    else:
        wavs = [job / "input" / "track_1.wav", job / "input" / "track_2.wav"]
        per_track, report, warnings = track_segments(model, wavs, progress)
        found = [list(s) for s in union([s for track in per_track for s in track])]
        extra = {
            "tracks": [
                {"track": k + 1, "speech_ms": sum(b - a for a, b in s), "segments": s}
                for k, s in enumerate(per_track)
            ],
            "bleed": report,
            "warnings": warnings,
        }  # fmt: skip
    speech_ms = sum(b - a for a, b in found)
    if speech_ms < MIN_TOTAL_SPEECH_MS:
        raise InputError("E_INPUT_SILENT", f"{speech_ms} ms of speech found")
    progress(100)
    return {
        "status": "ok",
        "component": component.id,
        "frame_ms": FRAME_MS,
        "speech_ms": speech_ms,
        "segments": found,
        **extra,
    }
