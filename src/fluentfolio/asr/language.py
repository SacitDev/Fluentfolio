"""The language check: Whisper tiny names the language of each long stretch of speech, on the CPU.

A stretch is only marked, never translated or removed: the transcript shows `[non-English]` before its
words, and the export says how long the marked speech is.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import soundfile as sf

from fluentfolio.core.files import read_json
from fluentfolio.core.paths import models_dir
from fluentfolio.models import catalogue
from fluentfolio.models.download import require

MODEL = "whisper-tiny"
RATE = 16000
MIN_SEGMENT_MS = 3000  # shorter speech gives the model too little to tell languages apart
# Marked only when English is very unlikely and one other language clearly likely, so that accented
# English stays unmarked. Generated noise and silence never reached 0.5 for any language.
ENGLISH_MAX = 0.05
OTHER_MIN = 0.70


def marked(probability: float, english: float) -> bool:
    """The mark-only rule: the most likely language clearly likely, and English very unlikely."""
    # With both bounds, the most likely language can never be English itself.
    return probability >= OTHER_MIN and english <= ENGLISH_MAX


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """The language step: one judgement per VAD segment of 3 s or more, per track when there are tracks."""
    record = read_json(job / "job.json")
    if not record["settings"].get("components", {}).get(MODEL, True):
        return {"status": "disabled", "segments": [], "non_english_ms": 0}
    component = catalogue.get(MODEL)
    folder = require(component, models_dir())
    from faster_whisper import WhisperModel

    model = WhisperModel(str(folder), device="cpu", compute_type="int8", local_files_only=True)
    vad = read_json(job / "stages" / "vad.json")
    tracks = vad.get("tracks") or [{"track": None, "segments": vad["segments"]}]
    todo = [(t["track"], a, b) for t in tracks for a, b in t["segments"] if b - a >= MIN_SEGMENT_MS]
    segments = []
    for n, (track, a, b) in enumerate(todo):
        wav = job / "input" / ("audio16k.wav" if track is None else f"track_{track}.wav")
        audio, _ = sf.read(str(wav), start=a * RATE // 1000, stop=b * RATE // 1000, dtype="float32")
        language, probability, ranked = model.detect_language(audio)
        english = dict(ranked).get("en", 0.0)
        row = {
            "start_ms": a, "end_ms": b, "language": language, "probability": round(probability, 3),
            "english": round(english, 3), "marked": marked(probability, english),
        }  # fmt: skip
        if track is not None:
            row["track"] = track
        segments.append(row)
        progress(100 * (n + 1) // len(todo))
    return {
        "status": "ok",
        "provenance": {
            "model": MODEL,
            "revision": component.revision,
            "device": "cpu",
            "engine": {name: metadata.version(name) for name in ("faster-whisper", "ctranslate2")},
            "settings": {
                "min_segment_ms": MIN_SEGMENT_MS,
                "english_max": ENGLISH_MAX,
                "other_min": OTHER_MIN,
            },
        },
        "segments": segments,
        "non_english_ms": sum(s["end_ms"] - s["start_ms"] for s in segments if s["marked"]),
    }
