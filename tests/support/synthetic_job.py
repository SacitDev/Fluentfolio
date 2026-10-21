"""A job folder with hand-made media, voice-detector and recognition results.

The steps after speech recognition then run through the real job loop without audio or models. The test
chooses every time and word, so the values it expects come from the test, not from the code.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fluentfolio import __version__
from fluentfolio.app import facade
from fluentfolio.asr.normalize import word
from fluentfolio.core.files import write_json
from fluentfolio.jobs import loop, store
from fluentfolio.setup.device import NO_GPU_STEP

JOB_ID = "20261013-120000-abcd"
MODEL = "parakeet-tdt-0.6b-v2"
REVISION = "0123456789abcdef0123456789abcdef01234567"
STAMP = "2026-10-13T12:00:00Z"
Spoken = Sequence[tuple[str, int, int, int]]  # text as the model wrote it, start ms, end ms, chunk


def words(spoken: Spoken, model: str = MODEL, confidence: float | None = None) -> list[dict[str, Any]]:
    """Words in the recogniser's own form, each with the chunk it was decoded in."""
    out = []
    for raw, start, end, chunk in spoken:
        w = word(raw, start, end, confidence, model)
        assert w is not None, raw
        out.append({**w, "chunk": chunk})
    return out


def roots(base: Path) -> loop.Roots:
    return loop.Roots(app=base / "app", data=base / "data", models=base / "models")


def make(
    base: Path,
    spoken: Spoken,
    segments: Sequence[tuple[int, int]],
    duration_ms: int,
    *,
    display_name: str = "practice.wav",
    style: str = "verbatim-best-effort",
    model: str = MODEL,
    marked: Sequence[tuple[int, int]] = (),
    music: Sequence[tuple[int, int]] = (),
    confidence: float | None = None,
) -> Path:
    """A solo job whose steps up to the speaker step have finished; returns its folder.

    `marked` holds the speech the language check took for another language, `music` the stretches where the
    media step heard background music, `confidence` the speech model's confidence in every word.
    """
    record = store.JobRecord(
        job_id=JOB_ID, created_utc=STAMP, updated_utc=STAMP, app_version=__version__, mode="solo",
        layout="mixed", asr=model, device={"requested": "auto", "resolved": "cpu", "reason": NO_GPU_STEP},
        input={"display_name": display_name, "bytes": 1000, "container": "wav", "sha256": "0" * 64},
    )  # fmt: skip
    for step, wall in (("media", 400), ("vad", 900), ("language", 700), ("asr", 5200), ("speakers", 1300)):
        record.stages.append(store.StageRecord(
            id=step, status="ok", started_utc=STAMP, ended_utc=STAMP, device="cpu", wall_ms=wall,
            peak_ram_mb=300,
        ))  # fmt: skip
    job = store.create(roots(base).data, record)
    quality = {
        "class": "clean",
        "snr_db": 31.5,
        "clipping_pct": 0.0,
        "level_dbfs": -22.0,
        "noise_dbfs": -53.5,
    }
    write_json(job / "stages" / "media.json", {
        "status": "ok", "duration_ms": duration_ms, "source": {"container": "wav"}, "quality": quality,
        "music_spans": [list(s) for s in music], "warnings": ["W_BACKGROUND_MUSIC"] if music else [],
    })  # fmt: skip
    write_json(job / "stages" / "vad.json", {
        "status": "ok", "component": "silero-vad", "frame_ms": 32,
        "speech_ms": sum(b - a for a, b in segments), "segments": [list(s) for s in segments],
    })  # fmt: skip
    checked = [{"start_ms": a, "end_ms": b, "language": "en", "probability": 0.99, "english": 0.99,
                "marked": False} for a, b in segments if b - a >= 3000]  # fmt: skip
    for row in checked:
        if (row["start_ms"], row["end_ms"]) in marked:
            row.update(language="es", probability=0.95, english=0.01, marked=True)
    write_json(job / "stages" / "language.json", {
        "status": "ok", "provenance": {"model": "whisper-tiny", "revision": REVISION, "device": "cpu"},
        "segments": checked, "non_english_ms": sum(b - a for a, b in marked),
    })  # fmt: skip
    provenance = {
        "model": model, "revision": REVISION, "transcript_style": style, "punctuation": "machine",
        "filler_capture": "best_effort" if style == "verbatim-best-effort" else "lower_bound",
    }  # fmt: skip
    write_json(
        job / "stages" / "asr.json",
        {"status": "ok", "provenance": provenance, "words": words(spoken, model, confidence)},
    )
    write_json(job / "stages" / "speakers.json", {
        "status": "ok", "backend": "single", "mode": "solo", "version": "sherpa-onnx 1.13.8",
        "second_voice": {"checked_by": "sherpa-onnx-diarizer", "voices": 1, "share": 0.0}, "warnings": [],
    })  # fmt: skip
    return job


LEFT_OUT = ("candidate_audio", "ai_audio", "phones", "alignment", "praat", "swiftf0", "opensmile", "nlp")


def plan(mode: str) -> list[loop.Stage]:
    """The mode's plan without the steps that need audio or a model; a hand-made job has neither."""
    return [s for s in facade.plan(mode) if s.id not in LEFT_OUT]


def finish(base: Path) -> store.JobRecord:
    """Run the rest of the solo plan through the job loop, which starts after the finished steps."""
    return loop.run_job(roots(base), JOB_ID, plan("solo"), audio_min=5, no_progress_s=60)
