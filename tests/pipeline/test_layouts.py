"""One track per person and two lined-up files, from decoding to word owners, with the real models."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fluentfolio import __version__
from fluentfolio.app import facade
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import app_root, models_dir
from fluentfolio.ingest import upload
from fluentfolio.jobs import loop, store
from fluentfolio.settings.model import Settings
from fluentfolio.settings.store import snapshot

pytestmark = pytest.mark.slow

ON_TRACK = {"E": "S1", "C": "S2"}  # the builder records the examiner on channel 1 or file 1
FIRST = ["media", "vad", "language", "asr", "speakers", "words"]  # the steps a layout changes


def truth_owner(word: dict[str, Any], segments: list[dict[str, Any]]) -> str | None:
    """Whoever the builder had talking longest during the word."""
    best, most = None, 0
    for s in segments:
        shared = min(word["end_ms"], s["end_ms"]) - max(word["start_ms"], s["start_ms"])
        if shared > most:
            best, most = ON_TRACK[s["speaker"]], shared
    return best


def run_layout(
    tmp_path: Path, layout: str, files: list[Path], offset_ms: int | None
) -> tuple[store.JobRecord, Path]:
    """An interview job in this layout, run through its first six steps on the CPU."""
    roots = loop.Roots(app=app_root(), data=tmp_path / "data", models=models_dir())
    settings, now = Settings(), store.utc_now()
    record = store.JobRecord(
        job_id=store.new_job_id(), created_utc=now, updated_utc=now, app_version=__version__,
        mode="interview", layout=layout, offset_ms=offset_ms, asr=settings.asr.model,
        device={"requested": "cpu", "resolved": "cpu"}, settings=snapshot(settings),
    )  # fmt: skip
    job = store.create(roots.data, record)
    record.input = upload.store_input(job, files[0], settings.limits)
    if len(files) == 2:
        record.input["second"] = upload.store_input(job, files[1], settings.limits, stem="source_2")
    store.write(roots.data, record)
    stages = facade.plan("interview", layout)[: len(FIRST)]
    assert [s.id for s in stages] == FIRST
    done = loop.run_job(roots, record.job_id, stages, audio_min=settings.limits.max_min, no_progress_s=300)
    return done, job


@pytest.mark.parametrize("name", ["CS-2TRACK", "CS-2FILE"])
def test_each_track_owns_its_words(
    tmp_path: Path, session_builder: ModuleType, tolerance: Any, name: str
) -> None:
    out = tmp_path / "session"
    truth = session_builder.build_derived(name, out)
    if name == "CS-2TRACK":
        layout, files, origins = "per-track", [out / "session.wav"], ["channel 1", "channel 2"]
    else:  # the candidate's file starts 1.5 s after the examiner's
        layout, files, origins = (
            "two-files",
            [out / "examiner.wav", out / "candidate.wav"],
            ["file 1", "file 2"],
        )
    record, job = run_layout(tmp_path, layout, files, truth.get("offset_ms"))
    assert [(s.id, s.status) for s in record.stages] == [(i, "ok") for i in FIRST], record.stages
    media = read_json(job / "stages" / "media.json")
    assert (media["layout"], [t["from"] for t in media["tracks"]]) == (layout, origins)
    assert (
        abs(media["duration_ms"] - truth["duration_ms"]) <= 10 and "W_TRACK_LENGTHS" not in media["warnings"]
    )
    # The other person leaks in 20 dB quieter: the louder track wins each frame, with no bleed warning.
    vad = read_json(job / "stages" / "vad.json")
    speakers = read_json(job / "stages" / "speakers.json")
    assert "W_TRACK_BLEED" not in vad["warnings"] and speakers["backend"] == "tracks"
    assert [(s["id"], s["kind"]) for s in speakers["speakers"]] == [("S1", "main"), ("S2", "main")]
    words = read_json(job / "stages" / "words.json")["words"]
    talk = {k: sum(s["end_ms"] - s["start_ms"] for s in truth["segments"] if s["speaker"] == k) for k in "EC"}
    found = [s["speech_ms"] for s in speakers["speakers"]]
    print(f"{name}: {len(words)} words; speech ms {found}, truth {talk}")
    # With one track per person the track is the person, so word ownership is role accuracy by speech time.
    timed = [(w["end_ms"] - w["start_ms"], w["speaker"] == truth_owner(w, truth["segments"])) for w in words]
    share = sum(ms for ms, right in timed if right) / sum(ms for ms, _ in timed)
    print(f"{name}: words owned by the right person {share:.4f} of word time")
    assert share >= tolerance("pipeline", "role_accuracy_interview_min")
    (prep,) = [s for s in truth["silences"] if s["label"] == "preparation"]
    # A word is in the silence when its middle is: the first word after it may start a little early.
    inside = [w for w in words if prep["start_ms"] <= (w["start_ms"] + w["end_ms"]) // 2 < prep["end_ms"]]
    assert len(inside) <= tolerance("pipeline", "words_in_preparation_silence")
