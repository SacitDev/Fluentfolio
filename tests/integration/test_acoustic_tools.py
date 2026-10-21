"""Praat, SwiftF0 and openSMILE on generated harmonic tones, whose pitch is known from how they were made."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from fluentfolio.analyzers import opensmile, praat, swiftf0
from fluentfolio.analyzers.measures import acoustic, fusion
from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs import loop
from fluentfolio.media import slices

RATE = 16_000
Limit = Callable[[str, str], float]


def harmonic(freq: np.ndarray, amp: float = 0.3) -> np.ndarray:
    """A tone with 1/k harmonics up to 4 kHz whose pitch follows `freq` (Hz per sample), phase unbroken."""
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    x = sum(np.sin(k * phase) / k for k in range(1, int(4000 // freq.max()) + 1))
    return (amp * x / np.abs(x).max()).astype(np.float32)


def steady(hz: float, seconds: float) -> np.ndarray:
    return np.full(int(RATE * seconds), hz)


def tone_job(
    job: Path, x: np.ndarray, quality: str = "clean", settings: dict[str, Any] | None = None
) -> Path:
    """A job folder whose one answer is the whole tone, with the candidate audio step already run."""
    (job / "input").mkdir(parents=True)
    sf.write(job / "input" / "audio16k.wav", x, RATE, subtype="PCM_16")
    ms = len(x) * 1000 // RATE
    write_json(job / "job.json", {"settings": settings or {}})
    write_json(
        job / "stages" / "media.json", {"status": "ok", "quality": {"class": quality}, "music_spans": []}
    )
    write_json(job / "stages" / "vad.json", {"status": "ok", "segments": [[0, ms]]})
    write_json(job / "stages" / "structure.json", {
        "candidates": [{"id": "candidate", "speaker": "S1"}], "speech": {"candidate": [[0, ms]]},
        "answers": [{"id": "A01", "candidate": "candidate", "part": 2, "start_ms": 0, "end_ms": ms}],
        "utterances": [{"id": "U001", "answer_id": "A01", "start_ms": 0, "end_ms": ms}],
    })  # fmt: skip
    write_json(job / "stages" / "candidate_audio.json", slices.run(job, lambda pct: None))
    return job


def test_praat_hears_a_120_hz_tone_as_a_flat_level_stretch(tmp_path: Path, tolerance: Limit) -> None:
    found = praat.run(tone_job(tmp_path / "job", harmonic(steady(120.0, 12.0))), lambda pct: None)
    row = found["candidates"]["candidate"]
    pitch = row["scopes"]["all"]
    assert (found["status"], pitch["used"]) == ("ok", ["A01"])
    assert abs(acoustic.hertz(pitch["p50_st"]) - 120.0) <= tolerance("integration", "praat_tone_hz")
    (unit,) = row["units"]
    assert (unit["flat"], unit["ending"]) == (True, "level")
    voice = row["voice"]["A01"]
    assert voice["runs"] == 1 and voice["run_ms"] > 11_000
    assert voice["jitter_pct"] <= tolerance("integration", "praat_tone_jitter_pct_max")
    assert voice["shimmer_pct"] <= tolerance("integration", "praat_tone_shimmer_pct_max")
    assert voice["hnr_db"] >= tolerance("integration", "praat_tone_hnr_db_min")


def test_praat_tracks_a_step_of_one_octave_as_12_semitones(tolerance: Limit) -> None:
    import parselmouth

    x = harmonic(np.concatenate([steady(100.0, 1.0), steady(200.0, 1.0)]))
    _, hz, _, _ = praat.track(parselmouth.Sound(x.astype(np.float64), RATE))
    st = [acoustic.semitones(f) for f in hz if f is not None]
    # Half the frames sit at each pitch, so the 20th and 80th percentiles fall one on each step.
    spread = acoustic.percentile(st, 80) - acoustic.percentile(st, 20)
    assert abs(spread - 12.0) <= tolerance("integration", "praat_step_st")  # 12 x log2(200 / 100)


def test_praat_hears_a_rise_at_the_end(tmp_path: Path) -> None:
    # The last 0.5 s glides from 120 to 160 Hz: the last 300 ms rise by 12 x log2(160 / 136) = 2.8 st.
    glide = np.concatenate([steady(120.0, 11.5), np.linspace(120.0, 160.0, RATE // 2)])
    found = praat.run(tone_job(tmp_path / "job", harmonic(glide)), lambda pct: None)
    assert found["candidates"]["candidate"]["units"][0]["ending"] == "rise"


def test_voice_quality_needs_clean_audio(tmp_path: Path) -> None:
    found = praat.run(tone_job(tmp_path / "job", harmonic(steady(120.0, 12.0)), "noisy"), lambda pct: None)
    row = found["candidates"]["candidate"]
    assert row["voice"]["A01"]["missing"]["code"] == praat.NOT_CLEAN
    assert row["scopes"]["all"]["missing"] is None  # pitch is still measured, at low reliability


@pytest.mark.parametrize(
    "settings",
    [{"components": {"praat": False}}, {"analyzers": {"pitch": False, "voice-quality": False}}],
)
def test_praat_switched_off_does_not_run(tmp_path: Path, settings: dict[str, Any]) -> None:
    job = tone_job(tmp_path / "job", harmonic(steady(120.0, 1.0)), settings=settings)
    found = praat.run(job, lambda pct: None)
    assert (found["status"], found["candidates"]) == ("disabled", {})


def test_without_candidate_audio_the_tools_skip(tmp_path: Path) -> None:
    job = tone_job(tmp_path / "job", harmonic(steady(120.0, 1.0)))
    write_json(job / "stages" / "candidate_audio.json", {"status": "failed", "code": "E_INTERNAL"})
    assert praat.run(job, lambda pct: None)["status"] == "skipped"
    assert swiftf0.run(job, lambda pct: None)["status"] == "skipped"


def test_swiftf0_hears_a_120_hz_tone(tmp_path: Path, tolerance: Limit) -> None:
    found = swiftf0.run(tone_job(tmp_path / "job", harmonic(steady(120.0, 12.0))), lambda pct: None)
    pitch = found["candidates"]["candidate"]["scopes"]["all"]
    assert (found["status"], pitch["used"]) == ("ok", ["A01"])
    assert abs(acoustic.hertz(pitch["p50_st"]) - 120.0) <= tolerance("integration", "swiftf0_tone_hz")


# openSMILE runs as the product runs it: the stage child, in a process of its own, on a job in the data
# folder. The tests' consent record lives only in each test's temporary data folder.
JOB_ID = "20261014-120000-abcd"
CONSENT = {"opensmile": {"utc": "2026-10-14T00:00:00Z", "source": "cli-tty"}}


def home() -> Path:
    return Path(os.environ["FLUENTFOLIO_HOME"])


def child(stage: str) -> dict[str, Any]:
    """Run one step's child on the test job and return its result file."""
    done = subprocess.run(
        loop.child_argv(stage, JOB_ID), capture_output=True, text=True, timeout=300, env=dict(os.environ)
    )
    assert done.returncode == 0, done.stderr[-2000:]
    found: dict[str, Any] = read_json(home() / "jobs" / JOB_ID / "analyzers" / f"{stage}.json")
    return found


def test_opensmile_without_consent_is_disabled_and_runs_nothing() -> None:
    tone_job(home() / "jobs" / JOB_ID, harmonic(steady(120.0, 3.0)))
    found = child("opensmile")
    assert (found["status"], found["code"]) == ("disabled", opensmile.CONSENT_MISSING)
    assert found["candidates"] == {} and "provenance" not in found


def test_opensmile_hears_a_120_hz_tone_the_same_way_twice(tolerance: Limit) -> None:
    write_json(home() / "consent.json", CONSENT)
    tone_job(home() / "jobs" / JOB_ID, harmonic(steady(120.0, 12.0)))
    first, again = child("opensmile"), child("opensmile")
    assert first["candidates"] == again["candidates"]
    assert (first["status"], first["analyzers"]) == ("ok", ["pitch", "loudness"])
    found = first["candidates"]["candidate"]
    pitch = found["scopes"]["all"]
    assert pitch["used"] == ["A01"]
    expected = 12 * math.log2(120 / 27.5)  # 25.51 semitones on the eGeMAPS scale
    assert abs(pitch["p50_st"] - expected) <= tolerance("integration", "opensmile_tone_st")
    assert len(found["functionals"]["all"]) == 88
    assert (found["loudness"]["all"]["missing"], found["loudness"]["all"]["speech_ms"]) == (None, 12_000)


def test_the_three_pitch_tools_agree_on_a_120_hz_tone(tolerance: Limit) -> None:
    write_json(home() / "consent.json", CONSENT)
    job = tone_job(home() / "jobs" / JOB_ID, harmonic(steady(120.0, 12.0)))
    found = {
        "praat": praat.run(job, lambda pct: None), "swiftf0": swiftf0.run(job, lambda pct: None),
        "opensmile": child("opensmile"),
    }  # fmt: skip
    rows = {tool: f["candidates"]["candidate"]["scopes"]["all"] for tool, f in found.items()}
    pitch = fusion.fuse(rows, {tool: f["provenance"]["version"] for tool, f in found.items()})
    assert [s.tool for s in pitch.median.sources] == ["praat", "swiftf0", "opensmile"]
    assert (pitch.median.agreement, pitch.range.agreement, pitch.median.nc_input) == ("agree", "agree", True)
    expected = 12 * math.log2(120 / 27.5)  # 25.51 semitones
    assert abs(pitch.statistics["p50"]["fused_st"] - expected) <= tolerance("integration", "fusion_agree_st")


def test_opensmile_hears_6_db_more_as_louder() -> None:
    write_json(home() / "consent.json", CONSENT)
    x = harmonic(steady(120.0, 3.0))
    job = home() / "jobs" / JOB_ID
    levels = []
    for gain in (1.0, 2.0):  # twice the amplitude is 6 dB more
        shutil.rmtree(job, ignore_errors=True)
        tone_job(job, x * gain)
        found = child("opensmile")["candidates"]["candidate"]["functionals"]["all"]
        levels.append(found["loudness_sma3_amean"])
    assert levels[1] > levels[0]


def test_loudness_is_not_measured_on_noisy_audio() -> None:
    write_json(home() / "consent.json", CONSENT)
    tone_job(home() / "jobs" / JOB_ID, harmonic(steady(120.0, 3.0)), "noisy")
    loudness = child("opensmile")["candidates"]["candidate"]["loudness"]["all"]
    assert loudness["missing"]["code"] == "audio_not_suited" and "variation" not in loudness


def test_a_gpl_module_in_the_process_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "parselmouth.praat", types.ModuleType("parselmouth.praat"))
    assert "parselmouth.praat" in opensmile.gpl_loaded()
