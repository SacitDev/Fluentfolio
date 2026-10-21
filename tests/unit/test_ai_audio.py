"""The optional audio for an AI chat: whole pieces up to 9.5 minutes, the candidate's own turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
import soundfile as sf

from fluentfolio.core.files import write_json
from fluentfolio.media import slices

RATE = 16_000
PRIMING = 1024  # samples the AAC encoder puts before the audio: one frame
PIECES = [
    {"wav_ms": 0, "orig_ms": 1000, "ms": 4000},
    {"wav_ms": 4000, "orig_ms": 9000, "ms": 3000},
    {"wav_ms": 7000, "orig_ms": 20000, "ms": 2000},
]


@pytest.mark.parametrize(("limit", "kept"), [(9_599, 2), (9_600, 3), (9_601, 3)])
def test_whole_pieces_with_a_gap_between_fit_up_to_the_limit(limit: int, kept: int) -> None:
    # By hand: 0-4000, then 300 ms of silence, 4300-7300, silence, 7600-9600.
    starts = [0, 4300, 7600]
    expected = [{**p, "audio_ms": s} for p, s in zip(PIECES, starts, strict=True)][:kept]
    assert slices.fit(PIECES, limit) == expected


def test_pieces_stay_in_time_order_and_one_too_long_is_cut_to_the_limit() -> None:
    longer = [{"wav_ms": 0, "orig_ms": 0, "ms": 4000}, {"wav_ms": 4000, "orig_ms": 5000, "ms": 9000},
              {"wav_ms": 13_000, "orig_ms": 15_000, "ms": 1000}]  # fmt: skip
    assert [p["orig_ms"] for p in slices.fit(longer, 10_000)] == [0]  # the 1 s piece after it is not taken
    assert slices.fit([{"wav_ms": 0, "orig_ms": 0, "ms": 700_000}], 569_500) == [
        {"wav_ms": 0, "orig_ms": 0, "ms": 569_500, "audio_ms": 0}
    ]


def tone(hz: float, ms: int) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * hz * np.arange(ms * RATE // 1000) / RATE)).astype(np.float32)


def mixed_job(tmp_path: Path, audio: bool) -> Path:
    """A recording where the candidate is a 440 Hz tone and the other person a 1000 Hz tone between."""
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    x = np.concatenate([np.zeros(RATE, np.float32), tone(440, 2000), tone(1000, 2000), tone(440, 2000),
                        np.zeros(RATE, np.float32)])  # fmt: skip
    sf.write(job / "input" / "audio16k.wav", x, RATE, subtype="PCM_16")
    write_json(job / "stages" / "media.json", {"music_spans": []})
    write_json(job / "stages" / "vad.json", {"status": "ok", "segments": []})
    structure = {"candidates": [{"id": "candidate", "speaker": "S1"}],
                 "speech": {"candidate": [[1000, 3000], [5000, 7000]]}}  # fmt: skip
    write_json(job / "stages" / "structure.json", structure)
    write_json(job / "job.json", {"settings": {"export": {"audio": audio}}})
    write_json(job / "stages" / "candidate_audio.json", slices.run(job, lambda pct: None))
    return job


def decoded(path: Path) -> tuple[np.ndarray, list[str], float]:
    with av.open(str(path)) as container:
        kinds = [s.type for s in container.streams]
        stream = container.streams.audio[0]
        samples = np.concatenate([f.to_ndarray().reshape(-1) for f in container.decode(stream)])
        return samples, kinds, container.duration / 1e6


def power(x: np.ndarray, hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    return float(spectrum[round(hz * len(x) / RATE)])


def test_the_audio_holds_the_candidates_pieces_with_a_silent_gap_and_no_other_voice(tmp_path: Path) -> None:
    job = mixed_job(tmp_path, audio=True)
    result = slices.for_ai_audio(job, lambda pct: None)
    row = result["candidates"]["candidate"]
    assert (result["status"], row["file"], row["duration_ms"]) == ("ok", "for_ai_audio.m4a", 4300)
    assert row["pieces"] == [{"audio_ms": 0, "orig_ms": 1000, "ms": 2000},
                             {"audio_ms": 2300, "orig_ms": 5000, "ms": 2000}]  # fmt: skip
    samples, kinds, seconds = decoded(job / "export" / "for_ai" / "for_ai_audio.m4a")
    assert kinds == ["audio"]  # no video, no second stream
    # AAC works in frames of 1024 samples: one frame of priming comes first and the last frame is padded.
    n = 4300 * RATE // 1000
    assert seconds == pytest.approx((n + PRIMING) / RATE, abs=1e-3)
    assert n + PRIMING <= len(samples) <= n + 2 * PRIMING
    # The other person's 1000 Hz tone is in the recording, between the two pieces, and must not be here.
    assert power(samples, 1000) < 1e-4 * power(samples, 440)
    # The gap between the pieces is silent (away from the encoder's smearing at its edges).
    gap = samples[PRIMING + 2100 * RATE // 1000 : PRIMING + 2200 * RATE // 1000]
    assert float(np.sqrt(np.mean(gap**2))) < 0.01


def test_with_the_setting_off_no_audio_is_made_and_an_old_file_is_removed(tmp_path: Path) -> None:
    job = mixed_job(tmp_path, audio=False)
    folder = job / "export" / "for_ai"
    folder.mkdir(parents=True)
    (folder / "for_ai_audio.m4a").write_bytes(b"from an earlier run")
    assert slices.for_ai_audio(job, lambda pct: None) == {"status": "disabled", "candidates": {}}
    assert not (folder / "for_ai_audio.m4a").exists()


def test_a_file_for_people_no_longer_in_the_job_is_removed(tmp_path: Path) -> None:
    job = mixed_job(tmp_path, audio=True)
    folder = job / "export" / "for_ai"
    folder.mkdir(parents=True)
    (folder / "for_ai_audio_person1.m4a").write_bytes(b"from a role-swap run")
    slices.for_ai_audio(job, lambda pct: None)
    assert sorted(p.name for p in folder.iterdir()) == ["for_ai_audio.m4a"]


def test_a_long_answer_gives_a_file_that_plays_for_at_most_9_5_minutes(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    quiet = (0.01 * np.random.default_rng(7).standard_normal(600 * RATE)).astype(np.float32)  # 10 min
    sf.write(job / "input" / "candidate_candidate.wav", quiet, RATE, subtype="PCM_16")
    write_json(job / "input" / "candidate_candidate.map.json", {"rate": RATE, "pieces": [
        {"wav_ms": 0, "orig_ms": 0, "ms": 600_000}]})  # fmt: skip
    made: dict[str, Any] = {
        "wav": "input/candidate_candidate.wav",
        "map": "input/candidate_candidate.map.json",
    }
    write_json(job / "stages" / "candidate_audio.json", {"status": "ok", "candidates": {"candidate": made}})
    write_json(job / "job.json", {"settings": {"export": {"audio": True}}})
    row = slices.for_ai_audio(job, lambda pct: None)["candidates"]["candidate"]
    _, _, seconds = decoded(job / "export" / "for_ai" / "for_ai_audio.m4a")
    # 569.5 s of samples and the priming frame: the file plays for 569.564 s, inside the 570 s limit.
    assert row["duration_ms"] == 569_500
    assert seconds == pytest.approx((569_500 * RATE // 1000 + PRIMING) / RATE, abs=1e-3) and seconds <= 570.0
