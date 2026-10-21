"""The background-music heuristic on generated signals, whose music or lack of it is known by construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from fluentfolio.media import quality

RATE = 16_000


def chord(n: int, amp: float) -> np.ndarray:
    """A held C major chord, three notes with 8 harmonics each, peak `amp`."""
    t = np.arange(n) / RATE
    x = sum(np.sin(2 * np.pi * f * h * t) / h for f in (261.63, 329.63, 392.0) for h in range(1, 9))
    return amp * x / np.abs(x).max()


def phrases(n: int, seed: int) -> np.ndarray:
    """Speech-like noise: 2 s phrases of bursts (250 ms on, 150 ms off), then a 1 s pause; RMS 0.1 when on."""
    t = np.arange(n) / RATE
    on = (t % 0.4 < 0.25) & (t % 3.0 < 2.0)
    return np.random.default_rng(seed).standard_normal(n) * 0.1 * on


def wav(path: Path, x: np.ndarray) -> Path:
    sf.write(path, x.astype(np.float32), RATE, subtype="FLOAT")
    return path


def samples(windows: int) -> int:
    """The length that holds exactly this many 3 s windows, 1 s apart, of 64 ms frames 10 ms apart."""
    frames = 300 + (windows - 1) * 100
    return (frames - 1) * 160 + 1024


def test_the_threshold_equals_its_tolerance_entry(tolerance: Any) -> None:
    assert quality.MUSIC_STABILITY_MIN == tolerance("media", "music_stability_min")


@pytest.mark.parametrize(
    "windows", [1, quality.CHUNK - 1, quality.CHUNK, quality.CHUNK + 1, 2 * quality.CHUNK + 1]
)
def test_a_held_chord_is_music_in_every_window_across_the_reads(tmp_path: Path, windows: int) -> None:
    path = wav(tmp_path / "chord.wav", chord(samples(windows), 0.3))
    found = quality.music_scores(path)
    assert [ms for ms, _ in found] == [k * 1000 for k in range(windows)]
    assert min(score for _, score in found) >= quality.MUSIC_STABILITY_MIN
    assert quality.music_spans(path) == [[0, (windows - 1) * 1000 + 3000]]


def test_a_recording_shorter_than_one_window_or_quieter_than_minus_60_dbfs_is_not_judged(
    tmp_path: Path,
) -> None:
    assert quality.music_scores(wav(tmp_path / "short.wav", chord(samples(1) - 1, 0.3))) == []
    # Peak 0.0012 is an RMS under 0.001 (-60 dBFS) for this chord.
    quiet = chord(samples(5), 0.0012)
    assert np.sqrt(np.mean(quiet**2)) < quality.QUIET_RMS
    assert quality.music_scores(wav(tmp_path / "quiet.wav", quiet)) == []


@pytest.mark.parametrize("floor", ["digital silence", "white noise 40 dB down"])
def test_speech_like_phrases_are_not_music_until_a_chord_fills_their_pauses(
    tmp_path: Path, floor: str
) -> None:
    n = samples(17)  # 19.06 s: windows start at 0-16 s, and the last ends at 19 s
    x = phrases(n, 1)
    if floor != "digital silence":
        x += np.random.default_rng(2).standard_normal(n) * 0.001
    assert quality.music_spans(wav(tmp_path / "speech.wav", x)) == []
    # The chord at peak 0.035 has an RMS of 0.0106, 19.5 dB under the bursts; its notes fill the pauses.
    mixed = quality.music_spans(wav(tmp_path / "mixed.wav", x + chord(n, 0.035)))
    assert mixed == [[0, 19_000]]
