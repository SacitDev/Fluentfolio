"""The experiments' data rules: non-speech levels, clip preparation, set B's lines and replacements."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

EVAL = Path(__file__).resolve().parents[2] / "eval"
CLIPS = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
RATE = 16000
FRAME = 160  # 10 ms


def _load(name: str) -> Any:
    # eval/ is a folder of scripts, not a package; a script that imports a sibling needs it registered.
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


_load("asr_metrics")
sets = _load("sets")


def dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2))))


def tone(ms: int, amplitude: float = 0.3, hz: float = 220.0) -> np.ndarray:
    t = np.arange(ms * RATE // 1000) / RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def quiet(ms: int, db: float | None = None) -> np.ndarray:
    """Digital silence, or noise whose every sample stays under the given level."""
    n = ms * RATE // 1000
    if db is None:
        return np.zeros(n, dtype=np.float32)
    return np.random.default_rng(1).uniform(-1, 1, n).astype(np.float32) * 10 ** (db / 20)


@pytest.mark.parametrize(("kind", "level"), [("silence", -60), ("pink_noise", -30), ("music", -20)])
def test_each_non_speech_clip_has_its_registered_length_and_level(kind: str, level: int) -> None:
    x = sets.non_speech(kind, 3)
    assert x.dtype == np.float32 and len(x) == 3 * RATE
    assert dbfs(x) == pytest.approx(level, abs=0.01)


def test_the_non_speech_clips_are_the_same_every_time_and_an_unknown_kind_is_refused() -> None:
    assert np.array_equal(sets.non_speech("pink_noise", 2), sets.non_speech("pink_noise", 2))
    with pytest.raises(ValueError, match="unknown non-speech clip"):
        sets.non_speech("white", 2)


def test_pink_noise_loses_3_db_of_power_per_octave() -> None:
    x = sets.non_speech("pink_noise", 20).astype(np.float64)
    segments = x[: len(x) // 4096 * 4096].reshape(-1, 4096) * np.hanning(4096)
    power = (np.abs(np.fft.rfft(segments, axis=1)) ** 2).mean(axis=0)
    hz = np.fft.rfftfreq(4096, 1 / RATE)
    band = (hz >= 100) & (hz <= 4000)
    slope = np.polyfit(np.log10(hz[band]), np.log10(power[band]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.05)  # power falls as 1/f: 10 log10(2) = 3 dB per octave


def test_a_quiet_clip_is_raised_to_minus_20_dbfs_and_a_loud_one_is_left_alone() -> None:
    soft = tone(1000, amplitude=0.01 * np.sqrt(2))  # RMS -40 dBFS
    assert dbfs(sets.level(soft)) == pytest.approx(-20, abs=0.05)
    loud = tone(1000, amplitude=0.4)  # RMS -11 dBFS, peak -8 dBFS
    assert np.array_equal(sets.level(loud), loud)


def test_the_peak_is_held_at_minus_6_dbfs() -> None:
    hot = tone(1000, amplitude=0.9)
    assert np.max(np.abs(sets.level(hot))) == pytest.approx(10 ** (-6 / 20), rel=1e-5)
    # A quiet clip with one loud click is raised only as far as the click allows.
    clicked = tone(1000, amplitude=0.01 * np.sqrt(2))
    clicked[8000] = 0.25
    assert np.max(np.abs(sets.level(clicked))) == pytest.approx(10 ** (-6 / 20), rel=1e-5)


def test_short_fragments_at_the_ends_are_dropped_and_long_silences_cut_to_100_ms() -> None:
    blip = tone(30, amplitude=0.5)
    x = np.concatenate([quiet(300), blip, quiet(200), tone(500), quiet(600), tone(500), quiet(300), blip])
    y = sets.tidy(x)
    # The 600 ms silence loses 500 ms; both 30 ms fragments sit 150 ms or more from the speech.
    assert len(y) == len(x) - 500 * RATE // 1000
    assert not np.any(y[: (300 + 30 + 200) * RATE // 1000])
    assert not np.any(y[-(300 + 30) * RATE // 1000 :])
    start = (300 + 30 + 200) * RATE // 1000
    assert np.array_equal(y[start : start + 8000], x[start : start + 8000])


def test_a_fragment_close_to_the_speech_and_a_silence_with_room_noise_are_kept() -> None:
    blip = tone(30, amplitude=0.5)
    near = np.concatenate([quiet(300), blip, quiet(100), tone(500), quiet(300)])
    assert np.array_equal(sets.tidy(near), near)  # only 100 ms from the speech
    # Room noise at -50 dBFS is quiet (under -40) but not deep (over -55): the pause stays as recorded.
    roomy = np.concatenate([tone(500), quiet(600, db=-50), tone(500)])
    assert np.array_equal(sets.tidy(roomy), roomy)


def test_a_noisy_recording_s_silences_are_cut_up_to_minus_37_dbfs() -> None:
    roomy = np.concatenate([tone(500), quiet(600, db=-50), tone(500)])
    y = sets.tidy(roomy, deep=10 ** (-37 / 20))
    assert len(y) == len(roomy) - 500 * RATE // 1000
    # The 50 ms kept on each side of the cut are the pause's own first and last frames.
    assert np.array_equal(y[: 550 * RATE // 1000], roomy[: 550 * RATE // 1000])
    assert np.array_equal(y[-550 * RATE // 1000 :], roomy[-550 * RATE // 1000 :])
    # Noise at -38.5 dBFS lies over the -40 dBFS quiet line: one 600 ms run only under the wider line.
    hum = np.concatenate([tone(500), quiet(250, db=-50), quiet(100, db=-38.5), quiet(250, db=-50), tone(500)])
    assert np.array_equal(sets.tidy(hum), hum)
    assert len(sets.tidy(hum, deep=10 ** (-37 / 20))) == len(hum) - 500 * RATE // 1000


def test_100_ms_is_kept_around_the_speech() -> None:
    x = np.concatenate([quiet(500), tone(500), quiet(1000)])
    y = sets.margin(x)
    assert len(y) == (100 + 500 + 100) * RATE // 1000
    assert np.array_equal(y, x[400 * RATE // 1000 : 1100 * RATE // 1000])


def test_a_prepared_clip_is_kept_when_the_builder_accepts_it_and_removed_when_not(tmp_path: Path) -> None:
    clip = np.concatenate([quiet(200), tone(800), quiet(600), tone(800), quiet(200)])
    kept = tmp_path / "kept.opus"
    assert sets.prepare(clip, kept, "kept") is True
    x, rate = sf.read(kept, dtype="float32")
    assert rate == RATE
    # 100 ms margins, 800 + 100 + 800 ms inside: the silence was cut to 100 ms.
    assert len(x) / RATE == pytest.approx(1.9, abs=0.02)
    roomy = np.concatenate([tone(800), quiet(300, db=-50), tone(800)])
    refused = tmp_path / "refused.opus"
    assert sets.prepare(roomy, refused, "refused") is False
    assert not refused.exists()
    # A clip whose own pauses are kept is never checked against the limit.
    assert sets.prepare(roomy, refused, "refused", tidied=False) is True and refused.exists()


def test_the_closest_unused_utterance_comes_first_and_ties_go_to_the_lower_id() -> None:
    seconds = {"7-1-3": 5.0, "7-1-1": 7.0, "7-1-2": 3.0, "7-2-0": 5.5, "8-1-0": 5.0}
    # 5.0 is used; 5.5 is 0.5 away; 3.0 and 7.0 tie at 2.0, so 7-1-1 goes before 7-1-2.
    assert sets.closest(seconds, "7", {"7-1-3"}, 5.0) == ["7-2-0", "7-1-1", "7-1-2"]
    assert sets.closest(seconds, "9", set(), 5.0) == []


def test_set_b_lines_keep_set_a_texts_in_the_mapped_voices() -> None:
    clips = json.loads((CLIPS / "clips.json").read_text(encoding="utf-8"))
    lines = sets.kokoro_lines()
    names = [x["name"] for x in lines]
    assert names == sorted(set(names))
    old = {"bf_emma": "bf_isabella", "bm_george": "bm_lewis", "af_bella": "af_nicole"}  # the registered map
    for x in lines:
        assert x["text"] == clips[x["name"]]["text"]
        assert x["voice"] == old[clips[x["name"]]["source"].split()[2]]
    by_name = {x["name"]: x["voice"] for x in lines}
    # The examiner's question is in CS-INT; fill_um is spliced only into the derived CS-HES.
    assert by_name["int_p1_q1"] == "bm_lewis" and "fill_um" not in by_name


@pytest.mark.parametrize(
    ("text", "count"),
    [("Where is your home town?", 5), ("I live in a flat in the centre of my town.", 12),
     ("Mm-hmm.", 2), ("[laughter] The museum was tedious.", 8), ("", 0)],
)  # fmt: skip
def test_syllables_are_hand_counts(text: str, count: int) -> None:
    # the 1 + mu-se-um 3 + was 1 + te-di-ous 3
    assert sets.syllables(text) == count


def test_the_hesitations_are_the_spliced_fillers_and_lead_words(
    built: Callable[[str], tuple[Path, dict[str, Any]]],
) -> None:
    _, truth = built("CS-HES")
    found = sets.hesitations(truth)
    spliced = [s for s in truth["segments"] if s["clip"].startswith(("fill_", "lead_"))]
    assert len(found) == 8 == len(spliced)
    assert found == [(s["start_ms"], s["end_ms"]) for s in spliced]
    assert sets.hesitations({"segments": [{"id": 3, "start_ms": 0, "end_ms": 5}]}) == []
