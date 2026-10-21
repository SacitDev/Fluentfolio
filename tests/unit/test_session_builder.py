"""The composed-session builder: exact clip edges and pauses, identical bytes twice, and its clip rules."""

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import soundfile as sf

Built = Callable[[str], tuple[Path, dict[str, Any]]]

SR = 16000
FRAME = 160  # 10 ms
LOUD = 0.01  # -40 dBFS, the documented edge rule, written out here rather than read from the builder
SPLIT = 15  # a quiet run of 150 ms or more separates two loud regions
SESSIONS = Path(__file__).resolve().parents[1] / "fixtures" / "sessions"
AUDIO = SESSIONS.parent / "audio"
SIX = ["CS-SOLO", "CS-INT", "CS-SWAP", "CS-OVL", "CS-WRONGMODE", "CS-NOISY"]


def loud_regions(path: Path) -> list[tuple[int, int]]:
    """Sample ranges of frames peaking at -40 dBFS or more, joined across quiet runs under 150 ms."""
    x, sr = sf.read(path, dtype="int16")
    assert sr == SR
    n = len(x) // FRAME
    loud = np.abs(x[: n * FRAME].astype(np.int32)).reshape(n, FRAME).max(axis=1) / 32768 >= LOUD
    regions: list[list[int]] = []
    for i in np.flatnonzero(loud):
        if regions and i - regions[-1][1] < SPLIT:
            regions[-1][1] = i + 1
        else:
            regions.append([i, i + 1])
    return [(a * FRAME, b * FRAME) for a, b in regions]


def truth_regions(truth: dict) -> list[tuple[int, int]]:
    regions: list[list[int]] = []
    for s in sorted(truth["segments"], key=lambda s: s["start_sample"]):
        if regions and s["start_sample"] - regions[-1][1] < SPLIT * FRAME:
            regions[-1][1] = max(regions[-1][1], s["end_sample"])
        else:
            regions.append([s["start_sample"], s["end_sample"]])
    return [(a, b) for a, b in regions]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ["CS-SOLO", "CS-INT", "CS-SWAP", "CS-OVL", "CS-HES"])
def test_edges_and_pauses_equal_the_truth(built: Built, name: str) -> None:
    folder, truth = built(name)
    found = loud_regions(folder / "session.wav")
    assert found == truth_regions(truth)
    gaps_ms = {(a // 16, b // 16) for (_, a), (b, _) in zip(found, found[1:], strict=False)}
    pauses = {(p["start_ms"], p["end_ms"]) for t in truth["turns"] for p in t["pauses"] if p["ms"] >= 250}
    # A pause with another voice inside it (a backchannel) is not one silent gap; the next test covers it.
    silent = {(a, b) for a, b in pauses if not any(a < s["start_ms"] < b for s in truth["segments"])}
    assert silent
    assert silent <= gaps_ms


def test_backchannel_sits_inside_a_candidate_pause_and_the_overlap_lasts_two_seconds(built: Built) -> None:
    _, truth = built("CS-OVL")
    nod = next(s for s in truth["segments"] if s["kind"] == "backchannel")
    pause = next(p for t in truth["turns"] if t["id"] == "p1a2" for p in t["pauses"] if p["ms"] == 1200)
    assert pause["start_ms"] + 400 == nod["start_ms"]
    assert nod["end_ms"] < pause["end_ms"]
    assert [o["end_ms"] - o["start_ms"] for o in truth["overlaps"]] == [2000]


@pytest.mark.parametrize("name", ["CS-INT", "CS-NOISY"])
def test_two_builds_give_identical_bytes(
    session_builder: ModuleType, built: Built, tmp_path: Path, name: str
) -> None:
    first, _ = built(name)  # the run's shared build, then a second one here
    session_builder.build(SESSIONS / f"{name}.json", tmp_path)
    for file in ("session.wav", "truth.json"):
        assert sha256(first / file) == sha256(tmp_path / file)


# Edges with another voice in the 10 ms outside them: CS-OVL's overlap covers one end of both its clips.
EDGES_NEXT_TO_A_VOICE = {"CS-SOLO": 0, "CS-INT": 0, "CS-SWAP": 0, "CS-OVL": 2}


@pytest.mark.parametrize("name", list(EDGES_NEXT_TO_A_VOICE))
def test_every_segment_edge_starts_on_sound_and_the_10_ms_outside_it_is_quiet(
    session_builder: ModuleType, built: Built, name: str
) -> None:
    recipe = session_builder.load_recipe(SESSIONS / f"{name}.json")
    segments, tracks, _ = session_builder.place(recipe, AUDIO)
    folder, _ = built(name)
    x, _ = sf.read(folder / "session.wav", dtype="int16")
    peak = np.abs(x.astype(np.int32)) / 32768
    spans = [(s["start_sample"], s["end_sample"]) for s in segments]
    skipped = 0
    for s in segments:
        a, b = s["start_sample"], s["end_sample"]
        # The speech itself starts and ends on a non-zero sample; in the file the noise floor is added on top,
        # so there the check is by frame: the edge frame reaches -40 dBFS.
        assert tracks[s["speaker"]][a] != 0
        assert tracks[s["speaker"]][b - 1] != 0
        assert a % FRAME == 0 and b % FRAME == 0
        assert peak[a : a + FRAME].max() >= LOUD
        assert peak[b - FRAME : b].max() >= LOUD
        if any(lo < a and hi > a - FRAME for lo, hi in spans):
            skipped += 1
        else:
            assert peak[a - FRAME : a].max() < LOUD
        if any(lo < b + FRAME and hi > b for lo, hi in spans):
            skipped += 1
        else:
            assert peak[b : b + FRAME].max() < LOUD
    assert skipped == EDGES_NEXT_TO_A_VOICE[name]


@pytest.mark.parametrize(("quiet_frames", "refused"), [(14, False), (15, True)])
def test_an_inner_quiet_run_of_150_ms_refuses_the_clip(
    session_builder: ModuleType, quiet_frames: int, refused: bool
) -> None:
    tone = 0.3 * np.cos(2 * np.pi * 200 * np.arange(50 * FRAME) / SR)
    pad = np.zeros(20 * FRAME)
    clip = np.concatenate([pad, tone, np.zeros(quiet_frames * FRAME), tone, pad]).astype(np.float32)
    if refused:
        with pytest.raises(ValueError, match="quiet run of 150 ms"):
            session_builder.trim(clip, "clip")
    else:
        assert len(session_builder.trim(clip, "clip")) == (50 + 14 + 50) * FRAME


def test_a_weak_edge_frame_is_cut_away(session_builder: ModuleType) -> None:
    weak = np.full(3 * FRAME, 10 ** (-37 / 20))  # above -40 dBFS, below the -34 dBFS kept at an edge
    tone = 0.3 * np.cos(2 * np.pi * 200 * np.arange(50 * FRAME) / SR)
    clip = np.concatenate([np.zeros(10 * FRAME), weak, tone, np.zeros(10 * FRAME)]).astype(np.float32)
    kept = session_builder.trim(clip, "clip")
    assert len(kept) == 50 * FRAME
    assert np.abs(kept[:FRAME]).max() >= 10 ** (-34 / 20)


def test_an_edge_frame_that_starts_on_a_zero_sample_is_cut_away(session_builder: ModuleType) -> None:
    # A sine starting at phase 0 has exactly 0 as its first sample; the next frame starts on sound.
    tone = (0.3 * np.sin(2 * np.pi * 200 * np.arange(50 * FRAME) / SR)).astype(np.float32)
    tone[0] = 0.0
    clip = np.concatenate(
        [np.zeros(10 * FRAME, dtype=np.float32), tone, np.zeros(10 * FRAME, dtype=np.float32)]
    )
    kept = session_builder.trim(clip, "clip")
    assert len(kept) == 49 * FRAME
    assert kept[0] != 0
    assert kept[-1] != 0


def test_derived_layouts(built: Built) -> None:
    _, base = built("CS-INT")
    two_track, stereo = built("CS-2TRACK")
    x, _ = sf.read(two_track / "session.wav", dtype="int16")
    assert x.shape == (base["samples"], 2)
    assert stereo["layout"] == "per-track"
    two_files, files = built("CS-2FILE")
    examiner = sf.info(two_files / "examiner.wav").frames
    candidate = sf.info(two_files / "candidate.wav").frames
    assert (files["offset_ms"], examiner - candidate) == (1500, 24000)  # 1.5 s at 16 kHz
    _, hes = built("CS-HES")
    clips = [s["clip"] for s in hes["segments"]]
    assert sum(c.startswith("fill_") for c in clips) == 4
    assert sum(c.startswith("lead_") for c in clips) == 4
    _, long = built("CS-LONG")
    assert long["duration_ms"] >= 15 * 60 * 1000
    copies = len(long["segments"]) // len(base["segments"])
    assert copies * len(base["segments"]) == len(long["segments"])
    assert long["duration_ms"] == copies * base["duration_ms"] + (copies - 1) * 2000


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_each_track_holds_its_own_voice_and_the_other_20_db_lower(built: Built) -> None:
    folder, truth = built("CS-2TRACK")
    x, _ = sf.read(folder / "session.wav", dtype="int16")
    channel = {"E": 0, "C": 1}  # examiner left, candidate right
    assert not truth["overlaps"]  # so each segment holds one voice and its leak only
    for s in truth["segments"]:
        span = x[s["start_sample"] : s["end_sample"]]
        own = rms(span[:, channel[s["speaker"]]])
        other = rms(span[:, 1 - channel[s["speaker"]]])
        # The recipe leaks each voice into the other track at -20 dB; the -60 dBFS floor adds well under 1 dB.
        assert 19.0 <= 20 * np.log10(own / other) <= 21.0


def test_the_two_files_are_the_two_tracks_with_the_candidate_starting_1_5_s_later(built: Built) -> None:
    two_track, _ = built("CS-2TRACK")
    two_files, truth = built("CS-2FILE")
    stereo, _ = sf.read(two_track / "session.wav", dtype="int16")
    examiner, _ = sf.read(two_files / "examiner.wav", dtype="int16")
    candidate, _ = sf.read(two_files / "candidate.wav", dtype="int16")
    assert truth["offset_ms"] == 1500
    assert np.array_equal(examiner, stereo[:, 0])
    assert np.array_equal(candidate, stereo[24000:, 1])  # 1.5 s x 16,000 samples cut from its start


def test_the_long_session_edges_equal_its_truth(built: Built) -> None:
    folder, truth = built("CS-LONG")
    assert loud_regions(folder / "session.wav") == truth_regions(truth)


def test_each_spliced_lead_word_is_the_first_word_after_it(built: Built) -> None:
    _, truth = built("CS-HES")
    ordered = sorted(truth["segments"], key=lambda s: s["start_sample"])
    leads = [(a, b) for a, b in zip(ordered, ordered[1:], strict=False) if a["clip"].startswith("lead_")]
    assert len(leads) == 4
    for lead, after in leads:
        assert after["text"].split()[0].strip(",.").lower() == lead["text"].strip(".").lower()


def test_the_builder_never_imports_the_package(tmp_path: Path) -> None:
    builder = Path(__file__).resolve().parents[1] / "support" / "session_builder.py"
    done = subprocess.run(
        [sys.executable, "-X", "importtime", str(builder), str(SESSIONS / "CS-SOLO.json"), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    imported = [line.rsplit("|", 1)[-1].strip() for line in done.stderr.splitlines() if "|" in line]
    assert "session_builder" not in imported  # run as a script, so it is __main__
    assert "numpy" in imported
    assert not [m for m in imported if m.split(".")[0] == "fluentfolio"]


def test_recipes_use_committed_clips_within_the_size_caps() -> None:
    clips = json.loads((AUDIO / "clips.json").read_text(encoding="utf-8"))
    files = sorted(AUDIO.glob("*.opus"))
    assert sorted(f.stem for f in files) == sorted(clips)
    sizes = [f.stat().st_size for f in files]
    assert max(sizes) <= 1_000_000
    assert sum(sizes) <= 8_000_000
    for name in SIX:
        recipe = json.loads((SESSIONS / f"{name}.json").read_text(encoding="utf-8"))
        assert {e["clip"] for e in recipe.get("events", [])} <= set(clips)
