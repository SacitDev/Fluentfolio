"""Speaker turns: word ownership, the speaker policy, the voice checks and the bleed rule, by hand."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs import stage
from fluentfolio.speakers import checks, ownership, pyannote, tracks
from fluentfolio.speakers.ownership import EDGE, OVERLAP


def word(start: int, end: int, **extra: object) -> dict:
    return {"text": "w", "start_ms": start, "end_ms": end, "flags": [], **extra}


def test_an_overlap_belongs_to_the_longer_turn_in_the_exclusive_turns() -> None:
    turns = [(0, 5000, "A"), (4000, 6000, "B")]
    # [4000, 5000) is covered by A (5000 ms long) and B (2000 ms): A keeps it.
    assert ownership.exclusive(turns) == [(0, 5000, "A"), (5000, 6000, "B")]
    assert ownership.overlaps(turns) == [(4000, 5000)]
    assert ownership.talk_ms([(0, 5000), (4000, 6000), (7000, 7500)]) == 6500  # 6000 + 500


def test_one_speakers_own_overlapping_turns_are_not_an_overlap() -> None:
    turns = [(0, 3000, "A"), (2000, 4000, "A"), (3500, 4500, "B")]
    assert ownership.overlaps(turns) == [(3500, 4000)]
    assert ownership.exclusive(turns) == [(0, 4000, "A"), (4000, 4500, "B")]


def test_each_word_goes_to_the_turn_that_covers_most_of_it() -> None:
    words = [word(4800, 5400), word(100, 900), word(5500, 5600), word(4500, 4600), word(3900, 4050)]
    ownership.assign(words, [(0, 5000, "A"), (4000, 6000, "B")])
    # 4800-5400: 200 ms with A (exclusive A ends at 5000), 400 ms with B.
    assert [w["speaker"] for w in words] == ["B", "A", "B", "A", "A"]
    # Midpoints 5100 and 4550 lie in the overlap [4000, 5000)? 5100 does not, 4550 does; 3975 does not.
    assert [OVERLAP in w["flags"] for w in words] == [False, False, False, True, False]
    assert not any(EDGE in w["flags"] for w in words)


@pytest.mark.parametrize(
    ("start", "speaker", "edge"),
    [
        (3499, "B", True),  # 499 ms after B's turn ends
        (3500, "B", True),  # exactly 500 ms: still close enough
        (3501, None, False),  # 501 ms: too far, so the speaker is unknown
    ],
)
def test_a_word_outside_every_turn_goes_to_a_turn_within_half_a_second(
    start: int, speaker: str | None, edge: bool
) -> None:
    words = [word(start, start + 100)]
    ownership.assign(words, [(0, 1000, "A"), (2000, 3000, "B")])
    assert (words[0]["speaker"], EDGE in words[0]["flags"]) == (speaker, edge)


def test_the_edge_and_bleed_rules_equal_their_tolerance_entries(tolerance: Any) -> None:
    assert ownership.EDGE_MS == tolerance("speakers", "edge_assign_ms")
    assert tracks.LOUDER_DB == tolerance("media", "bleed_louder_db")


def test_in_a_gap_the_nearer_turn_wins_and_a_word_without_length_inside_a_turn_is_not_an_edge() -> None:
    turns = [(0, 1000, "A"), (2000, 3000, "B")]
    words = [word(1200, 1300), word(1520, 1530), word(2500, 2500)]
    ownership.assign(words, turns)
    # Gaps: 200 ms to A and 700 ms to B; 520 ms to A and 470 ms to B; the last word sits inside B.
    assert [(w["speaker"], EDGE in w["flags"]) for w in words] == [("A", True), ("B", True), ("B", False)]


def test_with_one_track_per_person_a_word_belongs_to_its_track() -> None:
    words = [word(100, 200, track=2), word(4500, 4600, track=1)]
    ownership.assign(words, [(0, 5000, "S1"), (4000, 6000, "S2")], by_track=True)
    assert [w["speaker"] for w in words] == ["S2", "S1"]
    assert [OVERLAP in w["flags"] for w in words] == [False, True]


def job(tmp_path: Path, mode: str, speakers: dict | None, music: list[list[int]] | None = None) -> Path:
    folder = tmp_path / "job"
    (folder / "stages").mkdir(parents=True)
    write_json(folder / "job.json", {"mode": mode})
    write_json(folder / "stages" / "media.json", {"music_spans": music or []})
    write_json(folder / "stages" / "asr.json", {"words": [word(100, 400), word(2100, 2300)]})
    if speakers is not None:
        write_json(folder / "stages" / "speakers.json", speakers)
    return folder


def test_the_words_step_gives_no_speaker_when_the_speakers_step_failed(tmp_path: Path) -> None:
    for found in ({"status": "failed", "code": "E_STAGE_CRASHED"}, None):
        folder = job(tmp_path / str(found is None), "interview", found)
        result = ownership.run(folder)
        assert result["status"] == "partial" and result["reason"] == ownership.NO_SPEAKERS
        assert [w["speaker"] for w in result["words"]] == [None, None]
        assert result["counts"]["unknown_speaker"] == 2


def test_in_solo_every_word_is_the_one_speakers_whatever_the_check_found(tmp_path: Path) -> None:
    folder = job(tmp_path, "solo", {"status": "skipped", "backend": "single"})
    result = ownership.run(folder)
    assert result["status"] == "ok" and [w["speaker"] for w in result["words"]] == ["S1", "S1"]
    asr = read_json(folder / "stages" / "asr.json")
    assert all(w["flags"] == [] and "speaker" not in w for w in asr["words"])  # the recogniser's file is kept


@pytest.mark.parametrize(
    ("span", "flagged"),
    [
        ([0, 250], [False, False]),
        ([0, 251], [True, False]),
        ([2200, 2201], [False, True]),
        ([2201, 5000], [False, False]),
    ],
)
def test_a_word_whose_middle_is_in_background_music_is_flagged(
    tmp_path: Path, span: list[int], flagged: list[bool]
) -> None:  # fmt: skip
    # The words' middles are at 250 ms and 2200 ms; a span holds its start but not its end.
    result = ownership.run(job(tmp_path, "solo", {"status": "ok", "backend": "single"}, [span]))
    assert [ownership.MUSIC in w["flags"] for w in result["words"]] == flagged
    assert result["counts"][ownership.MUSIC] == sum(flagged)


def test_ids_follow_first_appearance() -> None:
    turns, ids = checks.relabel([(5000, 6000, "x"), (0, 1000, "y"), (2000, 3000, "x")])
    assert turns == [(0, 1000, "S1"), (2000, 3000, "S2"), (5000, 6000, "S2")]
    assert ids == {"y": "S1", "x": "S2"}


def talk(*ms: int) -> list[tuple[int, int, str]]:
    """One turn per speaker, laid end to end, with these talk times."""
    out, t = [], 0
    for k, length in enumerate(ms):
        out.append((t, t + length, f"S{k + 1}"))
        t += length
    return out


@pytest.mark.parametrize(
    ("times", "kinds", "warnings"),
    [
        ((50000, 50000), ["main", "main"], []),
        ((100000,), ["main"], ["W_ONE_VOICE"]),
        # 5000 of 100000 is exactly 5 %, so the third voice is a main one.
        ((47500, 47500, 5000), ["main", "main", "main"], ["W_THIRD_VOICE"]),
        # 4999 of 99999 is under 5 % (4999.95 ms), so it is "other".
        ((47500, 47500, 4999), ["main", "main", "other"], ["W_OTHER_VOICE"]),
        ((47500, 47500, 5001), ["main", "main", "main"], ["W_THIRD_VOICE"]),
        # A fourth voice is "other" however long it speaks: two-person modes allow three at most.
        ((30000, 30000, 30000, 10000), ["main", "main", "main", "other"], ["W_THIRD_VOICE", "W_OTHER_VOICE"]),
        ((95000, 5000 - 1), ["main", "other"], ["W_ONE_VOICE", "W_OTHER_VOICE"]),
    ],
)
def test_the_speaker_policy(times: tuple[int, ...], kinds: list[str], warnings: list[str]) -> None:
    result = checks.summarize("x", talk(*times), mode="interview")
    assert [s["kind"] for s in result["speakers"]] == kinds
    assert result["warnings"] == warnings
    assert set(warnings) <= {"W_ONE_VOICE", "W_THIRD_VOICE", "W_OTHER_VOICE"}  # the policy's only warnings
    assert [s["speech_ms"] for s in result["speakers"]] == list(times)


def test_overlap_time_is_recorded_and_no_embedding_is_kept() -> None:
    result = checks.summarize("pyannote-community-1", [(0, 5000, "S1"), (4000, 6000, "S2")], mode="interview")
    checks.add_similarity(result, 0.25)
    assert result["overlap_ms"] == 1000 and result["similarity"] == 0.25
    text = json.dumps(result)
    assert "embedding" not in text and len(text) < 1000  # numbers and labels only


@pytest.mark.parametrize(
    ("backend", "value", "warned"),
    [
        ("pyannote-community-1", 0.449, False), ("pyannote-community-1", 0.45, True),
        ("pyannote-community-1", 0.451, True), ("pyannote-community-1", None, False),
        # This embedder puts different voices higher on its scale, so 0.62 is still two voices.
        ("sherpa-onnx-diarizer", 0.62, False), ("sherpa-onnx-diarizer", 0.899, False),
        ("sherpa-onnx-diarizer", 0.9, True), ("sherpa-onnx-diarizer", 0.901, True),
    ],
)  # fmt: skip
def test_two_voices_that_sound_alike_cap_role_confidence(
    backend: str, value: float | None, warned: bool
) -> None:
    result = checks.summarize(backend, talk(50000, 50000), mode="interview")
    checks.add_similarity(result, value)
    assert ("W_SIMILAR_VOICES" in result["warnings"], result["role_confidence_cap"]) == (
        warned, "medium" if warned else None,
    )  # fmt: skip


def test_cosine_against_hand_values() -> None:
    assert checks.cosine(np.array([1.0, 0.0]), np.array([1.0, 1.0])) == pytest.approx(2**-0.5)
    assert checks.cosine(np.array([1.0, 0.0]), np.array([0.0, 0.0])) is None


@pytest.mark.parametrize(("other_ms", "warned"), [(4999, False), (5000, False), (5001, True)])
def test_a_second_voice_in_solo_warns_over_5_percent_of_speech(other_ms: int, warned: bool) -> None:
    turns = [(0, 100000 - other_ms, "a"), (100000, 100000 + other_ms, "b")]
    result = checks.second_voice("sherpa-onnx-diarizer", turns)
    assert result["second_voice"]["share"] == round(other_ms / 100000, 3)
    assert (result["warnings"] == ["W_SECOND_VOICE_SOLO"]) == warned


def frames(n: int, *runs: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for a, b in runs:
        mask[a:b] = True
    return mask


@pytest.mark.parametrize(
    ("ratio_db", "first", "second"),
    [
        (20.0, True, False),  # the other voice leaks in 20 dB lower: the louder track keeps the frames
        (6.0, True, False),  # exactly 6 dB louder still wins
        (5.9, True, True),  # closer than 6 dB: both keep the frames, which is overlap
        (0.0, True, True),
        (-6.0, False, True),
    ],
)
def test_the_louder_track_keeps_a_frame_when_louder_by_6_db(
    ratio_db: float, first: bool, second: bool
) -> None:
    n = 100
    voiced = [frames(n, (10, 60)), frames(n, (10, 60))]
    energy = [np.full(n, 10 ** (ratio_db / 10)), np.full(n, 1.0)]  # energy ratio in dB = 10 log10
    owned, both = tracks.resolve(voiced, energy)
    assert (bool(owned[0][30]), bool(owned[1][30])) == (first, second)
    assert bool(both[30]) == (first and second)
    assert not owned[0][70] and not owned[1][70]  # an unvoiced frame belongs to nobody


def test_a_frame_voiced_on_one_track_only_is_that_tracks() -> None:
    voiced = [frames(50, (0, 20)), frames(50, (20, 40))]
    owned, both = tracks.resolve(voiced, [np.full(50, 0.01), np.full(50, 1.0)])
    assert owned[0][:20].all() and owned[1][20:40].all() and not both.any()


def test_runs_of_frames_shorter_than_the_shortest_speech_are_dropped() -> None:
    mask = frames(40, (0, 8), (10, 18), (20, 40))  # 256 ms, 256 ms and 640 ms at 32 ms a frame
    assert tracks.mask_runs(mask, 32, 250) == [(0, 8), (10, 18), (20, 40)]
    assert tracks.mask_runs(mask, 32, 256) == [(20, 40)]  # a run must be longer than the limit


@pytest.mark.parametrize(("both_frames", "warned"), [(29, False), (30, False), (31, True)])
def test_heavy_bleed_on_both_tracks_suggests_the_mixed_layout(both_frames: int, warned: bool) -> None:
    voiced = [frames(100, (0, 100)), frames(100, (0, 100))]
    report, warnings = tracks.bleed_report(voiced, frames(100, (0, both_frames)), 32)
    assert report["both_share"] == both_frames / 100 and report["voiced_ms"] == 3200
    assert (warnings == ["W_TRACK_BLEED"]) == warned


def test_progress_counted_in_numpy_integers_reaches_the_parent_as_a_plain_number(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The embeddings step spans 30 to 95 percent: 30 + 65 * 13 // 26 = 30 + 32 = 62.
    pyannote._hook(stage.progress)("embeddings", None, completed=np.int64(13), total=np.int64(26))
    assert capsys.readouterr().out == '{"pct": 62}\n'
