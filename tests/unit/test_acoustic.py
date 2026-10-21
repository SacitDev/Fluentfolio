"""The rules the acoustic tools share, the prosody values read from Praat, and the candidate audio step.

Every expected number is worked out by hand in the test or comes from numpy, never from the code under test.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from fluentfolio.analyzers import opensmile, praat
from fluentfolio.analyzers.measures import acoustic, prosody
from fluentfolio.analyzers.measures.acoustic import Frame, Piece
from fluentfolio.analyzers.measures.values import Missing, Value
from fluentfolio.core.files import read_json, write_json
from fluentfolio.media import slices

Row = dict[str, Any]


def answer(aid: str, part: int | str, start: int = 0, end: int = 60_000) -> Row:
    return {"id": aid, "part": part, "start_ms": start, "end_ms": end}


@pytest.mark.parametrize(("q", "expected"), [(0, 1.0), (20, 1.6), (50, 2.5), (80, 3.4), (100, 4.0)])
def test_percentiles_interpolate_between_ranks(q: float, expected: float) -> None:
    # Sorted 1, 2, 3, 4: the q-th percentile sits at rank 3 * q / 100.
    assert acoustic.percentile([4.0, 1.0, 3.0, 2.0], q) == pytest.approx(expected)
    assert acoustic.percentile([4.0, 1.0, 3.0, 2.0], q) == pytest.approx(np.percentile([1, 2, 3, 4], q))


def test_semitones_count_from_27_5_hz() -> None:
    assert acoustic.semitones(27.5) == 0.0
    assert acoustic.semitones(55.0) == pytest.approx(12.0)  # one octave is 12 semitones
    assert acoustic.semitones(120.0) == pytest.approx(25.51, abs=0.01)
    assert acoustic.hertz(24.0) == pytest.approx(110.0)


PIECES = [Piece(0, 1000, 500), Piece(500, 3000, 200)]  # the candidate audio holds 0-500 and 500-700 ms


@pytest.mark.parametrize(
    ("wav_ms", "expected"), [(-5, None), (0, 0), (499.9, 0), (500, 1), (699, 1), (700, None), (900, None)]
)
def test_a_time_of_the_candidate_audio_falls_in_one_piece(wav_ms: float, expected: int | None) -> None:
    assert acoustic.piece_at(wav_ms, PIECES) == expected


def test_frames_are_placed_in_the_recording_and_in_their_answer() -> None:
    answers = [answer("A01", 1, 1080, 3050), answer("A02", 1, 3050, 5000)]
    found = acoustic.frames([50, 100, 600, 800], [110.0, 110.0, None, 220.0], PIECES, answers)
    # 50 ms is 1050 ms in the recording, before A01 starts; 800 ms lies past the last piece.
    assert found == [Frame(0, "A01", 100, pytest.approx(24.0)), Frame(1, "A02", 600, None)]


def test_scopes_are_the_total_and_each_part_an_answer_belongs_to() -> None:
    answers = [answer("A01", 1), answer("A02", 3), answer("A03", "unknown"), answer("A04", 1)]
    assert acoustic.scopes(answers) == {"all": ["A01", "A02", "A03", "A04"], "part1": ["A01", "A04"],
                                        "part3": ["A02"]}  # fmt: skip


def voiced(aid: str, k: int, st: float) -> list[Frame]:
    return [Frame(0, aid, float(t), st) for t in range(k)]


def test_the_voiced_minimum_equals_its_tolerance_entry(tolerance: Any) -> None:
    assert acoustic.MIN_VOICED_MS == tolerance("prosody", "min_voiced_ms_per_answer")
    assert acoustic.FLAT_MIN_MS == tolerance("prosody", "flat_min_ms")
    assert acoustic.VOICE_RUN_MS == tolerance("voice_quality", "min_run_ms")


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_an_answer_needs_the_voiced_minimum_for_prosody(delta: int) -> None:
    # One frame per ms, so the frame count is the voiced time in ms.
    found = voiced("A01", acoustic.MIN_VOICED_MS + delta, 12.0)
    row = acoustic.pitch_scopes(found, 1.0, [answer("A01", 2)])["all"]
    assert row["used"] == ([] if delta < 0 else ["A01"])
    assert (row["missing"] is None) == (delta >= 0)
    if delta < 0:
        assert row["missing"]["code"] == acoustic.INSUFFICIENT


def test_a_short_answer_is_left_out_of_its_scope_and_the_total() -> None:
    found = voiced("A01", 10_000, 12.0) + voiced("A02", 9_999, 24.0)
    rows = acoustic.pitch_scopes(found, 1.0, [answer("A01", 1), answer("A02", 2)])
    assert (rows["all"]["used"], rows["all"]["answers"], rows["all"]["p50_st"]) == (["A01"], 2, 12.0)
    assert rows["part2"]["missing"]["code"] == acoustic.INSUFFICIENT


def test_pitch_summary_by_hand() -> None:
    row = acoustic.pitch_summary([10.0, 12.0, 14.0, 16.0], 2500.0, 1, ["A01"])
    assert row["voiced_ms"] == 10_000
    assert (row["p20_st"], row["p50_st"], row["p80_st"]) == pytest.approx((11.2, 13.0, 14.8))
    assert row["sd_st"] == pytest.approx(math.sqrt(5))  # deviations 3, 1, 1, 3 from the mean of 13
    assert row["median_hz"] == pytest.approx(58.2705, abs=1e-4)  # 27.5 x 2 ** (13 / 12)


TIMES = [float(t) for t in range(0, 300, 10)]  # 30 frames of 10 ms: 0-290 ms


@pytest.mark.parametrize(
    ("per_ms", "expected"),
    [(0.01, "rise"), (-0.01, "fall"), (0.0, "level"), (0.0065, "level"), (0.0072, "rise"), (-0.0072, "fall")],
)
def test_an_ending_is_the_fitted_change_over_the_last_300_ms(per_ms: float, expected: str) -> None:
    # The change is the slope times the 290 ms the window spans: 0.0065 gives 1.9 st, 0.0072 gives 2.1 st.
    assert acoustic.ending(TIMES, [10 + per_ms * t for t in TIMES], 10.0) == expected


def test_an_ending_reads_only_the_last_300_ms() -> None:
    times = [float(t) for t in range(0, 1000, 10)]
    st = [20 - 0.05 * min(t, 690) for t in times]  # a steep fall that stops 300 ms before the end
    assert acoustic.ending(times, st, 10.0) == "level"


@pytest.mark.parametrize(("frames", "expected"), [(0, None), (9, None), (10, "level")])
def test_an_ending_needs_100_ms_of_voice(frames: int, expected: str | None) -> None:
    times = TIMES[:frames]
    assert acoustic.ending(times, [10 + 0.01 * t for t in times], 10.0) == expected


def ramp(top: float, n: int) -> list[float]:
    """n pitch values evenly from 0 to `top` semitones: their 10th-90th percentile range is 0.8 x top."""
    return [top * k / (n - 1) for k in range(n)]


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_a_flat_stretch_lasts_the_minimum_or_more(delta: int) -> None:
    assert acoustic.flat(acoustic.FLAT_MIN_MS + delta, ramp(1.9, 100), 10.0) is (delta >= 0)  # 1 s, 1.52 st


def test_a_flat_stretch_needs_1_s_of_pitch_inside_2_semitones() -> None:
    assert acoustic.flat(3000, ramp(1.9, 99), 10.0) is False  # 990 ms of pitch
    assert acoustic.flat(3000, ramp(3.0, 100), 10.0) is False  # a range of 2.4 st


def run_frames(*groups: tuple[int, int, bool]) -> list[Frame]:
    """Frames of 10 ms one after another: (count, piece, voiced) per group."""
    out, t = [], 0.0
    for count, piece, on in groups:
        for _ in range(count):
            out.append(Frame(piece, "A01", t, 12.0 if on else None))
            t += 10
    return out


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        (((100, 0, True),), [(0.0, 1000.0)]),
        (((99, 0, True),), []),
        (((60, 0, True), (1, 0, False), (60, 0, True)), []),  # an unvoiced frame breaks the run
        (((60, 0, True), (60, 1, True)), []),  # so does the join between two pieces
        (((150, 0, True), (1, 0, False), (100, 0, True)), [(0.0, 1500.0), (1510.0, 2510.0)]),
    ],
)
def test_voice_quality_uses_runs_of_1_s_or_more(
    groups: tuple[tuple[int, int, bool], ...], expected: list[tuple[float, float]]
) -> None:
    assert acoustic.voiced_runs(run_frames(*groups), 10.0) == expected


def test_a_gap_in_the_frames_breaks_a_run() -> None:
    frames = [Frame(0, "A01", float(t), 12.0) for t in [*range(0, 500, 10), *range(520, 1020, 10)]]
    assert acoustic.voiced_runs(frames, 10.0) == []


def test_a_tool_and_its_analyzers_are_on_unless_switched_off() -> None:
    assert praat.wanted({}) == ["pitch", "voice-quality"]
    assert praat.wanted({"components": {"praat": False}}) == []
    assert praat.wanted({"analyzers": {"voice-quality": False}}) == ["pitch"]
    assert acoustic.switched_on({"components": {"praat": False}}, "swiftf0", "pitch") is True
    assert acoustic.switched_on({"analyzers": {"pitch": False}}, "swiftf0", "pitch") is False


@pytest.mark.parametrize(
    ("used", "total", "expected"),
    [
        (1, 1, "From the one answer."),
        (3, 3, "From all 3 answers."),
        (2, 3, "From 2 of 3 answers; 1 has under 10 s of voiced speech."),
        (2, 5, "From 2 of 5 answers; 3 have under 10 s of voiced speech."),
    ],
)
def test_the_coverage_note_counts_the_answers_used(used: int, total: int, expected: str) -> None:
    assert prosody.coverage({"used": [f"A0{k}" for k in range(used)], "answers": total}) == expected


SUMMARY = {"answers": 3, "used": ["A01", "A02"], "voiced_ms": 25_000, "missing": None, "sd_st": 2.345,
           "voiced_share": 0.6789}  # fmt: skip
UNITS = [
    {"answer_id": "A01", "utterance_id": "U001", "flat": True, "ending": "rise"},
    {"answer_id": "A02", "utterance_id": "U002", "flat": False, "ending": "fall"},
    {"answer_id": "A02", "utterance_id": None, "flat": True, "ending": None},
    {"answer_id": "A03", "utterance_id": "U004", "flat": True, "ending": "rise"},  # A03 is too short
]


def test_scope_values_count_the_stretches_of_the_answers_used() -> None:
    found = prosody.scope_values(SUMMARY, UNITS, None)
    note = "From 2 of 3 answers; 1 has under 10 s of voiced speech."
    assert (found["flat_stretches"].value, found["flat_stretches"].note) == (2, f"{note} In U001, A02.")
    counts = [found[k].value for k in ("rising_endings", "falling_endings", "level_endings")]
    assert counts == [1, 1, 0]
    assert (found["f0_sd"].value, found["f0_sd"].unit) == (2.3, "semitones")
    assert (found["voiced_share"].value, found["voiced_share"].unit) == (67.9, "%")
    assert {v.note for k, v in found.items() if k != "flat_stretches"} == {note}


def test_scope_values_are_missing_with_a_reason() -> None:
    off = Missing("switched_off", "off")
    assert {v.missing for v in prosody.scope_values(SUMMARY, UNITS, off).values()} == {off}
    short = {**SUMMARY, "used": [], "missing": {"code": acoustic.INSUFFICIENT, "text": "too little"}}
    found = prosody.scope_values(short, UNITS, None)
    assert {v.value for v in found.values()} == {None}
    assert {v.missing for v in found.values()} == {Missing(acoustic.INSUFFICIENT, "too little")}
    assert codes(prosody.scope_values(None, [], None)) == {"no_value"}


def codes(values: dict[str, Value]) -> set[str]:
    return {v.missing.code for v in values.values() if v.missing}


def test_voice_values_round_and_keep_what_praat_could_not_measure() -> None:
    row = {
        "runs": 1, "run_ms": 1234, "jitter_pct": 1.2345, "shimmer_pct": 5.678, "hnr_db": 12.34,
        "f1_hz": 512.4, "f2_hz": None, "f3_hz": 2600.6, "missing": {"code": "undefined", "text": "x"},
    }  # fmt: skip
    found = prosody.voice_values(row, None)
    values = {k: v.value for k, v in found.items()}
    assert values == {"jitter": 1.23, "shimmer": 5.68, "hnr": 12.3, "f1": 512.0, "f2": None, "f3": 2601.0}
    assert found["f2"].missing is not None and found["f2"].missing.code == "undefined"
    assert found["jitter"].note == "Over 1 voiced stretch of 1 s or more, 1.2 s in all."
    assert prosody.voice_values({**row, "runs": 2}, None)["hnr"].note == (
        "Over 2 voiced stretches of 1 s or more, 1.2 s in all."
    )


def test_voice_values_are_missing_as_a_whole_with_a_reason() -> None:
    dirty = {"missing": {"code": praat.NOT_CLEAN, "text": "measured on clean audio only"}}
    assert codes(prosody.voice_values(dirty, None)) == {praat.NOT_CLEAN}
    off = Missing("switched_off", "off")
    assert {v.missing for v in prosody.voice_values(None, off).values()} == {off}
    assert codes(prosody.voice_values(None, None)) == {"no_value"}


def test_loudness_values_round_and_say_what_they_cover() -> None:
    row = {"speech_ms": 12_340, "missing": None, "variation": 0.31416, "peaks_per_s": 3.14159}
    found = prosody.loudness_values(row, None)
    assert {k: (v.value, v.unit) for k, v in found.items()} == {
        "loudness_variation": (0.31, "ratio"), "loudness_peaks": (3.14, "per s"),
    }  # fmt: skip
    assert found["loudness_peaks"].note == "Over 12.3 s of the candidate's speech, pauses left out."
    poor = {"speech_ms": 3000, "missing": {"code": "audio_not_suited", "text": "x"}}
    assert codes(prosody.loudness_values(poor, None)) == {"audio_not_suited"}
    off = Missing("not_enabled", "openSMILE not enabled: its licence is not accepted")
    assert {v.missing for v in prosody.loudness_values(None, off).values()} == {off}


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (None, False),  # no file
        ("not json", False),
        ({"partner": {"utc": "2026-10-14T00:00:00Z", "source": "ui"}}, False),
        ({"opensmile": {"utc": "2026-10-14T00:00:00Z", "source": "yes-flag"}}, False),
        ({"opensmile": {"utc": "", "source": "ui"}}, False),
        ({"opensmile": {"source": "ui"}}, False),
        ({"opensmile": {"utc": "2026-10-14T00:00:00Z", "source": "ui"}}, True),
        ({"opensmile": {"utc": "2026-10-14T00:00:00Z", "source": "cli-tty"}}, True),
    ],
)
def test_only_the_users_own_consent_record_lets_opensmile_run(
    tmp_path: Path, record: object, expected: bool
) -> None:
    if isinstance(record, str):
        (tmp_path / "consent.json").write_text(record, encoding="utf-8")
    elif record is not None:
        write_json(tmp_path / "consent.json", record)
    assert opensmile.consented(tmp_path) is expected


def conf_files(root: Path) -> list[Path]:
    """Every .conf file under a folder, past the virtual environment and the git folder."""
    found = []
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = [s for s in subfolders if s not in (".venv", ".git")]
        found += [Path(folder) / f for f in files if f.endswith(".conf")]
    return found


def test_no_opensmile_config_file_is_in_the_repository(tmp_path: Path) -> None:
    # openSMILE runs its built-in eGeMAPSv02 only: a .conf file would be a feature set of our own.
    assert conf_files(Path(__file__).resolve().parents[2]) == []
    planted = tmp_path / "src" / "analyzers" / "tuned.conf"
    planted.parent.mkdir(parents=True)
    planted.write_text("[componentInstances:cComponentManager]\n", encoding="utf-8")
    assert conf_files(tmp_path) == [planted]


def test_spans_join_in_order_with_their_place_in_the_recording() -> None:
    audio = np.arange(48_000, dtype=np.float32)  # 3 s at 16 kHz
    joined, pieces = slices.join(audio, [[1000, 1500], [2000, 2250], [2900, 2900], [2800, 3100]])
    # The empty span is dropped; the last one stops where the recording ends, at 3000 ms.
    assert pieces == [{"wav_ms": 0, "orig_ms": 1000, "ms": 500}, {"wav_ms": 500, "orig_ms": 2000, "ms": 250},
                      {"wav_ms": 750, "orig_ms": 2800, "ms": 200}]  # fmt: skip
    expected = np.concatenate([audio[16_000:24_000], audio[32_000:36_000], audio[44_800:48_000]])
    assert np.array_equal(joined, expected)


def test_a_candidate_on_a_track_of_their_own_reads_that_track() -> None:
    tracks = {"tracks": [{"track": 1}, {"track": 2}]}
    assert slices.source(tracks, "S2") == "track_2.wav"
    assert slices.source({}, "S1") == "audio16k.wav"


def slice_job(
    tmp_path: Path, speech: dict[str, list[list[int]]], music: list[list[int]] | None = None
) -> Path:
    job = tmp_path / "job"
    x = (0.3 * np.sin(2 * np.pi * 200 * np.arange(32_000) / 16_000)).astype(np.float32)  # 2 s
    (job / "input").mkdir(parents=True)
    sf.write(job / "input" / "audio16k.wav", x, 16_000, subtype="PCM_16")
    write_json(job / "stages" / "media.json", {"music_spans": music or []})
    write_json(job / "stages" / "vad.json", {"status": "ok", "segments": []})
    candidates = [{"id": cid, "speaker": "S1"} for cid in speech]
    write_json(job / "stages" / "structure.json", {"candidates": candidates, "speech": speech})
    return job


def test_the_candidate_audio_step_writes_each_candidates_speech_and_its_map(tmp_path: Path) -> None:
    job = slice_job(tmp_path, {"person1": [[100, 600], [900, 1400]], "person2": []})
    result = slices.run(job, lambda pct: None)
    one, two = result["candidates"]["person1"], result["candidates"]["person2"]
    assert (result["status"], one["speech_ms"], one["pieces"]) == ("ok", 1000, 2)
    assert (two["speech_ms"], two["wav"], two["map"]) == (0, None, None)
    found, rate = sf.read(job / one["wav"], dtype="float32")
    source = sf.read(job / "input" / "audio16k.wav", dtype="float32")[0]
    # 16-bit samples, so equal to one step; a shift of one sample would differ by far more.
    expected = np.concatenate([source[1600:9600], source[14_400:22_400]])
    assert rate == 16_000 and np.allclose(found, expected, atol=2 / 32768, rtol=0)
    assert read_json(job / one["map"])["pieces"] == [{"wav_ms": 0, "orig_ms": 100, "ms": 500},
                                                     {"wav_ms": 500, "orig_ms": 900, "ms": 500}]  # fmt: skip


def test_the_candidate_audio_leaves_out_background_music(tmp_path: Path) -> None:
    job = slice_job(tmp_path, {"person1": [[100, 600], [900, 1400]]}, music=[[300, 1000]])
    one = slices.run(job, lambda pct: None)["candidates"]["person1"]
    # 100-300 ms and 1000-1400 ms are left: 200 ms and 400 ms.
    assert (one["speech_ms"], one["pieces"]) == (600, 2)
    assert read_json(job / one["map"])["pieces"] == [{"wav_ms": 0, "orig_ms": 100, "ms": 200},
                                                     {"wav_ms": 200, "orig_ms": 1000, "ms": 400}]  # fmt: skip


def test_without_a_candidate_the_step_is_skipped(tmp_path: Path) -> None:
    assert slices.run(slice_job(tmp_path, {}), lambda pct: None)["status"] == "skipped"
