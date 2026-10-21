"""The pitch-tool agreement check's harness: its labels, shares and decision against hand-worked values."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fluentfolio.analyzers.measures import acoustic

EVAL = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL))
import pitch_fusion as pf  # noqa: E402
import pitch_tool  # noqa: E402


def v(p20: float | None, p50: float | None, p80: float | None, voiced_ms: int = 20000) -> dict[str, Any]:
    return {"p20": p20, "p50": p50, "p80": p80, "voiced_ms": voiced_ms, "why": None}


def test_the_values_here_are_the_pre_registered_ones() -> None:
    assert pf.TOOLS == ("praat", "swiftf0", "opensmile")
    assert (pf.AGREE, pf.APPROXIMATE) == (1.0, 3.2)
    assert pf.PREREG["decision"]["limit_share"] == 0.20 and pf.PREREG["calibration"]["tone_st_max"] == 0.3
    assert pf.semitones(55.0) == pytest.approx(12.0) and pf.hertz(24.0) == pytest.approx(110.0)
    # The product's unit, so a value read from a tool's output means the same here.
    assert acoustic.semitones(150.0) == pytest.approx(pf.semitones(150.0))


def test_spreads_are_the_largest_minus_the_smallest_tool_value_and_the_range_takes_the_worse_label() -> None:
    found = pf.judge({"praat": v(30.0, 33.0, 36.0), "swiftf0": v(30.5, 33.4, 37.5),
                      "opensmile": v(29.9, 33.2, 36.2)})  # fmt: skip
    assert found["tools"] == ["praat", "swiftf0", "opensmile"]
    assert found["spread_p20"] == pytest.approx(0.6) and found["spread_p50"] == pytest.approx(0.4)
    assert found["spread_p80"] == pytest.approx(1.5)
    # Ranges 6.0, 7.0 and 6.3: their spread is 1.0.
    assert found["spread_range"] == pytest.approx(1.0)
    assert (found["median_label"], found["range_label"], found["label"]) == (
        "agree",
        "approximate",
        "approximate",
    )


@pytest.mark.parametrize(
    ("spread", "expected"),
    [(0.0, "agree"), (1.0, "agree"), (1.0001, "approximate"), (3.2, "approximate"), (3.2001, "unreliable")],
)
def test_the_label_limits_are_1_and_3_2_semitones_inclusive(spread: float, expected: str) -> None:
    assert pf.label(spread) == expected
    # The same limit through the whole rule: one tool's median off by the spread (from 0, so the difference
    # is exactly the spread in binary floating point).
    found = pf.judge({"praat": v(0.0, 0.0, 0.0), "swiftf0": v(0.0, spread, 0.0)})
    assert found["median_label"] == expected and found["label"] == expected


def test_a_wide_p20_alone_makes_the_range_approximate() -> None:
    found = pf.judge({"praat": v(28.0, 33.0, 36.0), "swiftf0": v(30.0, 33.0, 36.5)})
    assert (found["spread_p20"], found["spread_p80"]) == (2.0, 0.5)
    assert (found["median_label"], found["range_label"], found["label"]) == (
        "agree",
        "approximate",
        "approximate",
    )


def test_the_clip_label_is_the_worse_of_the_median_and_the_range() -> None:
    found = pf.judge({"praat": v(30.0, 33.0, 36.0), "swiftf0": v(30.0, 37.0, 36.0)})
    assert (found["median_label"], found["range_label"], found["label"]) == (
        "unreliable",
        "agree",
        "unreliable",
    )
    assert (
        pf.worse("agree", "approximate") == "approximate" and pf.worse("unreliable", "agree") == "unreliable"
    )


def test_a_tool_with_under_5_s_of_voiced_speech_or_no_value_is_dropped() -> None:
    three = {"praat": v(30.0, 33.0, 36.0, 5000), "swiftf0": v(30.0, 33.5, 36.0, 4999),
             "opensmile": v(None, None, None, 0)}  # fmt: skip
    assert pf.judge(three) | {} == pf.judge(three)
    found = pf.judge(three)
    assert found["tools"] == ["praat"] and found["label"] == "single_tool" and found["spread_p50"] is None
    three["swiftf0"]["voiced_ms"] = 5000
    assert pf.judge(three)["tools"] == ["praat", "swiftf0"] and pf.judge(three)["label"] == "agree"
    none = {t: v(None, None, None, 0) for t in pf.TOOLS}
    assert pf.judge(none)["label"] == "too_little_speech"


def test_the_share_counts_only_clips_with_a_combined_label() -> None:
    labels = ["agree", "approximate", "unreliable", "single_tool", "too_little_speech", "agree"]
    assert pf.worse_share(labels) == pytest.approx(2 / 4)
    assert math.isnan(pf.worse_share(["single_tool", "too_little_speech"]))
    rows = [{"label": x, "item": f"c{k}"} for k, x in enumerate(labels)]
    found = pf.shares(rows)
    assert found["counts"] == {"agree": 2, "approximate": 1, "unreliable": 1, "too_little_speech": 1,
                               "single_tool": 1}  # fmt: skip
    assert (found["clips"], found["combined"], found["worse"], found["share"]) == (6, 4, 2, 0.5)
    low, high = found["interval"]
    # Resampled over the 4 clips with a combined label, so every draw has a share (no NaN).
    assert 0.0 <= low <= 0.5 <= high <= 1.0 and low < high
    assert pf.shares([{"label": "single_tool", "item": "x"}])["share"] is None


@pytest.mark.parametrize(
    ("worse", "previous", "starts"),
    [
        (4, False, "Decision: within the limit: 4 of 20"),
        (5, False, "Decision: over the limit: 5 of 20"),
        (5, True, "Decision: still over the limit after the one fix: 5 of 20"),
    ],
)
def test_more_than_20_percent_approximate_or_worse_is_over_the_limit(
    worse: int, previous: bool, starts: str
) -> None:
    found = {"worse": worse, "combined": 20, "share": worse / 20}
    assert pf.decide(True, found, previous).startswith(starts)
    assert pf.decide(False, found, previous).startswith("Decision: none: the harness calibration")
    assert pf.decide(True, {"share": None}, previous) == "Decision: none: no clean clip has a combined label"


def test_every_tool_median_on_every_tone_must_be_within_0_3_st() -> None:
    rows = [{"tone_hz": 100.0, "error_st": 0.3}, {"tone_hz": 150.0, "error_st": -0.2}]
    assert pf.calibration(rows)[0] is True
    assert pf.calibration([*rows, {"tone_hz": 220.0, "error_st": -0.31}])[0] is False
    assert pf.calibration([*rows, {"tone_hz": 220.0, "error_st": None}])[0] is False
    assert pf.calibration([])[0] is False


def test_a_tone_has_its_pitch_and_8_harmonics() -> None:
    x = pf.tone(150.0, 2.0)
    assert x.dtype == np.float32 and len(x) == 32000 and np.abs(x).max() == pytest.approx(0.3, abs=1e-6)
    spectrum = np.abs(np.fft.rfft(x))
    hz = np.fft.rfftfreq(len(x), 1 / pf.RATE)
    peaks = sorted(hz[np.argsort(spectrum)[-8:]])
    assert peaks == pytest.approx([150.0 * k for k in range(1, 9)])
    # 1/k amplitudes: the fundamental twice the second harmonic.
    assert spectrum[hz == 150.0][0] / spectrum[hz == 300.0][0] == pytest.approx(2.0, rel=1e-3)


def test_a_clip_job_is_one_answer_whose_candidate_audio_is_the_whole_input(tmp_path: Path) -> None:
    job = pf.clip_job(tmp_path / "job", pf.tone(120.0, 1.5))
    made = acoustic.candidate_audio(job)
    assert made is not None
    row = made["candidates"]["candidate"]
    pieces = acoustic.pieces(json.loads((job / row["map"]).read_text(encoding="utf-8"))["pieces"])
    assert pieces == [acoustic.Piece(0, 0, 1500)] and (job / row["wav"]).is_file()
    structure = json.loads((job / "stages" / "structure.json").read_text(encoding="utf-8"))
    assert [(a["id"], a["start_ms"], a["end_ms"]) for a in structure["answers"]] == [("A01", 0, 1500)]


def test_a_tool_that_did_not_run_or_measured_too_little_gives_no_value() -> None:
    results = {
        "praat": {"status": "ok", "candidates": {"candidate": {"scopes": {"all": {
            "p20_st": 30.0, "p50_st": 33.0, "p80_st": 36.0, "voiced_ms": 12000, "missing": None}}}}},
        "swiftf0": {"status": "ok", "candidates": {"candidate": {"scopes": {"all": {
            "voiced_ms": 3000, "missing": {"code": "insufficient_sample", "text": "..."}}}}}},
        "opensmile": {"status": "disabled", "code": "E_OPENSMILE_CONSENT_MISSING", "candidates": {}},
    }  # fmt: skip
    found = pf.tool_values(results, "candidate")
    assert found["praat"] == {"p20": 30.0, "p50": 33.0, "p80": 36.0, "voiced_ms": 12000, "why": None}
    assert (found["swiftf0"]["p50"], found["swiftf0"]["why"]) == (None, "insufficient_sample")
    assert (found["opensmile"]["p50"], found["opensmile"]["why"]) == (None, "E_OPENSMILE_CONSENT_MISSING")
    assert pf.judge(found)["label"] == "single_tool"
    # Only a step that finished counts, even if a failed one left values behind.
    results["praat"]["status"] = "failed"
    assert pf.tool_values(results, "candidate")["praat"]["p50"] is None


def test_each_tone_row_is_a_tool_median_against_the_true_pitch() -> None:
    st = pf.semitones(100.0)
    per_tool = {t: {"status": "ok", "candidates": {"candidate": {"scopes": {"all": {
        "p20_st": st - 0.1, "p50_st": st + d, "p80_st": st + 0.1, "voiced_ms": 12000, "missing": None}}}}}
        for t, d in zip(pf.TOOLS, (0.1, -0.2, 0.0), strict=True)}  # fmt: skip
    rows = pf.tone_rows({"tone-100.0": per_tool})
    assert [(r["tool"], round(r["error_st"], 6)) for r in rows] == [("praat", 0.1), ("swiftf0", -0.2),
                                                                   ("opensmile", 0.0)]  # fmt: skip
    assert rows[1]["median_hz"] == pytest.approx(100.0 * 2 ** (-0.2 / 12)) and rows[0]["label"] == "agree"


def test_the_change_between_runs_is_paired_over_the_same_clips() -> None:
    before = {"a": "approximate", "b": "approximate", "c": "agree", "d": "agree"}
    rows = [{"item": "a", "label": "agree"}, {"item": "b", "label": "approximate"},
            {"item": "c", "label": "agree"}, {"item": "d", "label": "agree"}]  # fmt: skip
    found = pf.change(rows, before)
    assert found is not None and found[0] == pytest.approx(0.25 - 0.5)
    assert found[1] <= found[0] <= found[2]
    assert pf.change(rows, {}) is None
    # A clip without a combined label in either run is left out of both shares.
    rows.append({"item": "e", "label": "approximate"})
    assert pf.change(rows, before | {"e": "single_tool"}) == found


def test_the_voices_clip_must_have_its_pre_registered_hash(tmp_path: Path) -> None:
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / Path(pf.PREREG["data"]["voices_source"]).name).write_bytes(b"not the clip")
    with pytest.raises(SystemExit, match="SHA-256"):
        pf.clean_clips(tmp_path, {"pitch_clips": {}})


def test_a_saved_tool_result_is_reused_and_a_failed_tool_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job"
    for tool in pf.TOOLS:
        pf.write(job / "analyzers" / f"{tool}.json", {"status": "ok", "tool": tool})

    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a saved result must not run again")

    monkeypatch.setattr(pf.subprocess, "run", never)
    assert {t: r["tool"] for t, r in pf.run_tools(job, tmp_path / "home").items()} == {t: t for t in pf.TOOLS}

    def fails(argv: list[str], **kwargs: Any) -> Any:
        assert kwargs["env"]["FLUENTFOLIO_HOME"] == str(tmp_path / "home")
        return pf.subprocess.CompletedProcess(argv, 1, "", "boom")

    monkeypatch.setattr(pf.subprocess, "run", fails)
    with pytest.raises(RuntimeError, match="praat failed"):
        pf.run_tools(tmp_path / "other", tmp_path / "home")


def test_the_run_needs_the_frozen_pre_registration(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"files": {pf.PREREG_FILE: "0" * 64}}), encoding="utf-8")
    argv = ["--data", str(tmp_path), "--splits", str(tmp_path / "s.json"), "--freeze", str(freeze),
            "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")]  # fmt: skip
    assert pf.main(argv) == 2 and not (tmp_path / "o").exists()


def test_the_tool_runner_takes_one_known_tool_and_a_job() -> None:
    assert pitch_tool.main(["praat"]) == 2 and pitch_tool.main(["crepe", "job"]) == 2
