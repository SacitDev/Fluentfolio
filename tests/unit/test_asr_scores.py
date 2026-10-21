"""The speech-model comparison's scoring and decisions, on hand-made words with hand-counted answers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"


def _load(name: str) -> Any:
    # eval/ is a folder of scripts, not a package; a script that imports a sibling needs it registered.
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


_load("asr_metrics")
scores = _load("asr_scores")
V2, V3, WHISPER, CANARY = scores.MODELS


def w(
    text: str, start: int, end: int, conf: float | None = None, flags: tuple[str, ...] = ()
) -> dict[str, Any]:
    """One word as the speech step writes it, in the first chunk."""
    return {"text": text, "start_ms": start, "end_ms": end, "confidence": conf, "chunk": 0,
            "flags": list(flags)}  # fmt: skip


def test_an_item_row_holds_the_hand_counted_errors_fillers_edits_times_and_unsure_words() -> None:
    words = [w("So", 0, 200, 0.9), w("uh", 300, 500, 0.8), w("I", 600, 700, 0.95), w("has", 800, 1000, 0.3),
             w("two", 1100, 1300, 0.9), w("cats", 1400, 1700, 0.85),
             w("today", 1800, 2100, 0.2, flags=("untimed",))]  # fmt: skip
    row = scores.text_row("So um I have two cats.", words, [[0, 2500]], [], False, 0.5)
    # so=so, um->uh, i=i, have->has, two=two, cats=cats, +today: 2 substitutions and 1 insertion in 6 words.
    assert (row["ref_words"], row["errors"], row["inserted"]) == (6, 3, 1)
    # Without fillers: "so i have two cats" against "so i has two cats today".
    assert (row["content_words"], row["content_errors"]) == (5, 2)
    assert (row["filler_tp"], row["filler_fp"], row["filler_fn"]) == (1, 0, 0)  # um and uh: both fillers
    assert row["grammar_edits"] == 1 and row["flags"] == [("have", "has", 3)]
    assert (row["words"], row["timed_words"]) == (7, 6)  # "today" has only its chunk's span
    # Wrong words: uh (0.8), has (0.3), today (0.2); under 0.5: has and today; no right word under it.
    assert (row["unsure_tp"], row["unsure_fp"], row["unsure_fn"]) == (2, 0, 1)
    assert row["output"] == "so uh i has two cats today" and "gaps" not in row
    no_threshold = scores.text_row("So um I have two cats.", words, [[0, 2500]], [], False, None)
    assert no_threshold["unsure_tp"] is None and no_threshold["unsure_fn"] is None


def test_a_disfluent_item_counts_repairs_kept_and_judges_each_voiced_gap_against_the_reference() -> None:
    words = [w("I", 0, 200), w("went", 800, 1000), w("to", 1050, 1150), w("the", 1200, 1300),
             w("store", 1900, 2300), w("today", 2800, 3100)]  # fmt: skip
    row = scores.text_row("I I um went to the sto* store today", words, [[0, 3200]], [[0, 3100]], True, None)
    # The repeated "I I" and the cut-off "sto*" are both missing from the output.
    assert (row["repairs_kept"], row["repairs_total"]) == (0, 2)
    # Voiced gaps (200 ms or more, all of it speech): I|went, the|store and store|today. I|went holds the
    # missing "um", the|store the missing cut-off, and nothing lies between store and today.
    assert (row["gaps"], row["gaps_correct"]) == (3, 2)


def test_a_gap_is_voiced_only_with_200_ms_of_speech_in_it_and_no_other_speaker_near() -> None:
    words = [w("a", 0, 200), w("b", 800, 1000), w("c", 1500, 1700), w("d", 2000, 2200)]
    speech = [[0, 1000], [1500, 2200]]
    # a|b: 600 ms of speech; b|c: none; c|d: 300 ms, but another speaker starts 250 ms after it.
    assert scores.gap_lefts(words, speech, [(2250, 2600)]) == [0]
    assert scores.gap_lefts(words, speech, []) == [0, 2]
    assert scores.gap_lefts(words, [[0, 200], [790, 1000]], []) == []  # 10 ms of speech in a|b


def _session() -> dict[str, Any]:
    segments = [("e1", "examiner", 0, 1000, "Tell me about tea"),
                ("c1", "candidate", 1500, 2500, "I like tea"), ("hes_0", "candidate", 2700, 3100, "um"),
                ("c2", "candidate", 3300, 4500, "and coffee")]  # fmt: skip
    return {
        "segments": [{"id": i, "role": r, "start_ms": a, "end_ms": b, "text": t}
                     for i, r, a, b, t in segments],
        "turns": [{"role": "examiner", "start_ms": 0, "end_ms": 1000},
                  {"role": "candidate", "start_ms": 1500, "end_ms": 4500}],
    }  # fmt: skip


SESSION_WORDS = [w("Tell", 30, 200), w("me", 250, 400), w("about", 450, 700), w("tea", 750, 980),
                 w("I", 1520, 1650), w("like", 1700, 2000), w("tea", 2300, 2480), w("and", 3320, 3500),
                 w("coffee", 3600, 4480)]  # fmt: skip


def test_a_session_row_holds_clip_edges_word_errors_and_voiced_gaps_against_the_spliced_hesitation() -> None:
    speech = [[0, 1000], [1500, 4500]]
    row = scores.session_row(_session(), SESSION_WORDS, [[0, 4600]], speech, [(2700, 3100)])
    # Edges of the three clips whose first and last words were found ("um" was not written):
    # e1 |30-0|, |980-1000|; c1 |1520-1500|, |2480-2500|; c2 |3320-3300|, |4480-4500|.
    assert row["edges"] == [30, 20, 20, 20, 20, 20]
    assert (row["ref_words"], row["errors"], row["words"], row["timed_words"]) == (10, 1, 9, 9)
    # The candidate's voiced gaps: like|tea (300 ms) holds nothing; tea|and holds the unwritten "um".
    assert (row["gaps"], row["gaps_correct"]) == (2, 1)
    without = scores.session_row(_session(), SESSION_WORDS, [[0, 4600]], speech, [])
    assert (without["gaps"], without["gaps_correct"]) == (2, 0)


def test_a_session_gap_near_the_examiner_is_not_voiced() -> None:
    truth = _session()
    # The examiner now speaks until 2000 ms: within 300 ms of like|tea (2000-2300), not of tea|and (2480-).
    truth["segments"][0]["end_ms"] = 2000
    row = scores.session_row(truth, SESSION_WORDS, [[0, 4600]], [[0, 4500]], [(2700, 3100)])
    assert (row["gaps"], row["gaps_correct"]) == (1, 1)


ROW_A = {"unit": "a", "seconds": 30, "ref_words": 10, "errors": 2, "content_words": 8, "content_errors": 1,
         "filler_tp": 1, "filler_fp": 1, "filler_fn": 0, "grammar_edits": 1, "inserted": 1, "words": 9,
         "timed_words": 9, "gaps": 2, "gaps_correct": 1, "edges": [10, 30], "unsure_tp": 1, "unsure_fp": 1,
         "unsure_fn": 1}  # fmt: skip
ROW_B = {"unit": "b", "seconds": 90, "ref_words": 30, "errors": 1, "content_words": 25, "content_errors": 0,
         "filler_tp": 2, "filler_fp": 0, "filler_fn": 2, "grammar_edits": 0, "inserted": 0, "words": 31,
         "timed_words": 30, "gaps": 0, "gaps_correct": 0, "edges": [20], "unsure_tp": 2, "unsure_fp": 0,
         "unsure_fn": 1, "repairs_kept": 1, "repairs_total": 4}  # fmt: skip


def test_a_set_s_measures_pool_its_items_counts_instead_of_averaging_their_rates() -> None:
    found = scores.pooled([ROW_A, ROW_B])
    assert found["wer_pct"] == pytest.approx(7.5)  # 3 / 40; the mean of 20 % and 3.33 % would be 11.67
    assert found["content_wer_pct"] == pytest.approx(100 / 33)
    # Fillers 3 found, 1 wrong, 2 missed: P 3/4, R 3/5, F1 6/9.
    assert (found["filler_precision"], found["filler_recall"]) == pytest.approx((0.75, 0.6))
    assert found["filler_f1_pct"] == pytest.approx(200 / 3)
    assert (found["grammar_edits_per_100"], found["inserted_per_100"]) == pytest.approx((2.5, 2.5))
    assert (found["repair_retention"], found["timed_share"]) == pytest.approx((0.25, 39 / 40))
    assert found["words_per_minute"] == pytest.approx(20.0)  # 40 words in 2 minutes
    assert (found["gap_precision"], found["gaps"]) == (0.5, 2)
    assert (found["median_edge_ms"], found["edges"]) == (20, 3)
    assert (found["unsure_precision"], found["unsure_recall"]) == pytest.approx((0.75, 0.6))


def test_a_measure_with_nothing_to_count_is_none() -> None:
    found = scores.pooled([{"unit": "a", "seconds": 0, "ref_words": 0, "words": 0}])
    assert found["wer_pct"] is None and found["words_per_minute"] is None and found["filler_f1_pct"] is None
    assert found["unsure_f1"] is None and found["median_edge_ms"] is None and found["gap_precision"] is None


def test_the_interval_resamples_units_and_the_difference_needs_the_same_units() -> None:
    same = [{**ROW_A, "unit": u, "ref_words": 100, "errors": 10} for u in "abc"]
    assert scores.interval(same, "wer_pct") == pytest.approx((10.0, 10.0, 10.0))  # every resample is 10 %
    better = [{**r, "errors": 5} for r in same]
    assert scores.difference(better, same, "wer_pct") == pytest.approx((-5.0, -5.0, -5.0))
    assert scores.difference(better[:2], same, "wer_pct") is None
    mixed = [{**ROW_A, "unit": "a", "ref_words": 100, "errors": 0}, {**ROW_A, "unit": "b", "ref_words": 100,
             "errors": 20}]  # fmt: skip
    value, low, high = scores.interval(mixed, "wer_pct")
    assert value == pytest.approx(10.0) and low == pytest.approx(0.0) and high == pytest.approx(20.0)


def _text(unit: str, **counts: int) -> dict[str, Any]:
    return {"unit": unit, "seconds": 10, "ref_words": 100, "errors": 0, **counts}


@pytest.mark.parametrize(
    ("errors", "inside"), [(9, False), (10, True), (35, True), (36, False)]
)  # fmt: skip
def test_the_calibration_holds_parakeet_v2_s_librispeech_wer_from_1_to_3_5(errors: int, inside: bool) -> None:
    librispeech = [{**_text("s", errors=errors), "ref_words": 1000}]  # 9 errors: 0.9 % WER
    found = {"librispeech": {V2: librispeech},
             "nyra_test": {CANARY: [_text("c", filler_tp=1, filler_fp=2, filler_fn=2)]}}  # fmt: skip
    ok, lines = scores.calibration(found)
    assert ok is inside and lines[0].endswith("yes" if inside else "NO")
    assert lines[1] == "canary_nyra_test_filler_f1_pct: 33.33 in [16.7, 36.7]: yes"  # 2 / 6


def test_the_calibration_fails_when_canary_s_filler_f1_is_outside_its_range() -> None:
    found = {"librispeech": {V2: [_text("s", errors=2)]},
             "nyra_test": {CANARY: [_text("c", filler_tp=1, filler_fp=1, filler_fn=2)]}}  # fmt: skip
    ok, lines = scores.calibration(found)
    assert not ok and lines[1] == "canary_nyra_test_filler_f1_pct: 40.00 in [16.7, 36.7]: NO"
    assert not scores.calibration({})[0]  # a model that did not run is no calibration


def _nyra(tp: int, fp: int, fn: int, units: str = "cd") -> list[dict[str, Any]]:
    return [_text(u, filler_tp=tp, filler_fp=fp, filler_fn=fn) for u in units]


def _edacc(errors: int) -> list[dict[str, Any]]:
    return [_text(u, errors=errors) for u in "st"]


@pytest.mark.parametrize(
    ("v3_fillers", "v3_errors", "proposed"),
    [((2, 0, 0), 11, True), ((2, 0, 0), 12, False), ((27, 23, 23), 10, False)],
)  # fmt: skip
def test_v3_is_proposed_only_5_filler_f1_points_up_and_at_most_1_wer_point_worse(
    v3_fillers: tuple[int, int, int], v3_errors: int, proposed: bool
) -> None:
    # v2: F1 50 % on every clip and WER 10 %; v3: F1 100 % (or 54 %, 4 points up) and WER 11 or 12 %.
    found = {"nyra_test": {V2: _nyra(1, 1, 1), V3: _nyra(*v3_fillers)},
             "edacc_test": {V2: _edacc(10), V3: _edacc(v3_errors)}}  # fmt: skip
    line = scores.default_model(found)
    assert ("propose Parakeet v3" in line) is proposed and ("Parakeet v2 stays" in line) is not proposed


def test_v3_is_not_proposed_when_the_filler_interval_reaches_0() -> None:
    # Pooled, v3's F1 is 20.8 points up (87.5 against 66.7), but clip d alone favours v2.
    v2 = [_text("c", filler_tp=0, filler_fp=2, filler_fn=2), _text("d", filler_tp=4)]
    v3 = [_text("c", filler_tp=4), _text("d", filler_tp=3, filler_fp=1, filler_fn=1)]
    f1 = scores.difference(v3, v2, "filler_f1_pct")
    assert f1[0] == pytest.approx(87.5 - 200 / 3) and f1[1] < 0
    found = {"nyra_test": {V2: v2, V3: v3}, "edacc_test": {V2: _edacc(10), V3: _edacc(10)}}
    assert "Parakeet v2 stays" in scores.default_model(found)


def _canary_sets(c_errors: int = 12, c_edge: int = 140) -> dict[str, Any]:
    return {
        "nyra_test": {CANARY: _nyra(1, 1, 1), WHISPER: _nyra(1, 1, 1)},
        "edacc_test": {CANARY: _edacc(c_errors), WHISPER: _edacc(10)},
        "set_b": {CANARY: [{"unit": "x", "edges": [c_edge]}], V2: [{"unit": "x", "edges": [100]}]},
    }


@pytest.mark.parametrize(
    ("change", "failed"),
    [({}, None), ({"c_errors": 13}, "WER"), ({"c_edge": 141}, "median clip edge"),
     ({"rtf": 1.01}, "CPU real-time factor"), ({"peak_gb": 6.6}, "peak RAM"), ({"rtf": None}, "CPU")],
)  # fmt: skip
def test_canary_is_added_only_when_every_condition_holds(change: dict[str, Any], failed: str | None) -> None:
    sets_args = {k: v for k, v in change.items() if k.startswith("c_")}
    measured = {k: v for k, v in change.items() if not k.startswith("c_")}
    line = scores.canary(_canary_sets(**sets_args), {CANARY: {"rtf": 1.0, "peak_gb": 6.5, **measured}})
    # At the limits (WER 12 = 10 + 2, edge 140 = 100 + 40, real-time factor 1.0, 6.5 GB) all conditions hold.
    if failed is None:
        assert "Canary: added next to Whisper; failed: none" in line
    else:
        assert "Canary: not added; failed: " + failed in line


def test_voiced_gap_markers_go_into_the_text_only_at_precision_0_7_on_both() -> None:
    def gaps(correct: int) -> list[dict[str, Any]]:
        return [{"unit": "u", "gaps": 100, "gaps_correct": correct}]

    found = {"composed_gaps": {V2: gaps(70), V3: gaps(70), WHISPER: gaps(90), CANARY: []},
             "nyra_test": {V2: gaps(70), V3: gaps(69), WHISPER: gaps(69), CANARY: gaps(90)}}  # fmt: skip
    line = scores.voiced_gap_markers(found)
    assert line.startswith(f"Decision: voiced-gap markers go into the text for the AI chat for {V2} (")
    found["nyra_test"][V2] = gaps(69)
    assert "markers stay in the details folder" in scores.voiced_gap_markers(found)


@pytest.mark.parametrize(("long_s", "slow"), [(179.9, False), (180.0, False), (180.1, True), (None, False)])
def test_the_decisions_are_one_line_each_and_parakeet_v2_over_3_minutes_on_the_cpu_is_reported(
    long_s: float | None, slow: bool
) -> None:
    found = {**_canary_sets(), "composed_gaps": {m: [] for m in scores.MODELS}}
    found["nyra_test"].update({V2: _nyra(1, 1, 1), V3: _nyra(1, 1, 1)})
    found["edacc_test"].update({V2: _edacc(10), V3: _edacc(10)})
    lines = scores.decide(found, {V2: {"threshold": 0.35}, CANARY: {"threshold": None}}, {}, long_s)
    assert len(lines) == 6 and all(line.startswith("Decision: ") for line in lines)
    assert lines[2].startswith(f"Decision: transcript style: {V2} verbatim-best-effort, {V3} verbatim-best-")
    assert f"{WHISPER} smoothed, {CANARY} smoothed" in lines[2]
    assert lines[4] == f"Decision: confidence thresholds, fitted on dev: {V2} 0.35, {CANARY} n/a"
    proposed = "propose GPU support for it; the product does not change"
    assert lines[5].endswith(proposed if slow else "no change")


def test_a_row_saved_by_the_runner_scores_in_its_item_s_own_time(tmp_path: Path) -> None:
    bakeoff = _load("asr_bakeoff")
    words = [w("hello", 3100, 3400, 0.9), w("there", 3500, 3900, 0.4)]
    saved = {"model": V2, "chunks": [[3000, 4000]], "record": {},
             "items": {"x": {"span": [3000, 4000], "words": words, "speech": [[3050, 3950]]}}}  # fmt: skip
    (tmp_path / "words" / "nyra_test" / V2).mkdir(parents=True)
    (tmp_path / "words" / "nyra_test" / V2 / "all.json").write_text(json.dumps(saved), encoding="utf-8")
    (tmp_path / "refs").mkdir()
    refs = {"x": {"ref": "hello there", "unit": "x", "seconds": 1.0}}
    (tmp_path / "refs" / "nyra_test.json").write_text(json.dumps(refs), encoding="utf-8")
    [row] = bakeoff.rows_of("nyra_test", V2, tmp_path, {V2: {"threshold": 0.5}})
    assert (row["set"], row["item"], row["unit"], row["seconds"]) == ("nyra_test", "x", "x", 1.0)
    assert row["errors"] == 0
    assert (row["timed_words"], row["unsure_fp"]) == (2, 1)  # "there" is right but under 0.5
    assert bakeoff.run_models("nyra_test", [V2, WHISPER]) == [V2, WHISPER, scores.PROMPTED]
    assert bakeoff.run_models("edacc_test", [V2, WHISPER]) == [V2, WHISPER]


def test_the_threshold_grid_runs_from_0_05_to_0_95_over_both_dev_sets() -> None:
    bakeoff = _load("asr_bakeoff")
    # Wrong words at 0.92 and right ones at 0.97: only the grid's last point, 0.95, separates them.
    edacc = [{"confidence": [(0.92, True), (0.97, False)]}]
    nyra = [{"confidence": [(0.92, True), (0.97, False), (0.97, False)]}]
    found = bakeoff.thresholds_from({"edacc_dev": {V2: edacc}, "nyra_dev": {V2: nyra}})
    assert found[V2]["threshold"] == 0.95 and found[V2]["f1"] == 1.0 and found[V2]["words"] == 5
    # A wrong word at 0.99 is over every grid point: F1 0, under the 0.2 floor, so no threshold.
    sure = [{"confidence": [(0.99, True), (0.5, False)]}]
    assert bakeoff.thresholds_from({"edacc_dev": {V2: sure}, "nyra_dev": {V2: []}})[V2]["threshold"] is None
    # At 0.04 it is under the first point, 0.05.
    low = [{"confidence": [(0.04, True), (0.5, False)]}]
    assert bakeoff.thresholds_from({"edacc_dev": {V2: low}, "nyra_dev": {V2: []}})[V2]["threshold"] == 0.05
