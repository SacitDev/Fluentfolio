"""The experiment scripts' own logic: the seeded item selection and the joining of items for one model run."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

EVAL = Path(__file__).resolve().parents[2] / "eval"


def _load(name: str) -> Any:
    # eval/ is a folder of scripts, not a package; a script that imports a sibling needs it registered.
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


_load("asr_metrics")
splits = _load("splits")
asr_run = _load("asr_run")
counter_pairs = _load("counter_pairs")
SPEC = {"speaker_cap_s": 180, "split_target_s": 1800, "l1_exclude": "english", "row_min_s": 1,
        "row_max_s": 30, "min_words": 3}  # fmt: skip


def row(
    speaker: str, n: int, seconds: float, l1: str = "Spanish", words: int = 5, scored: bool = True
) -> dict[str, Any]:
    item = f"shard.parquet:{speaker}{n}"
    return {"id": item, "speaker": speaker, "l1": l1, "scored": scored, "words": words, "seconds": seconds}


@pytest.mark.parametrize(("rows", "chosen"), [(41, ["b", "c"]), (42, ["a", "b"]), (43, ["a", "b"])])
def test_an_interview_speaker_needs_a_row_for_every_candidate_clip(rows: int, chosen: list[str]) -> None:
    # Speaker a has `rows` eligible rows and 5 more that are not eligible (an English L1 label never counts).
    found = [row("a", k, 10) for k in range(rows)] + [row("a", 100 + k, 10, l1="English") for k in range(5)]
    found += [row("b", k, 10) for k in range(42)] + [row("c", k, 10) for k in range(50)]
    assert splits.l2_speakers(found, ["a", "b", "c"], SPEC, 42, 2) == chosen


def test_the_replaced_clips_are_the_candidate_s_in_order_of_first_use() -> None:
    recipe = {"turns": {"t1": {"kind": "question"}, "t2": {"kind": "answer"}},
              "events": [{"clip": "q1", "turn": "t1"}, {"clip": "a2", "turn": "t2"},
                         {"clip": "a1", "turn": "t2"}, {"clip": "a2", "turn": "t2"},
                         {"clip": "q1", "turn": "t1"}]}  # fmt: skip
    assert splits.candidate_clips(recipe) == ["a2", "a1"]
    # The committed interview has 42, the number its pre-registration names.
    assert len(splits.candidate_clips(splits.sb.load_recipe(splits.RECIPES / "CS-INT.json"))) == 42


def test_the_rank_is_the_seeded_sha256_of_the_key() -> None:
    assert splits.rank("spk7", 20261013) == hashlib.sha256(b"20261013:spk7").hexdigest()


def test_the_selection_values_here_are_the_pre_registered_ones() -> None:
    prereg = tomllib.loads((EVAL / "prereg" / "asr_bakeoff.toml").read_text(encoding="utf-8"))
    assert {k: prereg["data"]["edacc"][k] for k in SPEC} == SPEC


def test_the_product_s_loop_guard_and_voiced_gap_values_are_the_pre_registered_ones() -> None:
    from fluentfolio.analyzers.measures import disfluency
    from fluentfolio.asr import base, onnx_engine

    metrics = tomllib.loads((EVAL / "prereg" / "asr_bakeoff.toml").read_text(encoding="utf-8"))["metrics"]
    loop = (onnx_engine.LOOP_TOKENS_PER_S, base.NGRAM, base.NGRAM_REPEATS)
    assert loop == tuple(metrics["canary_loop_guard_values"][k] for k in ("tokens_per_s", "run", "repeats"))
    gaps = (disfluency.GAP_MS, disfluency.GAP_SPEECH_MS, disfluency.OTHER_SPEAKER_MS)
    assert gaps == tuple(metrics["voiced_gap_values"][k] for k in ("gap_ms", "speech_ms", "other_speaker_ms"))


def test_the_pronunciation_gate_limits_are_the_frozen_tolerances(tolerance: Any) -> None:
    # The gate keeps its own copy of two limits; the frozen tolerances file is the one that counts.
    gate = _load("pronunciation_gate")
    assert gate.NATIVE_MAX_PER_MIN == tolerance("integration", "native_words_to_check_per_min_max")
    assert gate.PLANTED_MIN == tolerance("integration", "planted_substitution_flag_rate_min")


def test_only_rows_with_a_given_l1_without_english_of_1_to_30_s_and_3_words_are_eligible() -> None:
    rows = [row("a", 0, 10, l1=""), row("b", 0, 10, l1="Southern British English"),
            row("c", 0, 10, l1="ENGLISH"), row("d", 0, 0.9), row("e", 0, 30.1), row("f", 0, 10, words=2),
            row("g", 0, 1.0), row("h", 0, 30.0), row("i", 0, 10, words=3, l1="Hindi"),
            row("j", 0, 10, scored=False)]  # fmt: skip
    chosen = splits.edacc_selection(rows, SPEC, 20261013)
    assert sorted(r["speaker"] for r in chosen) == ["g", "h", "i"]


def test_a_row_is_unscored_only_when_its_whole_reference_is_the_placeholder(tmp_path: Path) -> None:
    placeholder = "IGNORE_TIME_SEGMENT_IN_SCORING"
    texts = [placeholder, f" {placeholder} ", f"I said {placeholder} once", "a real answer here"]
    audio = [{"bytes": _wav(1.5, 16000), "path": None} for _ in texts]
    table = pa.table({"speaker": ["s"] * 4, "l1": ["Polish "] * 4, "text": texts, "audio": audio})
    pq.write_table(table, tmp_path / "dev.parquet")
    rows = splits.edacc_rows(tmp_path / "dev.parquet", placeholder)
    assert [r["scored"] for r in rows] == [False, False, True, True]
    assert [r["id"] for r in rows] == [f"dev.parquet:{n}" for n in range(4)]
    assert rows[0]["l1"] == "Polish" and rows[3]["words"] == 4 and rows[0]["seconds"] == 1.5


def test_a_speaker_gives_at_most_180_s_and_every_row_left_out_would_pass_the_cap() -> None:
    durations = [30, 25, 30, 20, 29, 30, 28, 30, 12]  # 234 s in all, more than the cap
    rows = [row("a", n, s) for n, s in enumerate(durations)]
    chosen = splits.edacc_selection(rows, SPEC, 20261013)
    total = sum(r["seconds"] for r in chosen)
    assert total <= 180
    # Rows are taken first-fit in hash order and a total only grows, so whatever the order, each row left
    # out must still overflow the cap at the end.
    assert all(total + r["seconds"] > 180 for r in rows if r not in chosen)


def test_speakers_are_added_only_while_the_split_is_under_its_target() -> None:
    rows = [row(s, n, 30) for s in "abcd" for n in range(7)]  # each speaker: 6 rows fit, 180 s
    chosen = splits.edacc_selection(rows, {**SPEC, "split_target_s": 400}, 20261013)
    assert len(chosen) == 18 and len({r["speaker"] for r in chosen}) == 3  # 360 < 400, 540 >= 400


def test_until_takes_items_until_their_durations_reach_the_target() -> None:
    seconds = {f"u{k}": 4.0 for k in range(6)}
    chosen = splits.until(list(seconds), seconds, 10, 20261013)
    in_order = sorted(seconds, key=lambda i: hashlib.sha256(f"20261013:{i}".encode()).hexdigest())
    assert chosen == in_order[:3]  # 4 + 4 < 10 <= 12


def _wav(seconds: float, rate: int, channels: int = 1) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone] * channels, axis=1), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_audio_bytes_become_16_khz_mono() -> None:
    x = asr_run.mono16k(_wav(2.0, 32000, channels=2))
    assert x.dtype == np.float32 and x.ndim == 1 and len(x) == 32000
    spectrum = np.abs(np.fft.rfft(x))
    assert abs(np.argmax(spectrum) * 16000 / len(x) - 440) < 1


def test_items_are_joined_with_3_s_gaps_and_each_word_goes_back_to_its_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLUENTFOLIO_DEVICE", "cpu")  # restored after the test; the script sets it
    timed = [("one", 3100, 3900), ("edge", 4200, 4600), ("lost", 5300, 5700), ("two", 7500, 8500)]
    words = [{"text": t, "start_ms": a, "end_ms": b} for t, a, b in timed]
    segments = [[3000, 3800], [4400, 4600], [5000, 6400], [7100, 9000]]
    monkeypatch.setattr(asr_run.vad, "run", lambda job, progress: {"segments": segments})
    result = {"words": words, "provenance": {"word_times": True}, "chunks": [[3000, 9000]]}
    monkeypatch.setattr(asr_run.base, "run", lambda job, progress, offered_only=True: result)
    items = [("x", np.zeros(16000, np.float32)), ("y", np.zeros(32000, np.float32))]
    run = asr_run.run_items(items, "some-model", "gpu", tmp_path / "job")
    # x spans 3000-4000 ms and y 7000-9000 ms; a word's middle within 500 ms of an item's edge belongs to it.
    assert {k: [w["text"] for w in v] for k, v in run.words.items()} == {"x": ["one", "edge"], "y": ["two"]}
    # Speech segments reaching within 500 ms of an item go with it; 5000-6400 reaches 6400 < 7000 - 500.
    assert run.speech == {"x": [[3000, 3800], [4400, 4600]], "y": [[7100, 9000]]}
    assert run.spans == {"x": (3000, 4000), "y": (7000, 9000)} and run.chunks == [[3000, 9000]]
    audio, rate = sf.read(tmp_path / "job" / "input" / "audio16k.wav")
    assert rate == 16000 and len(audio) == 16000 * (3 + 1 + 3 + 2 + 3)
    assert json.loads((tmp_path / "job" / "job.json").read_text())["settings"]["asr"]["model"] == "some-model"
    assert asr_run.os.environ["FLUENTFOLIO_DEVICE"] == "gpu"


def test_a_model_without_word_times_gets_the_ctc_pass_and_a_silent_recording_no_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLUENTFOLIO_DEVICE", "cpu")
    asked: list[bool] = []
    untimed = {"words": [], "provenance": {"word_times": False}, "chunks": []}
    timed = {"words": [{"text": "hi", "start_ms": 3100, "end_ms": 3300}], "provenance": {"word_times": True},
             "chunks": [[3000, 4000]]}  # fmt: skip
    monkeypatch.setattr(asr_run.vad, "run", lambda job, progress: {"segments": [[3000, 4000]]})

    def speech_step(job: Path, progress: Any, offered_only: bool = True) -> dict[str, Any]:
        asked.append(offered_only)
        return untimed

    monkeypatch.setattr(asr_run.base, "run", speech_step)
    monkeypatch.setattr(asr_run.phones, "asr_times", lambda job, progress: timed)
    run = asr_run.run_items([("x", np.zeros(16000, np.float32))], "canary-1b-v2", "cpu", tmp_path / "a")
    # The model a learner cannot choose runs only because the comparison asks; its words get CTC times.
    assert asked == [False] and run.words["x"] == timed["words"] and run.record == {"word_times": True}

    def silent(job: Path, progress: Any) -> None:
        raise asr_run.vad.InputError("E_INPUT_SILENT", "0 ms of speech found")

    monkeypatch.setattr(asr_run.vad, "run", silent)
    run = asr_run.run_items([("x", np.zeros(16000, np.float32))], "canary-1b-v2", "cpu", tmp_path / "b")
    assert run.words == {"x": []} and run.speech == {"x": []} and run.record == {}


def test_the_flagged_word_is_marked_in_place() -> None:
    assert counter_pairs.marked(["she", "have", "a", "car"], 1) == "she [[have]] a car"
