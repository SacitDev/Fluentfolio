"""Speech recognition without models: word forms, chunks, VAD rules, the outside-speech flag, the record."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from fluentfolio.asr import base, normalize, vad
from fluentfolio.asr import language as language_check
from fluentfolio.asr.base import Heard
from fluentfolio.asr.onnx_engine import OnnxEngine
from fluentfolio.core.errors import AppError
from fluentfolio.models import catalogue
from fluentfolio.settings.model import Settings


@pytest.mark.parametrize(
    ("raw", "text", "filler", "cut_off"),
    [
        ("longer.", "longer", False, False),
        ("Hello,", "Hello", False, False),
        ("Um,", "um", True, False),
        ("[um]", "um", True, False),
        ("UMM", "um", True, False),
        ("Er", "uh", True, False),
        ("hmm...", "hmm", True, False),
        ("wor-", "wor", False, True),
        ("well—", "well", False, False),
        ("don’t", "don't", False, False),
        ("students'", "students'", False, False),
        ("“Quote”", "Quote", False, False),
        ("state-of-the-art", "state-of-the-art", False, False),
        ("3.5", "3.5", False, False),
    ],
)
def test_each_word_form(raw: str, text: str, filler: bool, cut_off: bool) -> None:
    w = normalize.word(raw, 100, 200, 0.98765, "m")
    assert w is not None
    assert (w["text"], w["is_filler"], w["cut_off"], w["raw"]) == (text, filler, cut_off, raw)


def test_punctuation_alone_is_no_word_and_confidence_is_kept_or_explained() -> None:
    assert [normalize.word(p, 0, 80, 0.5, "m") for p in (",", "...", "—", " ")] == [None] * 4
    w = normalize.word(" so ", 0, 80, 0.123456, "parakeet")
    assert w == {
        "text": "so", "raw": "so", "start_ms": 0, "end_ms": 80, "confidence": 0.1235,
        "confidence_reason": None, "is_filler": False, "cut_off": False, "model": "parakeet", "flags": [],
    }  # fmt: skip
    w = normalize.word("so", 0, 80, None, "m")
    assert w is not None and w["confidence"] is None and w["confidence_reason"]


@pytest.mark.parametrize(
    ("segments", "duration", "expected"),
    [
        ([[1000, 5000], [5500, 9000]], 20000, [(800, 9200)]),
        # a pause of 2000 ms starts a new chunk: 1999, 2000, 2001
        ([[1000, 2000], [3999, 5000]], 9000, [(800, 5200)]),
        ([[1000, 2000], [4000, 5000]], 9000, [(800, 2200), (3800, 5200)]),
        ([[1000, 2000], [4001, 5000]], 9000, [(800, 2200), (3801, 5200)]),
        # a chunk is at most 30 s with its pads: spans of 29999, 30000 and 30001 ms
        ([[0, 10000], [11000, 29999]], 40000, [(0, 30000)]),
        ([[0, 10000], [11000, 30000]], 40000, [(0, 30000)]),
        ([[0, 10000], [11000, 30001]], 40000, [(0, 10200), (10800, 30201)]),
        # 29.9 s of speech leaves 100 ms of room, shared by the two pads
        ([[1000, 30900]], 40000, [(950, 30950)]),
        # neighbours split by length share no audio: each pad is half the 300 ms pause at most
        ([[0, 29000], [29300, 31000]], 31100, [(0, 29150), (29150, 31100)]),
        # a run the VAD cut at 30 s: the pieces touch and get no pad at the cut
        ([[300, 25000], [25000, 40000]], 50000, [(100, 25000), (25000, 40200)]),
        ([], 5000, []),
    ],
)
def test_chunks_from_hand_worked_segments(
    segments: list[list[int]], duration: int, expected: list[tuple[int, int]]
) -> None:
    got = base.chunks(segments, duration)
    assert got == expected
    assert all(b - a <= base.MAX_CHUNK_MS for a, b in got)


def test_words_a_pause_or_more_from_speech_are_flagged() -> None:
    segments = [[1000, 2000], [5000, 6000]]
    spans = {
        (700, 750): True,  # ends 250 ms before speech
        (700, 751): False,
        (2250, 2300): True,  # starts 250 ms after speech
        (2249, 2300): False,
        (3400, 3600): True,  # in the middle of the pause
        (4600, 4750): True,
        (4700, 4800): False,
        (1200, 1500): False,
    }
    words = [{"start_ms": a, "end_ms": b, "flags": []} for a, b in spans]
    base.flag_outside_speech(words, segments)
    assert {(w["start_ms"], w["end_ms"]): bool(w["flags"]) for w in words} == spans
    alone = [{"start_ms": 0, "end_ms": 10, "flags": []}]
    base.flag_outside_speech(alone, [])
    assert alone[0]["flags"] == [base.HALLUCINATION]


def probs(*parts: tuple[float, int]) -> np.ndarray:
    return np.concatenate([np.full(n, p, dtype=np.float32) for p, n in parts])


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (probs((0.1, 2), (0.9, 10), (0.1, 9)), [(2, 12)]),
        # shortest run kept: 8 frames (256 ms) stay, 7 (224 ms) go
        (probs((0.1, 2), (0.9, 8), (0.1, 9)), [(2, 10)]),
        (probs((0.1, 2), (0.9, 7), (0.1, 9)), []),
        # between the two thresholds: speech goes on, but never starts
        (probs((0.9, 5), (0.4, 5), (0.1, 9)), [(0, 10)]),
        (probs((0.4, 20)), []),
        # four quiet frames join two runs; five end the first one
        (probs((0.9, 5), (0.1, 4), (0.9, 5), (0.1, 9)), [(0, 14)]),
        (probs((0.9, 10), (0.1, 5), (0.9, 10), (0.1, 9)), [(0, 10), (15, 25)]),
        # only a frame above the start threshold resets the quiet count
        (probs((0.9, 10), (0.1, 2), (0.4, 1), (0.1, 2), (0.9, 10)), [(0, 10), (15, 25)]),
        (probs((0.9, 10), (0.1, 2), (0.6, 1), (0.1, 2), (0.9, 10)), [(0, 25)]),
        (probs((0.1, 2), (0.9, 10)), [(2, 12)]),
    ],
)
def test_vad_runs_follow_the_silero_rule(p: np.ndarray, expected: list[tuple[int, int]]) -> None:
    assert vad.runs(p) == expected


def test_runs_longer_than_30_s_are_cut_at_their_least_speech_like_frame() -> None:
    assert vad.MAX_FRAMES == 937 and vad.CUT_FROM_FRAMES == 625  # 29.98 s and 20 s in 32 ms frames
    flat = probs((0.9, 2000))
    assert vad.split_long([(0, 936)], flat) == [(0, 936)]
    assert vad.split_long([(0, 937)], flat) == [(0, 937)]
    assert vad.split_long([(0, 938)], flat) == [(0, 625), (625, 938)]
    dip = flat.copy()
    dip[700] = 0.6
    assert vad.split_long([(0, 938)], dip) == [(0, 700), (700, 938)]
    assert vad.split_long([(0, 2000)], flat) == [(0, 625), (625, 1250), (1250, 2000)]  # 750 frames left


class Recognizer:
    def __init__(self, tokens: list[str], times: list[float], logprobs: list[float] | None) -> None:
        self.result = SimpleNamespace(tokens=tokens, timestamps=times, logprobs=logprobs)

    def recognize(self, audio: np.ndarray, sample_rate: int) -> SimpleNamespace:
        assert sample_rate == base.RATE
        return self.result


def heard(
    tokens: list[str], times: list[float], logprobs: list[float] | None, seconds: float, timed: bool = True
) -> list[Heard]:
    engine = object.__new__(OnnxEngine)  # the model itself is not needed to group its tokens
    engine.model = Recognizer(tokens, times, logprobs)
    engine.timed = timed
    return engine.heard(np.zeros(int(seconds * base.RATE), dtype=np.float32))


def untimed_engine(tokens: list[str]) -> OnnxEngine:
    engine = object.__new__(OnnxEngine)
    engine.model = Recognizer(tokens, [], None)
    engine.timed = False
    return engine


def test_tokens_become_words_with_end_times_and_confidence() -> None:
    got = heard(
        [" He", " could", " wa", "it", "."], [0.08, 0.24, 0.40, 0.48, 0.80], [-0.1, 0, -0.2, -0.4, -9], 1.0
    )
    assert [h.raw for h in got] == [" He", " could", " wait."]
    # one 80 ms step after the last spoken token; the late full stop neither ends the word nor counts
    assert [(h.start_s, h.end_s) for h in got] == [
        pytest.approx((0.08, 0.16)), pytest.approx((0.24, 0.32)), pytest.approx((0.40, 0.56)),
    ]  # fmt: skip
    assert [h.confidence for h in got] == pytest.approx([math.exp(-0.1), 1.0, math.exp(-0.3)])


def test_a_word_never_ends_after_the_next_one_starts_or_after_the_chunk() -> None:
    got = heard([" a", " b", " c"], [0.0, 0.04, 0.96], None, 1.0)
    assert [(h.start_s, h.end_s) for h in got] == [(0.0, 0.04), pytest.approx((0.04, 0.12)), (0.96, 1.0)]
    assert all(h.confidence is None for h in got)
    assert heard([], [], [], 1.0) == []


def test_models_without_a_speech_engine_are_refused() -> None:
    for model in ("whisper-tiny", "silero-vad"):
        with pytest.raises(AppError) as caught:
            base.engine_for(catalogue.get(model), Path("unused"))
        assert caught.value.code == "E_COMPONENT_UNAVAILABLE"
    # The language model runs on the Whisper engine too, yet it is never a speech model to choose.
    tiny = catalogue.get("whisper-tiny").model_copy(update={"engine": "faster-whisper"})
    assert base.runnable(tiny) is False


def test_the_speech_models_offered_their_default_and_their_transcript_styles() -> None:
    # The comparison of speech models took no decision, so the cautious choices hold: Parakeet v2 by
    # default, v3 and Whisper offered, Canary not offered, and only Whisper's text counted as smoothed.
    assert base.choices() == ["parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3", "whisper-large-v3-turbo"]
    assert Settings().asr.model == "parakeet-tdt-0.6b-v2"
    styles = {m: catalogue.get(m).transcript_style for m in base.choices()}
    assert styles == {
        "parakeet-tdt-0.6b-v2": "verbatim-best-effort", "parakeet-tdt-0.6b-v3": "verbatim-best-effort",
        "whisper-large-v3-turbo": "smoothed",
    }  # fmt: skip


def test_a_model_without_its_own_word_times_is_never_offered_and_runs_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = catalogue.get("canary-1b-v2")
    assert base.loadable(canary) and not base.runnable(canary) and "canary-1b-v2" not in base.choices()
    with pytest.raises(AppError) as caught:
        base.run(asr_job(tmp_path / "offered", "canary-1b-v2"), lambda pct: None)
    assert caught.value.code == "E_COMPONENT_UNAVAILABLE"
    engine = FakeEngine([[Heard("Hello", 0.0, 2.4, 0.9)], []])
    engine.timed, engine.dropped = False, 2
    monkeypatch.setattr(base, "require", lambda c, models: Path("unused"))
    monkeypatch.setattr(base, "engine_for", lambda c, folder: engine)
    result = base.run(asr_job(tmp_path / "asked", "canary-1b-v2"), lambda pct: None, offered_only=False)
    record = result["provenance"]
    assert (record["model"], record["word_times"], record["dropped_segments"]) == ("canary-1b-v2", False, 2)
    assert [(w["text"], w["start_ms"], w["end_ms"]) for w in result["words"]] == [("Hello", 800, 3200)]


@pytest.mark.parametrize(("model", "timed"), [("parakeet-tdt-0.6b-v2", True), ("canary-1b-v2", False)])
def test_the_engine_knows_from_the_catalogue_which_models_give_word_times(
    model: str, timed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import onnx_asr

    loaded = SimpleNamespace(with_timestamps=lambda: "timestamped")
    monkeypatch.setattr(onnx_asr, "load_model", lambda *args, **kwargs: loaded)
    engine = OnnxEngine(catalogue.get(model), Path("unused"))
    # Only the model without times of its own can loop, so only it gets the repeated-run guard.
    assert (engine.model, engine.timed, engine.flags_repeats) == ("timestamped", timed, not timed)
    assert engine.dropped == 0


def test_a_model_without_word_times_gives_each_word_its_whole_chunk() -> None:
    got = heard([" He", " could", " wa", "it", "."], [], [-0.1, 0, -0.2, -0.4, -9], 2.0, timed=False)
    assert [(h.raw, h.start_s, h.end_s) for h in got] == [(" He", 0.0, 2.0), (" could", 0.0, 2.0),
                                                          (" wait.", 0.0, 2.0)]  # fmt: skip
    assert [h.confidence for h in got] == pytest.approx([math.exp(-0.1), 1.0, math.exp(-0.3)])


@pytest.mark.parametrize(("count", "kept"), [(15, True), (16, True), (17, False)])
def test_a_segment_over_8_tokens_a_second_is_dropped(count: int, kept: bool) -> None:
    # 2 s of audio: 16 tokens is exactly 8 a second, the most the guard keeps.
    tokens = [f" w{i}" for i in range(count)]
    engine = untimed_engine(tokens)
    got = engine.heard(np.zeros(2 * base.RATE, dtype=np.float32))
    assert (len(got), engine.dropped) == ((count, 0) if kept else (0, 1))


def test_a_segment_with_a_4_word_run_said_3_times_is_dropped_and_only_without_word_times() -> None:
    thrice = [f" {w}" for w in "and then we went, and then we went, and then we went home".split()]
    twice = [f" {w}" for w in "and then we went and then we went home".split()]
    audio = np.zeros(4 * base.RATE, dtype=np.float32)  # 13 tokens in 4 s: well under 8 a second
    engine = untimed_engine(thrice)
    assert (engine.heard(audio), engine.dropped) == ([], 1)
    assert len(untimed_engine(twice).heard(audio)) == 9
    # Parakeet gives times and never loops this way, so its words are kept.
    assert len(heard(thrice, [0.1 * i for i in range(13)], None, 4.0)) == 13


def words_of(text: str) -> list[dict[str, Any]]:
    return [{"text": t, "flags": []} for t in text.split()]


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        # Hand-counted: "and then we went" starts at words 0, 4 and 8, so words 0-11 are flagged.
        ("and then we went and then we went and then we went home", list(range(12))),
        ("and then we went and then we went home", []),  # twice is not a loop
        ("And then we went, and Then we went and THEN we went", list(range(12))),  # case does not matter
        ("the the the the the the", list(range(6))),  # overlapping runs at 0, 1 and 2
        ("the the the the the", []),  # only two runs of four
        ("a b c a b c a b c a b c", list(range(12))),  # each of its three 4-word runs comes 3 times
    ],
)
def test_a_four_word_run_heard_three_times_is_flagged_on_every_word(text: str, flagged: list[int]) -> None:
    words = words_of(text.replace(",", ""))
    base.flag_repeats(words)
    assert [i for i, w in enumerate(words) if w["flags"] == [base.REPEATED]] == flagged
    assert all(w["flags"] in ([], [base.REPEATED]) for w in words)


class FakeEngine:
    device = "cpu"
    precision_ms = 80
    confidence = None
    settings = {"beam": "1"}
    flags_repeats = False
    timed = True
    dropped = 0

    def __init__(self, per_chunk: list[list[Heard]]) -> None:
        self.per_chunk = per_chunk
        self.lengths: list[int] = []

    def versions(self) -> dict[str, str]:
        return {"fake": "1.0"}

    def heard(self, audio: np.ndarray) -> list[Heard]:
        self.lengths.append(len(audio))
        return self.per_chunk.pop(0)


def asr_job(
    tmp_path: Path,
    model: str,
    segments: tuple[tuple[int, int], ...] = ((1000, 3000), (6000, 8000)),
    seconds: int = 10,
) -> Path:
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    (job / "stages").mkdir()
    (job / "job.json").write_text(json.dumps({"settings": {"asr": {"model": model}}}), encoding="utf-8")
    sf.write(job / "input" / "audio16k.wav", np.zeros(seconds * base.RATE), base.RATE, subtype="PCM_16")
    (job / "stages" / "vad.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")
    return job


class SilentOnLongInput(FakeEngine):
    """Hears one 500 ms word at each whole second of its input, but nothing at all in input over `limit_s`."""

    def __init__(self, limit_s: float) -> None:
        super().__init__([])
        self.limit = limit_s * base.RATE

    def heard(self, audio: np.ndarray) -> list[Heard]:
        self.lengths.append(len(audio))
        if len(audio) > self.limit or np.abs(audio).max(initial=0) > 0.25:  # noqa: PLR2004
            return []  # a loud stretch stands in for two voices at once, which this engine never reads
        return [Heard("w", t, t + 0.5, None) for t in range(len(audio) // base.RATE)]


# One chunk, 800-25200 ms with its pads; its longest pause is 12000-13000.
TWO_RUNS = ((1000, 12000), (13000, 25000))


def test_a_chunk_heard_as_nothing_is_read_again_in_two_pieces_split_at_its_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = SilentOnLongInput(limit_s=15)
    result = run_with(monkeypatch, asr_job(tmp_path, "parakeet-tdt-0.6b-v2", TWO_RUNS, 30), engine)
    assert result["chunks"] == [[800, 25200]]
    # The whole chunk, then 800-12500 and 12500-25200, cut in the middle of the pause.
    assert engine.lengths == [24400 * 16, 11700 * 16, 12700 * 16]
    words = result["words"]
    assert len(words) == 11 + 12 and words[0]["start_ms"] == 800 and words[11]["start_ms"] == 12500
    assert {w["chunk"] for w in words} == {0}
    # Within 200 ms of a word: 500 + 10 * 900 ms of the first run, 200 + 11 * 900 of the second, of 23000.
    assert result["rereads"] == [{"chunk": 0, "pieces": 2, "covered": 0.852, "missing_speech_ms": 0}]
    assert result["warnings"] == []


def test_a_chunk_that_stays_empty_is_kept_as_heard_and_warned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = SilentOnLongInput(limit_s=0)
    result = run_with(monkeypatch, asr_job(tmp_path, "parakeet-tdt-0.6b-v2", TWO_RUNS, 30), engine)
    # Three levels of halves under the chunk, each with 4 s of speech or more: 1 + 2 + 4 + 8 reads.
    assert len(engine.lengths) == 15 and result["words"] == []
    assert result["rereads"] == [{"chunk": 0, "pieces": 1, "covered": 0.0, "missing_speech_ms": 23000}]
    assert result["warnings"] == ["W_WORDS_MISSING"]


def test_a_piece_still_without_words_warns_even_when_the_chunk_is_mostly_heard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = asr_job(tmp_path, "parakeet-tdt-0.6b-v2", TWO_RUNS, 30)
    audio = np.zeros(30 * base.RATE)
    audio[2 * base.RATE : 3 * base.RATE] = 0.5
    sf.write(job / "input" / "audio16k.wav", audio, base.RATE, subtype="PCM_16")
    result = run_with(monkeypatch, job, SilentOnLongInput(limit_s=15))
    # Halves at 12500, then 800-12500 at 6650 and 800-6650 at 3725: 800-3725 holds the loud second and has
    # 2725 ms of speech, too little to judge, so 800-6650 ends with 1800 of its 5650 ms near a word.
    # The chunk: 1800 + 5 * 900 (6650-12500, the first word reaching back to 6450) + 10100 (12500-25200)
    # = 16400 of 23000 ms near a word.
    assert result["rereads"] == [{"chunk": 0, "pieces": 4, "covered": 0.713, "missing_speech_ms": 5650}]
    assert result["warnings"] == ["W_WORDS_MISSING"]


def test_a_short_chunk_without_words_is_not_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeEngine([[], []])  # 2 s of speech in each chunk: too little to judge
    result = run_with(monkeypatch, asr_job(tmp_path, "parakeet-tdt-0.6b-v2"), engine)
    assert len(engine.lengths) == 2 and (result["rereads"], result["warnings"]) == ([], [])


def test_a_stretch_out_of_reach_is_measured_in_speech_and_ends_at_a_pause_or_a_word() -> None:
    def word(start: int, end: int) -> dict[str, Any]:
        return {"start_ms": start, "end_ms": end}

    # A 100 ms gap is less than a pause, so 1000-2000 and 2100-3000 are one stretch of 1900 ms of speech.
    assert base.missed_run([], [(1000, 2000), (2100, 3000)], 0, 4000) == 1900
    # A word at 2100-2300 reaches 1900-2500 and splits it: 900 ms before, 500 after.
    assert base.missed_run([word(2100, 2300)], [(1000, 2000), (2100, 3000)], 0, 4000) == 900
    # A 249 ms gap goes on; a 250 ms gap is a pause and ends the stretch.
    assert base.missed_run([], [(1000, 2000), (2249, 3000)], 0, 4000) == 1751
    assert base.missed_run([], [(1000, 2000), (2250, 3000)], 0, 4000) == 1000
    # Only speech inside [a, b) counts.
    assert base.missed_run([], [(1000, 2000), (2100, 3000)], 1500, 2500) == 900


# One chunk, 800-25200 ms with its pads: a 2 s question, a 1 s pause, a 21 s answer. Its longest pause is
# 3000-4000, so a re-read cuts it at 3500. Speech: 23000 ms.
QUESTION_ANSWER = ((1000, 3000), (4000, 25000))


@pytest.mark.parametrize(("missed", "reads"), [(999, 1), (1000, 3), (1001, 3)])
def test_a_chunk_that_misses_one_stretch_of_a_second_or_more_is_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missed: int, reads: int
) -> None:
    # The first word reaches back 200 ms to 1000 + missed, so the first `missed` ms of the question are out of
    # reach; the answer's one long word (3800-25000) reaches all of it. Covered: (23000 - missed) / 23000.
    whole = [Heard("w", (400 + missed) / 1000, 2.0, None), Heard("w", 3.0, 24.2, None)]
    left, right = [Heard("w", 0.2, 2.2, None)], [Heard("w", 0.3, 21.5, None)]  # 1000-3000, 3800-25000
    engine = FakeEngine([whole, left, right])
    job = asr_job(tmp_path, "parakeet-tdt-0.6b-v2", QUESTION_ANSWER, 30)
    result = run_with(monkeypatch, job, engine)
    assert len(engine.lengths) == reads
    if reads == 1:
        assert [w["start_ms"] for w in result["words"]] == [1200 + missed, 3800]
        assert (result["rereads"], result["warnings"]) == ([], [])
    else:
        assert engine.lengths[1:] == [2700 * 16, 21700 * 16]  # 800-3500 and 3500-25200
        assert [w["start_ms"] for w in result["words"]] == [1000, 3800]
        # Every ms of speech is now within 200 ms of a word.
        assert result["rereads"] == [{"chunk": 0, "pieces": 2, "covered": 1.0, "missing_speech_ms": 0}]
        assert result["warnings"] == []


def test_a_missed_stretch_that_stays_wordless_keeps_the_chunk_as_heard_without_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No word for the question, whole or alone (a laugh, say): 21000 of 23000 ms covered both times.
    answer = [Heard("w", 3.0, 24.2, None)]
    engine = FakeEngine([answer, [], [Heard("w", 0.3, 21.5, None)]])
    result = run_with(monkeypatch, asr_job(tmp_path, "parakeet-tdt-0.6b-v2", QUESTION_ANSWER, 30), engine)
    assert len(engine.lengths) == 3
    assert [w["start_ms"] for w in result["words"]] == [3800]
    assert (result["rereads"], result["warnings"]) == ([], [])


def run_with(monkeypatch: pytest.MonkeyPatch, job: Path, engine: FakeEngine) -> dict[str, Any]:
    monkeypatch.setattr(base, "require", lambda c, models: Path("unused"))
    monkeypatch.setattr(base, "engine_for", lambda c, folder: engine)
    return base.run(job, lambda pct: None)


def test_every_word_is_kept_as_heard_and_the_record_says_how(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeEngine([
        [Heard("Um,", 0.3, 0.5, 0.9), Heard("I", 0.6, 0.7, 0.8), Heard("I", 0.7, 0.8, 0.7),
         Heard("think", 0.8, 1.2, 0.6), Heard("tha-", 1.3, 1.5, 0.5)],
        [Heard("goes.", 0.2, 0.5, 0.4), Heard("—", 0.5, 0.6, 0.3)],
    ])  # fmt: skip
    result = run_with(monkeypatch, asr_job(tmp_path, "parakeet-tdt-0.6b-v2"), engine)
    assert result["chunks"] == [[800, 3200], [5800, 8200]] and engine.lengths == [2400 * 16, 2400 * 16]
    words = result["words"]
    assert [(w["text"], w["start_ms"], w["end_ms"], w["chunk"]) for w in words] == [
        ("um", 1100, 1300, 0), ("I", 1400, 1500, 0), ("I", 1500, 1600, 0), ("think", 1600, 2000, 0),
        ("tha", 2100, 2300, 0), ("goes", 6000, 6300, 1),
    ]  # fmt: skip
    assert [w["is_filler"] for w in words] == [True, False, False, False, False, False]
    assert [w["cut_off"] for w in words] == [False, False, False, False, True, False]
    assert {w["model"] for w in words} == {"parakeet-tdt-0.6b-v2"} and not any(w["flags"] for w in words)
    record = result["provenance"]
    assert record.pop("wall_ms") >= 0
    assert record == {
        "model": "parakeet-tdt-0.6b-v2", "revision": catalogue.get("parakeet-tdt-0.6b-v2").revision,
        "engine": {"fake": "1.0"}, "device": "cpu", "transcript_style": "verbatim-best-effort",
        "filler_capture": "best_effort", "word_times": True, "timestamp_precision_ms": 80, "confidence": None,
        "punctuation": "machine", "fallback": None, "dropped_segments": 0,
        "settings": {
            "max_chunk_ms": 30000, "break_gap_ms": 2000, "pad_ms": 200, "near_speech_ms": 250,
            "min_covered": 0.5, "beam": "1",
        },
    }  # fmt: skip


@pytest.mark.parametrize("guarded", [True, False])
def test_only_an_engine_known_to_loop_gets_the_repeat_flag_and_only_within_one_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guarded: bool
) -> None:
    loop = [Heard(w, 0.1 * i, 0.1 * i + 0.05, 0.9) for i, w in enumerate("so so so so so so".split())]
    split = [Heard("so", 0.1 * i, 0.1 * i + 0.05, 0.9) for i in range(5)]  # one run too few
    engine = FakeEngine([loop, split])
    engine.flags_repeats = guarded
    words = run_with(monkeypatch, asr_job(tmp_path, "whisper-large-v3-turbo"), engine)["words"]
    assert [w["chunk"] for w in words] == [0] * 6 + [1] * 5
    assert [base.REPEATED in w["flags"] for w in words] == [guarded] * 6 + [False] * 5


class Recorded:
    """Stands in for faster-whisper's model and keeps what the engine asked of it."""

    made: tuple[str, dict[str, Any]]
    options: dict[str, Any]

    def __init__(self, path: str, **kwargs: Any) -> None:
        Recorded.made = (path, kwargs)

    def transcribe(self, audio: np.ndarray, **options: Any) -> tuple[Any, None]:
        Recorded.options = options
        hello = SimpleNamespace(word=" Hello,", start=0.1, end=0.4, probability=0.9)
        return iter([SimpleNamespace(words=[hello]), SimpleNamespace(words=None)]), None


def test_whisper_runs_with_no_prompt_no_carried_text_and_its_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctranslate2
    import faster_whisper

    from fluentfolio.asr.whisper_engine import WhisperEngine

    seeds: list[int] = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", Recorded)
    monkeypatch.setattr(ctranslate2, "set_random_seed", seeds.append)
    monkeypatch.setenv("FLUENTFOLIO_DEVICE", "cpu")
    engine = WhisperEngine(catalogue.get("whisper-large-v3-turbo"), Path("model"))
    assert seeds == [0]  # the fallback temperatures sample, so a rerun needs a fixed seed
    assert engine.heard(np.zeros(base.RATE, dtype=np.float32)) == [Heard(" Hello,", 0.1, 0.4, 0.9)]
    assert Recorded.made == ("model", {"device": "cpu", "compute_type": "int8", "local_files_only": True})
    options = Recorded.options
    assert (options["initial_prompt"], options["condition_on_previous_text"], options["vad_filter"]) == (
        None, False, False,
    )  # fmt: skip
    assert (options["language"], options["word_timestamps"]) == ("en", True)
    assert (options["compression_ratio_threshold"], options["no_speech_threshold"]) == (2.4, 0.6)
    assert (options["beam_size"], options["log_prob_threshold"]) == (5, -1.0)
    assert options["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert (engine.device, engine.flags_repeats, engine.settings["initial_prompt"]) == ("cpu", True, "None")


def test_whisper_sent_to_the_gpu_without_one_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctranslate2

    from fluentfolio.asr.whisper_engine import WhisperEngine

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    monkeypatch.setenv("FLUENTFOLIO_DEVICE", "gpu")
    with pytest.raises(AppError) as caught:
        WhisperEngine(catalogue.get("whisper-large-v3-turbo"), Path("model"))
    assert caught.value.code == "E_GPU_UNAVAILABLE"


def test_whisper_sent_to_the_gpu_on_the_cpu_build_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    from fluentfolio.asr.whisper_engine import WhisperEngine

    monkeypatch.setitem(sys.modules, "nvidia", None)  # as on the CPU build, which has no nvidia package
    monkeypatch.delitem(sys.modules, "nvidia.cublas", raising=False)
    monkeypatch.setenv("FLUENTFOLIO_DEVICE", "gpu")
    with pytest.raises(AppError) as caught:
        WhisperEngine(catalogue.get("whisper-large-v3-turbo"), Path("model"))
    assert caught.value.code == "E_GPU_UNAVAILABLE"


@pytest.mark.parametrize(
    ("probability", "english", "expected"),
    [
        (0.70, 0.05, True),  # both bounds exactly
        (0.95, 0.01, True),
        (0.69, 0.05, False),  # another language not clearly likely
        (0.70, 0.051, False),  # English not unlikely enough
        (0.48, 0.18, False),  # generated noise at its most confident: a weak guess
        (0.99, 0.99, False),  # English itself
    ],
)
def test_speech_is_marked_only_when_english_is_very_unlikely_and_another_language_clear(
    probability: float, english: float, expected: bool
) -> None:
    assert language_check.marked(probability, english) is expected


def test_the_language_check_switched_off_in_settings_runs_no_model(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    settings = {"components": {"whisper-tiny": False}}
    (job / "job.json").write_text(json.dumps({"settings": settings}), encoding="utf-8")
    # No VAD result and no model: switched off, the step reads neither.
    assert language_check.run(job, lambda pct: None) == {
        "status": "disabled",
        "segments": [],
        "non_english_ms": 0,
    }


def test_a_smoothed_model_records_fillers_as_a_lower_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = run_with(monkeypatch, asr_job(tmp_path, "whisper-large-v3-turbo"), FakeEngine([[], []]))
    assert (result["provenance"]["transcript_style"], result["provenance"]["filler_capture"]) == (
        "smoothed", "lower_bound",
    )  # fmt: skip


def test_an_unknown_or_missing_model_stops_the_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for model in ("silero-vad", "no-such-model"):
        with pytest.raises(AppError) as caught:
            base.run(asr_job(tmp_path / model, model), lambda pct: None)
        assert caught.value.code == "E_COMPONENT_UNAVAILABLE"
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(tmp_path / "empty"))  # analysis never downloads
    with pytest.raises(AppError) as caught:
        base.run(asr_job(tmp_path / "missing", "parakeet-tdt-0.6b-v2"), lambda pct: None)
    assert caught.value.code == "E_MODEL_MISSING"
