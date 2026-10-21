"""Alignment without MFA or models: the text and dictionary MFA gets, its guard, its command, reading its
output, the CTC aligner on hand-made scores, and the rhythm measures on hand-made phones."""

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from fluentfolio.alignment import ctc, guard, mfa, phones
from fluentfolio.analyzers.measures import rhythm
from fluentfolio.core.enums import JobState
from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs import loop
from fluentfolio.models import catalogue
from tests.support import synthetic_job

# Said in one chunk, with one stretch of speech; "accom-" is broken off. Times in ms.
SPOKEN = [
    ("I", 1000, 1200, 0), ("moved", 1250, 1600, 0), ("in", 1650, 1750, 0), ("2010,", 1800, 2600, 0),
    ("accom-", 2700, 3000, 0), ("mhm", 3100, 3300, 0),
]  # fmt: skip
SEGMENTS = [(950, 3350)]
TOKENS = [["i"], ["moved"], ["in"], ["two", "thousand", "ten"], [], ["mhm"]]
# MFA's answer for the utterance, in seconds inside its cut, which starts 200 ms before "I" at 800 ms.
WORDS = [
    (0.0, 0.2, ""), (0.2, 0.4, "i"), (0.45, 0.8, "moved"), (0.85, 0.95, "in"), (1.0, 1.2, "two"),
    (1.2, 1.6, "thousand"), (1.6, 1.8, "ten"), (1.8, 2.3, ""), (2.3, 2.5, "mhm"), (2.5, 2.7, ""),
]  # fmt: skip
PHONES = [(0.0, 0.2, ""), (0.2, 0.4, "AY1"), (0.45, 0.55, "M"), (0.55, 0.7, "UW1"), (0.7, 0.75, "V"),
          (0.75, 0.8, "D"), (0.85, 0.9, "IH0"), (0.9, 0.95, "N"), (1.0, 1.1, "T"), (1.1, 1.2, "UW1"),
          (1.2, 1.3, "TH"), (1.3, 1.45, "AW1"), (1.45, 1.5, "Z"), (1.5, 1.55, "N"), (1.55, 1.6, "D"),
          (1.6, 1.65, "T"), (1.65, 1.75, "EH1"), (1.75, 1.8, "N"), (1.8, 2.3, ""), (2.3, 2.4, "M"),
          (2.4, 2.45, "HH"), (2.45, 2.5, "M"), (2.5, 2.7, "")]  # fmt: skip
DICTIONARY = (
    "i\t0.99\t0.1\t1.0\t1.0\tAY1\nin\tIH0 N\nmoved\tM UW1 V D\ntwo\tT UW1\nthousand\tTH AW1 Z AH0 N D\n"
)


def tier(name: str, rows: list[tuple[float, float, str]]) -> str:
    out = [f'        name = "{name}" ', "        intervals: size = 0 "]
    for k, (a, b, text) in enumerate(rows, 1):
        out += [f"        intervals [{k}]:", f"            xmin = {a} ", f"            xmax = {b} ",
                f'            text = "{text}" ']  # fmt: skip
    return "\n".join(out)


def textgrid(words: list[tuple[float, float, str]], phones: list[tuple[float, float, str]]) -> str:
    return "\n".join([
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "", "item []: ", "    item [1]:",
        tier("words", words), "    item [2]:", tier("phones", phones),
    ]) + "\n"  # fmt: skip


def prepared(base: Path, spoken: list, segments: list[tuple[int, int]], duration_ms: int) -> Path:
    """A solo job up to its structure step, with the MFA dictionary file in the models folder."""
    folder = synthetic_job.make(base, spoken, segments, duration_ms)
    stages = [s for s in synthetic_job.plan("solo") if s.id in ("words", "structure")]
    loop.run_job(synthetic_job.roots(base), synthetic_job.JOB_ID, stages, audio_min=5, no_progress_s=60)
    models = synthetic_job.roots(base).models / "mfa"
    models.mkdir(parents=True)
    (models / mfa.DICTIONARY).write_text(DICTIONARY, encoding="utf-8")
    return folder


@pytest.fixture
def job(tmp_path: Path) -> Path:
    return prepared(tmp_path, SPOKEN, SEGMENTS, 5000)


def mfa_output(base: Path, deviation: str = "1.5", words: list | None = None) -> Path:
    out = mfa.mfa_root(synthetic_job.roots(base).app, synthetic_job.JOB_ID) / "out"
    (out / "candidate").mkdir(parents=True)
    (out / "candidate" / "U001.TextGrid").write_text(textgrid(words or WORDS, PHONES), encoding="utf-8")
    (out / "alignment_analysis.csv").write_text(
        "file,begin,end,speaker,overall_log_likelihood,speech_log_likelihood,phone_duration_deviation,"
        f"max_running_short_interval,snr,intensity_deviation\nU001,0.0,2.7,candidate,-50,-48,{deviation},7,20,1\n",
        encoding="utf-8",
    )  # fmt: skip
    return out


def finish(base: Path, job: Path, status: str = "ok") -> dict:
    roots = synthetic_job.roots(base)
    return mfa.finish(job, roots.app, roots.models, {"status": status})


@pytest.mark.parametrize(
    ("n", "words"),
    [(0, "zero"), (13, "thirteen"), (40, "forty"), (99, "ninety nine"), (105, "one hundred five"),
     (1000, "one thousand"), (2010, "two thousand ten"), (1_000_001, "one million one")],
)  # fmt: skip
def test_numbers_are_spelt_out_for_the_dictionary(n: int, words: str) -> None:
    assert " ".join(mfa.number_words(n)) == words


def test_the_aligners_get_lower_case_words_with_numbers_spelt_out() -> None:
    assert mfa.tokens("2010") == ["two", "thousand", "ten"]
    assert mfa.tokens("Don’t") == ["don't"]  # the curly apostrophe of a transcript
    assert mfa.tokens("well-known") == ["well", "known"]
    assert mfa.tokens("21st") == ["twenty", "first"]
    assert mfa.tokens("3.5") == ["three", "point", "five"]
    assert mfa.tokens("1,000") == ["one", "thousand"]
    assert mfa.tokens("Café") == ["cafe"]


def test_the_mini_dictionary_holds_the_words_said_and_the_missing_hesitation_sounds(tmp_path: Path) -> None:
    full = tmp_path / "full.dict"
    full.write_text(DICTIONARY + "zebra\tZ IY1 B R AH0\n", encoding="utf-8")
    lines, missing = mfa.mini_dictionary(full, {"i", "two", "ten", "mm", "mhm"})
    assert lines == ["i\t0.99\t0.1\t1.0\t1.0\tAY1", "two\tT UW1", "mm\tM", "mhm\tM HH M"]
    assert missing == {"ten"}  # MFA guesses its pronunciation


def test_each_utterance_is_cut_with_its_words_and_a_broken_word_is_left_out(job: Path) -> None:
    (cut,) = mfa.cuts(job)
    assert (cut.id, cut.speaker, cut.wav) == ("U001", "candidate", "audio16k.wav")
    # 200 ms on each side: less than half the gap to the start and to the end of the recording.
    assert (cut.start_ms, cut.end_ms, cut.words, cut.tokens) == (800, 3500, [0, 1, 2, 3, 4, 5], TOKENS)


def test_mfa_runs_with_speaker_adaptation_and_inside_the_jobs_own_folder(tmp_path: Path) -> None:
    app, models = tmp_path / "app", tmp_path / "models"
    tool = mfa.program(app, models, "20261014-000000-abcd")
    root = app / "tmp" / "mfa" / "20261014-000000-abcd"
    python = "python.exe" if sys.platform == "win32" else "bin/python"
    head = [str(app / "envs" / "mfa" / python), "-I", "-m", "montreal_forced_aligner", "align"]
    assert tool.argv[:5] == head
    assert tool.argv[5:9] == [str(root / "corpus"), str(root / "dictionary.dict"),
                              str(models / "mfa" / mfa.ACOUSTIC), str(root / "out")]  # fmt: skip
    assert "--single_speaker" not in tool.argv
    assert tool.argv[tool.argv.index("--dither") + 1] == "0"  # the same recording gives the same times
    assert tool.env["MFA_ROOT_DIR"] == str(root / "root")
    assert {Path(tool.env[k]) for k in ("HOME", "USERPROFILE")} == {root / "home"}
    first = app / "envs" / "mfa" if sys.platform == "win32" else app / "envs" / "mfa" / "bin"
    assert tool.scratch == root and tool.path[0] == first


def test_mfa_times_are_used_when_the_guard_passes(tmp_path: Path, job: Path) -> None:
    mfa_output(tmp_path)
    result = finish(tmp_path, job)
    ((row,),) = [result["utterances"]]
    assert (result["status"], row["method"], row["reason"]) == ("ok", "mfa", None)
    times = [(w["start_ms"], w["end_ms"], w["method"], w["g2p"]) for w in row["words"]]
    assert times == [
        (1000, 1200, "mfa", False), (1250, 1600, "mfa", False), (1650, 1750, "mfa", False),
        (1800, 2600, "mfa", True),  # "ten" is not in the dictionary, so its sounds were guessed
        (2700, 3000, "unaligned", False), (3100, 3300, "mfa", False),
    ]  # fmt: skip
    assert result["mfa"]["aligned"] == result["mfa"]["utterances"] == 1


def test_mfa_exiting_0_without_a_textgrid_counts_as_failed(tmp_path: Path, job: Path) -> None:
    out = mfa_output(tmp_path)
    (out / "candidate" / "U001.TextGrid").unlink()
    write_json(job / "stages" / "phones.json", {"status": "ok", "utterances": [{
        "id": "U001", "status": "ok",
        "words": [[1010, 1190], [1240, 1610], [1640, 1760], [1790, 2610], None, [3090, 3310]],
    }]})  # fmt: skip
    result = finish(tmp_path, job)
    (row,) = result["utterances"]
    assert (result["status"], row["method"], row["reason"]) == ("partial", "ctc", "no_textgrid")
    assert [(w["start_ms"], w["method"]) for w in row["words"]][3:5] == [(1790, "ctc"), (2700, "unaligned")]


def test_mfa_aligning_one_of_two_utterances_leaves_a_gap(tmp_path: Path) -> None:
    spoken = [
        ("I", 1000, 1200, 0), ("moved", 1250, 1600, 0), ("in.", 1650, 1750, 0), ("Do", 2500, 2600, 0),
        ("you", 2650, 2750, 0), ("like", 2800, 2950, 0), ("it?", 3000, 3200, 0), ("Yes,", 4000, 4200, 0),
        ("I", 4250, 4400, 0), ("do.", 4450, 4700, 0),
    ]  # fmt: skip
    folder = prepared(tmp_path, spoken, [(950, 1800), (2450, 3250), (3950, 4750)], 6000)
    assert [c.id for c in mfa.cuts(folder)] == ["U001", "U002"]
    mfa_output(tmp_path, words=[*WORDS[:4], (0.95, 1.15, "")])  # a TextGrid for the first only
    result = finish(tmp_path, folder)
    assert [(r["id"], r["method"], r["reason"]) for r in result["utterances"]] == [
        ("U001", "mfa", None), ("U002", "asr", "no_textgrid"),
    ]  # fmt: skip
    assert (result["status"], result["mfa"]["aligned"], result["mfa"]["utterances"]) == ("partial", 1, 2)


def test_without_mfa_or_ctc_times_the_speech_models_times_stay(tmp_path: Path, job: Path) -> None:
    result = finish(tmp_path, job, status="unavailable")
    (row,) = result["utterances"]
    assert (result["status"], row["method"]) == ("partial", "asr")
    assert [w["start_ms"] for w in row["words"]] == [s[1] for s in SPOKEN]
    off = finish(tmp_path, job, status="disabled")
    assert off["status"] == "ok" and off["mfa"]["version"] is None


@pytest.mark.parametrize(
    ("system", "chosen", "off"),
    [("windows", None, True), ("linux", None, False), ("windows", True, False), ("linux", False, True)],
)
def test_mfa_waits_to_be_switched_on_where_it_is_experimental(
    tmp_path: Path, job: Path, monkeypatch: pytest.MonkeyPatch, system: str, chosen: bool | None, off: bool
) -> None:
    # A list that names one system shows both sides of the rule; the shipped list names both (test_models).
    real = catalogue.get
    one = real("mfa").model_copy(update={"experimental_on": ["windows"]})
    monkeypatch.setattr(mfa.catalogue, "get", lambda c: one if c == "mfa" else real(c))
    monkeypatch.setattr(mfa, "SYSTEM", system)
    choose(job, chosen)
    roots = synthetic_job.roots(tmp_path)
    found = mfa.prepare(job, roots.app, roots.models)
    assert (found == {"status": "disabled"}) is off
    assert finish(tmp_path, job, status="disabled")["mfa"]["experimental"] is (system == "windows")


def test_the_pronunciation_dictionary_is_its_own_download_or_the_copy_mfa_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    monkeypatch.setattr(phones, "models_dir", lambda: models)
    part = catalogue.get(phones.DICTIONARY)
    (f,) = part.files_for(catalogue.SYSTEM)
    own, mfa_copy = models / "other" / part.id / f.path, models / "mfa" / mfa.DICTIONARY
    assert phones.dictionary() is None
    mfa_copy.parent.mkdir(parents=True)
    mfa_copy.write_bytes(b"x" * (f.size - 1))  # cut short, so not used
    assert phones.dictionary() is None
    mfa_copy.write_bytes(b"x" * f.size)
    assert phones.dictionary() == mfa_copy
    own.parent.mkdir(parents=True)
    own.write_bytes(b"x" * f.size)
    assert phones.dictionary() == own


@pytest.mark.parametrize("system", ["windows", "linux"])
def test_mfa_as_shipped_runs_only_when_chosen_and_is_experimental_on_both_systems(
    tmp_path: Path, job: Path, monkeypatch: pytest.MonkeyPatch, system: str
) -> None:
    monkeypatch.setattr(mfa, "SYSTEM", system)
    roots = synthetic_job.roots(tmp_path)
    choose(job, None)
    assert mfa.prepare(job, roots.app, roots.models) == {"status": "disabled"}
    assert finish(tmp_path, job, status="disabled")["mfa"]["experimental"] is True


def choose(job: Path, chosen: bool | None) -> None:
    """Record the learner's MFA choice in the job's settings; None is no choice."""
    record = read_json(job / "job.json")
    record["settings"]["components"] = {} if chosen is None else {"mfa": chosen}
    write_json(job / "job.json", record)


@pytest.mark.parametrize(
    ("deviation", "words", "reason"),
    [
        ("9.0", None, "phone_duration"),
        ("1.5", [*WORDS[:3], (0.85, 0.95, "on"), *WORDS[4:]], "words_differ"),
        ("1.5", [w for w in WORDS if w[2] != "mhm"], "words_differ"),
    ],
)
def test_a_guard_failure_or_other_words_fall_back(
    tmp_path: Path, job: Path, deviation: str, words: list | None, reason: str
) -> None:
    mfa_output(tmp_path, deviation, words)
    (row,) = finish(tmp_path, job)["utterances"]
    assert (row["method"], row["reason"]) == ("asr", reason)
    assert row["phones"] == []  # the TextGrid's phones are not used, so rhythm leaves this answer out


# The guard on hand-made rows: a passing utterance, then each rule broken alone.
GOOD_PHONES = [(0, 100, "HH"), (100, 250, "AH0"), (250, 300, "L"), (300, 450, "OW1"), (500, 600, "Y"),
               (600, 800, "UW1")]  # fmt: skip
GOOD_WORDS = [(0, 450), (500, 800)]
SPEECH = [[0, 450], [500, 800]]
WINDOW = (0, 900)


def squeezed(n: int) -> list[tuple[int, int, str]]:
    """The first word "hello" (0-450 ms) with `n` phones of 10 ms after its first phone."""
    tiny = [(100 + 10 * k, 110 + 10 * k, "AH0") for k in range(n)]
    return [(0, 100, "HH"), *tiny, (100 + 10 * n, 450, "OW1")]


def test_a_clean_utterance_passes_the_guard() -> None:
    assert guard.check(1.5, GOOD_PHONES, GOOD_WORDS, GOOD_WORDS, SPEECH, WINDOW) is None


def test_each_guard_rule_fires_alone() -> None:
    good = (GOOD_WORDS, GOOD_WORDS, SPEECH, WINDOW)
    assert guard.check(guard.T1 + 0.1, GOOD_PHONES, *good) == "phone_duration"
    short = squeezed(guard.T2 + 1)
    assert guard.short_run(short) == guard.T2 + 1 and guard.short_run(GOOD_PHONES) == 0
    assert guard.check(1.5, short + GOOD_PHONES[4:], *good) == "short_phones"
    # "you", 2 phones, may last 0.3 + 2 x 0.25 = 0.8 s: stretched to 0.9 s it fails.
    long = [*GOOD_PHONES[:4], (500, 600, "Y"), (600, 1400, "UW1")]
    stretched = [(0, 450), (500, 1400)]
    assert guard.check(1.5, long, stretched, stretched, [[0, 450], [500, 1400]], (0, 1500)) == "long_word"
    moved = [(0, 450), (1100, 1400)]  # the second word's middle 0.6 s from the speech model's
    assert guard.check(1.5, [*GOOD_PHONES[:4], (1100, 1200, "Y"), (1200, 1400, "UW1")], moved, GOOD_WORDS,
                       [[0, 450], [1100, 1400]], (0, 1500)) == "word_midpoints"  # fmt: skip
    speech = [[0, 450], [500, 800], [900, 1410]]  # 0.51 s of detected speech after the last word
    assert guard.check(1.5, GOOD_PHONES, GOOD_WORDS, GOOD_WORDS, speech, (0, 1500)) == "speech_outside_words"


def test_the_guard_limits_sit_exactly_at_the_edge() -> None:
    assert guard.check(guard.T1, GOOD_PHONES, GOOD_WORDS, GOOD_WORDS, SPEECH, WINDOW) is None
    at = [(0, 450), (500, 800)], [[0, 450], [500, 800], [900, 1400]]
    assert guard.check(1.5, GOOD_PHONES, at[0], at[0], at[1], (0, 1500)) is None  # 0.5 s outside passes
    # 50-100 and 300-950 are not covered: 50 + 650 ms.
    assert guard.outside_ms([[0, 1000]], [(100, 200), (150, 300)], (50, 950)) == 700
    # A run of T2 phones under 11 ms passes; phones of exactly 11 ms are not short.
    run = squeezed(guard.T2)
    assert guard.short_run(run) == guard.T2
    assert guard.check(1.5, run + GOOD_PHONES[4:], GOOD_WORDS, GOOD_WORDS, SPEECH, WINDOW) is None
    assert guard.short_run([(0, 100, "HH"), (100, 111, "AH0"), (111, 122, "L"), (122, 133, "OW1")]) == 0
    # "you", 2 phones, lasting exactly 0.3 + 2 x 0.25 = 0.8 s passes.
    you, spans = [*GOOD_PHONES[:4], (500, 600, "Y"), (600, 1300, "UW1")], [(0, 450), (500, 1300)]
    assert guard.check(1.5, you, spans, spans, [[0, 450], [500, 1300]], (0, 1500)) is None
    # A word's middle exactly 0.5 s from the speech model's is not moved.
    later = [(0, 450), (1000, 1300)]
    assert guard.check(1.5, [*GOOD_PHONES[:4], (1000, 1100, "Y"), (1100, 1300, "UW1")], later, GOOD_WORDS,
                       [[0, 450], [1000, 1300]], (0, 1500)) is None  # fmt: skip


@pytest.mark.parametrize(("moved", "reason"), [(1, None), (2, "word_midpoints")])
def test_one_moved_word_in_ten_passes_and_two_fail(moved: int, reason: str | None) -> None:
    ten = [(k * 200, k * 200 + 100) for k in range(10)]
    # The speech model hears the last `moved` words 0.6 s earlier than MFA places them.
    heard = [*ten[: 10 - moved], *[(s - 600, e - 600) for s, e in ten[10 - moved :]]]
    phones = [(s, e, "AH0") for s, e in ten]
    assert guard.check(1.5, phones, ten, heard, [list(w) for w in ten], (0, 2000)) == reason


def test_mfa_reads_its_analysis_file(tmp_path: Path, job: Path) -> None:
    out = mfa_output(tmp_path, deviation="")
    assert mfa.analysis(out) == {"U001": None} and mfa.analysis(tmp_path) == {}


def test_mfa_times_need_the_utterances_row_in_the_analysis_file(tmp_path: Path, job: Path) -> None:
    out = mfa_output(tmp_path)
    (out / "alignment_analysis.csv").write_text("file,phone_duration_deviation\n", encoding="utf-8")
    (row,) = finish(tmp_path, job)["utterances"]
    assert (row["method"], row["reason"]) == ("asr", "no_analysis")


def test_mfa_words_must_match_the_transcript_one_for_one() -> None:
    found = [(0, 100, ""), (100, 300, "i"), (300, 500, "<unk>"), (500, 600, "sil")]
    # "<unk>" is MFA's label for a word it could not place in its dictionary.
    assert mfa.match([["i"], ["zorbing"]], found) == [(100, 300), (300, 500)]
    assert mfa.match([["i"]], found) is None  # MFA has a word more
    assert mfa.match([["i"], ["zorbing"], ["now"]], found) is None  # or a word less
    assert mfa.match([["you"], ["zorbing"]], found) is None


def test_a_cut_takes_at_most_half_the_gap_to_the_next_word_or_the_end(tmp_path: Path) -> None:
    timed = (("Hello", 0, 400), ("yes", 500, 800), ("I", 900, 1000), ("do", 1300, 1500))
    words = [{"text": t, "start_ms": a, "end_ms": b, "cut_off": False} for t, a, b in timed]
    write_json(tmp_path / "stages" / "words.json", {"words": words})
    write_json(tmp_path / "stages" / "media.json", {"duration_ms": 1600, "music_spans": []})
    write_json(tmp_path / "stages" / "structure.json", {
        "answers": [{"id": "A1", "candidate": "candidate"}],
        "utterances": [{"id": "U001", "first_word": 1, "last_word": 1, "answer_id": "A1"},
                       {"id": "U002", "first_word": 2, "last_word": 3, "answer_id": "A1"}],
    })  # fmt: skip
    first, second = mfa.cuts(tmp_path)
    # 100 ms to each neighbour word and to the end of the recording: 50 ms of audio on that side.
    assert [(c.start_ms, c.end_ms) for c in (first, second)] == [(450, 850), (850, 1550)]


# The CTC aligner on scores made by hand: blank everywhere but where each unit is said.
VOCAB = {"<pad>": 0, "a": 1, "b": 2, "aɪ": 3, "ɪ": 4}


def scores(frames: int, said: dict[int, int]) -> np.ndarray:
    p = np.full((frames, len(VOCAB)), 0.01)
    p[:, 0] = 0.96
    for t, unit in said.items():
        p[t, :] = 0.01
        p[t, unit] = 0.96
    return np.log(p)


def test_espeak_phones_become_the_models_units_longest_first() -> None:
    units = {k: v for k, v in VOCAB.items() if k != "<pad>"}
    assert ctc.tokenize(["ˈaɪ", "b"], units) == [3, 2]
    assert ctc.tokenize(["a", "ɪ"], units) == [1, 4]
    assert ctc.tokenize(["q"], units) == []  # no unit for it


def test_ctc_places_each_unit_where_it_is_said() -> None:
    lp = scores(10, {2: 1, 3: 1, 6: 2, 7: 2})
    assert ctc.align(lp, [1, 2], blank=0) == [(2, 3), (6, 7)]
    assert ctc.word_times(lp, [[1], [2], []], blank=0, offset_ms=1000) == [(1040, 1080), (1120, 1160), None]


def untimed_job(tmp_path: Path, tracks: list[int] | None) -> Path:
    """A speech result without word times: each word spans its chunk, as the speech step leaves it."""
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    names = ["audio16k.wav"] if tracks is None else ["track_0.wav", "track_1.wav"]
    for name, seconds in zip(names, (4.0, 2.1), strict=False):  # a short second track
        sf.write(job / "input" / name, np.zeros(int(seconds * 16000)), 16000, subtype="PCM_16")
    chunks = [[1000, 1400], [2000, 2200]]
    texts = [("a", 0), ("q", 0), ("b", 0)] + [("b", 1)] * 6
    words = [{"text": t, "start_ms": chunks[k][0], "end_ms": chunks[k][1], "chunk": k, "flags": []}
             for t, k in texts]  # fmt: skip
    record = {"word_times": False, "timestamp_precision_ms": 80}
    asr = {"provenance": record, "chunks": chunks, "words": words}
    if tracks is not None:
        asr["chunk_tracks"] = tracks
    write_json(job / "stages" / "asr.json", asr)
    return job


def fake_phone_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[int]:
    """The phone model's files and run replaced by hand-made scores; returns each chunk's sample count."""
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    lengths: list[int] = []

    def run(model: Any, audio: np.ndarray) -> np.ndarray:
        lengths.append(len(audio))
        # 400 ms is 20 frames: "a" said at frames 2-3 and "b" at frame 10; anything shorter is all blank.
        return scores(len(audio) // 320, {2: 1, 3: 1, 10: 2} if len(audio) == 6400 else {})  # noqa: PLR2004

    monkeypatch.setattr(phones, "require", lambda component, models: folder)
    monkeypatch.setattr(phones, "session", lambda path: None)
    monkeypatch.setattr(phones, "scores", run)
    monkeypatch.setattr(phones, "pronounce", lambda words: {"a": ["a"], "b": ["b"], "q": ["q"]})
    return lengths


def test_words_without_times_get_the_ctc_times_of_their_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lengths = fake_phone_model(monkeypatch, tmp_path)
    result = phones.asr_times(untimed_job(tmp_path, None), lambda pct: None)
    got = [(w["text"], w["start_ms"], w["end_ms"], w["flags"]) for w in result["words"]]
    assert got == [
        # 1000 + 2 * 20 to 1000 + 4 * 20; "q" has no unit, so it has no length and stays after "a"; then "b"
        ("a", 1040, 1080, []), ("q", 1080, 1080, ["untimed"]), ("b", 1200, 1220, []),
        # Six "b" need 11 frames (a blank between repeats), and 200 ms holds 10: the chunk keeps its span.
        *[("b", 2000, 2200, ["untimed"])] * 6,
    ]  # fmt: skip
    assert lengths == [6400, 3200]
    record = result["provenance"]
    assert (record["word_times"], record["timestamp_precision_ms"]) == (True, ctc.FRAME_MS)
    assert (record["times"]["method"], record["times"]["model"], record["times"]["untimed"]) == (
        "ctc", "phone-model", 7,
    )  # fmt: skip


def test_each_chunk_is_timed_on_its_own_track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lengths = fake_phone_model(monkeypatch, tmp_path)
    phones.asr_times(untimed_job(tmp_path, [0, 1]), lambda pct: None)
    assert lengths == [6400, 1600]  # the second chunk read from the 2.1 s track ends at its end


def test_a_repeated_unit_needs_a_blank_between() -> None:
    assert ctc.align(scores(2, {0: 1, 1: 1}), [1, 1], blank=0) is None
    assert ctc.align(scores(3, {0: 1, 2: 1}), [1, 1], blank=0) == [(0, 0), (2, 2)]
    assert ctc.align(scores(3, {}), [], blank=0) is None
    # Even where every frame sounds like the unit, one blank frame must sit between the two.
    found = ctc.align(scores(4, {0: 1, 1: 1, 2: 1, 3: 1}), [1, 1], blank=0)
    assert found is not None and found[1][0] - found[0][1] == 2


def test_log_softmax_rows_are_probabilities() -> None:
    out = ctc.log_softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(np.exp(out).sum(axis=1), 1.0) and math.isclose(out[1, 0], 0.0, abs_tol=1e-12)


# Rhythm on hand-made phones; the expected values are worked out by hand in the comments. Four copies of one
# pattern, 1 s apart, hold exactly the 20 vowel sounds the values need.
RHYTHM_PHONES = [[0, 100, "AH0"], [100, 150, "T"], [150, 250, "IY1"], [250, 300, "S"], [300, 500, "OW1"],
                 [500, 560, "IY0"], [600, 700, "AA1"]]  # fmt: skip
TWENTY_VOWELS = [[a + 1000 * k, b + 1000 * k, label] for k in range(4) for a, b, label in RHYTHM_PHONES]


def test_rhythm_measures_match_a_hand_count() -> None:
    found = rhythm.measures([TWENTY_VOWELS])
    # Per copy: vowel stretches 100, 100 and 260 (OW1 and IY0 touch), a pause, then 100; consonants 50 and 50.
    # nPVI-V: per copy the pairs (100, 100) and (100, 260): 100 x (0 + 160 / 180) / 2 = 44.4, over 8 pairs.
    # VarcoV: twelve 100s and four 260s, mean 140, sample SD sqrt((12 x 40^2 + 4 x 120^2) / 15) = 71.55,
    # so 51.1. %V: 4 x 560 / (4 x 660) = 84.8. Vowel sounds 100, 100, 200, 60 and 100: mean 112 over 20.
    assert {k: (v.value, v.n) for k, v in found.items()} == {
        "npvi_v": (44.4, 8), "varco_v": (51.1, 16), "percent_v": (84.8, 28),
        "vowel_ms": (112, 20), "consonant_ms": (50, 8),
    }  # fmt: skip


def test_rhythm_needs_20_vowel_sounds(tolerance: Any) -> None:
    assert rhythm.MIN_VOWELS == tolerance("rhythm", "min_vowels")
    nineteen = TWENTY_VOWELS[:-1]  # the last copy's AA1 left out
    found = rhythm.measures([nineteen])
    assert {(v.missing.code, v.n) for v in found.values() if v.missing} == {("insufficient_sample", 19)}
    assert all(v.value is None for v in found.values())
    twenty_one = [*TWENTY_VOWELS, [4000, 4100, "EH1"]]
    for enough in (TWENTY_VOWELS, twenty_one):
        assert all(v.value is not None for v in rhythm.measures([enough]).values())


def test_without_mfa_phones_rhythm_says_why_it_is_missing() -> None:
    found = rhythm.measures([])
    assert {v.missing.code for v in found.values() if v.missing} == {rhythm.NO_MFA}


def test_enough_vowels_can_still_leave_a_value_without_its_intervals() -> None:
    # 20 vowel sounds, each alone between pauses: no two share a stretch, so nPVI-V has no pair.
    apart = rhythm.measures([[[200 * k, 200 * k + 100, "AH0"] for k in range(20)]])
    assert apart["npvi_v"].missing.code == rhythm.TOO_FEW and apart["varco_v"].value == 0.0
    # 20 vowel sounds that touch make one interval, so VarcoV has nothing to vary.
    joined = rhythm.measures([[[100 * k, 100 * k + 100, "AH0"] for k in range(20)]])
    assert joined["varco_v"].missing.code == rhythm.TOO_FEW and joined["percent_v"].value == 100.0


def test_rhythm_reaches_the_details_from_the_alignment_file(tmp_path: Path, job: Path) -> None:
    mfa_output(tmp_path)
    alignment = finish(tmp_path, job)
    (row,) = alignment["utterances"]
    assert row["method"] == "mfa" and row["phones"]
    row["phones"] = TWENTY_VOWELS  # the fixture's own phones hold 6 vowel sounds, under the 20 rhythm needs
    write_json(job / "stages" / "alignment.json", alignment)
    # The fixture's run of the first steps ended the job; run the rest as a job that was stopped.
    write_json(job / "job.json", {**read_json(job / "job.json"), "state": JobState.INTERRUPTED.value})
    synthetic_job.finish(tmp_path)
    (candidate,) = read_json(job / "evidence.json")["candidates"]
    details = {m["id"]: m for m in candidate["details"]["measures"] if m["scope"] == "all"}
    assert details["DM05"]["kind"] == "context" and details["DM05"]["context_only"] is True
    assert details["DM05"]["sources"][0]["tool"] == "mfa" and details["DM05"]["value"] is not None
    rhythm_ids = {"DM03", "DM04", "DM05", "DM06", "DM07"}
    assert rhythm_ids <= set(details) and not rhythm_ids & {m["id"] for m in candidate["measures"]}
