"""The pronunciation fit of each phone and the words-to-check rules, on hand-made frame scores and words."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from fluentfolio.alignment import gop, phones
from fluentfolio.analyzers.measures import collect
from fluentfolio.analyzers.measures import pronunciation as pr
from fluentfolio.asr import normalize
from fluentfolio.speakers.ownership import OVERLAP

TOLERANCES = Path(__file__).resolve().parents[1] / "tolerances.toml"
X = -10.0  # a unit the frame does not hold


def table(*rows: list[float]) -> np.ndarray:
    """Frame scores with the blank in column 0 and units 1 to 4."""
    return np.array(rows, dtype=np.float64)


def rival(log_probs: np.ndarray) -> np.ndarray:
    return log_probs[:, 1:].max(axis=1)


# Frame 0 holds unit 1, frame 1 a blank, frame 2 unit 2: the written phones 1, 2 fit exactly.
CLEAN = table([X, 0, X, X, X], [0, X, X, X, X], [X, X, 0, X, X])
# Frame 0 holds unit 3 (0), not unit 1 (-5); its blank scores -4.
SWAPPED = table([-4, -5, X, 0, X], [0, X, X, X, X], [X, X, 0, X, X])


def test_a_phone_the_model_hears_best_scores_zero() -> None:
    assert gop.phone_scores(CLEAN, rival(CLEAN), [1, 2], 0) == [0.0, 0.0]


def test_a_swapped_phone_scores_the_gap_between_the_two_best_paths() -> None:
    # Written 1, 2: unit 1 on frame 0 (-5), blank (0), unit 2 (0) = -5; every other path is -14 or lower.
    # Unit 1 free: frame 0 takes its best unit (0), so the best path scores 0 and unit 1 gets -5 - 0 = -5.
    # Unit 2 free: frame 2 already holds unit 2, so the best path stays at -5 and unit 2 gets 0.
    assert gop.phone_scores(SWAPPED, rival(SWAPPED), [1, 2], 0) == [-5.0, 0.0]


def test_a_free_phone_may_join_an_equal_neighbour_without_a_blank() -> None:
    # Frames: unit 1, unit 3, blank. Written 1, 1 must put a blank between: 1, blank (-10), 1 (-10) = -20.
    # Second unit free: 1, then any unit straight after it (3, 0), blank (0) = 0, so it gets -20.
    # First unit free: it may be 1 itself and lead straight to the written 1 (-10), blank (0) = -10.
    frames = table([X, 0, X, X, X], [X, X, X, 0, X], [0, X, X, X, X])
    assert gop.phone_scores(frames, rival(frames), [1, 1], 0) == [-10.0, -20.0]


def test_units_that_cannot_fit_the_frames_have_no_scores() -> None:
    assert gop.phone_scores(CLEAN, rival(CLEAN), [1, 2, 3, 4], 0) is None


def test_what_is_heard_merges_repeats_and_drops_blanks() -> None:
    best = [1, 1, 0, 1, 2, 2, 0]  # the best unit of each frame
    frames = np.full((len(best), 5), X)
    frames[np.arange(len(best)), best] = 0
    assert gop.heard(frames, 0) == [1, 1, 2]


def test_a_word_takes_the_form_that_fits_it_best() -> None:
    # Form 0 writes the word as units 1, 2 (lowest phone -5); form 1 as 3, 2, which the frames hold exactly.
    (fit,) = gop.fit_words(SWAPPED, [[[1, 2]], [[3, 2]]], 0)
    assert fit == gop.Fit(form=1, score=0.0, phones=[0.0, 0.0], heard=[3, 2], frames=(0, 2))
    (only,) = gop.fit_words(SWAPPED, [[[1, 2]]], 0)
    assert only is not None and (only.form, only.score, only.heard) == (0, -5.0, [3, 2])


def test_each_word_gets_its_own_frames_and_a_word_without_units_none() -> None:
    first, empty, second = gop.fit_words(CLEAN, [[[1], [], [2]]], 0)
    assert first is not None and second is not None and empty is None
    assert (first.frames, second.frames, first.heard, second.heard) == ((0, 0), (2, 2), [1], [2])


def test_no_word_is_fitted_when_no_form_can_be_placed() -> None:
    assert gop.fit_words(CLEAN, [[[1, 2, 3, 4]]], 0) == [None]


def test_a_word_reports_both_forms_and_the_sounds_of_the_one_that_fits() -> None:
    vocab = {"<pad>": 0, "p": 1, "ɛ": 2, "n": 3, "æ": 4, "<unk>": 5}
    frames = np.full((3, 6), X)
    frames[[0, 1, 2], [1, 4, 3]] = 0  # the model hears the uk form
    us, uk = {"pen": ["p", "ˈɛ", "n"]}, {"pen": ["p", "ˈæ", "n"]}
    (fit,) = phones.fits(frames, [["pen"]], [us, uk], vocab)
    assert fit == {
        "score": 0.0, "phones": [0.0, 0.0, 0.0], "forms": ["pɛn", "pæn"],
        "expected": ["p", "æ", "n"], "heard": ["p", "æ", "n"], "listed": [],
    }  # fmt: skip
    (us_only,) = phones.fits(frames, [["pen"]], [us], vocab)
    assert us_only is not None and (us_only["score"], us_only["phones"]) == (-10.0, [0.0, -10.0, 0.0])


def test_the_dictionary_lines_become_pronunciations_in_the_phone_model_s_symbols() -> None:
    # By hand from the ARPAbet chart: L AY1 V is /laɪv/ and L IH1 V /lɪv/; in Y UW1 ZH AH0 L IY0 the
    # unstressed AH and IY are the reduced /ə/ and /i/. A hesitation line has no scores; a line with a
    # phone outside the chart (spn, a noise label) is left out.
    lines = [
        "live\t0.08\t0.13\t1.06\t0.99\tL AY1 V",
        "live\t0.99\t0.29\t0.52\t1.05\tL IH1 V",
        "usually\t0.99\t0.05\t1.2\t0.96\tY UW1 ZH AH0 L IY0",
        "mhm\tM HH M",
        "noise\t0.5\tspn",
        "bird\t0.99\t0.1\t1.0\t1.0\tB ER1 D",
        "better\t0.99\t0.1\t1.0\t1.0\tB EH1 T ER0",
    ]
    assert phones.listed_forms(lines) == {
        "live": ["laɪv", "lɪv"], "usually": ["juːʒəli"], "mhm": ["mhm"], "bird": ["bɜːd"], "better": ["bɛtɚ"],
    }  # fmt: skip


def test_a_word_of_several_tokens_keeps_its_joined_pronunciations_up_to_the_cap() -> None:
    vocab = {"<pad>": 0, "p": 1, "ɛ": 2, "n": 3, "<unk>": 4}
    frames = np.full((3, 5), X)
    frames[[0, 1, 2], [1, 2, 3]] = 0
    said = [{"pen": ["p", "ɛ", "n"], "a": [], "b": []}]
    listed = {"pen": ["pɛn"], "a": ["a1", "a2", "a3"], "b": ["b1", "b2", "b3"]}
    (fit,) = phones.fits(frames, [["pen"]], said, vocab, listed)
    assert fit is not None and fit["listed"] == ["pɛn"]
    (other,) = phones.fits(frames, [["pen", "c"]], [{**said[0], "c": []}], vocab, listed)
    assert other is not None and other["listed"] == []  # "c" has no dictionary line
    (both,) = phones.fits(frames, [["pen", "a", "b"]], said, vocab, listed)
    # 1 x 3 x 3 = 9 choices, the first 8 kept in order.
    assert both is not None and both["listed"] == [f"pɛna{i}b{j}" for i in (1, 2, 3) for j in (1, 2, 3)][:8]


def test_a_word_heard_as_one_of_its_dictionary_pronunciations_is_not_listed() -> None:
    # The verb "live": eSpeak gives /laɪv/ in both forms, the dictionary also /lɪv/, which the model heard.
    live = {"score": -9.0, "forms": ["laɪv", "laɪv"], "expected": ["l", "aɪ", "v"], "heard": ["l", "ɪ", "v"]}
    assert pr.flagged(live, -2) is True  # a fit stored before the dictionary forms: listed as before
    assert pr.flagged({**live, "listed": ["laɪv", "lɪv"]}, -2) is False
    assert pr.flagged({**live, "listed": ["laɪv"], "heard": ["l", "ɪ", "v"]}, -2) is True


@pytest.mark.parametrize(
    ("forms", "listed", "heard", "listed_now"),
    [
        # The verb "console" is K AH0 N S OW1 L, /kənsoʊl/ in the chart; the model writes its vowel /oː/.
        (["kɑːnsoʊl", "kɒnsəʊl"], ["kɑːnsoʊl", "kənsoʊl"], ["k", "ə", "n", "s", "oː", "l"], False),
        (["kɑːnsoʊl", "kɒnsəʊl"], [], ["k", "ɒ", "n", "s", "oʊ", "l"], False),  # the UK form, other spelling
        # "meridian" is M ER0 IH1 D IY0 AH0 N, /mɚɪdiən/; the model heard /ə/ and /ɹ/ apart.
        (["mɚɹɪdiən", "mɚɹɪdiən"], ["mɚɪdiən"], ["m", "ə", "ɹ", "ɪ", "d", "i", "ə", "n"], False),
        # eSpeak's US "theatres" has /i/ with no length mark; the model heard /iː/.
        (["θiəɾɚz", "θɪətəz"], ["θiːətɚz"], ["θ", "iː", "ə", "ɾ", "ɚ", "z"], False),
        # Real swaps stay listed: a short vowel for a long one, /ʊ/ for /ə/, /d/ for /t/.
        (["ʃɪp", "ʃɪp"], [], ["ʃ", "iː", "p"], True),
        (["fʊl", "fʊl"], [], ["f", "uː", "l"], True),
        (["juːʒuːəli", "juːʒuəli"], ["juːʒəwəli", "juːʒəli"], ["j", "uː", "ʒ", "ʊ", "l", "i"], True),
        (["tɛn", "tɛn"], ["tɛn"], ["d", "ɛ", "n"], True),
    ],
)
def test_one_sound_written_with_other_symbols_is_heard_as_written(
    forms: list[str], listed: list[str], heard: list[str], listed_now: bool
) -> None:
    fit = {"score": -9.0, "forms": forms, "expected": list(forms[0]), "heard": heard, "listed": listed}
    assert pr.flagged(fit, -2) is listed_now


@pytest.mark.parametrize(
    ("text", "guessed", "confidence", "threshold", "reason"),
    [
        ("the", False, 0.9, None, pr.WEAK),
        ("To", False, 0.9, None, pr.WEAK),  # the list is read in lower case
        ("think", True, 0.9, None, pr.GUESSED),
        ("think", False, 0.4, 0.5, pr.UNSURE),
        ("think", False, None, 0.5, pr.UNSURE),  # no confidence cannot pass a set threshold
        ("think", False, 0.5, 0.5, None),
        ("think", False, None, None, None),  # no threshold set yet: no word is held back for it
        ("three", False, 0.9, None, None),  # a content word for counting, but it has no reduced form
    ],
)
def test_the_words_left_out_of_the_check(
    text: str, guessed: bool, confidence: float | None, threshold: float | None, reason: str | None
) -> None:
    assert pr.skip_reason(text, guessed, confidence, threshold) == reason


@pytest.mark.parametrize(
    ("expected", "heard", "swaps"),
    [
        (["θ", "ɪ", "ŋ", "k"], ["s", "ɪ", "ŋ", "k"], [("θ", "s")]),
        (["p", "ɛ", "n"], ["p", "æ", "n"], [("ɛ", "æ")]),
        (["b", "æ", "θ"], ["b", "æ"], []),  # a sound left out is not a swap
        (["w", "ɛ", "s", "t"], ["v", "ɛ", "s"], [("w", "v")]),
        (["θ", "ɪ", "n"], [], []),
    ],
)
def test_swapped_sounds_come_from_the_closest_match(
    expected: list[str], heard: list[str], swaps: list[tuple[str, str]]
) -> None:
    assert pr.substitutions(expected, heard) == swaps


def word(n: int, score: float, skip: str | None = None, expected: str = "pɛn", heard: str = "pæn") -> pr.Word:
    """A word whose heard sounds differ from its form by one swap unless told otherwise."""
    fit = {"score": score, "forms": [expected, expected], "expected": list(expected), "heard": list(heard)}
    return pr.Word("U001", n, f"w{n}", skip, fit)


def test_each_answer_lists_every_word_below_the_line_in_time_order() -> None:
    scores = [-0.1, -5, -3, -0.5, -4, -6, -2.5, -7]  # six below -2: -5, -3, -4, -6, -2.5 and -7
    found = pr.check([(3000, [word(n, s) for n, s in enumerate(scores)])], flag_below=-2)
    assert [p.at[0] for p in found.listed] == [1, 2, 4, 5, 6, 7]
    assert found.listed[0].text == "w1: expected /pɛn/, heard /pæn/"


def test_a_word_heard_as_one_of_its_forms_is_not_listed_however_low_it_scores() -> None:
    """The model has some sound runs as one unit and as parts: one run heard as two parts scores low."""
    us = {"score": -8.1, "forms": ["taɪɚləs", "taɪələs"], "expected": ["t", "aɪɚ", "l", "ə", "s"],
          "heard": ["t", "aɪ", "ɚ", "l", "ə", "s"]}  # fmt: skip
    uk = {**us, "heard": ["t", "aɪ", "ə", "l", "ə", "s"]}
    changed = {**us, "heard": ["t", "aɪ", "ɚ", "l", "ɪ", "s"]}  # one sound differs from both forms
    words = [pr.Word("U001", n, "tireless", None, fit) for n, fit in enumerate([us, uk, changed])]
    found = pr.check([(3000, [*words, word(3, 0.0), word(4, 0.0)])], flag_below=-2)
    assert [p.at[0] for p in found.listed] == [2]
    assert found.listed[0].text == "tireless: expected /taɪɚləs/ (US) or /taɪələs/ (UK), heard /taɪɚlɪs/"
    assert found.coverage.startswith("Checked 5 of 5 words")  # the two heard as written were checked
    assert [pr.flagged(f, -2) for f in (us, uk, changed)] == [False, False, True]
    assert pr.flagged({**changed, "score": -2.0}, -2) is False  # on the line is not under it


def test_a_word_exactly_on_the_line_is_not_listed() -> None:
    found = pr.check([(3000, [word(n, -2.0 if n == 0 else 0.0) for n in range(5)])], flag_below=-2)
    assert found.listed == []


def test_us_and_uk_forms_are_both_shown_when_they_differ() -> None:
    bath = pr.Word(
        "U002", 3, "bath", None, {"score": -4, "forms": ["bæθ", "baθ"], "expected": [], "heard": ["s"]}
    )
    found = pr.check([(3000, [bath, *(word(n, 0.0) for n in range(4))])], flag_below=-2)
    assert [(p.utterance_id, p.at, p.text) for p in found.listed] == [
        ("U002", (3, 3), "bath: expected /bæθ/ (US) or /baθ/ (UK), heard /s/")
    ]


def test_patterns_count_listed_words_against_every_checked_word_with_the_sound() -> None:
    words = [
        word(0, -5, expected="θɪn", heard="sɪn"),
        word(1, -4, expected="θɪŋk", heard="sɪŋk"),
        word(2, 0.0, expected="θɔː", heard="θɔː"),  # said right: counted in the 3, not as a swap
        word(3, -3, expected="pɛn", heard="pæn"),  # one swap of its kind is not a pattern
        word(4, 0.0),
    ]
    found = pr.check([(3000, words)], flag_below=-2)
    assert found.patterns == "/θ/ heard as /s/ in 2 of 3 words"
    assert (
        pr.check([(3000, [word(n, 0.0) for n in range(5)])], flag_below=-2).patterns == "no repeated pattern"
    )


def test_the_coverage_line_says_what_was_left_out_and_why() -> None:
    first = [word(n, 0.0) for n in range(5)] + [
        word(5, 0.0, pr.WEAK),
        word(6, 0.0, pr.WEAK),
        word(7, 0.0, pr.GUESSED),
    ]
    short = [word(n, 0.0) for n in range(6)]
    found = pr.check([(3000, first), (1500, short)], flag_below=-2)
    assert found.coverage == (
        "Checked 5 of 14 words (not checked: 6 words in answers too short to check; "
        "2 words with a normal reduced form; 1 word whose expected sounds were guessed)."
    )
    assert (
        pr.check([(3000, [word(n, 0.0) for n in range(5)])], flag_below=-2).coverage
        == "Checked 5 of 5 words."
    )
    alone = pr.check([(1000, [word(0, 0.0)])], flag_below=-2).coverage
    assert alone == "Checked 0 of 1 word (not checked: 1 word in answers too short to check)."


def test_the_words_of_an_answer_as_the_check_sees_them() -> None:
    said = ["Well,", "um", "think", "wor-", "the", "Zyx", "pen", "ten", "next", "one"]
    words = [normalize.word(w, 100 * i, 100 * i + 90, None, "m") for i, w in enumerate(said)]
    words[6]["flags"] = [OVERLAP]
    fit = {"score": 0.0}
    fits = [fit, None, fit, None, fit, fit, fit, None]
    fitted = {"U001": {"fits": fits, "guessed": [False] * 5 + [True, False, False]}}
    utterances = [
        {"id": "U001", "answer_id": "A01", "first_word": 0, "last_word": 7},
        {"id": "U002", "answer_id": "A01", "first_word": 8, "last_word": 8},  # the phone step had no row
        {"id": "U003", "answer_id": "A02", "first_word": 9, "last_word": 9},  # another answer's
    ]
    ((length, found),) = collect._check_words(
        [{"id": "A01", "start_ms": 1500, "end_ms": 4000}], utterances, words, fitted
    )
    assert length == 2500
    assert [(w.utterance_id, w.position, w.text, w.skip) for w in found] == [
        ("U001", 0, "Well", None), ("U001", 2, "think", None), ("U001", 4, "the", pr.WEAK),
        ("U001", 5, "Zyx", pr.GUESSED), ("U001", 6, "pen", pr.OVERLAP), ("U001", 7, "ten", pr.NO_FIT),
        ("U002", 0, "next", pr.NO_FIT),
    ]  # fmt: skip


def floors() -> dict[str, float]:
    found = tomllib.loads(TOLERANCES.read_text(encoding="utf-8"))["pronunciation"]
    return {k: v["floor"] for k, v in found.items()}


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_an_answer_needs_the_minimum_length_and_words_to_be_checked(delta: int) -> None:
    limits = floors()
    assert (pr.MIN_ANSWER_MS, pr.MIN_WORDS) == (limits["min_answer_ms"], limits["min_words"])
    long_enough = pr.check([(pr.MIN_ANSWER_MS + delta, [word(n, -9) for n in range(6)])], flag_below=-2)
    enough_words = pr.check([(3000, [word(n, -9) for n in range(pr.MIN_WORDS + delta)])], flag_below=-2)
    assert (len(long_enough.listed) > 0, len(enough_words.listed) > 0) == (delta >= 0, delta >= 0)
