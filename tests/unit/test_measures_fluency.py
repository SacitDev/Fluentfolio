"""Fluency timing on hand-built VAD segments, with every expected value worked out by hand."""

import pytest

from fluentfolio.analyzers.measures.fluency import latency, pauses, place, timing
from fluentfolio.analyzers.measures.values import Missing, Value, rounded

# One answer from 0 to 20 s; speech 0-4, 4.5-9, 9.3-15 and 16-20 s.
TURN = [(0, 20000)]
SPEECH = [(0, 4000), (4500, 9000), (9300, 15000), (16000, 20000)]
WORDS = ["cat"] * 68  # 68 one-syllable words


def test_hand_computed_answer():
    m = timing(TURN, SPEECH, WORDS)
    # gaps 0.5, 0.3 and 1.0 s are all at least 0.25 s: 3 pauses in 20 s = 3 / 20 * 60 = 9.0 per minute
    assert m["pauses_per_min"].value == 9.0
    assert m["pauses_per_min"].n == 3
    assert m["mean_pause"].value == 0.6  # (0.5 + 0.3 + 1.0) / 3
    assert m["speaking_time"].value == 18.2  # 4.0 + 4.5 + 5.7 + 4.0
    assert m["speech_rate"].value == 3.4  # 68 / 20
    assert m["articulation_rate"].value == 3.7  # 68 / 18.2 = 3.736
    assert m["mean_length_of_run"].value == 17.0  # 68 syllables / 4 runs
    assert m["words_per_minute"].value == 204.0  # 68 words / (20 / 60) min


def test_longer_pause_views():
    m = timing(TURN, SPEECH, WORDS)
    assert m["pauses_500ms"].value == 2  # 0.5 and 1.0 s; 0.5 counts because the rule is "at least"
    assert m["pauses_1s"].value == 1  # 1.0 s only


@pytest.mark.parametrize(("gap_ms", "found"), [(249, 0), (250, 1), (251, 1)])
def test_pause_is_at_least_250_ms(gap_ms, found):
    speech = [(0, 1000), (1000 + gap_ms, 2000 + gap_ms)]
    assert len(pauses([(0, 2000 + gap_ms)], speech)) == found


def test_short_gap_counts_as_speaking_time():
    m = timing([(0, 2000)], [(0, 1000), (1200, 2000)], ["cat"] * 4)
    assert m["speaking_time"].value == 2.0  # the 0.2 s gap is not a pause, so it stays in the run
    assert m["mean_length_of_run"].value == 4.0  # one run holds all 4 syllables
    assert m["pauses_per_min"].value == 0.0


def test_read_aloud_question_is_left_out():
    # Solo: the learner reads the question from 0 to 5 s, then answers from 6 to 40 s.
    speech = [(0, 5000), (6000, 20000), (20500, 40000)]
    m = timing([(6000, 40000)], speech, ["cat"] * 68)
    assert m["speech_rate"].value == 2.0  # 68 / 34 s; the question time is not in the answer
    assert m["pauses_1s"].value == 0  # the 1 s silence after the question is not inside the answer
    assert m["pauses_per_min"].n == 1  # only the 0.5 s pause at 20 s
    assert m["speaking_time"].value == 33.5  # 14.0 + 19.5


def test_partner_turn_is_never_a_pause():
    # Two 1 s answers with the partner speaking from 1.5 to 9.5 s in between.
    speech = [(0, 1000), (1500, 9500), (10000, 11000)]
    m = timing([(0, 1000), (10000, 11000)], speech, ["cat"] * 4)
    assert m["pauses_per_min"].n == 0
    assert m["speech_rate"].value == 2.0  # 4 syllables / 2 s of answers, not / 11 s
    assert m["speaking_time"].value == 2.0


def test_no_speech_gives_reasons_not_numbers():
    m = timing([(0, 5000)], [(6000, 7000)], ["cat"])
    for value in m.values():
        assert value.value is None
        assert value.missing is not None and value.missing.code == "no_speech"
    assert m["speech_rate"].unit == "syll/s"


def test_no_pause_has_no_mean_pause():
    m = timing([(0, 3000)], [(0, 3000)], ["cat"] * 3)
    assert m["mean_pause"].value is None  # never 0 and never a default number
    assert m["mean_pause"].missing.code == "no_pauses"


def test_no_countable_words_gives_reasons():
    m = timing([(0, 3000)], [(0, 3000)], ["um", "uh"])
    for key in ("speech_rate", "articulation_rate", "mean_length_of_run", "words_per_minute"):
        assert m[key].value is None and m[key].missing.code == "no_words"
    assert m["speaking_time"].value == 3.0


def test_uncounted_words_are_named():
    m = timing([(0, 2000)], [(0, 2000)], ["cat", "2010", "um"])
    assert m["speech_rate"].value == 0.5  # 1 syllable / 2 s; the number and the filler give none
    assert m["speech_rate"].note.endswith(": 1")
    assert m["words_per_minute"].value == 60.0  # 2 words that are not fillers / (2 / 60) min


def test_pause_place_in_text():
    # words start at 0, 0.5 and 2 s; the pause 1.0-1.8 s has its middle at 1.4 s, after two words
    assert place([(1000, 1800)], [0, 500, 2000]) == [2]


def test_latency():
    # question ends at 10 s; answers start at 10.85 s, 9.9 s (an overlap) and 21 s (11 s later)
    m = latency([(10000, 10850), (10000, 9900), (10000, 21000)], "interview")
    assert m["latency_median"].value == 0.9  # only 0.85 s is used; 0.85 rounds half up to 0.9
    assert m["latency_median"].n == 1
    assert m["latency_median"].note.endswith(": 1")  # the overlap start is left out
    assert m["latency_over_10s"].value == 1  # flagged for checking, not averaged


def test_latency_median_and_ten_second_edge():
    m = latency([(0, 600), (0, 1000), (0, 10000), (0, 10001)], "role-swap")
    assert m["latency_median"].value == 1.0  # median of 0.6, 1.0 and 10.0 s
    assert m["latency_over_10s"].value == 1  # 10.0 s is not over 10 s; 10.001 s is


def test_latency_missing():
    solo = latency([(0, 500)], "solo")
    assert solo["latency_median"].value is None
    assert solo["latency_median"].missing.code == "not_valid_in_mode"
    none = latency([(1000, 900)], "interview")
    assert none["latency_median"].missing.code == "no_timed_answers"
    assert none["latency_over_10s"].value == 0


@pytest.mark.parametrize(
    ("pairs", "prepared", "code"),
    [
        ([], 1, "after_preparation"),  # Part 2 alone: its talk follows the preparation minute
        ([], 0, "no_timed_answers"),  # nothing to time, and no preparation minute to explain why
        ([(0, 600)], 1, None),  # a scope with Part 2 and other answers: those others are timed
    ],
)
def test_a_part_2_talk_after_the_preparation_minute_has_no_response_time(
    pairs: list[tuple[int, int]], prepared: int, code: str | None
) -> None:
    m = latency(pairs, "interview", prepared)
    found = m["latency_median"].missing
    assert (found.code if found else None) == code
    if code == "after_preparation":
        assert m["latency_over_10s"].missing.code == code  # not "0 answers after more than 10 s"


def test_rounding():
    assert rounded(3.7363, 1) == 3.7
    assert rounded(0.649, 1) == 0.6
    assert rounded(0.25, 1) == 0.3  # half up, not to the nearest even digit
    with pytest.raises(ValueError):
        rounded(0.12345, 4)  # never 4 or more decimals


def test_value_has_a_number_or_a_reason():
    with pytest.raises(ValueError):
        Value(None, "s")
    with pytest.raises(ValueError):
        Value(0.55, "s", Missing("no_speech", "x"))
