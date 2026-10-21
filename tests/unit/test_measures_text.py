"""Measures read from the transcript, with hand-counted expected values."""

import pytest
import wordfreq

from fluentfolio.analyzers.measures import syllables
from fluentfolio.analyzers.measures.discourse import markers
from fluentfolio.analyzers.measures.disfluency import (
    filled_pauses,
    listed_phrases,
    repetitions,
    self_corrections,
    voiced_gaps,
)
from fluentfolio.analyzers.measures.lexical import mattr, profile
from fluentfolio.analyzers.measures.values import find, tokens

HAND_COUNTS = {
    "cat": 1, "table": 2, "banana": 3, "beautiful": 3, "education": 4, "curiosity": 5, "university": 5,
    "opportunity": 5, "strengths": 1, "I": 1,
    # one word for each spelling rule
    "makes": 1, "places": 2, "tables": 2, "smiles": 1, "walked": 1, "wanted": 2, "studied": 2, "being": 2,
    "someone": 2, "maybe": 2, "movie": 2, "actual": 3, "nation": 2, "didn't": 2, "don't": 1, "centre": 2,
    "idea": 3,
}  # fmt: skip


@pytest.mark.parametrize(("word", "expected"), HAND_COUNTS.items())
def test_syllables_hand_counts(word, expected):
    assert syllables.count(word) == expected


def test_syllable_total_leaves_out_fillers_and_numbers():
    # cat 1 + table 2; "um" is a filler and "2010" has no letters to count
    assert syllables.total(["um", "cat", "2010", "table"]) == (3, 1)
    assert syllables.count("2010") is None


def test_repetition_candidates():
    found = repetitions([("U007", "I I went to the the market")])
    assert [(p.utterance_id, p.at, p.text) for p in found] == [
        ("U007", (0, 1), "I I"),
        ("U007", (4, 5), "the the"),
    ]


@pytest.mark.parametrize(
    ("text", "at", "shown"),
    [
        ("I went I went home", (0, 3), "I went I went"),  # a repeated pair
        ("the um the cat", (0, 2), "the um the"),  # a hesitation sound between does not hide it
        ("the the the cat", (0, 2), "the the the"),  # one candidate, not two
    ],
)
def test_repetition_shapes(text, at, shown):
    found = repetitions([("U1", text)])
    assert [(p.at, p.text) for p in found] == [(at, shown)]


def test_self_correction_candidates():
    found = self_corrections([("U003", "I go- I went")])
    assert [(p.utterance_id, p.at, p.text) for p in found] == [("U003", (1, 3), "go- -> went")]
    assert self_corrections([("U1", "she bou- bought it")])[0].text == "bou- -> bought"
    assert self_corrections([("U1", "and then I wa-")])[0].text == "wa-"  # broken off with no replacement
    assert self_corrections([("U1", "that is right, that is")]) == []  # "that is" is not a repair


@pytest.mark.parametrize(
    ("gap", "speech", "found"),
    # A word ends at 1000 and the next starts at 1000 + gap; speech is heard from 1000 to 1000 + speech.
    [(199, 199, False), (200, 199, False), (200, 200, True), (201, 201, True), (900, 199, False),
     (900, 200, True), (900, 201, True)],
)  # fmt: skip
def test_a_voiced_gap_needs_200_ms_between_words_and_200_ms_of_speech_in_it(gap, speech, found):
    got = voiced_gaps([(500, 1000), (1000 + gap, 1500 + gap)], [(1000, 1000 + speech)], [])
    assert got == ([(1000, 1000 + gap, speech)] if found else [])


def test_a_voiced_gap_adds_up_speech_inside_it_and_keeps_away_from_the_other_speaker():
    words = [(0, 400), (1000, 1300), (1310, 1600), (2600, 3000)]
    # Speech 300-700 overlaps 400-1000 by 300 ms; 1650-1750 and 2500-2700 add 100 + 100 to 1600-2600.
    speech = [(300, 700), (1650, 1750), (2500, 2700)]
    assert voiced_gaps(words, speech, []) == [(400, 1000, 300), (1600, 2600, 200)]
    # Another speaker 299 ms after the second gap ends blocks it; at 300 ms it does not.
    assert voiced_gaps(words, speech, [(2899, 3500)]) == [(400, 1000, 300)]
    assert voiced_gaps(words, speech, [(2900, 3500)]) == [(400, 1000, 300), (1600, 2600, 200)]
    assert voiced_gaps(words, speech, [(0, 101)]) == [(1600, 2600, 200)]  # 299 ms before 400
    assert voiced_gaps([(0, 400)], speech, []) == []  # one word has no gap


def test_like_you_know_i_mean_are_listed_not_counted():
    found = listed_phrases([("U2", "you know, I mean, it was like huge")])
    assert [(p.at, p.text) for p in found] == [((0, 1), "you know"), ((2, 3), "I mean"), ((6, 6), "like")]
    # "like" is never a hesitation sound: 0 per minute
    assert filled_pauses(["I", "like", "football"], 60000, "verbatim-best-effort").value == 0.0


def test_filled_pauses_per_minute():
    words = ["um", "I", "uh", "went", "er"]
    verbatim = filled_pauses(words, 30000, "verbatim-best-effort")
    assert verbatim.value == 6.0  # 3 hesitation sounds / 0.5 min
    assert verbatim.note is None
    smoothed = filled_pauses(words, 30000, "smoothed")
    assert smoothed.value == 6.0
    assert smoothed.note.startswith("lower bound")
    empty = filled_pauses(words, 0, "smoothed")
    assert empty.value is None and empty.missing.code == "no_speech"


def test_tokens_drop_lone_punctuation_and_mark_cut_offs():
    assert tokens("Well , I wa- ...") == [("Well", False), ("I", False), ("wa", True)]


def test_phrase_finder_prefers_the_longer_phrase():
    assert find(["first", "of", "all", "first"], ["first", "first of all"]) == [
        (0, 2, "first of all"),
        (3, 3, "first"),
    ]


def test_discourse_markers():
    found = markers(
        [("U1", "First of all I think it is good. However, actually"), ("U2", "so first I think")]
    )
    # i think 2; first of all, however, actually, so and first 1 each
    assert found.counts == {"i think": 2, "first of all": 1, "however": 1, "actually": 1, "so": 1, "first": 1}
    assert found.variety.value == 0.86  # 6 distinct / 7 used = 0.857
    assert found.variety.n == 7
    assert [p.utterance_id for p in found.found] == ["U1"] * 4 + ["U2"] * 3


def test_no_markers_no_variety():
    found = markers([("U1", "you know I mean it was like huge")])  # listed phrases are not markers
    assert found.counts == {}
    assert found.variety.value is None and found.variety.missing.code == "no_markers"


def test_mattr_window_of_50():
    words = [f"w{i}" for i in range(49)] + ["w0", "w49"]
    # window 1: w0..w48 and w0 again, 49 types / 50 = 0.98; window 2: w1..w48, w0, w49, 50 / 50 = 1.0
    assert mattr(words).value == 0.99
    assert mattr(words).n == 51
    assert mattr(["a"] * 60).value == 0.02  # 1 type in every window of 50


@pytest.mark.parametrize(("n", "value"), [(49, None), (50, 1.0), (51, 1.0)])
def test_mattr_needs_50_words(n, value):
    result = mattr([f"w{i}" for i in range(n)])
    assert result.value == value
    assert result.n == n
    if value is None:
        assert result.missing.code == "too_few_words"


def test_content_word_frequency_shares():
    # The reference lists put "house" and "travel" in the top 2,000, "neighborhood" between 2,000 and
    # 5,000, and "photosynthesis" outside the top 5,000.
    top2k, top5k = set(wordfreq.top_n_list("en", 2000)), set(wordfreq.top_n_list("en", 5000))
    assert {"house", "travel"} <= top2k and "neighborhood" in top5k - top2k and "photosynthesis" not in top5k
    lex = profile([("U1", "I like the house and travel."), ("U2", "Um, the neighborhood, photosynthesis")])
    assert lex.outside_2k.value == 50.0  # 2 of the 4 content words
    assert lex.outside_5k.value == 25.0  # 1 of 4
    assert lex.outside_5k.n == 4
    assert [(p.utterance_id, p.at, p.text) for p in lex.rare] == [("U2", (3, 3), "photosynthesis")]


def test_names_are_left_out_and_a_word_is_common_when_its_base_form_is():
    # wordfreq ranks: cooked 5,950 (cook 2,248), festivals 8,666 (festival 1,940), villages 5,466 (village
    # 1,621); Hilda and Bedford are outside the top 10,000. The ranks are checked here, not assumed.
    top2k, top5k = set(wordfreq.top_n_list("en", 2000)), set(wordfreq.top_n_list("en", 5000))
    assert {"festival", "village"} <= top2k and "cook" in top5k - top2k
    assert not {"cooked", "festivals", "villages", "hilda", "bedford"} & top5k
    said = [("U1", "Hilda cooked at festivals in villages near Bedford")]
    plain = profile(said)  # no parse: every content word is looked up as it stands
    assert (plain.outside_2k.value, plain.outside_5k.value, plain.outside_5k.n) == (100.0, 100.0, 5)
    assert [p.text for p in plain.rare] == ["hilda", "cooked", "festivals", "villages", "bedford"]
    lemmas = {("U1", 1): "cook", ("U1", 3): "festival", ("U1", 5): "village"}
    parsed = profile(said, names={("U1", 0), ("U1", 7)}, lemmas=lemmas)
    assert parsed.outside_2k.value == 33.3  # 1 of 3: only "cook" is outside the top 2,000
    assert (parsed.outside_5k.value, parsed.outside_5k.n, parsed.rare) == (0.0, 3, [])
    assert parsed.mattr.n == plain.mattr.n == 8  # the names are still words for word variety


def test_possessive_is_looked_up_as_its_noun():
    lex = profile([("U1", "my family's house")])
    assert lex.outside_2k.value == 0.0  # "family" and "house" are both common


def test_hesitations_and_cut_offs_are_not_words():
    lex = profile([("U1", "um the gar- garden")])
    assert lex.mattr.n == 2  # "the" and "garden"
    assert lex.outside_2k.n == 1  # "garden"


def test_no_content_words():
    lex = profile([("U1", "I am the one")])
    assert lex.outside_2k.value is None and lex.outside_2k.missing.code == "no_content_words"
    assert lex.rare == []
