"""The speech-model comparison's normaliser and grammar-edit counter, on hand-made sentences."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# eval/ is not a package; its dataclasses need the module registered before it runs.
_SPEC = importlib.util.spec_from_file_location(
    "asr_metrics", Path(__file__).resolve().parents[2] / "eval" / "asr_metrics.py"
)
assert _SPEC is not None and _SPEC.loader is not None
metrics = sys.modules.setdefault("asr_metrics", importlib.util.module_from_spec(_SPEC))
_SPEC.loader.exec_module(metrics)


@pytest.mark.parametrize(
    ("n", "words"),
    [(0, "zero"), (13, "thirteen"), (40, "forty"), (99, "ninety nine"), (105, "one hundred five"),
     (2023, "two thousand twenty three"), (1_000_000, "one million")],
)  # fmt: skip
def test_numbers_are_spelled_in_words(n: int, words: str) -> None:
    assert metrics.spelled(n) == words


def test_the_normaliser_spells_numbers_keeps_filler_tags_and_drops_other_tags() -> None:
    text = "Well, I think 20% of the 1st-years [laughter] said <noise> um... [UH] it's 1,500."
    assert metrics.normalise(text) == (
        "well i think twenty percent of the first years said um uh it's one thousand five hundred".split()
    )


def test_a_backchannel_is_one_word_and_never_a_filler() -> None:
    words = metrics.normalise("Uh-huh, I agree. Mm-hmm... huh-uh, uh, no.")
    assert words == ["uhhuh", "i", "agree", "mmhmm", "huhuh", "uh", "no"]
    assert [w for w in words if w in metrics.FILLERS] == ["uh"]


@pytest.mark.parametrize(
    ("text", "words"),
    [("21st", "twenty first"), ("100th", "one hundredth"), ("12th", "twelfth"), ("3rd", "third"),
     ("20th", "twentieth"), ("3.5", "three point five")],
)  # fmt: skip
def test_ordinals_and_decimals(text: str, words: str) -> None:
    assert metrics.normalise(text) == words.split()


@pytest.mark.parametrize(
    ("ref", "hyp", "counted"),
    [("go", "goes", True), ("walk", "walked", True), ("use", "using", True), ("make", "making", True),
     ("use", "used", True), ("thing", "things", True), ("is", "are", True), ("a", "the", True),
     ("he", "she", True), ("has", "had", True),
     ("is", "his", False), ("cat", "dog", False), ("go", "went", False), ("he", "he", False),
     ("have", "is", False), ("their", "there", False), ("an", "and", False), ("a", "as", False),
     ("in", "ind", False),
     ("hi", "his", True)],  # a blind spot the rule keeps: the precision check on real output measures these
)  # fmt: skip
def test_a_grammar_edit_is_a_closed_group_swap_or_one_ending(ref: str, hyp: str, counted: bool) -> None:
    assert metrics.grammar_edit(ref, hyp) is counted


def test_only_aligned_substitutions_count_and_the_rate_is_per_100_reference_words() -> None:
    # "go" became "goes" and "the" was left out: one substitution counted, a deletion never.
    ref, hyp = "he go to the school every day".split(), "he goes to school every day".split()
    assert metrics.grammar_edits(metrics.align(ref, hyp)) == [metrics.Pair("go", "goes", 1)]
    # "have" became "has" and "a" became "the": two edits in four words.
    two = ("she have a car".split(), "she has the car".split())
    added = ("i like it".split(), "i like it a lot".split())  # inserted words are not substitutions
    assert metrics.grammar_edit_rate([two]) == 50.0
    assert metrics.grammar_edit_rate([added]) == 0.0
    # 3 edits over 7 + 4 = 11 reference words.
    assert metrics.grammar_edit_rate([(ref, hyp), two]) == pytest.approx(300 / 11)


def test_an_empty_side_aligns_every_word_to_nothing() -> None:
    assert metrics.align(["a"], []) == [metrics.Pair("a", None, 0)]
    assert metrics.align([], ["a"]) == [metrics.Pair(None, "a", None)]


def test_word_errors_count_substitutions_deletions_and_insertions() -> None:
    assert metrics.word_errors("the cat sat on the mat".split(), "the cat sat on mat".split()) == (1, 6)
    assert metrics.word_errors("a b c".split(), "a x c d".split()) == (2, 3)  # one swap, one added
    assert metrics.word_errors([], ["a", "b"]) == (2, 0)
    assert metrics.without_fillers(["um", "i", "uh", "went", "erm"]) == ["i", "went"]


def test_fillers_are_matched_by_place_whatever_their_spelling() -> None:
    # The first "um" is missing; "uh" is found where it was said: 1 found, 0 invented, 1 missed.
    assert metrics.filler_counts("um i think uh it is".split(), "i think um it is".split()) == (1, 0, 1)
    assert metrics.filler_counts(["i", "er", "went"], ["i", "um", "went"]) == (1, 0, 0)
    assert metrics.filler_counts(["i", "the", "went"], ["i", "uh", "went"]) == (0, 1, 0)
    assert metrics.prf(1, 0, 1) == pytest.approx((1.0, 0.5, 2 / 3))
    assert metrics.prf(0, 0, 0) == (None, None, None)
    assert metrics.prf(0, 2, 0) == (0.0, None, 0.0)


def test_cut_off_marks_are_read_before_the_normaliser_removes_them() -> None:
    words, cut = metrics.marked_reference("I wa* went [UH] to the the shop.")
    assert words == ["i", "wa", "went", "uh", "to", "the", "the", "shop"]
    assert cut == [False, True, False, False, False, False, False, False]
    with pytest.raises(ValueError, match="token by token"):
        metrics.marked_reference("a <noise here> b")  # one tag written across a space


def test_a_repair_is_kept_when_the_output_has_both_copies_or_the_broken_word_s_letters() -> None:
    raw = "I wa* went [UH] to the the shop."
    # "want" starts with "wa", and both "the" are there: 2 of 2 kept; the filler is not a repair.
    assert metrics.repairs(raw, "i want went to the the shop".split()) == (2, 2)
    assert metrics.repairs(raw, "i went to the shop".split()) == (0, 2)


def test_word_times_must_lie_in_their_chunk_and_never_go_back() -> None:
    chunks = [[0, 1000], [2000, 3000]]
    rows = [(0, 100, 200, []), (0, 150, 900, []), (0, 120, 300, []), (1, 1900, 2100, []),
            (1, 2000, 3000, ["untimed"]), (1, 2500, 2600, [])]  # fmt: skip
    words = [{"chunk": c, "start_ms": a, "end_ms": b, "flags": f} for c, a, b, f in rows]
    # Out: 120 starts before 150, 1900 lies outside its chunk, and the untimed word has no time of its own.
    assert metrics.valid_times(words, chunks) == (3, 6)


def word(text: str, start: int, end: int, confidence: float | None = None) -> dict[str, object]:
    return {"text": text, "start_ms": start, "end_ms": end, "confidence": confidence, "chunk": 0, "flags": []}


def test_clip_edges_need_the_clip_s_first_and_last_words_recognised() -> None:
    clips = [(1000, 2000, "Hello there."), (3000, 4000, "Good morning"), (5000, 6000, "In 2010")]
    words = [word("Hello", 1010, 1300), word("there", 1400, 1990), word("good", 3050, 3300),
             word("mourning", 3400, 3990), word("in", 5020, 5200), word("2010", 5300, 5980)]  # fmt: skip
    # Clip 2's last word was misheard; "2010" is three reference words and ends the third clip.
    assert metrics.clip_edges(clips, words) == [10, 10, 20, 20]


def test_unsure_words_are_the_substituted_or_inserted_ones_and_the_threshold_fits_them() -> None:
    words = [word("I", 0, 1, 0.9), word("want", 1, 2, 0.4), word("home", 2, 3, 0.8), word("now", 3, 4, 0.3),
             word("ok", 4, 5, None)]  # fmt: skip
    scored = metrics.confidence_targets(["i", "went", "home", "ok"], words)
    assert scored == [(0.9, False), (0.4, True), (0.8, False), (0.3, True)]  # no confidence, not scored
    grid = [round(0.05 * k, 10) for k in range(1, 20)]
    # Every threshold over 0.4 and up to 0.8 finds both wrong words and no right one; the lowest is 0.45.
    best = metrics.best_threshold(scored, grid, 0.2)
    assert (best["threshold"], best["f1"], best["tp"], best["fp"], best["fn"]) == (0.45, 1.0, 2, 0, 0)
    none = metrics.best_threshold([(0.5, False)], grid, 0.2)
    assert none["threshold"] is None and none["f1"] == 0.0 and "under 0.2" in none["reason"]


NYRA = "I think [UH] we should go go home [laughter] now"


@pytest.mark.parametrize(
    ("left", "correct"),
    # Gaps after the output's word `left`: the missing filler, the half-kept "go go" (either side, as the
    # output's "go" may be either copy), the sound tag; nothing between "I" and "think" or "we" and "should".
    [(0, False), (1, True), (2, False), (3, True), (4, True), (5, True)],
)
def test_a_nyra_gap_is_correct_when_the_reference_holds_a_hesitation_the_output_lacks(
    left: int, correct: bool
) -> None:
    words = [word(w, 0, 0) for w in "I think we should go home now".split()]
    assert metrics.nyra_gap_correct(NYRA, words, left) is correct


def test_a_broken_off_word_counts_for_a_gap_only_when_the_output_lacks_it() -> None:
    raw = "we wa* walked home"
    assert metrics.nyra_gap_correct(raw, [word(w, 0, 0) for w in ("we", "walked", "home")], 0) is True
    assert metrics.nyra_gap_correct(raw, [word(w, 0, 0) for w in ("we", "wa", "walked", "home")], 0) is False


def test_a_composed_gap_is_correct_over_an_unwritten_truth_hesitation() -> None:
    hesitations = [(1000, 1400)]
    words = [word("so", 0, 900), word("we", 1500, 2000)]
    assert metrics.composed_gap_correct((900, 1500), hesitations, words) is True
    written = [*words, word("um", 1100, 1300)]
    assert metrics.composed_gap_correct((900, 1500), hesitations, written) is False
    assert metrics.composed_gap_correct((2000, 2500), hesitations, words) is False
