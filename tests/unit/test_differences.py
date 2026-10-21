"""The differences between the learner's transcript and Fluentfolio's, against answers worked out by hand."""

from __future__ import annotations

from fluentfolio.export.differences import SHOWN, Piece, compare


def one(text: str) -> list[Piece]:
    return [Piece("U001", "A001", text)]


def test_a_changed_content_word_gives_one_row_with_two_words_of_context():
    rows, match = compare(one("He go to school every day."), "Speaker A: He goes to school every day.\n")
    # 6 words each side, 5 line up; nothing before "He", so only one word of context on the left
    assert rows == [{"utterance_id": "U001", "ours": "He go to school", "users": "He goes to school"}]
    assert match == 83.3


def test_a_function_word_alone_makes_no_row_but_lowers_the_match():
    rows, match = compare(one("I like the cats"), "I like cats")
    assert (rows, match) == ([], 75.0)  # 3 of the 4 words of the longer one line up


def test_fillers_case_punctuation_marks_and_cut_off_words_are_left_out():
    text = "Um, I REALLY [pause 0.6 s] like wor- working on [?pizza], er, a well-known [non-English] place."
    assert compare(one(text), "i really like working on pizza a well known place") == ([], 100.0)


def test_hidden_characters_in_the_learners_transcript_are_cleaned_like_ours():
    users = "Speaker A: He goes to sch\u200bool\u202e every day\U000e0041\n"
    rows, match = compare(one("He go to school every day"), users)
    assert [r["users"] for r in rows] == ["He goes to school"] and match == 83.3


SESSION = [
    Piece(None, None, "Do you like music?"),
    Piece("U001", "A001", "Yes I like jazz"),
    Piece("U002", "A001", "and rock music"),
    Piece(None, None, "Why?"),
]


def test_words_only_the_learner_has_inside_an_answer_go_to_the_utterance_before_them():
    users = (
        "Speaker A: Do you like music?\nSpeaker B: Yes I like jazz and classical rock music\nSpeaker A: Why?"
    )
    rows, match = compare(SESSION, users)
    # Context stays inside U002: one word before ("and"), two after
    assert rows == [{"utterance_id": "U002", "ours": "and rock music", "users": "and classical rock music"}]
    assert match == 92.3  # 12 of 13


def test_words_only_the_learner_has_at_the_edge_of_an_answer_or_in_a_question_give_no_row():
    edge = "Do you like music Yes I like jazz and rock music thank goodness Why"
    assert compare(SESSION, edge)[0] == []
    question = "Do you love music Yes I like jazz and rock music Why"
    assert compare(SESSION, question)[0] == []


def test_changes_up_to_four_words_apart_share_a_row_and_five_apart_do_not():
    rows, _ = compare(one("I go to the big market on Sunday"), "I went to the big markets on Sunday")
    assert rows == [{
        "utterance_id": "U001", "ours": "I go to the big market on Sunday",
        "users": "I went to the big markets on Sunday",
    }]  # fmt: skip
    rows, _ = compare(one("I go to the big old market on Sunday"), "I went to the big old markets on Sunday")
    assert [r["ours"] for r in rows] == ["I go to the big old market on Sunday"]  # exactly four apart
    rows, _ = compare(one("I go home and then later I eat lunch"), "I went home and then later I ate lunch")
    assert [(r["ours"], r["users"]) for r in rows] == [
        ("I go home and", "I went home and"),
        ("later I eat lunch", "later I ate lunch"),
    ]


def test_a_long_run_is_quoted_up_to_the_limit():
    extra = [f"word{k}x" for k in range(SHOWN + 3)]  # letters and digits: not content words
    long_run = ["house", *(f"place{chr(97 + k)}" for k in range(SHOWN + 2))]
    rows, _ = compare(
        one("we sat in the garden all day"), f"we sat in the {' '.join(long_run)} garden all day"
    )
    assert len(rows) == 1
    shown = rows[0]["users"].split()
    assert shown == ["in", "the", *long_run[:SHOWN], "...", "garden", "all"]
    assert compare(one("we sat down"), f"we sat {' '.join(extra)} down")[0] == []  # no content word differs


def test_frequent_words_still_line_up_in_a_long_session():
    # 244 words, "and" and "the" 81 times each: past the 200 words where difflib's automatic junk rule
    # would stop frequent words from lining up between two changes.
    start = " ".join(f"item{k} and the" for k in range(80))
    _, match = compare(one(f"{start} red and the blue"), f"{start} green and the yellow")
    assert match == 99.2  # 242 of 244; with the junk rule, "and the" would not line up: 240 of 244


def test_nothing_to_compare_gives_no_match_and_an_empty_transcript_gives_zero():
    assert compare([], "") == ([], None)
    rows, match = compare(one("cats sleep"), "")
    assert match == 0.0 and rows == [{"utterance_id": "U001", "ours": "cats sleep", "users": ""}]
