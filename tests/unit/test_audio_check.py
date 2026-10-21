"""The audio check's span: where it may lie, how many words it needs, and where it is in the audio file.

Times are written out by hand, so each expected span and audio position can be read off the test.
"""

from __future__ import annotations

from typing import Any

import pytest

from fluentfolio.export.audio_check import (
    WITHHELD,
    Span,
    answer_start,
    audio_ms,
    choose,
    hidden_text,
    utterances_of,
    withhold_items,
    withhold_rows,
)

Row = dict[str, Any]
# The requirement's values, written here rather than read from the module, so a changed limit fails a test.
MIN_WORDS = 12  # a span holds 12 words or more
SPAN_MS = 8_000  # a span lasts 8.0 s
WHOLE = [{"orig_ms": 0, "audio_ms": 0, "ms": 600_000}]  # one piece: the audio file is the recording


def word(start: int, end: int, raw: str = "word", filler: bool = False) -> Row:
    return {"start_ms": start, "end_ms": end, "raw": raw, "is_filler": filler}


def packed(n: int, first: int = 1000, step: int = 500) -> list[Row]:
    """n words of 400 ms, one every `step` ms from `first`, then one that runs over the end of a span
    started at `first`, so exactly n words lie inside that span."""
    words = [word(first + k * step, first + k * step + 400, f"w{k}") for k in range(n)]
    return [*words, word(first + SPAN_MS - 200, first + SPAN_MS + 300, "over")]


def answer(aid: str, part: int | str, words: list[Row], first: int = 0) -> tuple[Row, Row]:
    a = {"id": aid, "part": part, "start_ms": words[0]["start_ms"], "end_ms": words[-1]["end_ms"]}
    u = {"id": f"U{aid}", "answer_id": aid, "first_word": first, "last_word": first + len(words) - 1}
    return a, u


@pytest.mark.parametrize(("n", "found"), [(MIN_WORDS - 1, False), (MIN_WORDS, True), (MIN_WORDS + 1, True)])
def test_a_span_needs_twelve_words_inside_it(n: int, found: bool) -> None:
    words = packed(n, step=7000 // n)
    a, u = answer("A01", 1, words)
    span = choose([a], [u], words, set(), WHOLE)
    assert (span is not None) is found
    if span:
        assert (span.orig_start_ms, span.orig_end_ms) == (1000, 9000)
        assert span.hidden == tuple(range(n))  # the word that runs over the end stays shown
        assert hidden_text(span, words) == " ".join(f"w{k}" for k in range(n))


def test_hesitation_sounds_are_hidden_but_not_counted() -> None:
    words = packed(MIN_WORDS, step=7000 // MIN_WORDS)
    words[3] = {**words[3], "raw": "uh", "is_filler": True}
    a, u = answer("A01", 1, words)
    assert choose([a], [u], words, set(), WHOLE) is None
    longer = packed(MIN_WORDS + 1, step=7000 // (MIN_WORDS + 1))
    longer[3] = {**longer[3], "raw": "uh", "is_filler": True}
    a, u = answer("A01", 1, longer)
    span = choose([a], [u], longer, set(), WHOLE)
    assert span is not None and 3 in span.hidden


def test_a_span_lies_inside_one_answer() -> None:
    words = packed(MIN_WORDS + 4, step=400)  # 16 words in 6.4 s, then the word over the end
    a, u = answer("A01", 1, words)
    short = {**a, "end_ms": 1000 + SPAN_MS - 1}
    assert choose([short], [u], words, set(), WHOLE) is None
    assert choose([{**short, "end_ms": 1000 + SPAN_MS}], [u], words, set(), WHOLE) is not None


@pytest.mark.parametrize("flagged", [0, 5, MIN_WORDS])  # the first word, one inside, the one over the end
def test_no_unsure_overlapped_or_other_language_word_touches_the_span(flagged: int) -> None:
    words = packed(MIN_WORDS, step=7000 // MIN_WORDS)
    a, u = answer("A01", 1, words)
    assert choose([a], [u], words, {flagged}, WHOLE) is None
    assert choose([a], [u], words, {MIN_WORDS + 1}, WHOLE) is not None  # a word of no answer


def test_part_1_comes_first_then_part_3_then_part_2() -> None:
    blocks = [packed(MIN_WORDS, first=t, step=500) for t in (1000, 20_000, 40_000)]
    words = [w for b in blocks for w in b]
    n = len(blocks[0])
    parts = [2, 3, 1]  # in time order: a Part 2 answer, a Part 3 answer, then a Part 1 answer
    made = [answer(f"A0{k}", p, b, k * n) for k, (p, b) in enumerate(zip(parts, blocks, strict=True))]
    answers, utterances = [m[0] for m in made], [m[1] for m in made]
    span = choose(answers, utterances, words, set(), WHOLE)
    assert span is not None and span.orig_start_ms == 40_000
    span = choose(answers[:2], utterances[:2], words, set(), WHOLE)
    assert span is not None and span.orig_start_ms == 20_000


def test_the_span_is_found_in_the_audio_file_across_its_pieces() -> None:
    words = packed(MIN_WORDS, step=500)  # words from 1000 to 6900 ms, the one over the end at 8800-9300
    a, u = answer("A01", 1, words)
    # Two pieces with 300 ms of silence between them in the file; the recording's pause 5000-5500 is cut.
    pieces = [{"orig_ms": 500, "audio_ms": 0, "ms": 4500}, {"orig_ms": 5500, "audio_ms": 4800, "ms": 4500}]
    moved = {"start_ms": 5500, "end_ms": 5900}
    words = [w if not 5000 <= w["start_ms"] < 5500 else {**w, **moved} for w in words]
    span = choose([a], [u], words, set(), pieces)
    assert span == Span(1000, 9000, 500, 4800 + 3500, span.hidden)


def test_a_word_cut_out_of_the_audio_file_rules_the_span_out() -> None:
    words = packed(MIN_WORDS, step=500)
    a, u = answer("A01", 1, words)
    gap = [{"orig_ms": 0, "audio_ms": 0, "ms": 3000}, {"orig_ms": 3500, "audio_ms": 3300, "ms": 10_000}]
    # The word at 3000-3400 lies wholly in the cut stretch, so the file does not hold it.
    assert choose([a], [u], words, set(), gap) is None
    assert choose([a], [u], words, set(), []) is None  # no audio file at all


@pytest.mark.parametrize(
    ("start", "end", "found"),
    [
        (2999, 3400, True), (3000, 3400, False),  # 1 ms into the first piece, then just touching its end
        (3100, 3501, True), (3100, 3500, False),  # 1 ms into the second piece, then just touching its start
    ],
)  # fmt: skip
def test_a_word_that_reaches_into_a_piece_is_in_the_audio_file(start: int, end: int, found: bool) -> None:
    # Word times from a speech model often run into the pause next to a stretch of speech, so a word whose
    # middle lies in a cut pause can still be wholly in the file.
    words = packed(MIN_WORDS, step=500)
    words[4] = {**words[4], "start_ms": start, "end_ms": end}  # the word at 3000-3400
    a, u = answer("A01", 1, words)
    gap = [{"orig_ms": 0, "audio_ms": 0, "ms": 3000}, {"orig_ms": 3500, "audio_ms": 3300, "ms": 10_000}]
    span = choose([a], [u], words, set(), gap)
    assert (span is not None) is found
    if span:
        assert (span.audio_start_ms, span.audio_end_ms) == (1000, 9000 - 200)


@pytest.mark.parametrize(
    ("ms", "start", "expected"),
    [
        (1000, True, 1000), (1000, False, 1000),  # inside the first piece
        (3200, True, 3300), (3200, False, 3000),  # in the cut stretch: the next start, or the last end
        (3500, True, 3300), (13_500, False, 13_300),  # at a piece's start; at the last piece's end
        (14_000, True, None), (-1, False, None),  # after the file; before it
    ],
)  # fmt: skip
def test_times_of_the_recording_in_the_audio_file(ms: int, start: bool, expected: int | None) -> None:
    pieces = [{"orig_ms": 0, "audio_ms": 0, "ms": 3000}, {"orig_ms": 3500, "audio_ms": 3300, "ms": 10_000}]
    assert audio_ms(pieces, ms, start) == expected


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (1000, 2000, 1000),  # inside the first piece
        (2999, 3200, 2999), (3000, 3500, None),  # its last ms in the first piece; wholly in the cut stretch
        (3200, 3501, 3300),  # begins in the cut stretch and reaches 1 ms into the next piece: its start
        (13_499, 14_000, 13_299), (13_500, 14_000, None),  # the last ms of the file; after it
    ],
)  # fmt: skip
def test_where_an_answer_begins_in_the_audio_file(start: int, end: int, expected: int | None) -> None:
    pieces = [{"orig_ms": 0, "audio_ms": 0, "ms": 3000}, {"orig_ms": 3500, "audio_ms": 3300, "ms": 10_000}]
    assert answer_start(pieces, start, end) == expected


def test_quotes_and_rows_of_the_span_are_withheld_and_others_kept() -> None:
    span = Span(0, SPAN_MS, 0, SPAN_MS, (4, 5, 6))
    utterances = [
        {"id": "U001", "first_word": 0, "last_word": 4},
        {"id": "U002", "first_word": 5, "last_word": 9},
        {"id": "U003", "first_word": 10, "last_word": 12},
    ]
    within = utterances_of(span, utterances)
    assert within == {"U001", "U002"}
    items = [
        {"utterance_id": "U001", "first": 4, "last": 4, "text": "Museum"},  # a hidden word, other case
        {"utterance_id": "U002", "first": 0, "last": 1, "text": "uh uh"},  # no hidden word in it
        {"utterance_id": "U003", "first": 0, "last": 0, "text": "museum"},  # outside the span's utterances
    ]
    shown = withhold_items(items, within, {"museum", "ancient"})
    assert [i["text"] for i in shown] == [WITHHELD, "uh uh", "museum"]
    rows = [{"utterance_id": u, "ours": "a b", "users": "a c"} for u in ("U002", "U003")]
    assert withhold_rows(rows, within) == [
        {"utterance_id": "U002", "ours": WITHHELD, "users": WITHHELD}, rows[1],
    ]  # fmt: skip
