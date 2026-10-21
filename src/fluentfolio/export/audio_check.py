"""The audio check: one 8-second span of a candidate's answer whose words no exported file shows.

A chat that says it hears the audio writes what the candidate says in that span; the app then compares that
line with the hidden words, which stay in the job folder. A match shows the audio arrived and was decoded,
not that the chat heard it natively.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from fluentfolio.core.text import clean

Row = dict[str, Any]
SPAN_MS = 8_000
MIN_WORDS = 12  # words in the span, hesitation sounds not counted
WITHHELD = "[withheld for audio check]"
FILE = "stages/audio_check.json"  # the hidden words, kept in the job folder only
# Part 1 first: its answers are short and many, so hiding a few of their words costs the grading least.
PART_ORDER = {1: 0, 3: 1, 2: 2}


@dataclass(frozen=True)
class Span:
    orig_start_ms: int
    orig_end_ms: int
    audio_start_ms: int
    audio_end_ms: int
    hidden: tuple[int, ...]  # the speech model's words inside the span, hesitation sounds included


def _piece(pieces: Sequence[Row], ms: int) -> Row | None:
    return next((p for p in pieces if p["orig_ms"] <= ms < p["orig_ms"] + p["ms"]), None)


def audio_ms(pieces: Sequence[Row], ms: int, start: bool) -> int | None:
    """Where a time of the recording falls in the audio file. In a gap between pieces: the next piece's
    start for the start of a span (`start`), the last piece's end for its end; None if there is none."""
    if (p := _piece(pieces, ms)) is not None:
        shift: int = p["audio_ms"] - p["orig_ms"]
        return ms + shift
    if start:
        return next((p["audio_ms"] for p in pieces if p["orig_ms"] > ms), None)
    return max((p["audio_ms"] + p["ms"] for p in pieces if p["orig_ms"] + p["ms"] <= ms), default=None)


def answer_start(pieces: Sequence[Row], start_ms: int, end_ms: int) -> int | None:
    """Where an answer begins in the audio file: in the first piece that overlaps it; None if none does."""
    p = next((p for p in pieces if p["orig_ms"] < end_ms and start_ms < p["orig_ms"] + p["ms"]), None)
    return None if p is None else p["audio_ms"] + max(0, start_ms - p["orig_ms"])


def _heard(pieces: Sequence[Row], word: Row) -> bool:
    """The word is in the audio file: it overlaps one of the pieces the file was made of.

    Not its middle: the speech model's word times can run some tenths of a second into the silence around
    a stretch of speech, while the pieces follow the voice detector, which keeps to the speech itself.
    """
    return any(p["orig_ms"] < word["end_ms"] and word["start_ms"] < p["orig_ms"] + p["ms"] for p in pieces)


def choose(
    answers: Sequence[Row], utterances: Sequence[Row], words: Sequence[Row], excluded: Collection[int],
    pieces: Sequence[Row],
) -> Span | None:  # fmt: skip
    """The first span, Part 1 answers first, that starts at a word, lies inside one answer and in the audio
    file, holds 12 words or more, and touches no word in `excluded` (unsure, overlapped or not English)."""
    order = sorted(answers, key=lambda a: (PART_ORDER.get(a["part"], len(PART_ORDER)), a["start_ms"]))
    for a in order:
        mine = [u for u in utterances if u["answer_id"] == a["id"]]
        ids = [i for u in mine for i in range(u["first_word"], u["last_word"] + 1)]
        for k, i in enumerate(ids):
            start = words[i]["start_ms"]
            end = start + SPAN_MS
            if end > a["end_ms"]:
                break
            touched = [j for j in ids[k:] if words[j]["start_ms"] < end]
            inside = [j for j in touched if words[j]["end_ms"] <= end]
            counted = sum(not words[j]["is_filler"] for j in inside)
            if counted < MIN_WORDS or any(j in excluded for j in touched):
                continue
            first, last = audio_ms(pieces, start, True), audio_ms(pieces, end, False)
            if first is None or last is None or not all(_heard(pieces, words[j]) for j in touched):
                continue
            return Span(start, end, first, last, tuple(inside))
    return None


def tokens(text: str) -> set[str]:
    """Lower-case words, so a quote and a hidden word compare without case or punctuation."""
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def hidden_text(span: Span, words: Sequence[Row]) -> str:
    """The hidden words as the speech model wrote them, cleaned as the texts are, for the reveal and the
    answer check."""
    return clean(" ".join(str(words[i]["raw"]) for i in span.hidden))


def utterances_of(span: Span, utterances: Sequence[Row]) -> set[str]:
    """The utterances that hold a hidden word."""
    hidden = set(span.hidden)
    return {u["id"] for u in utterances if hidden & set(range(u["first_word"], u["last_word"] + 1))}


def withhold_items(items: Sequence[Row], within: Collection[str], hidden: Collection[str]) -> list[Row]:
    """Pointer items: one in a span's utterance that quotes a hidden word shows the mark instead."""
    return [
        {**item, "text": WITHHELD}
        if item["utterance_id"] in within and tokens(item["text"]) & set(hidden) else item
        for item in items
    ]  # fmt: skip


def withhold_rows(rows: Sequence[Row], within: Collection[str]) -> list[Row]:
    """Difference rows: both sides of a row in a span's utterance; the learner's side holds the words too."""
    return [{**r, "ours": WITHHELD, "users": WITHHELD} if r["utterance_id"] in within else r for r in rows]
