"""The learner's own transcript against Fluentfolio's: the places where a content word differs.

Only words count: fillers, cut-off words, case, punctuation and Fluentfolio's marks are left out, because
they say how a transcript was written, not what was said. A missing "the" alone makes no row; a different
noun or verb does. The two transcripts are lined up over the whole session, the examiner's turns too, so
the learner's answers line up even when the other transcript also holds the questions.

Known limit: words that only the learner's transcript has, at the very edge of an answer, give no row,
because nothing shows who said them; revisit if the learner's transcript is ever matched to speakers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import NamedTuple

from fluentfolio.analyzers.measures.values import FILLER_WORDS, tokens, word_list
from fluentfolio.core.text import clean

MAX_ROWS = 40  # in one for_ai file; the evidence keeps every row
CONTEXT = 2  # words shown on each side of a difference, so it can be found in its utterance
SHOWN = 12  # differing words quoted from each side; a longer run ends in "..."
FUNCTION_FILE = "function_words.txt"
# The marks Fluentfolio writes into its own transcript; "[?word]" keeps its word.
MARKS = re.compile(
    r"\[(?:pause [0-9.]+ s|non-English|overlap|question read aloud|withheld for audio check)\]"
)
# The neutral label a stored line starts with, as names were replaced when the transcript was taken in.
LABEL = re.compile(r"^Speaker [A-Z0-9]+:\s*")


class Word(NamedTuple):
    text: str  # as written, without the punctuation around it
    utterance: str | None  # None for the examiner's turns
    answer: str | None


class Piece(NamedTuple):
    """One stretch of the session in time order: an utterance of a candidate, or an examiner's turn."""

    utterance: str | None
    answer: str | None
    text: str


def words(text: str, utterance: str | None = None, answer: str | None = None) -> list[Word]:
    """The words of a text that take part in the comparison; "well-known" counts as two words."""
    out = []
    for word, cut in tokens(MARKS.sub(" ", text)):
        if cut or word.lower() in FILLER_WORDS:
            continue
        out += [Word(part, utterance, answer) for part in word.split("-") if part]
    return out


def users_words(text: str) -> list[Word]:
    """The learner's transcript through the same cleaning as Fluentfolio's text, line by line."""
    return [w for line in text.splitlines() for w in words(clean(LABEL.sub("", line.strip())))]


def _content(found: Sequence[Word], function: frozenset[str]) -> bool:
    return any((low := w.text.lower()) not in function and low.replace("'", "").isalpha() for w in found)


def _quote(side: Sequence[Word], start: int, end: int, left: int, right: int) -> str:
    """The differing words with `left` and `right` words of context; a long run is shortened."""
    differing = [w.text for w in side[start:end]]
    if len(differing) > SHOWN:
        differing = [*differing[:SHOWN], "..."]
    before = [w.text for w in side[start - left : start]]
    after = [w.text for w in side[end : end + right]]
    return " ".join([*before, *differing, *after])


def _reach(ours: Sequence[Word], start: int, step: int, utterance: str | None) -> int:
    """How many of the context words from `start`, going back (-1) or on (1), are in the utterance."""
    n = 0
    while n < CONTEXT and 0 <= start + n * step < len(ours) and ours[start + n * step].utterance == utterance:
        n += 1
    return n


def _owner(ours: Sequence[Word], i1: int, i2: int) -> Word | None:
    """The candidate word a change belongs to, or None when it lies in the examiner's turns."""
    if i1 < i2:
        return next((w for w in ours[i1:i2] if w.utterance), None)
    # Words only the learner's transcript has: inside an answer when both neighbours are in it.
    if 0 < i1 < len(ours) and ours[i1 - 1].answer and ours[i1 - 1].answer == ours[i1].answer:
        return ours[i1 - 1]
    return None


def compare(session: Sequence[Piece], users_text: str) -> tuple[list[dict[str, str]], float | None]:
    """Difference rows for the candidates' utterances in time order, and the word match in percent.

    The word match is the share of the longer transcript's words that line up with the same word in the
    other: 100 when they agree word for word, lower when either one has words the other lacks.
    """
    ours = [w for piece in session for w in words(piece.text, piece.utterance, piece.answer)]
    users = users_words(users_text)
    longer = max(len(ours), len(users))
    if not longer:
        return [], None
    # Frequent words must stay matchable, so the automatic junk rule for long sequences is off.
    keys = SequenceMatcher(
        None, [w.text.lower() for w in ours], [w.text.lower() for w in users], autojunk=False
    )
    function = frozenset(word_list(FUNCTION_FILE))
    changes: list[list[int]] = []  # [i1, i2, j1, j2] per row; changes close together share one row
    owners: list[Word] = []
    for tag, i1, i2, j1, j2 in keys.get_opcodes():
        if tag == "equal" or not _content([*ours[i1:i2], *users[j1:j2]], function):
            continue
        if (owner := _owner(ours, i1, i2)) is None:
            continue
        if changes and owners[-1].utterance == owner.utterance and i1 - changes[-1][1] <= 2 * CONTEXT:
            changes[-1][1], changes[-1][3] = i2, j2
        else:
            changes.append([i1, i2, j1, j2])
            owners.append(owner)
    rows = []
    for (i1, i2, j1, j2), owner in zip(changes, owners, strict=True):
        # Context stays inside the utterance; the learner's side shows as many words around the change.
        left = _reach(ours, i1 - 1, -1, owner.utterance)
        right = _reach(ours, i2, 1, owner.utterance)
        rows.append({
            "utterance_id": str(owner.utterance),
            "ours": _quote(ours, i1, i2, left, right),
            "users": _quote(users, j1, j2, min(left, j1), right),
        })  # fmt: skip
    matched = sum(block.size for block in keys.get_matching_blocks())
    return rows, round(100 * matched / longer, 1)
