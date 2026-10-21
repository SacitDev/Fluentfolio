"""Where the silent pauses fall in the answers' grammar, and the parser's clause counts and examples.

With the parser, a pause lies inside a clause when the words on both sides belong to one clause.
Without it only the transcript's sentence punctuation is known, so the measure becomes "internal
pauses": pauses inside a sentence, which says less and is marked low.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Sequence
from typing import Any

from fluentfolio.analyzers.measures.values import (
    NO_PAUSES,
    Missing,
    Pointer,
    Span,
    Value,
    absent,
    measured,
    rounded,
)

TYPES = ("conditional", "relative clause", "passive", "modal of speculation")
EXAMPLES = 2  # examples listed per structure type
SENTENCE_ENDS = (".", "?", "!")
# A word that can border a pause: its start in ms, its text, and its clause (None without a parse).
Border = tuple[int, str, int | None]
Row = dict[str, Any]


def locate(pauses: Sequence[Span], words: Sequence[Border], parsed: bool) -> list[bool | None]:
    """Per pause of one answer: True inside a clause (a sentence, without a parse), False at a boundary.

    None when a side has no word. A word comes before a pause when it starts before the pause's middle,
    as the transcript's pause marks are placed.
    """
    starts = [w[0] for w in words]
    out: list[bool | None] = []
    for a, b in pauses:
        k = bisect_left(starts, (a + b) / 2)
        if k in (0, len(words)):
            out.append(None)
            continue
        (_, text, left), (_, _, right) = words[k - 1], words[k]
        out.append(left == right if parsed else not text.endswith(SENTENCE_ENDS))
    return out


def share(found: Sequence[bool | None], why: Missing | None) -> dict[str, Value]:
    """The pauses inside a clause as a percent of the placed pauses; internal pauses when `why` is given."""
    key = "internal_pauses" if why else "pauses_in_clause"
    placed = [f for f in found if f is not None]
    if not placed:
        return {key: absent(NO_PAUSES, "There was no pause between two words to place.", "%", n=0)}
    notes = []
    if len(placed) < len(found):
        notes.append(f"{len(placed)} of {len(found)} pauses have a word on both sides; only these count.")
    if why:
        notes.append(
            f"No clause boundaries ({why.text}), so this counts pauses inside a sentence of the transcript's "
            "punctuation."
        )
    return {key: measured(100 * sum(placed) / len(placed), "%", n=len(placed), note=" ".join(notes) or None)}


def clause_counts(answers: Sequence[Row]) -> Row:
    """Clauses per unit (a main clause with the clauses that depend on it) and subordinate clauses."""
    units = sum(a["units"] for a in answers)
    per_unit = rounded(sum(a["clauses"] for a in answers) / units, 1) if units else None
    return {"clauses_per_unit": per_unit, "subordinate": sum(a["subordinate"] for a in answers)}


def examples(found: Sequence[Row]) -> tuple[list[Pointer], dict[str, int]]:
    """The first examples of each structure type in time order, and how many of each type there are."""
    counts: Counter[str] = Counter()
    listed = []
    for f in found:
        if counts[f["type"]] < EXAMPLES:
            listed.append(Pointer(f["utterance_id"], (f["at"][0], f["at"][1]), f"{f['type']}: {f['text']}"))
        counts[f["type"]] += 1
    return listed, {t: counts[t] for t in TYPES}
