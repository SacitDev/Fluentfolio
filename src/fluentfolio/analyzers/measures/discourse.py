"""Discourse markers in the answers: which ones, how often, and how varied, from the app's own marker list."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from fluentfolio.analyzers.measures.values import (
    NO_MARKERS,
    Pointer,
    Utterance,
    Value,
    absent,
    find,
    measured,
    tokens,
    word_list,
)

MARKERS_FILE = "discourse_markers.txt"


@dataclass(frozen=True)
class Markers:
    counts: dict[str, int]  # marker -> times used, most used first
    variety: Value  # distinct markers / markers used
    found: list[Pointer]


def markers(utterances: Sequence[Utterance]) -> Markers:
    """Find each marker of the list; a longer marker wins over a shorter one inside it ("first of all")."""
    found = []
    for uid, text in utterances:
        words = [w for w, _ in tokens(text)]
        for first, last, marker in find([w.lower() for w in words], word_list(MARKERS_FILE)):
            found.append(Pointer(uid, (first, last), marker))
    counts = dict(Counter(p.text for p in found).most_common())
    total = len(found)
    if total == 0:
        variety = absent(
            NO_MARKERS, "No marker from the list was used, so there is no variety to measure.", "ratio"
        )
    else:
        variety = measured(len(counts) / total, "ratio", places=2, n=total)
    return Markers(counts, variety, found)
