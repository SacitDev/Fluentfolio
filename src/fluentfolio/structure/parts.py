"""Parts 1, 2 and 3 of one candidate's test: from the question list, else the long turn, then text cues.

The long turn (Part 2) is the anchor: the longest answer of 45 s or more, after topic-card words or 15 s of
near-silence (the minute to prepare). The card may come a few short exchanges earlier (a question about the
topic, "can I start?"); those exchanges are Part 2 too. Without an anchor every Part is "unknown"; nothing is
guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

LONG_TURN_MS = 45000
PREP_MS = 15000
FOLLOW_UP_MS = 15000  # a short answer after the long turn stays in Part 2 ...
FOLLOW_UP_WITHIN_MS = 45000  # ... when it starts this soon after the long turn ends
FOLLOW_UPS = 2


@dataclass(frozen=True)
class Step:
    """One answer and what was said to the candidate just before it."""

    start_ms: int  # the first question or prompt before the answer, else the answer's start
    answer_start_ms: int
    end_ms: int
    cue_card: bool  # the turns before it hold topic-card words
    part_cue: int | None  # the Part they announce, if any
    listed_part: int | None  # the Part heading of the listed question it answers, if any
    silence_ms: int  # the longest stretch without words since the candidate's previous answer

    @property
    def answer_ms(self) -> int:
        return self.end_ms - self.answer_start_ms


@dataclass(frozen=True)
class Parts:
    labels: list[int | str]  # per step: 1, 2, 3 or "unknown"
    source: str  # auto or question_list
    confidence: str
    anchor: int | None  # the step of the long turn
    follow_ups: int  # short answers after the long turn kept in Part 2


def detect(steps: list[Step]) -> Parts:
    """The Part of each step: a listed question's heading first, then the timing; never 3 -> 2 or 2 -> 1."""
    found = _timing(steps)
    listed = [s.listed_part for s in steps]
    if all(p is None for p in listed):
        return found
    labels: list[int | str] = []
    current = next(p for p in listed if p is not None)
    for p, auto in zip(listed, found.labels, strict=True):
        current = max(current, p if p is not None else current if auto == "unknown" else int(auto))
        labels.append(current)
    anchor = found.anchor if found.anchor is not None and labels[found.anchor] == 2 else None  # noqa: PLR2004
    return Parts(labels, "question_list", "high", anchor, found.follow_ups if anchor is not None else 0)


def _timing(steps: list[Step]) -> Parts:
    """Parts from the long turn; a cue moves a boundary by one exchange at most and never breaks the order."""
    unknown: list[int | str] = ["unknown"] * len(steps)
    long = [k for k, s in enumerate(steps) if s.answer_ms >= LONG_TURN_MS]
    if not long:
        return Parts(unknown, "auto", "low", None, 0)
    k = max(long, key=lambda j: steps[j].answer_ms)
    card = _card(steps, k)
    cues = int(card is not None) + int(steps[k].silence_ms >= PREP_MS)
    if cues == 0:
        return Parts(unknown, "auto", "low", None, 0)
    first = k if card is None else card  # the topic card opens Part 2
    labels: list[int | str] = [1 if j < first else 2 if j <= k else 3 for j in range(len(steps))]
    follow = 0
    for j in range(k + 1, min(k + 1 + FOLLOW_UPS, len(steps))):
        s = steps[j]
        late = s.answer_start_ms - steps[k].end_ms > FOLLOW_UP_WITHIN_MS
        if s.part_cue == 3 or s.answer_ms >= FOLLOW_UP_MS or late:  # noqa: PLR2004 - a Part 3 cue
            break
        labels[j] = 2
        follow += 1
    if k > 0 and steps[k - 1].part_cue == 2:  # noqa: PLR2004 - the topic was given one exchange earlier
        labels[k - 1] = 2
    return Parts(labels, "auto", "high" if cues == 2 else "medium", k, follow)  # noqa: PLR2004 - both cues


def _card(steps: list[Step], k: int) -> int | None:
    """The step the topic card came before: the long turn, or an earlier one with only short answers since."""
    for j in range(k, -1, -1):
        if j < k and steps[j].answer_ms >= FOLLOW_UP_MS:
            return None  # a full answer since the card: the card was for something else
        if steps[j].cue_card:
            return j
    return None


def spans(steps: list[Step], labels: list[int | str]) -> list[tuple[int, int, int]]:
    """(Part, start, end) of each known Part: from its first question to its last answer's end."""
    out: list[tuple[int, int, int]] = []
    for s, label in zip(steps, labels, strict=True):
        if label == "unknown":
            continue
        if out and out[-1][0] == label:
            out[-1] = (out[-1][0], out[-1][1], s.end_ms)
        else:
            out.append((int(label), s.start_ms, s.end_ms))
    return out
