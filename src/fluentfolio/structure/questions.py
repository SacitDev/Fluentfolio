"""Sentences and what each one asks, the solo split into questions read aloud and answers, the question list.

The cue lists are the app's own data files. A cue match is a sign to weigh, never a certainty: the structure
step combines it with who speaks, for how long, and what follows.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from fluentfolio.analyzers.measures.values import find, word_list
from fluentfolio.asr.normalize import split

Word = dict[str, Any]
Span = tuple[int, int]  # start and end in ms
ENDS = (".", "?", "!")
LEAD = frozenset(
    {"and", "so", "ok", "okay", "now", "right", "well", "alright", "then"}
)  # skipped before a cue
CARD_LINE = frozenset({"what", "where", "when", "why", "who", "whom", "whose", "which", "how", "explain"})
CARD_GAP_MS = 3000  # the lines of a topic card are read one after another, with short gaps
# The speech model may write no full stop before a question; a silence as long as the longest pause view
# of the fluency measures (1 s) ends the sentence, so the question does not take the answer's last words.
SENTENCE_GAP_MS = 1000
CLARIFY_MAX_WORDS = 8  # "Sorry, could you say that again?" is short; a real question about "sorry" is longer
MATCH_SHARE = 0.6  # a listed question matches text that holds this share of its words, at least two of them
MATCH_MIN_WORDS = 2
HEADING = re.compile(r"^\W*part\s*(1|2|3|one|two|three)\W*$", re.IGNORECASE)
PART_NUMBERS = {"1": 1, "2": 2, "3": 3, "one": 1, "two": 2, "three": 3}
LIST_MARK = re.compile(r"^\s*(?:\d+\s*[.)]|[-*+])\s*")


def lower(words: list[Word], ids: list[int]) -> list[str]:
    return [str(words[i]["text"]).lower() for i in ids]


def _last_mark(raw: str) -> str:
    return raw.rstrip("\"')]")[-1:]


def long_silences(speech: Sequence[Sequence[int]]) -> list[Span]:
    """The gaps of at least SENTENCE_GAP_MS between detected speech segments, which are in time order."""
    return [(a[1], b[0]) for a, b in pairwise(speech) if b[0] - a[1] >= SENTENCE_GAP_MS]


def _after_silences(words: list[Word], ids: list[int], silences: Sequence[Span]) -> set[int]:
    """For each long silence, the word after the one word gap it overlaps most.

    The speech model's word times often run into a silence (960 ms between words around 1.4 s of quiet), so
    the detected speech marks the silence too, as for the pause measures. A short word can lie inside a
    silence the detector heard as quiet; the silence then touches the gaps on both sides of it, and only
    the larger overlap is the sentence end.
    """
    found = set()
    for s, e in silences:
        touched = [
            (min(words[b]["start_ms"], e) - max(words[a]["end_ms"], s), b)
            for a, b in pairwise(ids)
            if s <= words[b]["start_ms"] and e >= words[a]["end_ms"]
        ]
        if touched:
            found.add(max(touched)[1])
    return found


def sentences(words: list[Word], ids: list[int], silences: Sequence[Span] = ()) -> list[list[int]]:
    """The word indexes `ids` cut into sentences after each . ? or ! and at each long silence.

    `silences` holds the long gaps of the detected speech (long_silences); without them only the word
    times show a silence.
    """
    starts = _after_silences(words, ids, silences)
    out: list[list[int]] = [[]]
    for i in ids:
        gap = words[i]["start_ms"] - words[out[-1][-1]]["end_ms"] if out[-1] else 0
        if out[-1] and (gap >= SENTENCE_GAP_MS or i in starts):
            out.append([])
        out[-1].append(i)
        if _last_mark(words[i]["raw"]) in ENDS:
            out.append([])
    return [s for s in out if s]


def _opens(low: list[str], name: str) -> bool:
    k = 0
    while k < len(low) - 1 and low[k] in LEAD:
        k += 1
    return any(low[k : k + len(p.split())] == p.split() for p in word_list(name))


def kind(words: list[Word], sentence: list[int], after_pause: bool = False) -> str:
    """ "question" (a question mark or a question opener), "prompt" (a request to speak) or "statement".

    After a pause that ended a sentence with no . ? or !, an opener alone is not enough: a learner who
    searches for a word ("how do you say ...", "what is it called ...") starts the same way.
    """
    low = lower(words, sentence)
    opener = not after_pause and _opens(low, "question_openers.txt")
    if _last_mark(words[sentence[-1]]["raw"]) == "?" or opener:
        return "question"
    return "prompt" if _opens(low, "prompt_cues.txt") else "statement"


def kinds(words: list[Word], found: list[list[int]]) -> list[str]:
    """The kind of each sentence of `found`, in order."""
    return [
        kind(words, s, k > 0 and _last_mark(words[found[k - 1][-1]]["raw"]) not in ENDS)
        for k, s in enumerate(found)
    ]


def holds(words: list[Word], ids: list[int], name: str) -> bool:
    return bool(find(lower(words, ids), word_list(name)))


def part_cue(words: list[Word], ids: list[int]) -> int | None:
    """The Part a turn announces ("Now, in part three..."), if any; the first cue counts."""
    cues = {" ".join(line.split()[1:]): int(line.split()[0]) for line in word_list("part_cues.txt")}
    found = find(lower(words, ids), cues)
    return cues[found[0][2]] if found else None


@dataclass(frozen=True)
class Asking:
    likeness: float  # 1 when the turn ends with a question or prompt, 0.5 when one comes earlier, else 0
    kind: str  # as an examiner turn: question, prompt, cue_card or other
    cue_card: bool


def asking(words: list[Word], ids: list[int]) -> Asking:
    """How much one turn asks for an answer; a short request to repeat ("Sorry?") never counts as asking."""
    found = kinds(words, sentences(words, ids))
    asks = [k != "statement" for k in found]
    card = holds(words, ids, "cue_card_words.txt")
    likeness = 1.0 if asks[-1] else 0.5 if any(asks) else 0.0
    if len(ids) <= CLARIFY_MAX_WORDS and holds(words, ids, "clarification_cues.txt"):
        likeness = 0.0
    if card and any(asks):
        turn_kind = "cue_card"
    else:
        turn_kind = "question" if "question" in found else "prompt" if "prompt" in found else "other"
    return Asking(likeness, turn_kind, card)


def ask_from(words: list[Word], ids: list[int], silences: Sequence[Span] = ()) -> int:
    """The first word of the questions or prompts that close a turn; its first word when none close it."""
    found = sentences(words, ids, silences)
    found_kinds = kinds(words, found)
    k = len(found)
    while k and found_kinds[k - 1] != "statement":
        k -= 1
    return found[k][0] if k < len(found) else ids[0]


def solo_split(
    words: list[Word], listed: list[Listed] | None = None, silences: Sequence[Span] = ()
) -> list[tuple[str, list[int]]]:
    """Solo: the words cut into ("read_aloud", ids) and ("answer", ids) blocks, in time order.

    With a question list, a sentence that matches a listed question is read aloud; without one, a question
    or prompt sentence is. The lines of a topic card that follow without a long gap belong to the card.
    """
    blocks: list[tuple[str, list[int]]] = []
    card = False
    found = sentences(words, list(range(len(words))), silences)
    for s, s_kind in zip(found, kinds(words, found), strict=True):
        low = lower(words, s)
        if listed is not None:
            asked = match(listed, low) is not None
        else:
            asked = s_kind != "statement"
        prev = blocks[-1] if blocks else None
        if prev and prev[0] == "read_aloud" and card and not asked:
            gap = words[s[0]]["start_ms"] - words[prev[1][-1]]["end_ms"]
            k = 1 if low[0] == "and" and len(low) > 1 else 0
            asked = gap < CARD_GAP_MS and low[k] in CARD_LINE
        role = "read_aloud" if asked else "answer"
        if prev and prev[0] == role:
            prev[1].extend(s)
        else:
            blocks.append((role, list(s)))
            card = False
        card = card or (role == "read_aloud" and holds(words, s, "cue_card_words.txt"))
    return blocks


@dataclass(frozen=True)
class Listed:
    text: str
    part: int | None  # from the "Part 1/2/3" heading above it, if any


def parse_list(text: str) -> list[Listed]:
    """The questions of a plain-text list, one per line, with the Part of the heading above each."""
    out = []
    part = None
    for line in text.splitlines():
        if heading := HEADING.match(line.strip()):
            part = PART_NUMBERS[heading.group(1).lower()]
            continue
        question = LIST_MARK.sub("", line).strip()
        if any(c.isalpha() for c in question):
            out.append(Listed(question, part))
    return out


def _bag(text: str) -> set[str]:
    return {w.lower() for w in (split(raw)[0] for raw in text.split()) if w}


def match(listed: list[Listed], low: list[str]) -> int | None:
    """The listed question whose words the text holds best (a share of 0.6 or more), by its index."""
    have = set(low)
    best, score = None, 0.0
    for k, q in enumerate(listed):
        want = _bag(q.text)
        hit = len(want & have)
        share = hit / len(want) if want else 0.0
        if hit >= MATCH_MIN_WORDS and share >= MATCH_SHARE and share > score:
            best, score = k, share
    return best
