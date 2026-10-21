"""Fluency timing of a set of candidate answers: pauses from the VAD, rates from transcript syllables.

The caller passes the answers of one Part, or of the whole test, with read-aloud questions and the
Part 2 preparation time already left out. Nothing here is "better" when higher or lower: a rate or a
pause count describes how the answers were spoken, not how good they were.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from itertools import pairwise
from statistics import median

from fluentfolio.analyzers.measures import syllables
from fluentfolio.analyzers.measures.values import (
    AFTER_PREPARATION,
    FILLER_WORDS,
    NO_ANSWERS,
    NO_PAUSES,
    NO_SPEECH,
    NO_WORDS,
    NOT_IN_MODE,
    Span,
    Value,
    absent,
    counted,
    measured,
)
from fluentfolio.core.enums import Mode

PAUSE_MS = 250  # a silence of 250 ms or more inside an answer is a pause
VIEWS_MS = (500, 1000)
LATENCY_CHECK_MS = 10000  # a longer wait before an answer is flagged for checking, not averaged
LATENCY_MODES = (Mode.INTERVIEW, Mode.ROLE_SWAP)  # solo has no examiner turn to measure from
UNITS = {
    "speaking_time": "s",
    "speech_rate": "syll/s",
    "articulation_rate": "syll/s",
    "pauses_per_min": "per min",
    "mean_pause": "s",
    "pauses_500ms": "count",
    "pauses_1s": "count",
    "mean_length_of_run": "syll",
    "words_per_minute": "words/min",
}


def runs(turns: Sequence[Span], speech: Sequence[Span]) -> list[list[Span]]:
    """Per answer, the stretches of speech between pauses; VAD segments are in time order.

    A gap shorter than a pause belongs to the speech around it, as in the usual definition of
    articulation rate (syllables per second of speaking time, pauses removed).
    """
    out = []
    for a, b in turns:
        merged: list[list[int]] = []
        for s, e in speech:
            s, e = max(s, a), min(e, b)
            if s >= e:
                continue
            if merged and s - merged[-1][1] < PAUSE_MS:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out.append([(s, e) for s, e in merged])
    return out


def pauses(turns: Sequence[Span], speech: Sequence[Span]) -> list[Span]:
    """The silent pauses inside the answers; the silence before an answer starts is latency, not a pause."""
    return [(end, start) for turn in runs(turns, speech) for (_, end), (start, _) in pairwise(turn)]


def place(found: Sequence[Span], word_starts: Sequence[int]) -> list[int]:
    """For each pause, how many words (in time order) start before its middle: its place in the text."""
    return [bisect_left(word_starts, (a + b) / 2) for a, b in found]


def timing(turns: Sequence[Span], speech: Sequence[Span], words: Sequence[str]) -> dict[str, Value]:
    """The timing measures of the answers; `words` are the transcript words spoken in them."""
    answer_ms = sum(b - a for a, b in turns)
    stretches = [r for turn in runs(turns, speech) for r in turn]
    if not stretches:
        return {k: absent(NO_SPEECH, "No speech was found in these answers.", u) for k, u in UNITS.items()}
    speaking_ms = sum(e - s for s, e in stretches)
    gaps = [b - a for a, b in pauses(turns, speech)]
    minutes = answer_ms / 60000
    out = {
        "speaking_time": measured(speaking_ms / 1000, "s"),
        "pauses_per_min": measured(len(gaps) / minutes, "per min", n=len(gaps)),
        "mean_pause": measured(sum(gaps) / len(gaps) / 1000, "s", n=len(gaps))
        if gaps
        else absent(NO_PAUSES, "There was no pause of 250 ms or more to average.", "s", n=0),
        "pauses_500ms": counted(sum(g >= VIEWS_MS[0] for g in gaps)),
        "pauses_1s": counted(sum(g >= VIEWS_MS[1] for g in gaps)),
    }
    syll, uncounted = syllables.total(words)
    spoken = [w for w in words if w.lower() not in FILLER_WORDS]
    note = f"words without letters (such as numbers) not counted: {uncounted}" if uncounted else None
    if syll == 0:
        text = "The transcript has no countable words in these answers."
        for k in ("speech_rate", "articulation_rate", "mean_length_of_run", "words_per_minute"):
            out[k] = absent(NO_WORDS, text, UNITS[k])
        return out
    out["speech_rate"] = measured(syll / (answer_ms / 1000), "syll/s", n=syll, note=note)
    out["articulation_rate"] = measured(syll / (speaking_ms / 1000), "syll/s", n=syll, note=note)
    out["mean_length_of_run"] = measured(syll / len(stretches), "syll", n=len(stretches), note=note)
    out["words_per_minute"] = measured(len(spoken) / minutes, "words/min", n=len(spoken))
    return out


def latency(pairs: Sequence[Span], mode: str, prepared: int = 0) -> dict[str, Value]:
    """Response latency from (question end, answer start) word times in ms: the median and the long waits.

    `prepared` answers followed the Part 2 preparation minute; they have no response time, so a scope that
    holds only those says so rather than that no answer started in time.
    """
    if mode not in LATENCY_MODES or (prepared and not pairs):
        code, text = (
            (NOT_IN_MODE, "Response latency is measured only when someone asks the questions.")
            if mode not in LATENCY_MODES
            else (AFTER_PREPARATION, "The answer follows the Part 2 preparation minute, not a question.")
        )
        return {"latency_median": absent(code, text, "s"), "latency_over_10s": absent(code, text, "count")}
    waits = [start - end for end, start in pairs]
    usable = [w for w in waits if 0 <= w <= LATENCY_CHECK_MS]
    overlaps = sum(w < 0 for w in waits)
    note = f"answers that started before the question ended, left out: {overlaps}" if overlaps else None
    long_waits = counted(
        sum(w > LATENCY_CHECK_MS for w in waits), note="check: longer than 10 s, not averaged"
    )
    if not usable:
        text = "No answer started 0 to 10 s after its question ended."
        return {"latency_median": absent(NO_ANSWERS, text, "s", n=0), "latency_over_10s": long_waits}
    return {
        "latency_median": measured(median(usable) / 1000, "s", n=len(usable), note=note),
        "latency_over_10s": long_waits,
    }
