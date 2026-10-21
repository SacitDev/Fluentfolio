"""The value every measure returns: a rounded number, or no number and the reason why."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import cache
from importlib import resources

from fluentfolio.asr.normalize import FILLERS, split

Span = tuple[int, int]  # start and end in ms
Utterance = tuple[str, str]  # its ID ("U031") and its text

# Reason codes for a value that cannot be measured. A missing value is never a weakness of the speaker.
NO_SPEECH = "no_speech"
NO_WORDS = "no_words"
NO_PAUSES = "no_pauses"
NOT_IN_MODE = "not_valid_in_mode"
AFTER_PREPARATION = "after_preparation"  # the Part 2 talk follows the preparation minute, not a question
NOT_APPLICABLE = (NOT_IN_MODE, AFTER_PREPARATION)  # the measure does not apply here: shown as "n/a"
NO_ANSWERS = "no_timed_answers"
TOO_FEW_WORDS = "too_few_words"
NO_CONTENT_WORDS = "no_content_words"
NO_MARKERS = "no_markers"

FILLER_WORDS = frozenset(FILLERS) | frozenset(FILLERS.values())  # every spelling: "er" as well as "uh"
MAX_PLACES = 3  # a measure is never shown with 4 or more decimals


@dataclass(frozen=True)
class Missing:
    code: str
    text: str


@dataclass(frozen=True)
class Value:
    """A number with its unit, or `value` None and `missing` saying why; never both, never neither."""

    value: float | None
    unit: str
    missing: Missing | None = None
    n: int | None = None  # how many items (pauses, words, answers) the value rests on
    note: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.missing is None):
            raise ValueError("a value has either a number or a reason it is missing")


@dataclass(frozen=True)
class Pointer:
    """One candidate for the learner or the AI to check: where it is and the words involved."""

    utterance_id: str
    at: tuple[int, int]  # first and last token index inside the utterance
    text: str


def rounded(x: float, places: int) -> float:
    """Round half up, as people do by hand (0.25 -> 0.3), not to the nearest even digit."""
    if not 0 <= places <= MAX_PLACES:
        raise ValueError(f"{places} decimals")
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def measured(x: float, unit: str, places: int = 1, n: int | None = None, note: str | None = None) -> Value:
    value = rounded(x, places)
    # A value kept to whole units reads "121 Hz", not "121.0 Hz", which would claim a decimal it lacks.
    return Value(int(value) if places == 0 else value, unit, n=n, note=note)


def counted(k: int, unit: str = "count", note: str | None = None) -> Value:
    return Value(k, unit, n=k, note=note)


def absent(code: str, text: str, unit: str, n: int | None = None) -> Value:
    return Value(None, unit, Missing(code, text), n=n)


def tokens(text: str) -> list[tuple[str, bool]]:
    """The words of an utterance as written, each with whether it was cut off ("wor-")."""
    out = []
    for raw in text.split():
        word, cut_off = split(raw)
        if word:
            out.append((word, cut_off))
    return out


@cache
def word_list(name: str) -> tuple[str, ...]:
    """The lines of a word list in the package data folder, lower case, comments left out."""
    text = resources.files("fluentfolio.data").joinpath(name).read_text(encoding="utf-8")
    lines = (line.strip().lower() for line in text.splitlines())
    return tuple(line for line in lines if line and not line.startswith("#"))


def find(words: Sequence[str], phrases: Iterable[str]) -> list[tuple[int, int, str]]:
    """Where phrases occur in lower-case words: (first, last index, phrase); longest first, no overlap."""
    parts = sorted((p.split() for p in phrases), key=len, reverse=True)
    hits = []
    i = 0
    while i < len(words):
        for p in parts:
            if list(words[i : i + len(p)]) == p:
                hits.append((i, i + len(p) - 1, " ".join(p)))
                i += len(p)
                break
        else:
            i += 1
    return hits
