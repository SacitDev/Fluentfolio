"""Words to check for pronunciation (X06) and the line saying what the check covered (X07).

Never a score: a word is listed when the phone model hears one of its sounds clearly differently from
both its US and its UK form. An accent is not an error, and a word that is not listed may still be unclear.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fluentfolio.analyzers.measures.values import Pointer, word_list

ANALYZER = "pronunciation"
ACCENT = "An accent is not an error."
# The highest word score at which native read speech gives a median of at most 1 flagged word per minute,
# as eval/pronunciation_gate.py found it (rounded down); a word scoring below it is listed.
FLAG_BELOW = -7.277
VALIDATION = (
    "Experimental: did not pass its accuracy check. On native read speech (LibriSpeech test-clean, "
    "40 speakers) it lists a median of 0.998 words per minute, but it flagged 2 of 20 planted sound swaps "
    "(90% needed). Its sound scores agree with expert scores at r = 0.43 (speechocean762 test, 2,500 "
    "recordings). Measured on the CPU on 2026-10-19."
)
# Starting values for a fair sample (tests/tolerances.toml); a shorter answer is not checked.
MIN_ANSWER_MS = 2000
MIN_WORDS = 5
MIN_PATTERN = 2  # a sound heard as the same other sound in this many listed words is a pattern
# The model, eSpeak and the dictionary write some sounds differently: with or without a length mark, the
# r-coloured "er" as one symbol or two, and the vowel of "go" in three ways. Each pair here is one sound; the
# vowels of "ship" and "sheep" stay apart.
SAME_SOUND = (("ː", ""), ("ɚ", "əɹ"), ("oʊ", "o"), ("əʊ", "o"))

WEAK = "weak_form"
GUESSED = "guessed_pronunciation"
UNSURE = "unsure_word"
OVERLAP = "other_voice"
MUSIC = "background_music"
NO_FIT = "not_placed"
SHORT = "answer_too_short"
SKIPPED = {  # each follows "<n> word(s)"
    WEAK: "with a normal reduced form",
    GUESSED: "whose expected sounds were guessed",
    UNSURE: "the speech model was unsure of",
    OVERLAP: "spoken over another voice",
    MUSIC: "heard over background music",
    NO_FIT: "the phone model could not place",
    SHORT: "in answers too short to check",
}


@dataclass(frozen=True)
class Word:
    """One word of an answer as the check sees it."""

    utterance_id: str
    position: int  # token index inside the utterance
    text: str
    skip: str | None  # why it is not checked, or None
    fit: dict[str, Any]  # its entry in stages/phones.json; empty when it has none


@dataclass(frozen=True)
class Found:
    listed: list[Pointer]  # every word below the flag line, in time order
    patterns: str
    coverage: str


def switched_on(settings: Mapping[str, Any]) -> bool:
    """Whether a job's settings run the check; as for every analyzer, one missing from the table is on."""
    return bool(settings.get("analyzers", {}).get(ANALYZER, True))


def skip_reason(text: str, guessed: bool, confidence: float | None, threshold: float | None) -> str | None:
    """Why a word is left out of the check, or None; `threshold` is the speech model's (None: not set)."""
    if text.lower() in word_list("weak_forms.txt"):
        return WEAK
    if guessed:
        return GUESSED
    if threshold is not None and (confidence is None or confidence < threshold):
        return UNSURE
    return None


def substitutions(expected: Sequence[str], heard: Sequence[str]) -> list[tuple[str, str]]:
    """The (expected, heard) sound pairs where the closest match of the two sequences swaps one sound."""
    n, m = len(expected), len(heard)
    cost = [[i + j if i * j == 0 else 0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = expected[i - 1] == heard[j - 1]
            cost[i][j] = min(cost[i - 1][j - 1] + (not same), cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    pairs = []
    i, j = n, m
    while i and j:
        if cost[i][j] == cost[i - 1][j - 1] + (expected[i - 1] != heard[j - 1]):
            if expected[i - 1] != heard[j - 1]:
                pairs.append((expected[i - 1], heard[j - 1]))
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _one_spelling(sounds: str) -> str:
    for variant, sound in SAME_SOUND:
        sounds = sounds.replace(variant, sound)
    return sounds


def heard_as_written(fit: Mapping[str, Any]) -> bool:
    """Whether the model heard the sounds of one of the word's forms or dictionary pronunciations.

    Such a word can still score low: the model has some runs of sounds both as one unit and as parts,
    and hearing the parts where the form has the unit costs the written path with no sound different. The
    dictionary adds what eSpeak's one form per word misses, such as "live" as a verb and reduced forms.
    """
    heard = _one_spelling("".join(fit["heard"]))
    written = ["".join(fit["expected"]), *fit["forms"], *fit.get("listed", ())]
    return heard in {_one_spelling(form) for form in written}


def flagged(fit: Mapping[str, Any], flag_below: float) -> bool:
    """Whether a checked word is listed: it scores below the line and was not heard as written."""
    return bool(fit["score"] < flag_below) and not heard_as_written(fit)


def _pointer(word: Word) -> Pointer:
    us, uk = word.fit["forms"]
    expected = f"/{us}/" if us == uk else f"/{us}/ (US) or /{uk}/ (UK)"
    text = f"{word.text}: expected {expected}, heard /{''.join(word.fit['heard'])}/"
    return Pointer(word.utterance_id, (word.position, word.position), text)


def check(answers: Sequence[tuple[int, Sequence[Word]]], flag_below: float) -> Found:
    """The words to check in one candidate's answers, each given as (its length in ms, its words in order)."""
    listed: list[Pointer] = []
    reasons: Counter[str] = Counter()
    checked: list[Word] = []
    all_low: list[Word] = []
    for length_ms, words in answers:
        fair = [w for w in words if w.skip is None]
        if length_ms < MIN_ANSWER_MS or len(fair) < MIN_WORDS:
            reasons[SHORT] += len(words)
            continue
        reasons.update(w.skip for w in words if w.skip is not None)
        checked += fair
        low = [w for w in fair if flagged(w.fit, flag_below)]
        all_low += low
        listed += [_pointer(w) for w in low]
    return Found(listed, _patterns(checked, all_low), _coverage(len(checked), reasons))


def _patterns(checked: list[Word], low: list[Word]) -> str:
    """Sounds heard as the same other sound in several listed words, as counts, never a share."""
    swaps: Counter[tuple[str, str]] = Counter()
    for w in low:
        swaps.update(set(substitutions(w.fit["expected"], w.fit["heard"])))
    found = []
    for (sound, other), k in swaps.most_common():
        if k >= MIN_PATTERN:
            n = sum(sound in w.fit["expected"] for w in checked)
            found.append(f"/{sound}/ heard as /{other}/ in {k} of {n} words")
    return "; ".join(found) or "no repeated pattern"


def _words(n: int) -> str:
    return f"{n} word" if n == 1 else f"{n} words"


def _coverage(checked: int, reasons: Counter[str]) -> str:
    total = checked + sum(reasons.values())
    left = "; ".join(f"{_words(k)} {SKIPPED[code]}" for code, k in reasons.most_common())
    return f"Checked {checked} of {_words(total)}" + (f" (not checked: {left})." if left else ".")
