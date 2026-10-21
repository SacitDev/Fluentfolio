"""Word variety and word frequency: MATTR over 50-word windows, and content words beyond common English.

Names of people and places are not vocabulary, so the parser's names are left out of the frequency profile,
and a word counts as common when its own form or its base form is common ("dishes" as "dish").
Known limit: wordfreq ranks American spellings, so a British spelling ("neighbourhood") can count as
rare; revisit if the frequency lists add British forms. Without the parser, names and base forms are not
known, so a name can count as rare.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from fluentfolio.analyzers.measures.values import (
    FILLER_WORDS,
    NO_CONTENT_WORDS,
    TOO_FEW_WORDS,
    Pointer,
    Utterance,
    Value,
    absent,
    measured,
    tokens,
    word_list,
)

WINDOW = 50
TOPS = (2000, 5000)
FUNCTION_FILE = "function_words.txt"


@dataclass(frozen=True)
class Lexical:
    mattr: Value
    outside_2k: Value  # % of content words outside the 2,000 most frequent English words
    outside_5k: Value
    rare: list[Pointer]  # each content word outside the top 5,000, at its first use


@cache
def top(n: int) -> frozenset[str]:
    # Loaded on first use: the import takes about a sixth of a second, which every command would pay at start.
    import wordfreq

    return frozenset(wordfreq.top_n_list("en", n))


def mattr(words: Sequence[str]) -> Value:
    """Moving-average type-token ratio: the share of different words in each 50-word window, averaged.

    Unlike a plain type-token ratio, it does not fall just because an answer is longer.
    """
    n = len(words)
    if n < WINDOW:
        return absent(TOO_FEW_WORDS, f"MATTR needs at least {WINDOW} words; there were {n}.", "ratio", n=n)
    seen = Counter(words[:WINDOW])
    total = len(seen)
    for i in range(WINDOW, n):
        seen[words[i]] += 1
        seen[words[i - WINDOW]] -= 1
        if seen[words[i - WINDOW]] == 0:
            del seen[words[i - WINDOW]]
        total += len(seen)
    return measured(total / ((n - WINDOW + 1) * WINDOW), "ratio", places=2, n=n)


def _base(word: str) -> str:
    """A possessive looked up as its noun: "mum's" as "mum"."""
    return word[:-2] if word.endswith("'s") else word


def profile(
    utterances: Sequence[Utterance],
    unsure: Collection[tuple[str, int]] = (),
    names: Collection[tuple[str, int]] = (),
    lemmas: Mapping[tuple[str, int], str] | None = None,
) -> Lexical:
    """MATTR and the content-word frequency profile; hesitation sounds and cut-off words left out.

    Each collection is keyed by (utterance ID, token index): `unsure` words may be misheard and are never
    listed as rare; `names` are left out of the profile (they still count for MATTR); `lemmas` gives a
    word's base form where it differs from the word.
    """
    function = frozenset(word_list(FUNCTION_FILE))
    base_forms = lemmas or {}
    words: list[str] = []
    content: list[tuple[str, int, str, str]] = []  # utterance ID, token index, word, base form
    for uid, text in utterances:
        for i, (w, cut) in enumerate(tokens(text)):
            low = w.lower()
            if cut or low in FILLER_WORDS:
                continue
            words.append(low)
            if low not in function and low.replace("'", "").isalpha() and (uid, i) not in names:
                content.append((uid, i, _base(low), base_forms.get((uid, i), _base(low))))
    if not content:
        text = "There were no content words to look up."
        none = absent(NO_CONTENT_WORDS, text, "%", n=0)
        return Lexical(mattr(words), none, none, [])

    def rare_at(k: int, word: str, base: str) -> bool:
        return word not in top(k) and base not in top(k)

    shares = [
        measured(100 * sum(rare_at(k, w, b) for _, _, w, b in content) / len(content), "%", n=len(content))
        for k in TOPS
    ]
    rare: dict[str, Pointer] = {}
    for uid, i, w, b in content:
        if rare_at(TOPS[1], w, b) and w not in rare and (uid, i) not in unsure:
            rare[w] = Pointer(uid, (i, i), w)
    return Lexical(mattr(words), shares[0], shares[1], list(rare.values()))
