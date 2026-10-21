"""Syllables counted from the spelling of the transcript's words, with no audio and no dictionary.

Known limit: a spelling rule, so some words are off by one ("video" gets 2, "quiet" 1, "homework" 3);
revisit if a pronouncing dictionary with a permissive licence becomes a dependency.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from fluentfolio.analyzers.measures.values import FILLER_WORDS

GROUP = re.compile(r"[aeiouy]+")
# Vowel pairs spoken as two syllables (cu-ri-o-si-ty, me-di-a, stad-i-um, ac-tu-al), except where the
# letter before makes the first vowel a glide (na-tion, so-cial, re-li-gion, lan-guage, qual-i-ty);
# also a vowel before a final -ing (be-ing) and a final -ea (i-de-a).
HIATUS = re.compile(r"(?<![cstxg])i(?=[aou])|(?<![qg])u(?=a)|[aeiouy](?=ing$)|e(?=a$)")
SILENT_E = re.compile(r"[^aeiouy]e$")
SOUNDED_LE = re.compile(r"[^aeiouy][lr]e$")  # ta-ble, cen-tre
SILENT_ES = re.compile(r"(?:[^aeiouyscxzghl]|[aeiouy]l)es$")  # makes, smiles: one; plac-es, tab-les: two
SILENT_ED = re.compile(r"[^aeiouytd]ed$")  # walk-ed is one syllable; want-ed, need-ed two
SOUNDED_NT = re.compile(r"[^aeiournt]nt$")  # did-n't and is-n't add one; don't and can't do not
SOME = "some"  # some-one, some-thing, some-times: the e of "some" is silent inside the word
EXCEPTIONS = {"maybe": 2}


def count(word: str) -> int | None:
    """The syllables of one written word, or None when it holds no letters (a number such as "2010")."""
    w = "".join(c for c in word.lower() if c.isalpha())
    if not w or not w.isascii():
        return None
    if w in EXCEPTIONS:
        return EXCEPTIONS[w]
    if w.startswith(SOME) and len(w) > len(SOME) + 1:
        rest = count(w[len(SOME) :])
        return None if rest is None else 1 + rest
    n = len(GROUP.findall(w)) + len(HIATUS.findall(w))
    if SOUNDED_NT.search(w) and "'" in word:
        n += 1
    if n > 1 and (
        (SILENT_E.search(w) and not SOUNDED_LE.search(w)) or SILENT_ES.search(w) or SILENT_ED.search(w)
    ):
        n -= 1
    return max(n, 1)


def total(words: Iterable[str]) -> tuple[int, int]:
    """Syllables in the spoken words (fillers left out), and how many words could not be counted."""
    syllables = uncounted = 0
    for word in words:
        if word.lower() in FILLER_WORDS:
            continue
        k = count(word)
        if k is None:
            uncounted += 1
        else:
            syllables += k
    return syllables, uncounted
