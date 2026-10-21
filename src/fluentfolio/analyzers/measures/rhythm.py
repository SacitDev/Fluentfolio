"""Speech rhythm from MFA's phones: how much vowel lengths vary (nPVI-V, VarcoV) and the vowel share (%V).

These are context only: no norm says what a value means for a learner, so they stay in the details.
They come from MFA's phones alone; the CTC aligner's word times say nothing about phone lengths.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from fluentfolio.analyzers.measures.acoustic import INSUFFICIENT
from fluentfolio.analyzers.measures.values import Value, absent, measured

Phone = Sequence[int | str]  # start ms, end ms and the ARPAbet label, such as "AH0"
VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
# Rhythm values from fewer vowel sounds than this say "insufficient sample" instead of a number.
MIN_VOWELS = 20
NO_MFA = "no_mfa"
TOO_FEW = "too_few_vowels"
NO_MFA_TEXT = "MFA did not align these answers (it is off, not installed, or its check failed)."
TOO_FEW_TEXT = "Too few vowels between pauses for this value."


def vowel(label: str) -> bool:
    return label.rstrip("012") in VOWELS


def intervals(utterances: Sequence[Sequence[Phone]]) -> tuple[list[list[int]], list[int]]:
    """Vowel interval lengths in ms per stretch of speech between pauses, and consonant interval lengths.

    Neighbouring vowels make one interval, as do neighbouring consonants.
    """
    stretches: list[list[int]] = []
    consonants: list[int] = []
    for phones in utterances:
        end = None
        kind = None
        for start, stop, label in ((int(p[0]), int(p[1]), str(p[2])) for p in phones):
            length = stop - start
            if end is None or start > end:  # a pause starts a new stretch
                stretches.append([])
                kind = None
            now = vowel(label)
            if now and kind is True:
                stretches[-1][-1] += length
            elif now:
                stretches[-1].append(length)
            elif kind is False:
                consonants[-1] += length
            else:
                consonants.append(length)
            kind, end = now, stop
    return stretches, consonants


def measures(utterances: Sequence[Sequence[Phone]]) -> dict[str, Value]:
    """nPVI-V, VarcoV and %V of the MFA-aligned utterances of one scope, and the mean vowel and
    consonant phone lengths."""
    keys = {"npvi_v": "index", "varco_v": "index", "percent_v": "%", "vowel_ms": "ms", "consonant_ms": "ms"}
    if not any(utterances):
        return {k: absent(NO_MFA, NO_MFA_TEXT, unit) for k, unit in keys.items()}
    n = sum(vowel(str(p[2])) for u in utterances for p in u)
    if n < MIN_VOWELS:
        text = f"Insufficient sample: {n} vowel sounds, under the {MIN_VOWELS} these values need."
        return {k: absent(INSUFFICIENT, text, unit, n=n) for k, unit in keys.items()}
    stretches, consonants = intervals(utterances)
    vowels = [d for s in stretches for d in s]
    pairs = [(a, b) for s in stretches for a, b in zip(s, s[1:], strict=False)]
    sounds = [(int(p[1]) - int(p[0]), vowel(str(p[2]))) for u in utterances for p in u]
    out: dict[str, Value] = {}
    if pairs:
        pvi = 100 * statistics.mean(abs(a - b) / ((a + b) / 2) for a, b in pairs)
        out["npvi_v"] = measured(pvi, "index", 1, n=len(pairs))
    else:
        out["npvi_v"] = absent(TOO_FEW, TOO_FEW_TEXT, "index", n=0)
    if len(vowels) > 1:
        out["varco_v"] = measured(100 * statistics.stdev(vowels) / statistics.mean(vowels), "index", 1,
                                  n=len(vowels))  # fmt: skip
    else:
        out["varco_v"] = absent(TOO_FEW, TOO_FEW_TEXT, "index", n=len(vowels))
    out["percent_v"] = measured(100 * sum(vowels) / (sum(vowels) + sum(consonants)), "%", 1, n=len(sounds))
    for key, wanted in (("vowel_ms", True), ("consonant_ms", False)):
        lengths = [d for d, is_vowel in sounds if is_vowel is wanted]
        out[key] = (
            measured(statistics.mean(lengths), "ms", 0, n=len(lengths))
            if lengths else absent(TOO_FEW, TOO_FEW_TEXT, "ms", n=0)
        )  # fmt: skip
    return out
