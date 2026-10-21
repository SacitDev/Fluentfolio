"""The check on MFA's output: MFA can stretch a word across a pause and still exit 0.

Each utterance passes every rule, or the next aligner's times are used for it.
"""

from __future__ import annotations

from collections.abc import Sequence

# Starting limits until real learner recordings set them: just above the largest values on the clean
# composed test sessions (a deviation of 4.66 and a run of 4 short phones in 51 utterances).
T1 = 5.0  # MFA's phone-duration deviation: the largest |z| of a phone's log duration in the utterance
T2 = 4  # the longest run of phones shorter than 11 ms
SHORT_MS = 11
WORD_BASE_MS = 300  # a word may last 0.3 s plus 0.25 s per phone
WORD_PER_PHONE_MS = 250
MIDPOINT_MS = 500  # a word's middle this far from the speech model's is moved
MIDPOINT_SHARE = 0.10  # more than this share of moved words fails the utterance
OUTSIDE_MS = 500  # more detected speech than this outside all words fails it

Span = tuple[int, int]  # start and end in ms
Interval = tuple[int, int, str]  # a phone: start and end in ms, and its label


def short_run(phones: Sequence[Interval]) -> int:
    """The longest run of phones under 11 ms in one utterance.

    MFA's own column for this is not used: it starts its count once for all the utterances it checks and
    ends a run only when the run sets a new maximum, so it can report a run that no utterance has.
    """
    run = best = 0
    for a, b, _ in phones:
        run = run + 1 if b - a < SHORT_MS else 0
        best = max(best, run)
    return best


def outside_ms(speech: Sequence[Sequence[int]], words: Sequence[Span], window: Span) -> int:
    """Detected speech inside the window that no word covers, in ms."""
    total = 0
    for a, b in speech:
        pieces = [(max(a, window[0]), min(b, window[1]))]
        for s, e in words:
            pieces = [q for p in pieces for q in ((p[0], min(p[1], s)), (max(p[0], e), p[1])) if q[0] < q[1]]
        total += sum(e - s for s, e in pieces if s < e)
    return total


def check(
    deviation: float | None, phones: Sequence[Interval], words: Sequence[Span], asr: Sequence[Span],
    speech: Sequence[Sequence[int]], window: Span, unplaced: Sequence[Span] = (),
) -> str | None:  # fmt: skip
    """The first rule an utterance breaks, or None when MFA's times can be used.

    `words` and `asr` hold MFA's and the speech model's span of the same words, in order. `unplaced` holds
    the speech model's span of words MFA was not given (broken-off ones), whose speech is not a gap.
    """
    if deviation is not None and deviation > T1:
        return "phone_duration"
    if short_run(phones) > T2:
        return "short_phones"
    for s, e in words:
        inside = sum(s <= (a + b) / 2 < e for a, b, _ in phones)
        if e - s > WORD_BASE_MS + WORD_PER_PHONE_MS * inside:
            return "long_word"
    moved = sum(abs((s + e) - (a + b)) / 2 > MIDPOINT_MS for (s, e), (a, b) in zip(words, asr, strict=True))
    if words and moved / len(words) > MIDPOINT_SHARE:
        return "word_midpoints"
    if outside_ms(speech, [*words, *unplaced], window) > OUTSIDE_MS:
        return "speech_outside_words"
    return None
