"""Hesitation and repair in the transcript: filled pauses per minute and candidate lists to check.

Repetitions and self-corrections are candidates found by simple rules, never counts: a repeated word
can be emphasis ("very very"), and only a listener can tell. "like", "you know" and "I mean" are
listed where they occur and never counted, because they are often ordinary words.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from fluentfolio.analyzers.measures.values import (
    FILLER_WORDS,
    NO_SPEECH,
    Pointer,
    Utterance,
    Value,
    absent,
    find,
    measured,
    tokens,
)

VERBATIM = "verbatim-best-effort"
REPEAT_WORDS = (2, 1)  # a repeated pair ("I went I went") is tried before a repeated word
RETRACE = 3  # words before a cut-off that a restart may repeat ("I go- I went")
LISTED = ("like", "you know", "i mean")
# A voiced gap: a stretch this long between two words of one utterance, in which the voice detector hears
# at least this much speech, may hold a hesitation the speech model did not write. Near the other
# speaker, the speech heard may be theirs, so no gap is marked there.
GAP_MS = 200
GAP_SPEECH_MS = 200
OTHER_SPEAKER_MS = 300


def filled_pauses(words: Sequence[str], answer_ms: int, transcript_style: str) -> Value:
    """Hesitation sounds (um, uh, er, ...) per minute of answer time."""
    if answer_ms <= 0:
        return absent(NO_SPEECH, "No answer time to divide by.", "per min")
    k = sum(w.lower() in FILLER_WORDS for w in words)
    note = (
        None
        if transcript_style == VERBATIM
        else "lower bound: this transcript leaves out some hesitation sounds"
    )
    return measured(k / (answer_ms / 60000), "per min", n=k, note=note)


def _spoken(text: str) -> list[tuple[int, str, bool]]:
    """Token index, word and cut-off flag of each token that is not a hesitation sound."""
    return [(i, w, cut) for i, (w, cut) in enumerate(tokens(text)) if w.lower() not in FILLER_WORDS]


def repetitions(utterances: Sequence[Utterance]) -> list[Pointer]:
    """Words or word pairs said again straight away, hesitation sounds between them ignored."""
    out = []
    for uid, text in utterances:
        all_words = [w for w, _ in tokens(text)]
        spoken = _spoken(text)
        seq = [w.lower() for _, w, _ in spoken]
        j = 0
        while j < len(seq):
            for k in REPEAT_WORDS:
                if seq[j : j + k] == seq[j + k : j + 2 * k]:
                    end = j + 2 * k
                    while seq[end : end + k] == seq[j : j + k]:
                        end += k
                    first, last = spoken[j][0], spoken[end - 1][0]
                    out.append(Pointer(uid, (first, last), " ".join(all_words[first : last + 1])))
                    j = end
                    break
            else:
                j += 1
    return out


def self_corrections(utterances: Sequence[Utterance]) -> list[Pointer]:
    """Each word broken off ("go-") with the word that replaced it, after any repeated lead-in."""
    out = []
    for uid, text in utterances:
        spoken = _spoken(text)
        for q, (pos, word, cut) in enumerate(spoken):
            if not cut:
                continue
            before = [w.lower() for _, w, _ in spoken[max(0, q - RETRACE) : q]]
            after = [w.lower() for _, w, _ in spoken[q + 1 :]]
            m = max(m for m in range(len(before) + 1) if after[:m] == before[len(before) - m :])
            if q + 1 + m < len(spoken):
                target_pos, target, _ = spoken[q + 1 + m]
                out.append(Pointer(uid, (pos, target_pos), f"{word}- -> {target}"))
            else:
                out.append(Pointer(uid, (pos, pos), f"{word}-"))
    return out


def voiced_gaps(
    words: Sequence[tuple[int, int]], speech: Sequence[tuple[int, int]], others: Sequence[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    """Gaps between neighbouring words of one utterance that hold speech: (start, end, speech heard), in ms.

    `words` are the utterance's (start, end) in order, `speech` the voice detector's segments, and
    `others` the other speaker's turns.
    """
    out = []
    for (_, a), (b, _) in pairwise(words):
        if b - a < GAP_MS:
            continue
        heard = sum(max(0, min(e, b) - max(s, a)) for s, e in speech)
        near = any(s < b + OTHER_SPEAKER_MS and e > a - OTHER_SPEAKER_MS for s, e in others)
        if heard >= GAP_SPEECH_MS and not near:
            out.append((a, b, heard))
    return out


def listed_phrases(utterances: Sequence[Utterance]) -> list[Pointer]:
    """Every "like", "you know" and "I mean", to read in context; never a count of fillers."""
    out = []
    for uid, text in utterances:
        words = [w for w, _ in tokens(text)]
        for first, last, _ in find([w.lower() for w in words], LISTED):
            out.append(Pointer(uid, (first, last), " ".join(words[first : last + 1])))
    return out
