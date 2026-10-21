"""Fluentfolio's CTC aligner: known phones placed on the phone model's frame scores by Viterbi, in numpy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FRAME_MS = 20  # the phone model gives one frame per 320 samples at 16 kHz
STRESS = str.maketrans("", "", "ˈˌ")  # espeak's stress marks; the model has no unit for them
STAY, STEP, JUMP = 0, 1, 2


def log_softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    x = logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return np.asarray(x - np.log(np.exp(x).sum(axis=-1, keepdims=True)))


def tokenize(phones: Sequence[str], vocab: Mapping[str, int]) -> list[int]:
    """The model's units for espeak's phones, longest match first within each phone.

    A symbol the model has no unit for is left out, so a word may keep fewer units than phones.
    """
    longest = max(map(len, vocab))
    units = []
    for phone in phones:
        text, i = phone.translate(STRESS), 0
        while i < len(text):
            for n in range(min(longest, len(text) - i), 0, -1):
                if text[i : i + n] in vocab:
                    units.append(vocab[text[i : i + n]])
                    i += n
                    break
            else:
                i += 1
    return units


def align(log_probs: NDArray[np.float64], units: Sequence[int], blank: int) -> list[tuple[int, int]] | None:
    """The first and last frame of each unit on the best CTC path, or None when the units cannot fit."""
    frames = len(log_probs)
    repeats = sum(a == b for a, b in zip(units, units[1:], strict=False))
    if not units or frames < len(units) + repeats:  # a repeated unit needs a blank frame between
        return None
    states = np.full(2 * len(units) + 1, blank)
    states[1::2] = units
    n = len(states)
    # A unit may follow the unit before it directly, skipping the blank, unless the two are the same.
    jump_ok = np.zeros(n, dtype=bool)
    jump_ok[3::2] = states[3::2] != states[1:-2:2]
    emit = log_probs[:, states]
    score = np.full(n, -np.inf)
    score[:2] = emit[0, :2]
    back = np.zeros((frames, n), dtype=np.int8)
    for t in range(1, frames):
        step = np.concatenate(([-np.inf], score[:-1]))
        jump = np.where(jump_ok, np.concatenate(([-np.inf, -np.inf], score[:-2])), -np.inf)
        choices = np.stack((score, step, jump))
        best = choices.argmax(axis=0)
        back[t] = best
        score = choices[best, np.arange(n)] + emit[t]
    state = n - 1 if score[n - 1] >= score[n - 2] else n - 2
    if not np.isfinite(score[state]):
        return None
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    spans = []
    for k in range(len(units)):
        hit = np.flatnonzero(path == 2 * k + 1)
        spans.append((int(hit[0]), int(hit[-1])))
    return spans


def word_times(
    log_probs: NDArray[np.float64], words: Sequence[Sequence[int]], blank: int, offset_ms: int
) -> list[tuple[int, int] | None] | None:
    """Start and end in ms of each word, given its units; None for a word without units.

    Returns None when the utterance's units cannot be placed at all.
    """
    spans = align(log_probs, [u for w in words for u in w], blank)
    if spans is None:
        return None
    out: list[tuple[int, int] | None] = []
    k = 0
    for w in words:
        if not w:
            out.append(None)
            continue
        first, last = spans[k][0], spans[k + len(w) - 1][1]
        out.append((offset_ms + first * FRAME_MS, offset_ms + (last + 1) * FRAME_MS))
        k += len(w)
    return out
