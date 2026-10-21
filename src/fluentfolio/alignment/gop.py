"""How well each expected phone of a word fits the phone model's frame scores (goodness of pronunciation).

A phone's score compares two CTC paths over the word's frames: the best path through the phones as
written, and the best path when that one phone may be any sound. It is 0 when the written phone is what
the model hears best there, and falls as another sound fits better. Nothing is trained; the line below
which a word is flagged is set on native speech by eval/pronunciation_gate.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fluentfolio.alignment import ctc


@dataclass(frozen=True)
class Fit:
    """One word under the form (such as US or UK English) that fits it best."""

    form: int  # position of that form in the list given
    score: float  # the word's lowest phone score
    phones: list[float]  # each expected unit's score, in order
    heard: list[int]  # the units the model hears best over the word's frames
    frames: tuple[int, int]  # the word's first and last frame


def heard(log_probs: NDArray[np.float64], blank: int) -> list[int]:
    """The best unit in each frame, with repeats merged and blanks dropped (greedy CTC decoding)."""
    out: list[int] = []
    before = blank
    for unit in log_probs.argmax(axis=1).tolist():
        if unit != before and unit != blank:
            out.append(unit)
        before = unit
    return out


def best_path(emit: NDArray[np.float64], jump_ok: NDArray[np.bool_]) -> float:
    """The score of the best CTC path through the state columns of `emit`: blank, unit, blank, ..., blank."""
    n = emit.shape[1]
    score = np.full(n, -np.inf)
    score[:2] = emit[0, :2]
    for t in range(1, len(emit)):
        step = np.concatenate(([-np.inf], score[:-1]))
        jump = np.where(jump_ok, np.concatenate(([-np.inf, -np.inf], score[:-2])), -np.inf)
        score = np.maximum(np.maximum(score, step), jump) + emit[t]
    return float(max(score[-1], score[-2]))


def phone_scores(
    log_probs: NDArray[np.float64], rival: NDArray[np.float64], units: Sequence[int], blank: int
) -> list[float] | None:
    """Each unit's score over these frames; `rival` holds each frame's best non-blank log-probability.

    None when the units cannot be placed on the frames.
    """
    states = np.full(2 * len(units) + 1, blank)
    states[1::2] = units
    jump_ok = np.zeros(len(states), dtype=bool)
    jump_ok[3::2] = states[3::2] != states[1:-2:2]  # as in the aligner: no direct step between equal units
    emit = log_probs[:, states]
    written = best_path(emit, jump_ok)
    if not np.isfinite(written):
        return None
    out = []
    for i in range(len(units)):
        free, free_jump = emit.copy(), jump_ok.copy()
        free[:, 2 * i + 1] = rival  # this unit may be any sound
        free_jump[2 * i + 1 : 2 * i + 4 : 2] = True  # a free unit may follow or lead to any unit directly
        out.append(written - best_path(free, free_jump))
    return out


def fit_words(
    log_probs: NDArray[np.float64], forms: Sequence[Sequence[Sequence[int]]], blank: int
) -> list[Fit | None]:
    """Each word's best fit over the forms; `forms[f][w]` holds word w's units in form f.

    A word gets None when no form gives it units, or when no form can be placed on the utterance.
    """
    rival = np.delete(log_probs, blank, axis=1).max(axis=1)
    best: list[Fit | None] = [None] * len(forms[0])
    for f, form in enumerate(forms):
        spans = ctc.align(log_probs, [u for word in form for u in word], blank)
        if spans is None:
            continue
        k = 0
        for w, units in enumerate(form):
            if not units:
                continue
            first, last = spans[k][0], spans[k + len(units) - 1][1]
            k += len(units)
            frames = slice(first, last + 1)
            scores = phone_scores(log_probs[frames], rival[frames], units, blank)
            old = best[w]
            if scores is not None and (old is None or min(scores) > old.score):
                best[w] = Fit(f, min(scores), scores, heard(log_probs[frames], blank), (first, last))
    return best
