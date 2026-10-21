"""Audio quality of the 16 kHz mono WAV: level, noise floor, an SNR estimate, the band limit and one class.

It also looks for background music, with a heuristic and no model (see `music_spans`).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fluentfolio.core.time import union

RATE = 16000
FRAME = 480  # 30 ms
NFFT = 512
BLOCK_FRAMES = 2000  # one minute per read, so a three-hour file never sits in memory
SPEECH_BAND = (100.0, 4000.0)  # frame power is measured here, so rumble and hiss do not hide speech
FLOOR = 1e-10  # -100 dBFS; digital silence counts as this
NOISE_PERCENTILE = 10
EDGE_DROP_DB = 60.0  # the band limit: the highest frequency within 60 dB of the strongest one

# Class rules, checked in this order (the first match wins). Starting values, chosen on synthetic and composed
# test sessions; the SNR is this module's estimate, not a lab measure.
NOISY_BELOW_DB = 20.0
GATED_FLOOR_DB = -75.0  # a lossy file whose background is this quiet was gated by noise suppression
NARROW_UP_TO_HZ = 5000.0  # telephone audio stops near 4 kHz
CLEAN_FROM_DB = 35.0
# No speech: the speech band never gets loud, or never rises above its own background.
SILENT_LEVEL_DB = -60.0
SILENT_SNR_DB = 4.0

# Background music. Speech leaves the dips between its syllables, 3 to 8 a second, empty or noisy; music
# fills them with held notes whose harmonics stay where they were. So in the quietest frames of each window,
# the harmonic pattern of the spectrum is compared with the one 300 ms later.
MUSIC_N = 1024  # 64 ms frames: the harmonics of a 100 Hz voice lie about 6 bins apart
MUSIC_HOP = 160  # 10 ms
MUSIC_BAND = (100.0, 3000.0)
ENVELOPE_BINS = 9  # 140 Hz: the spectral envelope, taken away to leave the harmonics
LAG = 30  # frames (300 ms): a held note is still there, a syllable has moved on
WINDOW, STEP = 300, 100  # 3 s windows, 1 s apart
TROUGH_PERCENTILE = 40  # the quietest 40% of a window's frames
QUIET_RMS = 1e-3  # -60 dBFS: a frame this quiet holds no note, and a window this quiet is not judged
# Chosen by looking at synthetic music under speech and at speech, hum and noise without music.
MUSIC_STABILITY_MIN = 0.3
CHUNK = 60  # windows per read, so a long file never sits in memory whole

_WINDOW = np.hanning(FRAME)
_BINS = np.fft.rfftfreq(NFFT, 1 / RATE)
_BAND = (_BINS >= SPEECH_BAND[0]) & (_BINS <= SPEECH_BAND[1])
# A one-sided spectrum sum, scaled so that a frame's band power is its mean square within the band.
_SCALE = NFFT / 2 * float(np.sum(_WINDOW**2))
_MUSIC_WINDOW = np.hanning(MUSIC_N).astype(np.float32)
_MUSIC_BINS = np.fft.rfftfreq(MUSIC_N, 1 / RATE)
_MUSIC_BAND = (_MUSIC_BINS >= MUSIC_BAND[0]) & (_MUSIC_BINS <= MUSIC_BAND[1])


def _blocks(wav: Path) -> Iterator[np.ndarray]:
    """The file as power spectra of whole 30 ms frames, about one minute at a time."""
    for block in sf.blocks(str(wav), blocksize=FRAME * BLOCK_FRAMES, dtype="float32"):
        n = len(block) // FRAME
        if n:
            frames = block[: n * FRAME].reshape(n, FRAME).astype(np.float64)
            yield np.abs(np.fft.rfft(frames * _WINDOW, n=NFFT, axis=1)) ** 2


def _db(power: float) -> float:
    return round(10 * float(np.log10(max(power, FLOOR))), 1)


def measure(wav: Path) -> dict[str, Any]:
    """Level, noise floor and SNR estimate in the speech band, and the band limit of the louder frames."""
    power = np.concatenate([np.sum(s[:, _BAND], axis=1) / _SCALE for s in _blocks(wav)])
    middle = float(np.median(power))
    speech = float(np.mean(power[power >= middle]))  # the louder half of the frames
    noise = max(float(np.percentile(power, NOISE_PERCENTILE)), FLOOR)
    spectrum = np.zeros(len(_BINS))
    for s in _blocks(wav):  # a second pass, because the louder half is known only now
        spectrum += np.sum(s[np.sum(s[:, _BAND], axis=1) / _SCALE >= middle], axis=0)
    level = 10 * np.log10(spectrum + 1e-30)
    edge = float(_BINS[np.nonzero(level >= level.max() - EDGE_DROP_DB)[0][-1]])
    return {
        "level_dbfs": _db(speech),
        "noise_dbfs": _db(noise),
        "snr_db": round(_db(speech) - _db(noise), 1),
        "band_limit_hz": int(edge),
    }


def silent(q: dict[str, Any]) -> bool:
    """Silence, steady noise or a steady tone: nothing in the speech band stands out."""
    return bool(q["level_dbfs"] < SILENT_LEVEL_DB or q["snr_db"] < SILENT_SNR_DB)


def _fine(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each frame's harmonic pattern (log spectrum less its envelope, mean 0) and its RMS level."""
    frames = np.lib.stride_tricks.sliding_window_view(x, MUSIC_N)[::MUSIC_HOP]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    logs = np.log(np.abs(np.fft.rfft(frames * _MUSIC_WINDOW, axis=1))[:, _MUSIC_BAND] + 1e-6)
    sums = np.concatenate([np.zeros((len(logs), 1), logs.dtype), np.cumsum(logs, axis=1)], axis=1)
    envelope = (sums[:, ENVELOPE_BINS:] - sums[:, :-ENVELOPE_BINS]) / ENVELOPE_BINS
    half = ENVELOPE_BINS // 2
    fine = logs[:, half : half + envelope.shape[1]] - envelope
    return fine - fine.mean(axis=1, keepdims=True), rms


def _held(fine: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """How alike each frame's harmonic pattern is to the one LAG frames later; 0 where either is quiet."""
    a, b = fine[:-LAG], fine[LAG:]
    size = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    heard = (rms[:-LAG] >= QUIET_RMS) & (rms[LAG:] >= QUIET_RMS) & (size > 0)
    return np.where(heard, np.sum(a * b, axis=1) / np.where(size > 0, size, 1.0), 0.0)


def music_scores(wav: Path) -> list[tuple[int, float]]:
    """Each judged 3 s window's start in ms and how steady the harmonics in its quietest frames are."""
    total = (sf.info(str(wav)).frames - MUSIC_N) // MUSIC_HOP + 1
    out = []
    for first in range(0, total - WINDOW + 1, CHUNK * STEP):
        last = min(first + (CHUNK - 1) * STEP + WINDOW, total)
        x = sf.read(
            str(wav), start=first * MUSIC_HOP, stop=(last - 1) * MUSIC_HOP + MUSIC_N, dtype="float32"
        )[0]
        fine, rms = _fine(x)
        held = _held(fine, rms)
        for start in range(0, len(rms) - WINDOW + 1, STEP):
            level = rms[start : start + WINDOW]
            if level.mean() < QUIET_RMS:
                continue
            troughs = level[:-LAG] <= np.percentile(level, TROUGH_PERCENTILE)
            score = float(np.median(held[start : start + WINDOW - LAG][troughs]))
            out.append(((first + start) * MUSIC_HOP * 1000 // RATE, score))
    return out


def music_spans(wav: Path) -> list[list[int]]:
    """Where background music seems to play, in ms: the judged windows at or over the threshold, merged."""
    length = WINDOW * MUSIC_HOP * 1000 // RATE
    found = [(ms, ms + length) for ms, score in music_scores(wav) if score >= MUSIC_STABILITY_MIN]
    return [list(span) for span in union(found)]


def classify(q: dict[str, Any], *, lossy: bool) -> str:
    """One of clean, laptop, meeting_codec, phone or noisy; classes that lower reliabilities come first."""
    if q["snr_db"] < NOISY_BELOW_DB:
        return "noisy"
    if lossy and q["noise_dbfs"] <= GATED_FLOOR_DB:
        return "meeting_codec"
    if q["band_limit_hz"] <= NARROW_UP_TO_HZ:
        return "phone"
    if q["snr_db"] >= CLEAN_FROM_DB:
        return "clean"
    return "laptop"
