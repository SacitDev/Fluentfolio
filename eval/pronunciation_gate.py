"""The pronunciation check's three numbers, against limits fixed before it first ran.

1. Flag line: the highest word score line at which native read speech (LibriSpeech test-clean, every
   utterance, grouped by speaker) gives a median of at most 1 flagged word per minute of audio. It is
   rounded down to 3 decimals, which can only flag fewer words.
2. Planted substitutions: 20 synthetic sentences each say one word with one sound changed ("sink" for
   "think"). Scored against the intended sentence, at least 90 % of those words must be listed: below the
   line and not heard as written, as the app lists them.
3. speechocean762 test split: Pearson r of each phone's score against the experts' phone scores (0 to 2),
   at least 0.30, with the experts' expected phones as the only form.
Words are checked as the app checks them: the same phone model, US and UK forms, the dictionary's
pronunciations for a word heard as written, and the same words left out.
If a number misses its limit, every output of the pronunciation check says it is experimental.

Usage, from publishable/: python eval/pronunciation_gate.py --data DIR --phone-model DIR --dictionary FILE
--data holds librispeech/LibriSpeech/test-clean and speechocean762/test-00000-of-00001.parquet; --phone-model
is the phone model's folder (vocab.json, onnx/model.onnx) and --dictionary the aligner's pronunciation
dictionary, both as the app downloads them. Their SHA-256 values are printed with the data's.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import soundfile as sf

from fluentfolio.alignment import mfa, phones
from fluentfolio.analyzers.measures.pronunciation import flagged, heard_as_written, skip_reason

NATIVE_MAX_PER_MIN = 1.0
PLANTED_MIN = 0.90
PEARSON_MIN = 0.30

PLANTED = 20
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "audio"
LIBRISPEECH = "librispeech/LibriSpeech/test-clean"
LIBRISPEECH_ARCHIVE = "librispeech/test-clean.tar.gz"
SPEECHOCEAN = "speechocean762/test-00000-of-00001.parquet"
# The experts' ARPAbet phones as the phone model's units; an unstressed AH and ER are the reduced vowels.
ARPABET = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AH0": "ə", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ",
    "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜː", "ER0": "ɚ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h",
    "IH": "ɪ", "IY": "iː", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ",
    "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː",
    "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}  # fmt: skip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def note(text: str) -> None:
    print(text, file=sys.stderr, flush=True)  # progress, so a long run shows it is alive


class Models(NamedTuple):
    """The downloaded files the check runs on, given by path so that only the measured code is imported."""

    phone_model: Path
    dictionary: Path


class Scorer:
    """The app's phone model, forms and word rules, for many transcripts at once."""

    def __init__(self, transcripts: list[list[str]], models: Models) -> None:
        self.vocab = json.loads((models.phone_model / "vocab.json").read_text(encoding="utf-8"))
        self.model = phones.session(models.phone_model / "onnx" / "model.onnx")
        needed = sorted({t for words in transcripts for w in words for t in mfa.tokens(w)})
        self.said = [phones.pronounce(needed, form) for form in phones.FORMS]
        lines, self.guessed = mfa.mini_dictionary(models.dictionary, set(needed))
        self.listed = phones.listed_forms(lines)

    def words(self, audio: np.ndarray, words: list[str]) -> list[tuple[str, dict[str, Any] | None]]:
        """Each word with its fit, or None when the app would not check it."""
        tokens = [mfa.tokens(w) for w in words]
        fits = phones.fits(phones.scores(self.model, audio), tokens, self.said, self.vocab, self.listed)
        out = []
        for word, toks, fit in zip(words, tokens, fits, strict=True):
            guessed = any(t in self.guessed for t in toks)
            out.append((word, fit if skip_reason(word, guessed, None, None) is None else None))
        return out


def native(root: Path, models: Models) -> tuple[dict[str, list[float]], dict[str, float], int]:
    """Each speaker's checked word scores and minutes of audio, and the number of utterances.

    A word heard as written is never listed, whatever its score, so its score is left out here.
    """
    items = []
    for listing in sorted(root.glob("*/*/*.trans.txt")):
        for line in listing.read_text(encoding="utf-8").splitlines():
            utterance, text = line.split(" ", 1)
            items.append((utterance.split("-")[0], listing.parent / f"{utterance}.flac", text.split()))
    scorer = Scorer([words for _, _, words in items], models)
    scores: dict[str, list[float]] = defaultdict(list)
    minutes: dict[str, float] = defaultdict(float)
    for n, (speaker, path, words) in enumerate(items, 1):
        audio, rate = sf.read(path, dtype="float32")
        if rate != phones.RATE:
            raise ValueError(f"{path.name}: {rate} Hz")
        minutes[speaker] += len(audio) / rate / 60
        found = [fit for _, fit in scorer.words(audio, words) if fit is not None]
        scores[speaker] += [fit["score"] for fit in found if not heard_as_written(fit)]
        if n % 200 == 0:
            note(f"native: {n} of {len(items)} utterances")
    return scores, minutes, len(items)


def median_rate(scores: dict[str, np.ndarray], minutes: dict[str, float], line: float) -> float:
    """The median over speakers of words scoring below the line, per minute."""
    return float(np.median([np.searchsorted(s, line) / minutes[k] for k, s in scores.items()]))


def flag_line(scores: dict[str, np.ndarray], minutes: dict[str, float]) -> float:
    """The highest score line whose median native rate stays within the limit (the rate only rises)."""
    lines = np.unique(np.concatenate(list(scores.values())))
    low, high = 0, len(lines) - 1  # the lowest line flags nothing, so it always qualifies
    while low < high:
        mid = (low + high + 1) // 2
        if median_rate(scores, minutes, float(lines[mid])) <= NATIVE_MAX_PER_MIN:
            low = mid
        else:
            high = mid - 1
    return float(lines[low])


def planted(line: float, models: Models) -> tuple[int, list[str]]:
    """How many planted words fall below the line, with one line of detail per clip and the controls."""
    clips = json.loads((FIXTURES / "clips.json").read_text(encoding="utf-8"))
    pairs = [(clips[f"sub{k:02d}_ok"]["text"].split(), clips[f"sub{k:02d}_bad"]["text"].split())
             for k in range(1, PLANTED + 1)]  # fmt: skip
    scorer = Scorer([ok for ok, _ in pairs], models)
    caught, lines, false_targets, others, other_flags = 0, [], 0, 0, 0
    for k, (ok, bad) in enumerate(pairs, 1):
        target = next(
            i for i, (a, b) in enumerate(zip(ok, bad, strict=True)) if mfa.tokens(a) != mfa.tokens(b)
        )
        for name in ("bad", "ok"):
            path = FIXTURES / f"sub{k:02d}_{name}.opus"
            audio, rate = sf.read(path, dtype="float32")
            if rate != phones.RATE or audio.ndim != 1:
                raise ValueError(f"{path.name}: expected 16 kHz mono")
            found = scorer.words(audio, ok)
            word, fit = found[target]
            low = fit is not None and flagged(fit, line)
            rest = [f for i, (_, f) in enumerate(found) if i != target and f is not None]
            others += len(rest)
            other_flags += sum(flagged(f, line) for f in rest)
            if name == "ok":
                false_targets += low
                continue
            caught += low
            shown = "not checked" if fit is None else (
                f"score {fit['score']:.3f}, expected {'/'.join(fit['forms'])}, heard {''.join(fit['heard'])}"
            )  # fmt: skip
            lines.append(
                f"  sub{k:02d} {word!r} said as {bad[target]!r}: {'flagged' if low else 'missed'} ({shown})"
            )
    lines.append(
        f"  controls: the intended word is flagged in {false_targets} of {PLANTED} correct clips; "
        f"{other_flags} of {others} other checked words in all {2 * PLANTED} clips are flagged"
    )
    return caught, lines


def speechocean(path: Path, models: Models) -> tuple[float, int, int, int]:
    """Pearson r of phone scores against the experts' scores; utterances, phones scored, phones left out."""
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["words", "audio"]).to_pylist()
    scorer = Scorer([], models)
    ours: list[float] = []
    theirs: list[float] = []
    left_out = 0
    for n, row in enumerate(rows, 1):
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if rate != phones.RATE:
            raise ValueError(f"utterance {n}: {rate} Hz")
        names = [f"w{i}" for i in range(len(row["words"]))]
        expected = {}
        for name, word in zip(names, row["words"], strict=True):
            mapped = [ARPABET.get(p) or ARPABET.get(p.rstrip("012")) for p in word["phones"]]
            expected[name] = [] if None in mapped else mapped  # a phone not in the table: word left out
        log_probs = phones.scores(scorer.model, audio)
        fits = phones.fits(log_probs, [[name] for name in names], [expected], scorer.vocab)
        for word, fit in zip(row["words"], fits, strict=True):
            if fit is None or len(fit["phones"]) != len(word["phones"]):
                left_out += len(word["phones"])
                continue
            ours += fit["phones"]
            theirs += word["phones-accuracy"]
        if n % 200 == 0:
            note(f"speechocean762: {n} of {len(rows)} utterances")
    return float(np.corrcoef(ours, theirs)[0, 1]), len(rows), len(ours), left_out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--phone-model", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    args = parser.parse_args(argv)
    data, models = args.data, Models(args.phone_model, args.dictionary)
    started = time.monotonic()
    print(f"limits: native median at most {NATIVE_MAX_PER_MIN} flagged words per minute; planted at least "
          f"{PLANTED_MIN:.2f} flagged; phone-level Pearson r at least {PEARSON_MIN:.2f}")  # fmt: skip
    print(f"data: {LIBRISPEECH_ARCHIVE} sha256 {sha256(data / LIBRISPEECH_ARCHIVE)}")
    print(f"data: {SPEECHOCEAN} sha256 {sha256(data / SPEECHOCEAN)}")
    for path in sorted(FIXTURES.glob("sub*.opus")):
        print(f"data: tests/fixtures/audio/{path.name} sha256 {sha256(path)}")
    for name in ("vocab.json", "onnx/model.onnx"):
        print(f"model: phone model {name} sha256 {sha256(models.phone_model / name)}")
    print(f"model: dictionary {models.dictionary.name} sha256 {sha256(models.dictionary)}")

    found, minutes, utterances = native(data / LIBRISPEECH, models)
    scores = {k: np.sort(np.array(v)) for k, v in found.items()}
    exact = flag_line(scores, minutes)
    line = math.floor(exact * 1000) / 1000
    rates = [np.searchsorted(s, line) / minutes[k] for k, s in scores.items()]
    median = median_rate(scores, minutes, line)
    native_ok = median <= NATIVE_MAX_PER_MIN  # rounded down, so this holds
    print(f"1 native: {utterances} utterances, {len(scores)} speakers, {sum(minutes.values()):.1f} min, "
          f"{sum(len(s) for s in scores.values())} checked words")  # fmt: skip
    print(f"1 native: flag line {exact:.6f}, used as {line:.3f}")
    print(f"1 native: median {median:.3f} flagged words per minute (speakers from {min(rates):.3f} to "
          f"{max(rates):.3f}): {'PASS' if native_ok else 'FAIL'}")  # fmt: skip

    caught, detail = planted(line, models)
    planted_ok = caught / PLANTED >= PLANTED_MIN
    print(f"2 planted: {caught} of {PLANTED} substituted words flagged ({caught / PLANTED:.2f}): "
          f"{'PASS' if planted_ok else 'FAIL'}")  # fmt: skip
    print("\n".join(detail))

    r, rows, scored, left_out = speechocean(data / SPEECHOCEAN, models)
    pearson_ok = r >= PEARSON_MIN
    print(f"3 speechocean762: {rows} utterances, {scored} phones scored, {left_out} left out; Pearson r "
          f"{r:.3f}: {'PASS' if pearson_ok else 'FAIL'}")  # fmt: skip
    print(f"time: {time.monotonic() - started:.0f} s")
    passed = native_ok and planted_ok and pearson_ok
    print(f"pronunciation check: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
