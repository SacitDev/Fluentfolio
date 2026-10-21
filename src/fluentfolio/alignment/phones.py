"""The phones step: the phone model on each candidate utterance, the CTC aligner's word times from it, and
how well each word fits its expected phones; and the word-time step for a speech model that gives no times.

The model's frame scores never leave this process; stages/phones.json holds times and per-word fits only.
The expected phones come from eSpeak NG (US and UK English), through the copy inside the espeakng-loader
package.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from itertools import islice, product
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fluentfolio.alignment import ctc, gop, mfa
from fluentfolio.analyzers.measures import pronunciation
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import models_dir
from fluentfolio.models import catalogue
from fluentfolio.models.download import local_path, require

MODEL = "phone-model"
DICTIONARY = "pronunciation-dictionary"
RATE = 16000
UNTIMED = "untimed"  # a word the CTC aligner could not place, which keeps its chunk's span
SPECIAL = ("<pad>", "<s>", "</s>", "<unk>")  # model tokens that are not phones; <pad> is the CTC blank
FORMS = ("en-us", "en-gb")  # a word may match either form of its expected sounds
# The dictionary's ARPAbet phones in the eSpeak symbols the phone model hears; an unstressed AH, ER and IY are
# the reduced vowels eSpeak writes.
ARPABET = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AH0": "ə", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ",
    "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜː", "ER0": "ɚ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h",
    "IH": "ɪ", "IY": "iː", "IY0": "i", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ",
    "UW": "uː", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}  # fmt: skip
LISTED_MAX = 8  # dictionary pronunciations kept for a word of several tokens, whose choices multiply


def espeak(language: str) -> Any:
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    # The package's own library and voice data, never an eSpeak NG installed on the computer.
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    return EspeakBackend(language)


def pronounce(words: list[str], language: str = "en-us") -> dict[str, list[str]]:
    """Each word's phones as eSpeak NG says them."""
    from phonemizer.separator import Separator

    found = espeak(language).phonemize(words, separator=Separator(phone=" ", word=None), strip=True)
    return {w: p.split() for w, p in zip(words, found, strict=True)}


def session(path: Path) -> Any:
    import onnxruntime as ort

    ort.disable_telemetry_events()  # the Windows trace path ignores ORT_DISABLE_TELEMETRY
    options = ort.SessionOptions()
    options.log_severity_level = 3
    # Named explicitly: onnxruntime also offers a remote (Azure) provider, and analysis stays offline.
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def scores(model: Any, audio: np.ndarray) -> np.ndarray:
    """Log-probabilities per 20 ms frame, after the scaling the model was trained with."""
    x = audio.astype(np.float64)
    x = ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)  # zero mean, unit variance
    logits = model.run(None, {model.get_inputs()[0].name: x[None, :]})[0][0]
    return ctc.log_softmax(logits)


def fits(
    log_probs: np.ndarray, words: Sequence[Sequence[str]], said: Sequence[Mapping[str, list[str]]],
    vocab: Mapping[str, int], listed: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any] | None]:  # fmt: skip
    """How well each word fits the better of its forms; `said` holds each form's phones per token.

    `listed` holds each token's dictionary pronunciations, kept with the fit so that a word heard as one of
    them is not listed. None for a word that has no phones or cannot be placed.
    """
    units = {k: v for k, v in vocab.items() if k not in SPECIAL}
    names = {v: k for k, v in vocab.items()}
    phones = [[[p for t in w for p in form[t]] for w in words] for form in said]
    forms = [[ctc.tokenize(p, units) for p in form] for form in phones]
    out: list[dict[str, Any] | None] = []
    for w, fit in enumerate(gop.fit_words(log_probs, forms, vocab["<pad>"])):
        out.append(None if fit is None else {
            "score": round(fit.score, 3),
            "phones": [round(s, 3) for s in fit.phones],
            "forms": ["".join(form[w]).translate(ctc.STRESS) for form in phones],
            "expected": [names[u] for u in forms[fit.form][w]],
            "heard": [names[u] for u in fit.heard if names[u] not in SPECIAL],
            "listed": _joined(words[w], listed or {}),
        })  # fmt: skip
    return out


def _joined(tokens: Sequence[str], listed: Mapping[str, Sequence[str]]) -> list[str]:
    """A word's dictionary pronunciations over its tokens; none when a token has none."""
    if not tokens or any(t not in listed for t in tokens):
        return []
    return ["".join(choice) for choice in islice(product(*(listed[t] for t in tokens)), LISTED_MAX)]


def listed_forms(lines: Sequence[str]) -> dict[str, list[str]]:
    """Each word's pronunciations from its dictionary lines; a line with an unmapped phone is left out."""
    # Known limit: every listed form counts, so a word said without its last sound ("last" as "las") is not
    # listed when the dictionary has that form; revisit if the gate gets dropped-sound pairs.
    forms: dict[str, list[str]] = {}
    for line in lines:
        word, *_, arpa = line.split("\t")
        mapped = [ARPABET.get(p) or ARPABET.get(p.rstrip("012")) for p in arpa.split()]
        if mapped and None not in mapped:
            forms.setdefault(word, []).append("".join(p for p in mapped if p))
    return forms


def asr_times(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """The speech step's result with CTC word times, for a model that gives none (Canary).

    Each chunk's words are placed on the phone model's frames over that chunk. A word that cannot be
    placed keeps its chunk's span and the flag "untimed", so it is never mistaken for a timed word.
    """
    asr = read_json(job / "stages" / "asr.json")
    started = time.monotonic()
    component = catalogue.get(MODEL)
    folder = require(component, models_dir())
    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    units = {k: v for k, v in vocab.items() if k not in SPECIAL}
    model = session(folder / "onnx" / "model.onnx")
    said = pronounce(sorted({w["text"].lower() for w in asr["words"]}))
    tracks = asr.get("chunk_tracks")
    for k, (a, b) in enumerate(asr["chunks"]):
        words = [w for w in asr["words"] if w["chunk"] == k]
        wav = job / "input" / ("audio16k.wav" if tracks is None else f"track_{tracks[k]}.wav")
        audio, _ = sf.read(str(wav), start=a * RATE // 1000, stop=b * RATE // 1000, dtype="float32")
        word_units = [ctc.tokenize(said[w["text"].lower()], units) for w in words]
        times = ctc.word_times(scores(model, audio), word_units, vocab["<pad>"], a) if words else []
        last = a
        for w, t in zip(words, times or [None] * len(words), strict=True):
            if t is not None:
                w["start_ms"], w["end_ms"] = t
            else:
                w["flags"].append(UNTIMED)
                if times is not None:  # a word without phones among placed ones: no length, kept in order
                    w["start_ms"] = w["end_ms"] = last
            last = w["end_ms"]
        progress(100 * (k + 1) // len(asr["chunks"]))
    asr["words"].sort(key=lambda w: (w["start_ms"], w.get("track", 0)))
    asr["provenance"].update(
        word_times=True, timestamp_precision_ms=ctc.FRAME_MS,
        times={"method": "ctc", "model": MODEL, "revision": component.revision,
               "untimed": sum(UNTIMED in w["flags"] for w in asr["words"]),
               "wall_ms": int((time.monotonic() - started) * 1000)},
    )  # fmt: skip
    return asr


def dictionary() -> Path | None:
    """The US English pronunciation dictionary: its own download, or the copy an MFA install holds."""
    part = catalogue.get(DICTIONARY)
    (f,) = part.files_for(catalogue.SYSTEM)
    own = local_path(part, f, models_dir())
    found = [
        p for p in (own, models_dir() / "mfa" / mfa.DICTIONARY) if p.is_file() and p.stat().st_size == f.size
    ]
    return found[0] if found else None


def known(needed: list[str]) -> tuple[set[str] | None, dict[str, list[str]]]:
    """The words the pronunciation dictionary lacks, whose expected sounds are a guess (None without it), and
    the pronunciations it lists for the others."""
    path = dictionary()
    if path is None:
        return None, {}
    lines, missing = mfa.mini_dictionary(path, set(needed))
    return missing, listed_forms(lines)


def run(job: Path, progress: Callable[[int], None]) -> dict[str, Any]:
    """CTC word times for every candidate utterance, in the cuts MFA gets too, and how well each word fits.

    The word times need the CTC aligner, the fits need the pronunciation check; each is left out when off.
    """
    settings = read_json(job / "job.json")["settings"]
    on = settings.get("components", {})
    timed = on.get("ctc-aligner", True)
    checked = pronunciation.switched_on(settings)
    if not (on.get(MODEL, True) and on.get("espeak-ng", True) and (timed or checked)):
        return {"status": "disabled", "utterances": []}
    component = catalogue.get(MODEL)
    folder = require(component, models_dir())
    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    blank = vocab["<pad>"]
    units = {k: v for k, v in vocab.items() if k not in SPECIAL}
    model = session(folder / "onnx" / "model.onnx")
    cuts = [c for c in mfa.cuts(job) if any(c.tokens)]
    needed = sorted({t for c in cuts for w in c.tokens for t in w})
    said = [pronounce(needed, form) for form in (FORMS if checked else FORMS[:1])]
    unknown, listed = known(needed) if checked else (None, {})
    rows = []
    for n, c in enumerate(cuts):
        audio, _ = sf.read(
            str(job / "input" / c.wav), start=c.start_ms * RATE // 1000, stop=c.end_ms * RATE // 1000,
            dtype="float32",
        )  # fmt: skip
        log_probs = scores(model, audio) if len(audio) else None
        row: dict[str, Any] = {"id": c.id, "status": "off", "words": []}
        if timed:
            word_units = [ctc.tokenize([p for t in w for p in said[0][t]], units) for w in c.tokens]
            times = None if log_probs is None else ctc.word_times(log_probs, word_units, blank, c.start_ms)
            row["status"] = "failed" if times is None else "ok"
            row["words"] = [] if times is None else [None if t is None else list(t) for t in times]
        if checked:
            row["fits"] = [None] * len(c.tokens) if log_probs is None else (
                fits(log_probs, c.tokens, said, vocab, listed)
            )  # fmt: skip
            row["guessed"] = None if unknown is None else [any(t in unknown for t in w) for w in c.tokens]
        rows.append(row)
        progress(100 * (n + 1) // len(cuts))
    return {
        "status": "ok",
        "provenance": {
            "model": MODEL,
            "revision": component.revision,
            "device": "cpu",
            "engine": {
                name: metadata.version(name) for name in ("onnxruntime", "phonemizer", "espeakng-loader")
            },
            "language": "en-us",
            "forms": list(FORMS) if checked else None,
            "dictionary": None if unknown is None else mfa.DICTIONARY,
        },  # fmt: skip
        "utterances": rows,
    }
