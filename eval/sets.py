"""The experiments' data, each set made by the rule its pre-registration fixes.

EdAcc and Nyra rows, LibriSpeech utterances, the non-speech clips, and the composed sessions: set A as
committed, set B with every voice replaced, and the L2 interviews with an EdAcc speaker as the candidate.
Every session comes from the test suite's own builder, so its truth is made the same way as set A's.
Nothing here imports the app. Usage, from publishable/:
  python eval/sets.py kokoro-lines OUT.json        the lines set B needs from the synthetic voices
  python eval/sets.py set-b DATA SPLITS KOKORO OUT  set B's clips and recipes, prepared like set A's
"""

from __future__ import annotations

import io
import json
import re
import shutil
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "support"))
sys.path.insert(0, str(ROOT / "eval"))
import session_builder as sb  # noqa: E402  (eval/ is a folder of scripts, not a package)
from splits import candidate_clips, edacc_rows, eligible  # noqa: E402

ASR = tomllib.loads((ROOT / "eval" / "prereg" / "asr_bakeoff.toml").read_text("utf-8"))
SEED = ASR["seed"]
RATE = sb.SR
CLIPS = json.loads((sb.FIXTURES / "audio" / "clips.json").read_text(encoding="utf-8"))
VOICE = re.compile(r"synthetic voice (\S+)")
LIBRI = re.compile(r"LibriSpeech test-clean (\d+-\d+-\d+)")
TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>")
VOWELS = set("aeiouyæɑɒɔəɛɜɪʊʌɐɚɝɘɵɤɯœøɞʉɨᵻ")
HAND = {
    "area": 3,
    "museum": 3,
    "tedious": 3,
    "theatre": 3,
    "mm": 1,
    "hmm": 1,
}  # espeak merges or spells these
DEEP = 10 ** (-55 / 20)  # set A cut only silence frames under -55 dBFS, a few dB over its speaker's floor
# Set B's LibriSpeech speakers have noise floors up to about -50 dBFS, some of its synthetic voices breathe
# in their pauses, and Opus lowers frames near the -40 dBFS quiet line by up to 2 dB, joining short
# silences into refused ones; so set B's silences are cut up to -37 dBFS.
NOISY = 10 ** (-37 / 20)


# ---- rows of the public sets -------------------------------------------------------------------------


def parquet_rows(path: Path, columns: list[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            yield n, row
            n += 1


def edacc_items(folder: Path, ids: list[str]) -> dict[str, tuple[str, bytes]]:
    """The reference text and audio bytes of the named rows ("<shard file>:<row>")."""
    wanted: dict[str, set[int]] = {}
    for item in ids:
        name, row = item.rsplit(":", 1)
        wanted.setdefault(name, set()).add(int(row))
    return {
        f"{name}:{n}": (row["text"] or "", row["audio"]["bytes"])
        for name, numbers in wanted.items()
        for n, row in parquet_rows(folder / name, ["text", "audio"])
        if n in numbers
    }


def nyra_items(path: Path) -> list[dict[str, Any]]:
    """Every clip of one Nyra split: ID, the verbatim reference, audio bytes and seconds."""
    return [
        {"id": str(r["id"]), "text": r["verbatim_transcript"] or "", "audio": r["audio"]["bytes"],
         "seconds": float(r["duration_in_s"])}
        for _, r in parquet_rows(path, ["id", "verbatim_transcript", "audio", "duration_in_s"])
    ]  # fmt: skip


def libri_texts(root: Path) -> dict[str, str]:
    return {
        utt: text
        for trans in sorted((root / "test-clean").rglob("*.trans.txt"))
        for utt, text in (line.split(" ", 1) for line in trans.read_text(encoding="utf-8").splitlines())
    }


def libri_audio(root: Path, utt: str) -> np.ndarray:
    speaker, chapter, _ = utt.split("-")
    x, rate = sf.read(root / "test-clean" / speaker / chapter / f"{utt}.flac", dtype="float32")
    return resample(x, rate)


def joined(root: Path, utts: list[str], gap_ms: int) -> np.ndarray:
    """Utterances one after another with silence between them."""
    gap = np.zeros(gap_ms * RATE // 1000, dtype=np.float32)
    pieces = [libri_audio(root, u) for u in utts]
    return np.concatenate([p for x in pieces for p in (gap, x)][1:])


def non_speech(kind: str, seconds: int) -> np.ndarray:
    """Silence with a -60 dBFS noise floor, pink noise at -30 dBFS or the music bed at -20 dBFS (RMS)."""
    n = seconds * RATE
    rng = np.random.default_rng(SEED)
    if kind == "silence":
        return (rng.standard_normal(n) * 10 ** (-60 / 20)).astype(np.float32)
    if kind == "pink_noise":
        spectrum = np.fft.rfft(rng.standard_normal(n))
        spectrum[1:] /= np.sqrt(np.fft.rfftfreq(n, 1 / RATE)[1:])  # 1/f power
        spectrum[0] = 0
        x, level_db = np.fft.irfft(spectrum, n=n), -30
    elif kind == "music":
        x, level_db = sb.music(n, SEED), -20
    else:
        raise ValueError(f"unknown non-speech clip {kind!r}")
    return (x / np.sqrt(np.mean(x**2)) * 10 ** (level_db / 20)).astype(np.float32)


# ---- clips prepared as set A's were -------------------------------------------------------------------


def resample(x: np.ndarray, rate: int) -> np.ndarray:
    """Mono at 16 kHz, with the resampler the fixture clips were made with."""
    x = x.mean(axis=1) if x.ndim == 2 else x  # noqa: PLR2004
    if rate == RATE:
        return x.astype(np.float32)
    res = av.AudioResampler(format="flt", layout="mono", rate=RATE)
    frame = av.AudioFrame.from_ndarray(x.reshape(1, -1).astype(np.float32), format="flt", layout="mono")
    frame.sample_rate = rate
    out = [f.to_ndarray().reshape(-1) for f in [*res.resample(frame), *res.resample(None)]]
    return np.concatenate(out).astype(np.float32)


def level(x: np.ndarray) -> np.ndarray:
    """A quiet clip raised to a -20 dBFS active level; never turned down except to keep the peak at -6."""
    n = len(x) // sb.FRAME
    rms = np.sqrt((x[: n * sb.FRAME].reshape(n, sb.FRAME).astype(np.float64) ** 2).mean(axis=1))
    active = rms >= rms.max() * 10 ** (-30 / 20)
    gain = max(1.0, 10 ** (-20 / 20) / np.sqrt(np.mean(rms[active] ** 2)))
    return (x * min(gain, 10 ** (-6 / 20) / np.max(np.abs(x)))).astype(np.float32)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def tidy(x: np.ndarray, deep: float = DEEP) -> np.ndarray:
    """Loud fragments under 50 ms cut off by 150 ms of quiet at either end dropped; inner silences shortened.

    Inside a quiet run (frames under -40 dBFS, or under `deep` when that is higher) longer than 100 ms,
    frames under `deep` are removed from the middle of the run until it lasts 100 ms.
    """
    n = len(x) // sb.FRAME
    x = x[: n * sb.FRAME].copy()
    loud = runs(sb.frame_peaks(x) >= sb.LOUD)
    short, apart = 5, 15  # frames: a fragment under 50 ms, after 150 ms of quiet
    while len(loud) > 1 and loud[0][1] - loud[0][0] < short and loud[1][0] - loud[0][1] >= apart:
        x[: loud[0][1] * sb.FRAME] = 0
        loud = loud[1:]
    while len(loud) > 1 and loud[-1][1] - loud[-1][0] < short and loud[-1][0] - loud[-2][1] >= apart:
        x[loud[-1][0] * sb.FRAME :] = 0
        loud = loud[:-1]
    peaks = sb.frame_peaks(x)
    first, last = loud[0][0], loud[-1][1]
    keep = np.ones(n, dtype=bool)
    for a, b in runs(peaks[first:last] < max(sb.LOUD, deep)):
        a, b = a + first, b + first
        cut = sorted((i for i in range(a, b) if peaks[i] < deep), key=lambda i: abs(i - (a + b) / 2))
        for i in cut[: max(0, (b - a) - 10)]:
            keep[i] = False
    return np.asarray(x.reshape(n, sb.FRAME)[keep].reshape(-1))


def margin(x: np.ndarray) -> np.ndarray:
    """100 ms kept around the speech (frames at -50 dBFS or more)."""
    loud = np.flatnonzero(sb.frame_peaks(x) >= 10 ** (-50 / 20))
    return x[max(0, (int(loud[0]) - 10) * sb.FRAME) : min(len(x), (int(loud[-1]) + 11) * sb.FRAME)]


def prepare(x: np.ndarray, path: Path, name: str, tidied: bool = True, deep: float = DEEP) -> bool:
    """Write the clip as set A's were made (level, shortened silences, margin, -75 dBFS dither, Opus).

    True when the builder accepts it with 30 ms to spare under its 150 ms inner-quiet limit; else the file
    is removed. `tidied=False` keeps a recording's own pauses, for clips whose inner silences are not cut.
    """
    y = margin(tidy(level(x), deep) if tidied else level(x))
    y = (y + np.random.default_rng(len(name)).standard_normal(len(y)) * 10 ** (-75 / 20)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, RATE, format="OGG", subtype="OPUS")
    if not tidied:
        return True
    try:
        quiet = runs(sb.frame_peaks(sb.trim(sb.load_clip(path), name)) < sb.LOUD)
    except ValueError:
        quiet = [(0, sb.MAX_QUIET)]
    if max((b - a for a, b in quiet), default=0) <= sb.MAX_QUIET - 3:
        return True
    path.unlink()
    return False


def syllables(text: str) -> int:
    """Syllables from espeak-ng's phones (US English), one word at a time, as set A's were counted."""
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer.separator import Separator

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    words = re.findall(r"[a-z']+", TAGS.sub(" ", text.lower()))
    if not words:
        return 0
    phones = EspeakBackend("en-us").phonemize(words, separator=Separator(phone=" ", word=None), strip=True)
    return sum(
        HAND.get(w, sum(1 for p in ph.split() if set(p) & VOWELS))
        for w, ph in zip(words, phones, strict=True)
    )


# ---- composed sessions ---------------------------------------------------------------------------------


def recipes() -> list[dict[str, Any]]:
    """The six committed recipes, with every base and edit applied."""
    return [sb.load_recipe(p) for p in sorted(sb.SESSIONS.glob("*.json"))]


def used_clips(recipe: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(e["clip"] for e in recipe["events"]))


def kokoro_lines() -> list[dict[str, str]]:
    """Every synthetic-voice line of the six recipes, with set B's voice for it and the same text."""
    voices = ASR["data"]["composed"]["set_b_voices"]
    found = {c: VOICE.match(CLIPS[c]["source"]) for r in recipes() for c in used_clips(r)}
    return [
        {"name": c, "voice": voices[m.group(1)], "text": CLIPS[c]["text"]}
        for c, m in sorted(found.items())
        if m is not None
    ]


def closest(seconds: dict[str, float], speaker: str, used: set[str], target: float) -> list[str]:
    """The speaker's unused utterances, the closest in duration to the target first (ties by ID)."""
    own = (u for u in seconds if u.split("-")[0] == speaker and u not in used)
    return sorted(own, key=lambda u: (abs(seconds[u] - target), u))


def set_b(data: Path, splits: dict[str, Any], kokoro: Path, out: Path) -> dict[str, Any]:
    """Set B's clips and recipes: each synthetic line in its new voice, each LibriSpeech clip replaced by
    the recipe's new speaker's unused utterance closest in duration (ties by ID), passed over when the
    builder refuses it; seeds + 1000. Returns which utterance replaced which clip."""
    root = data / "librispeech" / "LibriSpeech"
    texts = libri_texts(root)
    seconds = {p.stem: sf.info(p).duration for p in sorted((root / "test-clean").rglob("*.flac"))}
    lines = {x["name"]: x["voice"] for x in kokoro_lines()}
    report: dict[str, Any] = {}
    for recipe in recipes():
        name, folder = recipe["name"], out / recipe["name"]
        speaker, used, meta = splits["set_b_speakers"].get(name), set(), {}
        for clip in used_clips(recipe):
            if clip in lines:
                x, rate = sf.read(kokoro / f"{clip}.wav", dtype="float32")
                if not prepare(resample(x, rate), folder / "audio" / f"{clip}.opus", clip, deep=NOISY):
                    raise ValueError(f"{name}: the builder refuses {clip} in its new voice")
                meta[clip] = {"text": CLIPS[clip]["text"], "source": f"synthetic voice {lines[clip]}",
                              "licence": "Apache-2.0"}  # fmt: skip
                continue
            old = LIBRI.match(CLIPS[clip]["source"]).group(1)  # type: ignore[union-attr]
            path = folder / "audio" / f"{clip}.opus"
            choices = closest(seconds, speaker, used, seconds[old])
            utt = next((u for u in choices if prepare(libri_audio(root, u), path, clip, deep=NOISY)), None)
            if utt is None:
                raise ValueError(f"{name}: no utterance of speaker {speaker} replaces {clip}")
            used.add(utt)
            report.setdefault(name, {})[clip] = utt
            meta[clip] = {
                "text": texts[utt],
                "source": f"LibriSpeech test-clean {utt}",
                "licence": "CC-BY-4.0",
            }
        for m in meta.values():
            m["syllables"] = syllables(m["text"])
        (folder / "audio" / "clips.json").write_text(
            json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8"
        )
        b = {**recipe, "name": f"{name}-B", "seed": recipe["seed"] + 1000}
        (folder / "recipe.json").write_text(json.dumps(b, indent=1), encoding="utf-8")
    return report


def build_b(prepared: Path, name: str, out: Path) -> dict[str, Any]:
    recipe = json.loads((prepared / name / "recipe.json").read_text(encoding="utf-8"))
    return sb.build_recipe(recipe, out, prepared / name / "audio")


def l2_rows(data: Path, split: str, speakers: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Each speaker's eligible EdAcc rows in the split, as the speech-model comparison defines them."""
    spec = ASR["data"]["edacc"]
    rows = [r for s in spec[f"{split}_shards"] for r in edacc_rows(data / "edacc" / Path(s["file"]).name,
                                                                    spec["not_scored"])]  # fmt: skip
    return {s: [r for r in rows if r["speaker"] == s and eligible(r, spec)] for s in speakers}


def l2_interview(data: Path, rows: list[dict[str, Any]], name: str, out: Path) -> dict[str, Any]:
    """CS-INT with each candidate clip replaced by the speaker's unused row closest in duration (ties by ID).

    The rows keep their own pauses: only their outer edges are truth, so the builder's inner-quiet limit,
    made for clips whose pauses are truth, is lifted while they are placed.
    """
    recipe = sb.load_recipe(sb.SESSIONS / "CS-INT.json")
    recipe["name"] = name
    folder, meta, chosen = out / "audio", {}, {}
    folder.mkdir(parents=True, exist_ok=True)
    replaced = set(candidate_clips(recipe))
    available = list(rows)
    for clip in used_clips(recipe):
        if clip not in replaced:
            shutil.copyfile(sb.FIXTURES / "audio" / f"{clip}.opus", folder / f"{clip}.opus")
            meta[clip] = CLIPS[clip]
            continue
        target = len(sb.trim(sb.load_clip(sb.FIXTURES / "audio" / f"{clip}.opus"), clip)) / RATE
        row = min(available, key=lambda r: (abs(r["seconds"] - target), r["id"]))
        available.remove(row)
        chosen[clip] = row["id"]
    found = edacc_items(data / "edacc", list(chosen.values()))
    for clip, item in chosen.items():
        text, audio = found[item]
        x, rate = sf.read(io.BytesIO(audio), dtype="float32")
        prepare(resample(x, rate), folder / f"{clip}.opus", clip, tidied=False)
        meta[clip] = {"text": text, "syllables": syllables(text), "source": f"EdAcc {item}",
                      "licence": "CC-BY-SA-4.0"}  # fmt: skip
    (folder / "clips.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    quiet, sb.MAX_QUIET = sb.MAX_QUIET, 10**9
    try:
        truth = sb.build_recipe(recipe, out, folder)
    finally:
        sb.MAX_QUIET = quiet
    truth["rows"] = chosen
    (out / "truth.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")
    return truth


def hesitations(truth: dict[str, Any]) -> list[tuple[int, int]]:
    """The fillers and repeated lead-in words spliced into CS-HES, as (start, end) in ms."""
    return [(s["start_ms"], s["end_ms"]) for s in truth["segments"] if str(s["id"]).startswith("hes_")]


def main(argv: list[str]) -> int:
    if argv[:1] == ["kokoro-lines"]:
        Path(argv[1]).write_text(json.dumps(kokoro_lines(), indent=1, ensure_ascii=False), encoding="utf-8")
        return 0
    if argv[:1] == ["set-b"]:
        data, splits, kokoro, out = (Path(a) for a in argv[1:5])
        report = set_b(data, json.loads(splits.read_text(encoding="utf-8")), kokoro, out)
        (out / "replaced.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
