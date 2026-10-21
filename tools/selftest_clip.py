"""Build the self-test clip in src/fluentfolio/assets/selftest/ from two LibriSpeech test-clean speakers.

Usage, from the repository root:
  uv run --locked python tools/selftest_clip.py <LibriSpeech test-clean folder> [<out folder>]

The clip is a short two-person exchange laid out by the test session builder, so its truth is exact: who
speaks when, and each utterance's text. The script never imports fluentfolio. The candidate's median
pitch is measured here with Praat (through Parselmouth) on the candidate's own stretches of the encoded
clip, in two passes as the product does; Praat is deterministic, so the self-test can hold the installed
Praat to this value.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "support"))
import session_builder as sb  # noqa: E402

OUT = ROOT / "src" / "fluentfolio" / "assets" / "selftest"
EXAMINER, CANDIDATE = "1089", "121"  # a male and a female reader, so their voices are easy to tell apart
ASKS_S, ANSWERS_S = (2.5, 4.5), (5.0, 10.0)  # the lengths of the utterances each speaker gets
# Two answers, each of several utterances: the pitch measures use only answers with 10 s of voiced speech.
ANSWERS, ANSWER_MS = 2, 19000
# A pause of over 2 s inside an answer starts a new speech-model chunk, so each read sentence is one
# utterance for the aligner, as in the sessions the alignment check was tuned on.
ASK_GAP_MS, ANSWER_GAP_MS, WITHIN_GAP_MS = 800, 700, 2200
LEVEL_DBFS, PEAK_DBFS, DITHER_DBFS = -20, -6, -75
MAX_BYTES = 1_500_000
FIRST_PASS_HZ = (60.0, 700.0)


def level(x: np.ndarray) -> np.ndarray:
    """Raise the active speech to -20 dBFS, keeping the peak at -6 dBFS."""
    n = len(x) // sb.FRAME
    rms = np.sqrt((x[: n * sb.FRAME].reshape(n, sb.FRAME).astype(np.float64) ** 2).mean(axis=1))
    active = rms >= rms.max() * 10 ** (-30 / 20)
    gain = 10 ** (LEVEL_DBFS / 20) / np.sqrt(np.mean(rms[active] ** 2))
    return (x * min(gain, 10 ** (PEAK_DBFS / 20) / np.max(np.abs(x)))).astype(np.float32)


def shorten_silences(x: np.ndarray) -> np.ndarray:
    """Cut every inner quiet run (frames under -40 dBFS) down to 100 ms, so the builder accepts the clip."""
    n = len(x) // sb.FRAME
    frames = x[: n * sb.FRAME].reshape(n, sb.FRAME)
    quiet = sb.frame_peaks(x[: n * sb.FRAME]) < sb.LOUD
    loud = np.flatnonzero(~quiet)
    keep = np.zeros(n, dtype=bool)
    keep[loud[0] : loud[-1] + 1] = True
    run = 0
    for i in range(loud[0], loud[-1] + 1):
        run = run + 1 if quiet[i] else 0
        keep[i] = run <= 10  # noqa: PLR2004 - 10 frames of 10 ms
    return frames[keep].reshape(-1)


def utterances(folder: Path, speaker: str, seconds: tuple[float, float]) -> list[tuple[str, Path, str]]:
    """The speaker's utterances whose length lies in `seconds`, in corpus order: (ID, file, text)."""
    found = []
    for trans in sorted((folder / speaker).rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt, text = line.split(" ", 1)
            path = trans.parent / f"{utt}.flac"
            if seconds[0] <= sf.info(str(path)).duration <= seconds[1]:
                found.append((utt, path, text))
    return found


def prepare(found: list[tuple[str, Path, str]], clips: Path, enough: int, by_length: bool) -> list[str]:
    """Write utterances the builder accepts as 16 kHz Opus clips until there are `enough`; return their names.

    `enough` counts clips, or with `by_length` the milliseconds of speech after the builder's trim.
    """
    names: list[str] = []
    have = 0
    meta = (
        json.loads((clips / "clips.json").read_text(encoding="utf-8"))
        if (clips / "clips.json").exists()
        else {}
    )
    for utt, path, text in found:
        x, rate = sf.read(str(path), dtype="float32")
        if rate != sb.SR:
            raise ValueError(f"{utt}: {rate} Hz, expected {sb.SR}")
        y = shorten_silences(level(x))
        y = y + np.random.default_rng(len(meta)).standard_normal(len(y)).astype(np.float32) * 10 ** (
            DITHER_DBFS / 20
        )
        name = f"ls_{utt.replace('-', '_')}"
        sf.write(clips / f"{name}.opus", y, sb.SR, format="OGG", subtype="OPUS")
        try:
            kept = sb.trim(sb.load_clip(clips / f"{name}.opus"), name)
        except ValueError as exc:
            print(f"refused {utt}: {exc}")
            (clips / f"{name}.opus").unlink()
            continue
        meta[name] = {"text": text, "syllables": 0, "source": f"LibriSpeech test-clean {utt}"}
        names.append(name)
        have += len(kept) * 1000 // sb.SR if by_length else 1
        if have >= enough:
            break
    (clips / "clips.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if have < enough:
        raise ValueError(f"only {have} of {enough} {'ms' if by_length else 'utterances'} fit")
    return names


def median_f0(sound: parselmouth.Sound) -> float:
    """Two passes: a wide range first, then 0.75 of its lower quartile to 1.5 of its upper quartile."""
    first = sound.to_pitch(time_step=0.01, pitch_floor=FIRST_PASS_HZ[0], pitch_ceiling=FIRST_PASS_HZ[1])
    hz = first.selected_array["frequency"]
    q25, q75 = np.percentile(hz[hz > 0], [25, 75])
    second = sound.to_pitch(time_step=0.01, pitch_floor=0.75 * q25, pitch_ceiling=1.5 * q75)
    hz = second.selected_array["frequency"]
    return float(np.median(hz[hz > 0]))


def main(corpus: Path, out: Path) -> int:
    with tempfile.TemporaryDirectory() as folder:
        return build(corpus, out, Path(folder))


def build(corpus: Path, out: Path, work: Path) -> int:
    (work / "clips").mkdir(parents=True)
    asks = prepare(utterances(corpus, EXAMINER, ASKS_S), work / "clips", ANSWERS, by_length=False)
    turns, events = {}, []
    found = utterances(corpus, CANDIDATE, ANSWERS_S)
    for k, ask in enumerate(asks):
        answer = prepare(found, work / "clips", ANSWER_MS, by_length=True)
        found = found[[f"ls_{u.replace('-', '_')}" for u, _, _ in found].index(answer[-1]) + 1 :]
        turns[f"q{k}"] = {"speaker": "E", "kind": "question", "part": 1}
        turns[f"a{k}"] = {"speaker": "C", "kind": "answer", "part": 1}
        events.append({"clip": ask, "turn": f"q{k}", "gap_ms": ASK_GAP_MS if k else 500})
        events += [{"clip": c, "turn": f"a{k}", "gap_ms": WITHIN_GAP_MS if i else ANSWER_GAP_MS}
                   for i, c in enumerate(answer)]  # fmt: skip
    recipe = {"name": "selftest", "mode": "interview", "speakers": ["E", "C"], "seed": 60,
              "turns": turns, "events": events}  # fmt: skip
    truth = sb.build_recipe(recipe, work / "session", work / "clips")
    out.mkdir(parents=True, exist_ok=True)
    audio, _ = sf.read(str(work / "session" / "session.wav"), dtype="float32")
    sf.write(out / "selftest.opus", audio, sb.SR, format="OGG", subtype="OPUS")
    size = (out / "selftest.opus").stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"selftest.opus is {size} bytes, more than {MAX_BYTES}")
    decoded = sb.load_clip(out / "selftest.opus")
    segments = [
        {"speaker": s["speaker"], "role": s["role"], "turn": s["turn"], "start_ms": s["start_ms"],
         "end_ms": s["end_ms"], "text": s["text"], "source": s["source"]}
        for s in truth["segments"]
    ]  # fmt: skip
    own = [decoded[s["start_ms"] * 16 : s["end_ms"] * 16] for s in segments if s["role"] == "candidate"]
    f0 = median_f0(parselmouth.Sound(np.concatenate(own).astype(np.float64), sampling_frequency=sb.SR))
    shipped = {
        "duration_ms": truth["duration_ms"],
        "speakers": {"E": "examiner", "C": "candidate"},
        "speech_ms": sum(s["end_ms"] - s["start_ms"] for s in segments),
        "candidate_median_f0_hz": round(f0, 2),
        "segments": segments,
    }
    # LF on every system, as git keeps these files, so a rebuild on Windows matches them byte for byte.
    (out / "truth.json").write_text(json.dumps(shipped, indent=1) + "\n", encoding="utf-8", newline="\n")
    reference = "\n".join(s["text"] for s in segments) + "\n"
    (out / "reference.txt").write_text(reference, encoding="utf-8", newline="\n")
    answers = {}
    for s in segments:
        answers[s["turn"]] = answers.get(s["turn"], 0) + s["end_ms"] - s["start_ms"]
    print(f"selftest.opus: {size} bytes, {truth['duration_ms']} ms, {len(segments)} utterances, "
          f"turn lengths (ms) {answers}, candidate median F0 {f0:.2f} Hz")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else OUT))  # noqa: PLR2004
