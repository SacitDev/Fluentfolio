"""The pitch-tool agreement check, run as eval/prereg/pitch_fusion.toml says.

Usage, from publishable/ (DATA holds librispeech/ and voices/ as tools/fetch_test_data.py leaves them; WORK
keeps the built sessions, the jobs and each tool's output; OUT gets the results):
  python eval/pitch_fusion.py --data DATA --splits FILE --freeze FILE --work WORK --out OUT
Options: --previous DIR (the OUT of a first run, after the one allowed fix of the inputs: its label shares
are kept beside the new ones); --limit N (the first N clean clips and composed sessions: a check run only).
Each tone and clean clip is one answer of a job whose candidate audio is the whole input; Praat, SwiftF0
and openSMILE (under a test consent record in WORK's own data folder) each run on it in a process of their
own. The six composed sessions go through the app's command line, and each tool's values are read from the
job. Spreads and labels are computed here by the fusion rule, never taken from the app's own fusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL))
import sets  # noqa: E402  (eval/ is a folder of scripts, not a package)
import stats  # noqa: E402

Row = dict[str, Any]
PREREG_PATH = EVAL / "prereg" / "pitch_fusion.toml"
PREREG_FILE = "eval/prereg/pitch_fusion.toml"
PREREG = tomllib.loads(PREREG_PATH.read_text(encoding="utf-8"))
TOOLS = tuple(PREREG["tools"]["tools"])
AGREE, APPROXIMATE = PREREG["tools"]["agree_st"], PREREG["tools"]["approximate_st"]
REF_HZ = 27.5  # semitones re 27.5 Hz, the fusion rule's unit
MIN_VOICED_MS = 5000  # a tool with less voiced speech is dropped (the fusion rule)
ORDER = ("agree", "approximate", "unreliable")  # worse to the right
SHARES = (*ORDER, "too_little_speech", "single_tool")
RATE = 16000
HARMONICS = 8
CONSENT = {"opensmile": {"utc": "2026-10-13T00:00:00Z", "source": "cli-tty"}}  # test consent, own folder
TIMEOUT_S = 1800  # one tool on one input, or one session's analysis


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semitones(hz: float) -> float:
    return 12 * math.log2(hz / REF_HZ)


def hertz(st: float) -> float:
    return REF_HZ * 2 ** (st / 12)


# ---- inputs ----------------------------------------------------------------------------------------------


def tone(hz: float, seconds: float) -> np.ndarray:
    """A harmonic tone: the first 8 harmonics at 1/k amplitude, peak 0.3 of full scale."""
    t = np.arange(int(RATE * seconds)) / RATE
    x = sum(np.sin(2 * np.pi * k * hz * t) / k for k in range(1, HARMONICS + 1))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def clip_job(job: Path, audio: np.ndarray) -> Path:
    """A job whose one candidate has one answer, the whole input, as the candidate audio step leaves it."""
    ms = len(audio) * 1000 // RATE
    audio = audio[: ms * RATE // 1000]
    (job / "input").mkdir(parents=True, exist_ok=True)
    sf.write(job / "input" / "audio16k.wav", audio, RATE, subtype="PCM_16")
    write(
        job / "input" / "candidate.map.json",
        {"rate": RATE, "pieces": [{"wav_ms": 0, "orig_ms": 0, "ms": ms}]},
    )
    write(job / "job.json", {"settings": {}})
    write(job / "stages" / "media.json", {"status": "ok", "quality": {"class": "clean"}, "music_spans": []})
    write(job / "stages" / "structure.json", {
        "candidates": [{"id": "candidate", "speaker": "S1"}], "speech": {"candidate": [[0, ms]]},
        "answers": [{"id": "A01", "candidate": "candidate", "part": None, "start_ms": 0, "end_ms": ms}],
        "utterances": [{"id": "U001", "answer_id": "A01", "start_ms": 0, "end_ms": ms}],
    })  # fmt: skip
    made = {"source": "audio16k.wav", "speech_ms": ms, "pieces": 1, "wav": "input/audio16k.wav",
            "map": "input/candidate.map.json"}  # fmt: skip
    write(job / "stages" / "candidate_audio.json", {"status": "ok", "candidates": {"candidate": made}})
    return job


def clean_clips(data: Path, splits: Row) -> dict[str, np.ndarray]:
    """The 19 LibriSpeech clips of the selection, joined with 300 ms gaps, and the VOiCES clip, in order."""
    root = data / "librispeech" / "LibriSpeech"
    gap = PREREG["data"]["gap_ms"]
    out = {f"libri-{s}": sets.joined(root, utts, gap) for s, utts in splits["pitch_clips"].items()}
    voices = data / "voices" / Path(PREREG["data"]["voices_source"]).name
    if sha256(voices) != PREREG["data"]["voices_sha256"]:
        raise SystemExit(f"{voices} does not have the pre-registered SHA-256")
    x, rate = sf.read(voices, dtype="float32")
    out["voices"] = sets.resample(x, rate)
    return out


# ---- the tools -------------------------------------------------------------------------------------------


def run_tools(job: Path, home: Path) -> dict[str, Row]:
    """Each tool's step result on the job, each in its own process; a result already saved is reused."""
    env = {**os.environ, "FLUENTFOLIO_HOME": str(home), "CUDA_VISIBLE_DEVICES": "-1"}
    out = {}
    for tool in TOOLS:
        path = job / "analyzers" / f"{tool}.json"
        if not path.is_file():
            argv = [sys.executable, str(EVAL / "pitch_tool.py"), tool, str(job)]
            done = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
            if done.returncode != 0:
                raise RuntimeError(
                    f"{tool} failed on {job} (exit code {done.returncode}): {done.stderr[-2000:]}"
                )
        out[tool] = read(path)
    return out


def tool_values(results: dict[str, Row], candidate: str) -> dict[str, Row]:
    """Each tool's p20, p50 and p80 in semitones and its voiced time over all of a candidate's answers."""
    out = {}
    for tool, found in results.items():
        scope = found.get("candidates", {}).get(candidate, {}).get("scopes", {}).get("all")
        if found["status"] != "ok" or scope is None:
            out[tool] = {"p20": None, "p50": None, "p80": None, "voiced_ms": 0,
                         "why": found.get("code") or found["status"]}  # fmt: skip
            continue
        why = scope["missing"]["code"] if scope["missing"] else None
        out[tool] = {"p20": scope.get("p20_st"), "p50": scope.get("p50_st"), "p80": scope.get("p80_st"),
                     "voiced_ms": scope["voiced_ms"], "why": why}  # fmt: skip
    return out


# ---- the fusion rule's labels ----------------------------------------------------------------------------


def label(spread: float) -> str:
    return "agree" if spread <= AGREE else "approximate" if spread <= APPROXIMATE else "unreliable"


def worse(a: str, b: str) -> str:
    return max(a, b, key=ORDER.index)


def judge(values: dict[str, Row]) -> Row:
    """The spreads (largest minus smallest tool value) and the labels of one candidate or clip.

    A tool without a value or with under 5 s of voiced speech is dropped; the range takes the worse of the
    p20 and p80 labels, and the clip label is the worse of the median's and the range's.
    """
    used = [t for t, v in values.items() if v["p50"] is not None and v["voiced_ms"] >= MIN_VOICED_MS]
    row: Row = {"tools": used, "spread_p20": None, "spread_p50": None, "spread_p80": None,
                "spread_range": None, "median_label": None, "range_label": None}  # fmt: skip
    if len(used) < 2:
        return row | {"label": "single_tool" if used else "too_little_speech"}
    for stat in ("p20", "p50", "p80"):
        found = [values[t][stat] for t in used]
        row[f"spread_{stat}"] = max(found) - min(found)
    ranges = [values[t]["p80"] - values[t]["p20"] for t in used]
    row["spread_range"] = max(ranges) - min(ranges)
    row["median_label"] = label(row["spread_p50"])
    row["range_label"] = worse(label(row["spread_p20"]), label(row["spread_p80"]))
    return row | {"label": worse(row["median_label"], row["range_label"])}


def worse_share(labels: list[str]) -> float:
    """The share of approximate or worse among the labels with a combined value; NaN when there are none."""
    combined = [x for x in labels if x in ORDER]
    return sum(x != "agree" for x in combined) / len(combined) if combined else float("nan")


def tone_rows(results: dict[str, dict[str, Row]]) -> list[Row]:
    """Per tone and tool: the median against the true pitch, in semitones."""
    rows = []
    for name, per_tool in results.items():
        true = float(name.removeprefix("tone-"))
        values = tool_values(per_tool, "candidate")
        for tool, v in values.items():
            heard = v["p50"]
            rows.append({"tone_hz": true, "tool": tool, "median_hz": None if heard is None else hertz(heard),
                         "error_st": None if heard is None else heard - semitones(true),
                         "label": judge(values)["label"]})  # fmt: skip
    return rows


def calibration(rows: list[Row]) -> tuple[bool, str]:
    """Every tool's median on every tone within 0.3 st of the true pitch."""
    most = PREREG["calibration"]["tone_st_max"]
    errors = [abs(r["error_st"]) if r["error_st"] is not None else math.inf for r in rows]
    worst = max(errors) if errors else math.inf
    ok = worst <= most
    return ok, (f"calibration: {len(rows)} tool medians on {len({r['tone_hz'] for r in rows})} tones, the "
                f"largest error {worst:.3f} st (at most {most}): {'holds' if ok else 'FAILS'}")  # fmt: skip


# ---- composed sessions -----------------------------------------------------------------------------------


def composed(work: Path, home: Path, limit: int | None) -> list[Row]:
    """The candidates of the committed recipes, each session analysed by the app's command line."""
    rows = []
    for recipe in sorted(sets.sb.SESSIONS.glob("*.json"))[:limit]:
        folder = work / "sessions" / recipe.stem
        if not (folder / "truth.json").is_file():
            sets.sb.build(recipe, folder)
        job = analysed(folder, work / "analysis" / f"{recipe.stem}.json", home)
        results = {tool: read(job / "analyzers" / f"{tool}.json") for tool in TOOLS}
        candidates = sorted({c for found in results.values() for c in found.get("candidates", {})})
        for candidate in candidates:
            values = tool_values(results, candidate)
            rows.append({"set": "composed", "item": recipe.stem, "candidate": candidate, "seconds": None,
                         **judge(values), **flat(values)})  # fmt: skip
    return rows


def analysed(folder: Path, saved: Path, home: Path) -> Path:
    """The app's job folder for the session, analysed on the CPU with the default settings."""
    if saved.is_file():
        return Path(read(saved)["job"])
    mode = read(folder / "truth.json")["mode"]
    argv = [sys.executable, "-m", "fluentfolio", "analyze", str(folder / "session.wav"), "--mode", mode,
            "--device", "cpu", "--json"]  # fmt: skip
    if mode != "solo":
        argv.append("--partner-consent")  # the composed voices stand for people who agreed
    env = {**os.environ, "FLUENTFOLIO_HOME": str(home), "CUDA_VISIBLE_DEVICES": "-1"}
    done = subprocess.run(argv, env=env, capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT_S)
    lines = [line for line in done.stdout.splitlines() if line.startswith('{"schema"')]
    envelope = json.loads(lines[-1]) if lines else {"ok": False}
    if done.returncode != 0 or not envelope["ok"]:
        raise RuntimeError(f"the analysis of {folder.name} failed (exit code {done.returncode})")
    job = envelope["data"]["outputs"]["job_dir"]
    write(saved, {"job": job})
    return Path(job)


def flat(values: dict[str, Row]) -> Row:
    return {f"{t}_{k}": v[k] for t, v in values.items() for k in ("p20", "p50", "p80", "voiced_ms", "why")}


# ---- the report ------------------------------------------------------------------------------------------


COLUMNS = ["set", "item", "candidate", "seconds", "label", "median_label", "range_label", "spread_p20",
           "spread_p50", "spread_p80", "spread_range", "tools",
           *[f"{t}_{k}" for t in TOOLS for k in ("p20", "p50", "p80", "voiced_ms", "why")]]  # fmt: skip


def write_rows(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({**r, "tools": "+".join(r["tools"])} for r in rows)


def shares(rows: list[Row]) -> Row:
    """Each label's count, and the approximate-or-worse share with its 95 % bootstrap interval over the
    clips it is taken over: those with a combined label."""
    counts = Counter(r["label"] for r in rows)
    combined = [r["label"] for r in rows if r["label"] in ORDER]
    method = PREREG["statistics"]
    found: Row = {"clips": len(rows), "counts": {x: counts[x] for x in SHARES}, "combined": len(combined),
                  "worse": sum(x != "agree" for x in combined), "share": None, "interval": None}  # fmt: skip
    if combined:
        found["share"] = worse_share(combined)
        draw = stats.bootstrap(combined, worse_share, method["resamples"], method["seed"])
        found["interval"] = [draw[1], draw[2]]
    return found


def change(rows: list[Row], before: dict[str, str]) -> list[float] | None:
    """This run's share minus the first run's, with the paired interval over the same clips: those with a
    combined label in both runs."""
    now = {r["item"]: r["label"] for r in rows}
    items = [i for i, x in now.items() if x in ORDER and before.get(i) in ORDER]
    if not items:
        return None
    method = PREREG["statistics"]

    def first(chosen: list[str]) -> float:
        return worse_share([now[i] for i in chosen])

    def second(chosen: list[str]) -> float:
        return worse_share([before[i] for i in chosen])

    return list(stats.paired(items, first, second, method["resamples"], method["seed"]))


def decide(calibrated: bool, found: Row, previous: bool) -> str:
    limit = PREREG["decision"]["limit_share"]
    if not calibrated:
        return ("Decision: none: the harness calibration is outside its range, so no decision is taken until "
                "the cause is found")  # fmt: skip
    if found["share"] is None:
        return "Decision: none: no clean clip has a combined label"
    told = (f"{found['worse']} of {found['combined']} clean clips with a combined label "
            f"({100 * found['share']:.1f} %) are approximate or worse, against at most "
            f"{100 * limit:.0f} %")  # fmt: skip
    if found["share"] <= limit:
        return f"Decision: within the limit: {told}; the agreement limits stay as they are"
    if previous:
        return (f"Decision: still over the limit after the one fix: {told}; the limits stay and the accuracy "
                "page says so")  # fmt: skip
    return (f"Decision: over the limit: {told}; fix the inputs once (range settings or voicing input) and "
            "run again")  # fmt: skip


def fmt(x: Any) -> str:
    return "n/a" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def report(found: Row, tones: list[Row], clean: list[Row], mixed: list[Row], limit: int | None) -> list[str]:
    lines = [f"prereg_sha256: {found['prereg_sha256']}", found["decision"], "", "# Pitch-tool agreement", ""]
    if limit is not None:
        lines += [
            f"A small check run on the first {limit} clean clips and composed sessions: not a result.",
            "",
        ]
    s = found["clean"]
    interval = "n/a" if s["interval"] is None else f"[{s['interval'][0]:.3f}, {s['interval'][1]:.3f}]"
    lines += [f"- {found['calibration']}",
              f"- clean clips: {s['clips']}; labels " + ", ".join(f"{k} {v}" for k, v in s["counts"].items()),
              f"- approximate or worse: {s['worse']} of {s['combined']} with a combined label, share "
              f"{fmt(s['share'])}, 95 % interval over clips {interval}"]  # fmt: skip
    if found.get("previous"):
        p, d = found["previous"], found["change"]
        lines.append(f"- first run: labels {p['counts']}, share {fmt(p['share'])}; change "
                     + ("n/a" if d is None else f"{d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]"))  # fmt: skip
    lines += ["", "## Tones (true pitch against each tool's median)", "",
              "| tone Hz | tool | median Hz | error st | label |", "|---|---|---|---|---|"]  # fmt: skip
    shown = ("tone_hz", "tool", "median_hz", "error_st", "label")
    lines += ["| " + " | ".join(fmt(r[k]) for k in shown) + " |" for r in tones]
    keys = ["item", "candidate", "seconds", "label", "median_label", "range_label", "spread_p20",
            "spread_p50", "spread_p80", "spread_range", *[f"{t}_p50" for t in TOOLS]]  # fmt: skip
    for title, rows in (("Clean clips", clean), ("Composed sessions (reported, not decided on)", mixed)):
        lines += ["", f"## {title}: spreads in semitones (largest minus smallest tool value)", "",
                  "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]  # fmt: skip
        lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    lines += ["", "## Tool versions", "", *[f"- {k}: {v}" for k, v in found["versions"].items()], ""]
    return lines


def versions(results: list[dict[str, Row]]) -> dict[str, str]:
    seen: dict[str, set[str]] = {}
    for per_tool in results:
        for tool, found in per_tool.items():
            if "provenance" in found:
                seen.setdefault(tool, set()).add(found["provenance"]["version"])
    return {t: "; ".join(sorted(v)) for t, v in sorted(seen.items())}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--data", "--splits", "--freeze", "--work", "--out"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    frozen = read(args.freeze)["files"].get(PREREG_FILE)
    if frozen != sha256(PREREG_PATH):
        print(f"the pre-registration's hash must be the frozen one ({frozen})")
        return 2
    home = args.work / "home"
    write(home / "consent.json", CONSENT)
    tone_results = {}
    for hz in PREREG["data"]["tones_hz"]:
        job = args.work / "jobs" / f"tone-{hz}"
        if not (job / "job.json").is_file():
            clip_job(job, tone(hz, PREREG["data"]["tone_s"]))
        tone_results[f"tone-{hz}"] = run_tools(job, home)
    tones = tone_rows(tone_results)
    calibrated, calibration_line = calibration(tones)
    print(calibration_line, flush=True)
    clean, clip_results = [], []
    for name, audio in list(clean_clips(args.data, read(args.splits)).items())[: args.limit]:
        job = args.work / "jobs" / name
        if not (job / "job.json").is_file():
            clip_job(job, audio)
        results = run_tools(job, home)
        clip_results.append(results)
        values = tool_values(results, "candidate")
        clean.append({"set": "clean", "item": name, "candidate": "candidate", "seconds": len(audio) / RATE,
                      **judge(values), **flat(values)})  # fmt: skip
        print(f"{name}: {clean[-1]['label']}", flush=True)
    mixed = composed(args.work, home, args.limit)
    found: Row = {"prereg_sha256": sha256(PREREG_PATH), "limit": args.limit, "calibration": calibration_line,
                  "calibrated": calibrated, "tones": tones, "clean": shares(clean), "composed": shares(mixed),
                  "versions": versions([*tone_results.values(), *clip_results])}  # fmt: skip
    if args.previous:
        old = (args.previous / "per_clip.csv").read_text(encoding="utf-8").splitlines()
        before = {r["item"]: r["label"] for r in csv.DictReader(old) if r["set"] == "clean"}
        found["previous"] = read(args.previous / "metrics.json")["clean"]
        found["change"] = change(clean, before)
    found["decision"] = decide(calibrated, found["clean"], args.previous is not None)
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / "per_clip.csv", [*clean, *mixed])
    write(args.out / "metrics.json", found)
    lines = report(found, tones, clean, mixed, args.limit)
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
