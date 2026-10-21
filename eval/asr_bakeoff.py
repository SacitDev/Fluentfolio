"""The speech-model comparison, run as eval/prereg/asr_bakeoff.toml says; its scores come from asr_scores.py.

Usage, from publishable/ (DATA holds edacc/, nyra/, librispeech/ as tools/fetch_test_data.py leaves them;
WORK keeps the jobs, the saved words and the built sessions; OUT gets the results):
  python eval/asr_bakeoff.py dev --data DATA --splits FILE --work WORK --out OUT
  python eval/asr_bakeoff.py test --data DATA --splits FILE --set-b DIR --freeze FILE --work WORK --out OUT
Options: --models (default: all four), --limit N (the first N items of each set: a small check run, never
a result), --device gpu|cpu (Whisper's device; the ONNX models run on the CPU only).
The dev phase fits each model's confidence threshold on EdAcc dev and Nyra dev and writes
dev_thresholds.json to OUT. The test phase refuses to start without it, or when the pre-registration's
hash differs from the one the freeze record holds. Saved words are reused, so a stopped run resumes.
metrics.json and per_item.csv depend only on the saved words; timing.json and report.md also hold times.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL))
import asr_run  # noqa: E402  (eval/ is a folder of scripts, not a package)
import asr_scores as scores  # noqa: E402
import sets  # noqa: E402
from asr_metrics import PREREG, best_threshold  # noqa: E402

from fluentfolio.asr import whisper_engine  # noqa: E402

PREREG_PATH = EVAL / "prereg" / "asr_bakeoff.toml"
PREREG_FILE = "eval/prereg/asr_bakeoff.toml"
DEV = ("edacc_dev", "nyra_dev")
TEXT = ("librispeech", "edacc_test", "nyra_test", "non_speech")
SESSIONS = ("set_b", "cs_hes")
PROMPTED_SETS = ("non_speech", "nyra_test")
CPU_SUBSET, LONG = "nyra_cpu_subset", "cs_long"
SAMPLE_S = 0.5  # how often the process tree's resident memory is read


@dataclass
class Item:
    id: str
    unit: str
    ref: str
    audio: np.ndarray

    @property
    def seconds(self) -> float:
        return len(self.audio) / asr_run.RATE


# ---- the data --------------------------------------------------------------------------------------------


def built(names: list[str], make: Any, folder: Path) -> list[Item]:
    """Composed sessions, each built once into the work folder; an item is one whole session."""
    items = []
    for n in names:
        if not (folder / n / "truth.json").is_file():
            make(n, folder / n)
        audio, _ = sf.read(folder / n / "session.wav", dtype="float32")
        items.append(Item(n, n, "", audio))
    return items


def load(name: str, data: Path, splits: dict[str, Any], work: Path, set_b: Path | None) -> list[Item]:
    """One set's items in their registered order."""
    if name.startswith("edacc_"):
        rows = splits["edacc"][name.removeprefix("edacc_")]["items"]
        found = sets.edacc_items(data / "edacc", [r["id"] for r in rows])
        items = []
        for r in rows:
            text, audio = found[r["id"]]
            items.append(Item(r["id"], r["speaker"], text, asr_run.mono16k(audio)))
        return items
    if name.startswith("nyra_"):
        split = "dev" if name == "nyra_dev" else "test"
        clips = sets.nyra_items(data / "nyra" / Path(PREREG["data"]["nyra"][split]["file"]).name)
        if name == CPU_SUBSET:
            order = {c: k for k, c in enumerate(splits[CPU_SUBSET])}
            clips = sorted((c for c in clips if c["id"] in order), key=lambda c: order[c["id"]])
        return [Item(c["id"], c["id"], c["text"], asr_run.mono16k(c["audio"])) for c in clips]
    if name == "librispeech":
        root = data / "librispeech" / "LibriSpeech"
        texts = sets.libri_texts(root)
        utts = splits["librispeech_subset"]
        return [Item(u, u.split("-")[0], texts[u], sets.libri_audio(root, u)) for u in utts]
    if name == "non_speech":
        kinds, seconds = PREREG["data"]["composed"]["non_speech"], PREREG["data"]["composed"]["non_speech_s"]
        return [Item(k, k, "", sets.non_speech(k, seconds)) for k in kinds]
    folder = work / "sessions"
    if name == "set_b":
        if set_b is None:
            raise SystemExit("the test phase needs --set-b")
        names = sorted(p.name for p in set_b.iterdir() if (p / "recipe.json").is_file())
        return built(names, lambda n, out: sets.build_b(set_b, n, out), folder)
    return built(["CS-HES" if name == "cs_hes" else "CS-LONG"], sets.sb.build_derived, folder)


def references(name: str, items: list[Item], work: Path) -> None:
    """The set's references, units and durations beside its saved words, so scoring needs no data."""
    path = work / "refs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    found = {i.id: {"ref": i.ref, "unit": i.unit, "seconds": round(i.seconds, 3)} for i in items}
    path.write_text(json.dumps(found, indent=1, ensure_ascii=False), encoding="utf-8")


# ---- the runs --------------------------------------------------------------------------------------------


def run(name: str, items: list[Item], model: str, device: str, work: Path) -> None:
    """One model over one set, saved as words per item; each composed session runs as a job of its own."""
    groups = [[i] for i in items] if name in SESSIONS else [items]
    for group in groups:
        path = work / "words" / name / model / f"{group[0].id if name in SESSIONS else 'all'}.json"
        if path.is_file():
            continue
        job = work / "jobs" / name / model / path.stem
        shutil.rmtree(job, ignore_errors=True)
        prompt = PREREG["candidates"][2]["filler_prompt"]
        whisper_engine.OPTIONS["initial_prompt"] = prompt if model == scores.PROMPTED else None
        started = time.monotonic()
        try:
            found = asr_run.run_items([(i.id, i.audio) for i in group], model.split("+")[0], device, job)
        finally:
            whisper_engine.OPTIONS["initial_prompt"] = None
        saved = {
            "model": model, "wall_s": round(time.monotonic() - started, 3),
            "audio_s": round(sum(i.seconds for i in group), 3), "record": found.record,
            "chunks": found.chunks,
            "items": {i: {"span": found.spans[i], "words": found.words[i], "speech": found.speech[i]}
                      for i in found.spans},
        }  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved), encoding="utf-8")
        print(f"{name} {model} {path.stem}: {saved['audio_s']} s of audio in {saved['wall_s']} s", flush=True)


def tree_rss(process: psutil.Process) -> int:
    return process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))


def cpu_timing(name: str, items: list[Item], model: str, work: Path) -> None:
    """The model on the CPU in a process of its own, its tree's resident memory read every 0.5 s."""
    folder = work / "cpu" / name
    out = folder / f"{model}.json"
    if out.is_file():
        return
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "items.npz", **{i.id: i.audio for i in items})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "FLUENTFOLIO_DEVICE": "cpu"}
    child = subprocess.Popen([sys.executable, __file__, "cpu-child", str(folder), model], env=env)
    peak, process = 0, psutil.Process(child.pid)
    while child.poll() is None:
        try:
            peak = max(peak, tree_rss(process))
        except psutil.Error:  # the child ended between the poll and the read
            continue
        time.sleep(SAMPLE_S)
    if child.returncode != 0:
        raise RuntimeError(f"the CPU run of {model} on {name} failed with exit code {child.returncode}")
    found = json.loads((folder / f"{model}.child.json").read_text(encoding="utf-8"))
    out.write_text(json.dumps({**found, "peak_rss_bytes": peak}, indent=1), encoding="utf-8")
    print(f"cpu {name} {model}: {found}, peak {peak / 1e9:.2f} GB", flush=True)


def cpu_child(folder: Path, model: str) -> None:
    with np.load(folder / "items.npz") as saved:
        items = [(k, saved[k]) for k in saved.files]
    started = time.monotonic()
    record = asr_run.run_items(items, model, "cpu", folder / "jobs" / model).record
    steps_s = (record.get("wall_ms", 0) + record.get("times", {}).get("wall_ms", 0)) / 1000
    audio_s = sum(len(a) for _, a in items) / asr_run.RATE
    result = {"audio_s": round(audio_s, 3), "wall_s": round(time.monotonic() - started, 3),
              "steps_s": round(steps_s, 3), "rtf": round(steps_s / audio_s, 4),
              "device": record.get("device"), "dropped_segments": record.get("dropped_segments")}  # fmt: skip
    (folder / f"{model}.child.json").write_text(json.dumps(result), encoding="utf-8")


# ---- scoring -----------------------------------------------------------------------------------------


def saved_runs(name: str, model: str, work: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text("utf-8")) for p in sorted((work / "words" / name / model).glob("*.json"))]


def rows_of(name: str, model: str, work: Path, thresholds: dict[str, Any]) -> list[scores.Row]:
    """Every item row of one model on one set, from its saved words, each in its item's own time."""
    refs = json.loads((work / "refs" / f"{name}.json").read_text(encoding="utf-8"))
    threshold = thresholds.get(model, {}).get("threshold")
    out = []
    for saved in saved_runs(name, model, work):
        for item, found in saved["items"].items():
            start = found["span"][0]
            words = scores.shifted(found["words"], start)
            speech = [[s - start, e - start] for s, e in found["speech"]]
            chunks = [[a - start, b - start] for a, b in saved["chunks"]]
            if name in SESSIONS:
                truth = json.loads((work / "sessions" / item / "truth.json").read_text(encoding="utf-8"))
                hes = sets.hesitations(truth) if name == "cs_hes" else []
                row = scores.session_row(truth, words, chunks, speech, hes)
            else:
                disfluent = name.startswith("nyra")
                row = scores.text_row(refs[item]["ref"], words, chunks, speech, disfluent, threshold)
            out.append({"set": name, "model": model, "item": item, "unit": refs[item]["unit"],
                        "seconds": refs[item]["seconds"], **row})  # fmt: skip
    order = {item: k for k, item in enumerate(refs)}
    return sorted(out, key=lambda r: order[r["item"]])


def run_models(name: str, models: list[str]) -> list[str]:
    """The models run on a set: on the sets that count invented text, Whisper with the filler prompt too."""
    return [*models, scores.PROMPTED] if name in PROMPTED_SETS and scores.WHISPER in models else models


def all_rows(names: tuple[str, ...], models: list[str], work: Path, limits: dict[str, Any]) -> scores.Sets:
    return {name: {m: rows_of(name, m, work, limits) for m in run_models(name, models)} for name in names}


def write_rows(path: Path, found: scores.Sets) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scores.COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for models in found.values():
            for rows in models.values():
                for r in rows:
                    writer.writerow({**r, "edges": ";".join(str(e) for e in r.get("edges") or [])})


def thresholds_from(found: scores.Sets) -> dict[str, Any]:
    """Each model's confidence threshold, fitted on EdAcc dev and Nyra dev together."""
    low, high, step = PREREG["decisions"]["confidence"]["grid"]
    grid = [round(low + k * step, 2) for k in range(round((high - low) / step) + 1)]
    out = {}
    for model in found[DEV[0]]:
        scored = [pair for name in DEV for r in found[name][model] for pair in r["confidence"]]
        best = best_threshold(scored, grid, PREREG["decisions"]["confidence"]["min_f1"])
        out[model] = {**best, "words": len(scored)}
    return out


SHOWN = ("wer_pct", "content_wer_pct", "filler_f1_pct", "grammar_edits_per_100", "repair_retention",
         "gap_precision", "median_edge_ms", "words_per_minute")  # fmt: skip


def metrics_of(found: scores.Sets) -> dict[str, Any]:
    """Each set's and model's pooled measures, with the main ones' bootstrap intervals."""
    out: dict[str, Any] = {}
    for name, models in found.items():
        for model, rows in models.items():
            pooled: dict[str, Any] = {**scores.pooled(rows), "items": len(rows)}
            pooled["intervals"] = {m: scores.interval(rows, m) for m in SHOWN if pooled.get(m) is not None}
            out.setdefault(name, {})[model] = pooled
    return out


def timing_of(names: tuple[str, ...], models: list[str], work: Path) -> dict[str, Any]:
    """Each run's speech-step time (Canary's CTC pass included) over its audio, and Canary's dropped loops."""
    out: dict[str, Any] = {"runs": {}, "cpu": {}}
    for name in names:
        for model in models:
            runs = saved_runs(name, model, work)
            steps_ms = sum(r["record"].get("wall_ms", 0) + r["record"].get("times", {}).get("wall_ms", 0)
                           for r in runs)  # fmt: skip
            audio_s = sum(r["audio_s"] for r in runs)
            out["runs"].setdefault(name, {})[model] = {
                "audio_s": round(audio_s, 1), "steps_s": round(steps_ms / 1000, 1),
                "rtf": round(steps_ms / 1000 / audio_s, 4) if audio_s else None,
                "device": next((r["record"]["device"] for r in runs if r["record"]), None),
                "dropped_segments": sum(r["record"].get("dropped_segments") or 0 for r in runs),
            }  # fmt: skip
    for path in sorted((work / "cpu").glob("*/*.json")):
        if not path.name.endswith(".child.json"):
            found = json.loads(path.read_text(encoding="utf-8"))
            found["peak_gb"] = round(found["peak_rss_bytes"] / 1e9, 3)
            out["cpu"].setdefault(path.parent.name, {})[path.stem] = found
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- the report ------------------------------------------------------------------------------------------

TEXT_COLUMNS = [
    ("wer_pct", "WER %", 2), ("content_wer_pct", "WER without fillers %", 2),
    ("filler_precision", "filler P", 3), ("filler_recall", "filler R", 3),
    ("filler_f1_pct", "filler F1 %", 1), ("grammar_edits_per_100", "grammar edits /100 (not reliable)", 2),
    ("inserted_per_100", "inserted /100", 2), ("timed_share", "timed share", 3),
    ("unsure_precision", "unsure P", 3), ("unsure_recall", "unsure R", 3), ("unsure_f1", "unsure F1", 3),
]  # fmt: skip
GAP_COLUMNS = [("gap_precision", "voiced-gap precision", 3), ("gaps", "voiced gaps", 0)]
NYRA_COLUMNS = [*TEXT_COLUMNS, ("repair_retention", "repair retention", 3), *GAP_COLUMNS]
SESSION_COLUMNS = [
    ("median_edge_ms", "median clip-edge error ms", 0), ("edges", "edges", 0), ("wer_pct", "WER %", 2),
    ("timed_share", "timed share", 3), *GAP_COLUMNS,
]  # fmt: skip
NOISE_COLUMNS = [("words_per_minute", "words written per minute", 2)]


def table(metrics: dict[str, Any], name: str, columns: list[tuple[str, str, int]]) -> list[str]:
    """One set's measures, a row per model, with the 95 % interval after each main measure."""
    lines = ["| model | items | " + " | ".join(label for _, label, _ in columns) + " |"]
    lines.append("|" + "---|" * (len(columns) + 2))
    for model, found in metrics.get(name, {}).items():
        cells = []
        for key, _, places in columns:
            cell = scores.fmt(found.get(key), places)
            if found.get("intervals", {}).get(key):
                _, low, high = found["intervals"][key]
                cell += f" [{low:.{places}f}, {high:.{places}f}]"
            cells.append(cell)
        lines.append(f"| {model} | {found['items']} | " + " | ".join(cells) + " |")
    return lines


def flagged(found: scores.Sets, name: str, shown: int = 10) -> list[str]:
    """The grammar-edit counter's first flags per model in the pre-registered hash order, to read by eye."""
    seed, lines = PREREG["seed"], []
    for model, rows in found.get(name, {}).items():
        flags = [(r["item"], ref, hyp, k) for r in rows for ref, hyp, k in r.get("flags", [])]
        flags.sort(key=lambda f: hashlib.sha256(f"{seed}:{f[0]}:{f[3]}".encode()).hexdigest())
        pairs = ", ".join(f"{ref} -> {hyp}" for _, ref, hyp, _ in flags[:shown]) or "none"
        lines.append(f"- {model}: {len(flags)} flags; the first {min(shown, len(flags))}: {pairs}")
    return lines


SECTIONS = [
    ("librispeech", TEXT_COLUMNS, "LibriSpeech test-clean subset (read speech; harness check)"),
    ("edacc_test", TEXT_COLUMNS, "EdAcc test selection (intervals by speaker)"),
    ("nyra_test", NYRA_COLUMNS, "Nyra test split (intervals by clip)"),
    ("set_b", SESSION_COLUMNS, "Composed set B (intervals by session)"),
    ("cs_hes", SESSION_COLUMNS, "CS-HES (fillers and repeated lead words at known times)"),
    ("non_speech", NOISE_COLUMNS, "Non-speech clips: silence, pink noise, music (120 s each)"),
]


def decisions(found: scores.Sets, timing: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """The decision lines, or why none is taken."""
    ok, _ = scores.calibration(found)
    if not ok:
        return ["Decision: none: the harness calibration is outside its range, so no decision is taken until "
                "the cause is found"]  # fmt: skip
    if any(m not in found["nyra_test"] for m in scores.MODELS):
        return ["Decision: none: not every model ran"]
    subset = timing["cpu"].get(CPU_SUBSET, {})
    cpu = {m: {"rtf": t["rtf"], "peak_gb": t["peak_gb"]} for m, t in subset.items()}
    long_s = timing["cpu"].get(LONG, {}).get(scores.V2, {}).get("steps_s")
    return scores.decide(found, thresholds, cpu, long_s)


def times(timing: dict[str, Any]) -> list[str]:
    """The runs' speech-step times, then the CPU runs' times and peak memory, as two tables."""
    lines = ["| set | model | device | audio s | speech step s | real-time factor |"]
    lines.append("|---|---|---|---|---|---|")
    for name, models in timing["runs"].items():
        for model, t in models.items():
            cells = [name, model, t["device"], t["audio_s"], t["steps_s"], t["rtf"]]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")
    lines.append("| CPU run | model | audio s | speech step s | real-time factor | peak RAM GB (1e9 bytes) |")
    lines.append("|---|---|---|---|---|---|")
    for name, models in timing["cpu"].items():
        for model, t in models.items():
            cells = [name, model, t["audio_s"], t["steps_s"], t["rtf"], t["peak_gb"]]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return lines


def report(found: scores.Sets, metrics: dict[str, Any], timing: dict[str, Any], thresholds: dict[str, Any],
           limit: int | None) -> str:  # fmt: skip
    _, checks = scores.calibration(found)
    lines = [f"prereg_sha256: {sha256(PREREG_PATH)}", *decisions(found, timing, thresholds), ""]
    lines += ["# Speech-model comparison", ""]
    if limit is not None:
        lines += [f"A small check run on the first {limit} items of each set: not a result.", ""]
    lines += ["## Harness calibration", "", *[f"- {c}" for c in checks],
              "- The Canary filler F1 is this comparison's position-matched F1; the Nyra benchmark's own "
              "scorer was not run, so it is not the published figure's scoring.", ""]  # fmt: skip
    for name, columns, what in SECTIONS:
        lines += [f"## {what}", "", *table(metrics, name, columns), ""]
    lines += ["## Grammar-edit flags on EdAcc test (the counter is not reliable)", "",
              *flagged(found, "edacc_test"), ""]  # fmt: skip
    lines += ["## Times and memory", "", *times(timing), ""]
    dropped = [f"{n}: {ms[scores.CANARY]['dropped_segments']}" for n, ms in timing["runs"].items()
               if scores.CANARY in ms]  # fmt: skip
    lines += ["## Canary's loop guard", "", f"Segments dropped: {'; '.join(dropped) or 'none run'}.", ""]
    shown = "; ".join(f"{m} {scores.fmt(t.get('threshold'))} (dev F1 {scores.fmt(t.get('f1'), 3)}, "
                      f"{t['words']} words)" for m, t in thresholds.items())  # fmt: skip
    lines += ["## Confidence thresholds, fitted on dev", "", shown, "",
              f"The filler-prompt Whisper run ({scores.PROMPTED}) only counts invented text, on the "
              "non-speech clips and Nyra test; its rows are in those tables.", ""]  # fmt: skip
    return "\n".join(lines)


# ---- the phases ------------------------------------------------------------------------------------------


def write(out: Path, stem: str, found: scores.Sets, extra: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics_of(found)
    write_rows(out / f"per_item{stem}.csv", found)
    text = json.dumps({**extra, "sets": metrics}, indent=1)
    (out / f"metrics{stem}.json").write_text(text, encoding="utf-8")
    return metrics


def main(argv: list[str]) -> int:
    if argv[:1] == ["cpu-child"]:
        cpu_child(Path(argv[1]), argv[2])
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["dev", "test"])
    for name in ("--data", "--splits", "--work", "--out"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--set-b", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--models", nargs="+", default=list(scores.MODELS), choices=scores.MODELS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args(argv)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    thresholds_path = args.out / "dev_thresholds.json"
    if args.phase == "dev":
        for name in DEV:
            items = load(name, args.data, splits, args.work, None)[: args.limit]
            references(name, items, args.work)
            for model in args.models:
                run(name, items, model, args.device, args.work)
        found = all_rows(DEV, args.models, args.work, {})
        thresholds = thresholds_from(found)
        thresholds_path.write_text(json.dumps(thresholds, indent=1), encoding="utf-8")
        write(args.out, "_dev", found, {"prereg_sha256": sha256(PREREG_PATH), "limit": args.limit})
        print(json.dumps(thresholds, indent=1))
        return 0
    frozen = json.loads(args.freeze.read_text("utf-8"))["files"].get(PREREG_FILE) if args.freeze else None
    if frozen != sha256(PREREG_PATH):
        print(f"the pre-registration's hash {sha256(PREREG_PATH)} is not the frozen one ({frozen})")
        return 2
    if not thresholds_path.is_file():
        print("the dev phase has not written its confidence thresholds yet")
        return 2
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    for name in (*TEXT, *SESSIONS):
        items = load(name, args.data, splits, args.work, args.set_b)[: args.limit]
        references(name, items, args.work)
        for model in run_models(name, args.models):
            run(name, items, model, args.device, args.work)
    subset = load(CPU_SUBSET, args.data, splits, args.work, None)[: args.limit]
    for model in args.models:
        cpu_timing(CPU_SUBSET, subset, model, args.work)
    if scores.V2 in args.models and args.limit is None:  # 15 minutes: left out of a small check run
        cpu_timing(LONG, load(LONG, args.data, splits, args.work, None), scores.V2, args.work)
    found = all_rows((*TEXT, *SESSIONS), args.models, args.work, thresholds)
    found["composed_gaps"] = {m: found["set_b"][m] + found["cs_hes"][m] for m in args.models}
    metrics = write(args.out, "", {k: v for k, v in found.items() if k != "composed_gaps"},
                    {"prereg_sha256": sha256(PREREG_PATH), "limit": args.limit})  # fmt: skip
    timing = timing_of((*TEXT, *SESSIONS), args.models, args.work)
    (args.out / "timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    text = report(found, metrics, timing, thresholds, args.limit)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print("\n".join(line for line in text.splitlines() if line.startswith(("prereg_sha256", "Decision"))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
