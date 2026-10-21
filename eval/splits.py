"""The item lists that eval/prereg/*.toml choose by seeded hashes, computed once from metadata and saved.

Reads only IDs, speakers, sex and L1 labels, reference texts and durations (from the audio headers); no
model runs here. Usage, from publishable/: python eval/splits.py --data DIR --out FILE
DIR holds edacc/, nyra/ and librispeech/LibriSpeech/ as tools/fetch_test_data.py leaves them.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "tests" / "support"))
import session_builder as sb  # noqa: E402  (eval/ is a folder of scripts, not a package)
from asr_metrics import normalise  # noqa: E402

PREREG = ROOT / "eval" / "prereg"
RECIPES = ROOT / "tests" / "fixtures" / "sessions"


def rank(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def edacc_rows(path: Path, not_scored: str) -> list[dict[str, Any]]:
    """Every row of one shard: ID, speaker, L1, word count, duration, and whether it is scored at all."""
    rows, n = [], 0
    columns = ["speaker", "l1", "text", "audio"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            seconds = round(sf.info(io.BytesIO(row["audio"]["bytes"])).duration, 3)  # the header only
            text = row["text"] or ""
            rows.append({"id": f"{path.name}:{n}", "speaker": row["speaker"], "l1": (row["l1"] or "").strip(),
                         "scored": text.strip() != not_scored, "words": len(normalise(text)),
                         "seconds": seconds})  # fmt: skip
            n += 1
    return rows


def eligible(r: dict[str, Any], spec: dict[str, Any]) -> bool:
    """A row the comparison may use: a given L1 that is not English, scored, 1-30 s, 3 words or more."""
    return bool(r["l1"] and spec["l1_exclude"] not in r["l1"].lower() and r["scored"]
                and spec["row_min_s"] <= r["seconds"] <= spec["row_max_s"]
                and r["words"] >= spec["min_words"])  # fmt: skip


def edacc_selection(rows: list[dict[str, Any]], spec: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Speakers with a given L1 that is not English, by hash, up to speaker_cap_s each, to split_target_s."""
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for r in (r for r in rows if eligible(r, spec)):
        by_speaker.setdefault(r["speaker"], []).append(r)
    chosen, total = [], 0.0
    for speaker in sorted(by_speaker, key=lambda s: rank(s, seed)):
        if total >= spec["split_target_s"]:
            break
        own = 0.0
        for r in sorted(by_speaker[speaker], key=lambda r: rank(r["id"], seed)):
            if own + r["seconds"] <= spec["speaker_cap_s"]:
                chosen.append(r)
                own += r["seconds"]
        total += own
    return chosen


def candidate_clips(recipe: dict[str, Any]) -> list[str]:
    """The clips a recipe's candidate says, in order of first use: the clips an L2 interview replaces."""
    candidate = {t for t, info in recipe["turns"].items() if sb.ROLE[info["kind"]] == "candidate"}
    used = dict.fromkeys(e["clip"] for e in recipe["events"])
    return [c for c in used if any(e["clip"] == c and e["turn"] in candidate for e in recipe["events"])]


def l2_speakers(
    rows: list[dict[str, Any]], speakers: list[str], spec: dict[str, Any], need: int, n: int
) -> list[str]:
    """The first n speakers in order with at least `need` eligible rows, so that every candidate clip of the
    interview gets a row of its own."""
    counts = Counter(r["speaker"] for r in rows if eligible(r, spec))
    return [s for s in speakers if counts[s] >= need][:n]


def librispeech(folder: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Every test-clean utterance's duration, and each test-clean speaker's sex."""
    seconds = {p.stem: sf.info(p).duration for p in sorted((folder / "test-clean").rglob("*.flac"))}
    sex = {}
    for line in (folder / "SPEAKERS.TXT").read_text(encoding="utf-8").splitlines():
        fields = [f.strip() for f in line.split("|")]
        if not line.startswith(";") and len(fields) >= 3 and fields[2] == "test-clean":
            sex[fields[0]] = fields[1]  # only the ID and the sex; the reader's name is not read
    return seconds, sex


def until(ids: list[str], seconds: dict[str, float], target: float, seed: int) -> list[str]:
    """IDs in hash order, taken until their durations reach the target."""
    chosen, total = [], 0.0
    for i in sorted(ids, key=lambda i: rank(i, seed)):
        if total >= target:
            break
        chosen.append(i)
        total += seconds[i]
    return chosen


def recipe_speakers() -> dict[str, str]:
    """Each committed recipe's LibriSpeech candidate speaker; a derived recipe has its base's."""
    events = {}
    for path in sorted(RECIPES.glob("*.json")):
        recipe = json.loads(path.read_text(encoding="utf-8"))
        events[recipe["name"]] = recipe
    out = {}
    for name, recipe in events.items():
        source = events[recipe.get("base", name)]
        found = {e["clip"].split("_")[1] for e in source.get("events", []) if e["clip"].startswith("ls_")}
        if found:
            (out[name],) = found
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    asr, mfa, pitch = (tomllib.loads((PREREG / f"{name}.toml").read_text(encoding="utf-8"))
                       for name in ("asr_bakeoff", "mfa_gate", "pitch_fusion"))  # fmt: skip
    seed, edacc, l2 = asr["seed"], asr["data"]["edacc"], mfa["data"]["l2_interviews_per_split"]
    need = len(candidate_clips(sb.load_recipe(RECIPES / "CS-INT.json")))
    out: dict[str, Any] = {"seed": seed, "edacc": {}, "l2_rows_needed": need}
    for split in ("dev", "test"):
        shards = [args.data / "edacc" / Path(s["file"]).name for s in edacc[f"{split}_shards"]]
        rows = [r for path in shards for r in edacc_rows(path, edacc["not_scored"])]
        chosen = edacc_selection(rows, edacc, seed)
        speakers = list(dict.fromkeys(r["speaker"] for r in chosen))
        interview = l2_speakers(rows, speakers, edacc, need, l2)
        out["edacc"][split] = {"rows_read": len(rows), "items": chosen, "speakers": speakers,
                               "l2_interview_speakers": interview}  # fmt: skip
    test_file = args.data / "nyra" / Path(asr["data"]["nyra"]["test"]["file"]).name
    table = pq.read_table(test_file, columns=["id", "duration_in_s"]).to_pylist()
    nyra = {str(r["id"]): r["duration_in_s"] for r in table}
    out["nyra_cpu_subset"] = until(list(nyra), nyra, asr["metrics"]["cpu_subset_s"], seed)
    seconds, sex = librispeech(args.data / "librispeech" / "LibriSpeech")
    out["librispeech_subset"] = until(list(seconds), seconds, asr["data"]["librispeech"]["target_s"], seed)
    used = recipe_speakers()
    free = sorted(s for s in sex if s not in used.values())
    out["set_b_speakers"] = {
        name: min((s for s in free if sex[s] == sex[old]), key=lambda s: rank(f"{name}:{s}", seed))
        for name, old in used.items()
    }
    clips = {}
    for speaker in sorted(free, key=lambda s: rank(s, seed))[: pitch["data"]["librispeech_clips"]]:
        own, total = [], 0.0
        for utt in sorted(u for u in seconds if u.split("-")[0] == speaker):
            if total >= pitch["data"]["clip_min_s"]:
                break
            own.append(utt)
            total += seconds[utt] + (pitch["data"]["gap_ms"] / 1000 if len(own) > 1 else 0.0)
        clips[speaker] = own
    out["pitch_clips"] = clips
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for split in ("dev", "test"):
        e = out["edacc"][split]
        total = sum(r["seconds"] for r in e["items"])
        print(f"edacc {split}: {e['rows_read']} rows read, {len(e['items'])} items, "
              f"{len(e['speakers'])} speakers, {total:.1f} s; {len(e['l2_interview_speakers'])} interview "
              f"speakers with {need} or more eligible rows")  # fmt: skip
    print(f"nyra cpu subset: {len(out['nyra_cpu_subset'])} clips; librispeech subset: "
          f"{len(out['librispeech_subset'])} utterances; set B speakers: {out['set_b_speakers']}; "
          f"pitch clips: {len(clips)} speakers")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
