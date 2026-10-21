"""Twenty aligned pairs that the grammar-edit counter flags in Whisper's output on the EdAcc dev selection.

A reviewer judges from the text whether each flagged change is a real grammar edit; the counter is used
in the comparison only if at least 0.8 of the pairs are (eval/prereg/asr_bakeoff.toml). Dev data only.
Usage, from publishable/: python eval/counter_pairs.py --data DIR --splits FILE --heard FILE --out FILE
[--job DIR] [--device gpu|cpu]. When --heard does not exist, the model runs in --job and its words are saved
there; when it exists, the counter runs again on those saved words, with no model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr_run  # noqa: E402  (eval/ is a folder of scripts, not a package)
from asr_metrics import align, grammar_edits, normalise  # noqa: E402

PREREG = Path(__file__).resolve().parent / "prereg" / "asr_bakeoff.toml"


def rows(folder: Path, ids: list[str]) -> dict[str, tuple[str, bytes]]:
    """The reference text and audio bytes of the named rows ("<shard file>:<row>")."""
    wanted: dict[str, set[int]] = {}
    for item in ids:
        name, row = item.rsplit(":", 1)
        wanted.setdefault(name, set()).add(int(row))
    out = {}
    for name, numbers in wanted.items():
        n = 0
        for batch in pq.ParquetFile(folder / name).iter_batches(batch_size=64, columns=["text", "audio"]):
            for row in batch.to_pylist():
                if n in numbers:
                    out[f"{name}:{n}"] = (row["text"] or "", row["audio"]["bytes"])
                n += 1
    return out


def marked(words: list[str], index: int) -> str:
    return " ".join(f"[[{w}]]" if k == index else w for k, w in enumerate(words))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--data", "--splits", "--heard", "--out"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args(argv)
    prereg = tomllib.loads(PREREG.read_text(encoding="utf-8"))
    seed, wanted, model = prereg["seed"], prereg["counter_check"]["pairs"], prereg["counter_check"]["model"]
    items = [r["id"] for r in json.loads(args.splits.read_text(encoding="utf-8"))["edacc"]["dev"]["items"]]
    found = rows(args.data / "edacc", items)
    if args.heard.exists():
        heard = json.loads(args.heard.read_text(encoding="utf-8"))
        source = f"saved words in {args.heard.name}"
    elif args.job is None:
        parser.error("--job is needed when --heard does not exist yet")
    else:
        heard = asr_run.transcribe(
            [(i, asr_run.mono16k(found[i][1])) for i in items], model, args.device, args.job
        )
        args.heard.parent.mkdir(parents=True, exist_ok=True)
        args.heard.write_text(json.dumps(heard, indent=1), encoding="utf-8")
        source = f"the model run on {args.device}"
    flagged, words = [], 0
    for item in items:
        ref = normalise(found[item][0])
        hyp = normalise(" ".join(w["text"] for w in heard[item]))
        words += len(ref)
        pairs = align(ref, hyp)
        for p in grammar_edits(pairs):
            k = next(i for i, q in enumerate(pairs) if q is p)  # the pair's place among the aligned words
            hyp_index = sum(1 for q in pairs[:k] if q.hyp is not None)
            flagged.append((item, p, marked(ref, p.ref_index), marked(hyp, hyp_index)))
    flagged.sort(key=lambda f: hashlib.sha256(f"{seed}:{f[0]}:{f[1].ref_index}".encode()).hexdigest())
    shown = min(wanted, len(flagged))
    lines = [
        f"Grammar-edit counter check: {model} on the EdAcc dev selection ({len(items)} rows, {words} "
        f"reference words after the normaliser). The counter flagged {len(flagged)} substitutions "
        f"({100 * len(flagged) / words:.2f} per 100 words); the {shown} below come first in "
        f"order of sha256('{seed}:<row ID>:<word index>'). Words from {source}.",
        "Each pair: the reference (what the speaker said) and the model's output, the flagged word in [[ ]].",
        "",
    ]
    for n, (item, p, ref_text, hyp_text) in enumerate(flagged[:wanted], 1):
        lines += [
            f"{n}. {item}   {p.ref} -> {p.hyp}",
            f"   reference: {ref_text}",
            f"   output:    {hyp_text}",
            "",
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(lines[0])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
