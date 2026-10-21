"""The speech-model comparison's scores from saved words: per item, per set and model, and the decisions.

Everything here reads what the comparison's runs saved and the frozen pre-registration; no model runs, so
scoring the same saved words twice gives the same bytes. The rules are eval/prereg/asr_bakeoff.toml's.
Only the voiced-gap rule is the app's own (fluentfolio.analyzers), because that rule is what is judged.
"""

from __future__ import annotations

import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402  (eval/ is a folder of scripts, not a package)
from asr_metrics import (  # noqa: E402
    PREREG,
    align,
    clip_edges,
    composed_gap_correct,
    confidence_targets,
    filler_counts,
    grammar_edits,
    normalise,
    nyra_gap_correct,
    prf,
    repairs,
    tokens_of,
    valid_times,
    without_fillers,
    word_errors,
)

from fluentfolio.analyzers.measures.disfluency import voiced_gaps

Row = dict[str, Any]
Sets = dict[str, dict[str, list[Row]]]  # set name -> model -> item rows
V2, V3, WHISPER, CANARY = (c["id"] for c in PREREG["candidates"])
MODELS = (V2, V3, WHISPER, CANARY)
PROMPTED = f"{WHISPER}+prompt"  # Whisper run again with the filler prompt, for invented text only
RULES = PREREG["decisions"]
# The frozen counter check found the grammar-edit rate not reliable (0.75 < 0.8), so by the rules it drops
# out of the Canary decision and the transcript style takes the conservative default; it is still reported.
if PREREG["counter_check"]["reliable"]:
    raise ValueError("these decisions follow the frozen counter check, which found the counter not reliable")
STYLE = {V2: "verbatim-best-effort", V3: "verbatim-best-effort", WHISPER: "smoothed", CANARY: "smoothed"}
COLUMNS = ["set", "model", "item", "unit", "seconds", "ref_words", "errors", "content_words",
           "content_errors", "filler_tp", "filler_fp", "filler_fn", "grammar_edits", "inserted",
           "repairs_kept", "repairs_total", "words", "timed_words", "gaps", "gaps_correct", "edges",
           "unsure_tp", "unsure_fp", "unsure_fn", "output"]  # fmt: skip


# ---- one item ----------------------------------------------------------------------------------------


def gap_lefts(
    words: list[Row], speech: Sequence[Sequence[int]], others: Sequence[tuple[int, int]]
) -> list[int]:
    """The places k of the output words whose gap to word k + 1 the app's rule marks as voiced."""
    spans = [(int(w["start_ms"]), int(w["end_ms"])) for w in words]
    found = {(a, b) for a, b, _ in voiced_gaps(spans, [(int(s), int(e)) for s, e in speech], others)}
    return [k for k in range(len(spans) - 1) if (spans[k][1], spans[k + 1][0]) in found]


def unsure(scored: list[tuple[float, bool]], threshold: float | None) -> tuple[int, int, int] | None:
    """Wrong words under the threshold, right words under it, and wrong words at or over it."""
    if threshold is None:
        return None
    tp = sum(c < threshold and w for c, w in scored)
    fp = sum(c < threshold and not w for c, w in scored)
    return tp, fp, sum(c >= threshold and w for c, w in scored)


def text_row(
    ref_raw: str, words: list[Row], chunks: list[list[int]], speech: list[list[int]], disfluent: bool,
    threshold: float | None,
) -> Row:  # fmt: skip
    """One item of a set with a reference text: every count the set's measures need."""
    ref = normalise(ref_raw)
    hyp, _ = tokens_of(words)
    pairs = align(ref, hyp)
    flags = grammar_edits(pairs)
    tp, fp, fn = filler_counts(ref, hyp)
    content_errors, content_words = word_errors(without_fillers(ref), without_fillers(hyp))
    timed, total = valid_times(words, chunks)
    scored = confidence_targets(ref, words)
    found = unsure(scored, threshold) or (None, None, None)
    row: Row = {
        "ref_words": len(ref), "errors": sum(p.ref != p.hyp for p in pairs), "content_words": content_words,
        "content_errors": content_errors, "filler_tp": tp, "filler_fp": fp, "filler_fn": fn,
        "grammar_edits": len(flags), "inserted": sum(p.ref is None for p in pairs),
        "words": total, "timed_words": timed, "unsure_tp": found[0], "unsure_fp": found[1],
        "unsure_fn": found[2], "output": " ".join(hyp), "confidence": scored,
        "flags": [(p.ref, p.hyp, p.ref_index) for p in flags],
    }  # fmt: skip
    if disfluent:  # the disfluency set: repairs and voiced gaps against its verbatim reference
        row["repairs_kept"], row["repairs_total"] = repairs(ref_raw, hyp)
        lefts = gap_lefts(words, speech, [])
        row["gaps"], row["gaps_correct"] = len(lefts), sum(nyra_gap_correct(ref_raw, words, k) for k in lefts)
    return row


def shifted(spans: list[Row], by: int) -> list[Row]:
    """Words moved from the joined recording's time to their own item's."""
    return [{**w, "start_ms": w["start_ms"] - by, "end_ms": w["end_ms"] - by} for w in spans]


def session_row(
    truth: Row, words: list[Row], chunks: list[list[int]], speech: list[list[int]],
    hesitations: list[tuple[int, int]],
) -> Row:  # fmt: skip
    """One composed session in its own time: clip edges, word times, voiced gaps and word errors.

    The utterances are the truth's candidate turns and the other speaker's speech its examiner segments.
    """
    segments = sorted(truth["segments"], key=lambda s: s["start_ms"])
    edges = clip_edges([(s["start_ms"], s["end_ms"], s["text"]) for s in segments], words)
    others = [(s["start_ms"], s["end_ms"]) for s in segments if s["role"] != "candidate"]
    gaps = correct = 0
    for turn in (t for t in truth["turns"] if t["role"] == "candidate"):
        mine = [w for w in words if turn["start_ms"] <= (w["start_ms"] + w["end_ms"]) / 2 <= turn["end_ms"]]
        for k in gap_lefts(mine, speech, others):
            gaps += 1
            correct += composed_gap_correct((mine[k]["end_ms"], mine[k + 1]["start_ms"]), hesitations, words)
    hyp, _ = tokens_of(words)
    errors, n = word_errors(normalise(" ".join(s["text"] for s in segments)), hyp)
    timed, total = valid_times(words, chunks)
    return {"ref_words": n, "errors": errors, "words": total, "timed_words": timed, "gaps": gaps,
            "gaps_correct": correct, "edges": edges, "output": " ".join(hyp)}  # fmt: skip


# ---- a set ---------------------------------------------------------------------------------------------


def _share(a: int, b: int, scale: float = 1.0) -> float | None:
    return scale * a / b if b else None


def pooled(rows: Sequence[Row]) -> dict[str, float | None]:
    """A set's measures from its items' summed counts, so a resample of units stays pooled."""

    def total(key: str) -> int:
        return sum(r.get(key) or 0 for r in rows)

    p, r, f1 = prf(total("filler_tp"), total("filler_fp"), total("filler_fn"))
    edges = [e for row in rows for e in row.get("edges") or []]
    asked = any(row.get("unsure_tp") is not None for row in rows)
    up, ur, uf = prf(total("unsure_tp"), total("unsure_fp"), total("unsure_fn")) if asked else (None,) * 3
    minutes = sum(row.get("seconds") or 0 for row in rows) / 60
    return {
        "wer_pct": _share(total("errors"), total("ref_words"), 100),
        "content_wer_pct": _share(total("content_errors"), total("content_words"), 100),
        "filler_precision": p, "filler_recall": r, "filler_f1_pct": None if f1 is None else 100 * f1,
        "grammar_edits_per_100": _share(total("grammar_edits"), total("ref_words"), 100),
        "inserted_per_100": _share(total("inserted"), total("ref_words"), 100),
        "repair_retention": _share(total("repairs_kept"), total("repairs_total")),
        "timed_share": _share(total("timed_words"), total("words")),
        "words_per_minute": total("words") / minutes if minutes else None,
        "gap_precision": _share(total("gaps_correct"), total("gaps")), "gaps": total("gaps"),
        "median_edge_ms": statistics.median(edges) if edges else None, "edges": len(edges),
        "unsure_precision": up, "unsure_recall": ur, "unsure_f1": uf,
    }  # fmt: skip


def by_unit(rows: Sequence[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = {}
    for r in rows:
        out.setdefault(str(r["unit"]), []).append(r)
    return out


def _metric(groups: dict[str, list[Row]], name: str) -> Callable[[list[str]], float]:
    def value(chosen: list[str]) -> float:
        found = pooled([r for u in chosen for r in groups[u]])[name]
        return float("nan") if found is None else float(found)

    return value


def interval(rows: Sequence[Row], name: str) -> tuple[float, float, float] | None:
    """The measure and its 95 % percentile interval from a bootstrap over the set's units."""
    groups = by_unit(rows)
    if not groups or pooled(rows)[name] is None:
        return None
    method = PREREG["statistics"]
    return stats.bootstrap(sorted(groups), _metric(groups, name), method["resamples"], method["seed"])


def difference(first: Sequence[Row], second: Sequence[Row], name: str) -> tuple[float, float, float] | None:
    """First model minus second on the same units: the difference and its paired 95 % interval."""
    a, b = by_unit(first), by_unit(second)
    if not a or sorted(a) != sorted(b) or None in (pooled(first)[name], pooled(second)[name]):
        return None
    method = PREREG["statistics"]
    return stats.paired(sorted(a), _metric(a, name), _metric(b, name), method["resamples"], method["seed"])


# ---- the decisions ---------------------------------------------------------------------------------------


def fmt(x: float | None, places: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


def signed(found: tuple[float, float, float] | None) -> str:
    return "n/a" if found is None else f"{found[0]:+.2f} [{found[1]:+.2f}, {found[2]:+.2f}]"


def calibration(sets: Sets) -> tuple[bool, list[str]]:
    """The harness checks: Parakeet v2's LibriSpeech WER and Canary's Nyra test filler F1 in their ranges."""
    lines, ok = [], True
    checks = [("librispeech", V2, "wer_pct", "parakeet_v2_librispeech_wer_pct"),
              ("nyra_test", CANARY, "filler_f1_pct", "canary_nyra_test_filler_f1_pct")]  # fmt: skip
    for name, model, metric, key in checks:
        low, high = PREREG["calibration"][key]
        rows = sets.get(name, {}).get(model, [])
        found = pooled(rows)[metric] if rows else None
        inside = found is not None and low <= found <= high
        ok &= inside
        lines.append(f"{key}: {fmt(found)} in [{low}, {high}]: {'yes' if inside else 'NO'}")
    return ok, lines


def default_model(sets: Sets) -> str:
    rule = RULES["default_model"]
    f1 = difference(sets["nyra_test"][V3], sets["nyra_test"][V2], "filler_f1_pct")
    wer = difference(sets["edacc_test"][V3], sets["edacc_test"][V2], "wer_pct")
    # At least 5 points higher with an interval above 0, and a WER at most 1 point higher.
    switch = f1 is not None and f1[0] >= rule["filler_f1_margin_points"] and f1[1] > 0
    switch = switch and wer is not None and wer[0] <= rule["wer_margin_points"]
    verdict = "propose Parakeet v3 to the maintainer" if switch else "Parakeet v2 stays"
    return (f"Decision: default model: {verdict} (v3 minus v2: filler F1 {signed(f1)} points on Nyra test, "
            f"WER {signed(wer)} points on EdAcc test)")  # fmt: skip


def canary(sets: Sets, cpu: dict[str, dict[str, float]]) -> str:
    rule = RULES["canary"]
    nyra, edacc, composed = sets["nyra_test"], sets["edacc_test"], sets["set_b"]
    c_f1, w_f1 = (pooled(nyra[m])["filler_f1_pct"] for m in (CANARY, WHISPER))
    c_wer, w_wer = (pooled(edacc[m])["wer_pct"] for m in (CANARY, WHISPER))
    c_edge, v2_edge = (pooled(composed[m])["median_edge_ms"] for m in (CANARY, V2))
    rtf, ram = cpu.get(CANARY, {}).get("rtf"), cpu.get(CANARY, {}).get("peak_gb")
    wer_max = None if w_wer is None else w_wer + rule["wer_margin_points"]
    edge_max = None if v2_edge is None else v2_edge + rule["edge_margin_ms"]
    checks = [
        (f"filler F1 {fmt(c_f1)} at least Whisper's {fmt(w_f1)}", c_f1, w_f1),
        (f"WER {fmt(c_wer)} at most Whisper's + {rule['wer_margin_points']}, {fmt(wer_max)}", wer_max, c_wer),
        (f"median clip edge {fmt(c_edge, 0)} ms at most {fmt(edge_max, 0)}", edge_max, c_edge),
        (f"CPU real-time factor {fmt(rtf, 3)} at most {rule['cpu_rtf_max']}", rule["cpu_rtf_max"], rtf),
        (f"peak RAM {fmt(ram)} GB at most {rule['peak_ram_gb_max']}", rule["peak_ram_gb_max"], ram),
    ]  # each check holds when its second value is at least its third
    failed = [text for text, high, low in checks if high is None or low is None or high < low]
    held = [text for text, _, _ in checks if text not in failed]
    verdict = "added next to Whisper" if not failed else "not added"
    return (f"Decision: Canary: {verdict}; failed: {'; '.join(failed) or 'none'}; "
            f"held: {'; '.join(held) or 'none'}; the grammar-edit condition drops out, as the counter is "
            "not reliable")  # fmt: skip


def voiced_gap_markers(sets: Sets) -> str:
    least = RULES["voiced_gaps"]["min_precision"]
    passed, shown = [], []
    for m in MODELS:
        a = pooled(sets["composed_gaps"][m])["gap_precision"]
        b = pooled(sets["nyra_test"][m])["gap_precision"]
        shown.append(f"{m} {fmt(a)} composed, {fmt(b)} Nyra test")
        if a is not None and b is not None and a >= least and b >= least:
            passed.append(m)
    where = "stay in the details folder"
    if passed:
        where = f"go into the text for the AI chat for {', '.join(passed)}"
    return (f"Decision: voiced-gap markers {where} (precision {least} or more needed on both: "
            f"{'; '.join(shown)})")  # fmt: skip


def decide(
    sets: Sets, thresholds: dict[str, Row], cpu: dict[str, dict[str, float]], long_s: float | None
) -> list[str]:
    """One "Decision:" line per pre-registered decision, each with the numbers it rests on."""
    limit = RULES["parakeet_cpu"]["max_cpu_s_for_15_min"]
    slow = long_s is not None and long_s > limit
    return [
        default_model(sets),
        canary(sets, cpu),
        "Decision: transcript style: " + ", ".join(f"{m} {s}" for m, s in STYLE.items())
        + " (the grammar-edit counter is not reliable, so the conservative default applies)",
        voiced_gap_markers(sets),
        "Decision: confidence thresholds, fitted on dev: "
        + ", ".join(f"{m} {fmt(t.get('threshold'))}" for m, t in thresholds.items()),
        f"Decision: Parakeet v2 on the CPU took {fmt(long_s, 1)} s for the 15-minute session "
        f"(limit {limit} s): "
        + ("propose GPU support for it; the product does not change" if slow else "no change"),
    ]  # fmt: skip
