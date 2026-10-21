"""The self-test: the bundled clip through every installed and switched-on part, held to its known answers.

The caller runs a throwaway interview job on the clip (two LibriSpeech readers, CC BY 4.0) with the normal
steps. This module checks each part's result against truth.json and reference.txt with wide limits, and
keeps the outcome in the app folder's state.json, where Doctor and the components list read it.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from importlib import resources
from itertools import permutations
from pathlib import Path
from typing import Any, Literal

from fluentfolio.core.enums import JobState, Status
from fluentfolio.core.errors import catalogue as codes
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.paths import install_state
from fluentfolio.doctor.checks import ComponentRow
from fluentfolio.jobs import store

ASSETS = resources.files("fluentfolio") / "assets" / "selftest"
CANDIDATE = "candidate"  # the candidate's ID in an interview job
PEOPLE = 2  # the clip's two readers
WER_MAX = 0.10
AGREEMENT_MIN = 0.90  # the share of the true speech time given to the right person
SPEECH_SHARE = 0.10  # the voice detector's speech time against the truth
PRAAT_SHARE = 0.02  # Praat is deterministic, so the installed copy must give the bundled median
SEMITONES_MAX = 1.0  # the other pitch tools against Praat
EDGE_MS, EDGE_SHARE = 50, 0.80  # MFA: utterance edges against the truth, and how many must hold
WORDS_TO_CHECK_MAX = 1  # clean read speech should give almost none
BUDGET_MS = {"gpu": 180_000, "cpu": 360_000}
DAMAGED = "E_HASH_MISMATCH"  # a model file of the wrong size, found when a step loads it
REDOWNLOAD = "a file of this part is missing or damaged: re-download this model"
WORD = re.compile(r"[a-z0-9']+")
PITCH_TOOLS = ("swiftf0", "opensmile")
NO_PITCH = "found no pitch in the test clip"
# The parts each step runs, when its result alone decides them; the other steps are checked one by one.
STEP_PARTS = {
    "language": ("whisper-tiny",),
    "phones": ("phone-model", "espeak-ng", "ctc-aligner"),
    "nlp": ("spacy-en-core-web-sm",),
}

PartStatus = Literal["ok", "failed", "unavailable", "disabled", "not-tested"]
Row = dict[str, Any]


@dataclass
class Part:
    status: PartStatus
    reason: str = ""
    values: dict[str, float | int] = field(default_factory=dict)


def clip() -> Path:
    """The bundled recording, as a file path the analysis can copy."""
    return Path(str(ASSETS / "selftest.opus"))


def truth() -> Row:
    found: Row = read_json(Path(str(ASSETS / "truth.json")))
    return found


def reference() -> list[str]:
    return words(Path(str(ASSETS / "reference.txt")).read_text(encoding="utf-8"))


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def wer(reference: list[str], hypothesis: list[str]) -> float:
    """Word error rate: the fewest substitutions, deletions and insertions, over the reference length."""
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        prev, row[0] = row[0], i
        for j, hyp in enumerate(hypothesis, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ref != hyp))
    return row[-1] / max(1, len(reference))


def agreement(turns: list[Row], segments: list[Row]) -> float:
    """The share of true speech time the best match of found to true speakers gives to the right person."""
    found = sorted({t["speaker"] for t in turns})
    people = sorted({s["speaker"] for s in segments})
    if len(found) < len(people):
        return 0.0
    overlap = {(f, p): 0 for f in found for p in people}
    for t in turns:
        for s in segments:
            both = min(t["end_ms"], s["end_ms"]) - max(t["start_ms"], s["start_ms"])
            overlap[(t["speaker"], s["speaker"])] += max(0, both)
    best = max(sum(overlap[(f, p)] for f, p in zip(pick, people, strict=True))
               for pick in permutations(found, len(people)))  # fmt: skip
    return best / sum(s["end_ms"] - s["start_ms"] for s in segments)


def edges_held(aligned: list[Row], segments: list[Row]) -> tuple[int, int]:
    """How many of the candidate's true utterances have MFA words with both edges within EDGE_MS of truth."""
    mine = [s for s in segments if s["role"] == "candidate"]
    held = 0
    for s in mine:
        inside = [w for w in aligned if s["start_ms"] <= (w["start_ms"] + w["end_ms"]) / 2 <= s["end_ms"]]
        if inside:
            first, last = inside[0]["start_ms"] - s["start_ms"], inside[-1]["end_ms"] - s["end_ms"]
            held += abs(first) <= EDGE_MS and abs(last) <= EDGE_MS
    return held, len(mine)


def semitones(a: float, b: float) -> float:
    return abs(12 * math.log2(a / b))


def _check(ok: bool, values: dict[str, float | int], reason: str) -> Part:
    return Part("ok", "", values) if ok else Part("failed", reason, values)


def _from_record(rec: store.StageRecord | None) -> Part | None:
    """The part's status when its step did not finish; None when the step's values must decide."""
    if rec is None or rec.status is None:
        return Part("not-tested", "an earlier step stopped, so this one did not run")
    code = rec.error_code
    known = codes()
    if rec.status in (Status.DISABLED, Status.UNAVAILABLE):
        reason = known[code].message if code in known else "switched off in Settings"
        return Part("disabled" if rec.status is Status.DISABLED else "unavailable", reason)
    if rec.status is Status.SKIPPED:
        return Part("not-tested", "the step it needs did not finish, so this one did not run")
    if rec.status is Status.FAILED:
        return _failed(code)
    return None


def _failed(code: str | None) -> Part:
    """A step that stopped: a model never downloaded only limits the app; anything else is a failure."""
    if code == "E_MODEL_MISSING":
        return Part("unavailable", codes()[code].message)
    return Part("failed", REDOWNLOAD if code == DAMAGED else f"it stopped with {code or 'an error'}")


def _vad(job: Path, known: Row) -> Part:
    found = int(read_json(job / "stages" / "vad.json")["speech_ms"])
    want = int(known["speech_ms"])
    return _check(abs(found - want) <= SPEECH_SHARE * want, {"speech_ms": found, "truth_ms": want},
                  f"speech time {found} ms, against {want} ms in the clip")  # fmt: skip


def _asr(job: Path) -> Part:
    heard = [w for item in read_json(job / "stages" / "asr.json")["words"] for w in words(item["text"])]
    rate = wer(reference(), heard)
    return _check(rate <= WER_MAX, {"wer": round(rate, 4)}, f"word error rate {rate:.1%}, more than 10%")


def _language(job: Path) -> Part:
    other = int(read_json(job / "stages" / "language.json")["non_english_ms"])
    return _check(other == 0, {"non_english_ms": other},
                  f"{other} ms of the English clip heard as another language")  # fmt: skip


def _speakers(job: Path, known: Row) -> tuple[str, Part]:
    found = read_json(job / "stages" / "speakers.json")
    share = agreement(found["turns"], known["segments"])
    n = sum(s["kind"] == "main" for s in found["speakers"])
    part = _check(n == PEOPLE and share >= AGREEMENT_MIN, {"speakers": n, "agreement": round(share, 4)},
                  f"{n} speakers found, {share:.0%} of the speech given to the right person")  # fmt: skip
    return str(found["backend"]), part


def _mfa(job: Path, known: Row) -> Part:
    found = read_json(job / "stages" / "alignment.json")
    mfa = found["mfa"]
    if mfa["status"] != Status.OK:
        rec = store.StageRecord(id="mfa", status=Status(mfa["status"]), error_code=mfa.get("code"))
        return _from_record(rec) or Part("not-tested")
    aligned = [w for u in found["utterances"] if u["method"] == "mfa" for w in u["words"]]
    held, n = edges_held(aligned, known["segments"])
    return _check(held >= EDGE_SHARE * n, {"edges_held": held, "utterances": n},
                  f"{held} of {n} utterances aligned with both edges within {EDGE_MS} ms")  # fmt: skip


def median_hz(job: Path, tool: str) -> float | None:
    """A pitch tool's median over the candidate's answers, or None when it gave none."""
    found = read_json(job / "analyzers" / f"{tool}.json")["candidates"].get(CANDIDATE)
    value = found["scopes"]["all"].get("median_hz") if found else None
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _pitch(job: Path, known: Row, stages: dict[str, store.StageRecord]) -> dict[str, Part]:
    """Praat against the bundled median; the other tools against Praat, or the bundled median without it."""
    want = float(known["candidate_median_f0_hz"])
    parts: dict[str, Part] = {}
    praat = None
    if (p := _from_record(stages.get("praat"))) is None:
        praat = median_hz(job, "praat")
        p = _check(praat is not None and abs(praat - want) <= PRAAT_SHARE * want,
                   {"median_hz": round(praat or 0, 2), "truth_hz": want},
                   f"median pitch {praat:.1f} Hz, against {want} Hz" if praat else NO_PITCH)  # fmt: skip
    parts["praat"] = p
    against = praat if p.status == "ok" else want
    for tool in PITCH_TOOLS:
        if (p := _from_record(stages.get(tool))) is None:
            hz = median_hz(job, tool)
            gap = semitones(hz, against) if hz else None
            if gap is None:
                p = Part("failed", NO_PITCH, {"median_hz": 0})
            else:
                p = _check(gap <= SEMITONES_MAX, {"median_hz": round(hz or 0, 2), "semitones": round(gap, 2)},
                           f"median pitch {gap:.1f} semitones from Praat's")  # fmt: skip
        parts[tool] = p
    return parts


def _words_to_check(job: Path) -> Part:
    pointers = read_json(job / "analyzers" / "measures.json")["candidates"][CANDIDATE]["pointers"]
    n = len(pointers["words_to_check"])
    return _check(n <= WORDS_TO_CHECK_MAX, {"words_to_check": n}, f"{n} words to check in clean read speech")


def evaluate(job: Path, record: store.JobRecord, rows: list[ComponentRow]) -> dict[str, Part]:
    """Every part the analysis would use, checked against the clip's truth where its step ran."""
    known = truth()
    stages = {s.id: s for s in record.stages}
    parts: dict[str, Part] = {}
    if (p := _from_record(stages.get("vad"))) is None:
        p = _vad(job, known)
    parts["silero-vad"] = p
    if (p := _from_record(stages.get("asr"))) is None:
        p = _asr(job)
    parts[str(record.asr)] = p
    if (p := _from_record(stages.get("speakers"))) is None:
        backend, p = _speakers(job, known)
        parts[backend] = p
        fallback = stages["speakers"].fallback
        if fallback and fallback["to"] != "cpu":  # the first choice stopped and the other backend ran
            first = next((r.id for r in rows if r.kind == "diarizer" and r.id != backend), None)
            if first:
                parts[first] = _failed(fallback["after"])
    else:
        for r in rows:
            if r.kind == "diarizer":
                parts[r.id] = p or Part("not-tested")
    for step, ids in STEP_PARTS.items():
        p = _from_record(stages.get(step)) or (_language(job) if step == "language" else Part("ok"))
        parts.update(dict.fromkeys(ids, p))
    pronunciation = record.settings.get("analyzers", {}).get("pronunciation", True)  # missing is on
    if pronunciation:  # only the pronunciation check reads the dictionary, in the phones step
        parts["pronunciation-dictionary"] = parts["espeak-ng"]
    if (p := _from_record(stages.get("alignment"))) is None:
        p = _mfa(job, known)
    parts["mfa"] = p
    parts.update(_pitch(job, known, stages))
    if pronunciation and parts["phone-model"].status == "ok" and _from_record(stages.get("measures")) is None:
        parts["phone-model"] = _words_to_check(job)
    return _with_rows(parts, rows)


def _with_rows(parts: dict[str, Part], rows: list[ComponentRow]) -> dict[str, Part]:
    """Only the parts in use. A failure a step saw stands; a part that is not ready says why it is not."""
    out = {}
    for r in rows:
        found = parts.get(r.id)
        if found is not None and found.status == "failed":
            out[r.id] = found
        elif r.status == "needs-consent":
            out[r.id] = Part("disabled", r.reason)
        elif r.status not in ("ready", "self-test-failed"):
            out[r.id] = Part("unavailable", r.reason)
        else:
            out[r.id] = found or Part("not-tested", "these settings did not use it")
    return out


def passed(parts: dict[str, Part], record: store.JobRecord) -> bool:
    """The analysis finished and no part gave a wrong result; a part switched off or missing only limits."""
    finished = record.state in (JobState.COMPLETED, JobState.COMPLETED_WITH_GAPS)
    return finished and not any(p.status == "failed" for p in parts.values())


def save(app: Path, parts: dict[str, Part], record: store.JobRecord, elapsed_ms: int) -> Row:
    """Keep the outcome in state.json, where Doctor and the components list read it."""
    device = str(record.device.get("resolved") or "cpu")
    outcome = {
        "utc": store.utc_now(), "passed": passed(parts, record), "device": device,
        "elapsed_ms": elapsed_ms, "budget_ms": BUDGET_MS[device],
        "components": {k: asdict(v) for k, v in parts.items()},
    }  # fmt: skip
    state = install_state(app)
    state["selftest"] = outcome
    app.mkdir(parents=True, exist_ok=True)
    write_json(app / "state.json", state)
    return outcome
