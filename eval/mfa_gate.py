"""The MFA accuracy gate, run as eval/prereg/mfa_gate.toml says, on the system it runs on.

Usage, from publishable/ (DATA holds edacc/ and voices/ as tools/fetch_test_data.py leaves them; WORK keeps
the built sessions, the jobs and MFA's own output, and its path holds a space and non-ASCII letters, as a
learner's may; OUT gets the results):
  python eval/mfa_gate.py dev --data DATA --splits FILE --work WORK --out OUT
  python eval/mfa_gate.py test --data DATA --splits FILE --set-b DIR --freeze FILE --work WORK --out OUT
Option: --limit N (the first N sessions of each set and no 15-minute one: a small check run, never a result).
Each session is analysed once by the app's command line on the CPU with Parakeet v2 and MFA switched off;
that job gives arm P its words and every arm the voice detector's speech. MFA then runs through the app's
own alignment functions on copies of the job: the app's words (P, twice on the test sessions), the
reference words of each truth clip with the app's CTC times (T), and that reference with words deleted (M).
MFA's own output is kept, so the dev phase can apply the app's check with each pair of limits it tries; the
test phase uses the pair the dev phase chose. metrics.json and per_utterance.csv depend only on MFA's and
the CTC aligner's output; resources.json and report.md also hold times.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL))
import sets  # noqa: E402  (eval/ is a folder of scripts, not a package)
import stats  # noqa: E402
from asr_metrics import FILLER_TAGS, TAG, normalise  # noqa: E402
from asr_metrics import PREREG as ASR_PREREG  # noqa: E402

from fluentfolio.alignment import guard, mfa, phones  # noqa: E402

Row = dict[str, Any]
PREREG_PATH = EVAL / "prereg" / "mfa_gate.toml"
PREREG_FILE = "eval/prereg/mfa_gate.toml"
PREREG = tomllib.loads(PREREG_PATH.read_text(encoding="utf-8"))
SEED = PREREG["seed"]
RULE = PREREG["arms"]["m_rule"]
LIMITS = PREREG["thresholds"]
ASR = ASR_PREREG["candidates"][0]["id"]  # Parakeet TDT 0.6B v2, the app's default
SYSTEM = "windows" if mfa.IS_WINDOWS else "linux"
START = (guard.T1, guard.T2)
ARMS = ("P", "T", "M")
SAMPLE_S = 0.5  # how often a process tree's resident memory is read
TIMEOUT_S = 3600  # one analysis or one MFA call
if mfa.IS_WINDOWS:
    KEPT = ("SYSTEMROOT", "WINDIR", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "USERPROFILE",
            "LOCALAPPDATA", "APPDATA")  # fmt: skip
else:
    KEPT = ("HOME", "LANG", "LC_ALL")
CALIBRATION = "tests/integration/test_alignment.py::test_mfa_times_the_clip_within_40_ms_of_silencedetect"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def quiet(pct: int) -> None:
    """Progress is not shown."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- processes -------------------------------------------------------------------------------------------


def tree_rss(process: psutil.Process) -> int:
    return process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))


def watched(argv: list[str], env: dict[str, str], cwd: Path | None, log: Path) -> tuple[int, float, int]:
    """Run a program, its output to `log`: its exit code, wall time and its tree's peak resident memory."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started, peak = time.monotonic(), 0
    with log.open("wb") as sink:
        child = subprocess.Popen(argv, env=env, cwd=cwd, stdout=sink, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            if time.monotonic() - started > TIMEOUT_S:
                for p in [*process.children(recursive=True), process]:
                    p.kill()
                break
            try:
                peak = max(peak, tree_rss(process))
            except psutil.Error:  # the child ended between the poll and the read
                pass
            time.sleep(SAMPLE_S)
        child.wait()
    return child.returncode, round(time.monotonic() - started, 3), peak


def mfa_env(tool: mfa.Program) -> dict[str, str]:
    """MFA's environment as the app builds one for a step: an allow-list, MFA's env folders first on PATH."""
    env = {name: os.environ[name] for name in KEPT if name in os.environ}
    if mfa.IS_WINDOWS:
        system = [Path(os.environ["SYSTEMROOT"]) / "System32"]
    else:
        system = [Path("/usr/bin"), Path("/bin")]
    env["PATH"] = os.pathsep.join(str(p) for p in (*tool.path, Path(sys.executable).parent, *system))
    tmp = tool.scratch / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env.update(TMP=str(tmp), TEMP=str(tmp), TMPDIR=str(tmp), CUDA_VISIBLE_DEVICES="-1")
    return env | tool.env


# ---- sessions and the app's analysis ---------------------------------------------------------------------


@dataclass
class Session:
    name: str
    folder: Path  # session.wav and truth.json

    @property
    def truth(self) -> Row:
        return read(self.folder / "truth.json")


def sessions(
    phase: str, data: Path, splits: Row, work: Path, set_b: Path | None, limit: int | None = None
) -> list[Session]:
    """The phase's sessions in their registered order, each built once into the work folder.

    With a limit, the first sessions of each set, and no 15-minute session: a small check run.
    """
    root, makers = work / "sessions", {}
    for part in PREREG["data"][phase]:
        if part == "set_a":
            for recipe in sorted(sets.sb.SESSIONS.glob("*.json"))[:limit]:
                makers[recipe.stem] = lambda f, r=recipe: sets.sb.build(r, f)
        elif part == "set_b":
            if set_b is None:
                raise SystemExit("the test phase needs --set-b")
            for n in sorted(p.name for p in set_b.iterdir() if (p / "recipe.json").is_file())[:limit]:
                makers[f"{n}-B"] = lambda f, n=n: sets.build_b(set_b, n, f)
        elif part.startswith("l2_"):
            split = part.removeprefix("l2_")
            speakers = splits["edacc"][split]["l2_interview_speakers"][:limit]
            names = [f"L2-{split}-{k + 1}" for k in range(len(speakers))]
            if not all((root / n / "truth.json").is_file() for n in names):
                rows = sets.l2_rows(data, split, speakers)
                for n, s in zip(names, speakers, strict=True):
                    makers[n] = lambda f, n=n, mine=rows[s]: sets.l2_interview(data, mine, n, f)
            makers.update({n: None for n in names if n not in makers})
        elif not (limit is not None and part == "CS-LONG"):
            makers[part] = lambda f, n=part: sets.sb.build_derived(n, f)
    for name, make in makers.items():
        if make is not None and not (root / name / "truth.json").is_file():
            make(root / name)
    return [Session(name, root / name) for name in makers]


def analysed(session: Session, work: Path) -> Row:
    """The app's own analysis of the session, on the CPU with MFA off: its job folder, time and memory."""
    path = work / "analysis" / f"{session.name}.json"
    if path.is_file():
        return read(path)
    home = work / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.toml").write_text("[components]\nmfa = false\n", encoding="utf-8")
    argv = [sys.executable, "-m", "fluentfolio", "analyze", str(session.folder / "session.wav"), "--mode",
            session.truth["mode"], "--asr", ASR, "--device", "cpu", "--json"]  # fmt: skip
    if session.truth["mode"] != "solo":
        argv.append("--partner-consent")  # the composed voices stand for people who agreed
    env = {**os.environ, "FLUENTFOLIO_HOME": str(home), "CUDA_VISIBLE_DEVICES": "-1"}
    log = work / "analysis" / f"{session.name}.log"
    code, wall_s, peak = watched(argv, env, None, log)
    text = log.read_bytes().decode("utf-8", "replace")
    lines = [line for line in text.splitlines() if line.startswith('{"schema"')]
    envelope = json.loads(lines[-1]) if lines else {"ok": False}
    if code != 0 or not envelope["ok"]:
        raise RuntimeError(f"the analysis of {session.name} failed (exit code {code}); see {log}")
    found = {"job": envelope["data"]["outputs"]["job_dir"], "wall_s": wall_s, "peak_bytes": peak}
    write(path, found)
    return found


# ---- the arms --------------------------------------------------------------------------------------------


def copy_job(base: Path, folder: Path, job_id: str) -> Path:
    """A copy of the job under its own ID (MFA's folder is per job), with MFA switched on."""
    shutil.rmtree(folder, ignore_errors=True)
    shutil.copytree(base, folder)
    record = read(folder / "job.json")
    record["job_id"] = job_id
    record["settings"].setdefault("components", {})["mfa"] = True
    write(folder / "job.json", record)
    return folder


def minus(segments: list[list[int]], spans: list[tuple[int, int]]) -> list[list[int]]:
    """The segments with the spans cut out."""
    out = [list(s) for s in segments]
    for a, b in spans:
        out = [p for s, e in out for p in ([s, min(e, a)], [max(s, b), e]) if p[0] < p[1]]
    return out


def even(words: list[Row], start: int, end: int) -> None:
    """Times shared out by letters, for a clip the CTC aligner could not place."""
    sizes = [max(1, len(w["text"])) for w in words]
    at = start
    for w, size in zip(words, sizes, strict=True):
        w["start_ms"], at = at, at + (end - start) * size // sum(sizes)
        w["end_ms"] = at


def ctc_timed(words: list[Row], found: Row, clip: Row) -> None:
    """The CTC aligner's times for an utterance's words; a word it could not place gets no length."""
    if found["status"] != "ok":
        even(words, clip["start_ms"], clip["end_ms"])
        return
    last = clip["start_ms"]
    for w, t in zip(words, found["words"], strict=True):
        w["start_ms"], w["end_ms"] = (last, last) if t is None else t
        last = w["end_ms"]


def truth_job(job: Path, truth: Row) -> None:
    """Arm T: one utterance per candidate clip with its reference words; other clips' words are neighbours.

    Every word first spans its clip; the app's CTC aligner then times the candidate words on the clip's
    cut, and those times stand in for the speech model's, which the check compares MFA's times with.
    """
    clips = sorted(truth["segments"], key=lambda s: (s["start_ms"], s["end_ms"]))
    words: list[Row] = []
    answers, utterances, own = [], [], {}
    for s in clips:
        first = len(words)
        words += [{"text": t, "start_ms": s["start_ms"], "end_ms": s["end_ms"], "cut_off": False, "flags": [],
                   "speaker": s["speaker"], "confidence": None} for t in s["text"].split()]  # fmt: skip
        if s["role"] == "candidate" and len(words) > first:
            uid = str(s["id"])
            while uid in own:  # a clip used twice in one session
                uid += "+"
            own[uid] = s
            answers.append({"id": uid, "candidate": s["speaker"]})
            utterances.append({"id": uid, "answer_id": uid, "first_word": first, "last_word": len(words) - 1})
    speech = read(job / "stages" / "vad.json")["segments"]
    people = {s["speaker"] for s in own.values()}
    others = {c: [(s["start_ms"], s["end_ms"]) for s in clips if s["speaker"] != c] for c in people}
    heard = {c: minus(speech, spans) for c, spans in others.items()}
    write(job / "stages" / "words.json", {"words": words})
    write(job / "stages" / "structure.json", {"answers": answers, "utterances": utterances, "speech": heard})
    timed = {u["id"]: u for u in phones.run(job, quiet)["utterances"]}
    for u in utterances:
        ctc_timed(words[u["first_word"] : u["last_word"] + 1], timed[u["id"]], own[u["id"]])
    write(job / "stages" / "words.json", {"words": words})
    write(job / "stages" / "phones.json", phones.run(job, quiet))
    write(job / "truth_clips.json", own)


def _order(uid: str, positions: list[int]) -> list[int]:
    return sorted(positions, key=lambda i: hashlib.sha256(f"{uid}:{i}".encode()).hexdigest())


def deletions(session: str, uid: str, n: int) -> list[int]:
    """The word positions arm M deletes from an utterance of n words; an utterance keeps one word at least.

    Interior positions go first, in hash order; an utterance with too few of them loses an edge word too.
    """
    number = int(hashlib.sha256(f"{SEED}:{session}:{uid}".encode()).hexdigest(), 16)
    if number % 100 >= RULE["share_pct"]:
        return []
    order = _order(uid, list(range(1, n - 1))) + _order(uid, sorted({0, n - 1}))
    return sorted(order[: min(1 + number % RULE["words"], n - 1)])


def mismatch_job(job: Path, session: str) -> None:
    """Arm M from a copy of arm T's job: the chosen words deleted, the others keeping their times."""
    words = read(job / "stages" / "words.json")["words"]
    structure = read(job / "stages" / "structure.json")
    gone: set[int] = set()
    lost = {}
    for u in structure["utterances"]:
        drop = deletions(session, u["id"], u["last_word"] - u["first_word"] + 1)
        gone.update(u["first_word"] + i for i in drop)
        lost[u["id"]] = len(drop)
    new = {i: k for k, i in enumerate(i for i in range(len(words)) if i not in gone)}
    for u in structure["utterances"]:
        kept = [new[i] for i in range(u["first_word"], u["last_word"] + 1) if i in new]
        u["first_word"], u["last_word"] = kept[0], kept[-1]
    write(job / "stages" / "words.json", {"words": [w for i, w in enumerate(words) if i not in gone]})
    write(job / "stages" / "structure.json", structure)
    write(job / "stages" / "phones.json", phones.run(job, quiet))
    write(job / "deleted.json", lost)


# ---- MFA and the check -----------------------------------------------------------------------------------


def align(job: Path, keep: Path, app: Path, models: Path) -> Row:
    """MFA on the job through the app's own functions; its output is kept for the check to apply again."""
    found = keep.parent / f"{keep.name}.json"
    if found.is_file():
        return read(found)
    shutil.rmtree(keep, ignore_errors=True)
    shutil.rmtree(mfa.mfa_root(app, read(job / "job.json")["job_id"]), ignore_errors=True)
    tool = mfa.prepare(job, app, models)
    if isinstance(tool, dict):
        result = {"ran": tool, "wall_s": 0.0, "peak_bytes": 0}
    else:
        try:
            log = keep.parent / f"{keep.name}.log"
            code, wall_s, peak = watched(tool.argv, mfa_env(tool), tool.cwd, log)
            if (tool.scratch / "out").is_dir():
                shutil.copytree(tool.scratch / "out", keep)
        finally:
            shutil.rmtree(tool.scratch, ignore_errors=True)
        ran = {"status": "ok" if code == 0 else "failed", "exit_code": code}
        result = {"ran": ran, "wall_s": wall_s, "peak_bytes": peak}
    write(found, result)
    return result


def checked(job: Path, ran: Row, app: Path, models: Path, pairs: list[tuple[float, int]]) -> list[Row]:
    """The app's step result on MFA's kept output, once per pair of check limits (T1, T2)."""
    root = mfa.mfa_root(app, read(job / "job.json")["job_id"])
    if (job.parent / "mfa_out").is_dir():
        shutil.copytree(job.parent / "mfa_out", root / "out", dirs_exist_ok=True)
    try:
        out = []
        for t1, t2 in pairs:
            guard.T1, guard.T2 = t1, t2
            out.append(mfa.finish(job, app, models, ran))
        return out
    finally:
        guard.T1, guard.T2 = START
        shutil.rmtree(root, ignore_errors=True)


def start_of(segment: Row) -> int:
    return int(segment["start_ms"])


def first_last(texts: list[str]) -> tuple[str | None, str | None]:
    """The first and the last normalised word of the texts in order."""
    tokens = [t for text in texts for t in normalise(text)]
    return (tokens[0], tokens[-1]) if tokens else (None, None)


def edge_sound(text: str, end: bool) -> str | None:
    """The sound tag a reference text starts (or ends) with, such as laughter or overlap; a filler tag is a
    word, so it does not count."""
    text = text.strip().lower()
    found = [m.group() for m in TAG.finditer(text) if (m.end() == len(text) if end else m.start() == 0)]
    return found[0] if found and found[0] not in FILLER_TAGS else None


def ref_first_last(texts: list[str]) -> tuple[str | None, str | None]:
    """The reference's first and last word. Where its text starts or ends with a sound tag, the clip's edge
    is that sound, not a word, and no output word equals it, so the utterance is not judged on its edges."""
    first, last = first_last(texts)
    if not texts:
        return first, last
    return edge_sound(texts[0], end=False) or first, edge_sound(texts[-1], end=True) or last


@dataclass
class Arm:
    """One arm's job as the scoring reads it: its words, utterances, CTC rows and truth."""

    session: str
    arm: str
    run: int
    job: Path

    def __post_init__(self) -> None:
        self.words = read(self.job / "stages" / "words.json")["words"]
        utterances = read(self.job / "stages" / "structure.json")["utterances"]
        self.spans = {u["id"]: (u["first_word"], u["last_word"]) for u in utterances}
        self.ctc = {u["id"]: u for u in read(self.job / "stages" / "phones.json")["utterances"]}
        own = self.job / "truth_clips.json"
        self.own = read(own) if own.is_file() else None
        lost = self.job / "deleted.json"
        self.lost = read(lost) if lost.is_file() else {}

    def clips(self, uid: str, truth: list[Row]) -> list[Row]:
        """The truth clips an utterance is judged against: arms T and M their own; arm P the candidate
        clips its words overlap."""
        if self.own is not None:
            return [self.own[uid]]
        first, last = self.spans[uid]
        a, b = self.words[first]["start_ms"], max(w["end_ms"] for w in self.words[first : last + 1])
        return [c for c in truth if c["start_ms"] < b and c["end_ms"] > a]


def utterance_rows(arm: Arm, result: Row, truth: Row) -> list[Row]:
    """One row per utterance MFA was given: how each aligner did, and the truth edges it is judged against."""
    candidate = sorted((s for s in truth["segments"] if s["role"] == "candidate"), key=start_of)
    rows: list[Row] = []
    for u in result["utterances"]:
        given = [w for w in u["words"] if w["method"] != "unaligned"]
        if not given:
            continue
        first, last = arm.spans[u["id"]]
        over = arm.clips(u["id"], candidate)
        ref = ref_first_last([c["text"] for c in over])
        out = first_last([w["text"] for w in arm.words[first : last + 1]])
        timed = arm.ctc.get(u["id"], {"status": "missing", "words": []})
        ctc = [t for t in timed["words"] if t is not None]
        mfa_ok = u["method"] == "mfa"
        rows.append({
            "session": arm.session, "arm": arm.arm, "run": arm.run, "utterance": u["id"],
            "words": len(u["words"]), "deleted": arm.lost.get(u["id"], 0), "method": u["method"],
            "reason": u["reason"], "ctc_status": timed["status"],
            "truth_start": over[0]["start_ms"] if over else None,
            "truth_end": over[-1]["end_ms"] if over else None,
            "ref_first": ref[0], "ref_last": ref[1], "out_first": out[0], "out_last": out[1],
            "mfa_start": given[0]["start_ms"] if mfa_ok else None,
            "mfa_end": given[-1]["end_ms"] if mfa_ok else None,
            "ctc_start": ctc[0][0] if ctc else None, "ctc_end": ctc[-1][1] if ctc else None,
        })  # fmt: skip
    return rows


# ---- the measures ----------------------------------------------------------------------------------------


def passes(r: Row, aligner: str) -> bool:
    """Whether the aligner's times stand for the utterance: MFA's passed the app's check, or CTC's ran."""
    return r["method"] == "mfa" if aligner == "mfa" else r["ctc_status"] == "ok"


def edge_errors(rows: list[Row], aligner: str) -> list[tuple[Row, int, int]]:
    """The passed utterances whose first and last words match the reference, each with the aligner's start
    and end errors against the truth edges, in ms."""
    matched = [r for r in rows if passes(r, aligner) and r[f"{aligner}_start"] is not None
               and r["truth_start"] is not None
               and (r["out_first"], r["out_last"]) == (r["ref_first"], r["ref_last"])]  # fmt: skip
    return [(r, abs(r[f"{aligner}_start"] - r["truth_start"]), abs(r[f"{aligner}_end"] - r["truth_end"]))
            for r in matched]  # fmt: skip


def measures(rows: list[Row], aligner: str) -> dict[str, Any]:
    """The pre-registered measures of one aligner ("mfa" or "ctc") over pooled utterance rows."""
    ok = [passes(r, aligner) for r in rows]
    passed = [r for r, good in zip(rows, ok, strict=True) if good]
    pairs = [(a, b) for _, a, b in edge_errors(rows, aligner)]
    edges = [e for pair in pairs for e in pair]
    silent = sum(max(p) > PREREG["metrics"]["silent_failure_ms"] for p in pairs)
    deleted = [good for r, good in zip(rows, ok, strict=True) if r["deleted"]]
    return {
        "utterances": len(rows), "passed": len(passed),
        "pass_rate": len(passed) / len(rows) if rows else None,
        "edge_matched": len(pairs), "edges": len(edges),
        "median_edge_ms": statistics.median(edges) if edges else None,
        "p90_edge_ms": float(np.percentile(edges, 90)) if edges else None,
        "silent_failure_rate": silent / len(pairs) if pairs else None,
        "deleted_utterances": len(deleted),
        "mismatch_detection": deleted.count(False) / len(deleted) if deleted else None,
    }  # fmt: skip


def grid() -> list[tuple[float, int]]:
    (a1, b1, s1), (a2, b2, s2) = PREREG["guard_fit"]["t1_grid"], PREREG["guard_fit"]["t2_grid"]
    t1 = [round(a1 + k * s1, 2) for k in range(round((b1 - a1) / s1) + 1)]
    return [(x, y) for x in t1 for y in range(a2, b2 + 1, s2)]


def choose(table: list[Row]) -> Row:
    """The fitted pair: the qualifying pair with the highest arm-P pass rate, ties to the smaller T1, then T2.

    A pair qualifies with arm-M mismatch detection of 80 % or more and arm-P silent failures of 2 % or less;
    when none does, the starting values stay.
    """
    ok = [r for r in table if r["m_detection"] is not None and r["p_silent"] is not None
          and r["m_detection"] >= LIMITS["mismatch_detection_min"]
          and r["p_silent"] <= LIMITS["silent_failure_rate_max"]]  # fmt: skip
    if not ok:
        start = PREREG["guard_fit"]["start"]
        return {"t1": start["t1"], "t2": start["t2"], "fitted": False}
    best = min(ok, key=lambda r: (-r["p_pass"], r["t1"], r["t2"]))
    return {"t1": best["t1"], "t2": best["t2"], "fitted": True}


def decide(p: Row, m: Row, same: dict[str, bool], long_run: Row | None) -> tuple[str, list[str]]:
    """PASS when every threshold holds (arm P, and arm M for detection), with the lines it rests on."""
    gb = None if long_run is None else long_run["peak_bytes"] / 1e9
    checks = [  # name, value, limit, and 1 when the value must be at least the limit, -1 at most
        ("guard pass rate", p["pass_rate"], LIMITS["guard_pass_rate_min"], 1),
        ("median clip-edge error ms", p["median_edge_ms"], LIMITS["median_edge_error_ms_max"], -1),
        ("silent-failure rate", p["silent_failure_rate"], LIMITS["silent_failure_rate_max"], -1),
        ("mismatch detection (arm M)", m["mismatch_detection"], LIMITS["mismatch_detection_min"], 1),
        ("CS-LONG peak RAM of MFA's process tree, GB", gb, LIMITS["long_session_peak_ram_gb_max"], -1),
    ]
    lines, ok = [], True
    for name, value, limit, sign in checks:
        held = value is not None and (value - limit) * sign >= 0
        ok &= held
        shown = "n/a" if value is None else round(value, 4)
        bound = "at least" if sign > 0 else "at most"
        lines.append(f"{name}: {shown} ({bound} {limit}): {'holds' if held else 'FAILS'}")
    steady = bool(same) and all(same.values())
    completed = long_run is not None and long_run["ran"]["status"] == "ok"
    lines.append(f"determinism: {sum(same.values())} of {len(same)} sessions gave identical word times in "
                 f"{LIMITS['determinism_runs']} runs: {'holds' if steady else 'FAILS'}")  # fmt: skip
    lines.append(f"CS-LONG completes: {'holds' if completed else 'FAILS'}")
    return ("PASS" if ok and steady and completed else "FAIL"), lines


def outcome(
    test: bool, calibrated: bool, measured: Row, same: dict[str, bool], long_run: Row | None
) -> tuple[str, list[str]]:
    """The report's decision line and the lines it rests on; the dev phase decides nothing."""
    verdict, lines = decide(measured["P"]["mfa"], measured["M"]["mfa"], same, long_run)
    if not test:
        lines = [line for line in lines if not line.startswith(("CS-LONG", "determinism"))]
        return "Decision: none: the dev phase fits the check's limits only", [
            *lines, "determinism and CS-LONG: measured in the test phase only"]  # fmt: skip
    if not calibrated:
        return (
            "Decision: none: the harness calibration failed, so no decision until the cause is found",
            lines,
        )
    return f"Decision: {verdict}", lines


def _value(rows: list[Row], aligner: str, name: str) -> Callable[[list[str]], float]:
    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(r["session"], []).append(r)

    def value(chosen: list[str]) -> float:
        found = measures([r for s in chosen for r in groups[s]], aligner)[name]
        return float("nan") if found is None else float(found)

    return value


def interval(rows: list[Row], aligner: str, name: str) -> tuple[float, float, float] | None:
    """The measure and its 95 % percentile interval from a bootstrap over sessions."""
    if measures(rows, aligner)[name] is None:
        return None
    method, units = PREREG["statistics"], sorted({r["session"] for r in rows})
    return stats.bootstrap(units, _value(rows, aligner, name), method["resamples"], method["seed"])


def difference(rows: list[Row], name: str) -> tuple[float, float, float] | None:
    """MFA minus the CTC aligner on the same sessions, with its paired 95 % interval."""
    if None in (measures(rows, "mfa")[name], measures(rows, "ctc")[name]):
        return None
    method, units = PREREG["statistics"], sorted({r["session"] for r in rows})
    first, second = _value(rows, "mfa", name), _value(rows, "ctc", name)
    return stats.paired(units, first, second, method["resamples"], method["seed"])


def arm_measures(rows: list[Row]) -> dict[str, Any]:
    """Each arm's measures for both aligners (arm P from its first run), with intervals over sessions."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm and r["run"] == 1]
        found: dict[str, Any] = {a: measures(mine, a) for a in ("mfa", "ctc")}
        names = ("pass_rate", "median_edge_ms", "silent_failure_rate", "mismatch_detection")
        found["intervals"] = {f"{a}_{n}": interval(mine, a, n) for a in ("mfa", "ctc") for n in names}
        found["mfa_minus_ctc"] = {n: difference(mine, n) for n in ("pass_rate", "median_edge_ms")}
        out[arm] = found
    return out


def word_tiers(keep: Path) -> dict[str, list[tuple[int, int, str]]]:
    """Every word interval MFA wrote, per TextGrid, before any check."""
    grids = sorted(keep.rglob("*.TextGrid"))
    return {p.relative_to(keep).as_posix(): mfa.textgrid(p, 0)["words"] for p in grids}


def determinism(found: list[Run]) -> dict[str, bool]:
    """Per session, whether MFA's two runs of arm P wrote the same word intervals (and wrote some)."""
    kept = {(a.session, a.arm, a.run): a.job.parent / "mfa_out" for _, a, _ in found}
    out = {}
    for name in dict.fromkeys(a.session for _, a, _ in found):
        one, two = word_tiers(kept[(name, "P", 1)]), word_tiers(kept[(name, "P", 2)])
        out[name] = bool(one) and one == two
    return out


def calibration(data: Path, work: Path) -> tuple[bool, str]:
    """The app's VOiCES test: MFA's first word start and last word end within 40 ms of silencedetect's."""
    env = {**os.environ, "FF_TEST_DATA": str(data), "CUDA_VISIBLE_DEVICES": "-1"}
    argv = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-s", "-m", "slow", CALIBRATION]
    code, _, _ = watched(argv, env, EVAL.parent, work / "calibration.log")
    text = (work / "calibration.log").read_bytes().decode("utf-8", "replace")
    found = re.search(r"MFA (\d+)-(\d+) ms, silencedetect (\d+)-(\d+) ms", text)
    if found is None:
        return False, f"calibration: the test printed no edges (exit code {code}): FAILS"
    a, b, c, d = (int(x) for x in found.groups())
    worst, most = max(abs(a - c), abs(b - d)), PREREG["calibration"]["voices_edge_ms_max"]
    ok = code == 0 and worst <= most
    return ok, (f"calibration: MFA {a}-{b} ms, silencedetect {c}-{d} ms on the VOiCES clip, the largest edge "
                f"difference {worst} ms (at most {most}), test exit code {code}: "
                f"{'holds' if ok else 'FAILS'}")  # fmt: skip


# ---- the phases ------------------------------------------------------------------------------------------


def job_for(session: Session, base: Path, work: Path, arm: str, run: int) -> Path:
    """The arm's job folder, made once."""
    folder = work / "runs" / session.name / f"{arm}{run}"
    job = folder / "job"
    if (folder / "ready").is_file():
        return job
    source = work / "runs" / session.name / "T1" / "job" if arm == "M" else base
    copy_job(source, job, f"e2-{session.name}-{arm}{run}")
    if arm == "T":
        truth_job(job, session.truth)
    elif arm == "M":
        mismatch_job(job, session.name)
    (folder / "ready").write_text("", encoding="utf-8")
    return job


Run = tuple[Session, Arm, Row]  # the session, the arm's job, and MFA's run on it


def run_all(
    chosen: list[Session], runs: list[tuple[str, int]], work: Path, app: Path, models: Path
) -> tuple[list[Run], list[Row]]:
    """Every session's analysis and every arm's MFA run, resumed from what is saved."""
    found, resources = [], []
    for session in chosen:
        product = analysed(session, work)
        for arm, run in runs:
            job = job_for(session, Path(product["job"]), work, arm, run)
            ran = align(job, job.parent / "mfa_out", app, models)
            found.append((session, Arm(session.name, arm, run, job), ran))
            resources.append({
                "session": session.name, "arm": arm, "run": run, "mfa": ran["ran"]["status"],
                "mfa_s": ran["wall_s"], "mfa_gb": round(ran["peak_bytes"] / 1e9, 3),
                "analysis_s": product["wall_s"], "analysis_gb": round(product["peak_bytes"] / 1e9, 3),
            })  # fmt: skip
            print(f"{session.name} {arm}{run}: MFA {ran['ran']['status']} in {ran['wall_s']} s", flush=True)
    return found, resources


def all_rows(found: list[Run], pairs: list[tuple[float, int]], app: Path, models: Path) -> list[list[Row]]:
    """Every utterance row of every run, once per pair of check limits."""
    out: list[list[Row]] = [[] for _ in pairs]
    for session, arm, ran in found:
        truth = session.truth
        for k, result in enumerate(checked(arm.job, ran["ran"], app, models, pairs)):
            out[k] += utterance_rows(arm, result, truth)
    return out


def fit(per_pair: list[list[Row]], pairs: list[tuple[float, int]]) -> tuple[Row, list[Row]]:
    """The dev phase's table of every pair of limits, and the pair it chooses."""
    table = []
    for (t1, t2), rows in zip(pairs, per_pair, strict=True):
        p = measures([r for r in rows if r["arm"] == "P"], "mfa")
        m = measures([r for r in rows if r["arm"] == "M"], "mfa")
        table.append({"t1": t1, "t2": t2, "p_pass": p["pass_rate"], "p_silent": p["silent_failure_rate"],
                      "m_detection": m["mismatch_detection"]})  # fmt: skip
    return choose(table), table


COLUMNS = ["session", "arm", "run", "utterance", "words", "deleted", "method", "reason", "ctc_status",
           "truth_start", "truth_end", "ref_first", "ref_last", "out_first", "out_last", "mfa_start",
           "mfa_end", "ctc_start", "ctc_end"]  # fmt: skip


def write_rows(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any) -> str:
    return "n/a" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def silent_lines(rows: list[Row]) -> list[str]:
    """Arm P's silent failures in its first run, with the CTC aligner's errors on the same utterances: when
    both sit far from the truth edge, that edge may hold a sound the reference text lacks."""
    limit = PREREG["metrics"]["silent_failure_ms"]
    first = [r for r in rows if r["arm"] == "P" and r["run"] == 1]
    lines = []
    for r, a, b in edge_errors(first, "mfa"):
        if max(a, b) > limit:
            ctc = "n/a" if r["ctc_start"] is None else (
                f"{abs(r['ctc_start'] - r['truth_start'])} / {abs(r['ctc_end'] - r['truth_end'])}"
            )  # fmt: skip
            lines.append(f"| {r['session']} | {r['utterance']} | {a} / {b} | {ctc} |")
    head = [f"## Arm P's silent failures (an edge error over {limit} ms), with the CTC aligner's errors", ""]
    if not lines:
        return [*head, "none"]
    return [*head, "| session | utterance | MFA start / end error, ms | CTC start / end error, ms |",
            "|---|---|---|---|", *lines]  # fmt: skip


def tables(found: dict[str, Any], resources: list[Row]) -> list[str]:
    """The measures per arm and aligner, their intervals, and each run's time and memory."""
    names = ["utterances", "passed", "pass_rate", "edge_matched", "median_edge_ms", "p90_edge_ms",
             "silent_failure_rate", "deleted_utterances", "mismatch_detection"]  # fmt: skip
    lines = ["| arm | aligner | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 2)]
    lines += [f"| {arm} | {a} | " + " | ".join(fmt(found[arm][a][n]) for n in names) + " |"
              for arm in ARMS for a in ("mfa", "ctc")]  # fmt: skip
    lines += ["", "95 % intervals over sessions, and MFA minus CTC with its paired interval:", ""]
    for arm in ARMS:
        shown = dict(found[arm]["intervals"])
        shown.update({f"mfa_minus_ctc_{k}": v for k, v in found[arm]["mfa_minus_ctc"].items()})
        cells = [f"{k} {fmt(v[0])} [{fmt(v[1])}, {fmt(v[2])}]" for k, v in shown.items() if v is not None]
        lines.append(f"- arm {arm}: " + ("; ".join(cells) or "none"))
    keys = ["session", "arm", "run", "mfa", "mfa_s", "mfa_gb", "analysis_s", "analysis_gb"]
    lines += ["", "## Each run's wall time (s) and its process tree's peak resident memory (GB of 1e9 bytes)",
              "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]  # fmt: skip
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in resources]
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["dev", "test"])
    for name in ("--data", "--splits", "--work", "--out"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--set-b", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if " " not in str(args.work) or str(args.work).isascii():
        print("the work folder's path must hold a space and non-ASCII letters, as the setup asks")
        return 2
    fit_path, test = args.out / "dev_fit.json", args.phase == "test"
    frozen = read(args.freeze)["files"].get(PREREG_FILE) if args.freeze else None
    if test and (frozen != sha256(PREREG_PATH) or not fit_path.is_file()):
        print(f"the pre-registration's hash must be the frozen one ({frozen}), and dev_fit.json must exist")
        return 2
    app, models = Path(os.environ["FLUENTFOLIO_APP_ROOT"]), phones.models_dir()
    args.out.mkdir(parents=True, exist_ok=True)
    calibrated, calibration_line = calibration(args.data, args.work)
    print(calibration_line, flush=True)
    chosen = sessions(args.phase, args.data, read(args.splits), args.work, args.set_b, args.limit)
    runs = [("P", 1), ("T", 1), ("M", 1), *([("P", 2)] if test else [])]
    found, resources = run_all(chosen, runs, args.work, app, models)
    extra = {"prereg_sha256": sha256(PREREG_PATH), "system": SYSTEM, "limit": args.limit,
             "calibrated": calibrated}  # fmt: skip
    if test:
        pick = read(fit_path)["chosen"]
        rows = all_rows(found, [(pick["t1"], pick["t2"])], app, models)[0]
    else:
        pairs = grid()
        per_pair = all_rows(found, pairs, app, models)
        pick, table = fit(per_pair, pairs)
        write(fit_path, {**extra, "chosen": pick, "table": table})
        rows = per_pair[pairs.index((pick["t1"], pick["t2"]))]
    stem = "" if test else "_dev"
    write_rows(args.out / f"per_utterance{stem}.csv", rows)
    measured = arm_measures(rows)
    same = determinism(found) if test else {}
    write(args.out / f"metrics{stem}.json", {**extra, "guard": pick, "arms": measured, "determinism": same})
    write(args.out / f"resources{stem}.json", resources)
    long_run = next((ran for s, a, ran in found if s.name == "CS-LONG" and (a.arm, a.run) == ("P", 1)), None)
    decision, lines = outcome(test, calibrated, measured, same, long_run)
    where = "fitted on dev" if pick["fitted"] else "the starting values: no pair qualified on dev"
    report = [f"prereg_sha256: {sha256(PREREG_PATH)}", decision, "", f"# MFA accuracy gate on {SYSTEM}", ""]
    if args.limit is not None:
        report += [f"A small check run on the first {args.limit} sessions of each set: not a result.", ""]
    report += [f"- {calibration_line}", f"- the check's limits: T1 {pick['t1']}, T2 {pick['t2']} ({where})",
               *[f"- {line}" for line in lines], "", *tables(measured, resources), "",
               *silent_lines(rows), ""]  # fmt: skip
    (args.out / f"report{stem}.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:2] + lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
