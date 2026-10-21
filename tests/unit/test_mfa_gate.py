"""The MFA gate's harness: its arms, its measures and its decision, on hand-made jobs and rows."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fluentfolio.alignment import guard, mfa

EVAL = Path(__file__).resolve().parents[2] / "eval"


def _load(name: str) -> Any:
    # eval/ is a folder of scripts, not a package; a script that imports a sibling needs it registered.
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


_load("asr_metrics")
gate = _load("mfa_gate")
PREREG = tomllib.loads((EVAL / "prereg" / "mfa_gate.toml").read_text(encoding="utf-8"))


def put(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_the_check_s_starting_limits_and_the_m_rule_are_the_pre_registered_ones() -> None:
    assert (guard.T1, guard.T2) == (PREREG["guard_fit"]["start"]["t1"], PREREG["guard_fit"]["start"]["t2"])
    assert PREREG["arms"]["m_rule"] == {"share_pct": 30, "words": 3}  # the M text: 30 %, 1 + (number % 3)
    assert PREREG["metrics"]["silent_failure_ms"] == 300  # the silent-failure text: over 300 ms
    pairs = gate.grid()
    assert len(pairs) == 9 * 7 and pairs[0] == (4.0, 2) and pairs[-1] == (8.0, 8) and (5.5, 3) in pairs


def test_arm_m_deletes_by_the_pre_registered_hash_rule() -> None:
    chosen = 0
    for k in range(600):
        uid, n = f"u{k}", 2 + k % 9
        number = int(hashlib.sha256(f"20261013:CS-X:{uid}".encode()).hexdigest(), 16)
        found = gate.deletions("CS-X", uid, n)
        if number % 100 >= 30:
            assert found == []
            continue
        chosen += 1
        wanted = min(1 + number % 3, n - 1)

        def by_hash(positions: set[int], uid: str = uid) -> list[int]:
            return sorted(positions, key=lambda i: hashlib.sha256(f"{uid}:{i}".encode()).hexdigest())

        interior = by_hash(set(range(1, n - 1)))
        assert len(found) == wanted and len(found) < n  # one word is always left to align
        # Interior words first, in hash order; with too few of them an edge word goes too.
        assert found == sorted((interior + by_hash({0, n - 1}))[:wanted])
        assert set(found) >= set(interior[:wanted])
    assert 0.25 < chosen / 600 < 0.35
    assert gate.deletions("CS-X", "any", 1) == []  # a one-word utterance keeps its word


def test_spans_are_cut_out_of_segments() -> None:
    assert gate.minus([[0, 1000], [1500, 3000]], [(800, 1600), (2000, 2100)]) == [
        [0, 800], [1600, 2000], [2100, 3000]
    ]  # fmt: skip
    assert gate.minus([[100, 200]], [(0, 300)]) == []


def test_ctc_times_are_taken_and_an_unplaced_word_gets_no_length() -> None:
    words = [{"text": t, "start_ms": 0, "end_ms": 0} for t in ("so", "-", "yes")]
    clip = {"start_ms": 1000, "end_ms": 2000}
    gate.ctc_timed(words, {"status": "ok", "words": [[1100, 1300], None, [1500, 1900]]}, clip)
    assert [(w["start_ms"], w["end_ms"]) for w in words] == [(1100, 1300), (1300, 1300), (1500, 1900)]
    gate.ctc_timed(words, {"status": "failed", "words": []}, clip)  # shared out by letters: 2, 1 and 3
    assert [(w["start_ms"], w["end_ms"]) for w in words] == [(1000, 1333), (1333, 1499), (1499, 1999)]


def _job(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    job = tmp_path / "job"
    put(job / "job.json", {"job_id": "j1", "mode": "interview", "settings": {"components": {}}})
    put(job / "stages" / "media.json", {"duration_ms": 9000})
    put(job / "stages" / "vad.json", {"segments": [[0, 1100], [1950, 6100], [6400, 8000]]})
    truth = {"segments": [
        {"id": "q1", "role": "examiner", "speaker": "E", "start_ms": 0, "end_ms": 1000,
         "text": "Do you cook?"},
        {"id": "a1", "role": "candidate", "speaker": "C", "start_ms": 2000, "end_ms": 4000,
         "text": "Yes, I cook every day"},
        {"id": "a1", "role": "candidate", "speaker": "C", "start_ms": 4500, "end_ms": 6000,
         "text": "mostly rice"},
        {"id": "q2", "role": "examiner", "speaker": "E", "start_ms": 6500, "end_ms": 8000, "text": "Why?"},
    ]}  # fmt: skip
    return job, truth


def test_arm_t_writes_one_utterance_per_candidate_clip_that_the_app_s_cuts_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, truth = _job(tmp_path)
    times = {"a1": [[2050, 2300], [2400, 2500], [2600, 2900], [3000, 3400], [3500, 3900]],
             "a1+": [[4550, 5100], [5200, 5900]]}  # fmt: skip
    calls: list[int] = []

    def ctc(path: Path, progress: Any) -> dict[str, Any]:
        calls.append(1)
        return {
            "status": "ok",
            "utterances": [{"id": k, "status": "ok", "words": v} for k, v in times.items()],
        }

    monkeypatch.setattr(gate.phones, "run", ctc)
    gate.truth_job(job, truth)
    structure = json.loads((job / "stages" / "structure.json").read_text(encoding="utf-8"))
    # The clip used twice gets its own ID; each candidate clip is one answer and one utterance.
    assert [u["id"] for u in structure["utterances"]] == ["a1", "a1+"] and len(calls) == 2
    # The other speaker's clips (0-1000 and 6500-8000) are cut out of the voice detector's speech.
    assert structure["speech"] == {"C": [[1000, 1100], [1950, 6100], [6400, 6500]]}
    cuts = mfa.cuts(job)
    assert [c.tokens for c in cuts] == [
        [["yes"], ["i"], ["cook"], ["every"], ["day"]],
        [["mostly"], ["rice"]],
    ]
    # Windows: the CTC times padded by 200 ms, as half the gap to each neighbour word is more than that.
    assert [(c.start_ms, c.end_ms) for c in cuts] == [(1850, 4100), (4350, 6100)]
    assert all(c.speaker == "C" for c in cuts)
    own = json.loads((job / "truth_clips.json").read_text(encoding="utf-8"))
    assert own["a1+"]["text"] == "mostly rice"


def test_arm_m_deletes_words_from_arm_t_and_keeps_the_others_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, truth = _job(tmp_path)
    times = {"a1": [[2050, 2300], [2400, 2500], [2600, 2900], [3000, 3400], [3500, 3900]],
             "a1+": [[4550, 5100], [5200, 5900]]}  # fmt: skip
    monkeypatch.setattr(gate.phones, "run", lambda path, progress: {
        "utterances": [{"id": k, "status": "ok", "words": v} for k, v in times.items()]})  # fmt: skip
    gate.truth_job(job, truth)
    monkeypatch.setattr(gate, "deletions", lambda session, uid, n: [1, 3] if uid == "a1" else [])
    gate.mismatch_job(job, "CS-X")
    cuts = mfa.cuts(job)
    assert [c.tokens for c in cuts] == [[["yes"], ["cook"], ["day"]], [["mostly"], ["rice"]]]
    words = json.loads((job / "stages" / "words.json").read_text(encoding="utf-8"))["words"]
    assert [(w["text"], w["start_ms"]) for w in words if w["speaker"] == "C"] == [
        ("Yes,", 2050), ("cook", 2600), ("day", 3500), ("mostly", 4550), ("rice", 5200)
    ]  # fmt: skip
    assert json.loads((job / "deleted.json").read_text(encoding="utf-8")) == {"a1": 2, "a1+": 0}


def _rows_job(tmp_path: Path, own: bool) -> Any:
    job = tmp_path / ("own" if own else "app")
    words = [{"text": t, "start_ms": a, "end_ms": b} for t, a, b in
             [("Yes", 2010, 2300), ("I", 2400, 2500), ("cook", 2600, 3900), ("Mostly", 4560, 5100),
              ("rice.", 5200, 5990)]]  # fmt: skip
    put(job / "stages" / "words.json", {"words": words})
    utterances = [
        {"id": "u1", "first_word": 0, "last_word": 2},
        {"id": "u2", "first_word": 3, "last_word": 4},
    ]
    put(job / "stages" / "structure.json", {"utterances": utterances})
    put(job / "stages" / "phones.json", {"utterances": [
        {"id": "u1", "status": "ok", "words": [[2000, 2290], None, [2610, 3700]]},
        {"id": "u2", "status": "failed", "words": []}]})  # fmt: skip
    if own:
        clips = {
            "u1": {"start_ms": 2000, "end_ms": 4000, "text": "Yes I cook"},
            "u2": {"start_ms": 4500, "end_ms": 6000, "text": "mostly rice"},
        }
        put(job / "truth_clips.json", clips)
        put(job / "deleted.json", {"u1": 0, "u2": 1})
    return gate.Arm("S", "T" if own else "P", 1, job)


RESULT = {"utterances": [
    {"id": "u1", "method": "mfa", "reason": None, "words": [
        {"i": 0, "start_ms": 2020, "end_ms": 2300, "method": "mfa"},
        {"i": 1, "start_ms": 2400, "end_ms": 2500, "method": "mfa"},
        {"i": 2, "start_ms": 2600, "end_ms": 4301, "method": "mfa"}]},
    {"id": "u2", "method": "ctc", "reason": "long_word", "words": [
        {"i": 3, "start_ms": 4560, "end_ms": 5100, "method": "asr"},
        {"i": 4, "start_ms": 5200, "end_ms": 5990, "method": "asr"}]},
]}  # fmt: skip
TRUTH = {"segments": [
    {"role": "examiner", "start_ms": 0, "end_ms": 1000, "text": "Do you cook"},
    {"role": "candidate", "start_ms": 2000, "end_ms": 3000, "text": "Yes I"},
    {"role": "candidate", "start_ms": 3200, "end_ms": 4000, "text": "cook"},
    {"role": "candidate", "start_ms": 4500, "end_ms": 6000, "text": "mostly rice"},
]}  # fmt: skip


def test_an_app_utterance_is_judged_against_the_candidate_clips_it_overlaps(tmp_path: Path) -> None:
    rows = gate.utterance_rows(_rows_job(tmp_path, own=False), RESULT, TRUTH)
    first, second = rows
    # u1's words span 2010-3900: two candidate clips, 2000-3000 "Yes I" and 3200-4000 "cook".
    assert (first["truth_start"], first["truth_end"], first["ref_first"], first["ref_last"]) == (
        2000,
        4000,
        "yes",
        "cook",
    )
    assert (first["out_first"], first["out_last"], first["mfa_start"], first["mfa_end"]) == (
        "yes",
        "cook",
        2020,
        4301,
    )
    assert (first["ctc_status"], first["ctc_start"], first["ctc_end"]) == ("ok", 2000, 3700)
    assert (second["method"], second["mfa_start"], second["ctc_status"], second["ctc_start"]) == (
        "ctc", None, "failed", None
    )  # fmt: skip
    assert second["out_last"] == "rice" and second["deleted"] == 0


@pytest.mark.parametrize(
    ("texts", "edges"),
    [
        (["SO SEE YOU ON VACATION <LAUGH>"], ("so", "<laugh>")),
        (["<LAUGH> THAT'S NOT GOOD"], ("<laugh>", "good")),
        (["STICKY <LAUGH> TOO SUNNY TOO STICKY"], ("sticky", "sticky")),  # a sound inside is no edge
        (["WHERE YOU LAND AND <NO-SPEECH> <OVERLAP>"], ("where", "<overlap>")),
        (["<laugh> yeah", "fine thanks <LAUGH> "], ("<laugh>", "<laugh>")),  # the first and the last clip
        (["[UH] I THINK SO [UM]"], ("uh", "um")),  # a filler tag is a word
        (["I THINK SO"], ("i", "so")),
        ([], (None, None)),
    ],
)
def test_a_reference_that_starts_or_ends_with_a_sound_has_that_sound_as_its_edge(
    texts: list[str], edges: tuple[str | None, str | None]
) -> None:
    assert gate.ref_first_last(texts) == edges


def test_an_utterance_whose_clip_ends_in_laughter_is_not_judged_on_its_edges(tmp_path: Path) -> None:
    laughing = {"segments": [{**s, "text": s["text"] + " <LAUGH>"} if s["text"] == "cook" else s
                             for s in TRUTH["segments"]]}  # fmt: skip
    rows = gate.utterance_rows(_rows_job(tmp_path, own=False), RESULT, laughing)
    assert (rows[0]["ref_last"], rows[0]["out_last"]) == ("<laugh>", "cook")
    found = gate.measures(rows, "mfa")
    assert (found["passed"], found["edge_matched"], found["median_edge_ms"]) == (1, 0, None)


def test_the_report_lists_each_silent_failure_of_arm_p_with_the_ctc_errors() -> None:
    def row(utterance: str, end: int, arm: str = "P", run: int = 1, ctc: int | None = 2000) -> dict[str, Any]:
        return {"session": "S", "arm": arm, "run": run, "utterance": utterance, "method": "mfa",
                "ctc_status": "ok", "truth_start": 2000, "truth_end": 4000, "ref_first": "yes",
                "ref_last": "cook", "out_first": "yes", "out_last": "cook", "mfa_start": 2010, "mfa_end": end,
                "ctc_start": ctc, "ctc_end": None if ctc is None else 4372}  # fmt: skip

    rows = [row("u1", 4352), row("u2", 4300), row("u3", 4400, arm="T"), row("u4", 4400, run=2),
            row("u5", 3600, ctc=None)]  # fmt: skip
    lines = gate.silent_lines(rows)
    assert lines[0] == "## Arm P's silent failures (an edge error over 300 ms), with the CTC aligner's errors"
    # u1 is 352 ms off at its end; u2 is exactly 300 ms off, which is not over the limit.
    assert lines[4:] == ["| S | u1 | 10 / 352 | 0 / 372 |", "| S | u5 | 10 / 400 | n/a |"]
    assert gate.silent_lines([row("u2", 4300)])[2:] == ["none"]


def test_the_measures_count_passes_edges_silent_failures_and_detections(tmp_path: Path) -> None:
    rows = gate.utterance_rows(_rows_job(tmp_path, own=True), RESULT, TRUTH)
    found = gate.measures(rows, "mfa")
    # u1 passed and its edge words match: |2020 - 2000| = 20 and |4301 - 4000| = 301 > 300, a silent failure.
    assert (found["utterances"], found["passed"], found["pass_rate"], found["edge_matched"]) == (2, 1, 0.5, 1)
    assert (found["median_edge_ms"], found["silent_failure_rate"]) == (160.5, 1.0)
    assert found["p90_edge_ms"] == pytest.approx(20 + 0.9 * 281)
    # u2 lost a word and MFA's times were refused for it: detected.
    assert (found["deleted_utterances"], found["mismatch_detection"]) == (1, 1.0)
    ctc = gate.measures(rows, "ctc")  # u1's CTC edges 0 and 300 ms: at the limit, not over it
    assert (ctc["passed"], ctc["median_edge_ms"], ctc["silent_failure_rate"], ctc["mismatch_detection"]) == (
        1, 150.0, 0.0, 1.0
    )  # fmt: skip
    rows[0]["out_first"] = "so"  # an edge word the reference does not start with: no edges
    assert gate.measures(rows, "mfa")["edge_matched"] == 0


def _pair(
    t1: float, t2: int, p_pass: float, p_silent: float | None, m_detection: float | None
) -> dict[str, Any]:
    return {"t1": t1, "t2": t2, "p_pass": p_pass, "p_silent": p_silent, "m_detection": m_detection}


def test_the_fit_takes_the_qualifying_pair_with_the_highest_pass_rate_and_ties_to_smaller_limits() -> None:
    table = [_pair(4.0, 2, 0.99, 0.03, 0.9), _pair(4.5, 3, 0.95, 0.02, 0.8), _pair(5.0, 2, 0.95, 0.0, 0.85),
             _pair(4.5, 4, 0.95, 0.01, 0.79), _pair(6.0, 2, 0.96, 0.01, None)]  # fmt: skip
    # 4.0/2 fails on silent failures, 4.5/4 on detection, 6.0/2 has none; 4.5/3 and 5.0/2 tie at 0.95.
    assert gate.choose(table) == {"t1": 4.5, "t2": 3, "fitted": True}
    assert gate.choose(table[:1]) == {"t1": 5.0, "t2": 4, "fitted": False}  # none qualifies: the start


P_OK = {"pass_rate": 0.8, "median_edge_ms": 50, "silent_failure_rate": 0.02}
M_OK = {"mismatch_detection": 0.8}
LONG_OK = {"ran": {"status": "ok"}, "peak_bytes": 3_000_000_000}


@pytest.mark.parametrize(
    ("p", "m", "same", "long_run", "verdict"),
    [(P_OK, M_OK, {"a": True}, LONG_OK, "PASS"),
     ({**P_OK, "pass_rate": 0.79}, M_OK, {"a": True}, LONG_OK, "FAIL"),
     ({**P_OK, "median_edge_ms": 50.5}, M_OK, {"a": True}, LONG_OK, "FAIL"),
     ({**P_OK, "silent_failure_rate": 0.021}, M_OK, {"a": True}, LONG_OK, "FAIL"),
     ({**P_OK, "median_edge_ms": None}, M_OK, {"a": True}, LONG_OK, "FAIL"),
     (P_OK, {"mismatch_detection": 0.79}, {"a": True}, LONG_OK, "FAIL"),
     (P_OK, M_OK, {"a": True, "b": False}, LONG_OK, "FAIL"),
     (P_OK, M_OK, {}, LONG_OK, "FAIL"),
     (P_OK, M_OK, {"a": True}, {**LONG_OK, "peak_bytes": 3_000_000_001}, "FAIL"),
     (P_OK, M_OK, {"a": True}, {**LONG_OK, "ran": {"status": "failed"}}, "FAIL"),
     (P_OK, M_OK, {"a": True}, None, "FAIL")],
)  # fmt: skip
def test_the_gate_passes_only_when_every_threshold_holds_at_its_limit(
    p: dict[str, Any], m: dict[str, Any], same: dict[str, bool], long_run: dict[str, Any] | None, verdict: str
) -> None:
    found, lines = gate.decide(p, m, same, long_run)
    assert found == verdict and len(lines) == 7


def test_only_a_calibrated_test_phase_decides() -> None:
    measured = {"P": {"mfa": P_OK}, "M": {"mfa": M_OK}}
    assert gate.outcome(True, True, measured, {"a": True}, LONG_OK)[0] == "Decision: PASS"
    assert gate.outcome(True, False, measured, {"a": True}, LONG_OK)[0].startswith(
        "Decision: none: the harness"
    )
    decision, lines = gate.outcome(False, True, measured, {}, None)
    assert decision == "Decision: none: the dev phase fits the check's limits only"
    # The dev phase has no second run and no 15-minute session, so those lines are not judged there.
    assert len(lines) == 5 and not any(x.startswith(("determinism:", "CS-LONG")) for x in lines)
    assert lines[-1] == "determinism and CS-LONG: measured in the test phase only"


GRID = """item [1]:
    name = "words"
        intervals [1]:
            xmin = 0.0
            xmax = 0.25
            text = "{word}"
item [2]:
    name = "phones"
        intervals [1]:
            xmin = 0.0
            xmax = 0.25
            text = "HH"
"""


def test_two_runs_are_the_same_only_when_every_word_interval_is(tmp_path: Path) -> None:
    def found(*words: str) -> list[Any]:
        out = []
        for run, word in enumerate(words, start=1):
            keep = tmp_path / "runs" / "S" / f"P{run}" / "mfa_out"
            (keep / "C").mkdir(parents=True, exist_ok=True)
            if word:
                (keep / "C" / "u1.TextGrid").write_text(GRID.format(word=word), encoding="utf-8")
            out.append((None, SimpleNamespace(session="S", arm="P", run=run, job=keep.parent / "job"), {}))
        return out

    assert gate.determinism(found("hello", "hello")) == {"S": True}
    assert gate.determinism(found("hello", "yellow")) == {"S": False}
    for grid in (tmp_path / "runs").rglob("*.TextGrid"):
        grid.unlink()
    assert gate.determinism(found("", "")) == {"S": False}  # no output at all is not a repeatable run


def test_the_check_runs_with_each_pair_on_the_kept_output_and_then_restores_the_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "run" / "job"
    put(job / "job.json", {"job_id": "e2-S-P1"})
    (job.parent / "mfa_out" / "C").mkdir(parents=True)
    (job.parent / "mfa_out" / "C" / "u1.TextGrid").write_text("x", encoding="utf-8")
    app = tmp_path / "app"
    seen = []

    def finish(path: Path, a: Path, models: Path, ran: dict[str, Any]) -> dict[str, Any]:
        seen.append(
            (guard.T1, guard.T2, (mfa.mfa_root(a, "e2-S-P1") / "out" / "C" / "u1.TextGrid").is_file())
        )
        return {"t1": guard.T1}

    monkeypatch.setattr(gate.mfa, "finish", finish)
    out = gate.checked(job, {"status": "ok"}, app, app / "models", [(4.0, 2), (6.5, 8)])
    assert seen == [(4.0, 2, True), (6.5, 8, True)] and out == [{"t1": 4.0}, {"t1": 6.5}]
    assert (guard.T1, guard.T2) == gate.START and not mfa.mfa_root(app, "e2-S-P1").exists()


def test_the_calibration_holds_within_40_ms_of_silencedetect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(printed: str, code: int) -> None:
        def watched(argv: list[str], env: dict[str, str], cwd: Path, log: Path) -> tuple[int, float, int]:
            log.write_text(printed, encoding="utf-8")
            return code, 1.0, 0

        monkeypatch.setattr(gate, "watched", watched)

    fake("MFA 611-3114 ms, silencedetect 571-3154 ms; MFA {...}", 0)  # 40 ms at both edges
    assert gate.calibration(tmp_path, tmp_path)[0]
    fake("MFA 612-3114 ms, silencedetect 571-3154 ms", 0)  # 41 ms
    assert not gate.calibration(tmp_path, tmp_path)[0]
    fake("MFA 571-3154 ms, silencedetect 571-3154 ms", 1)  # the test failed for another reason
    assert not gate.calibration(tmp_path, tmp_path)[0]
    fake("no edges", 0)
    assert not gate.calibration(tmp_path, tmp_path)[0]


def test_the_sessions_come_in_registered_order_and_a_check_run_leaves_out_the_long_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []

    def make(name: str, folder: Path) -> None:
        built.append(name)
        put(folder / "truth.json", {"name": name})

    monkeypatch.setattr(gate.sets.sb, "build", lambda recipe, f: make(recipe.stem, f))
    monkeypatch.setattr(gate.sets, "build_b", lambda prepared, n, f: make(f"{n}-B", f))
    monkeypatch.setattr(gate.sets, "l2_rows", lambda data, split, speakers: {s: [s] for s in speakers})
    monkeypatch.setattr(gate.sets, "l2_interview", lambda data, rows, n, f: make(n, f))
    monkeypatch.setattr(gate.sets.sb, "build_derived", lambda n, f: make(n, f))
    splits = {"edacc": {s: {"l2_interview_speakers": ["x", "y", "z", "w"]} for s in ("dev", "test")}}
    set_b = tmp_path / "set_b"
    for n in ("CS-SOLO", "CS-INT"):
        put(set_b / n / "recipe.json", {})
    dev = gate.sessions("dev", tmp_path, splits, tmp_path / "w1", None)
    assert [s.name for s in dev] == ["CS-INT", "CS-NOISY", "CS-OVL", "CS-SOLO", "CS-SWAP", "CS-WRONGMODE",
                                     "L2-dev-1", "L2-dev-2", "L2-dev-3", "L2-dev-4"]  # fmt: skip
    test = gate.sessions("test", tmp_path, splits, tmp_path / "w2", set_b, limit=1)
    assert [s.name for s in test] == ["CS-INT-B", "L2-test-1", "CS-HES"]
    built.clear()
    assert [s.name for s in gate.sessions("dev", tmp_path, splits, tmp_path / "w1", None)] == [
        s.name for s in dev
    ]
    assert built == []  # built once
