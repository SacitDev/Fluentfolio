"""The three modes on the composed sessions, from audio to the export, with the real models.

Each session runs once per module; the scores compare the structure step with the builder's truth.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from statistics import mean, median
from types import ModuleType
from typing import Any

import pytest

from fluentfolio.analyzers.measures import fluency
from fluentfolio.app import facade
from fluentfolio.core.enums import JobState
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.paths import app_root, data_root, models_dir
from fluentfolio.core.time import inside, total, union
from fluentfolio.evidence import build
from fluentfolio.jobs import loop
from fluentfolio.models import catalogue
from fluentfolio.speakers.ownership import OVERLAP, overlaps
from fluentfolio.structure.build import UNKNOWN_CANDIDATE

pytestmark = pytest.mark.slow
SESSIONS = Path(__file__).resolve().parents[1] / "fixtures" / "sessions"
Row = dict[str, Any]
Run = tuple[Row, facade.Analysis]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory, session_builder: ModuleType) -> Callable[..., Run]:
    """Build a session and analyse it in the declared mode, once per module."""
    done: dict[tuple[str, str], Run] = {}

    def run(name: str, mode: str, **options: Any) -> Run:
        if (name, mode) not in done:
            out = tmp_path_factory.mktemp(name)
            truth = session_builder.build(SESSIONS / f"{name}.json", out / "session")
            if mode != "solo":  # the composed voices stand for people who agreed
                options.setdefault("partner_consent", "cli-flag")
            done[(name, mode)] = (truth, facade.analyze(out / "session" / "session.wav", mode, **options))
        return done[(name, mode)]

    return run


def stage(job: Path, name: str) -> Row:
    result: Row = read_json(job / "stages" / f"{name}.json")
    return result


def codes(job: Path) -> list[str]:
    return [w["code"] for w in read_json(job / "evidence.json")["manifest"]["warnings"]]


def resumed(job: Path, mode: str, backend: str, change: Callable[[Path], None]) -> Path:
    """A copy of a finished job, changed by `change` and run again from the first result it lacks."""
    copy = data_root().parent / "resumed" / "jobs" / job.name
    shutil.copytree(job, copy)
    change(copy)
    write_json(copy / "job.json", {**read_json(copy / "job.json"), "state": JobState.INTERRUPTED.value})
    roots = loop.Roots(app=app_root(), data=copy.parents[1], models=models_dir())
    loop.run_job(roots, job.name, facade.plan(mode, "mixed", backend), audio_min=90, no_progress_s=600)
    return copy


def first_question_of(truth: Row, speaker: str) -> Row:
    return next(s for s in truth["segments"] if s["kind"] == "question" and s["speaker"] == speaker)


def long_turn_ms(truth: Row, speaker: str) -> int:
    """The length of the speaker's Part 2 answer in the truth: their first Part 2 answer turn."""
    mine = [t for t in truth["turns"] if t["kind"] == "answer" and t["speaker"] == speaker]
    t = next(t for t in mine if t["part"] == 2)  # noqa: PLR2004
    return int(t["end_ms"] - t["start_ms"])


def truth_pauses(turn: Row) -> list[int]:
    """The inserted silences in one truth turn that count as pauses: 250 ms or more, by definition."""
    return [p["ms"] for p in turn["pauses"] if p["ms"] >= 250]  # noqa: PLR2004


def check_fluency(name: str, truth: Row, scope: Row, tolerance: Any) -> None:
    """Compare one candidate's whole-test pauses, latency and speech rate with the truth."""
    answers = [t for t in truth["turns"] if t["kind"] == "answer"]
    found = scope["pauses"]
    per_answer = [
        (len([p for p in found if t["start_ms"] <= (p[0] + p[1]) / 2 < t["end_ms"]]), len(truth_pauses(t)))
        for t in answers
    ]
    expected_ms = [ms for t in answers for ms in truth_pauses(t)]
    mean_found, mean_expected = mean(b - a for a, b in found), mean(expected_ms)
    # The truth's latency: from the end of a question turn to the start of the answer turn after it.
    asked = [(a, b) for a, b in pairwise(truth["turns"]) if (a["kind"], b["kind"]) == ("question", "answer")]
    waits = [b["start_ms"] - a["end_ms"] for a, b in asked]
    latency = scope["values"]["latency_median"]["value"] * 1000
    rate_expected = sum(t["syllables"] for t in answers) / sum(t["end_ms"] - t["start_ms"] for t in answers)
    rate = scope["values"]["speech_rate"]["value"] / 1000
    print(f"{name}: pauses per answer {per_answer} (found, truth), mean pause {mean_found:.0f} ms (truth "
          f"{mean_expected:.0f}), latency {latency:.0f} ms (truth median {median(waits):.0f} over "
          f"{len(waits)}), speech rate {rate * 1000} syll/s (truth {rate_expected * 1000:.3f})")  # fmt: skip
    for (n_found, n_expected), t in zip(per_answer, answers, strict=True):
        assert abs(n_found - n_expected) <= tolerance("pipeline", "pause_count_per_answer"), t["id"]
    assert abs(mean_found - mean_expected) <= tolerance("pipeline", "pause_mean_ms")
    assert abs(latency - median(waits)) <= tolerance("pipeline", "latency_ms")
    assert abs(rate / rate_expected - 1) <= tolerance("pipeline", "speech_rate_relative")


def test_cs_int_roles_parts_and_the_long_turn(
    runs: Callable[..., Run], scoring: ModuleType, tolerance: Any
) -> None:
    truth, done = runs("CS-INT", "interview")
    assert scoring.gaps(done.job) == []
    words, structure = stage(done.job, "words")["words"], stage(done.job, "structure")
    roles, part = (
        scoring.role_accuracy(truth, words, structure),
        scoring.part_accuracy(truth, words, structure),
    )
    der = scoring.der(truth, stage(done.job, "speakers")["turns"])
    (c,) = structure["candidates"]
    expected = long_turn_ms(truth, next(s["speaker"] for s in truth["segments"] if s["kind"] == "answer"))
    print(f"CS-INT: role accuracy {roles:.4f}, part accuracy {part:.4f}, pyannote DER {der:.4f}, "
          f"part2 {c['part2']} (long turn in the truth {expected} ms), "
          f"confidence {c['roles']['confidence']}")  # fmt: skip
    assert roles >= tolerance("pipeline", "role_accuracy_interview_min")
    assert part >= tolerance("pipeline", "part_accuracy_min")
    assert der <= tolerance("pipeline", "der_pyannote_max")
    assert abs(c["part2"]["long_turn_ms"] - expected) <= tolerance("pipeline", "long_turn_s") * 1000
    (anchor,) = [a for a in structure["answers"] if a["after_preparation"]]
    assert anchor["part"] == 2 and c["part2"]["prep_excluded_ms"] >= 15_000  # noqa: PLR2004
    assert "W_ROLES_UNAVAILABLE" not in codes(done.job)
    measures = read_json(done.job / "analyzers" / "measures.json")["candidates"][c["id"]]
    # The Part 2 talk follows the preparation minute, so Part 2 has no response time, and says why.
    part2 = measures["scopes"]["part2"]["values"]
    assert part2["latency_median"]["missing"]["code"] == "after_preparation"
    assert part2["latency_over_10s"]["missing"]["code"] == "after_preparation"
    check_fluency("CS-INT", truth, measures["scopes"]["all"], tolerance)
    # Minutes of answers hold far more vowel sounds than rhythm needs, so MFA's phones give its values where
    # MFA is on by default. Where it is experimental it waits to be switched on, and rhythm says why it is
    # missing (test_mfa_chosen.py switches it on).
    (candidate,) = read_json(done.job / "evidence.json")["candidates"]
    vowel_ms = [m for m in candidate["details"]["measures"] if m["id"] == "DM06" and m["scope"] == "all"]
    assert len(vowel_ms) == 1
    if catalogue.get("mfa").on_by_default(catalogue.SYSTEM):
        assert vowel_ms[0]["value"] is not None and vowel_ms[0]["sources"][0]["tool"] == "mfa"
        assert vowel_ms[0]["n"] >= tolerance("rhythm", "min_vowels")  # n counts the vowel sounds
    else:
        assert vowel_ms[0]["value"] is None and vowel_ms[0]["missing"]["code"] == "no_mfa"
        tools = read_json(done.job / "evidence.json")["manifest"]["tools"]
        (alignment,) = [t for t in tools if t["id"] == "alignment"]
        assert alignment["reason"] == f"MFA is off; it is {build.MFA_EXPERIMENTAL}"


def test_cs_swap_finds_both_people_and_their_parts(
    runs: Callable[..., Run], scoring: ModuleType, tolerance: Any
) -> None:
    truth, done = runs("CS-SWAP", "role-swap", swap_style="once")
    assert scoring.gaps(done.job) == []
    words, structure = stage(done.job, "words")["words"], stage(done.job, "structure")
    roles, part = (
        scoring.role_accuracy(truth, words, structure),
        scoring.part_accuracy(truth, words, structure),
    )
    der = scoring.der(truth, stage(done.job, "speakers")["turns"])
    one, two = structure["candidates"]
    who = {v: k for k, v in scoring.persons(truth).items()}  # person1 -> the builder's speaker
    swap = first_question_of(truth, who["person1"])  # the first to answer asks second
    print(f"CS-SWAP: role accuracy {roles:.4f}, part accuracy {part:.4f}, pyannote DER {der:.4f}, "
          f"swap times {one['roles']['swap_times_ms']} (the second examiner's first question "
          f"{swap['start_ms']}-{swap['end_ms']}), confidence {one['roles']['confidence']}, "
          f"part2 {one['part2']} and {two['part2']}")  # fmt: skip
    assert roles >= tolerance("pipeline", "role_accuracy_role_swap_min")
    assert part >= tolerance("pipeline", "part_accuracy_min")
    assert der <= tolerance("pipeline", "der_pyannote_max")
    (found,) = one["roles"]["swap_times_ms"]
    assert swap["start_ms"] - 1000 <= found <= swap["end_ms"]
    assert (one["heard_questions_first"], two["heard_questions_first"]) == (False, True)
    names = sorted(p.name for p in (done.job / "export" / "for_ai").iterdir())
    assert names == ["for_ai_person1.txt", "for_ai_person2.txt"]  # one text per person
    for c in (one, two):
        expected = long_turn_ms(truth, who[c["id"]])
        assert abs(c["part2"]["long_turn_ms"] - expected) <= tolerance("pipeline", "long_turn_s") * 1000


def spoken_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def data_words(path: Path) -> set[str]:
    """The words of the lines between a text's boundary lines that quote speech: utterances, pointer quotes,
    difference rows and questions. The app's own labels and notes are left out, as they are no one's words."""
    lines = path.read_text(encoding="utf-8").split("\n")
    start = next(i for i, line in enumerate(lines) if line.endswith(" START OF DATA"))
    end = next(i for i, line in enumerate(lines) if line.endswith(" END OF DATA"))
    quoted = [line for line in lines[start + 1 : end] if re.match(r"[UQ]\d+ \|", line)]
    return spoken_words("\n".join(quoted))


def test_each_persons_text_holds_their_own_canary_words_and_never_the_others(
    runs: Callable[..., Run], scoring: ModuleType
) -> None:
    truth, done = runs("CS-SWAP", "role-swap", swap_style="once")
    person = scoring.persons(truth)  # the builder's speaker -> person1 or person2
    answers = {s: set[str]() for s in person}
    for t in truth["turns"]:
        if t["kind"] == "answer":
            answers[t["speaker"]] |= spoken_words(t["text"])
    # A canary is a word only this person's answers hold: no question, prompt or other answer says it.
    canaries = {}
    for s in person:
        elsewhere = set[str]().union(
            *(spoken_words(t["text"]) for t in truth["turns"] if (t["kind"], t["speaker"]) != ("answer", s))
        )
        canaries[person[s]] = {w for w in answers[s] - elsewhere if len(w) >= 6}  # noqa: PLR2004
    folder = done.job / "export" / "for_ai"
    texts = {p: data_words(folder / f"for_ai_{p}.txt") for p in ("person1", "person2")}
    one, two = canaries["person1"], canaries["person2"]
    print(f"CS-SWAP canaries: {len(one)} and {len(two)} words; heard in their own text "
          f"{len(one & texts['person1'])} and {len(two & texts['person2'])}")  # fmt: skip
    for own, other in (("person1", "person2"), ("person2", "person1")):
        assert canaries[own] & texts[own], own  # the speech model heard some of them
        assert not canaries[own] & texts[other], sorted(canaries[own] & texts[other])


def test_cs_solo_splits_questions_read_aloud_from_answers(
    runs: Callable[..., Run], scoring: ModuleType, tolerance: Any
) -> None:
    truth, done = runs("CS-SOLO", "solo")
    assert scoring.gaps(done.job) == []
    words, structure = stage(done.job, "words")["words"], stage(done.job, "structure")
    roles, part = (
        scoring.role_accuracy(truth, words, structure),
        scoring.part_accuracy(truth, words, structure),
    )
    owner = scoring.answers_of(structure)
    # Each word on the side of the truth turn that holds it, or of the nearest one when the speech model puts
    # it in a gap: no answer word read aloud (no full stop before a question), no question word in an answer.
    sides = [scoring.nearest_segment(w, truth["segments"])["kind"] == "answer" for w in words]
    lost = [i for i, answer in enumerate(sides) if answer and i not in owner]
    taken = [i for i, answer in enumerate(sides) if not answer and i in owner]
    print(f"CS-SOLO: read-aloud split accuracy {roles:.4f}, part accuracy {part:.4f}, "
          f"{len(structure['questions'])} questions, {len(structure['answers'])} answers, "
          f"{len(lost)} answer words read aloud, {len(taken)} question words in answers")  # fmt: skip
    assert lost == [] and taken == []
    assert roles >= tolerance("pipeline", "role_accuracy_solo_min")
    assert part >= tolerance("pipeline", "part_accuracy_min")
    assert {q["kind"] for q in structure["questions"]} == {"read_aloud"}
    # The questions read aloud are not the candidate's speech.
    asked = [(s["start_ms"], s["end_ms"]) for s in truth["segments"] if s["kind"] == "read_aloud"]
    assert total(inside(structure["speech"]["candidate"], asked)) <= 0.10 * total(asked)  # noqa: PLR2004


def test_a_second_voice_in_solo_warns(runs: Callable[..., Run]) -> None:
    _, done = runs("CS-WRONGMODE", "solo")
    assert "W_SECOND_VOICE_SOLO" in codes(done.job)


def test_cs_wrongmode_run_again_as_an_interview_finds_the_roles_without_transcribing_again(
    runs: Callable[..., Run], scoring: ModuleType, tolerance: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth, solo = runs("CS-WRONGMODE", "solo")
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(solo.job.parents[1]))  # the data folder the module's runs use
    asr = (solo.job / "stages" / "asr.json").read_bytes()
    offer = facade.rerun_offer(solo.record.job_id, "interview")
    assert offer.first == "speakers" and offer.estimate_ms
    done = facade.rerun(solo.record.job_id, "interview", partner_consent="cli-flag")
    assert scoring.gaps(done.job) == [] and done.record.rerun_of == solo.record.job_id
    assert (done.job / "stages" / "asr.json").read_bytes() == asr  # not transcribed again
    assert (solo.job / "stages" / "asr.json").read_bytes() == asr  # and the solo job is as it was
    kept = {s.id: s for s in solo.record.stages}
    after = {s.id: s for s in done.record.stages}
    assert all(after[step] == kept[step] for step in ("media", "vad", "language", "asr"))
    assert after["speakers"].fallback is None  # pyannote itself separated the voices
    words, structure = stage(done.job, "words")["words"], stage(done.job, "structure")
    roles = scoring.role_accuracy(truth, words, structure)
    der = scoring.der(truth, stage(done.job, "speakers")["turns"])
    took = sum(r.wall_ms or 0 for r in done.record.stages if r.id not in ("media", "vad", "language", "asr"))
    print(f"CS-WRONGMODE run again as an interview: role accuracy {roles:.4f}, pyannote DER {der:.4f}, "
          f"estimate {offer.estimate_ms} ms, steps took {took} ms")  # fmt: skip
    assert roles >= tolerance("pipeline", "role_accuracy_interview_min")
    assert "W_SECOND_VOICE_SOLO" not in codes(done.job)


def test_one_voice_in_an_interview_warns_and_names_no_candidate(runs: Callable[..., Run]) -> None:
    _, done = runs("CS-SOLO", "interview")
    assert done.record.state is JobState.COMPLETED_WITH_GAPS, done.record.stages
    assert {"W_ONE_VOICE", "W_ROLES_UNAVAILABLE"} <= set(codes(done.job))
    assert read_json(done.job / "evidence.json")["candidates"] == []


def test_cs_ovl_flags_the_overlap_and_keeps_one_pause_around_the_backchannel(
    runs: Callable[..., Run], scoring: ModuleType, tolerance: Any
) -> None:
    truth, done = runs("CS-OVL", "interview")
    words, structure = stage(done.job, "words")["words"], stage(done.job, "structure")
    turns = stage(done.job, "speakers")["turns"]
    (overlap,) = [
        (o["start_ms"], o["end_ms"]) for o in truth["overlaps"] if o["end_ms"] - o["start_ms"] >= 1000
    ]  # noqa: PLR2004
    shared = overlaps([(t["start_ms"], t["end_ms"], t["speaker"]) for t in turns])
    found = total(inside(union(shared), [overlap]))
    flagged = [w for w in words if OVERLAP in w["flags"] and overlap[0] <= w["start_ms"] < overlap[1]]
    speech = structure["speech"]["candidate"]
    kept = total(inside(speech, [overlap]))
    der = scoring.der(truth, turns)
    print(f"CS-OVL: overlap found {found} ms of {overlap[1] - overlap[0]}, {len(flagged)} words flagged, "
          f"candidate speech kept in it {kept} ms, pyannote DER {der:.4f}")  # fmt: skip
    least = tolerance("pipeline", "overlap_found_min_s") * 1000
    assert found >= least and flagged
    assert kept <= (overlap[1] - overlap[0]) - least
    assert der <= tolerance("pipeline", "der_pyannote_max")
    # The examiner's "mm-hmm" inside a candidate pause leaves one pause there, not two.
    (bc,) = [s for s in truth["segments"] if s["kind"] == "backchannel"]
    answers = [s for s in truth["segments"] if s["kind"] == "answer"]
    before = max((s for s in answers if s["end_ms"] <= bc["start_ms"]), key=lambda s: s["end_ms"])
    after = min((s for s in answers if s["start_ms"] >= bc["end_ms"]), key=lambda s: s["start_ms"])
    spans = [(a["start_ms"], a["end_ms"]) for a in structure["answers"]]
    found_pauses = fluency.pauses(spans, [(a, b) for a, b in speech])
    gap = [p for p in found_pauses if p[0] < after["start_ms"] and p[1] > before["end_ms"]]
    assert len(gap) == 1 and gap[0][0] <= bc["start_ms"] and gap[0][1] >= bc["end_ms"], gap


def test_cs_int_forced_to_one_voice_cluster_gives_no_roles(runs: Callable[..., Run]) -> None:
    _, done = runs("CS-INT", "interview")

    def one_cluster(job: Path) -> None:
        found = stage(job, "speakers")
        found["turns"] = [{**t, "speaker": "S1"} for t in found["turns"]]
        speech_ms = sum(s["speech_ms"] for s in found["speakers"])
        found["speakers"] = [{"id": "S1", "speech_ms": speech_ms, "share": 1.0, "kind": "main"}]
        found["warnings"] = ["W_ONE_VOICE"]
        write_json(job / "stages" / "speakers.json", found)
        (job / "stages" / "words.json").unlink()

    job = resumed(done.job, "interview", "auto", one_cluster)
    assert read_json(job / "job.json")["state"] == JobState.COMPLETED_WITH_GAPS
    assert read_json(job / "evidence.json")["candidates"] == [] and "W_ROLES_UNAVAILABLE" in codes(job)
    assert sorted(p.name for p in (job / "export" / "for_ai").iterdir()) == ["for_ai.txt"]
    lines = (job / "export" / "for_ai" / "for_ai.txt").read_text(encoding="utf-8").split("\n")
    assert UNKNOWN_CANDIDATE in lines and any(x.startswith("S1 | ") for x in lines)


@pytest.mark.parametrize("name", ["CS-INT", "CS-SWAP", "CS-OVL"])
def test_sherpa_on_the_composed_sessions(runs: Callable[..., Run], scoring: ModuleType, name: str) -> None:
    mode = "role-swap" if name == "CS-SWAP" else "interview"
    truth, done = runs(name, mode, **({"swap_style": "once"} if name == "CS-SWAP" else {}))

    def no_speakers(job: Path) -> None:
        (job / "stages" / "speakers.json").unlink()

    job = resumed(done.job, mode, "sherpa", no_speakers)
    found = stage(job, "speakers")
    words, structure = stage(job, "words")["words"], stage(job, "structure")
    assert found["backend"] == "sherpa-onnx-diarizer"
    der = scoring.der(truth, found["turns"])
    roles = scoring.role_accuracy(truth, words, structure) if structure["candidates"] else 0.0
    caps = {c["roles"]["confidence"] for c in structure["candidates"]}
    print(f"{name} with sherpa: DER {der:.4f}, role accuracy {roles:.4f}, confidence {sorted(caps)}")
    assert structure["candidates"]  # each session has a candidate, so the confidence check is never empty
    assert caps <= {"low", "medium"}  # roles from the no-account backend are at most medium
