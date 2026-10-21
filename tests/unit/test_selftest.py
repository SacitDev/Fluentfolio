"""The self-test's checks on hand-made step files laid out from the bundled clip's own truth.

Each limit is tried just inside and just outside its edge; the expected values come from the clip's truth
and the stated limits, never from the code under test.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.app import facade
from fluentfolio.cli.main import main
from fluentfolio.core.enums import Device, JobState, Status
from fluentfolio.core.errors import catalogue as codes
from fluentfolio.core.files import read_json, write_json
from fluentfolio.doctor import checks, selftest
from fluentfolio.doctor.checks import ComponentRow
from fluentfolio.jobs import loop, store
from fluentfolio.models import catalogue
from fluentfolio.settings.model import Settings
from tests.support.in_process import Command

Row = dict[str, Any]
ASR = "parakeet-tdt-0.6b-v2"
IN_USE = (
    ASR, "whisper-tiny", "silero-vad", "pyannote-community-1", "sherpa-onnx-diarizer", "mfa", "ctc-aligner",
    "phone-model", "espeak-ng", "pronunciation-dictionary", "praat", "swiftf0", "opensmile",
    "spacy-en-core-web-sm",
)  # fmt: skip
STEPS = (
    "media", "vad", "language", "asr", "speakers", "words", "structure", "candidate_audio", "phones",
    "alignment", "praat", "swiftf0", "opensmile", "nlp", "measures", "evidence", "export",
)  # fmt: skip
KNOWN = selftest.truth()
WANT_HZ = KNOWN["candidate_median_f0_hz"]
SPOKEN = [s for s in KNOWN["segments"] if s["role"] == "candidate"]


def pitch(hz: float | None) -> Row:
    return {"status": "ok", "candidates": {"candidate": {"scopes": {"all": {"median_hz": hz}}}}}


def clean() -> dict[str, Row]:
    """Step files as a perfect run would write them: every value equal to the clip's truth."""
    halves = [((s["start_ms"] + s["end_ms"]) // 2, s) for s in SPOKEN]
    return {
        "stages/vad.json": {"status": "ok", "speech_ms": KNOWN["speech_ms"]},
        "stages/language.json": {"status": "ok", "non_english_ms": 0},
        "stages/asr.json": {"status": "ok", "words": [{"text": w} for w in selftest.reference()]},
        "stages/speakers.json": {
            "status": "ok", "backend": "pyannote-community-1",
            "turns": [{"start_ms": s["start_ms"], "end_ms": s["end_ms"], "speaker": f"S{s['speaker']}"}
                      for s in KNOWN["segments"]],
            "speakers": [{"id": "SE", "kind": "main"}, {"id": "SC", "kind": "main"}],
        },
        "stages/alignment.json": {
            "status": "ok", "mfa": {"status": "ok"},
            "utterances": [{"method": "mfa", "words": [{"start_ms": s["start_ms"], "end_ms": mid},
                                                         {"start_ms": mid, "end_ms": s["end_ms"]}]}
                           for mid, s in halves],
        },
        "analyzers/praat.json": pitch(WANT_HZ),
        "analyzers/swiftf0.json": pitch(WANT_HZ),
        "analyzers/opensmile.json": pitch(WANT_HZ),
        "analyzers/measures.json": {"candidates": {"candidate": {"pointers": {"words_to_check": []}}}},
    }  # fmt: skip


def record(
    pronunciation: bool = True, state: JobState = JobState.COMPLETED, **stages: store.StageRecord
) -> store.JobRecord:
    found = [stages.get(s) or store.StageRecord(id=s, status=Status.OK) for s in STEPS]
    return store.JobRecord(
        job_id="20261015-000000-0001", created_utc="2026-10-15T00:00:00Z", updated_utc="2026-10-15T00:00:00Z",
        app_version="0", mode="interview", layout="mixed", asr=ASR, selftest=True, state=state,
        device={"requested": "auto", "resolved": "cpu", "reason": ""},
        settings={"analyzers": {"pronunciation": pronunciation}}, stages=found,
    )  # fmt: skip


def rows(**statuses: str) -> list[ComponentRow]:
    """The parts in use, all ready unless named; a name's dashes are written as underscores."""
    found = {k.replace("_", "-"): v for k, v in statuses.items()}
    return [
        ComponentRow(id=i, title=i, kind=catalogue.get(i).kind, status=found.get(i, "ready"),  # type: ignore[arg-type]
                     reason=f"row reason of {i}", on=True, size_bytes=0, licence_class="", consent=None)
        for i in IN_USE
    ]  # fmt: skip


def job(tmp_path: Path, change: Callable[[dict[str, Row]], None] | None = None) -> Path:
    files = clean()
    if change:
        change(files)
    for name, content in files.items():
        write_json(tmp_path / "job" / name, content)
    return tmp_path / "job"


def evaluate(tmp_path: Path, change: Callable[[dict[str, Row]], None] | None = None, **kw: Any) -> Row:
    found = selftest.evaluate(job(tmp_path, change), kw.pop("rec", None) or record(), kw.pop("rows", rows()))
    return {k: (p.status, p.reason, p.values) for k, p in found.items()}


def test_a_run_equal_to_the_truth_passes_every_part_it_used(tmp_path: Path) -> None:
    found = selftest.evaluate(job(tmp_path), record(), rows())
    assert {k: p.status for k, p in found.items()} == {
        ASR: "ok", "whisper-tiny": "ok", "silero-vad": "ok", "pyannote-community-1": "ok",
        "sherpa-onnx-diarizer": "not-tested", "mfa": "ok", "ctc-aligner": "ok", "phone-model": "ok",
        "espeak-ng": "ok", "pronunciation-dictionary": "ok", "praat": "ok", "swiftf0": "ok",
        "opensmile": "ok", "spacy-en-core-web-sm": "ok",
    }  # fmt: skip
    assert found["sherpa-onnx-diarizer"].reason == "these settings did not use it"
    assert found[ASR].values == {"wer": 0.0} and found["mfa"].values == {"edges_held": 7, "utterances": 7}
    assert selftest.passed(found, record())


def test_the_bundled_clip_is_the_one_its_truth_describes() -> None:
    assert selftest.clip().stat().st_size <= 1_500_000 and (selftest.ASSETS / "NOTICE").is_file()
    assert len(KNOWN["segments"]) == len(selftest.ASSETS.joinpath("reference.txt").read_text().splitlines())
    assert KNOWN["speech_ms"] == sum(s["end_ms"] - s["start_ms"] for s in KNOWN["segments"])
    assert {s["speaker"] for s in KNOWN["segments"]} == {"E", "C"} and len(SPOKEN) >= 5  # noqa: PLR2004
    assert 50_000 <= KNOWN["duration_ms"] <= 65_000  # about a minute


def set_value(path: str, *keys: str, value: Any) -> Callable[[dict[str, Row]], None]:
    def change(files: dict[str, Row]) -> None:
        place = files[path]
        for k in keys[:-1]:
            place = place[k]
        place[keys[-1]] = value

    return change


TENTH = KNOWN["speech_ms"] // 10  # 10% of the true speech time, rounded down to whole ms


def vad(ms: int) -> Callable[[dict[str, Row]], None]:
    return set_value("stages/vad.json", "speech_ms", value=ms)


def median(tool: str, hz: float | None) -> Callable[[dict[str, Row]], None]:
    return set_value(
        f"analyzers/{tool}.json", "candidates", "candidate", "scopes", "all", "median_hz", value=hz
    )


def to_check(n: int) -> Callable[[dict[str, Row]], None]:
    found = [{"text": f"word {k}"} for k in range(n)]
    return set_value(
        "analyzers/measures.json", "candidates", "candidate", "pointers", "words_to_check", value=found
    )


def voices(n: int) -> Callable[[dict[str, Row]], None]:
    return set_value(
        "stages/speakers.json", "speakers", value=[{"id": f"S{k}", "kind": "main"} for k in range(n)]
    )


SPEECH = KNOWN["speech_ms"]
ST = 2 ** (1 / 12)  # one semitone as a frequency ratio


@pytest.mark.parametrize(
    ("change", "part", "status"),
    [
        (vad(SPEECH + TENTH), "silero-vad", "ok"),
        (vad(SPEECH - TENTH), "silero-vad", "ok"),
        (vad(SPEECH + TENTH + 1), "silero-vad", "failed"),
        (vad(SPEECH - TENTH - 1), "silero-vad", "failed"),
        (median("praat", WANT_HZ * 1.0199), "praat", "ok"),
        (median("praat", WANT_HZ * 0.9801), "praat", "ok"),
        (median("praat", WANT_HZ * 1.0201), "praat", "failed"),
        (median("praat", None), "praat", "failed"),
        (median("swiftf0", WANT_HZ * ST**0.99), "swiftf0", "ok"),
        (median("swiftf0", WANT_HZ * ST**1.01), "swiftf0", "failed"),
        (median("opensmile", WANT_HZ / ST**0.99), "opensmile", "ok"),
        (median("opensmile", WANT_HZ / ST**1.01), "opensmile", "failed"),
        (median("opensmile", WANT_HZ * 2), "opensmile", "failed"),  # an octave jump
        (to_check(1), "phone-model", "ok"),
        (to_check(2), "phone-model", "failed"),
        (set_value("stages/language.json", "non_english_ms", value=1), "whisper-tiny", "failed"),
        (voices(1), "pyannote-community-1", "failed"),
        (voices(3), "pyannote-community-1", "failed"),
    ],
)  # fmt: skip
def test_each_limit_holds_just_inside_and_fails_just_outside(
    tmp_path: Path, change: Callable[[dict[str, Row]], None], part: str, status: str
) -> None:
    found = selftest.evaluate(job(tmp_path, change), record(), rows())
    assert found[part].status == status, found[part]
    others = [p for k, p in found.items() if k != part and p.status == "failed"]
    assert others == []  # one change fails only its own part


def swapped_words(k: int) -> Callable[[dict[str, Row]], None]:
    def change(files: dict[str, Row]) -> None:
        for w in files["stages/asr.json"]["words"][:k]:
            w["text"] = "zzz"

    return change


def test_the_word_error_rate_may_reach_ten_percent_and_no_more(tmp_path: Path) -> None:
    n = len(selftest.reference())
    allowed = n // 10  # substitutions within 10% of the reference words
    assert evaluate(tmp_path / "a", swapped_words(allowed))[ASR][0] == "ok"
    status, reason, values = evaluate(tmp_path / "b", swapped_words(allowed + 1))[ASR]
    assert status == "failed" and values == {"wer": round((allowed + 1) / n, 4)} and "more than 10%" in reason


def given_away(ms: int) -> Callable[[dict[str, Row]], None]:
    """The first candidate utterance's first `ms` given to the examiner's voice."""

    def change(files: dict[str, Row]) -> None:
        turns = files["stages/speakers.json"]["turns"]
        k = next(i for i, t in enumerate(turns) if t["speaker"] == "SC")
        t = turns[k]
        turns[k : k + 1] = [
            {**t, "speaker": "SE", "end_ms": t["start_ms"] + ms},
            {**t, "start_ms": t["start_ms"] + ms},
        ]

    return change


def test_speaker_agreement_may_lose_ten_percent_of_the_speech_and_no_more(tmp_path: Path) -> None:
    assert SPOKEN[0]["end_ms"] - SPOKEN[0]["start_ms"] > TENTH + 1  # the change fits in one utterance
    assert evaluate(tmp_path / "a", given_away(TENTH))["pyannote-community-1"][0] == "ok"
    status, _, values = evaluate(tmp_path / "b", given_away(TENTH + 1))["pyannote-community-1"]
    assert status == "failed" and values == {"speakers": 2, "agreement": round(1 - (TENTH + 1) / SPEECH, 4)}


def edges_moved(k: int, ms: int) -> Callable[[dict[str, Row]], None]:
    def change(files: dict[str, Row]) -> None:
        for u in files["stages/alignment.json"]["utterances"][:k]:
            u["words"][0]["start_ms"] += ms

    return change


def test_mfa_needs_eighty_percent_of_utterances_with_edges_within_50_ms(tmp_path: Path) -> None:
    n = len(SPOKEN)
    may_miss = n - math.ceil(0.8 * n)  # utterances that may miss and still leave 80%
    assert evaluate(tmp_path / "a", edges_moved(n, 50))["mfa"][0] == "ok"  # 50 ms off is still held
    assert evaluate(tmp_path / "b", edges_moved(may_miss, 51))["mfa"][0] == "ok"
    status, reason, values = evaluate(tmp_path / "c", edges_moved(may_miss + 1, 51))["mfa"]
    assert status == "failed" and values == {"edges_held": n - may_miss - 1, "utterances": n}
    assert reason == f"{n - may_miss - 1} of {n} utterances aligned with both edges within 50 ms"


def test_words_to_check_count_only_when_the_pronunciation_check_is_on(tmp_path: Path) -> None:
    assert evaluate(tmp_path, to_check(2), rec=record(pronunciation=False))["phone-model"] == ("ok", "", {})


def test_the_dictionary_is_tested_only_by_the_pronunciation_check(tmp_path: Path) -> None:
    off = evaluate(tmp_path, rec=record(pronunciation=False))
    assert off["pronunciation-dictionary"] == ("not-tested", "these settings did not use it", {})
    # Without its own download the phones step reads the copy an MFA install holds; the part is still missing.
    missing = evaluate(tmp_path, rows=rows(pronunciation_dictionary="needs-download"))
    part = "pronunciation-dictionary"
    assert missing[part][:2] == ("unavailable", f"row reason of {part}")


def test_a_constant_pitch_tool_fails_against_praat(tmp_path: Path) -> None:
    """A pitch tool stuck at one value, here 220 Hz, gives that value as its median."""
    found = selftest.evaluate(job(tmp_path, median("swiftf0", 220.0)), record(), rows())
    gap = 12 * math.log2(220.0 / WANT_HZ)  # hand formula, more than 1 semitone for this clip
    assert gap > 1 and found["swiftf0"].status == "failed"
    assert found["swiftf0"].values == {"median_hz": 220.0, "semitones": round(gap, 2)}
    assert not selftest.passed(found, record())


@pytest.mark.parametrize("tool", ["praat", "swiftf0"])
def test_a_pitch_tool_that_finds_no_pitch_fails_in_words_with_no_infinite_value(
    tmp_path: Path, tool: str
) -> None:
    status, reason, values = evaluate(tmp_path, median(tool, None))[tool]
    assert (status, reason) == ("failed", "found no pitch in the test clip")
    assert all(math.isfinite(v) for v in values.values()) and "semitones" not in values


def test_without_praat_the_other_tools_are_held_to_the_bundled_value(tmp_path: Path) -> None:
    off = store.StageRecord(id="praat", status=Status.DISABLED)
    found = evaluate(tmp_path, median("swiftf0", WANT_HZ * ST**0.5), rec=record(praat=off))
    assert found["praat"][0] == "disabled" and found["swiftf0"][0] == "ok"
    assert found["swiftf0"][2]["semitones"] == 0.5  # noqa: PLR2004


DAMAGED = "a file of this part is missing or damaged: re-download this model"


@pytest.mark.parametrize(
    ("step", "status", "code", "parts", "expected", "reason"),
    [
        ("nlp", Status.FAILED, "E_HASH_MISMATCH", ["spacy-en-core-web-sm"], "failed", DAMAGED),
        ("nlp", Status.FAILED, "E_MODEL_MISSING", ["spacy-en-core-web-sm"], "unavailable",
         codes()["E_MODEL_MISSING"].message),
        ("nlp", Status.FAILED, "E_STAGE_CRASHED", ["spacy-en-core-web-sm"], "failed",
         "it stopped with E_STAGE_CRASHED"),
        ("opensmile", Status.DISABLED, "E_OPENSMILE_CONSENT_MISSING", ["opensmile"], "disabled",
         codes()["E_OPENSMILE_CONSENT_MISSING"].message),
        ("swiftf0", Status.DISABLED, None, ["swiftf0"], "disabled", "switched off in Settings"),
        ("phones", Status.UNAVAILABLE, "E_COMPONENT_UNAVAILABLE",
         ["phone-model", "espeak-ng", "ctc-aligner", "pronunciation-dictionary"], "unavailable",
         codes()["E_COMPONENT_UNAVAILABLE"].message),
        ("language", Status.SKIPPED, None, ["whisper-tiny"], "not-tested",
         "the step it needs did not finish, so this one did not run"),
        ("speakers", None, None, ["pyannote-community-1", "sherpa-onnx-diarizer"], "not-tested",
         "an earlier step stopped, so this one did not run"),
    ],
)  # fmt: skip
def test_a_step_that_did_not_finish_decides_its_parts(
    tmp_path: Path, step: str, status: Status | None, code: str | None, parts: list[str], expected: str,
    reason: str,
) -> None:  # fmt: skip
    rec = record(**{step: store.StageRecord(id=step, status=status, error_code=code)})
    found = evaluate(tmp_path, rec=rec)
    assert {p: found[p][:2] for p in parts} == dict.fromkeys(parts, (expected, reason))
    assert all(v[0] in ("ok", "not-tested") for k, v in found.items() if k not in parts)


@pytest.mark.parametrize(
    ("mfa", "expected"),
    [
        ({"status": "unavailable", "code": "E_COMPONENT_UNAVAILABLE"}, ("unavailable",
                                                                      codes()["E_COMPONENT_UNAVAILABLE"].message)),
        ({"status": "failed", "code": "E_STAGE_CRASHED"}, ("failed", "it stopped with E_STAGE_CRASHED")),
        ({"status": "disabled"}, ("disabled", "switched off in Settings")),
        ({"status": "skipped"}, ("not-tested", "the step it needs did not finish, so this one did not run")),
    ],
)  # fmt: skip
def test_mfa_that_did_not_align_is_reported_by_its_own_status(
    tmp_path: Path, mfa: Row, expected: tuple[str, str]
) -> None:
    found = evaluate(tmp_path, set_value("stages/alignment.json", "mfa", value=mfa))
    assert found["mfa"][:2] == expected and found["ctc-aligner"][0] == "ok"


def test_when_the_first_speaker_backend_stops_the_other_one_is_checked(tmp_path: Path) -> None:
    switched = store.StageRecord(
        id="speakers", status=Status.OK, fallback={"to": "speakers-sherpa", "after": "E_STAGE_CRASHED"}
    )
    found = evaluate(tmp_path, set_value("stages/speakers.json", "backend", value="sherpa-onnx-diarizer"),
                     rec=record(speakers=switched))  # fmt: skip
    assert found["sherpa-onnx-diarizer"][0] == "ok"
    assert found["pyannote-community-1"][:2] == ("failed", "it stopped with E_STAGE_CRASHED")


def test_a_gpu_retry_on_the_cpu_fails_no_backend(tmp_path: Path) -> None:
    retried = store.StageRecord(id="speakers", status=Status.OK, fallback={"to": "cpu", "after": "E_GPU_OOM"})
    found = evaluate(tmp_path, rec=record(speakers=retried))
    assert found["pyannote-community-1"][0] == "ok" and found["sherpa-onnx-diarizer"][0] == "not-tested"


def test_a_part_that_is_not_ready_says_why_unless_a_step_saw_it_fail(tmp_path: Path) -> None:
    damaged = store.StageRecord(id="nlp", status=Status.FAILED, error_code="E_HASH_MISMATCH")
    found = evaluate(
        tmp_path, rec=record(nlp=damaged),
        rows=rows(opensmile="needs-consent", mfa="needs-download", spacy_en_core_web_sm="needs-download",
                  praat="self-test-failed"),
    )  # fmt: skip
    assert found["opensmile"][:2] == ("disabled", "row reason of opensmile")
    assert found["mfa"][:2] == ("unavailable", "row reason of mfa")
    assert found["spacy-en-core-web-sm"][:2] == ("failed", DAMAGED)  # the step's failure stands
    assert found["praat"][0] == "ok"  # a part that failed before is tested again


def test_only_the_parts_in_use_are_reported(tmp_path: Path) -> None:
    fewer = [r for r in rows() if r.id not in ("opensmile", "sherpa-onnx-diarizer")]
    assert set(evaluate(tmp_path, rows=fewer)) == set(IN_USE) - {"opensmile", "sherpa-onnx-diarizer"}


@pytest.mark.parametrize(
    ("state", "statuses", "expected"),
    [
        (JobState.COMPLETED, ["ok", "not-tested"], True),
        (JobState.COMPLETED_WITH_GAPS, ["ok", "disabled", "unavailable"], True),
        (JobState.COMPLETED, ["ok", "failed"], False),
        (JobState.FAILED, ["ok", "not-tested"], False),
        (JobState.INTERRUPTED, ["ok"], False),
    ],
)
def test_it_passes_when_the_analysis_finished_and_no_part_failed(
    state: JobState, statuses: list[str], expected: bool
) -> None:
    parts = {f"p{k}": selftest.Part(s) for k, s in enumerate(statuses)}  # type: ignore[arg-type]
    assert selftest.passed(parts, record(state=state)) is expected


def test_the_outcome_is_kept_and_the_components_list_names_the_failed_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, data = tmp_path / "app", tmp_path / "data"
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(app))
    write_json(app / "state.json", {"backend": "cpu", "models_dir": str(tmp_path / "models")})
    parts = {"praat": selftest.Part("failed", "median pitch 1.0 Hz, against 2 Hz", {"median_hz": 1.0})}
    outcome = selftest.save(app, parts, record(), 61_260)
    kept = read_json(app / "state.json")
    assert kept["backend"] == "cpu" and kept["selftest"] == outcome  # the install state stays
    assert outcome["passed"] is False and (outcome["elapsed_ms"], outcome["budget_ms"]) == (61_260, 360_000)
    assert outcome["components"]["praat"] == {
        "status": "failed",
        "reason": "median pitch 1.0 Hz, against 2 Hz",
        "values": {"median_hz": 1.0},
    }
    row = next(r for r in checks.components(Settings(), loop.Roots(app, data, tmp_path / "models"))
               if r.id == "praat")  # fmt: skip
    assert (row.status, row.reason) == ("self-test-failed", "median pitch 1.0 Hz, against 2 Hz")


def test_a_self_test_job_left_by_a_crash_is_hidden_and_then_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, app = tmp_path / "data", tmp_path / "app"
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(data))
    left = record(state=JobState.RUNNING)
    mine = record().model_copy(update={"job_id": "20261015-000000-0002", "selftest": False})
    store.create(data, left)
    store.create(data, mine)
    assert [r.job_id for r in facade.jobs(10)] == [mine.job_id]
    assert store.selftest_jobs(data) == [left.job_id]
    loop.housekeeping(loop.Roots(app, data, tmp_path / "models"), retention_days=0)
    assert store.list_ids(data) == [mine.job_id]


def outcome(**parts: tuple[str, str]) -> Row:
    found = {k.replace("_", "-"): {"status": s, "reason": r, "values": {}} for k, (s, r) in parts.items()}
    passed = all(p["status"] != "failed" for p in found.values())
    return {
        "utc": "",
        "passed": passed,
        "device": "cpu",
        "elapsed_ms": 61_300,
        "budget_ms": 360_000,
        "components": found,
    }


@pytest.mark.parametrize(
    ("given", "exit_code", "last_line"),
    [
        (outcome(praat=("ok", "")), 0, "Ready. The self-test took 61 s on the CPU."),
        ({**outcome(praat=("ok", "")), "elapsed_ms": 360_000}, 0,
         "Ready. The self-test took 360 s on the CPU."),  # at the budget
        ({**outcome(praat=("ok", "")), "elapsed_ms": 360_001}, 0,
         "It took longer than the 6 min it should take on the CPU."),
        (outcome(praat=("ok", ""), opensmile=("disabled", "it waits for consent"), mfa=("unavailable", "X")),
         0, "Ready, with limits: opensmile is off because it waits for consent; mfa is off because X."),
        (outcome(praat=("failed", "median pitch 1 Hz"), swiftf0=("failed", "y"), mfa=("disabled", "z")), 3,
         "The self-test failed: praat, swiftf0. Run `fluentfolio doctor`."),
    ],
)  # fmt: skip
def test_the_command_says_ready_names_the_limits_or_names_the_failed_parts(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, given: Row, exit_code: int,
    last_line: str,
) -> None:  # fmt: skip
    Command(capsys, monkeypatch)  # the test process keeps its signal handlers and Hugging Face settings
    asked: list[Device | None] = []
    monkeypatch.setattr(facade, "self_test", lambda device, progress: asked.append(device) or given)
    assert main(["self-test", "--device", "cpu"]) == exit_code
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == last_line and asked == [Device.CPU]
    assert printed[0].split()[:2] == [next(iter(given["components"])), given["components"]["praat"]["status"]]


def test_the_command_reports_a_failure_as_its_error_with_the_failed_parts(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Command(capsys, monkeypatch)
    monkeypatch.setattr(facade, "self_test", lambda device, progress: outcome(praat=("failed", "x")))
    exit_code, found, _ = run("self-test")
    assert exit_code == 3 and found["ok"] is False and found["data"]["passed"] is False  # noqa: PLR2004
    assert found["error"]["code"] == "E_SELFTEST_FAILED" and found["error"]["message"].endswith("(praat)")


@pytest.mark.parametrize(
    ("reference", "hypothesis", "rate"),
    [
        ("a b c", "a b c", 0.0),
        ("a b c", "a x c d", 2 / 3),  # one substitution and one insertion over 3 words
        ("a b c d", "a c", 2 / 4),  # two deletions
        ("a b", "", 1.0),
        ("", "a", 1.0),  # an insertion over an empty reference still counts against a length of 1
    ],
)
def test_word_error_rate_counts_the_fewest_edits(reference: str, hypothesis: str, rate: float) -> None:
    assert selftest.wer(reference.split(), hypothesis.split()) == pytest.approx(rate)


def test_words_ignore_case_and_punctuation() -> None:
    assert selftest.words("It's DONE, isn't it?") == ["it's", "done", "isn't", "it"]


def seg(speaker: str, start: int, end: int, role: str = "candidate") -> Row:
    return {"speaker": speaker, "start_ms": start, "end_ms": end, "role": role}


TWO = [seg("E", 0, 1000, "examiner"), seg("C", 1000, 3000)]


@pytest.mark.parametrize(
    ("turns", "share"),
    [
        ([seg("S1", 0, 1000), seg("S2", 1000, 3000)], 1.0),
        ([seg("S2", 0, 1000), seg("S1", 1000, 3000)], 1.0),  # the labels' names do not matter
        ([seg("S1", 0, 1000), seg("S2", 1000, 2500), seg("S1", 2500, 3000)], 2500 / 3000),
        ([seg("S1", 0, 3000)], 0.0),  # one voice for two people
        ([seg("S1", 0, 1000), seg("S2", 1000, 2900), seg("S3", 2900, 3000)], 2900 / 3000),  # a short third
    ],
)
def test_agreement_takes_the_best_match_of_found_to_true_speakers(turns: list[Row], share: float) -> None:
    assert selftest.agreement(turns, TWO) == pytest.approx(share)


def word(start: int, end: int) -> Row:
    return {"start_ms": start, "end_ms": end}


@pytest.mark.parametrize(
    ("aligned", "held"),
    [
        ([word(1050, 1500), word(1600, 2950)], 1),  # both edges 50 ms off: held
        ([word(950, 1500), word(1600, 3050)], 1),  # 50 ms early and 50 ms late: held
        ([word(1051, 1500), word(1600, 2950)], 0),  # the start 51 ms off
        ([word(1050, 1500), word(1600, 2949)], 0),  # the end 51 ms off
        ([word(4000, 4500)], 0),  # nothing inside the utterance
    ],
)
def test_mfa_edges_hold_within_50_ms(aligned: list[Row], held: int) -> None:
    assert selftest.edges_held(aligned, TWO) == (held, 1)
