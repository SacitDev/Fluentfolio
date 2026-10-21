"""The command line's own rules: when it may ask, what progress it shows, and which exit code a job gives."""

from __future__ import annotations

import inspect
import io
from pathlib import Path
from types import ModuleType

import click
import pytest

from fluentfolio.app import facade
from fluentfolio.cli import commands, setup_commands
from fluentfolio.cli.main import cli, command_of, main
from fluentfolio.core.enums import JobState
from fluentfolio.core.errors import AppError
from fluentfolio.core.errors import catalogue as codes
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.paths import data_root
from fluentfolio.jobs.store import JobRecord
from fluentfolio.models import catalogue
from fluentfolio.setup import components as installer
from tests.support.in_process import Command

SEGMENTS = [(1000, 4000)]
SPOKEN = [
    ("I", 1000, 1200, 0), ("like", 1250, 1500, 0), ("my", 1550, 1700, 0), ("home", 1750, 2100, 0),
    ("town", 2150, 2500, 0), ("because", 2550, 3000, 0), ("it", 3050, 3200, 0), ("is", 3250, 3400, 0),
    ("quiet.", 3450, 3950, 0),
]  # fmt: skip

# Each analyze flag and the facade parameter it fills, written out by hand from the command's help.
ANALYZE_FLAGS = {
    "file": "source", "file2": "second", "mode": "mode", "layout": "layout", "offset": "offset_ms",
    "swap_style": "swap_style", "questions": "questions", "transcript": "transcript", "device": "device",
    "asr": "asr", "partner_consent": "partner_consent", "confirm_long": "confirm_long",
    "light_run": "light_run", "detach": "detach",
}  # fmt: skip
OUTPUT_ONLY = {"as_json", "quiet"}  # these shape what is printed, not what is analysed
# --dry-run calls facade.dry_run instead, which takes every flag except those that only a job uses.
JOB_ONLY = {"swap_style", "questions", "transcript", "detach"}
# Every facade function and the command that calls it; the web routes join this table with the web app.
FACADE_COMMANDS = {
    "analyze": "analyze", "status": "status", "wait": "status", "jobs": "jobs list", "resume": "resume",
    "cancel": "cancel", "export": "export", "export_text": "export", "prompt": "prompt",
    "check_answer": "check-answer", "hidden_words": "check-answer", "dry_run": "analyze",
    "start_detached": "analyze", "resume_detached": "resume", "doctor": "doctor", "self_test": "self-test",
    "components": "components list", "in_use": "components install", "install_plan": "components install",
    "install": "components install", "consent": "components consent", "remove": "components remove",
    "check_removable": "components remove",
    "token_status": "token status", "token_set": "token set", "token_clear": "token clear",
    "agent_guide": "guide agents", "write_skill": "guide agents",
    "token_file_place": "token status",
    "delete_job": "jobs delete", "settings_show": "settings show", "settings_set": "settings set",
    "rerun": "rerun", "rerun_offer": "rerun", "serve": "serve",
}  # fmt: skip
# Helpers, or actions of the web pages only.
NOT_A_COMMAND = {
    "plan", "steps", "check_layout", "move_folder", "probe", "estimate_ms", "job_estimate_ms",
    "diagnostic_report", "estimates", "job_steps", "results", "job_evidence", "for_ai_files", "export_files",
    "review", "corrections", "settings_restore", "delete_all", "check_updates",
}  # fmt: skip


def test_the_envelope_lists_the_settings_warnings_then_the_jobs_own(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    job = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, 12_000)
    speakers = read_json(job / "stages" / "speakers.json")
    write_json(job / "stages" / "speakers.json", {**speakers, "warnings": ["W_SECOND_VOICE_SOLO"]})
    record = synthetic_job.finish(tmp_path)
    assert record.state is JobState.COMPLETED
    done = facade.Analysis(record, job, [("W_SETTINGS_INVALID", "general.log_level")])
    found = commands.job_warnings(done)
    assert [w.code for w in found] == ["W_SETTINGS_INVALID", "W_SECOND_VOICE_SOLO"]
    assert found[0].message.endswith("(general.log_level)")


class Terminal(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


@pytest.mark.parametrize(
    ("stdin", "stderr", "no_input", "expected"),
    [(True, True, None, True), (True, True, "1", False), (True, True, "0", True), (False, True, None, False),
     (True, False, None, False)],
)  # fmt: skip
def test_a_question_is_asked_only_at_a_terminal_and_never_with_no_input_set(
    monkeypatch: pytest.MonkeyPatch, stdin: bool, stderr: bool, no_input: str | None, expected: bool
) -> None:
    monkeypatch.setattr("sys.stdin", Terminal(stdin))
    monkeypatch.setattr("sys.stderr", Terminal(stderr))
    if no_input is None:
        monkeypatch.delenv("FLUENTFOLIO_NO_INPUT", raising=False)
    else:
        monkeypatch.setenv("FLUENTFOLIO_NO_INPUT", no_input)
    assert commands.can_prompt() is expected


def test_no_stdin_at_all_is_no_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", None)
    assert commands.can_prompt() is False


def test_progress_names_each_step_and_each_tenth_of_its_child(capsys: pytest.CaptureFixture[str]) -> None:
    show = commands.Progress(quiet=False)
    show("media", {"step": 1, "of": 18})
    for pct in (5, 10, 15, 19, 20, 35):
        show("media", {"pct": pct})
    show("vad", {"pct": 50})  # a late event of a step that is not the current one
    show("vad", {"step": 2, "of": 18})
    assert capsys.readouterr().err.splitlines() == [
        "[1/18] media", "  media 10%", "  media 20%", "  media 35%", "[2/18] vad",
    ]  # fmt: skip


def test_quiet_progress_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    show = commands.Progress(quiet=True)
    show("media", {"step": 1, "of": 18})
    show("media", {"pct": 50})
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("quiet", [False, True])
def test_a_warning_before_the_run_is_shown_even_when_quiet(
    quiet: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    detail = "the phones step needs about 2.0 GB"
    commands.Progress(quiet=quiet)("", {"warning": "W_FREE_RAM", "detail": detail})
    entry = codes()["W_FREE_RAM"]
    assert capsys.readouterr() == ("", f"Warning: {entry.message} ({detail}) {entry.hint}\n")


def record(state: JobState, error: dict[str, str] | None = None) -> facade.Analysis:
    stamp = "2026-10-15T00:00:00Z"
    job = JobRecord(job_id="20261015-000000-abcd", created_utc=stamp, updated_utc=stamp, app_version="x",
                    state=state, error=error)  # fmt: skip
    return facade.Analysis(job, Path("unused"), [])


@pytest.mark.parametrize(
    ("state", "error", "here", "exit_code", "code"),
    [
        (JobState.COMPLETED, None, True, 0, None),
        (JobState.COMPLETED_WITH_GAPS, None, False, 0, None),
        (JobState.RUNNING, None, False, 8, None),
        (JobState.FAILED, {"code": "E_JOB_FAILED", "stage": "asr"}, True, 7, "E_JOB_FAILED"),
        (JobState.FAILED, {"code": "E_MODEL_MISSING", "stage": "asr"}, True, 7, "E_MODEL_MISSING"),
        (JobState.FAILED, {"code": "E_INPUT_SILENT", "stage": "media"}, False, 7, "E_INPUT_SILENT"),
        (JobState.FAILED, {"code": "E_CONSENT_REQUIRED", "stage": "speakers"}, True, 7, "E_CONSENT_REQUIRED"),
        (JobState.FAILED, {"code": "E_STAGE_CRASHED", "stage": "asr"}, True, 7, "E_STAGE_CRASHED"),
        (JobState.FAILED, None, False, 7, "E_JOB_FAILED"),
        (JobState.CANCELLED, {"code": "E_JOB_CANCELLED"}, True, 7, "E_JOB_CANCELLED"),
        (JobState.INTERRUPTED, {"code": "E_INTERRUPTED"}, True, 130, "E_INTERRUPTED"),
        (JobState.INTERRUPTED, {"code": "E_INTERRUPTED"}, False, 7, "E_INTERRUPTED"),
        (JobState.INTERRUPTED, {"code": "E_JOB_INTERRUPTED"}, True, 7, "E_JOB_INTERRUPTED"),
    ],
)
def test_a_jobs_state_gives_0_7_or_8_and_130_only_for_a_stop_in_this_process(
    state: JobState, error: dict[str, str] | None, here: bool, exit_code: int, code: str | None
) -> None:
    found, problem = commands.job_outcome(record(state, error), here=here)
    assert found == exit_code and (problem.code if problem else None) == code
    if error and "stage" in error:
        assert problem is not None and problem.detail == f"job 20261015-000000-abcd, step {error['stage']}"


@pytest.mark.parametrize(
    ("args", "expected"),
    [(["jobs", "list", "--json"], "jobs list"), (["--json", "status", "x"], "status"), (["nope"], ""),
     (["analyze", "--mode", "solo", "a.wav"], "analyze"), (["--version", "--json"], "")],
)  # fmt: skip
def test_the_envelope_names_the_command_from_the_arguments(args: list[str], expected: str) -> None:
    assert command_of(args) == expected


def test_a_part_that_failed_to_install_names_its_code_once_then_the_next_step(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    Command(capsys, monkeypatch)  # the test process keeps its signal handlers and Hugging Face settings
    praat = catalogue.components()["praat"]
    monkeypatch.setattr(facade, "install_plan", lambda ids, every: facade.InstallPlan([praat], 0))
    monkeypatch.setattr(facade, "install", lambda planned, *heard: {"praat": AppError("E_NETWORK")})
    monkeypatch.setattr(facade, "in_use", lambda: [])
    assert main(["components", "install", "praat"]) == 3  # noqa: PLR2004 - setup, from the exit code table
    line = capsys.readouterr().out.splitlines()[-1]
    hint = AppError("E_NETWORK").entry.hint
    assert line.startswith("praat: E_NETWORK: ") and line.count("E_") == 1, line
    assert line.endswith(f". Next step: {hint}"), line


def test_the_download_line_gives_the_share_the_speed_and_the_time_left_every_two_seconds(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr(setup_commands.time, "monotonic", lambda: clock[0])
    meter = setup_commands.DownloadMeter(200_000_000)
    meter.part = "Speech recognition"
    for at, written in ((101.0, 10_000_000), (102.0, 50_000_000)):
        clock[0] = at
        meter(written)
    meter.next_file()  # the first file ended at 50 MB
    clock[0] = 162.0
    meter(10_000_000)
    # By hand: 50 MB in 2 s is 25 MB/s, so 150 MB take 6 s; then 60 MB in 62 s, so 140 MB take 145 s.
    assert capsys.readouterr().err.splitlines() == [
        "25% of 200 MB downloaded, 25.0 MB/s, less than a minute left. Now: Speech recognition.",
        "30% of 200 MB downloaded, 1.0 MB/s, about 3 min left. Now: Speech recognition.",
    ]
    start = setup_commands.DownloadMeter(200_000_000)  # under 1 MB the speed is mostly the connecting
    start.written = 999_999
    assert start.line(start.start + 10) == "0% of 200 MB downloaded."


def test_install_all_with_yes_leaves_the_licence_consent_absent(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Command(capsys, monkeypatch)
    facade.settings_set("components.opensmile", "true")  # in use, so --all would take it if it could
    monkeypatch.setattr(installer.download, "download", lambda c, models, *heard: None)  # nothing is fetched
    exit_code, found, _ = run("components", "install", "--all", "--yes")
    smile = next(r for r in facade.components() if r.id == "opensmile")
    installed = {p["id"] for p in found["data"]["parts"] if not p["code"]}
    assert exit_code == 0 and smile.status == "needs-consent" and "opensmile" not in installed
    assert not (data_root() / installer.CONSENT_FILE).exists()


def test_each_analyze_flag_fills_a_facade_parameter() -> None:
    flags = {p.name for p in commands.analyze.params} - OUTPUT_ONLY - {"dry_run"}
    assert flags == set(ANALYZE_FLAGS)
    accepted = inspect.signature(facade.analyze).parameters
    assert all(ANALYZE_FLAGS[flag] in accepted for flag in flags)
    planned = inspect.signature(facade.dry_run).parameters
    assert all(ANALYZE_FLAGS[flag] in planned for flag in flags - JOB_ONLY)


def test_token_set_says_that_accepting_the_pyannote_terms_shares_contact_details() -> None:
    token = cli.commands["token"]
    assert isinstance(token, click.Group)
    help_text = " ".join((token.commands["set"].help or "").split())
    assert "Accepting them shares your contact details with pyannote's makers." in help_text


def test_every_facade_function_has_its_command() -> None:
    public = {
        name for name, value in vars(facade).items()
        if inspect.isfunction(value) and value.__module__ == facade.__name__ and not name.startswith("_")
    }  # fmt: skip
    assert public == set(FACADE_COMMANDS) | NOT_A_COMMAND
    names: set[str] = set()
    for name, command in cli.commands.items():
        names.add(name)
        if isinstance(command, click.Group):
            names |= {f"{name} {sub}" for sub in command.commands}
    assert set(FACADE_COMMANDS.values()) <= names


def test_no_command_that_takes_the_partners_agreement_has_a_yes_switch() -> None:
    # --yes answers size and delete questions only, so it can never stand in for the other person's agreement.
    leaves = [
        (f"{name} {sub}", c) for name, group in cli.commands.items() if isinstance(group, click.Group)
        for sub, c in group.commands.items()
    ] + [(name, c) for name, c in cli.commands.items() if not isinstance(c, click.Group)]  # fmt: skip
    switches = {name: {o for p in c.params for o in p.opts} for name, c in leaves}
    agreeing = sorted(name for name, opts in switches.items() if "--partner-consent" in opts)
    assert agreeing == ["analyze", "rerun"]
    assert all("--yes" not in switches[name] for name in agreeing)
    answering = {name for name, opts in switches.items() if "--yes" in opts}
    assert {"jobs delete", "components install"} <= answering  # the switch exists, just not there
