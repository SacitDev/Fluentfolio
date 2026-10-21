"""The commands of the fluentfolio command line; each one calls the facade and maps its outcome."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from fluentfolio.app import facade
from fluentfolio.cli import envelope as env
from fluentfolio.cli.envelope import emit, warning
from fluentfolio.core.enums import Device, JobState, Layout, Mode
from fluentfolio.core.errors import AppError
from fluentfolio.core.files import read_json
from fluentfolio.export.answer_check import NO_AUDIO_LINE
from fluentfolio.export.bundle import NAMES
from fluentfolio.export.for_ai import PROMPT_VERSION, approx_tokens
from fluentfolio.jobs.store import utc_now
from fluentfolio.settings.model import Settings

FINISHED = (JobState.COMPLETED, JobState.COMPLETED_WITH_GAPS)
JOB_FAILED, STILL_RUNNING, INTERRUPTED = 7, 8, 130
AUDIO_RULE = "AC-11"
AUDIO_FAILED = (
    "Ignore Pronunciation and the overall estimate from this answer: the chat did not hear the audio. "
    "Ask again without the audio file, so it estimates Pronunciation from the clues in the text."
)
CONSENT_QUESTION = "Did the other person agree to be recorded and analysed on this computer?"
PROGRESS_STEP = 10  # a child's percentage is shown each time it grows by this much
NEEDS = {
    "partner_consent": "the other person's agreement (--partner-consent, only when they said so)",
    "confirm_long": "a long recording (--confirm-long, or --light-run for a faster run), after seeing the "
    "estimate",
}

json_option = click.option(
    "--json", "as_json", is_flag=True, help="Print one JSON object for scripts and AI agents."
)
job_argument = click.argument("job_id", metavar="JOB")
# A deletion's one plain limit, told before it happens.
DELETE_NOTE = "Deleted files skip the Recycle Bin; on an SSD or a synced folder this is not secure erasure."


def confirmed(question: str, yes: bool, detail: str) -> bool:
    """A size or delete question: --yes answers it; with no person at a terminal, the command stops."""
    if yes:
        return True
    if not can_prompt():
        raise AppError("E_CONFIRM_NEEDED", detail)
    return click.confirm(question, default=False, err=True)


yes_option = click.option("--yes", is_flag=True, help="Answer the size or delete question with yes. It never "
                          "accepts a licence or sets a token.")  # fmt: skip


def can_prompt() -> bool:
    """Questions are asked only of a person at a terminal, never of a script or an agent."""
    stdin = sys.stdin
    return (
        bool(stdin and stdin.isatty() and sys.stderr.isatty())
        and os.environ.get("FLUENTFOLIO_NO_INPUT") != "1"
    )


def refuse(command: str, exc: AppError, as_json: bool) -> int:
    """A command that could not do what was asked: the envelope, or the message and next step on stderr."""
    if as_json:
        return emit(command, exc.entry.exit_code, None, [], exc)
    click.echo(str(exc), err=True)  # the error's own text starts with its code
    click.echo(f"Next step: {exc.entry.hint}", err=True)
    return exc.entry.exit_code


class Progress:
    """Each step's name and its child's percentage on stderr while a job runs in the foreground."""

    def __init__(self, quiet: bool) -> None:
        self.quiet, self.stage, self.pct = quiet, "", 0

    def __call__(self, stage: str, event: dict[str, Any]) -> None:
        if "warning" in event:  # before the run starts; shown with --quiet too, since it is not progress
            notice = warning(event["warning"], event.get("detail", ""))
            click.echo(f"Warning: {notice.message} {notice.hint}", err=True)
            return
        if self.quiet:
            return
        if "step" in event:
            self.stage, self.pct = stage, 0
            click.echo(f"[{event['step']}/{event['of']}] {stage}", err=True)
        elif (
            stage == self.stage
            and isinstance(pct := event.get("pct"), int)
            and pct >= self.pct + PROGRESS_STEP
        ):
            self.pct = pct
            click.echo(f"  {stage} {pct}%", err=True)


def _ms_between(start: str, end: str) -> int:
    stamp = "%Y-%m-%dT%H:%M:%SZ"
    return int((datetime.strptime(end, stamp) - datetime.strptime(start, stamp)).total_seconds() * 1000)


def outputs(job: Path, files: list[Path] | None = None) -> env.Outputs:
    """Where a job's evidence, ZIP and texts for an AI chat are; `files` points at copies elsewhere."""
    evidence = job / "evidence.json"
    found = {p.name: p for p in files or []}
    zipped = found.get(f"fluentfolio_{job.name}.zip", job / "export" / f"fluentfolio_{job.name}.zip")
    for_ai = []
    for c in read_json(evidence)["candidates"] if evidence.is_file() else []:
        path = found.get(NAMES[c["id"]], job / "export" / "for_ai" / NAMES[c["id"]])
        if path.is_file():
            tokens = approx_tokens(path.read_text(encoding="utf-8"))
            for_ai.append(env.ForAi(candidate=c["id"], path=path.as_posix(), approx_tokens=tokens))
    return env.Outputs(
        job_dir=job.as_posix(), evidence=evidence.as_posix() if evidence.is_file() else None,
        zip=zipped.as_posix() if zipped.is_file() else None, for_ai=for_ai,
    )  # fmt: skip


def audio_files(job: Path, files: list[Path] | None = None) -> list[env.AudioFile]:
    """The optional audio file of each candidate, where the learner finds it, with its length."""
    found = {p.name: p for p in files or []}
    made = job / "stages" / "ai_audio.json"
    rows = read_json(made)["candidates"] if made.is_file() else {}
    return [
        env.AudioFile(
            candidate=cid, path=found.get(row["file"], job / "export" / "for_ai" / row["file"]).as_posix(),
            duration_ms=row["duration_ms"],
        )
        for cid, row in rows.items()
    ]  # fmt: skip


def no_audio_reason(done: facade.Analysis) -> str:
    """Why a job asked for its audio file has none: the step's own reason, or the code it ended with."""
    made = done.job / "stages" / "ai_audio.json"
    reason = read_json(made).get("reason") if made.is_file() else None
    step = next((s for s in done.record.stages if s.id == "ai_audio"), None)
    return reason or f"the audio step ended with {step.error_code if step else 'no result'}"


def job_data(done: facade.Analysis) -> env.JobData:
    """What an agent needs about a job: state, outputs, limits, and what ran where for how long.

    It holds no transcript text and no input file name.
    """
    record, job = done.record, done.job
    evidence = read_json(job / "evidence.json") if (job / "evidence.json").is_file() else None
    tools = {t["id"]: t for t in evidence["manifest"]["tools"]} if evidence else {}
    media = job / "stages" / "media.json"
    running = record.state is JobState.RUNNING
    stage = None
    if running and record.stage is not None:
        stage = env.StageAt(id=record.stage, index=len(record.stages), total=facade.steps(record))
    return env.JobData(
        job_id=record.job_id, state=record.state, stage=stage, created_utc=record.created_utc,
        updated_utc=record.updated_utc,
        elapsed_ms=_ms_between(record.created_utc, utc_now() if running else record.updated_utc),
        estimate_ms=facade.job_estimate_ms(record), mode=record.mode, asr=record.asr,
        device=record.device.get("resolved"),
        # The decoded length once the media step has one; before that, the length read before the job began.
        audio_duration_ms=(read_json(media) if media.is_file() else {}).get("duration_ms")
        or record.input.get("duration_ms"),
        outputs=outputs(job),
        limits=[
            env.Limit(component=t["id"], code=t["reason_code"] or t["status"], message=t["reason"] or "")
            for t in tools.values() if t["status"] != "ok"
        ],
        stub=evidence["manifest"]["stub"] if evidence else None,
        commit=evidence["manifest"]["commit"] if evidence else None,
        stages=[
            env.StageRun(id=s.id, status=s.status, ran=tools.get(s.id, {}).get("ran"),
                         version=tools.get(s.id, {}).get("version"), device=s.device, wall_ms=s.wall_ms,
                         peak_ram_mb=s.peak_ram_mb, error_code=s.error_code)
            for s in record.stages
        ],
    )  # fmt: skip


def job_warnings(done: facade.Analysis) -> list[env.Notice]:
    """The settings' warnings, then the job's own as its evidence lists them."""
    evidence = done.job / "evidence.json"
    found = read_json(evidence)["manifest"]["warnings"] if evidence.is_file() else []
    return [warning(code, detail) for code, detail in done.warnings] + [warning(w["code"]) for w in found]


def job_outcome(done: facade.Analysis, *, here: bool, started: bool = False) -> tuple[int, AppError | None]:
    """The exit code of a job's state, 0, 7 or 8, and the error behind a 7.

    A job that failed for any reason gives 7; its error's code and category say why, so an agent can tell a
    missing model from a failed step. `here` says this process ran the job, so a Ctrl+C that stopped it
    gives 130. `started` says this process only started it in the background, so a running job is the
    expected outcome, 0.
    """
    record = done.record
    if record.state in FINISHED:
        return 0, None
    if record.state is JobState.RUNNING:
        return (0 if started else STILL_RUNNING), None
    error = record.error or {"code": "E_JOB_FAILED"}
    step = f", step {error['stage']}" if error.get("stage") else ""
    problem = AppError(error["code"], f"job {record.job_id}{step}")
    return (INTERRUPTED if here and problem.code == "E_INTERRUPTED" else JOB_FAILED), problem


def report_job(
    command: str, done: facade.Analysis, as_json: bool, exit_code: int, problem: AppError | None
) -> int:
    """Print a job's state, its texts for an AI chat and its limits; IDs, counts and paths only."""
    data = job_data(done)
    where = f" (step {data.stage.index} of {data.stage.total}: {data.stage.id})" if data.stage else ""
    summary = f"job {data.job_id}: {data.state}{where}"
    if as_json:
        click.echo(summary, err=True)
        return emit(command, exit_code, data, job_warnings(done), problem)
    click.echo(summary)
    for item in data.outputs.for_ai:
        click.echo(f"Text for an AI chat: {item.path} (about {item.approx_tokens} tokens)")
    for limit in data.limits:
        click.echo(f"Limit: {limit.component}: {limit.message or limit.code}")
    if problem is not None:
        click.echo(str(problem), err=True)
        click.echo(f"Next step: {problem.entry.hint.replace('JOB', data.job_id)}", err=True)
    return exit_code


def _offset_ms(ctx: click.Context, param: click.Parameter, value: str | None) -> int | None:
    """mm:ss or mm:ss.s, as the time line of the first file shows it."""
    if value is None:
        return None
    minutes, sep, seconds = value.partition(":")
    try:
        ms = round((int(minutes) * 60 + float(seconds)) * 1000) if sep else -1
    except ValueError:
        ms = -1
    if ms < 0 or not seconds.replace(".", "", 1).isdigit():
        raise click.BadParameter("use mm:ss or mm:ss.s, for example 01:30.5")
    return ms


FILE = click.Path(exists=True, dir_okay=False, path_type=Path)


def examples(*lines: str) -> str:
    """A help epilog of example commands, kept on their own lines."""
    return "\b\nExamples:\n" + "\n".join(f"  {line}" for line in lines)


@click.command(epilog=examples(
    "fluentfolio analyze practice.m4a --mode solo",
    "fluentfolio analyze mock.m4a --mode interview --partner-consent --json",
))  # fmt: skip
@click.argument("file", type=FILE)
@click.argument("file2", type=FILE, required=False)
@click.option("--mode", required=True, type=click.Choice([m.value for m in Mode]),
              help="solo: you read each question aloud, then answer it. interview: one person asks, one "
                   "answers. role-swap: two learners take turns asking.")  # fmt: skip
@click.option(
    "--layout",
    type=click.Choice(list(facade.LAYOUTS)),
    default=Layout.MIXED.value,
    show_default=True,
    help="mixed: one voice track for both; per-track: one track per person; two-files: FILE2 too.",
)
@click.option("--offset", callback=_offset_ms, help="two-files: where FILE2 starts in FILE, as mm:ss.")
@click.option("--swap-style", type=click.Choice(list(facade.SWAP_STYLES)),
              help="role-swap: once (or once per Part, the default) or per-question.")  # fmt: skip
@click.option("--questions", type=FILE, help="Your question list (.txt), with optional Part 1/2/3 headings.")
@click.option(
    "--transcript",
    type=FILE,
    help="Your own transcript (.txt, .md, .vtt, .srt); the names found in its speaker labels are replaced,"
    " but some labels and forms of a name are not found (docs/limitations.md).",
)
@click.option("--device", type=click.Choice([d.value for d in Device]), help="Where the analysis runs.")
@click.option("--asr", help="The speech recognition model for this run (a component ID).")
@click.option("--partner-consent", is_flag=True,
              help="interview and role-swap: the other person agreed to be recorded and analysed on this "
                   "computer. Pass it only when they said so.")  # fmt: skip
@click.option("--confirm-long", is_flag=True,
              help="Go on with a recording over 60 minutes, which takes long. Pass it only after the person "
                   "saw the estimate from --dry-run.")  # fmt: skip
@click.option("--light-run", is_flag=True,
              help="Over 90 minutes: a faster run with the MFA aligner, the phone model and openSMILE "
                   "off for this recording. The results say so.")  # fmt: skip
@click.option("--dry-run", is_flag=True,
              help="Check the file and show its length, the time estimate, the steps, what is missing and "
                   "what a person must decide; start nothing.")  # fmt: skip
@click.option("--detach", is_flag=True,
              help="Run in the background and return at once; follow it with status --wait.")  # fmt: skip
@click.option("--quiet", is_flag=True, help="Show no progress on stderr.")
@json_option
def analyze(
    file: Path, file2: Path | None, mode: str, layout: str, offset: int | None, swap_style: str | None,
    questions: Path | None, transcript: Path | None, device: str | None, asr: str | None,
    partner_consent: bool, confirm_long: bool, light_run: bool, dry_run: bool, detach: bool, quiet: bool,
    as_json: bool,
) -> int:  # fmt: skip
    """Analyse a practice recording and write the evidence and the text for an AI chat."""
    if dry_run and detach:
        raise click.UsageError("--dry-run starts nothing, so it takes no --detach")
    consent: facade.Consent | None = "cli-flag" if partner_consent else None
    if mode != Mode.SOLO and consent is None and can_prompt() and not dry_run:
        consent = "cli-tty" if click.confirm(CONSENT_QUESTION, default=False, err=True) else None
    options: dict[str, Any] = {
        "device": Device(device) if device else None, "asr": asr, "layout": layout, "second": file2,
        "offset_ms": offset, "partner_consent": consent, "confirm_long": confirm_long, "light_run": light_run,
    }  # fmt: skip
    try:
        if dry_run:
            return report_plan(facade.dry_run(file, mode, **options), as_json)
        done = facade.analyze(
            file, mode, swap_style=swap_style, questions=questions, transcript=transcript, detach=detach,
            on_progress=Progress(quiet), **options,
        )  # fmt: skip
    except AppError as exc:
        return refuse("analyze", exc, as_json)
    return report_job("analyze", done, as_json, *job_outcome(done, here=not detach, started=detach))


def report_plan(planned: facade.Planned, as_json: bool) -> int:
    """What `analyze --dry-run` found: the file, the estimate, the steps, and what is missing or needed."""
    data = env.DryRun(
        probe=[env.Probe(**p) for p in planned.probe], audio_duration_ms=planned.duration_ms,
        estimate_ms=planned.estimate_ms, device=planned.device["resolved"], will_run=planned.will_run,
        missing=planned.missing, needs=planned.needs,  # type: ignore[arg-type]
    )  # fmt: skip
    if as_json:
        return emit("analyze", 0, data, [warning(code, detail) for code, detail in planned.warnings])
    first = planned.probe[0]
    click.echo(
        f"Length: {planned.duration_ms // 1000} s; {first['container']}, {first['codec']}, "
        f"{first['sample_rate']} Hz, {first['channels']} channel(s), {first['tracks']} track(s)"
    )
    click.echo(
        f"Estimate: about {max(1, round(planned.estimate_ms / 60000))} min on the {data.device.upper()}"
    )
    click.echo(f"Steps: {', '.join(planned.will_run)}")
    if planned.missing:
        click.echo(f"Not ready, so the results will have limits: {', '.join(planned.missing)}")
    for need in planned.needs:
        click.echo(f"Needs a person's decision: {NEEDS[need]}")
    for code, detail in planned.warnings:
        notice = warning(code, detail)
        click.echo(f"Warning: {notice.message} {notice.hint}")
    return 0


@click.command(epilog=examples(
    "fluentfolio status 20261013-101500-ab12",
    "fluentfolio status 20261013-101500-ab12 --wait --timeout 100 --json",
))  # fmt: skip
@job_argument
@click.option("--wait", is_flag=True, help="Wait until the job stops running, or until the timeout.")
@click.option("--timeout", type=click.IntRange(min=0), default=100, show_default=True,
              help="With --wait: seconds to wait. Exit code 8: the job still runs; ask again.")  # fmt: skip
@json_option
def status(job_id: str, wait: bool, timeout: int, as_json: bool) -> int:
    """A job's state, step, time so far, outputs and limits."""
    try:
        done = facade.status(job_id)
        if wait and done.record.state is JobState.RUNNING:
            click.echo(f"Waiting up to {timeout} s for job {job_id} to stop running.", err=True)
            done = facade.wait(job_id, timeout)
    except AppError as exc:
        return refuse("status", exc, as_json)
    exit_code, problem = job_outcome(done, here=False)
    # Without --wait the command did its job by reading the state, so it exits 0 whatever the state is.
    return report_job("status", done, as_json, *((exit_code, problem) if wait else (0, None)))


@click.command(epilog=examples(
    "fluentfolio resume 20261013-101500-ab12", "fluentfolio resume 20261013-101500-ab12 --json"
))  # fmt: skip
@job_argument
@click.option("--detach", is_flag=True,
              help="Run in the background and return at once; follow it with status --wait.")  # fmt: skip
@click.option("--quiet", is_flag=True, help="Show no progress on stderr.")
@json_option
def resume(job_id: str, detach: bool, quiet: bool, as_json: bool) -> int:
    """Run a job again from its first unfinished step and every step after it; a completed job stays as is."""
    try:
        done = (
            facade.resume_detached(job_id) if detach else facade.resume(job_id, on_progress=Progress(quiet))
        )
    except AppError as exc:
        return refuse("resume", exc, as_json)
    return report_job("resume", done, as_json, *job_outcome(done, here=not detach, started=detach))


@click.command(epilog=examples(
    "fluentfolio rerun 20261013-101500-ab12 --mode interview",
    "fluentfolio rerun 20261013-101500-ab12 --mode solo --json",
))  # fmt: skip
@job_argument
@click.option("--mode", required=True, type=click.Choice([m.value for m in Mode]),
              help="The mode to find the roles in this time.")  # fmt: skip
@click.option("--partner-consent", is_flag=True,
              help="From solo to interview or role-swap: the other person agreed to be recorded and analysed "
                   "on this computer. Pass it only when they said so.")  # fmt: skip
@click.option("--detach", is_flag=True,
              help="Run in the background and return at once; follow it with status --wait.")  # fmt: skip
@click.option("--quiet", is_flag=True, help="Show no progress on stderr.")
@json_option
def rerun(job_id: str, mode: str, partner_consent: bool, detach: bool, quiet: bool, as_json: bool) -> int:
    """Run a finished job again as another mode, in a new job; its speech is not transcribed again."""
    consent: facade.Consent | None = "cli-flag" if partner_consent else None
    try:
        offer = facade.rerun_offer(job_id, mode)
        if offer.needs_consent and consent is None and can_prompt():
            consent = "cli-tty" if click.confirm(CONSENT_QUESTION, default=False, err=True) else None
        if offer.estimate_ms is not None and not quiet:
            minutes = max(1, round(offer.estimate_ms / 60000))
            where = offer.device["resolved"].upper()
            click.echo(
                f"Runs again from the {offer.first} step: about {minutes} min on the {where}.", err=True
            )
        done = facade.rerun(job_id, mode, partner_consent=consent, detach=detach, on_progress=Progress(quiet))
    except AppError as exc:
        return refuse("rerun", exc, as_json)
    return report_job("rerun", done, as_json, *job_outcome(done, here=not detach, started=detach))


@click.command("_run", hidden=True)
@job_argument
def run_detached(job_id: str) -> int:
    """The runner that --detach starts: the job in the foreground of a process with no window."""
    try:
        done = facade.resume(job_id)
    except AppError:
        return 1  # nobody reads this process's output; the job's own state says what happened
    return job_outcome(done, here=True)[0]


@click.command(epilog=examples(
    "fluentfolio cancel 20261013-101500-ab12", "fluentfolio cancel 20261013-101500-ab12 --json"
))  # fmt: skip
@job_argument
@json_option
def cancel(job_id: str, as_json: bool) -> int:
    """Ask a running job to stop; it stops within seconds and keeps its finished steps."""
    try:
        record, requested = facade.cancel(job_id)
    except AppError as exc:
        return refuse("cancel", exc, as_json)
    if as_json:
        return emit("cancel", 0, env.CancelData(job_id=job_id, state=record.state, requested=requested), [])
    click.echo(f"job {job_id}: " + ("asked to stop" if requested else f"not running ({record.state})"))
    return 0


@click.group(epilog=examples("fluentfolio jobs list", "fluentfolio jobs list --limit 1 --json"))
def jobs() -> None:
    """The jobs in your data folder."""


@jobs.command("list", epilog=examples("fluentfolio jobs list", "fluentfolio jobs list --limit 1 --json"))
@click.option("--limit", type=click.IntRange(min=1), default=10, show_default=True, help="How many jobs.")
@json_option
def jobs_list(limit: int, as_json: bool) -> int:
    """The newest jobs first: ID, start time, state, mode and length; never file names."""
    rows = [
        env.JobRow(job_id=r.job_id, created_utc=r.created_utc, state=r.state, mode=r.mode,
                   audio_duration_ms=r.input.get("duration_ms"))
        for r in facade.jobs(limit)
    ]  # fmt: skip
    if as_json:
        return emit("jobs list", 0, env.JobList(jobs=rows), [])
    for row in rows:
        length = f"{row.audio_duration_ms // 1000} s" if row.audio_duration_ms is not None else "-"
        click.echo(f"{row.job_id}  {row.created_utc}  {row.state:<19}  {row.mode or '-':<9}  {length}")
    if not rows:
        click.echo("No jobs yet.")
    return 0


@jobs.command("delete", epilog=examples(
    "fluentfolio jobs delete 20261013-101500-ab12",
    "fluentfolio jobs delete 20261013-101500-ab12 --yes --json",
))  # fmt: skip
@job_argument
@yes_option
@json_option
def jobs_delete(job_id: str, yes: bool, as_json: bool) -> int:
    """Delete one job: its copy of the recording, its results and its temporary files."""
    facade.status(job_id)  # an unknown job is refused before the question
    deleted = confirmed(f"Delete job {job_id}? {DELETE_NOTE}", yes, "a job to delete")
    if deleted:
        facade.delete_job(job_id)
    if as_json:
        return emit("jobs delete", 0, env.Deleted(job_id=job_id, deleted=deleted), [])
    click.echo(f"Deleted job {job_id}. {DELETE_NOTE}" if deleted else "Nothing was deleted.")
    return 0


@click.command(epilog=examples(
    "fluentfolio export 20261013-101500-ab12 --out ./for-my-ai",
    "fluentfolio export 20261013-101500-ab12 --print --candidate person1",
))  # fmt: skip
@job_argument
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              help="Copy the texts for an AI chat, any audio file for it and the ZIP here.")  # fmt: skip
@click.option("--print", "show", is_flag=True, help="Print one person's text for an AI chat to stdout.")
@click.option("--candidate", type=click.Choice(list(NAMES)),
              help="role-swap: whose text --print shows (person1 or person2).")  # fmt: skip
@click.option("--audio", is_flag=True,
              help="Also make the audio file of each person's own speech for the pronunciation check "
                   "(at most 9.5 min each; the partner can still be heard where the speakers were mixed up "
                   "or both spoke at once). "
                   "Only this step and the export run again.")  # fmt: skip
@json_option
def export(
    job_id: str, out: Path | None, show: bool, candidate: str | None, audio: bool, as_json: bool
) -> int:
    """Give the paths of a finished job's texts for an AI chat and its ZIP; write a missing part again."""
    if show and (as_json or out is not None or audio):
        raise click.UsageError("--print prints the text alone, so it takes none of --json, --out and --audio")
    try:
        if show:
            click.echo(facade.export_text(job_id, candidate), nl=False)
            return 0
        done, files, changed = facade.export(job_id, out, audio=audio)
    except AppError as exc:
        return refuse("export", exc, as_json)
    copies = files if out is not None else None
    where, sound = outputs(done.job, copies), audio_files(done.job, copies)
    if as_json:
        return emit("export", 0, env.Exported(outputs=where, audio=sound, changed=changed), [])
    for item in where.for_ai:
        click.echo(f"Text for an AI chat: {item.path} (about {item.approx_tokens} tokens)")
    for clip in sound:
        minutes = clip.duration_ms / 60000
        click.echo(f"Audio for an AI chat: {clip.path} ({minutes:.1f} min of one person's own speech)")
    if audio and not sound:
        click.echo(f"No audio file: {no_audio_reason(done)}.")
    if where.zip:
        click.echo(f"ZIP: {where.zip}")
    click.echo("Written again." if changed else "Nothing was missing, so nothing was written again.")
    return 0


@click.command(epilog=examples("fluentfolio prompt", "fluentfolio prompt --kind pronunciation --json"))
@click.option("--kind", type=click.Choice(list(facade.PROMPT_FILES)), default="grading", show_default=True,
              help="grading: the full grading prompt. pronunciation: the one for an audio file.")  # fmt: skip
@json_option
def prompt(kind: str, as_json: bool) -> int:
    """Print a prompt for an AI chat on its own, without any data ("Copy prompt only")."""
    text = facade.prompt(kind)
    if as_json:
        return emit("prompt", 0, env.PromptData(kind=kind, prompt_version=PROMPT_VERSION, text=text), [])  # type: ignore[arg-type]
    click.echo(text, nl=False)
    return 0


@click.command("check-answer", epilog=examples(
    "fluentfolio check-answer 20261013-101500-ab12 answer.md",
    "fluentfolio check-answer 20261013-101500-ab12 answer.md --candidate person2 --json",
))  # fmt: skip
@job_argument
@click.argument("answer", metavar="ANSWER_FILE", type=FILE)
@click.option("--candidate", type=click.Choice(list(NAMES)),
              help="role-swap: which person the answer is about (person1 or person2).")  # fmt: skip
@json_option
def check_answer(job_id: str, answer: Path, candidate: str | None, as_json: bool) -> int:
    """Check an AI chat's answer (.md or .txt) against the job's text: each rule, PASS, FAIL or N/A."""
    try:
        report = facade.check_answer(job_id, answer, candidate)
    except AppError as exc:
        return refuse("check-answer", exc, as_json)
    heard = next(r for r in report.results if r.rule == AUDIO_RULE)
    if as_json:
        data = env.Checked(
            job_id=job_id, passed=report.passed, counts=report.counts,
            results=[env.RuleResult(rule=r.rule, name=r.name, verdict=r.verdict, detail=r.detail)
                     for r in report.results],
        )  # fmt: skip
        return emit("check-answer", 0, data, [])
    counts = report.counts
    click.echo(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['N/A']} not applicable")
    for r in report.results:
        click.echo(f"{r.rule} {r.name}: {r.verdict}" + (f" ({r.detail})" if r.detail else ""))
    words = facade.hidden_words(job_id, candidate)
    # The advice fits only an audio check that was due: a file was made and the answer has its line.
    if heard.verdict == "FAIL" and words is not None and heard.detail != NO_AUDIO_LINE:
        click.echo(AUDIO_FAILED)
    # Shown only after the check, and never in --json, which carries no transcript text.
    if words is not None:
        click.echo(f"The hidden words of the audio check: {words}")
    return 0


@click.group(epilog=examples("fluentfolio guide agents", "fluentfolio guide agents --write-skill ./skills"))
def guide() -> None:
    """Guides for programs that run Fluentfolio."""


@guide.command("agents", epilog=examples(
    "fluentfolio guide agents", "fluentfolio guide agents --write-skill ./skills --json",
))  # fmt: skip
@click.option("--write-skill", "folder", type=click.Path(file_okay=False, path_type=Path),
              help="Write FOLDER/fluentfolio/SKILL.md, the short guide for agents that read "
                   "Agent Skills.")  # fmt: skip
@json_option
def guide_agents(folder: Path | None, as_json: bool) -> int:
    """Print the guide for AI agents: the usual path, the JSON reply, the exit codes and what not to do."""
    written = facade.write_skill(folder) if folder is not None else None
    if as_json:
        path = written.as_posix() if written else None
        return emit("guide agents", 0, env.GuideData(text=facade.agent_guide(), path=path), [])
    click.echo(f"Wrote {written.as_posix()}." if written else facade.agent_guide(), nl=written is not None)
    return 0


@click.group(epilog=examples("fluentfolio settings show", "fluentfolio settings set export.audio true"))
def settings() -> None:
    """Show or change the settings in settings.toml, as the Settings page does."""


def settings_lines(found: Settings) -> list[str]:
    """One `section.key = value` line per setting, in TOML spelling."""
    lines = []
    for section, values in found.model_dump(mode="json").items():
        if isinstance(values, dict):
            lines += [f"{section}.{key} = {json.dumps(value)}" for key, value in values.items()]
    return lines


@settings.command("show", epilog=examples("fluentfolio settings show", "fluentfolio settings show --json"))
@json_option
def settings_show(as_json: bool) -> int:
    """Show every setting and its value; a value in settings.toml replaces the default.

    With --json, each duration is in whole milliseconds under a _ms name, as in every reply.
    """
    loaded, path = facade.settings_show()
    notes = [warning(code, detail) for code, detail in loaded.warnings]
    if as_json:
        data = env.settings_data(loaded.settings, path.as_posix(), loaded.read_only)
        return emit("settings show", 0, data, notes)
    for line in settings_lines(loaded.settings):
        click.echo(line)
    for note in notes:
        click.echo(f"Note: {note.message}", err=True)
    return 0


@settings.command("set", epilog=examples(
    "fluentfolio settings set export.audio true", "fluentfolio settings set limits.confirm_min 90 --json",
))  # fmt: skip
@click.argument("key", metavar="KEY")
@click.argument("value", metavar="VALUE")
@json_option
def settings_set(key: str, value: str, as_json: bool) -> int:
    """Change one setting, named section.key; it applies to jobs started after it, never to a running one."""
    try:
        found, path = facade.settings_set(key, value)
    except AppError as exc:
        return refuse("settings set", exc, as_json)
    if as_json:
        return emit("settings set", 0, env.settings_data(found, path.as_posix(), read_only=False), [])
    line = next(x for x in settings_lines(found) if x.startswith(f"{key} = "))
    click.echo(f"Saved in {path.as_posix()}: {line}")
    return 0


@click.command(epilog=examples("fluentfolio serve", "fluentfolio serve --port 8765 --no-browser"))
@click.option(
    "--port", type=click.IntRange(0, 65535), default=0, show_default=True,
    help="The port on 127.0.0.1; 0 picks a free one.",
)  # fmt: skip
@click.option("--no-browser", is_flag=True, help="Print the file to open instead of opening the browser.")
def serve(port: int, no_browser: bool) -> int:
    """Start the web app on this computer and open it; close the window, press Ctrl+C or use Quit to stop."""
    return facade.serve(port, browser=not no_browser)


COMMANDS: list[Callable[..., Any]] = [
    analyze, status, jobs, resume, rerun, cancel, export, prompt, check_answer, guide, settings, serve,
    run_detached,
]  # fmt: skip
