"""The commands that check and set up Fluentfolio: doctor, components and token."""

from __future__ import annotations

import math
import shutil
import sys
import time
from collections.abc import Callable
from importlib import util
from pathlib import Path
from typing import Any

import click

from fluentfolio.app import facade
from fluentfolio.cli import envelope as env
from fluentfolio.cli.commands import (
    Progress,
    can_prompt,
    confirmed,
    examples,
    json_option,
    refuse,
    yes_option,
)
from fluentfolio.cli.envelope import emit, warning
from fluentfolio.core.enums import Device
from fluentfolio.core.errors import AppError
from fluentfolio.core.files import size_text
from fluentfolio.doctor.checks import ComponentRow
from fluentfolio.models import catalogue
from fluentfolio.setup import components as installer
from fluentfolio.setup.token import TokenStatus

SETUP_FAILED = 3
GB = 1e9
LIVE_EVERY_S, LINE_EVERY_S = 0.5, 2.0  # how often the download line is drawn in a terminal, or written
SPEED_AFTER_BYTES = 1_000_000  # the first bytes come while connecting, so their speed would mislead


class DownloadMeter:
    """How much of the planned download is on disk, with the speed and the time left, on stderr: drawn again
    in place in a terminal, otherwise a line every 2 s, which the Components page reads its bar from."""

    def __init__(self, total: int) -> None:
        self.total, self.live = total, sys.stderr.isatty()
        self.finished = self.written = 0  # the files before this one, and this one so far
        self.part, self.drawn = "", False
        self.start = self.shown = time.monotonic()

    def next_file(self) -> None:
        self.finished, self.written = self.finished + self.written, 0

    def __call__(self, written: int) -> None:
        self.written, now = written, time.monotonic()
        if now - self.shown < (LIVE_EVERY_S if self.live else LINE_EVERY_S):
            return
        self.shown = now
        if self.live:
            width = shutil.get_terminal_size().columns - 1
            click.echo("\r" + self.line(now)[:width].ljust(width), err=True, nl=False)
            self.drawn = True
        else:
            click.echo(self.line(now), err=True)

    def end(self) -> None:
        """End a line drawn in place, so the next line starts below it."""
        if self.drawn:
            click.echo("", err=True)
            self.drawn = False

    def line(self, now: float) -> str:
        done, seconds = self.finished + self.written, now - self.start
        share = int(min(done / self.total, 1) * 100) if self.total else 0
        text = f"{share}% of {size_text(self.total)} downloaded"
        if done >= SPEED_AFTER_BYTES and seconds >= 1:
            rate = done / seconds
            left = max(self.total - done, 0) / rate
            text += f", {rate / 1e6:.1f} MB/s, " + (
                "less than a minute left" if left < 60 else f"about {math.ceil(left / 60)} min left"  # noqa: PLR2004
            )
        return f"{text}. Now: {self.part}." if self.part else f"{text}."


def _rows(found: list[ComponentRow]) -> list[env.ComponentInfo]:
    return [env.ComponentInfo(**vars(r)) for r in found]


@click.command(epilog=examples("fluentfolio doctor", "fluentfolio doctor --report report.txt --json"))
@click.option("--report", type=click.Path(dir_okay=False, path_type=Path),
              help="Also write a report to share when asking for help: no names, recordings, transcripts or "
                   "tokens. Read it first.")  # fmt: skip
@json_option
def doctor(report: Path | None, as_json: bool) -> int:
    """Check the install, the disks, the computer and each part; exit code 3 when no analysis can run."""
    found, written = facade.doctor(report)
    exit_code = SETUP_FAILED if found.blocking else 0
    if as_json:
        data = env.DoctorData(
            blocking=found.blocking, executable=found.executable, data_dir=found.data_dir,
            device=env.DeviceChoice(**found.device), report=written.as_posix() if written else None,
            checks=[env.CheckRow(**vars(c)) for c in found.checks],
        )  # fmt: skip
        # The first blocking problem is the error; the checks stay in `data`, as a failed job's status does.
        first = next((c for c in found.checks if c.blocking and c.code), None)
        problem = AppError(first.code, first.message) if first and first.code else None
        return emit("doctor", exit_code, data, [], problem)
    for c in found.checks:
        click.echo(f"[{c.level}] {c.group}: {c.message}" + (f" {c.code}: {c.fix}" if c.code else ""))
    if written:
        click.echo(f"Diagnostic report: {written.as_posix()}. Read it before you share it.")
    click.echo("No analysis can run until the problems above are fixed." if found.blocking else "Ready.")
    return exit_code


@click.command(
    "self-test", epilog=examples("fluentfolio self-test", "fluentfolio self-test --device cpu --json")
)
@click.option("--device", type=click.Choice([d.value for d in Device]), help="Where the test runs.")
@json_option
def self_test(device: str | None, as_json: bool) -> int:
    """Run a bundled 58 s clip through every part in use and check each result; exit code 3 if one fails."""
    try:
        outcome = facade.self_test(Device(device) if device else None, Progress(quiet=as_json))
    except AppError as exc:
        return refuse("self-test", exc, as_json)
    parts = [env.SelfTestPart(id=k, **v) for k, v in outcome["components"].items()]
    data = env.SelfTestData(
        passed=outcome["passed"], device=outcome["device"], elapsed_ms=outcome["elapsed_ms"],
        budget_ms=outcome["budget_ms"], components=parts,
    )  # fmt: skip
    failed = ", ".join(p.id for p in parts if p.status == "failed") or "the analysis stopped"
    problem = None if data.passed else AppError("E_SELFTEST_FAILED", failed)
    exit_code = 0 if data.passed else SETUP_FAILED
    if as_json:
        return emit("self-test", exit_code, data, [], problem)
    for p in parts:
        click.echo(f"{p.id:<28} {p.status:<12} {p.reason}")
    limits = [f"{p.id} is off because {p.reason}" for p in parts if p.status in ("unavailable", "disabled")]
    if not data.passed:
        click.echo(f"The self-test failed: {failed}. Run `fluentfolio doctor`.")
    elif limits:
        click.echo("Ready, with limits: " + "; ".join(limits) + ".")
    else:
        click.echo(
            f"Ready. The self-test took {round(data.elapsed_ms / 1000)} s on the {data.device.upper()}."
        )
    if data.elapsed_ms > data.budget_ms:
        budget = f"{data.budget_ms // 60000} min"
        click.echo(f"It took longer than the {budget} it should take on the {data.device.upper()}.")
    return exit_code


@click.group(epilog=examples("fluentfolio components list", "fluentfolio components list --json"))
def components() -> None:
    """The parts: speech models, speaker separation, the aligners and the measuring tools."""


@components.command(
    "list", epilog=examples("fluentfolio components list", "fluentfolio components list --json")
)
@json_option
def components_list(as_json: bool) -> int:
    """Every part with its status, download size and licence class."""
    rows = facade.components()
    if as_json:
        return emit("components list", 0, env.ComponentList(components=_rows(rows)), [])
    for r in rows:
        size = f"{r.size_bytes / GB:.2f} GB" if r.size_bytes else "-"
        click.echo(f"{r.id:<28} {r.status:<16} {'on ' if r.on else 'off'}  {size:>8}  {r.licence_class}"
                   + (f"  ({r.reason})" if r.reason else ""))  # fmt: skip
    return 0


def _outcomes(ids: list[str], errors: dict[str, AppError | None] | None = None) -> list[env.PartOutcome]:
    """Each part's status now, and the error that stopped it, if one did."""
    status, errors = {r.id: r.status for r in facade.components()}, errors or {}
    return [
        env.PartOutcome(id=i, status=status[i], code=e.code if (e := errors.get(i)) else None) for i in ids
    ]


@components.command("install", epilog=examples(
    "fluentfolio components install phone-model", "fluentfolio components install --all --yes --json",
))  # fmt: skip
@click.argument("ids", nargs=-1, metavar="ID...")
@click.option("--all", "every", is_flag=True, help="Every part the analyses use that is not ready yet.")
@yes_option
@json_option
def components_install(ids: tuple[str, ...], every: bool, yes: bool, as_json: bool) -> int:
    """Download parts, or make MFA's environment, with every file checked against its SHA-256."""
    if bool(ids) == every:
        raise click.UsageError("name the parts to install, or give --all")
    planned = facade.install_plan(list(ids), every)
    names = ", ".join(c.id for c in planned.parts)
    size = size_text(planned.download_bytes)
    if planned.download_bytes and not confirmed(f"Download {size} for {names}?", yes, f"{size} to download"):
        click.echo("Nothing was downloaded.", err=True)
        planned = facade.InstallPlan([], planned.download_bytes)
    total = len(planned.parts)
    seen: list[str] = []
    meter = DownloadMeter(planned.download_bytes)

    def on_part(c: catalogue.Component) -> None:
        seen.append(c.id)
        meter.end()
        meter.part = c.title
        extra = (
            " and its environment from conda-forge, which takes some minutes" if c.install == "env" else ""
        )
        click.echo(f"[{len(seen)}/{total}] {c.id}{extra}", err=True)

    def on_file(f: catalogue.ModelFile) -> None:
        meter.end()
        meter.next_file()
        click.echo(f"  {f.path} ({size_text(f.size)})", err=True)

    done = facade.install(planned, on_part, on_file, meter)
    meter.end()
    failed = [e for e in done.values() if e is not None]
    not_ready = [r.id for r in facade.in_use() if r.status != "ready"]
    exit_code = failed[0].entry.exit_code if failed else 0
    if as_json:
        data = env.Installed(download_bytes=planned.download_bytes, parts=_outcomes(list(done), done),
                             not_ready=_outcomes(not_ready))  # fmt: skip
        return emit("components install", exit_code, data, [], failed[0] if failed else None)
    for i, problem in done.items():
        click.echo(f"{i}: " + (f"{problem} Next step: {problem.entry.hint}" if problem else "ok"))
    if not_ready:
        click.echo(f"Still not ready: {', '.join(not_ready)}. See fluentfolio components list.")
    return exit_code


def _consent_ids() -> list[str]:
    return [c.id for c in catalogue.components().values() if c.requires_consent]


@components.command("consent", epilog=examples(
    "fluentfolio components consent opensmile", "fluentfolio components consent opensmile --json",
))  # fmt: skip
@click.argument("component_id", metavar="ID", type=click.Choice(_consent_ids()))
@json_option
def components_consent(component_id: str, as_json: bool) -> int:
    """Read a part's licence and accept it yourself by typing I ACCEPT; only a person at a terminal can."""
    row = next(r for r in facade.components() if r.id == component_id)
    recorded = row.consent is None
    if recorded:
        if not can_prompt():
            raise AppError(
                "E_OPENSMILE_CONSENT_MISSING", "only a person typing I ACCEPT at a terminal gives it"
            )
        click.echo(installer.OPENSMILE_TERMS, err=True)
        typed = click.prompt(f"Type {installer.ACCEPT} to accept, or press Enter to stop", default="",
                             show_default=False, err=True)  # fmt: skip
        if typed.strip() != installer.ACCEPT:
            raise AppError("E_OPENSMILE_CONSENT_MISSING", "not accepted, so nothing was recorded")
        record = facade.consent(component_id, "cli-tty")
    else:
        record = row.consent or {}
    package = catalogue.get(component_id).package
    needed = package is not None and util.find_spec(package) is None
    if as_json:
        data = env.ConsentData(component=component_id, consent=env.ConsentRecord(**record), recorded=recorded,
                               installer_needed=needed)  # fmt: skip
        return emit("components consent", 0, data, [])
    click.echo(
        f"Consent {'recorded' if recorded else 'was given before'}: {record['utc']} ({record['source']})."
    )
    if needed:
        click.echo(
            "One more step: close Fluentfolio and run the installer again (Repair); it adds this part."
        )
    return 0


@components.command("remove", epilog=examples(
    "fluentfolio components remove whisper-large-v3-turbo", "fluentfolio components remove mfa --yes --json",
))  # fmt: skip
@click.argument("ids", nargs=-1, required=True, metavar="ID...")
@yes_option
@json_option
def components_remove(ids: tuple[str, ...], yes: bool, as_json: bool) -> int:
    """Delete downloaded parts or MFA's environment; install them again at any time."""
    facade.check_removable(list(ids))
    removed = list(ids) if confirmed(f"Delete the files of {', '.join(ids)}?", yes, "files to delete") else []
    if removed:
        facade.remove(removed)
    if as_json:
        return emit("components remove", 0, env.Removed(parts=_outcomes(removed)), [])
    click.echo(f"Removed: {', '.join(removed)}." if removed else "Nothing was deleted.")
    return 0


@click.group(epilog=examples("fluentfolio token status", "fluentfolio token set"))
def token() -> None:
    """Your own Hugging Face token, which pyannote's speaker separation needs. It is never printed."""


def _token_data(found: TokenStatus) -> env.TokenData:
    return env.TokenData(present=found.present, valid=found.valid, pyannote_terms_ok=found.pyannote_terms_ok)


def _token_warnings(found: TokenStatus) -> list[env.Notice]:
    notices = [warning("W_TOKEN_FILE_PLACE", place)] if (place := facade.token_file_place()) else []
    if found.valid and found.pyannote_terms_ok is False:
        notices.append(warning("E_HF_TERMS_NOT_ACCEPTED"))  # the token works; the model page is the next step
    return notices


def _show_token(found: TokenStatus, notices: list[env.Notice]) -> None:
    def word(value: bool | None) -> str:
        return "not tested" if value is None else "yes" if value else "no"

    click.echo(f"Token saved: {word(found.present)}. Accepted by Hugging Face: {word(found.valid)}. "
               f"pyannote terms accepted: {word(found.pyannote_terms_ok)}.")  # fmt: skip
    for n in notices:
        click.echo(f"{n.code}: {n.message} Next step: {n.hint}", err=True)


@token.command("status", epilog=examples("fluentfolio token status", "fluentfolio token status --json"))
@json_option
def token_status(as_json: bool) -> int:
    """Whether a token is saved, works, and has the pyannote terms accepted; this asks huggingface.co."""
    found = facade.token_status()
    notices = _token_warnings(found)
    if as_json:
        return emit("token status", 0, _token_data(found), notices)
    _show_token(found, notices)
    return 0


@token.command("set", epilog=examples("fluentfolio token set", "fluentfolio token set --stdin --json"))
@click.option("--stdin", "from_stdin", is_flag=True,
              help="Read the token from standard input, one line, instead of a hidden prompt.")  # fmt: skip
@json_option
def token_set(from_stdin: bool, as_json: bool) -> int:
    """Save your own Hugging Face read token, after testing it; a token that works is never replaced.

    The token is for pyannote: accept its terms on its model page first. Accepting them shares your contact
    details with pyannote's makers.
    """
    if from_stdin:
        value = sys.stdin.readline() if sys.stdin else ""
    elif can_prompt():
        value = click.prompt("Hugging Face read token (hidden)", hide_input=True, err=True)
    else:
        raise AppError(
            "E_TOKEN_MISSING", "type it at the hidden prompt in a terminal, or give it with --stdin"
        )
    found, saved = facade.token_set(value)
    notices = _token_warnings(found)
    if as_json:
        return emit("token set", 0, env.TokenSaved(token=_token_data(found), saved=saved), notices)
    click.echo("Token saved." if saved else "A token that works was saved already, so nothing changed.")
    _show_token(found, notices)
    return 0


@token.command("clear", epilog=examples("fluentfolio token clear", "fluentfolio token clear --yes --json"))
@click.option("--yes", is_flag=True, help="Remove it without asking.")
@json_option
def token_clear(yes: bool, as_json: bool) -> int:
    """Remove the token, but only one that Fluentfolio saved; any other token stays."""
    agreed = confirmed("Remove the Hugging Face token that Fluentfolio saved?", yes, "a token to remove")
    removed = facade.token_clear() if agreed else False
    if as_json:
        return emit("token clear", 0, env.TokenCleared(removed=removed), [])
    if removed or not agreed:
        click.echo("Token removed." if removed else "Nothing was removed.")
    else:
        click.echo("Fluentfolio saved no token, so nothing was removed.")
    return 0


COMMANDS: list[Callable[..., Any]] = [doctor, self_test, components, token]
