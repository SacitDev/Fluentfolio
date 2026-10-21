"""The diagnostic report: what Doctor found, the parts, the settings and the last jobs, with nothing private.

Every configured folder becomes a placeholder (`<MODELS>`, `<DATA>`, `<APP>`, `~`), and the log redaction
removes tokens and the user name. Of a job only job.json is read, without its display name: never the
recording, the transcripts, the removed names or the job's own logs, and never MFA's log folder.
"""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path

from fluentfolio import __version__
from fluentfolio.core import logs
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import source_commit
from fluentfolio.doctor.checks import ComponentRow, Report
from fluentfolio.jobs import store
from fluentfolio.jobs.loop import Roots

JOBS = 5  # the newest jobs whose records go in
LOG_LINES = 200  # the end of the app log


def _forms(path: Path) -> set[str]:
    """A folder as it can appear in text: with either slash, and with the backslashes JSON doubles."""
    text = str(path)
    return {text, path.as_posix(), json.dumps(text)[1:-1]}


def scrub(text: str, roots: Roots) -> str:
    """Folders to placeholders, then the log redaction (tokens, the home folder and the user name)."""
    pairs = [(form, name) for path, name in (
        (roots.models, "<MODELS>"), (roots.data, "<DATA>"), (roots.app, "<APP>"), (Path.home(), "~"),
    ) for form in _forms(path)]  # fmt: skip
    # The longest first, so a folder inside another keeps its own name.
    for form, name in sorted(pairs, key=lambda p: -len(p[0])):
        text = re.sub(re.escape(form), name, text, flags=re.IGNORECASE)
    return logs.redact(text)


def _job(job: Path) -> str:
    try:
        record = read_json(job / "job.json")
    except (OSError, ValueError):
        return f"{job.name}: job.json could not be read"
    given = record.get("input", {})
    for item in (given, given.get("second") or {}):
        item.pop("display_name", None)  # the recording's own file name
    return json.dumps(record, indent=1, ensure_ascii=True)


def _app_log(data: Path) -> list[str]:
    path = data / "logs" / "app.log"
    if not path.is_file():
        return ["(no app log yet)"]
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_LINES:]


def build(report: Report, rows: list[ComponentRow], roots: Roots) -> str:
    """The report's text, ready to read before it is shared."""
    lines = [
        "Fluentfolio diagnostic report",
        f"Made: {store.utc_now()}",
        f"Version: {__version__} (commit {source_commit() or 'unknown'})",
        f"Python: {platform.python_version()}; OS: {platform.platform(terse=True)}",
        f"Folders: app {roots.app}; data {roots.data}; models {roots.models}",
        f"Command: {report.executable or 'not found'}",
        f"Device: {report.device.get('choice')} ({report.device.get('reason')})",
        "",
        "## Doctor",
        *(f"[{c.level}] {c.group} / {c.id}: {c.message}" + (f" {c.code}." if c.code else "")
          for c in report.checks),
        "",
        "## Parts",
        *(f"{r.id}: {r.status}" + (f" ({r.reason})" if r.reason else "") + f"; {'on' if r.on else 'off'}"
          for r in rows),
        "",
        "## Settings (settings.toml)",
    ]  # fmt: skip
    settings = roots.data / "settings.toml"
    lines += settings.read_text(encoding="utf-8").splitlines() if settings.is_file() else ["(the defaults)"]
    lines += ["", f"## The newest jobs (job.json only, at most {JOBS})"]
    ids = sorted(store.list_ids(roots.data), reverse=True)[:JOBS]
    lines += [_job(store.job_dir(roots.data, i)) for i in ids] or ["(no jobs)"]
    lines += ["", f"## App log (the last {LOG_LINES} lines)", *_app_log(roots.data)]
    return scrub("\n".join(lines) + "\n", roots)
