"""Doctor's checks and the components list: cheap tests of the install, the disks, the computer and each part.

No check imports a model library or uses the network: package lookups without import, file sizes, the
installer's state.json and the graphics driver's own tool only.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from importlib import util
from pathlib import Path
from typing import Any, Literal

import psutil

from fluentfolio import __version__
from fluentfolio.alignment import mfa
from fluentfolio.analyzers.opensmile import consented
from fluentfolio.asr.base import choices
from fluentfolio.core.enums import Device
from fluentfolio.core.errors import SetupError
from fluentfolio.core.files import read_json, size_text
from fluentfolio.core.paths import install_state
from fluentfolio.jobs.loop import Roots
from fluentfolio.models import catalogue, download
from fluentfolio.models.catalogue import SYSTEM, Component
from fluentfolio.settings.model import Settings
from fluentfolio.setup import device as gpu
from fluentfolio.setup import mfa_env
from fluentfolio.setup import token as hf_token

Level = Literal["ok", "warning", "problem"]
Status = Literal[
    "ready", "needs-download", "needs-token", "needs-consent", "self-test-failed", "not-available"
]
GB = 2**30
# Free space in the drive makers' GB (10**9 bytes), as the size of every download is shown; a 3-hour recording
# decodes to about 0.35 GB, and models need more.
DISK_PROBLEM, DISK_WARNING = 1 * 10**9, 5 * 10**9
RAM_SLOW_GB = 12  # an "8 GB" computer reports a little under 8 GB, a "16 GB" one a little under 16
ALWAYS_NEEDED = ("silero-vad",)  # with the chosen speech model, every analysis needs these
DECODER = "av"  # the audio decoder every analysis starts with
FIXES = {
    "needs-download": "Run `fluentfolio components install {id}`.",
    "needs-token": "Save your own Hugging Face token: run `fluentfolio token set`.",
    "needs-consent": "Read its licence, then run `fluentfolio components consent {id}` yourself.",
    "self-test-failed": "Run `fluentfolio self-test` again; if it still fails, run "
    "`fluentfolio components install {id}` to download it again.",
    "not-available": "Close Fluentfolio and run the installer again (Repair).",
}
CODES = {
    "needs-download": "E_MODEL_MISSING", "needs-token": "E_TOKEN_MISSING",
    "needs-consent": "E_OPENSMILE_CONSENT_MISSING", "self-test-failed": "E_SELFTEST_FAILED",
    "not-available": "E_COMPONENT_UNAVAILABLE",
}  # fmt: skip


@dataclass(frozen=True)
class Check:
    group: str  # Install, Disk, Compute, Media, Analyzers, Hugging Face, Network or Offline readiness
    id: str
    level: Level
    message: str
    code: str | None = None
    fix: str = ""
    blocking: bool = False  # no analysis can run until it is fixed


@dataclass(frozen=True)
class ComponentRow:
    id: str
    title: str
    kind: str
    status: Status
    reason: str
    on: bool  # switched on in Settings, or on by default
    size_bytes: int  # its download for this system
    licence_class: str
    consent: dict[str, str] | None  # the learner's own consent record, for a part that needs one


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    executable: str | None = None
    data_dir: str = ""
    device: dict[str, str] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return any(c.blocking for c in self.checks)


def token_present() -> bool:
    """Whether huggingface_hub finds a saved token; the value itself never leaves the library's call."""
    from huggingface_hub import get_token

    return get_token() is not None


def consent_record(data: Path, component_id: str) -> dict[str, str] | None:
    """When and where the learner gave a part's consent, or None without a valid record."""
    if not consented(data):
        return None
    record = read_json(data / "consent.json")[component_id]
    return {"utc": str(record["utc"]), "source": str(record["source"])}


def _files_status(c: Component, roots: Roots) -> tuple[Status, str] | None:
    """Whether every file of a downloaded part, or the MFA env and its files, is in place."""
    if c.install == "env":
        files = [(mfa_env.file_target(f.path, roots.app, roots.models), f.size) for f in c.files_for(SYSTEM)]
        whole = all(p.is_file() and p.stat().st_size == size for p, size in files)
        if not whole or not mfa.python(mfa.env_dir(roots.app)).is_file():
            return "needs-download", "its environment is not installed"
        if not mfa.env_current(roots.app):
            return "needs-download", "its environment needs an update"
        return None
    try:
        download.require(c, roots.models)
    except SetupError as exc:
        if exc.code == "E_HASH_MISMATCH":
            return "needs-download", "a file is damaged or cut short: download this model again"
        return "needs-download", "not downloaded yet"
    return None


def component_status(
    c: Component, roots: Roots, state: dict[str, Any], has_token: bool, found: dict[str, Status]
) -> tuple[Status, str]:
    """One part's install status and why; `found` holds the statuses of the parts already looked at."""
    if SYSTEM not in c.os:
        return "not-available", f"not available on {SYSTEM.capitalize()}"
    if c.requires_consent and not consented(roots.data):
        return "needs-consent", "its licence allows non-commercial use only, so it waits for your consent"
    if c.install in ("venv", "venv-extra"):
        if c.package is None or util.find_spec(c.package) is None:
            if c.install == "venv-extra":  # the licence is accepted by now: say the next step
                return "not-available", (
                    "only the installer adds its package, so close Fluentfolio and run the installer again"
                    " (Repair)"
                )
            return "not-available", "its package is not installed"
        try:
            download.require(c, roots.models)
        except SetupError:
            return "not-available", "a file of its package is missing or damaged"
    elif c.install == "built-in":
        missing = [n for n in c.needs if found.get(n) != "ready"]
        if missing:
            return "not-available", f"it needs {', '.join(catalogue.get(n).name for n in missing)}"
    else:
        if c.gated and not has_token:
            return "needs-token", "it needs your own Hugging Face token and the model's terms accepted"
        if problem := _files_status(c, roots):
            return problem
    last = state.get("selftest", {}).get("components", {}).get(c.id, {})
    if last.get("status") == "failed":
        return "self-test-failed", str(last.get("reason") or "it failed the last self-test")
    return "ready", ""


def components(settings: Settings, roots: Roots) -> list[ComponentRow]:
    """Every part a learner is offered, with its status, size, licence class and consent state."""
    state, has_token = install_state(roots.app), token_present()
    parts = catalogue.offered().values()
    found: dict[str, tuple[Status, str]] = {}
    # A built-in part is ready when the parts it needs are, so it is looked at after all the others.
    for c in sorted(parts, key=lambda c: c.install == "built-in"):
        known = {k: status for k, (status, _) in found.items()}
        found[c.id] = component_status(c, roots, state, has_token, known)
    rows = []
    for c in parts:
        status, reason = found[c.id]
        rows.append(ComponentRow(
            id=c.id, title=c.title, kind=c.kind, status=status, reason=reason,
            on=catalogue.switched_on(settings.components, c.id),
            size_bytes=sum(f.size for f in c.files_for(SYSTEM)), licence_class=c.licence_class,
            consent=consent_record(roots.data, c.id) if c.requires_consent else None,
        ))  # fmt: skip
    return rows


def in_use(rows: list[ComponentRow], asr: str) -> list[ComponentRow]:
    """The parts an analysis would use: those switched on, and of the speech models only the chosen one."""
    return [r for r in rows if (r.id == asr if r.kind == "asr" else r.on)]


def _free(folder: Path) -> int:
    existing = next(p for p in (folder, *folder.parents) if p.exists())
    return shutil.disk_usage(existing).free


def _install(state: dict[str, Any]) -> list[Check]:
    version = state.get("version")
    if not version:  # the self-test also writes state.json, so only the installer's version tells
        return [Check("Install", "installer", "warning",
                      "This copy was not set up by the installer (its record in the app folder is missing).",
                      "W_NOT_INSTALLED", "Use the installer for a normal install.")]  # fmt: skip
    if version != __version__:
        return [Check("Install", "installer", "warning",
                      f"The installer set up version {version}, but this is {__version__}.",
                      "W_NOT_INSTALLED", "Run the installer again (Repair).")]  # fmt: skip
    return [Check("Install", "installer", "ok", f"Installed by the installer, version {version}.")]


def _disks(roots: Roots) -> list[Check]:
    checks = []
    for name, folder in (("data", roots.data), ("models", roots.models)):
        free = _free(folder)
        text = f"{size_text(free)} free on the drive of the {name} folder."
        fix = "Free some space on that drive, or move the folder in Settings."
        if free < DISK_PROBLEM:
            checks.append(Check("Disk", name, "problem", text, "E_DISK_FULL", fix, blocking=True))
        elif free < DISK_WARNING:
            checks.append(Check("Disk", name, "warning", text, "W_DISK_LOW", fix))
        else:
            checks.append(Check("Disk", name, "ok", text))
    return checks


def _compute(ram_bytes: int, where: dict[str, str]) -> list[Check]:
    ram_gb = ram_bytes / GB
    checks = [Check("Compute", "cpu", "ok", f"{os.cpu_count()} processor threads.")]
    if ram_gb < RAM_SLOW_GB:
        checks.append(Check("Compute", "ram", "warning",
                            f"{ram_gb:.1f} GB of memory: analyses work, but slowly. 16 GB is recommended.",
                            "W_LOW_RAM", "Close other programs while an analysis runs."))  # fmt: skip
    else:
        checks.append(Check("Compute", "ram", "ok", f"{ram_gb:.1f} GB of memory."))
    if where["resolved"] == Device.GPU:
        checks.append(Check("Compute", "gpu", "ok", f"The GPU is used: {where['reason']}"))
    elif where["requested"] == Device.CPU:
        checks.append(Check("Compute", "gpu", "ok", f"The CPU runs every step: {where['reason']}"))
    else:
        checks.append(Check(
            "Compute", "gpu", "warning", f"The CPU runs every step: {where['reason']}", "W_GPU_NOT_USED",
            "A GPU is optional; without one, analyses take longer.",
        ))  # fmt: skip
    return checks


def _media() -> list[Check]:
    if util.find_spec(DECODER) is None:
        return [Check("Media", "decoder", "problem", "The audio decoder (PyAV) is not installed.",
                      "E_COMPONENT_UNAVAILABLE", FIXES["not-available"], blocking=True)]  # fmt: skip
    return [Check("Media", "decoder", "ok", "The audio decoder (PyAV) is installed.")]


def _parts(settings: Settings, rows: list[ComponentRow]) -> list[Check]:
    """Each part as Doctor shows it; the chosen speech model and the voice detector block an analysis."""
    if settings.asr.model not in choices():
        return [Check(
            "Analyzers", "asr", "problem",
            f"The speech model in Settings ({settings.asr.model}) is not one you can choose.",
            "E_COMPONENT_UNAVAILABLE", "Choose a speech model in Settings.", blocking=True,
        )]  # fmt: skip
    needed = {settings.asr.model, *ALWAYS_NEEDED}
    checks = []
    for row in rows:
        if row.kind == "asr" and row.id not in needed:
            continue  # the other speech models are choices, listed by `fluentfolio components list`
        if not row.on and row.id not in needed:
            unchosen = row.id not in settings.components and SYSTEM in catalogue.get(row.id).experimental_on
            why = (
                "off. It did not pass its accuracy check on this system, so it runs only once you install it"
                if unchosen else "switched off in Settings"
            )  # fmt: skip
            checks.append(Check("Analyzers", row.id, "ok", f"{row.title}: {why}."))
        elif row.status == "ready":
            checks.append(Check("Analyzers", row.id, "ok", f"{row.title}: ready."))
        else:
            blocking = row.id in needed
            checks.append(Check(
                "Analyzers", row.id, "problem" if blocking else "warning", f"{row.title}: {row.reason}.",
                CODES[row.status], FIXES[row.status].format(id=row.id), blocking=blocking,
            ))  # fmt: skip
    return checks


def _hugging_face(rows: list[ComponentRow], has_token: bool, roots: Roots) -> list[Check]:
    if has_token:
        saved = "A Hugging Face token is saved. `fluentfolio token status` tests it online."
        checks = [Check("Hugging Face", "token", "ok", saved)]
    else:
        gated = [r.title for r in rows if r.status == "needs-token"]
        checks = [Check("Hugging Face", "token", "warning",
                        f"No Hugging Face token, so {', '.join(gated) or 'no gated model'} cannot be "
                        "downloaded. Solo recordings and the built-in speaker separation still work.",
                        "E_TOKEN_MISSING", FIXES["needs-token"])]  # fmt: skip
    if place := hf_token.file_place(roots.app, roots.data):
        checks.append(Check(
            "Hugging Face", "token-file", "warning",
            f"The Hugging Face token file is inside {place}, where it may be copied or shared.",
            "W_TOKEN_FILE_PLACE", "Point HF_TOKEN_PATH somewhere private, or unset it.",
        ))  # fmt: skip
    return checks


def executable() -> str | None:
    """The absolute path of the fluentfolio command next to this Python, or the one on PATH."""
    command = Path(sys.executable).parent / ("fluentfolio.exe" if sys.platform == "win32" else "fluentfolio")
    found = command if command.is_file() else shutil.which("fluentfolio")
    return Path(found).absolute().as_posix() if found else None


def run(settings: Settings, roots: Roots, ram_bytes: int | None = None) -> Report:
    """Every check, grouped as Doctor shows them; a problem marked blocking means no analysis can run."""
    rows = components(settings, roots)
    has_token = token_present()
    where = gpu.resolve(settings.device.choice, gpu_steps=True)
    later = [
        catalogue.get(r.id).name for r in in_use(rows, settings.asr.model)
        if r.status in ("needs-download", "needs-token")
    ]  # fmt: skip
    checks = [
        *_install(install_state(roots.app)), *_disks(roots),
        *_compute(psutil.virtual_memory().total if ram_bytes is None else ram_bytes, where),
        *_media(), *_parts(settings, rows), *_hugging_face(rows, has_token, roots),
        Check("Network", "network", "ok",
              "This check uses no network. `fluentfolio token status` tests access to Hugging Face."),
        Check("Offline readiness", "offline", "warning" if later else "ok",
              f"Still to download: {', '.join(later)}." if later
              else "Everything switched on is on this computer, so analyses run offline.",
              "W_NOT_OFFLINE" if later else None,
              "Run `fluentfolio components install --all`." if later else ""),
    ]  # fmt: skip
    return Report(
        checks=checks, executable=executable(), data_dir=roots.data.absolute().as_posix(),
        device={"choice": where["resolved"], "reason": where["reason"]},
    )  # fmt: skip
