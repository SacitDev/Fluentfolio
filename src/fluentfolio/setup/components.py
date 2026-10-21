"""Installing and removing the optional parts, and keeping the learner's own consent to openSMILE's licence.

Every download keeps its SHA-256 check; the MFA environment is made from the lock file shipped for this
system. Nothing here accepts a licence or saves a token for the learner: those stay the learner's own acts.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from importlib import util
from pathlib import Path

from fluentfolio.alignment import mfa
from fluentfolio.analyzers.opensmile import SOURCES, consented
from fluentfolio.core.errors import AppError, InputError, SetupError
from fluentfolio.core.files import read_json, write_json
from fluentfolio.jobs.loop import Roots
from fluentfolio.jobs.store import utc_now
from fluentfolio.models import catalogue, download
from fluentfolio.models.catalogue import SYSTEM, Component, ModelFile
from fluentfolio.setup import mfa_env

CONSENT_FILE = "consent.json"
ACCEPT = "I ACCEPT"
# The key clause of openSMILE's licence (the audEERING Research License, clauses A and B), in plain words.
OPENSMILE_TERMS = """\
openSMILE measures loudness and a third pitch track. Its licence, the audEERING Research License,
allows non-commercial use only: for example teaching, academic research and personal use at home.
It may not be used for commercial purposes, such as running a business, in or with a product
(free or paid), or for any other commercial gain. Some commercial research is allowed (clause B).
Any other commercial use, including the values it measures in a product, needs a commercial licence
from audEERING GmbH. Fluentfolio labels every export that holds an openSMILE value."""


def consent(data: Path, component_id: str, source: str) -> dict[str, str]:
    """Record the learner's own consent to a part's licence: when, and where it was given."""
    if source not in SOURCES or not catalogue.get(component_id).requires_consent:
        raise ValueError(f"no consent from {source!r} for {component_id!r}")
    path = data / CONSENT_FILE
    records = read_json(path) if path.is_file() else {}
    records[component_id] = {"utc": utc_now(), "source": source}
    data.mkdir(parents=True, exist_ok=True)
    write_json(path, records)
    return dict(records[component_id])


def with_needs(ids: list[str]) -> list[Component]:
    """The parts named and every part they need, each once, a needed part before the part that needs it."""
    every = catalogue.offered()  # a part kept only for the comparison in eval/ is not one a learner can name
    if unknown := [i for i in ids if i not in every]:
        raise InputError(
            "E_INPUT_OPTION", f"{len(unknown)} name(s) that no part has"
        )  # never the names typed
    order: list[str] = []

    def add(i: str) -> None:
        for n in every[i].needs:
            add(n)
        if i not in order:
            order.append(i)

    for i in ids:
        add(i)
    return [every[i] for i in order]


def missing_bytes(c: Component, roots: Roots) -> int:
    """The download size of the files a part still lacks, judged by size alone (the hash is checked later)."""
    if c.install not in ("download", "env"):
        return 0
    files = [(_target(c, f, roots), f) for f in c.files_for(SYSTEM)]
    return sum(f.size for path, f in files if not (path.is_file() and path.stat().st_size == f.size))


def _target(c: Component, f: ModelFile, roots: Roots) -> Path:
    return mfa_env.file_target(f.path, roots.app, roots.models) if c.install == "env" else (
        download.local_path(c, f, roots.models)
    )  # fmt: skip


def install(
    c: Component, roots: Roots, on_file: Callable[[ModelFile], None] | None = None,
    on_bytes: download.Progress | None = None,
) -> None:  # fmt: skip
    """Download or make one part; the caller holds work.lock and has the size confirmed."""
    if SYSTEM not in c.os:
        raise SetupError("E_COMPONENT_UNAVAILABLE", f"{c.id} on {SYSTEM}")
    if c.requires_consent and not consented(roots.data):
        raise AppError("E_OPENSMILE_CONSENT_MISSING")
    if c.install in ("venv", "venv-extra"):
        # A package changes the app's own Python, which only the installer may do: Windows keeps the
        # running app's modules locked.
        if c.package is None or util.find_spec(c.package) is None:
            raise SetupError("E_INSTALLER_NEEDED", c.id)
    elif c.install == "download":
        download.download(c, roots.models, on_file, on_bytes)
    elif c.install == "env":
        for f in c.files_for(SYSTEM):
            if on_file:
                on_file(f)
            target = mfa_env.file_target(f.path, roots.app, roots.models)
            download.fetch_file(c, f, target, on_bytes)
            if target == mfa_env.micromamba(roots.app) and sys.platform != "win32":
                target.chmod(0o755)
        if mfa_env.installed_version(roots.app) is None:
            mfa_env.create(roots.app)


def check_removable(c: Component) -> None:
    """Only downloaded parts and the MFA environment can be removed; the others belong to the app."""
    if c.install not in ("download", "env"):
        raise InputError("E_INPUT_OPTION", f"{c.id} is part of the app: switch it off in Settings instead")


def remove(c: Component, roots: Roots) -> None:
    """Delete a part's downloaded files or its environment; the caller holds work.lock."""
    check_removable(c)
    if c.install == "download":
        download.remove(c, roots.models)
    else:
        shutil.rmtree(mfa.env_dir(roots.app), ignore_errors=True)
        for f in c.files_for(SYSTEM):
            mfa_env.file_target(f.path, roots.app, roots.models).unlink(missing_ok=True)
