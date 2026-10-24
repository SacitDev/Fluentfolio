"""The MFA environment: made by micromamba from the lock file shipped for this system, inside the app folder.

The lock names every package with its build and checksum, so each install gets the same files. micromamba
keeps its cache and its profile files in the app folder, never in the user's own profile.
"""

from __future__ import annotations

import re
import shutil
import sys
from importlib import resources
from pathlib import Path

from fluentfolio.alignment import mfa
from fluentfolio.core import paths
from fluentfolio.core.errors import SetupError
from fluentfolio.jobs import runner

IS_WINDOWS = sys.platform == "win32"
CREATE_S = 3600.0  # about 200 packages to fetch; micromamba prints no progress lines the runner reads
CHECK_S = 300.0
VERSION = re.compile(r"\d+\.\d+\.\d+")


def micromamba(app: Path) -> Path:
    return app / "bin" / ("micromamba.exe" if IS_WINDOWS else "micromamba")


def file_target(path: str, app: Path, models: Path) -> Path:
    """Where an MFA file goes: micromamba into the app folder's bin/, the models into models/mfa/."""
    return micromamba(app) if path.startswith("micromamba") else models / "mfa" / path


def _env(app: Path) -> dict[str, str]:
    cache = app / "cache" / "mamba"
    home = cache / "home"
    env = runner.child_env(app, paths.data_root(), paths.models_dir(), None, cpu=True)
    env |= {"MAMBA_ROOT_PREFIX": str(cache), "CONDA_PKGS_DIRS": str(cache / "pkgs"), "HOME": str(home),
            "USERPROFILE": str(home)}  # fmt: skip
    if IS_WINDOWS:
        env |= {"APPDATA": str(home / "AppData" / "Roaming"), "LOCALAPPDATA": str(home / "AppData" / "Local")}
    return env


def installed_version(app: Path) -> str | None:
    """What `python -m montreal_forced_aligner version` says in the env, or None when it does not run."""
    env = mfa.env_dir(app)
    if not mfa.env_current(app) or not mfa.python(env).is_file():
        return None
    variables = runner.child_env(app, paths.data_root(), paths.models_dir(), None, cpu=True,
                                 extra_path=mfa.env_path(env))  # fmt: skip
    root = app / "tmp" / "misc" / "mfa_root"
    variables["MFA_ROOT_DIR"] = str(root)
    done = runner.run(
        [str(mfa.python(env)), "-I", "-m", "montreal_forced_aligner", "version"], env=variables,
        timeout_s=CHECK_S, no_progress_s=CHECK_S,
    )  # fmt: skip
    shutil.rmtree(root, ignore_errors=True)
    # The runner keeps every line that is not a progress event, which here is the tool's whole answer.
    found = [line.strip() for line in done.stderr_tail if VERSION.fullmatch(line.strip())]
    return found[-1] if done.exit_code == 0 and done.stopped is None and found else None


def create(app: Path) -> str:
    """Make the MFA env from the shipped lock, replacing any older one; returns the MFA version it runs."""
    exe = micromamba(app)
    if not exe.is_file():
        raise SetupError("E_MODEL_MISSING", "mfa")
    env = mfa.env_dir(app)
    shutil.rmtree(env, ignore_errors=True)  # an update replaces the whole env
    variables = _env(app)
    with resources.as_file(resources.files("fluentfolio.data").joinpath(mfa.LOCK)) as lock:
        done = runner.run(
            [str(exe), "create", "--no-rc", "-y", "-p", str(env),
             "--override-channels", "--channel", "conda-forge", "--file", str(lock)],
            env=variables, timeout_s=CREATE_S, no_progress_s=CREATE_S,
        )  # fmt: skip
    if done.exit_code or done.stopped:
        shutil.rmtree(env, ignore_errors=True)
        raise SetupError("E_COMPONENT_UNAVAILABLE", f"mfa: micromamba ended with {hex(done.exit_code)}")
    runner.run([str(exe), "clean", "--all", "-y"], env=variables, timeout_s=CHECK_S, no_progress_s=CHECK_S)
    found = installed_version(app)
    if found is None:
        raise SetupError("E_COMPONENT_UNAVAILABLE", "mfa: the new env does not start")
    return found
