"""App folders for tests with the real models: the downloaded parts are reused through folder links.

A link is a junction on Windows, which needs no admin right, and a symlink elsewhere. Removing the folder
that holds a link removes the link only, never the files it points to.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from fluentfolio.core.files import write_json
from fluentfolio.models import catalogue, download
from fluentfolio.models.catalogue import SYSTEM


def link(target: Path, at: Path) -> None:
    at.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(at))
    else:
        at.symlink_to(target, target_is_directory=True)


def make(app: Path, real_app: Path, models: Path, backend: str, mfa_env: bool = True) -> Path:
    """An installed app folder; without `mfa_env` MFA's environment folder is there but empty."""
    write_json(app / "state.json", {"backend": backend, "models_dir": str(models)})
    link(real_app / "bin", app / "bin")
    if mfa_env:
        link(real_app / "envs" / "mfa", app / "envs" / "mfa")
    else:
        (app / "envs" / "mfa").mkdir(parents=True)
    return app


def damaged_models(real: Path, out: Path, component_id: str) -> Path:
    """A models folder of links to the real one, except a copy of one part with its first file cut in half."""
    c = catalogue.get(component_id)
    damaged = download.local_path(c, c.files_for(SYSTEM)[0], out)
    top, own = damaged.relative_to(out).parts[:2]
    for entry in real.iterdir():
        if entry.name != top:
            link(entry, out / entry.name)
            continue
        for child in entry.iterdir():
            if child.name == own:
                shutil.copytree(child, out / top / own)
            elif child.is_dir():
                link(child, out / top / child.name)
            else:
                shutil.copy2(child, out / top / child.name)
    whole = damaged.read_bytes()
    damaged.write_bytes(whole[: len(whole) // 2])
    return out
