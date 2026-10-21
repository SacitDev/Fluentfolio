"""Where Fluentfolio keeps its code, models and data, and the rules for safe paths inside them."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path, PurePath
from typing import Any

from fluentfolio.core.files import read_json

IS_WINDOWS = sys.platform == "win32"
# Device names Windows reserves in every folder, whatever follows the first dot.
RESERVED = re.compile(r"(?i)con|prn|aux|nul|com[1-9]|lpt[1-9]")
MAX_APP_ROOT = 60
SOURCE_ROOT = Path(__file__).resolve().parents[3]  # the checkout that holds src/fluentfolio
COMMIT = re.compile(r"[0-9a-f]{40}")


def _local_folder(name: str) -> Path:
    if IS_WINDOWS:
        return Path(os.environ["LOCALAPPDATA"]) / name
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / name.lower()


def _installed_root() -> Path | None:
    # An installed venv lives at <AppRoot>/app/<version>/.venv.
    parents = Path(sys.prefix).resolve().parents
    if len(parents) > 2 and (parents[2] / "state.json").is_file():
        return parents[2]
    return None


def app_root() -> Path:
    """Code, tools and models; safe to delete and reinstall."""
    if os.environ.get("FLUENTFOLIO_APP_ROOT"):
        return Path(os.environ["FLUENTFOLIO_APP_ROOT"])
    return _installed_root() or _local_folder("Fluentfolio")


def install_state(root: Path) -> dict[str, Any]:
    """The installer's state.json, or an empty record before the first install."""
    path = root / "state.json"
    if not path.is_file():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def data_root() -> Path:
    """Recordings and results; never touched by a reinstall or an update."""
    if os.environ.get("FLUENTFOLIO_HOME"):
        return Path(os.environ["FLUENTFOLIO_HOME"])
    moved = install_state(app_root()).get("data_root")
    return Path(moved) if moved else _local_folder("Fluentfolio-data")


def models_dir() -> Path:
    root = app_root()
    moved = install_state(root).get("models_dir")
    return Path(moved) if moved else root / "models"


def job_tmp(root: Path, job_id: str) -> Path:
    """TMP, TEMP and TMPDIR of one job's children."""
    return root / "tmp" / job_id


def mfa_root(root: Path, job_id: str) -> Path:
    """MFA_ROOT_DIR of one job: ASCII, per job, deleted with the job."""
    return root / "tmp" / "mfa" / job_id


def hub_env(root: Path, models: Path) -> dict[str, str]:
    """The Hugging Face cache folders, set before huggingface_hub is imported, which reads them once."""
    return {
        "HF_HUB_CACHE": str(models / "hf"),
        "HF_XET_CACHE": str(root / "cache" / "hf-xet"),
        "HF_ASSETS_CACHE": str(root / "cache" / "hf-assets"),
    }


def path_problems(path: Path) -> list[str]:
    """Why a folder breaks the AppRoot rule on Windows (MFA, Kaldi and espeak fail on such paths)."""
    if not IS_WINDOWS:
        return []
    text = str(path)
    problems = []
    if not text.isascii():
        problems.append("letters outside A to Z")
    if " " in text:
        problems.append("a space")
    if len(text) >= MAX_APP_ROOT:
        problems.append(f"{len(text)} characters (at most {MAX_APP_ROOT - 1})")
    onedrive = [os.environ.get(v) for v in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")]
    if any(o and text.lower().startswith(o.lower()) for o in onedrive) or "onedrive" in text.lower():
        problems.append("inside OneDrive")
    if not _local_ntfs(path):
        problems.append("not a local NTFS disk")
    return problems


def _local_ntfs(path: Path) -> bool:
    if sys.platform != "win32":
        return True
    import ctypes

    drive = (path.drive or Path.cwd().drive) + "\\"
    kernel = ctypes.windll.kernel32
    if kernel.GetDriveTypeW(drive) != 3:  # DRIVE_FIXED
        return False
    name = ctypes.create_unicode_buffer(32)
    ok = kernel.GetVolumeInformationW(drive, None, 0, None, None, None, name, len(name))
    return bool(ok) and name.value == "NTFS"


def secure_umask() -> None:
    """On Linux, make every new file 0600 and every folder 0700."""
    if not IS_WINDOWS:
        os.umask(0o077)


def safe_join(base: Path, *parts: str) -> Path:
    """Join user-influenced names under base, refusing anything that could leave it or name a device."""
    for part in parts:
        pure = PurePath(part)
        if (
            not part
            or pure.is_absolute()
            or pure.drive
            or ":" in part
            or "\\" in part
            or "/" in part
            or part in (".", "..")
            or RESERVED.fullmatch(part.split(".")[0].rstrip(" "))
        ):
            raise ValueError("not a safe file name")
    joined = base.joinpath(*parts)
    if not joined.resolve().is_relative_to(base.resolve()):
        raise ValueError("not a safe file name")
    return joined


def source_commit(root: Path = SOURCE_ROOT) -> str | None:
    """The git commit the code runs from, read from .git without running git; None outside a checkout."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="ascii").strip()
        if not head.startswith("ref: "):
            return head if COMMIT.fullmatch(head) else None
        ref = head.removeprefix("ref: ")
        if (git / ref).is_file():
            found = (git / ref).read_text(encoding="ascii").strip()
        else:  # after git gc the branch lives only in packed-refs
            lines = (git / "packed-refs").read_text(encoding="ascii").splitlines()
            found = next((ln.split()[0] for ln in lines if ln.endswith(f" {ref}")), "")
    except (OSError, UnicodeDecodeError):
        return None
    return found if COMMIT.fullmatch(found) else None
