"""Folder resolution, safe joins, the app-folder path rule, owner-only folders and the command line start."""

import csv
import io
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from fluentfolio.cli.main import main
from fluentfolio.core import paths
from tests.support.installer import powershell


@pytest.mark.parametrize(
    "name",
    ["..\\..\\x.wav", "../x.wav", "..", ".", "", "CON.wav", "con", "nul.txt", "COM1", "lpt9.wav", "CON .wav",
     "a.wav:evil", "/etc/passwd", "C:\\x.wav", "C:x.wav", "sub/x.wav", "sub\\x.wav"],
)  # fmt: skip
def test_unsafe_names_never_join(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a safe file name"):
        paths.safe_join(tmp_path / "job", name)


@pytest.mark.parametrize("name", ["source.wav", "x y.wav", "Zoë.m4a", "console.wav", "a..b.wav"])
def test_ordinary_names_stay_inside_the_folder(name: str, tmp_path: Path) -> None:
    assert paths.safe_join(tmp_path / "job", name) == tmp_path / "job" / name


def test_each_job_has_its_own_temp_and_alignment_folders_in_the_app_folder(tmp_path: Path) -> None:
    job = "20261013-170509-abcd"
    assert paths.job_tmp(tmp_path, job) == tmp_path / "tmp" / job
    assert paths.mfa_root(tmp_path, job) == tmp_path / "tmp" / "mfa" / job


def test_folders_come_from_the_environment_then_the_install_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = tmp_path / "app"
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(app))
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "home"))
    assert (paths.app_root(), paths.data_root(), paths.models_dir()) == (
        app,
        tmp_path / "home",
        app / "models",
    )
    app.mkdir()
    state = {"data_root": str(tmp_path / "moved"), "models_dir": str(tmp_path / "m")}
    (app / "state.json").write_text(json.dumps(state), encoding="utf-8")
    assert paths.data_root() == tmp_path / "home"  # the environment wins
    monkeypatch.delenv("FLUENTFOLIO_HOME")
    assert (paths.data_root(), paths.models_dir()) == (tmp_path / "moved", tmp_path / "m")


def test_the_app_folder_path_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform != "win32":
        assert (
            paths.path_problems(Path("/home/zoë ü/a very long folder name that goes on and on and on")) == []
        )
        return
    root = Path(tmp_path.anchor)  # the local disk the tests run on
    synced = root / "Users" / "someone" / "OneDrive"
    monkeypatch.setenv("OneDrive", str(synced))
    assert paths.path_problems(root / "Fluentfolio") == []
    assert paths.path_problems(root / "Zoë Ü") == ["letters outside A to Z", "a space"]
    assert paths.path_problems(root / ("a" * 56)) == []  # 59 characters
    assert paths.path_problems(root / ("a" * 57)) == ["60 characters (at most 59)"]
    assert paths.path_problems(synced / "ff") == ["inside OneDrive"]


def test_a_new_folder_is_readable_by_its_owner_only(tmp_path: Path) -> None:
    if sys.platform == "win32":
        folder = tmp_path / "Fluentfolio"
        folder.mkdir()
        system32 = Path(os.environ["SYSTEMROOT"]) / "System32"
        who = [str(system32 / "whoami.exe"), "/user", "/fo", "csv", "/nh"]
        me = subprocess.run(who, capture_output=True, text=True, check=True).stdout
        sid = next(csv.reader(io.StringIO(me)))[1]

        def access_list() -> str:  # in SDDL form, which names accounts by SID
            saved = tmp_path / "acl.txt"
            subprocess.run([str(system32 / "icacls.exe"), str(folder), "/save", str(saved)], check=True,
                           capture_output=True)  # fmt: skip
            return saved.read_bytes().decode("utf-16-le").lstrip("\ufeff").splitlines()[1]

        assert "D:P" not in access_list()  # a new folder inherits its parent's entries
        done = powershell(tmp_path, ". $env:FF_SCRIPT; Set-OwnerOnly $env:FF_FOLDER", FF_FOLDER=str(folder))
        assert done.stdout.strip() == "True", done.stdout + done.stderr  # the installer's own step
        sddl = access_list()
        assert "D:P" in sddl  # protected: nothing is inherited from the drive any more
        aces = re.findall(r"\(A;([^;]*);([^;]*);;;([^)]+)\)", sddl)
        # SY is S-1-5-18 and BA is S-1-5-32-544; each has full control of the folder and all inside it.
        assert sorted(aces) == sorted([("OICI", "FA", sid), ("OICI", "FA", "SY"), ("OICI", "FA", "BA")])
        # A folder that already has its own entry for another account (here Everyone, S-1-1-0) keeps it after
        # the step, so the step says it failed and the installer stops.
        shared = tmp_path / "Shared"
        shared.mkdir()
        grant = [str(system32 / "icacls.exe"), str(shared), "/grant", "*S-1-1-0:(OI)(CI)M"]
        subprocess.run(grant, check=True, capture_output=True)
        done = powershell(tmp_path, ". $env:FF_SCRIPT; Set-OwnerOnly $env:FF_FOLDER", FF_FOLDER=str(shared))
        assert done.stdout.strip() == "False", done.stdout + done.stderr
    else:
        old = os.umask(0o022)
        try:
            paths.secure_umask()
            (tmp_path / "data").mkdir()
            (tmp_path / "data" / "job.json").write_text("{}", encoding="utf-8")
        finally:
            os.umask(old)
        assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / "data" / "job.json").stat().st_mode) == 0o600


def test_the_command_line_sets_the_model_caches_before_anything_loads(
    monkeypatch: pytest.MonkeyPatch, app_root: Path
) -> None:
    for name in ("HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["fluentfolio", "--version"])
    assert main() == 0
    assert os.environ["HF_HUB_CACHE"] == str(paths.models_dir() / "hf")
    assert os.environ["HF_XET_CACHE"] == str(app_root / "cache" / "hf-xet")
    assert os.environ["HF_ASSETS_CACHE"] == str(app_root / "cache" / "hf-assets")
