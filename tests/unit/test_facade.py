"""Moving the data or the models folder: copy, verify, switch, then delete; refused while a job runs."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from fluentfolio.app.facade import move_folder
from fluentfolio.core import paths
from fluentfolio.core.errors import AppError
from fluentfolio.jobs.locks import work_lock


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An app folder whose install record points the data folder at tmp_path/old-data."""
    app = tmp_path / "app"
    app.mkdir()
    state = {"backend": "cpu", "data_root": str(tmp_path / "old-data")}
    (app / "state.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(app))
    monkeypatch.delenv("FLUENTFOLIO_HOME")
    (tmp_path / "old-data" / "jobs" / "20261013-170509-abcd").mkdir(parents=True)
    (tmp_path / "old-data" / "jobs" / "20261013-170509-abcd" / "job.json").write_text("{}", encoding="utf-8")
    (tmp_path / "old-data" / "settings.toml").write_text("x = 1\n", encoding="utf-8")
    return app


def test_the_data_folder_moves_and_only_then_the_old_copy_goes(installed: Path, tmp_path: Path) -> None:
    new = tmp_path / "new data"
    assert move_folder("data", new) == new
    assert (new / "jobs" / "20261013-170509-abcd" / "job.json").is_file() and (
        new / "settings.toml"
    ).is_file()
    state = json.loads((installed / "state.json").read_text(encoding="utf-8"))
    assert state == {"backend": "cpu", "data_root": str(new)}  # other install facts are kept
    assert paths.data_root() == new and not (tmp_path / "old-data").exists()


def test_a_move_is_refused_while_a_job_runs(installed: Path, tmp_path: Path) -> None:
    with work_lock(tmp_path / "old-data"), pytest.raises(AppError) as caught:
        move_folder("data", tmp_path / "new")
    assert caught.value.code == "E_BUSY"
    assert not (tmp_path / "new").exists() and paths.data_root() == tmp_path / "old-data"


def test_a_data_folder_chosen_by_the_environment_is_not_moved(
    installed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "old-data"))
    with pytest.raises(AppError) as caught:
        move_folder("data", tmp_path / "new")
    assert (caught.value.code, caught.value.entry.exit_code) == ("E_HOME_SET_BY_ENV", 5)
    assert (tmp_path / "old-data" / "settings.toml").is_file() and not (tmp_path / "new").exists()


def test_a_folder_cannot_move_inside_itself(installed: Path, tmp_path: Path) -> None:
    with pytest.raises(AppError) as caught:
        move_folder("data", tmp_path / "old-data" / "inner")
    assert (caught.value.code, caught.value.detail) == ("E_MOVE_FAILED", "one folder is inside the other")


@pytest.mark.parametrize("inside", ["recordings", ""])  # a folder in the app folder, and that folder itself
def test_the_data_folder_cannot_move_into_the_app_folder(
    installed: Path, tmp_path: Path, inside: str
) -> None:
    # The uninstaller deletes the app folder whole, so recordings in it would go with the program.
    with pytest.raises(AppError) as caught:
        move_folder("data", installed / inside)
    detail = "the data folder would lie inside the app folder"
    assert (caught.value.code, caught.value.detail) == ("E_MOVE_FAILED", detail)
    assert paths.data_root() == tmp_path / "old-data" and (tmp_path / "old-data" / "settings.toml").is_file()
    beside = tmp_path / "app-data"  # a name that only starts like the app folder's
    assert move_folder("data", beside) == beside


def record(app: Path, **moved: str) -> None:
    """Point the install record at moved folders, as Settings does."""
    state = json.loads((app / "state.json").read_text(encoding="utf-8"))
    (app / "state.json").write_text(json.dumps(state | moved), encoding="utf-8")


@pytest.mark.parametrize("moved", ["data", "models"])
def test_a_folder_cannot_move_into_the_other_one(installed: Path, tmp_path: Path, moved: str) -> None:
    # A later move of the outer folder would delete the inner one with it.
    (tmp_path / "M").mkdir()
    record(installed, models_dir=str(tmp_path / "M"))
    other, other_kind = (tmp_path / "M", "models") if moved == "data" else (tmp_path / "old-data", "data")
    with pytest.raises(AppError) as caught:
        move_folder(moved, other / "inner")  # type: ignore[arg-type]
    detail = f"the {moved} folder would lie inside the {other_kind} folder"
    assert (caught.value.code, caught.value.detail) == ("E_MOVE_FAILED", detail)
    assert paths.data_root() == tmp_path / "old-data" and paths.models_dir() == tmp_path / "M"
    assert not (other / "inner").exists()


@pytest.mark.parametrize("held", ["data", "models", "app"])
def test_a_folder_that_holds_another_fluentfolio_folder_is_not_moved(
    installed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held: str
) -> None:
    # A move deletes the old folder with all it holds.
    outer = tmp_path / "M" if held == "data" else tmp_path / "old-data"
    moved = "models" if held == "data" else "data"
    if held == "data":  # recordings inside the models folder
        shutil.copytree(tmp_path / "old-data", outer / "recordings")
        record(installed, models_dir=str(outer), data_root=str(outer / "recordings"))
    elif held == "models":
        (outer / "models").mkdir()
        record(installed, models_dir=str(outer / "models"))
    else:  # an app folder installed into the data folder
        shutil.copytree(installed, outer / "app")
        monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(outer / "app"))
    before = sorted(p.relative_to(tmp_path) for p in outer.rglob("*"))
    with pytest.raises(AppError) as caught:
        move_folder(moved, tmp_path / "N")  # type: ignore[arg-type]
    detail = f"the {moved} folder holds the {held} folder, which the move would delete"
    assert (caught.value.code, caught.value.detail) == ("E_MOVE_FAILED", detail)
    assert sorted(p.relative_to(tmp_path) for p in outer.rglob("*")) == before
    assert not (tmp_path / "N").exists()


@pytest.mark.parametrize("given", ["relative", "a file"])
def test_a_target_that_is_not_the_full_path_of_a_folder_is_refused(
    installed: Path, tmp_path: Path, given: str
) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    target = Path("new-data") if given == "relative" else tmp_path / "notes.txt"
    with pytest.raises(AppError) as caught:
        move_folder("data", target)
    assert (caught.value.code, caught.value.detail) == ("E_MOVE_FAILED", "not the full path of a folder")
    assert paths.data_root() == tmp_path / "old-data" and (tmp_path / "old-data" / "settings.toml").is_file()
    assert not Path("new-data").exists()


def test_with_keep_old_the_old_copy_stays_and_the_server_files_are_not_copied(
    installed: Path, tmp_path: Path
) -> None:
    for name in ("server.json", "open.html", "app.lock"):  # what a running web server keeps there
        (tmp_path / "old-data" / name).write_text("{}", encoding="utf-8")
    new = tmp_path / "new"
    assert move_folder("data", new, keep_old=True) == new
    assert paths.data_root() == new and (tmp_path / "old-data" / "settings.toml").is_file()
    assert sorted(p.name for p in new.iterdir()) == ["jobs", "settings.toml"]


def test_the_models_folder_follows_the_path_rule(installed: Path, tmp_path: Path) -> None:
    (installed / "models" / "hf").mkdir(parents=True)
    (installed / "models" / "hf" / "model.onnx").write_bytes(b"\0" * 10)
    spaced = tmp_path / "my models"
    if sys.platform == "win32":
        with pytest.raises(AppError) as caught:
            move_folder("models", spaced)
        assert caught.value.code == "E_PATH_RULE" and "a space" in caught.value.detail.split(", ")
        assert (installed / "models" / "hf" / "model.onnx").is_file() and not spaced.exists()
    else:  # the rule protects Windows tools only
        assert move_folder("models", spaced) == spaced
        assert paths.models_dir() == spaced and (spaced / "hf" / "model.onnx").stat().st_size == 10
