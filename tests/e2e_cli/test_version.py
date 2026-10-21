"""The command starts and reports the version written in pyproject.toml."""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def expected_version() -> str:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]


def test_module_entry_prints_version(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "fluentfolio", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"fluentfolio, version {expected_version()}"


def test_console_script_prints_version(tmp_path: Path) -> None:
    exe = shutil.which("fluentfolio", path=str(Path(sys.executable).parent))
    assert exe is not None
    done = subprocess.run(
        [exe, "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert expected_version() in done.stdout


def test_unknown_command_is_a_usage_error(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "fluentfolio", "no-such-command"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert done.returncode == 2
    assert "no-such-command" in done.stderr
