"""`doctor` and `components list` in a real process: the exit code, the envelope and the report file."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.models import catalogue

Run = Callable[..., subprocess.CompletedProcess[bytes]]
Read = Callable[[subprocess.CompletedProcess[bytes]], Any]
STATUSES = {"ready", "needs-download", "needs-token", "needs-consent", "self-test-failed", "not-available"}


def fresh(tmp_path: Path) -> dict[str, str]:
    """An app folder with no models in it, and a token file of the test's own that does not exist."""
    app = tmp_path / "fresh-app"
    app.mkdir(parents=True)
    return {"FLUENTFOLIO_APP_ROOT": str(app), "HF_TOKEN_PATH": str(tmp_path / "hf" / "token"), "HF_TOKEN": ""}


def test_doctor_exits_3_when_no_analysis_can_run_and_says_why(
    tmp_path: Path, cli: Run, envelope: Read
) -> None:
    done = cli("doctor", "--json", env=fresh(tmp_path))
    found = envelope(done)
    assert done.returncode == 3, done.stderr
    data = found["data"]
    assert data["blocking"] is True and found["ok"] is False
    blocking = [c for c in data["checks"] if c["blocking"]]
    assert found["error"]["code"] == blocking[0]["code"] == "E_MODEL_MISSING"
    assert found["error"]["category"] == "setup" and "parakeet-tdt-0.6b-v2" in blocking[0]["fix"]
    assert data["data_dir"] == Path(os.environ["FLUENTFOLIO_HOME"]).absolute().as_posix()
    assert data["device"]["choice"] in ("cpu", "gpu") and data["device"]["reason"]
    assert data["executable"] and Path(data["executable"]).is_absolute() and data["report"] is None
    groups = {c["group"] for c in data["checks"]}
    expected = {
        "Install",
        "Disk",
        "Compute",
        "Media",
        "Analyzers",
        "Hugging Face",
        "Network",
        "Offline readiness",
    }
    assert groups == expected
    # Without --json: the same exit code, and each problem with its fix.
    plain = cli("doctor", env=fresh(tmp_path / "second"))
    assert plain.returncode == 3 and b"No analysis can run" in plain.stdout


def test_doctor_report_is_written_where_asked(tmp_path: Path, cli: Run, envelope: Read) -> None:
    target = tmp_path / "out" / "report.txt"
    found = envelope(cli("doctor", "--report", target, "--json", env=fresh(tmp_path)))
    assert found["data"]["report"] == target.absolute().as_posix()
    text = target.read_text(encoding="utf-8")
    assert text.startswith("Fluentfolio diagnostic report\n") and "## Doctor" in text
    assert str(tmp_path) not in text and tmp_path.as_posix() not in text


def test_components_list_gives_every_part_with_its_status(tmp_path: Path, cli: Run, envelope: Read) -> None:
    done = cli("components", "list", "--json", env=fresh(tmp_path))
    found = envelope(done)
    assert done.returncode == 0 and found["command"] == "components list"
    rows = {r["id"]: r for r in found["data"]["components"]}
    assert list(rows) == list(catalogue.offered()) and "canary-1b-v2" not in rows  # kept for eval/ only
    assert {r["status"] for r in rows.values()} <= STATUSES
    assert rows["parakeet-tdt-0.6b-v2"]["status"] == "needs-download"
    assert rows["pyannote-community-1"]["status"] == "needs-token"
    assert rows["opensmile"]["consent"] is None and rows["mfa"]["on"] is False
    assert rows["parakeet-tdt-0.6b-v2"]["size_bytes"] == 97 + 8998286 + 652184014 + 9384  # its four files


@pytest.mark.parametrize(
    ("args", "exit_code", "code"),
    [
        (["components", "consent", "opensmile"], 5, "E_OPENSMILE_CONSENT_MISSING"),
        (["components", "consent", "opensmile", "--yes"], 2, "E_USAGE"),
        (["components", "install", "opensmile", "--yes"], 5, "E_OPENSMILE_CONSENT_MISSING"),
        (["components", "install", "--all"], 5, "E_CONFIRM_NEEDED"),
        (["components", "install"], 2, "E_USAGE"),
        (["components", "install", "no-such-part", "--yes"], 4, "E_INPUT_OPTION"),
        (["components", "remove", "praat", "--yes"], 4, "E_INPUT_OPTION"),
        (["components", "remove", "phone-model"], 5, "E_CONFIRM_NEEDED"),
        (["token", "set"], 5, "E_TOKEN_MISSING"),
        (["token", "clear"], 5, "E_CONFIRM_NEEDED"),
    ],
    ids=lambda value: "-".join(value) if isinstance(value, list) else str(value),
)  # fmt: skip
def test_without_a_terminal_nothing_waits_and_yes_never_gives_consent(
    tmp_path: Path, cli: Run, envelope: Read, args: list[str], exit_code: int, code: str
) -> None:
    done = cli(*args, "--json", env=fresh(tmp_path), timeout=20)
    found = envelope(done)
    assert (done.returncode, found["error"]["code"]) == (exit_code, code), done.stderr
    assert not (Path(os.environ["FLUENTFOLIO_HOME"]) / "consent.json").exists()
    assert not (tmp_path / "hf").exists()  # no token written, by any command


def test_token_status_and_clear_without_a_token(tmp_path: Path, cli: Run, envelope: Read) -> None:
    env = fresh(tmp_path)
    status = envelope(cli("token", "status", "--json", env=env, timeout=20))
    assert status["data"] == {"present": False, "valid": None, "pyannote_terms_ok": None}
    cleared = envelope(cli("token", "clear", "--yes", "--json", env=env, timeout=20))
    assert cleared["ok"] is True and cleared["data"] == {"removed": False}


def test_a_component_already_in_the_app_installs_with_nothing_to_download(
    tmp_path: Path, cli: Run, envelope: Read
) -> None:
    found = envelope(cli("components", "install", "praat", "--json", env=fresh(tmp_path), timeout=20))
    assert found["ok"] is True and found["data"]["download_bytes"] == 0
    assert found["data"]["parts"] == [{"id": "praat", "status": "ready", "code": None}]
    assert "parakeet-tdt-0.6b-v2" in {p["id"] for p in found["data"]["not_ready"]}
