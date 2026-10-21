"""settings show and settings set in real processes: the file, its checks, a busy folder and a newer file."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.jobs.locks import work_lock

Run = Callable[..., subprocess.CompletedProcess[bytes]]
Read = Callable[[subprocess.CompletedProcess[bytes]], Any]

# The documented defaults as a reply gives them: each duration in whole milliseconds, worked out by hand.
DEFAULTS = {
    "schema_version": 1,
    "general": {"log_level": "info", "retention_ms": 0},
    "device": {"choice": "auto"},
    "asr": {"model": "parakeet-tdt-0.6b-v2"},
    "speakers": {"backend": "auto", "count": "auto"},
    "review": {"policy": "when-unsure"},
    "components": {},
    "analyzers": {"pronunciation": True},
    "limits": {
        "max_upload_mb": 2000,
        "min_ms": 10_000,
        "short_ms": 60_000,
        "confirm_ms": 3_600_000,
        "long_ms": 5_400_000,
        "max_ms": 10_800_000,
        "no_progress_ms": 600_000,
    },
    "export": {"audio": False},
    "ui": {"theme": "system"},
}


def test_settings_show_gives_the_defaults_until_a_set_changes_one(
    tmp_path: Path, cli: Run, envelope: Read
) -> None:
    home = tmp_path / "home"
    found = envelope(cli("settings", "show", "--json", home=home))
    assert found["data"]["settings"] == DEFAULTS
    assert (found["data"]["path"], found["data"]["read_only"]) == ((home / "settings.toml").as_posix(), False)
    assert not (home / "settings.toml").exists()  # showing writes nothing
    saved = envelope(cli("settings", "set", "export.audio", "true", "--json", home=home))
    assert saved["exit_code"] == 0 and saved["data"]["settings"] == {**DEFAULTS, "export": {"audio": True}}
    shown = cli("settings", "set", "limits.confirm_min", "90", home=home)
    assert shown.returncode == 0 and b"limits.confirm_min = 90" in shown.stdout
    again = envelope(cli("settings", "show", "--json", home=home))["data"]["settings"]
    limits = {**DEFAULTS["limits"], "confirm_ms": 5_400_000}  # type: ignore[dict-item]
    assert again == {**DEFAULTS, "export": {"audio": True}, "limits": limits}
    lines = cli("settings", "show", home=home).stdout.decode("utf-8").splitlines()
    assert "export.audio = true" in lines and "limits.confirm_min = 90" in lines
    assert "general.retention_days = 0" in lines and 'ui.theme = "system"' in lines


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("export.audio", "yes"),
        ("nosuch.key", "1"),
        ("components.no-such-part", "false"),  # not a component of the catalogue
        ("analyzers.fluency", "false"),  # needs no optional tool, so it cannot be switched off
        ("asr.model", "no-such-model"),
    ],
)
def test_settings_set_refuses_a_key_or_value_and_leaves_the_file_as_it_was(
    tmp_path: Path, cli: Run, envelope: Read, key: str, value: str
) -> None:
    home = tmp_path / "home"
    envelope(cli("settings", "set", "ui.theme", "dark", "--json", home=home))
    before = (home / "settings.toml").read_bytes()
    found = envelope(cli("settings", "set", key, value, "--json", home=home))
    assert (found["error"]["code"], found["exit_code"]) == ("E_SETTINGS_INVALID", 4)
    # A real setting is named; a name that is not one is never repeated, as it may be anything typed.
    assert (key in found["error"]["message"]) is (key in ("export.audio", "asr.model"))
    assert (home / "settings.toml").read_bytes() == before


def test_set_waits_for_a_job_and_never_rewrites_a_newer_file(
    tmp_path: Path, cli: Run, envelope: Read
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    with work_lock(home):  # a job runs
        found = envelope(cli("settings", "set", "export.audio", "true", "--json", home=home))
    assert (found["error"]["code"], found["exit_code"]) == ("E_BUSY", 6)
    assert not (home / "settings.toml").exists()
    newer = "schema_version = 99\n[export]\naudio = true\n"
    (home / "settings.toml").write_text(newer, encoding="utf-8")
    found = envelope(cli("settings", "set", "export.audio", "false", "--json", home=home))
    assert (found["error"]["code"], found["exit_code"]) == ("E_SETTINGS_READ_ONLY", 4)
    assert (home / "settings.toml").read_text(encoding="utf-8") == newer
    shown = envelope(cli("settings", "show", "--json", home=home))
    assert shown["data"]["read_only"] is True and shown["data"]["settings"]["export"] == {"audio": True}
    assert [w["code"] for w in shown["warnings"]] == ["W_SETTINGS_NEWER"]
