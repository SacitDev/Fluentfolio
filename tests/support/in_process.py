"""The command line run inside the test process, for tests whose fakes a child process would not see."""

from __future__ import annotations

import io
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.cli.main import main
from fluentfolio.core.paths import hub_env
from tests.support import cli_units


class Command:
    """Runs `fluentfolio <args> --json` through main(); returns the exit code, the envelope and all output."""

    def __init__(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        self.capsys, self.monkeypatch = capsys, monkeypatch
        monkeypatch.setattr(signal, "signal", lambda *args: None)  # the test process keeps its own handlers
        for name in hub_env(Path("."), Path(".")):
            monkeypatch.setenv(name, os.environ.get(name, ""))  # main() sets these; the test puts them back

    def __call__(
        self, *args: str, stdin: str = "", terminal: bool = False
    ) -> tuple[int, dict[str, Any], str]:
        self.monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
        # A person at a terminal, for the questions only such a person may answer.
        for module in ("commands", "setup_commands"):
            self.monkeypatch.setattr(f"fluentfolio.cli.{module}.can_prompt", lambda: terminal)
        exit_code = main([*args, "--json"])
        printed = self.capsys.readouterr()
        found = json.loads(printed.out)
        cli_units.check(found)
        return exit_code, found, printed.out + printed.err
