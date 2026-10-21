"""The error catalogue and the small exception hierarchy every layer raises."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from importlib import resources

EXIT_CODES = {"internal": 1, "setup": 3, "input": 4, "needs_user": 5, "busy": 6, "stage": 7}


@dataclass(frozen=True)
class Entry:
    code: str
    severity: str
    title: str
    message: str
    hint: str
    doc: str
    category: str | None = None
    retryable: bool = False
    exit: int | None = None

    @property
    def exit_code(self) -> int:
        """The process exit code for this error; warnings never change it."""
        if self.category is None:
            return 0
        return self.exit if self.exit is not None else EXIT_CODES[self.category]


@cache
def catalogue() -> dict[str, Entry]:
    """Every row of errors.toml, by code."""
    text = resources.files("fluentfolio.core").joinpath("errors.toml").read_text(encoding="utf-8")
    return {code: Entry(code=code, **row) for code, row in tomllib.loads(text).items()}


class AppError(Exception):
    """An error a learner can act on: a catalogue code plus optional numbers or IDs, never their text."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.entry = catalogue()[code]
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {self.entry.message}" + (f" ({detail})" if detail else ""))


class InputError(AppError):
    """The recording, a file or an ID the user gave cannot be used."""


class SetupError(AppError):
    """A component, folder or disk is not ready."""


class ResourceError(AppError):
    """The computer is busy or out of memory."""


class StageError(AppError):
    """A step of the analysis failed."""


class CancelledError(AppError):
    """The user or the system stopped the job."""
