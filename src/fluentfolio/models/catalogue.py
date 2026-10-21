"""The model catalogue (data/models.toml): every component, where its files come from, and their hashes."""

from __future__ import annotations

import sys
import tomllib
from functools import cache
from importlib import resources
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The only first hosts a catalogue URL may name; a download may be redirected, but only over https.
URL_HOSTS = ("huggingface.co", "github.com")
SHA256 = r"^[0-9a-f]{64}$"
REVISION = r"^[0-9a-f]{40}$"
SYSTEM = "windows" if sys.platform == "win32" else "linux"  # as the catalogue's os fields name it


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelFile(Frozen):
    path: str  # in the repository for Hugging Face; the local file name for a URL; in the package for venv
    sha256: str = Field(pattern=SHA256)
    size: int = Field(gt=0)
    url: str | None = None
    os: Literal["windows", "linux"] | None = None  # None: needed on every system
    unpacked_size: int | None = None  # an archive: its total size once unpacked, the budget for unpacking


class Component(Frozen):
    id: str
    title: str
    note: str
    kind: Literal["asr", "vad", "language", "diarizer", "aligner", "phones", "acoustic", "nlp"]
    licence: str
    licence_class: Literal["permissive", "cc-by", "gpl", "nc"]
    attribution: str
    host: Literal["huggingface.co", "github.com", "pypi", "conda-forge", "built-in"]
    repo: str | None = None
    revision: str | None = Field(default=None, pattern=REVISION)
    package: str | None = None  # install = "venv": the package that holds the files
    engine: Literal["onnx-asr", "faster-whisper"] | None = (
        None  # asr entries: the library that runs the model
    )
    engine_model: str | None = None  # the model's name in that library
    device: Literal["cpu", "cpu+gpu"]
    install: Literal["download", "venv", "venv-extra", "env", "built-in"]
    default_on: bool
    # The systems where the component did not pass its accuracy check: there it is off until the learner
    # switches it on, and called experimental wherever it runs.
    experimental_on: list[Literal["windows", "linux"]] = []
    gated: bool
    requires_consent: bool
    needs: list[str]
    stages: list[str]
    os: list[Literal["windows", "linux"]]
    transcript_style: Literal["verbatim-best-effort", "smoothed"] | None = None
    # False: kept only for the speech-model comparison in eval/, which needs its pinned files; a learner
    # never sees it, in any list, and cannot install it.
    offered: bool = True
    file: list[ModelFile]

    @property
    def name(self) -> str:
        """The title without its detail in brackets: "Praat", for a line that names several parts."""
        return self.title.split(" (")[0]

    def files_for(self, system: str) -> list[ModelFile]:
        """The files this system needs ("windows" or "linux")."""
        return [f for f in self.file if f.os in (None, system)]

    def on_by_default(self, system: str) -> bool:
        """On when the learner has not chosen: never where the component is experimental."""
        return self.default_on and system not in self.experimental_on


class Catalogue(Frozen):
    schema_: int = Field(alias="schema")
    component: list[Component]


@cache
def components() -> dict[str, Component]:
    """Every catalogue entry, by component ID."""
    text = resources.files("fluentfolio.data").joinpath("models.toml").read_text(encoding="utf-8")
    entries = Catalogue.model_validate(tomllib.loads(text)).component
    return {c.id: c for c in entries}


def offered() -> dict[str, Component]:
    """The entries a learner sees, installs and removes: all but those kept for the comparison in eval/."""
    return {k: c for k, c in components().items() if c.offered}


def get(component_id: str) -> Component:
    return components()[component_id]


def switched_on(chosen: dict[str, bool], component_id: str, system: str = SYSTEM) -> bool:
    """The learner's choice from a settings components table; with none, the default on this system."""
    choice = chosen.get(component_id)
    return get(component_id).on_by_default(system) if choice is None else choice
