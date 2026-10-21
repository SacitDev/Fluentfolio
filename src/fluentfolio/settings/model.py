"""The settings model: every key of settings.toml with its default, allowed values, label and help text.

The Settings page is drawn from this model: a section's title and tab, and a field's title, description
and choice labels. A choice without a label is shown as its value with a capital letter.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.config import JsonDict

from fluentfolio.core.enums import Device

SCHEMA_VERSION = 1


def _labels(**names: str) -> JsonDict:
    """The words a choice is shown with, by its value; a value with a dash is written as a keyword."""
    return {"labels": {key.replace("_", "-"): name for key, name in names.items()}}


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tab: ClassVar[Literal["basic", "advanced"]] = "basic"


class General(Section):
    model_config = ConfigDict(title="Logs and storage")
    tab = "advanced"
    log_level: Literal["debug", "info", "warning"] = Field(
        default="info",
        title="Log detail",
        description="How much the app log records. Debug helps when you report a problem.",
    )
    retention_days: int = Field(
        default=0, ge=0, title="Keep analyses for (days)",
        description="0 keeps every analysis. Older ones are deleted when the next analysis starts.",
    )  # fmt: skip


class DeviceChoice(Section):
    model_config = ConfigDict(title="Device")
    choice: Device = Field(
        default=Device.AUTO, title="Run on",
        description="Auto uses the graphics card once its self-test has passed, and the processor otherwise.",
        json_schema_extra=_labels(auto="Auto", cpu="Processor (CPU)", gpu="Graphics card (GPU)"),
    )  # fmt: skip


class Asr(Section):
    model_config = ConfigDict(title="Speech recognition")
    model: str = Field(
        default="parakeet-tdt-0.6b-v2", title="Speech model",
        description="The model that writes down what was said. The list shows this version's models.",
    )  # fmt: skip


class Speakers(Section):
    model_config = ConfigDict(title="Speaker separation")
    backend: Literal["auto", "pyannote", "sherpa"] = Field(
        default="auto", title="Tool",
        description="pyannote needs a Hugging Face token; sherpa-onnx needs none. Auto tries pyannote first.",
        json_schema_extra=_labels(auto="Auto", pyannote="pyannote", sherpa="sherpa-onnx"),
    )  # fmt: skip
    count: Literal["auto", "exactly-2"] = Field(
        default="auto", title="Number of people",
        description="Exactly 2 tells the tool that two people speak.",
        json_schema_extra=_labels(auto="Find it", exactly_2="Exactly 2"),
    )  # fmt: skip


class Review(Section):
    model_config = ConfigDict(title="Review form")
    policy: Literal["when-unsure", "always", "never"] = Field(
        default="when-unsure", title="Show the review form",
        description="The form lets you correct who spoke and where each Part starts, then analyses again.",
        json_schema_extra=_labels(
            when_unsure="Only when unsure", always="After every analysis", never="Never"
        ),
    )  # fmt: skip


class Limits(Section):
    model_config = ConfigDict(title="Recording limits")
    tab = "advanced"
    max_upload_mb: int = Field(
        default=2000, ge=1, title="Largest upload (MB)", description="Larger files are refused."
    )
    min_s: int = Field(
        default=10, ge=1, title="Shortest recording (s)", description="Shorter recordings are refused."
    )
    short_s: int = Field(
        default=60,
        ge=1,
        title="Short recording (s)",
        description="A shorter recording is analysed with a warning.",
    )
    confirm_min: int = Field(
        default=60,
        ge=1,
        title="Ask before analysing over (min)",
        description="A longer recording needs a tick first.",
    )
    long_min: int = Field(
        default=90, ge=1, title="Offer a light run over (min)",
        description="A light run leaves out the three slowest optional parts.",
    )  # fmt: skip
    max_min: int = Field(
        default=180, ge=1, title="Longest recording (min)", description="Longer recordings are refused."
    )
    no_progress_min: int = Field(
        default=10, ge=1, title="Stop a stuck step after (min)",
        description="A step that reports no progress for this long is stopped.",
    )  # fmt: skip


class Export(Section):
    model_config = ConfigDict(title="Copy for AI")
    audio: bool = Field(
        default=False, title="Make the audio file for an AI chat",
        description="A short file of your answers for chats that accept audio, made with each analysis.",
    )  # fmt: skip


class Ui(Section):
    model_config = ConfigDict(title="Look")
    theme: Literal["system", "light", "dark"] = Field(
        default="system", title="Theme", description="System follows your computer's setting."
    )


class Settings(Section):
    """A component or analyzer missing from its table is on; the catalogue and the registry list them all."""

    schema_version: int = SCHEMA_VERSION
    general: General = General()
    device: DeviceChoice = DeviceChoice()
    asr: Asr = Asr()
    speakers: Speakers = Speakers()
    review: Review = Review()
    components: dict[str, bool] = {}
    # Listed so Settings shows its switch. On, though experimental (it did not pass its accuracy check):
    # its words to check are clues for the AI's Pronunciation estimate.
    analyzers: dict[str, bool] = {"pronunciation": True}
    limits: Limits = Limits()
    export: Export = Export()
    ui: Ui = Ui()

    def component_on(self, component_id: str) -> bool:
        return self.components.get(component_id, True)

    def analyzer_on(self, analyzer_id: str) -> bool:
        return self.analyzers.get(analyzer_id, True)
