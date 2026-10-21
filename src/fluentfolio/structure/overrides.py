"""The review form's corrections, read from the job's overrides.json when the structure step runs again."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fluentfolio.core.errors import InputError
from fluentfolio.core.files import read_json

FILE = "overrides.json"


class PartEdit(BaseModel):
    """Part 2 from start to end, and where Part 3 starts; an answer takes the Part it starts in."""

    model_config = ConfigDict(extra="forbid", strict=True)
    part2: list[int] | None = Field(None, min_length=2, max_length=2)  # start and end in ms
    part3_start_ms: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> PartEdit:
        if self.part2 and not 0 <= self.part2[0] < self.part2[1]:
            raise ValueError("Part 2 must start before it ends")
        if self.part2 and self.part3_start_ms is not None and self.part3_start_ms < self.part2[1]:
            raise ValueError("Part 3 cannot start before Part 2 ends")
        return self

    def label(self, start_ms: int) -> int:
        if self.part3_start_ms is not None and start_ms >= self.part3_start_ms:
            return 3
        if self.part2:
            return 1 if start_ms < self.part2[0] else 2 if start_ms < self.part2[1] else 3
        return 1


class Overrides(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    swap_people: bool = False  # the person found asking was answering, and the other way round
    swap_times_ms: list[int] | None = None  # role-swap: when the roles change, replacing the ones found
    other_speakers: list[str] = []  # voices that are not one of the two people
    one_voice: Literal["candidate"] | None = None  # the one voice found is the candidate
    parts: dict[Literal["candidate", "person1", "person2"], PartEdit] = {}


def load(job: Path) -> Overrides:
    """The corrections saved for this job; none when the review form was not used."""
    path = job / FILE
    if not path.is_file():
        return Overrides()
    try:
        return Overrides.model_validate(read_json(path))
    except ValidationError as exc:
        raise InputError("E_OVERRIDES_INVALID", f"{exc.error_count()} problems") from None
