"""The evidence.json models; docs/schemas/evidence-v1.schema.json is generated from them.

A new measure adds an ID, not a field. Times are whole milliseconds inside the recording, and no field
holds a band: Fluentfolio measures, and only the learner's chosen AI gives an unofficial estimate.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"

Kind = Literal["measured", "estimated", "pointer", "context"]
Reliability = Literal["high", "medium", "low"]
Agreement = Literal["agree", "approximate", "unreliable", "too_little_speech", "single_tool"]
Confidence = Literal["low", "medium", "high"]
LicenceClass = Literal["permissive", "cc-by", "gpl", "nc"]
Status = Literal["ok", "partial", "skipped", "unavailable", "failed", "disabled"]
PartId = Literal[1, 2, 3, "unknown"]
Ms = int  # whole milliseconds; strict mode refuses 1.5


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Missing(Model):
    """Why a value is empty: the tool did not run, was switched off or had nothing to measure."""

    code: str
    text: str


class Source(Model):
    tool: str
    version: str
    weight: float = Field(ge=0, le=1)


class Measure(Model):
    """One value for one scope: "all", a Part or an answer ID; `value` and `missing` are never both set."""

    id: str
    name: str
    unit: str
    scope: str
    value: int | float | None  # counts stay whole numbers
    kind: Kind
    reliability: Reliability
    missing: Missing | None = None
    sources: list[Source]
    agreement: Agreement | None = None
    spread_st: float | None = None
    nc_input: bool = False
    licence_class: LicenceClass
    valid_in_mode: bool
    context_only: bool
    n: int | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _number_or_reason(self) -> Measure:
        if (self.value is None) == (self.missing is None):
            raise ValueError(f"{self.id}: a measure has either a value or a reason it is missing")
        return self


class PointerItem(Model):
    utterance_id: str
    first: int = Field(ge=0)  # token positions inside the utterance's words
    last: int = Field(ge=0)
    text: str


class Pointers(Model):
    """A list of candidates for the learner or the AI to check, never a count of mistakes."""

    id: str
    name: str
    items: list[PointerItem]
    shown: int = Field(ge=0)
    total: int = Field(ge=0)
    note: str | None = None  # a line about the whole list, such as what a check covered


class Answer(Model):
    id: str
    question_id: str | None
    part: PartId
    start_ms: Ms = Field(ge=0)
    end_ms: Ms = Field(ge=0)
    word_count: int = Field(ge=0)
    audio_ms: Ms | None = Field(ge=0)  # where the answer begins in the audio file; None: not in it


class Utterance(Model):
    id: str
    answer_id: str
    start_ms: Ms = Field(ge=0)
    end_ms: Ms = Field(ge=0)
    text: str


class PartSpan(Model):
    part: PartId
    start_ms: Ms = Field(ge=0)
    end_ms: Ms = Field(ge=0)
    source: Literal["auto", "user", "question_list"]
    confidence: Confidence


class Roles(Model):
    confidence: Confidence
    method: Literal["pyannote", "sherpa", "tracks", "single", "user"]
    swap_times_ms: list[Ms]


class Part2(Model):
    prep_excluded_ms: Ms = Field(ge=0)
    long_turn_ms: Ms = Field(ge=0)
    ended_by: Literal["stopped", "examiner_prompt", "cut_off", "unknown"]
    examiner_prompts_n: int = Field(ge=0)


class Details(Model):
    """Kept out of the text for an AI chat: no descriptor maps to these, so a chat would invent a meaning."""

    measures: list[Measure]
    pointers: list[Pointers]


class Candidate(Model):
    id: Literal["candidate", "person1", "person2"]
    label: Literal["the candidate", "Person 1", "Person 2"]
    speaker: str
    heard_questions_first: bool | None
    roles: Roles
    parts: list[PartSpan]
    speech_ms: Ms = Field(ge=0)
    part2: Part2 | None
    answers: list[Answer]
    utterances: list[Utterance]
    measures: list[Measure]
    pointers: list[Pointers]
    details: Details


class Speaker(Model):
    id: str
    speech_ms: Ms = Field(ge=0)
    source: Literal["pyannote", "sherpa", "tracks", "single"]


class ExaminerTurn(Model):
    """What was said to a candidate: by the examiner, or a question the learner read aloud in solo."""

    id: str
    start_ms: Ms = Field(ge=0)
    end_ms: Ms = Field(ge=0)
    kind: Literal["question", "prompt", "cue_card", "read_aloud", "other"]
    addressed_to: Literal["candidate", "person1", "person2"]  # in role-swap, each person asks the other
    text: str


class Difference(Model):
    utterance_id: str
    ours: str
    users: str


class Definition(Model):
    measure_id: str
    unit: str
    definition: str
    caveat: str
    how: str  # a few words for the text for an AI chat; empty for a measure kept out of it


class Tool(Model):
    """One step of the analysis as it ran; `ran` names what actually ran, so a fallback is never hidden."""

    id: str
    status: Status
    reason_code: str | None
    reason: str | None
    coverage: str | None
    version: str | None
    ran: str | None
    fallback_for: str | None
    device: Literal["cpu", "gpu"] | None
    wall_ms: Ms | None
    peak_ram_mb: int | None
    peak_vram_mb: int | None
    licence_class: LicenceClass | None


class WarningEntry(Model):
    code: str
    text: str


class Asr(Model):
    model: str
    revision: str
    transcript_style: Literal["verbatim-best-effort", "smoothed"]
    filler_capture: Literal["best_effort", "lower_bound", "unknown"]
    punctuation: Literal["machine"]
    confidence_threshold: (
        float | None
    )  # set per model by the recognition experiment; None marks no word unsure


class Audio(Model):
    duration_ms: Ms = Field(gt=0)
    quality_class: Literal["clean", "laptop", "meeting_codec", "phone", "noisy"]
    snr_db: float
    clipping_pct: float = Field(ge=0, le=100)
    music_warning: bool | None  # None: the music check did not run


class DeviceInfo(Model):
    requested: Literal["auto", "cpu", "gpu"]
    resolved: Literal["cpu", "gpu"]
    reason: str


class AudioExport(Model):
    produced: bool
    candidate_only: bool
    duration_ms: Ms = Field(ge=0)
    partner_voice_included: bool


class AudioCheck(Model):
    """One candidate's span, by its times only: the words of the span never leave the job folder."""

    candidate: Literal["candidate", "person1", "person2"]
    orig_start_ms: Ms = Field(ge=0)
    orig_end_ms: Ms = Field(ge=0)
    audio_start_ms: Ms = Field(ge=0)
    audio_end_ms: Ms = Field(ge=0)


class UserTranscript(Model):
    status: Literal["none", "ingested"]
    names_removed: bool
    word_match_pct: float | None
    differences_shown: int = Field(ge=0)
    differences_total: int = Field(ge=0)


class Manifest(Model):
    export_id: str = Field(pattern=r"^[0-9a-f]{6}$")
    job_id: str
    tool_version: str
    prompt_version: str | None
    created_utc: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
    python_assigns_band: Literal[False]
    stub: bool  # true only when a test backend stood in for the real models
    commit: str | None = Field(pattern=r"^[0-9a-f]{7,40}$")
    mode: Literal["solo", "interview", "role-swap"]
    layout: Literal["mixed", "per-track", "two-files"]
    swap_style: Literal["once", "per-question"] | None
    asr: Asr
    unsure_word_share_pct: float | None
    audio: Audio
    device: DeviceInfo
    audio_export: AudioExport | None
    audio_check: list[AudioCheck]  # empty with no audio file or no suitable span
    user_transcript: UserTranscript | None
    tools: list[Tool]
    warnings: list[WarningEntry]
    nc_component_used: bool
    overrides_applied: bool


class Evidence(Model):
    """The whole evidence package of one job."""

    schema_version: Literal["1.0.0"]
    manifest: Manifest
    speakers: list[Speaker]
    candidates: list[Candidate]
    examiner_turns: list[ExaminerTurn]
    differences: list[Difference]
    definitions: list[Definition]
    limitations: list[str]
    disclaimer: str

    @model_validator(mode="after")
    def _invariants(self) -> Evidence:
        duration = self.manifest.audio.duration_ms
        for where, start, end in spans(self.model_dump()):
            if not 0 <= start <= end <= duration:
                raise ValueError(f"{where}: {start}-{end} ms is not inside 0-{duration} ms")
        answers = [a.id for c in self.candidates for a in c.answers]
        if len(answers) != len(set(answers)):
            raise ValueError("an answer belongs to more than one candidate")
        for c in self.candidates:
            own = {a.id for a in c.answers}
            if stray := [u.id for u in c.utterances if u.answer_id not in own]:
                raise ValueError(f"{c.id}: utterances {stray} belong to another candidate's answers")
            if late := [t for t in c.roles.swap_times_ms if t > duration]:
                raise ValueError(f"{c.id}: swap times {late} are after the end of the recording")
        made = self.manifest.audio_export.duration_ms if self.manifest.audio_export else 0
        for check in self.manifest.audio_check:
            if not 0 <= check.audio_start_ms <= check.audio_end_ms <= made:
                raise ValueError(f"audio check of {check.candidate}: not inside the {made} ms audio file")
        return self


def spans(data: Any, where: str = "") -> Iterator[tuple[str, int, int]]:
    """Every (start, end) pair of the recording: `start_ms`/`end_ms`, and the audio check's `orig_*` pair."""
    if isinstance(data, dict):
        for prefix in ("", "orig_"):
            if f"{prefix}start_ms" in data and f"{prefix}end_ms" in data:
                yield where + prefix, data[f"{prefix}start_ms"], data[f"{prefix}end_ms"]
        for key, value in data.items():
            yield from spans(value, f"{where}/{key}")
    elif isinstance(data, list):
        for i, value in enumerate(data):
            yield from spans(value, f"{where}/{i}")


def schema_text() -> str:
    """The JSON Schema of evidence.json as committed in docs/schemas/evidence-v1.schema.json."""
    schema = Evidence.model_json_schema()
    schema["title"] = f"Fluentfolio evidence {SCHEMA_VERSION}"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"
