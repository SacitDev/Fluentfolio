"""The one JSON object a command prints with --json, the models of its data, and the schema made from them."""

from __future__ import annotations

import json
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fluentfolio import __version__
from fluentfolio.core.enums import JobState, Status
from fluentfolio.core.errors import AppError, catalogue
from fluentfolio.settings.model import Settings

SCHEMA = "fluentfolio.cli/1"
Person = Literal["candidate", "person1", "person2"]


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Problem(_Closed):
    """Why a command failed and what to do next; `hint` is a command to run where one helps."""

    code: str
    category: Literal["input", "setup", "needs_user", "busy", "stage", "internal"]
    message: str
    hint: str
    retryable: bool
    doc: str


class Notice(_Closed):
    code: str
    message: str
    hint: str


class StageAt(_Closed):
    id: str
    index: int  # counted from 1, in run order
    total: int


class ForAi(_Closed):
    candidate: Person
    path: str
    approx_tokens: int


class Outputs(_Closed):
    job_dir: str
    evidence: str | None
    zip: str | None
    for_ai: list[ForAi]


class Limit(_Closed):
    component: str
    code: str
    message: str


class StageRun(_Closed):
    id: str
    status: Status | None
    ran: str | None  # what actually ran, so a fallback is never hidden
    version: str | None
    device: str | None
    wall_ms: int | None
    peak_ram_mb: int | None
    error_code: str | None


class JobData(_Closed):
    """`analyze`, `status` and `resume`: a job's state, outputs and limits, never transcript text."""

    job_id: str
    state: JobState
    stage: StageAt | None
    created_utc: str
    updated_utc: str
    elapsed_ms: int
    estimate_ms: int | None
    mode: str | None
    asr: str | None
    device: str | None
    audio_duration_ms: int | None
    outputs: Outputs
    limits: list[Limit]
    stub: bool | None
    commit: str | None
    stages: list[StageRun]


class JobRow(_Closed):
    job_id: str
    created_utc: str
    state: JobState
    mode: str | None
    audio_duration_ms: int | None


class JobList(_Closed):
    jobs: list[JobRow]


class AudioFile(_Closed):
    candidate: Person
    path: str
    duration_ms: int


class Exported(_Closed):
    outputs: Outputs
    audio: list[AudioFile]  # the optional audio of each candidate's own speech; empty when none was made
    changed: bool  # false when every file was there already, so nothing was written again


class Deleted(_Closed):
    job_id: str
    deleted: bool  # false when the person said no


class SettingsData(_Closed):
    settings: dict[str, Any]  # settings.toml's sections, each duration in whole ms under a _ms name
    path: str  # settings.toml, which may not exist yet: then every value is the default
    read_only: bool  # written by a newer version: used as it is, never changed


# settings.toml names a duration in its own unit; a reply gives every duration in ms.
TO_MS = {"_s": 1000, "_min": 60_000, "_days": 86_400_000}


def settings_data(settings: Settings, path: str, read_only: bool) -> SettingsData:
    shown: dict[str, Any] = {}
    for section, values in settings.model_dump(mode="json").items():
        if not isinstance(values, dict):
            shown[section] = values
            continue
        shown[section] = {}
        for key, value in values.items():
            unit = next((u for u in TO_MS if key.endswith(u)), "")
            if unit:
                key, value = key.removesuffix(unit) + "_ms", value * TO_MS[unit]
            shown[section][key] = value
    return SettingsData(settings=shown, path=path, read_only=read_only)


class CancelData(_Closed):
    job_id: str
    state: JobState
    requested: bool  # false when the job was not running, so nothing was asked of it


class RuleResult(_Closed):
    rule: str
    name: str
    verdict: Literal["PASS", "FAIL", "N/A"]
    detail: str


class Checked(_Closed):
    job_id: str
    passed: bool
    counts: dict[str, int]
    results: list[RuleResult]


class PromptData(_Closed):
    kind: Literal["grading", "pronunciation"]
    prompt_version: str
    text: str


class GuideData(_Closed):
    """`guide agents`: the guide's text, and where `--write-skill` put the skill file, if asked."""

    text: str
    path: str | None


class VersionData(_Closed):
    version: str
    python: str
    os: str
    evidence_schema: str
    prompt_version: str


class Probe(_Closed):
    container: str
    codec: str
    sample_rate: int
    channels: int
    tracks: int
    lossy: bool
    duration_ms: int


class DryRun(_Closed):
    """`analyze --dry-run`: what the analysis would do, before any job exists."""

    probe: list[Probe]  # the file, then the second file of two-files
    audio_duration_ms: int  # the session: with two files, from the first file's start to the later end
    estimate_ms: int
    device: str
    will_run: list[str]  # the steps that run; the others are switched off
    missing: list[str]  # parts the run would use that are not ready, so its results have limits
    needs: list[Literal["partner_consent", "confirm_long"]]  # decisions a person makes first


class CheckRow(_Closed):
    group: str
    id: str
    level: Literal["ok", "warning", "problem"]
    message: str
    code: str | None
    fix: str
    blocking: bool


class DeviceChoice(_Closed):
    choice: str
    reason: str


class DoctorData(_Closed):
    blocking: bool
    checks: list[CheckRow]
    executable: str | None
    data_dir: str
    device: DeviceChoice
    report: str | None  # the diagnostic report written with --report


class ConsentRecord(_Closed):
    utc: str
    source: str


PartStatus = Literal[
    "ready", "needs-download", "needs-token", "needs-consent", "self-test-failed", "not-available"
]


class ComponentInfo(_Closed):
    id: str
    title: str
    kind: str
    status: PartStatus
    reason: str
    on: bool
    size_bytes: int
    licence_class: str
    consent: ConsentRecord | None


class ComponentList(_Closed):
    components: list[ComponentInfo]


class PartOutcome(_Closed):
    id: str
    status: PartStatus  # after the command
    code: str | None  # the error that stopped this part, if one did


class Installed(_Closed):
    """`components install`: each part tried, and the parts in use that are still not ready."""

    download_bytes: int
    parts: list[PartOutcome]
    not_ready: list[PartOutcome]


class Removed(_Closed):
    parts: list[PartOutcome]


class ConsentData(_Closed):
    component: str
    consent: ConsentRecord
    recorded: bool  # false when the consent had been given before
    installer_needed: bool  # the part's package comes with the next run of the installer


class TokenData(_Closed):
    """`token status`: whether a token is saved, works and has pyannote's terms accepted; never its value."""

    present: bool
    valid: bool | None
    pyannote_terms_ok: bool | None


class TokenSaved(_Closed):
    token: TokenData
    saved: bool  # false when a working token was there already, so nothing was written


class TokenCleared(_Closed):
    removed: bool  # false when Fluentfolio had saved no token, so nothing was removed


class SelfTestPart(_Closed):
    id: str
    status: Literal["ok", "failed", "unavailable", "disabled", "not-tested"]
    reason: str
    values: dict[str, float | int]  # what was measured on the clip, beside the clip's own value


class SelfTestData(_Closed):
    """`self-test`: the bundled clip through every part in use, each held to the clip's known answers."""

    passed: bool
    device: Literal["cpu", "gpu"]
    elapsed_ms: int
    budget_ms: int  # the time the self-test should take on this device at most
    components: list[SelfTestPart]


Data = (
    JobData | JobList | Exported | Deleted | CancelData | Checked | PromptData | GuideData | VersionData
    | DryRun | DoctorData | ComponentList | Installed | Removed | ConsentData | TokenData | TokenSaved
    | TokenCleared | SelfTestData | SettingsData
)  # fmt: skip


class Envelope(_Closed):
    model_config = ConfigDict(extra="forbid", validate_by_name=True)

    schema_: Literal["fluentfolio.cli/1"] = Field(alias="schema")
    command: str
    ok: bool
    exit_code: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 130]
    version: str
    data: Data | None
    warnings: list[Notice]
    error: Problem | None


def warning(code: str, detail: str = "") -> Notice:
    entry = catalogue()[code]
    return Notice(code=code, message=entry.message + (f" ({detail})" if detail else ""), hint=entry.hint)


def error(exc: AppError) -> Problem:
    e = exc.entry
    message = e.message + (f" ({exc.detail})" if exc.detail else "")
    return Problem(
        code=exc.code, category=e.category, message=message, hint=e.hint, retryable=e.retryable, doc=e.doc
    )


def emit(
    command: str, exit_code: int, data: Data | None, warnings: list[Notice], problem: AppError | None = None
) -> int:
    """Print the envelope as one ASCII line on stdout and return the exit code."""
    envelope = Envelope(
        schema=SCHEMA, command=command, ok=problem is None, exit_code=exit_code, version=__version__,
        data=data, warnings=warnings, error=error(problem) if problem else None,
    )  # fmt: skip
    sys.stdout.write(json.dumps(envelope.model_dump(mode="json", by_alias=True), ensure_ascii=True) + "\n")
    sys.stdout.flush()
    return exit_code


def schema_text() -> str:
    """The JSON Schema of the envelope as committed in docs/schemas/cli-v1.schema.json."""
    schema = Envelope.model_json_schema(by_alias=True)
    schema["title"] = f"Fluentfolio command line output ({SCHEMA})"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"
