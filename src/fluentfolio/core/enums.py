"""The spellings of modes, layouts, job states, step statuses and devices, in every file the app writes."""

from enum import StrEnum


class Mode(StrEnum):
    SOLO = "solo"
    INTERVIEW = "interview"
    ROLE_SWAP = "role-swap"


class Layout(StrEnum):
    MIXED = "mixed"
    PER_TRACK = "per-track"
    TWO_FILES = "two-files"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_GAPS = "completed_with_gaps"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class Status(StrEnum):
    """The status of a step, tool or analyzer; `disabled` is the user's choice and never a gap."""

    OK = "ok"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    DISABLED = "disabled"


class Device(StrEnum):
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"


# The job loop tells each child where to run ("gpu" or "cpu"), so a child sent to the GPU can say it has none.
DEVICE_VAR = "FLUENTFOLIO_DEVICE"
