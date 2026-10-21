"""How long an analysis takes and how much free memory its steps need, from each step's measured time and peak
memory for the recording's length, on the CPU or GPU."""

from __future__ import annotations

from typing import Any

from fluentfolio.core.enums import Device, Mode
from fluentfolio.jobs.loop import Stage
from fluentfolio.models import catalogue

# Seconds per step as (fixed, per minute of audio): least-squares lines through composed public-speech
# sessions of 3 to 120 minutes, run on one computer. Known limit: a slower computer takes longer; revisit
# when timings from a second computer exist.
CPU: dict[str, tuple[float, float]] = {
    "media": (0.46, 0.16),
    "vad": (0.63, 0.08),
    "language": (0.0, 0.91),
    "asr:parakeet-tdt-0.6b-v2": (4.86, 2.77),
    "asr:parakeet-tdt-0.6b-v3": (4.86, 5.18),  # timed at 60 minutes only, with v2's fixed part
    "asr:whisper-large-v3-turbo": (10.91, 15.29),  # timed at 4.45 and 18 minutes only
    "speakers:single": (0.0, 6.55),  # solo: the check for a second voice
    "speakers:tracks": (0.02, 0.0),
    "speakers:pyannote": (5.28, 15.54),
    "speakers:sherpa": (0.0, 7.25),
    "words": (0.02, 0.0),
    "structure": (0.04, 0.0),
    "candidate_audio": (0.37, 0.0),
    "ai_audio": (1.90, 0.03),
    "phones": (1.75, 1.93),  # with the pronunciation check on
    "alignment": (0.02, 0.0),  # MFA off: the step only writes the CTC aligner's times
    "alignment:mfa": (20.90, 0.36),
    "praat": (0.0, 0.14),
    "swiftf0": (0.32, 0.14),
    "opensmile": (0.0, 1.18),
    "nlp": (3.17, 0.02),
    "measures": (0.12, 0.01),
    "evidence": (0.05, 0.0),
    "export": (0.14, 0.0),
}
GPU = {"asr:whisper-large-v3-turbo": (1.15, 0.97), "speakers:pyannote": (8.67, 1.07)}
OFF = (0.4, 0.0)  # a step that is off still starts and writes why

# Each step's declared minimum of free memory in MB as (fixed, per minute of audio): the peak of its own
# process and of the programs it starts, measured on composed public-speech sessions of 18, 60 and 120
# minutes on one computer. The slope comes from 60 and 120 minutes; the fixed part covers every measured
# length. Steps inside the app's own process have none: that memory is in use when free memory is read.
# Known limit: a step measured at one length only borrows its slope, as noted; revisit with more runs.
RAM_CPU: dict[str, tuple[float, float]] = {
    "media": (191, 0.12),
    "vad": (401, 0.0),
    "language": (240, 0.02),
    "asr:parakeet-tdt-0.6b-v2": (1211, 0.10),
    "asr:parakeet-tdt-0.6b-v3": (1245, 0.10),  # 60 minutes only, with v2's slope
    "asr:whisper-large-v3-turbo": (1885, 0.0),  # 18 minutes only, flat as on the GPU
    "speakers:single": (172, 10.45),  # the same program as sherpa: one line through its solo run at 60
    "speakers:sherpa": (172, 10.45),  # minutes and its two-voice run at 18
    "speakers:pyannote": (2620, 8.78),  # 18 minutes only, with the slope it has on the GPU
    "candidate_audio": (48, 5.47),
    "ai_audio": (220, 1.82),
    "phones": (2289, 0.07),
    "alignment": (50, 0.0),  # too short to sample: the peak of a step that only starts and writes why
    "alignment:mfa": (462, 0.13),
    "praat": (152, 5.47),
    "swiftf0": (308, 3.32),
    "opensmile": (228, 17.12),
    "nlp": (631, 0.15),
}
RAM_GPU = {"asr:whisper-large-v3-turbo": (1749, 0.0), "speakers:pyannote": (1620, 8.78)}


def _running(
    stages: list[Stage], mode: str, asr: str, settings: dict[str, Any], opensmile_consented: bool
) -> list[tuple[Stage, str, bool]]:
    """Each step with its key and whether it runs or only starts and writes why."""
    off = off_steps(settings, opensmile_consented)
    mfa_on = catalogue.switched_on(settings.get("components", {}), "mfa")
    return [(stage, step_key(stage, mode, asr, mfa_on), stage.id not in off) for stage in stages]


def step_key(stage: Stage, mode: str, asr: str, mfa_on: bool) -> str:
    """The name a step's time is kept under: the model for recognition, the backend for speakers."""
    if stage.id == "asr":
        return f"asr:{asr}"
    if stage.id == "speakers":
        if mode == Mode.SOLO:
            return "speakers:single"
        return "speakers:tracks" if stage.inline else stage.child.replace("speakers-", "speakers:")
    if stage.id == "alignment" and mfa_on:
        return "alignment:mfa"
    return stage.id


def off_steps(settings: dict[str, Any], opensmile_consented: bool) -> set[str]:
    """The steps a job's settings switch off, by each step's own first check; openSMILE also needs consent."""
    chosen, analyzers = settings.get("components", {}), settings.get("analyzers", {})

    def on(component: str, *serves: str) -> bool:
        # A tool runs when it is on and at least one analyzer it serves is on too.
        return catalogue.switched_on(chosen, component) and (
            not serves or any(analyzers.get(a, True) for a in serves)
        )

    # The phone step gives the aligner's word times and the pronunciation check's fits; either keeps it on.
    timed, checked = on("ctc-aligner"), bool(analyzers.get("pronunciation", True))
    phones = on("phone-model") and on("espeak-ng") and (timed or checked)
    switches = {
        "language": on("whisper-tiny"), "phones": phones, "praat": on("praat", "pitch", "voice-quality"),
        "swiftf0": on("swiftf0", "pitch"),
        "opensmile": on("opensmile", "pitch", "loudness") and opensmile_consented,
        "nlp": on("spacy-en-core-web-sm", "grammar"),
        "ai_audio": bool(settings.get("export", {}).get("audio", False)),
    }  # fmt: skip
    return {step for step, running in switches.items() if not running}


def estimate_ms(
    stages: list[Stage], *, mode: str, asr: str, settings: dict[str, Any], opensmile_consented: bool,
    audio_ms: int, device: str,
) -> int:  # fmt: skip
    """The sum of each step's time for this length; the card's time only for steps that run on it."""
    minutes = audio_ms / 60000
    total = 0.0
    for stage, key, runs in _running(stages, mode, asr, settings, opensmile_consented):
        if not runs:
            base, per_min = OFF
        elif device == Device.GPU and stage.gpu and key in GPU:
            base, per_min = GPU[key]
        else:
            base, per_min = CPU[key]
        total += base + per_min * minutes
    return round(total * 1000)


def ram_needs_mb(
    stages: list[Stage], *, mode: str, asr: str, settings: dict[str, Any], opensmile_consented: bool,
    audio_ms: int, device: str,
) -> dict[str, int]:  # fmt: skip
    """Each running step's declared minimum of free memory for this length in MB; steps run one at a time."""
    minutes = audio_ms / 60000
    needs = {}
    for stage, key, runs in _running(stages, mode, asr, settings, opensmile_consented):
        table = RAM_GPU if device == Device.GPU and stage.gpu and key in RAM_GPU else RAM_CPU
        if runs and key in table:
            base, per_min = table[key]
            needs[stage.id] = round(base + per_min * minutes)
    return needs
