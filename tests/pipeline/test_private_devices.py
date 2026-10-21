"""The private role-swap recordings on each device: on the processor every step runs there, and on the
graphics card the speaker step runs on the card; neither falls back.

The device is asked for, never Auto: Auto picks the processor in an app folder whose GPU test has not run.
Only numbers are recorded and compared, so a failure report never shows a word or a path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.app import facade
from fluentfolio.core.enums import Device, JobState, Status
from fluentfolio.core.files import read_json

pytestmark = [pytest.mark.slow, pytest.mark.private_sample]
Row = dict[str, Any]


def run(path: Path, device: Device) -> tuple[list[Row], dict[str, object]]:
    """Analyse a recording on the device asked for; its steps, and the numbers both tests record."""
    start = time.monotonic()
    job = facade.analyze(path, "role-swap", device=device, partner_consent="cli-flag").job
    record = read_json(job / "job.json")
    stages = record["stages"]
    found = {
        "wall_s": round(time.monotonic() - start, 1),
        "finished": record["state"] == JobState.COMPLETED,
        "resolved": record["device"]["resolved"] == device,
        "steps": len(stages),
        "fallbacks": sum(s["fallback"] is not None for s in stages),
        "steps_not_ok": sum(s["status"] not in (Status.OK, Status.DISABLED) for s in stages),
        "peak_ram_mb": max(s["peak_ram_mb"] or 0 for s in stages),
    }
    return stages, found


@pytest.mark.parametrize("index", [0, 1], ids=["sample1", "sample2"])
def test_a_private_recording_runs_every_step_on_the_processor(
    index: int,
    private_recordings: list[Path],
    private_summary: Callable[[str, dict[str, object]], None],
) -> None:
    stages, found = run(private_recordings[index], Device.CPU)
    found["steps_not_on_cpu"] = sum(s["device"] != Device.CPU for s in stages)
    private_summary("cpu_path", {f"sample{index + 1}": found})
    assert found["finished"] and found["resolved"]
    assert found["steps_not_on_cpu"] == 0 and found["fallbacks"] == 0
    assert found["steps_not_ok"] == 0  # "disabled" is a choice: the audio step and openSMILE without consent


@pytest.mark.gpu
@pytest.mark.parametrize("index", [0, 1], ids=["sample1", "sample2"])
def test_a_private_recording_runs_its_speaker_step_on_the_graphics_card(
    index: int,
    private_recordings: list[Path],
    private_summary: Callable[[str, dict[str, object]], None],
) -> None:
    stages, found = run(private_recordings[index], Device.GPU)
    speakers = next(s for s in stages if s["id"] == "speakers")
    found["speakers_on_gpu"] = speakers["device"] == Device.GPU
    found["peak_vram_mb"] = max(s["peak_vram_mb"] or 0 for s in stages)
    private_summary("gpu_path", {f"sample{index + 1}": found})
    assert found["finished"] and found["resolved"] and found["speakers_on_gpu"]
    assert found["fallbacks"] == 0  # no retry on the processor after a card error
    assert found["steps_not_ok"] == 0
