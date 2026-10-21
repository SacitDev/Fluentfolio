"""The job loop with a stand-in stage child: each outcome, the CPU retry, cancel, Ctrl+C, resume, clean-up."""

import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import pytest

from fluentfolio import __version__
from fluentfolio.core.enums import JobState, Status
from fluentfolio.core.errors import AppError
from fluentfolio.core.files import read_json
from fluentfolio.jobs import loop, store
from fluentfolio.jobs import stage as entry
from fluentfolio.jobs.runner import SWITCHES, Outcome
from fluentfolio.jobs.stage import gpu_code
from fluentfolio.settings.model import Settings
from fluentfolio.settings.store import snapshot

FAKE = Path(__file__).parents[1] / "support" / "fake_stage.py"
CUDA_FAULT = Path(__file__).parents[1] / "support" / "cuda_fault_stage.py"


def fake_argv(stage_id: str, job_id: str) -> list[str]:
    return [sys.executable, "-I", str(FAKE), stage_id, job_id]


@pytest.fixture
def roots(tmp_path: Path) -> loop.Roots:
    return loop.Roots(app=tmp_path / "app", data=tmp_path / "data", models=tmp_path / "app" / "models")


def new_job(
    roots: loop.Roots,
    *,
    device: str = "cpu",
    version: str = __version__,
    job_id: str = "",
    requested: str = "",
) -> str:
    now = store.utc_now()
    record = store.JobRecord(
        job_id=job_id or store.new_job_id(), created_utc=now, updated_utc=now, app_version=version,
        device={"requested": requested or device, "resolved": device}, settings=snapshot(Settings()),
    )  # fmt: skip
    store.create(roots.data, record)
    return record.job_id


def plan(*ids: str, gpu: tuple[str, ...] = (), optional: tuple[str, ...] = (), base_s: float = 60.0) -> list:
    return [
        loop.Stage(
            id=i,
            result=f"stages/{i}.json",
            required=i not in optional,
            gpu=i in gpu,
            base_s=base_s,
            per_min_s=0,
        )
        for i in ids
    ]


def run(roots: loop.Roots, job_id: str, stages: list, no_progress_s: float = 30.0, **kw) -> store.JobRecord:
    return loop.run_job(
        roots, job_id, stages, audio_min=1.0, no_progress_s=no_progress_s, argv_for=fake_argv, **kw
    )


def steps(record: store.JobRecord) -> dict[str, store.StageRecord]:
    return {s.id: s for s in record.stages}


def calls(roots: loop.Roots, job_id: str) -> list[str]:
    path = store.job_dir(roots.data, job_id) / "calls.txt"
    return path.read_text(encoding="utf-8").split() if path.exists() else []


def gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_all_steps_ok_completes_and_records_each_one(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    record = run(roots, job_id, plan("ok-a", "ok-b"))
    assert record.state is JobState.COMPLETED and record.error is None
    assert [(s.id, s.status, s.device, s.peak_ram_mb) for s in record.stages] == [
        ("ok-a", Status.OK, "cpu", 12),
        ("ok-b", Status.OK, "cpu", 12),
    ]
    assert all(s.started_utc and s.ended_utc and s.wall_ms is not None for s in record.stages)
    assert store.read(roots.data, job_id, lock_free=True) == record  # what the caller got is what is on disk
    assert not (roots.app / "tmp" / job_id).exists()  # a job removes its own temp folder


def test_an_unavailable_optional_step_gives_gaps_not_a_failure(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    record = run(roots, job_id, plan("ok-a", "unavailable-b", "ok-c", optional=("unavailable-b",)))
    assert record.state is JobState.COMPLETED_WITH_GAPS
    assert steps(record)["unavailable-b"].status is Status.UNAVAILABLE
    assert steps(record)["unavailable-b"].error_code == "E_COMPONENT_UNAVAILABLE"
    assert steps(record)["ok-c"].status is Status.OK


def test_a_failed_required_step_fails_the_job_with_the_childs_own_code(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    record = run(roots, job_id, plan("ok-a", "failed-b", "ok-c"))
    assert record.state is JobState.FAILED
    assert record.error == {"code": "E_JOB_FAILED", "stage": "failed-b"}
    failed = steps(record)["failed-b"]
    assert (failed.status, failed.error_code, failed.exit_code_hex) == (
        Status.FAILED,
        "E_CUDA_RUNTIME",
        "0x1",
    )
    assert calls(roots, job_id) == ["ok-a", "failed-b"]  # nothing after a failed required step


def test_a_refused_input_fails_the_job_with_its_own_code_which_a_retry_cannot_pass(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    record = run(roots, job_id, plan("ok-a", "silent-b", "ok-c"))
    assert record.state is JobState.FAILED
    assert record.error == {"code": "E_INPUT_SILENT", "stage": "silent-b"}
    required = run(roots, new_job(roots, job_id="20261013-120000-bbbb"), plan("unavailable-a"))
    assert required.error == {"code": "E_COMPONENT_UNAVAILABLE", "stage": "unavailable-a"}


def test_a_failed_optional_step_never_fails_the_job(roots: loop.Roots) -> None:
    record = run(roots, new_job(roots), plan("failed-a", "ok-b", optional=("failed-a",)))
    assert record.state is JobState.COMPLETED_WITH_GAPS
    assert steps(record)["ok-b"].status is Status.OK


def test_a_silent_step_is_killed_with_its_children_by_the_watchdog(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    started = time.monotonic()
    record = run(roots, job_id, plan("spawn-a"), no_progress_s=1.5)
    assert time.monotonic() - started < 10
    assert (record.state, steps(record)["spawn-a"].error_code) == (JobState.FAILED, "E_STAGE_TIMEOUT")
    grandchild = int((store.job_dir(roots.data, job_id) / "grandchild.pid").read_text(encoding="utf-8"))
    assert gone(grandchild)


def test_the_backstop_ends_a_step_that_keeps_reporting_progress(roots: loop.Roots) -> None:
    # 0.4 s base, times 3 on the CPU: the step is ended after about 1.2 s although it reports progress.
    record = run(roots, new_job(roots), plan("slow-a", base_s=0.4), no_progress_s=30)
    slow = steps(record)["slow-a"]
    assert slow.error_code == "E_STAGE_TIMEOUT"
    assert slow.wall_ms is not None and 1_200 <= slow.wall_ms < 10_000
    assert slow.exit_code_hex is None  # the runner ended it; that is not a crash code


def test_a_gpu_out_of_memory_runs_once_more_on_the_cpu_and_says_so(roots: loop.Roots) -> None:
    job_id = new_job(roots, device="gpu")
    record = run(roots, job_id, plan("gpuoom-a", "gpuoom-b", gpu=("gpuoom-a",)))
    retried, cpu_only = steps(record)["gpuoom-a"], steps(record)["gpuoom-b"]
    assert (retried.status, retried.device, retried.fallback) == (
        Status.OK,
        "cpu",
        {"to": "cpu", "after": "E_GPU_OOM"},
    )
    assert (cpu_only.status, cpu_only.device, cpu_only.fallback) == (
        Status.OK,
        "cpu",
        None,
    )  # GPU hidden from it
    assert record.state is JobState.COMPLETED_WITH_GAPS  # a fallback is never silent
    assert calls(roots, job_id) == ["gpuoom-a", "gpuoom-a", "gpuoom-b"]
    log = (store.job_dir(roots.data, job_id) / "logs" / "job.log").read_text(encoding="utf-8")
    assert "W_CPU_FALLBACK [gpuoom-a] E_GPU_OOM" in log


def test_the_job_names_the_cpu_only_when_every_gpu_step_had_to_move_there(roots: loop.Roots) -> None:
    moved = run(roots, new_job(roots, device="gpu"), plan("gpuoom-a", "ok-b", gpu=("gpuoom-a",)))
    reason = "The GPU could not be used (E_GPU_OOM), so the CPU was used."
    assert moved.device == {"requested": "gpu", "resolved": "cpu", "reason": reason}
    kept = run(roots, new_job(roots, device="gpu"), plan("gpuoom-a", "ok-b", gpu=("gpuoom-a", "ok-b")))
    assert kept.device == {"requested": "gpu", "resolved": "gpu"}  # one GPU step stayed on the GPU
    assert read_json(store.job_dir(roots.data, moved.job_id) / "job.json")["device"]["resolved"] == "cpu"


@pytest.mark.parametrize(
    ("gpu", "seen"), [(("gpuoom-a",), "cpu"), (("gpuoom-a", "ok-b"), "gpu"), (("gpuoom-a", "ok-d"), "gpu")]
)
def test_the_steps_after_the_last_gpu_step_see_the_device_that_ran(
    roots: loop.Roots, gpu: tuple[str, ...], seen: str
) -> None:
    """The evidence and the export copy the device from job.json, so it is settled before they run."""
    record = run(roots, new_job(roots, device="gpu"), plan("gpuoom-a", "ok-b", "job-c", "ok-d", gpu=gpu))
    result = read_json(store.job_dir(roots.data, record.job_id) / "stages" / "job-c.json")
    assert result["record"]["device"]["resolved"] == record.device["resolved"] == seen
    # A GPU step after the one that read job.json still runs on the GPU: the device is not settled early.
    assert steps(record)["ok-d"].device == ("gpu" if "ok-d" in gpu else "cpu")


class OutOfMemoryError(RuntimeError):
    """Named like torch's error for a full graphics card."""


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (RuntimeError("CUDA failed with error out of memory"), "E_GPU_OOM"),  # CTranslate2
        (OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), "E_GPU_OOM"),  # torch
        (RuntimeError("cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED"), "E_GPU_OOM"),
        (RuntimeError("CUDA error: an illegal memory access was encountered"), "E_CUDA_RUNTIME"),
        (RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"), "E_CUDA_RUNTIME"),
        (RuntimeError("DefaultCPUAllocator: not enough memory"), None),
        (ValueError("could not convert string to float"), None),
    ],
)
def test_errors_from_the_graphics_card_get_the_codes_that_retry_on_the_cpu(
    error: Exception, code: str
) -> None:
    assert gpu_code(error) == code
    assert code is None or code in loop.GPU_RETRY


def test_a_cuda_error_inside_a_real_stage_child_runs_once_more_on_the_cpu(roots: loop.Roots) -> None:
    job_id = new_job(roots, device="gpu")
    record = loop.run_job(
        roots, job_id, plan("fault", gpu=("fault",)), audio_min=1.0, no_progress_s=60,
        argv_for=lambda stage_id, job: [sys.executable, "-I", str(CUDA_FAULT), stage_id, job],
    )  # fmt: skip
    step = steps(record)["fault"]
    assert (step.status, step.device, step.fallback) == (
        Status.OK,
        "cpu",
        {"to": "cpu", "after": "E_GPU_OOM"},
    )
    assert record.state is JobState.COMPLETED_WITH_GAPS


def test_the_cpu_retry_happens_exactly_once(roots: loop.Roots) -> None:
    job_id = new_job(roots, device="gpu")
    record = run(roots, job_id, plan("gpufail-a", gpu=("gpufail-a",)))
    step = steps(record)["gpufail-a"]
    assert (step.status, step.error_code, step.fallback) == (
        Status.FAILED, "E_CUDA_RUNTIME", {"to": "cpu", "after": "E_CUDA_RUNTIME"},
    )  # fmt: skip
    assert calls(roots, job_id) == ["gpufail-a", "gpufail-a"]


def backend_step(child: str, fallback: str) -> loop.Stage:
    return loop.Stage("spk", "stages/spk.json", required=False, gpu=True, base_s=60, per_min_s=0,
                      child=child, fallback=fallback)  # fmt: skip


@pytest.mark.parametrize(
    ("child", "device", "expected_calls", "after"),
    [
        ("failed@spk", "cpu", ["failed@spk", "ok@spk"], "E_CUDA_RUNTIME"),
        ("crash@spk", "cpu", ["crash@spk", "ok@spk"], "E_STAGE_CRASHED"),
        # On the GPU the one CPU retry comes first; the fallback follows when that fails too.
        ("gpufail@spk", "gpu", ["gpufail@spk", "gpufail@spk", "ok@spk"], "E_CUDA_RUNTIME"),
    ],
)
def test_a_failed_backend_hands_over_to_its_fallback_and_says_so(
    roots: loop.Roots, child: str, device: str, expected_calls: list[str], after: str
) -> None:
    job_id = new_job(roots, device=device)
    record = run(roots, job_id, [backend_step(child, "ok@spk")])
    step = steps(record)["spk"]
    assert (step.status, step.error_code, step.device) == (Status.OK, None, "cpu")
    assert step.fallback == {"to": "ok@spk", "after": after}
    assert calls(roots, job_id) == expected_calls
    assert record.state is JobState.COMPLETED_WITH_GAPS  # the fallback is never silent
    log = (store.job_dir(roots.data, job_id) / "logs" / "job.log").read_text(encoding="utf-8")
    assert f"[spk] {child} ended with {after}; running ok@spk instead" in log


@pytest.mark.parametrize(
    ("child", "reason"),
    [
        ("nogpu@spk", "The GPU could not be used (E_GPU_UNAVAILABLE), so the CPU was used."),
        ("gpufail@spk", "The GPU could not be used (E_CUDA_RUNTIME), so the CPU was used."),
        # The first choice could not start for another reason, such as a missing model: the card is fine.
        ("unavailable@spk", "The spk step ran ok@spk on the CPU after E_COMPONENT_UNAVAILABLE."),
    ],
)
def test_the_device_reason_blames_the_gpu_only_for_a_gpu_fault(
    roots: loop.Roots, child: str, reason: str
) -> None:
    # Auto chose the card, and it was lost during the job.
    record = run(roots, new_job(roots, device="gpu", requested="auto"), [backend_step(child, "ok@spk")])
    assert record.device == {"requested": "auto", "resolved": "cpu", "reason": reason}


def test_a_chosen_gpu_that_a_step_cannot_use_fails_that_step_and_no_other_backend_runs(
    roots: loop.Roots,
) -> None:
    job_id = new_job(roots, device="gpu")
    record = run(roots, job_id, [backend_step("nogpu@spk", "ok@spk"), *plan("nogpu-a", gpu=("nogpu-a",))])
    spk = steps(record)["spk"]
    assert (spk.status, spk.error_code, spk.fallback) == (Status.FAILED, "E_GPU_UNAVAILABLE", None)
    assert calls(roots, job_id) == ["nogpu@spk", "nogpu-a"]  # neither the CPU nor the other backend ran
    assert (record.state, record.error) == (
        JobState.FAILED,
        {"code": "E_GPU_UNAVAILABLE", "stage": "nogpu-a"},
    )
    assert record.device == {"requested": "gpu", "resolved": "gpu"}


def test_a_backend_that_works_needs_no_fallback(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    record = run(roots, job_id, [backend_step("ok@spk", "failed@spk")])
    assert (steps(record)["spk"].fallback, record.state) == (None, JobState.COMPLETED)
    assert calls(roots, job_id) == ["ok@spk"]


def test_a_crash_without_a_result_gives_its_exit_code_in_hex(roots: loop.Roots) -> None:
    step = steps(run(roots, new_job(roots), plan("crash-a")))["crash-a"]
    assert (step.error_code, step.exit_code_hex) == ("E_STAGE_CRASHED", "0x3")


def test_an_uncaught_memory_error_is_reported_as_out_of_memory(roots: loop.Roots) -> None:
    assert steps(run(roots, new_job(roots), plan("memory-a")))["memory-a"].error_code == "E_RAM_OOM"


@pytest.mark.parametrize(
    ("outcome", "result", "expected"),
    [
        (Outcome(0), {"status": "partial", "code": None}, (Status.PARTIAL, None)),
        (Outcome(1), {"status": "failed", "code": "E_GPU_OOM"}, (Status.FAILED, "E_GPU_OOM")),
        (Outcome(2), {"status": "ok"}, (Status.FAILED, "E_STAGE_CRASHED")),
        (Outcome(1), {"status": "failed", "code": "E_NOT_IN_THE_CATALOGUE"}, (Status.FAILED, "E_INTERNAL")),
        (Outcome(0xC0000017), None, (Status.FAILED, "E_RAM_OOM")),  # Windows STATUS_NO_MEMORY
        (Outcome(-9), None, (Status.FAILED, "E_RAM_OOM")),  # killed by the Linux OOM killer
        (
            Outcome(1, stderr_tail=["OSError: [WinError 1455] paging file"]),
            None,
            (Status.FAILED, "E_RAM_OOM"),
        ),
        (Outcome(-9, stopped="cancel"), None, (Status.FAILED, "E_JOB_CANCELLED")),
        (Outcome(1, stopped="watchdog"), {"status": "ok"}, (Status.FAILED, "E_STAGE_TIMEOUT")),
        (Outcome(0), None, (Status.FAILED, "E_STAGE_CRASHED")),
    ],
)
def test_outcome_rules(outcome: Outcome, result: dict | None, expected: tuple) -> None:
    assert loop.classify(outcome, result) == expected


def test_a_cancel_request_ends_the_step_and_its_children_within_seconds(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    folder = store.job_dir(roots.data, job_id)
    done: list[store.JobRecord] = []
    worker = threading.Thread(target=lambda: done.append(run(roots, job_id, plan("spawn-a", "ok-b"))))
    worker.start()
    deadline = time.monotonic() + 20
    while not (folder / "grandchild.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    asked = time.monotonic()
    (folder / "cancel.request").write_text("", encoding="utf-8")
    worker.join(20)
    assert time.monotonic() - asked < 10
    record = done[0]
    assert (record.state, record.error) == (JobState.CANCELLED, {"code": "E_JOB_CANCELLED"})
    assert not (folder / "cancel.request").exists()
    assert gone(int((folder / "grandchild.pid").read_text(encoding="utf-8")))
    assert calls(roots, job_id) == ["spawn-a"]


def test_ctrl_c_marks_the_job_interrupted_and_kills_the_child(roots: loop.Roots) -> None:
    job_id = new_job(roots)

    def interrupt(stage_id: str, event: dict) -> None:
        if event.get("pct") == 10:
            raise KeyboardInterrupt

    with pytest.raises(AppError) as caught:
        run(roots, job_id, plan("spawn-a"), on_progress=interrupt)
    assert (caught.value.code, caught.value.entry.exit_code) == ("E_INTERRUPTED", 130)
    assert caught.value.detail == f"job {job_id}"  # so a script knows which job to run again
    record = store.read(roots.data, job_id, lock_free=False)
    assert (record.state, record.error) == (JobState.INTERRUPTED, {"code": "E_INTERRUPTED"})
    assert gone(int((store.job_dir(roots.data, job_id) / "grandchild.pid").read_text(encoding="utf-8")))


def test_run_again_starts_at_the_first_unfinished_step(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    stages = plan("ok-a", "ok-speakers", "ok-c", "ok-d")
    folder = store.job_dir(roots.data, job_id)
    for stage, status in zip(stages, ("ok", "failed", "ok", "ok"), strict=True):
        (folder / stage.result).write_text(f'{{"status": "{status}"}}', encoding="utf-8")
    record = run(roots, job_id, stages)
    assert calls(roots, job_id) == ["ok-speakers", "ok-c", "ok-d"]
    assert record.state is JobState.COMPLETED
    assert [s.id for s in record.stages] == ["ok-speakers", "ok-c", "ok-d"]


def test_a_completed_job_runs_nothing_when_run_again(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    run(roots, job_id, plan("ok-a", "ok-b"))
    again = run(roots, job_id, plan("ok-a", "ok-b"))
    assert again.state is JobState.COMPLETED
    assert calls(roots, job_id) == ["ok-a", "ok-b"]


def test_a_finished_job_runs_only_the_named_steps_again_and_keeps_the_rest_in_plan_order(
    roots: loop.Roots,
) -> None:
    job_id = new_job(roots)
    stages = plan("ok-a", "unavailable-b", "ok-c", "ok-d", optional=("unavailable-b",))
    first = run(roots, job_id, stages)
    again = run(roots, job_id, stages, only={"ok-a", "ok-d"})
    assert calls(roots, job_id) == ["ok-a", "unavailable-b", "ok-c", "ok-d", "ok-a", "ok-d"]
    assert [s.id for s in again.stages] == ["ok-a", "unavailable-b", "ok-c", "ok-d"]
    assert steps(again)["ok-c"] == steps(first)["ok-c"]  # a step not named keeps its record
    assert steps(again)["ok-d"].started_utc >= steps(first)["ok-d"].ended_utc
    assert again.state is JobState.COMPLETED_WITH_GAPS  # the gap of the step not named stays
    assert store.read(roots.data, job_id, lock_free=True) == again


def test_only_named_steps_need_a_finished_job(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    stages = plan("ok-a", "failed-b", "ok-c")
    assert run(roots, job_id, stages).state is JobState.FAILED
    with pytest.raises(AppError) as caught:
        run(roots, job_id, stages, only={"ok-c"})
    assert caught.value.code == "E_JOB_NOT_FINISHED"
    assert calls(roots, job_id) == ["ok-a", "failed-b"]


def test_a_job_made_by_another_version_is_not_continued(roots: loop.Roots) -> None:
    job_id = new_job(roots, version="0.0.1")
    with pytest.raises(AppError) as caught:
        run(roots, job_id, plan("ok-a"))
    assert (caught.value.code, caught.value.entry.exit_code) == ("E_RESUME_OTHER_VERSION", 5)
    assert calls(roots, job_id) == []


def test_inline_steps_write_their_result_and_an_unexpected_error_is_internal(roots: loop.Roots) -> None:
    def broken(folder: Path) -> dict:
        raise ZeroDivisionError

    job_id = new_job(roots)
    stages = [
        loop.Stage(id="words", result="stages/words.json", inline=lambda folder: {"status": "ok", "n": 3}),
        loop.Stage(id="structure", result="stages/structure.json", inline=broken),
    ]
    record = run(roots, job_id, stages)
    folder = store.job_dir(roots.data, job_id)
    assert (folder / "stages" / "words.json").read_text(encoding="utf-8").count('"n": 3') == 1
    assert (steps(record)["structure"].error_code, record.state) == ("E_INTERNAL", JobState.FAILED)
    assert "ZeroDivisionError" in (folder / "logs" / "job.log").read_text(encoding="utf-8")


WORD = "".join(["zor", "blat"])  # a transcript word, built at run time: no code line of a traceback holds it


def broken(*args: object) -> dict:
    raise KeyError(f"no lemma for {WORD}")  # a lookup by word carries the word in its message


def test_an_unexpected_error_in_an_inline_step_reaches_the_job_log_without_its_message(
    roots: loop.Roots,
) -> None:
    job_id = new_job(roots)
    run(roots, job_id, [loop.Stage(id="structure", result="stages/structure.json", inline=broken)])
    log = (store.job_dir(roots.data, job_id) / "logs" / "job.log").read_text(encoding="utf-8")
    assert "[structure] the step failed: KeyError" in log and ", in broken" in log
    assert WORD not in log


def test_a_child_that_crashes_prints_the_errors_type_and_code_lines_but_not_its_message(
    roots: loop.Roots, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = new_job(roots)
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(roots.data))
    monkeypatch.setattr(entry, "_end_with_parent", lambda: None)  # the test process is no child
    monkeypatch.setattr(entry, "secure_umask", lambda: None)
    monkeypatch.setattr(entry, "broken", broken, raising=False)
    monkeypatch.setitem(entry.STAGES, "nlp", ("fluentfolio.jobs.stage", "broken", "analyzers/nlp.json"))
    assert entry.main(["nlp", job_id]) == 1
    printed = capsys.readouterr().err
    assert printed.startswith("KeyError\n") and ", in broken" in printed and WORD not in printed
    assert read_json(store.job_dir(roots.data, job_id) / "analyzers" / "nlp.json")["code"] == "E_INTERNAL"


def test_the_decoded_recordings_length_and_format_enter_job_json(roots: loop.Roots) -> None:
    source = {"container": "wav", "codec": "pcm_s24le", "sample_rate": 48000, "channels": 2, "tracks": 1}
    media = {"status": "ok", "duration_ms": 61250, "source": source}
    job_id = new_job(roots)
    run(roots, job_id, [loop.Stage(id="media", result="stages/media.json", inline=lambda folder: media)])
    found = read_json(store.job_dir(roots.data, job_id) / "job.json")["input"]
    assert found == {"duration_ms": 61250, "codec": "pcm_s24le", "channels": 2, "sample_rate": 48000}
    failed = new_job(roots)
    run(roots, failed, [loop.Stage(id="media", result="stages/media.json", inline=lambda folder: 1 / 0)])
    assert read_json(store.job_dir(roots.data, failed) / "job.json")["input"] == {}


def test_child_output_reaches_the_job_log_only_after_redaction(roots: loop.Roots) -> None:
    job_id = new_job(roots)
    folder = store.job_dir(roots.data, job_id)
    token = "hf_" + "Q7x" * 12
    other = "zK4" * 9  # a secret in another form, carried by a URL parameter and a request header
    home = str(Path.home() / "Recordings")
    (folder / "canary.txt").write_text(
        f"{token} {home} https://example.org/?token={other}&a=1\nAuthorization: Bearer {other}",
        encoding="utf-8",
    )
    run(roots, job_id, plan("leak-a"))
    log = (folder / "logs" / "job.log").read_text(encoding="utf-8")
    assert "[leak-a] hf_<REDACTED>" in log and "?token=<REDACTED>&a=1" in log
    assert "[leak-a] Authorization: <REDACTED>" in log
    assert token not in log and other not in log and str(Path.home()) not in log


def test_a_child_gets_only_the_built_environment(roots: loop.Roots, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_TOKEN", "PYTHONPATH", "HF_HOME", "PYTHONSTARTUP", "HF_DEBUG"):
        monkeypatch.setenv(name, "planted")
    job_id = new_job(roots)
    run(roots, job_id, plan("env-a"))
    env = read_json(store.job_dir(roots.data, job_id) / "stages" / "env-a.json")["env"]
    assert not {"HF_TOKEN", "PYTHONPATH", "HF_HOME", "PYTHONSTARTUP", "HF_DEBUG"} & set(env)
    assert {name: env[name] for name in SWITCHES} == SWITCHES
    assert env["TMP"] == env["TEMP"] == env["TMPDIR"] == str(roots.app / "tmp" / job_id)
    assert "planted" not in env.values()


def test_housekeeping_marks_dead_jobs_and_never_touches_a_live_process_named_in_a_file(
    roots: loop.Roots,
) -> None:
    foreign = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"])
    try:
        job_id = new_job(roots)
        record = store.read(roots.data, job_id, lock_free=False)
        store.write(roots.data, record)  # state running, and nobody holds work.lock
        (roots.data / "server.json").write_text(f'{{"pid": {foreign.pid}}}', encoding="utf-8")
        (store.job_dir(roots.data, job_id) / "child.pid").write_text(str(foreign.pid), encoding="utf-8")
        loop.housekeeping(roots, retention_days=0)
        assert store.read(roots.data, job_id, lock_free=False).state is JobState.INTERRUPTED
        assert foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait(10)


@dataclass(frozen=True)
class Program:
    argv: list[str]
    path: list[Path]
    env: dict[str, str]
    cwd: Path
    scratch: Path


def outside(code: str, exit_code: int) -> tuple[Callable[..., Any], Callable[..., dict[str, Any]], list]:
    """A prepare step that writes an input file for a program running `code`, and a finish that records."""
    seen: list = []

    def prepare(job: Path, app: Path, models: Path) -> Program:
        scratch = app / "tmp" / "program"
        scratch.mkdir(parents=True)
        (scratch / "in.txt").write_text("input", encoding="utf-8")
        argv = [sys.executable, "-I", "-c", f"{code}; import sys; sys.exit({exit_code})"]
        return Program(argv, [], {"PROGRAM_SETTING": "set"}, scratch, scratch)

    def finish(job: Path, app: Path, models: Path, ran: dict[str, Any]) -> dict[str, Any]:
        out = app / "tmp" / "program" / "out.txt"
        seen.append((ran, out.read_text(encoding="utf-8") if out.exists() else None))
        return {"status": "ok"}

    return prepare, finish, seen


def test_an_outside_program_runs_between_its_prepare_and_finish_steps(roots: loop.Roots) -> None:
    code = (
        "import os; open('out.txt', 'w').write(open('in.txt').read() + ' ' + os.environ['PROGRAM_SETTING'])"
    )
    prepare, finish, seen = outside(code, 0)
    job_id = new_job(roots)
    record = run(roots, job_id, [loop.Stage("tool", "stages/tool.json", prepare=prepare, finish=finish)])
    ((ran, output),) = seen
    assert (ran["status"], output) == ("ok", "input set")  # it ran in its folder, with its own setting
    assert steps(record)["tool"].status is Status.OK and record.state is JobState.COMPLETED
    assert not (roots.app / "tmp" / "program").exists()  # what it left behind is removed after finish


def test_an_outside_program_that_fails_reaches_finish_with_its_code(roots: loop.Roots) -> None:
    prepare, finish, seen = outside("pass", 3)
    run(roots, new_job(roots), [loop.Stage("tool", "stages/tool.json", prepare=prepare, finish=finish)])
    ((ran, _),) = seen
    assert (ran["status"], ran["code"], ran["exit_code_hex"]) == ("failed", "E_STAGE_CRASHED", "0x3")


def test_an_outside_program_that_runs_too_long_is_stopped_at_its_backstop(roots: loop.Roots) -> None:
    prepare, finish, seen = outside("import time; time.sleep(120)", 0)
    stage = loop.Stage("tool", "stages/tool.json", base_s=0.1, per_min_s=0, prepare=prepare, finish=finish)
    started = time.monotonic()
    run(roots, new_job(roots), [stage])
    ((ran, _),) = seen
    assert (ran["status"], ran["code"]) == ("failed", "E_STAGE_TIMEOUT")
    assert time.monotonic() - started < 30  # three times the 0.1 s backstop on the CPU, far from the 120 s


def test_an_outside_program_that_prepare_rules_out_never_starts(roots: loop.Roots) -> None:
    seen: list = []
    stage = loop.Stage(
        "tool", "stages/tool.json", prepare=lambda job, app, models: {"status": "disabled"},
        finish=lambda job, app, models, ran: seen.append(ran) or {"status": "disabled"},
    )  # fmt: skip
    record = run(roots, new_job(roots), [stage])
    assert seen == [{"status": "disabled"}] and steps(record)["tool"].status is Status.DISABLED


def test_housekeeping_deletes_jobs_older_than_the_retention_setting(roots: loop.Roots) -> None:
    old = new_job(roots, job_id="20200101-000000-abcd")
    recent = new_job(roots)
    loop.housekeeping(roots, retention_days=0)
    assert store.list_ids(roots.data) == sorted([old, recent])  # 0 keeps everything
    loop.housekeeping(roots, retention_days=30)
    assert store.list_ids(roots.data) == [recent]
