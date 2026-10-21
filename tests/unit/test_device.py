"""The device rule: Auto, GPU and CPU with stand-in driver answers, each failing condition with its reason."""

from pathlib import Path

import pytest

from fluentfolio.core.enums import Device
from fluentfolio.core.files import read_json, write_json
from fluentfolio.setup import device
from fluentfolio.setup.device import Card

GOOD = Card((8, 6), 24576, "581.57")
SMOKED = {"passed": True, "code": None, "utc": "2026-10-14T00:00:00Z"}
READY = {"backend": "cu130", "gpu_smoke": SMOKED}
NOT_INSTALLED = "GPU support is not installed. Run the installer again and choose GPU."


def unasked() -> list[Card]:
    raise AssertionError("the driver was asked although an earlier condition already failed")


@pytest.mark.parametrize(
    ("state", "cards", "hidden", "reason"),
    [
        ({}, unasked, False, NOT_INSTALLED),
        ({**READY, "backend": "cpu"}, unasked, False, NOT_INSTALLED),
        (READY, unasked, True, "CUDA_VISIBLE_DEVICES hides every graphics card."),
        (READY, lambda: [], False, "No NVIDIA graphics card found."),
        (READY, lambda: [Card((6, 1), 24576, "581.57")], False,
         "This graphics card is too old (compute 6.1; 7.5 is needed)."),
        (READY, lambda: [Card((7, 4), 24576, "581.57")], False,
         "This graphics card is too old (compute 7.4; 7.5 is needed)."),
        (READY, lambda: [Card((8, 6), 4095, "581.57")], False,
         "The graphics card has less than 4 GB of video memory."),
        (READY, lambda: [Card((8, 6), 24576, "579.99")], False,
         "Graphics driver 579.99 is older than 580. Update the driver."),
        ({"backend": "cu130"}, lambda: [GOOD], False, "The GPU test has not run yet. Using the CPU."),
        ({**READY, "gpu_smoke": {**SMOKED, "passed": False, "code": "E_CUDA_RUNTIME"}}, lambda: [GOOD], False,
         "The GPU test failed (E_CUDA_RUNTIME). Using the CPU."),
    ],
)  # fmt: skip
def test_auto_gives_the_first_condition_that_fails(
    state: dict, cards: object, hidden: bool, reason: str
) -> None:
    assert device.why_not_gpu(state, cards, hidden) == reason  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "card",
    [
        Card((7, 5), 4096, "580.00"),  # each lower bound exactly
        Card((7, 6), 4097, "581.12"),
        Card((12, 0), 32607, "590.48"),
        Card((8, 0), 40960, "580.65.06"),  # the Linux driver numbering
    ],
)
def test_auto_takes_the_gpu_when_every_condition_holds(card: Card) -> None:
    assert device.why_not_gpu(READY, lambda: [card], False) is None


def test_only_the_first_card_counts() -> None:
    old = Card((6, 1), 8192, "581.57")
    assert device.why_not_gpu(READY, lambda: [old, GOOD], False) is not None
    assert device.why_not_gpu(READY, lambda: [GOOD, old], False) is None


@pytest.mark.parametrize(
    ("value", "hides"), [("", True), (" ", True), ("-1", True), ("0", False), ("1", False)]
)
def test_cuda_visible_devices_hides_the_cards_only_when_it_lists_none(
    monkeypatch: pytest.MonkeyPatch, value: str, hides: bool
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    assert device.hidden() is hides
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert device.hidden() is False


def ready_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: dict) -> Path:
    app = tmp_path / "app"
    write_json(app / "state.json", state)
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(app))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return app


@pytest.mark.parametrize("requested", list(Device))
def test_with_no_step_that_can_use_the_gpu_everything_runs_on_the_cpu(requested: Device) -> None:
    assert device.resolve(requested, gpu_steps=False, cards=unasked) == {
        "requested": requested.value, "resolved": "cpu", "reason": "No step of this analysis uses the GPU.",
    }  # fmt: skip


def test_a_chosen_device_is_used_as_chosen_without_asking_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready_root(tmp_path, monkeypatch, {})
    assert device.resolve(Device.CPU, True, unasked) == {
        "requested": "cpu", "resolved": "cpu", "reason": "You chose the CPU.",
    }  # fmt: skip
    # GPU is forced even here, where Auto would refuse it; a step that finds no card fails with a code.
    assert device.resolve(Device.GPU, True, unasked) == {
        "requested": "gpu", "resolved": "gpu", "reason": "You chose the GPU.",
    }  # fmt: skip


def test_auto_reads_the_install_state_and_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready_root(tmp_path, monkeypatch, READY)
    assert device.resolve(Device.AUTO, True, lambda: [GOOD]) == {
        "requested": "auto", "resolved": "gpu", "reason": "The graphics card passed every check.",
    }  # fmt: skip
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert device.resolve(Device.AUTO, True, lambda: [GOOD]) == {
        "requested": "auto", "resolved": "cpu", "reason": "CUDA_VISIBLE_DEVICES hides every graphics card.",
    }  # fmt: skip


def test_the_drivers_answer_is_read_line_by_line_and_anything_else_is_skipped() -> None:
    lines = [
        "8.6, 24576, 590.48",
        "7.5, 4096, 580.65.06",
        "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.",
        "[N/A], [N/A], 580.00",
        "8.6, 8192",
    ]
    assert device.parse(lines) == [Card((8, 6), 24576, "590.48"), Card((7, 5), 4096, "580.65.06")]


def test_without_the_drivers_tool_there_is_no_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "nvidia_smi", lambda: None)
    assert device.probe() == []


def test_the_gpu_test_with_every_card_hidden_fails_and_is_kept_for_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = ready_root(tmp_path, monkeypatch, {"backend": "cu130", "data_root": "kept"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    result = device.check_gpu(app)
    assert (result["passed"], result["code"]) == (False, "E_GPU_UNAVAILABLE")
    state = read_json(app / "state.json")
    assert state["gpu_smoke"] == result and state["data_root"] == "kept"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert device.why_not_gpu(state, lambda: [GOOD], False) == (
        "The GPU test failed (E_GPU_UNAVAILABLE). Using the CPU."
    )
