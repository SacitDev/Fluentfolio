"""The run time estimate against sums worked out by hand from the step times, for every plan."""

import importlib
import os
from pathlib import Path

import pytest

from fluentfolio.app.facade import BACKENDS, plan
from fluentfolio.asr.base import choices
from fluentfolio.core.enums import Device, Layout, Mode
from fluentfolio.core.errors import SetupError
from fluentfolio.core.files import write_json
from fluentfolio.doctor import estimate
from fluentfolio.jobs.stage import STAGES
from fluentfolio.models import catalogue
from fluentfolio.settings.model import Settings
from fluentfolio.settings.store import snapshot

V2 = "parakeet-tdt-0.6b-v2"
MINUTE = 60_000
CONSENT = {"opensmile": {"utc": "2026-10-15T00:00:00Z", "source": "cli-tty"}}


def _estimate(mode: str, minutes: float, *, device: str = Device.CPU, light_run: bool = False,
              components: dict[str, bool] | None = None, consented: bool = False,
              layout: str = Layout.MIXED, asr: str = V2) -> int:  # fmt: skip
    settings = Settings.model_validate({"components": components or {}})
    frozen = snapshot(settings, light_run=light_run)
    return estimate.estimate_ms(
        plan(mode, layout, "auto", asr), mode=mode, asr=asr, settings=frozen, opensmile_consented=consented,
        audio_ms=round(minutes * MINUTE), device=device,
    )  # fmt: skip


def test_a_solo_recording_of_ten_minutes_on_the_cpu_matches_the_hand_sum() -> None:
    # By hand from the step table, solo with the default settings: openSMILE waits for consent and the AI
    # audio file is off (0.4 s each); MFA is experimental on both systems, so off until switched on, and
    # the alignment step only writes the aligner's times.
    # Fixed: 0.46 + 0.63 + 0 + 4.86 + 0 + 0.02 + 0.04 + 0.37 + 0.4 + 1.75 + 0.02 + 0 + 0.32 + 0.4
    #        + 3.17 + 0.12 + 0.05 + 0.14 = 12.75 s
    # Per minute: 0.16 + 0.08 + 0.91 + 2.77 + 6.55 + 1.93 + 0.14 + 0.14 + 0.02 + 0.01 = 12.71 s
    assert not any(catalogue.get("mfa").on_by_default(system) for system in ("windows", "linux"))
    for minutes, ms in {0: 12_750, 1: 25_460, 10: 139_850}.items():
        assert _estimate(Mode.SOLO, minutes) == ms


def test_the_graphics_card_times_count_only_for_steps_that_run_on_it() -> None:
    # An interview of 10 minutes: speaker separation is 5.28 + 15.54 * 10 = 160.68 s on the CPU and
    # 8.67 + 1.07 * 10 = 19.37 s on the card; Parakeet has no card time, so it stays as on the CPU.
    assert _estimate(Mode.INTERVIEW, 10) - _estimate(Mode.INTERVIEW, 10, device=Device.GPU) == 141_310
    # Solo has no step on the card, so the device changes nothing.
    assert _estimate(Mode.SOLO, 10, device=Device.GPU) == _estimate(Mode.SOLO, 10)
    # Whisper runs on the card: 10.91 + 15.29 * 10 = 163.81 s against 1.15 + 0.97 * 10 = 10.85 s, so 10
    # minutes save 152.96 s.
    whisper = "whisper-large-v3-turbo"
    assert _estimate(Mode.SOLO, 10, asr=whisper) - _estimate(
        Mode.SOLO, 10, asr=whisper, device=Device.GPU
    ) == (152_960)


def test_a_light_run_leaves_out_the_phone_model_mfa_and_opensmile() -> None:
    # 120 minutes, solo. The phone model takes 1.75 + 1.93 * 120 = 233.35 s; switched off it takes 0.4 s.
    # With the default settings MFA is off already and openSMILE waits for consent, so only that changes.
    assert _estimate(Mode.SOLO, 120) - _estimate(Mode.SOLO, 120, light_run=True) == 232_950
    # With MFA switched on and openSMILE consented, the run also saves MFA (20.90 + 0.36 * 120 = 64.10 s,
    # against 0.02 s for the aligner's times) and openSMILE (1.18 * 120 = 141.6 s, against 0.4 s):
    # 232.95 + 64.08 + 141.2.
    full = _estimate(Mode.SOLO, 120, components={"mfa": True}, consented=True)
    light = _estimate(Mode.SOLO, 120, components={"mfa": True}, consented=True, light_run=True)
    assert full - light == 438_230


def test_switches_in_settings_turn_their_steps_off() -> None:
    frozen = snapshot(Settings.model_validate({"components": {"praat": False, "whisper-tiny": False}}))
    assert estimate.off_steps(frozen, opensmile_consented=True) == {"praat", "language", "ai_audio"}
    frozen["export"]["audio"] = True
    assert estimate.off_steps(frozen, opensmile_consented=False) == {"praat", "language", "opensmile"}
    # The phone step needs both of its parts.
    phones = snapshot(Settings.model_validate({"components": {"espeak-ng": False}}))
    assert "phones" in estimate.off_steps(phones, opensmile_consented=True)


SWITCHES = [
    {}, {"components": {"whisper-tiny": False}}, {"components": {"phone-model": False}},
    {"components": {"espeak-ng": False}}, {"components": {"ctc-aligner": False}},
    {"components": {"ctc-aligner": False}, "analyzers": {"pronunciation": True}},
    {"components": {"praat": False}}, {"components": {"swiftf0": False}},
    {"components": {"opensmile": False}}, {"components": {"spacy-en-core-web-sm": False}},
    {"analyzers": {"pitch": False}}, {"analyzers": {"pitch": False, "voice-quality": False}},
    {"analyzers": {"pitch": False, "loudness": False}}, {"analyzers": {"grammar": False}},
    {"export": {"audio": True}},
]  # fmt: skip


@pytest.mark.parametrize("consent", [False, True])
@pytest.mark.parametrize("chosen", SWITCHES)
def test_the_steps_switched_off_are_those_that_say_disabled_when_run(
    chosen: dict[str, dict[str, bool]], consent: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each step is run for real on a job with these settings: a step that is on gets past its switch and
    # stops at the next thing it needs (an empty models folder, or no candidate audio).
    settings = Settings.model_validate({k: {**getattr(Settings(), k), **v} if k == "analyzers" else v
                                        for k, v in chosen.items()})  # fmt: skip
    frozen = snapshot(settings)
    job = tmp_path / "job"
    job.mkdir()
    write_json(job / "job.json", {"settings": frozen})
    if consent:
        write_json(Path(os.environ["FLUENTFOLIO_HOME"]) / "consent.json", CONSENT)
    empty = tmp_path / "models"
    empty.mkdir()
    for name in ("fluentfolio.asr.language", "fluentfolio.alignment.phones", "fluentfolio.analyzers.nlp"):
        monkeypatch.setattr(importlib.import_module(name), "models_dir", lambda: empty)
    off = estimate.off_steps(frozen, opensmile_consented=consent)
    for step in ("language", "phones", "praat", "swiftf0", "opensmile", "nlp", "ai_audio"):
        module, function, _ = STAGES[step]
        try:
            status = getattr(importlib.import_module(module), function)(job, lambda pct: None)["status"]
        except SetupError as exc:
            status = exc.code  # it went past its switch to look for its model
        assert (status == "disabled") == (step in off), (step, status, chosen, consent)


@pytest.mark.parametrize("mode", list(Mode))
def test_every_plan_has_a_time_for_each_of_its_steps(mode: str) -> None:
    layouts = [Layout.MIXED] if mode == Mode.SOLO else list(Layout)
    for layout in layouts:
        for backend in BACKENDS:
            for asr in choices():
                for mfa_on in (False, True):
                    for stage in plan(mode, layout, backend, asr):
                        key = estimate.step_key(stage, mode, asr, mfa_on)
                        assert key in estimate.CPU, (mode, layout, backend, asr, key)
                        assert key not in estimate.GPU or stage.gpu, key
                        # A step in a process of its own declares the free memory it needs.
                        assert (key in estimate.RAM_CPU) == (stage.inline is None), (mode, layout, asr, key)
                        assert key not in estimate.RAM_GPU or stage.gpu, key


def _needs(mode: str, minutes: float, *, device: str = Device.CPU, asr: str = V2,
           consented: bool = False) -> dict[str, int]:  # fmt: skip
    return estimate.ram_needs_mb(
        plan(mode, Layout.MIXED, "auto", asr), mode=mode, asr=asr, settings=snapshot(Settings()),
        opensmile_consented=consented, audio_ms=round(minutes * MINUTE), device=device,
    )  # fmt: skip


def test_the_free_memory_each_step_needs_matches_the_hand_lines() -> None:
    # Solo, 60 minutes, default settings: openSMILE waits for consent and the AI audio file is off, so
    # neither needs memory; the steps inside the app's own process are not listed. By hand, fixed plus
    # 60 times the slope, rounded: media 191 + 7.2, vad 401, language 240 + 1.2, Parakeet v2 1211 + 6,
    # the solo voice check 172 + 627, candidate audio 48 + 328.2, phones 2289 + 4.2, the aligner's times
    # 50, Praat 152 + 328.2, SwiftF0 308 + 199.2 and spaCy 631 + 9.
    assert _needs(Mode.SOLO, 60) == {
        "media": 198, "vad": 401, "language": 241, "asr": 1217, "speakers": 799, "candidate_audio": 376,
        "phones": 2293, "alignment": 50, "praat": 480, "swiftf0": 507, "nlp": 640,
    }  # fmt: skip
    # With consent openSMILE runs too: 228 + 17.12 * 120 = 2282.4 at two hours.
    assert _needs(Mode.SOLO, 120, consented=True)["opensmile"] == 2282


def test_the_graphics_card_lowers_the_memory_only_of_the_steps_on_it() -> None:
    # An interview of 60 minutes: pyannote needs 2620 + 8.78 * 60 = 3146.8 MB on the CPU and
    # 1620 + 526.8 = 2146.8 MB of the computer's memory when it runs on the card.
    cpu, gpu = _needs(Mode.INTERVIEW, 60), _needs(Mode.INTERVIEW, 60, device=Device.GPU)
    assert (cpu["speakers"], gpu["speakers"]) == (3147, 2147)
    assert {k: v for k, v in cpu.items() if k != "speakers"} == {
        k: v for k, v in gpu.items() if k != "speakers"
    }
    # Whisper: 1885 MB on the CPU, 1749 MB on the card; Parakeet has no card figure.
    whisper = "whisper-large-v3-turbo"
    assert _needs(Mode.SOLO, 10, asr=whisper)["asr"] == 1885
    assert _needs(Mode.SOLO, 10, asr=whisper, device=Device.GPU)["asr"] == 1749
