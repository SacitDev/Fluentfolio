"""Which steps an analysis runs for each mode, layout and speaker backend, and the layouts refused."""

from pathlib import Path

import click
import pytest

from fluentfolio.alignment import mfa
from fluentfolio.app.facade import analyze, check_layout, plan
from fluentfolio.asr.base import choices
from fluentfolio.cli.commands import _offset_ms
from fluentfolio.core.errors import AppError
from fluentfolio.settings.model import Limits
from fluentfolio.speakers import ownership, tracks

ORDER = [
    "media",
    "vad",
    "language",
    "asr",
    "speakers",
    "words",
    "structure",
    "candidate_audio",
    "ai_audio",
    "phones",
    "alignment",
    "praat",
    "swiftf0",
    "opensmile",
    "nlp",
    "measures",
    "evidence",
    "export",
]


def test_the_solo_plan_runs_the_children_then_the_steps_in_this_process() -> None:
    stages = plan("solo")
    assert [s.id for s in stages] == ORDER
    children = [s.id for s in stages if s.inline is None and s.prepare is None]
    assert children == [
        "media", "vad", "language", "asr", "speakers", "candidate_audio", "ai_audio", "phones", "praat",
        "swiftf0", "opensmile", "nlp",
    ]  # fmt: skip
    # Forced alignment is an outside program (MFA) that the loop starts between prepare and finish.
    (aligner,) = [s for s in stages if s.prepare is not None]
    found = (aligner.id, aligner.prepare, aligner.finish, aligner.gpu)
    assert found == ("alignment", mfa.prepare, mfa.finish, False)
    # Solo has one speaker; the speaker step only checks for a second voice, so it may fail softly.
    speakers = stages[4]
    assert (speakers.child, speakers.fallback, speakers.gpu, speakers.required) == (
        "speakers-sherpa", "", False, False,
    )  # fmt: skip
    assert stages[5].inline is ownership.run
    # The language check only marks speech, the phone and alignment steps have fallbacks down to the
    # speech model's own word times, the acoustic tools only add context measures, and without the parser
    # pauses are placed by the punctuation, so the analysis goes on without each of them. The audio for an
    # AI chat is an optional extra.
    assert [s.id for s in stages if not s.required] == [
        "language", "speakers", "candidate_audio", "ai_audio", "phones", "alignment", "praat", "swiftf0",
        "opensmile", "nlp",
    ]  # fmt: skip
    assert not any(s.gpu for s in stages)  # Parakeet runs on the CPU


@pytest.mark.parametrize(
    ("mode", "layout"),
    [
        ("solo", "mixed"),
        *((m, lay) for m in ("interview", "role-swap") for lay in ("mixed", "per-track", "two-files")),
    ],
)
def test_speech_recognition_is_a_required_step_in_every_plan(mode: str, layout: str) -> None:
    for model in choices():
        for backend in ("auto", "pyannote", "sherpa"):
            found = [s for s in plan(mode, layout, backend, model) if s.id == "asr"]
            assert [(s.required, s.inline, s.child) for s in found] == [(True, None, "")], (
                mode,
                layout,
                model,
            )


@pytest.mark.parametrize(
    ("model", "gpu"),
    [("parakeet-tdt-0.6b-v2", False), ("parakeet-tdt-0.6b-v3", False), ("whisper-large-v3-turbo", True)],
)
def test_only_whisper_among_the_speech_models_may_use_the_gpu(model: str, gpu: bool) -> None:
    assert {s.id: s for s in plan("solo", asr=model)}["asr"].gpu is gpu


def test_canary_stays_hidden_and_an_unknown_model_is_refused_before_any_job(tmp_path: Path) -> None:
    assert choices() == ["parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3", "whisper-large-v3-turbo"]
    for model in ("canary-1b-v2", "whisper-tiny", "no-such-model"):
        with pytest.raises(AppError) as caught:
            analyze(tmp_path / "practice.wav", "solo", asr=model)
        assert caught.value.code == "E_COMPONENT_UNAVAILABLE"
        assert model not in str(caught.value)  # a typed name is never repeated
    assert not any((tmp_path / "ffhome" / "jobs").glob("*"))


@pytest.mark.parametrize(
    ("backend", "child", "fallback"),
    [
        ("auto", "speakers-pyannote", "speakers-sherpa"),
        ("pyannote", "speakers-pyannote", ""),
        ("sherpa", "speakers-sherpa", ""),
    ],
)
@pytest.mark.parametrize("mode", ["interview", "role-swap"])
def test_a_mixed_two_person_recording_runs_the_chosen_backend_and_its_fallback(
    mode: str, backend: str, child: str, fallback: str
) -> None:
    speakers = {s.id: s for s in plan(mode, "mixed", backend)}["speakers"]
    assert (speakers.child, speakers.fallback, speakers.inline, speakers.required) == (
        child,
        fallback,
        None,
        False,
    )
    assert speakers.gpu == (child == "speakers-pyannote")  # only pyannote may use the GPU here


@pytest.mark.parametrize("layout", ["per-track", "two-files"])
def test_one_track_per_person_needs_no_separation_model(layout: str) -> None:
    stages = plan("interview", layout, "auto")
    assert [s.id for s in stages] == ORDER
    speakers = stages[4]
    assert (speakers.inline, speakers.child, speakers.fallback, speakers.gpu) == (tracks.run, "", "", False)


LIMITS = Limits(max_min=90)
MOST = 90 * 60000


@pytest.mark.parametrize(
    ("mode", "layout", "second", "offset_ms"),
    [
        ("interview", "two-files", Path("b.wav"), None),
        ("interview", "two-files", Path("b.wav"), 0),
        ("interview", "two-files", Path("b.wav"), MOST),
        ("interview", "per-track", None, None),
        ("role-swap", "mixed", None, None),
        ("solo", "mixed", None, None),
    ],
)
def test_layouts_that_fit_their_mode_and_files_are_accepted(
    mode: str, layout: str, second: Path | None, offset_ms: int | None
) -> None:
    check_layout(mode, layout, second, offset_ms, LIMITS)


@pytest.mark.parametrize(
    ("mode", "layout", "second", "offset_ms"),
    [
        ("interview", "stereo", None, None),
        ("solo", "per-track", None, None),
        ("solo", "two-files", Path("b.wav"), None),
        ("interview", "two-files", None, None),
        ("interview", "mixed", Path("b.wav"), None),
        ("interview", "per-track", None, 1500),
        ("interview", "two-files", Path("b.wav"), -1),
        ("interview", "two-files", Path("b.wav"), MOST + 1),
    ],
)
def test_a_layout_that_does_not_fit_is_refused(
    mode: str, layout: str, second: Path | None, offset_ms: int | None
) -> None:
    with pytest.raises(AppError) as caught:
        check_layout(mode, layout, second, offset_ms, LIMITS)
    assert caught.value.code == "E_INPUT_LAYOUT"


@pytest.mark.parametrize(
    ("mode", "style"), [("solo", "once"), ("interview", "per-question"), ("role-swap", "sometimes")]
)
def test_a_wrong_swap_style_is_refused_before_any_job(tmp_path: Path, mode: str, style: str) -> None:
    with pytest.raises(AppError) as caught:
        analyze(tmp_path / "practice.wav", mode, swap_style=style)
    assert caught.value.code == "E_INPUT_OPTION"
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("text", "ms"), [("01:30.5", 90_500), ("1:30", 90_000), ("0:00", 0), ("90:00", 5_400_000)]
)
def test_an_offset_is_read_as_minutes_and_seconds(text: str, ms: int) -> None:
    assert _offset_ms(click.Context(click.Command("analyze")), click.Option(["--offset"]), text) == ms


@pytest.mark.parametrize("text", ["90", "-1:00", "1:-5", "01:3x", "1:30.5.1", ":30"])
def test_an_offset_in_another_form_is_refused(text: str) -> None:
    with pytest.raises(click.BadParameter):
        _offset_ms(click.Context(click.Command("analyze")), click.Option(["--offset"]), text)
