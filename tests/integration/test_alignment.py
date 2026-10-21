"""Forced alignment on the VOiCES clip with the real MFA env and phone model, against FFmpeg's silencedetect;
and MFA's word times on one reader's LibriSpeech chapter, run twice.

Each VOiCES run reads the clip from a data folder whose path holds a space and non-ASCII letters.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from fluentfolio.alignment import mfa, phones
from fluentfolio.app import facade
from fluentfolio.asr import base, vad
from fluentfolio.core.enums import JobState
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.paths import app_root, data_root, mfa_root
from fluentfolio.jobs import loop, store
from fluentfolio.settings import store as settings_store
from fluentfolio.settings.model import Limits, Settings
from tests.support import synthetic_job

pytestmark = pytest.mark.slow
CLIP = "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
READER, CHAPTER = "4446", "2275"  # 46 utterances of one reader in LibriSpeech test-clean
GAP_MS = 1000
RHYTHM = ("DM03", "DM04", "DM05", "DM06", "DM07")
ARPABET_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
# The levels of the published check on this clip: speech starts at -35 dB and ends at -30 dB.
ONSET_DB, OFFSET_DB = -35, -30
LONGEST_PATH = 200  # characters; MFA's deepest file must stay under the old Windows path limit


Run = tuple[facade.Analysis, list[int]]


@pytest.fixture
def analyse(test_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Run]:
    """Analyse the clip in solo mode with the given components; also the longest path MFA left at its end."""

    def run(**components: bool) -> Run:
        home = tmp_path / "Dossier élève" / "données"
        monkeypatch.setenv("FLUENTFOLIO_HOME", str(home))
        home.mkdir(parents=True)
        # The clip is 3.4 s long, under the 10 s an analysis needs by default.
        settings_store.save(home / "settings.toml", Settings(limits=Limits(min_s=1), components=components))
        source = tmp_path / "Dossier élève" / CLIP
        shutil.copyfile(test_data / "voices" / CLIP, source)
        longest: list[int] = []
        real = mfa.finish

        def finish(job: Path, app: Path, models: Path, ran: dict[str, Any]) -> dict[str, Any]:
            longest.append(max((len(str(p)) for p in mfa_root(app, job.name).rglob("*")), default=0))
            return real(job, app, models, ran)

        monkeypatch.setattr(mfa, "finish", finish)  # watches the folder, then runs the real step
        return facade.analyze(source, "solo"), longest

    return run


def edges(done: facade.Analysis, method: str) -> tuple[int, int, dict[str, Any]]:
    """The first word's start and the last word's end, from the one utterance the method aligned."""
    alignment = read_json(done.job / "stages" / "alignment.json")
    (utterance,) = alignment["utterances"]
    assert utterance["method"] == method, utterance["reason"]
    timed = [w for w in utterance["words"] if w["method"] == method]
    assert timed and len(timed) == len(utterance["words"])  # every word got the method's own times
    return min(w["start_ms"] for w in timed), max(w["end_ms"] for w in timed), alignment


def rhythm(done: facade.Analysis) -> dict[str, dict[str, Any]]:
    (candidate,) = read_json(done.job / "evidence.json")["candidates"]
    return {m["id"]: m for m in candidate["details"]["measures"] if m["id"] in RHYTHM}


def test_mfa_times_the_clip_within_40_ms_of_silencedetect(
    analyse: Callable[..., Run], test_data: Path, silence: ModuleType, tolerance: Any
) -> None:
    clip = test_data / "voices" / CLIP
    onset, offset = silence.onset_ms(clip, ONSET_DB), silence.offset_ms(clip, OFFSET_DB)
    done, longest = analyse(mfa=True)  # chosen, as MFA is off by default where it is experimental
    assert done.record.state is JobState.COMPLETED, done.record.stages
    start, end, alignment = edges(done, "mfa")
    print(f"MFA {start}-{end} ms, silencedetect {onset}-{offset} ms; MFA {alignment['mfa']}; "
          f"longest path under MFA's folder {longest} characters")  # fmt: skip
    assert abs(start - onset) <= tolerance("integration", "mfa_edge_ms")
    assert abs(end - offset) <= tolerance("integration", "mfa_edge_ms")
    assert alignment["status"] == "ok" and alignment["mfa"]["aligned"] == alignment["mfa"]["utterances"] == 1
    assert alignment["mfa"]["version"] == "3.4.3"
    (utterance,) = alignment["utterances"]
    assert utterance["phones"] and not any(w["g2p"] for w in utterance["words"])
    found = rhythm(done)
    # MFA's phones reach rhythm, but this short clip holds fewer vowel sounds than its values need.
    vowels = sum(p[2].rstrip("012") in ARPABET_VOWELS for p in utterance["phones"])
    assert 0 < vowels < tolerance("rhythm", "min_vowels")
    assert all(found[k]["sources"][0]["tool"] == "mfa" and found[k]["n"] == vowels for k in RHYTHM)
    assert all(found[k]["value"] is None for k in RHYTHM)
    assert all(found[k]["missing"]["code"] == "insufficient_sample" for k in RHYTHM)
    # The job's MFA folder is gone after the run; deleting the job leaves nothing either.
    (deepest,) = longest
    assert 0 < deepest < LONGEST_PATH
    app = app_root()
    assert not mfa_root(app, done.job.name).exists()
    store.delete(data_root(), app, done.job.name)
    assert not done.job.exists() and not mfa_root(app, done.job.name).exists()


def reader_job(base: Path, test_data: Path) -> Path:
    """A solo job up to its speaker step: the chapter's utterances 1 s apart, each its own speech-model chunk.

    Each utterance's words are spread evenly over it, as MFA only needs them to cut the audio. The guard then
    turns down most of MFA's times, which differ from these, so the test reads MFA's own output.
    """
    folder = test_data / "librispeech" / "LibriSpeech" / "test-clean" / READER / CHAPTER
    lines = (folder / f"{READER}-{CHAPTER}.trans.txt").read_text(encoding="utf-8").splitlines()
    per_ms = mfa.RATE // 1000
    audio, spoken, segments, at = [], [], [], 0
    for k, line in enumerate(lines):
        name, text = line.split(" ", 1)
        clip, rate = sf.read(str(folder / f"{name}.flac"), dtype="int16")
        assert rate == mfa.RATE
        length = len(clip) // per_ms  # whole milliseconds, so the times below match the samples
        said = text.lower().split()
        spoken += [(w, at + i * length // len(said), at + (i + 1) * length // len(said), k)
                   for i, w in enumerate(said)]  # fmt: skip
        segments.append((at, at + length))
        audio += [clip[: length * per_ms], np.zeros(GAP_MS * per_ms, dtype=np.int16)]
        at += length + GAP_MS
    job = synthetic_job.make(base, spoken, segments, at)
    record = read_json(job / "job.json")
    # Chosen, as MFA is off by default where it is experimental.
    record["settings"]["components"] = {"mfa": True}
    write_json(job / "job.json", record)
    (job / "input").mkdir(exist_ok=True)
    sf.write(str(job / "input" / "audio16k.wav"), np.concatenate(audio), mfa.RATE, subtype="PCM_16")
    return job


def test_mfa_gives_the_same_word_times_when_run_again(
    test_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MFA adds random noise to the audio before its features unless told not to. On this chapter that moved
    # some word by one 10 ms frame in almost every pair of runs.
    grids: list[dict[str, Any]] = []
    real = mfa.finish

    def finish(job: Path, app: Path, models: Path, ran: dict[str, Any]) -> dict[str, Any]:
        out = mfa_root(app, job.name) / "out"
        found = sorted(out.rglob("*.TextGrid"))
        grids.append({p.relative_to(out).as_posix(): mfa.textgrid(p, 0) for p in found})
        return real(job, app, models, ran)

    monkeypatch.setattr(mfa, "finish", finish)  # keeps MFA's own output, then runs the real step
    stages = [s for s in facade.plan("solo") if s.id in ("words", "structure", "alignment")]
    for run in ("one", "two"):
        job = reader_job(tmp_path / run, test_data)
        roots = loop.Roots(app=app_root(), data=tmp_path / run / "data", models=phones.models_dir())
        loop.run_job(roots, synthetic_job.JOB_ID, stages, audio_min=10, no_progress_s=600)
        alignment = read_json(job / "stages" / "alignment.json")
        n = len(read_json(job / "stages" / "structure.json")["utterances"])
        assert alignment["mfa"]["status"] == "ok" and alignment["mfa"]["utterances"] == n
    # A word moved in about one utterance in ten, so the check needs many of them.
    one, two = grids
    assert len(one) == n >= 40 and all(grid["words"] for grid in one.values())
    print(f"{n} utterances; MFA {alignment['mfa']}")
    assert [name for name in one if one[name] != two.get(name)] == []


def test_canary_words_get_ctc_times_within_80_ms_of_silencedetect(
    tmp_path: Path, test_data: Path, silence: ModuleType, tolerance: Any
) -> None:
    # Known limit: Canary is kept for eval/ only, so this passes only where the model is already in the app
    # folder; revisit if Canary is offered or leaves the catalogue.
    clip = test_data / "voices" / CLIP
    onset, offset = silence.onset_ms(clip, ONSET_DB), silence.offset_ms(clip, OFFSET_DB)
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    (job / "stages").mkdir()
    shutil.copyfile(clip, job / "input" / "audio16k.wav")  # already 16 kHz mono
    write_json(job / "job.json", {"layout": "mixed", "settings": {"asr": {"model": "canary-1b-v2"}}})
    write_json(job / "stages" / "vad.json", vad.run(job, lambda pct: None))
    heard = base.run(job, lambda pct: None, offered_only=False)
    assert heard["provenance"]["word_times"] is False and heard["words"]
    assert all((w["start_ms"], w["end_ms"]) == tuple(heard["chunks"][w["chunk"]]) for w in heard["words"])
    write_json(job / "stages" / "asr.json", heard)
    timed = phones.asr_times(job, lambda pct: None)
    words = timed["words"]
    start, end = words[0]["start_ms"], words[-1]["end_ms"]
    print(f"Canary with CTC times {start}-{end} ms over {len(words)} words; silencedetect {onset}-{offset}")
    assert not any(phones.UNTIMED in w["flags"] for w in words)
    assert timed["provenance"]["times"]["untimed"] == 0
    assert abs(start - onset) <= tolerance("integration", "ctc_edge_ms")
    assert abs(end - offset) <= tolerance("integration", "ctc_edge_ms")
    for w, after in zip(words, words[1:], strict=False):
        a, b = heard["chunks"][w["chunk"]]
        assert a <= w["start_ms"] < w["end_ms"] <= b and w["end_ms"] <= after["start_ms"]
    assert (timed["provenance"]["word_times"], timed["provenance"]["times"]["method"]) == (True, "ctc")


def test_without_mfa_the_ctc_aligner_times_the_clip_within_80_ms(
    analyse: Callable[..., Run], test_data: Path, silence: ModuleType, tolerance: Any
) -> None:
    clip = test_data / "voices" / CLIP
    onset, offset = silence.onset_ms(clip, ONSET_DB), silence.offset_ms(clip, OFFSET_DB)
    done, longest = analyse(mfa=False)
    assert done.record.state is JobState.COMPLETED, done.record.stages
    start, end, alignment = edges(done, "ctc")
    print(f"CTC {start}-{end} ms, silencedetect {onset}-{offset} ms")
    assert abs(start - onset) <= tolerance("integration", "ctc_edge_ms")
    assert abs(end - offset) <= tolerance("integration", "ctc_edge_ms")
    assert alignment["mfa"]["status"] == "disabled" and longest == [0]  # nothing written for MFA
    # Rhythm needs MFA's phones; the CTC aligner's word times never stand in for them.
    assert {m["missing"]["code"] for m in rhythm(done).values()} == {"no_mfa"}
