"""The audio check on a composed session: one span in the audio file, and its words in no exported file."""

from __future__ import annotations

import os
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
import pytest

from fluentfolio.app import facade
from fluentfolio.core.files import read_json, write_json
from fluentfolio.export.for_ai import prompt

pytestmark = pytest.mark.slow
SPAN_MS = 8_000
MIN_WORDS = 12
FILLERS = {"uh", "um", "er", "erm", "ah", "hmm", "mm"}
OLD_UTC = "2026-01-01T00:00:00Z"  # a step's time before any test runs, so a step run again gets a later one


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def runs_of_three(text: str) -> set[tuple[str, ...]]:
    found = words(text)
    return {tuple(found[k : k + 3]) for k in range(len(found) - 2)}


def test_cs_solo_with_its_audio_file_withholds_one_part_1_span_everywhere(
    built: Callable[[str], tuple[Path, dict[str, Any]]],
) -> None:
    folder, truth = built("CS-SOLO")
    home = Path(os.environ["FLUENTFOLIO_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.toml").write_text("[export]\naudio = true\n", encoding="utf-8")
    # The learner's own transcript is the script itself, so the differences would show every word.
    own = folder / "own.txt"
    own.write_text("\n".join(t["text"] for t in truth["turns"]) + "\n", encoding="utf-8")
    job = facade.analyze(folder / "session.wav", "solo", transcript=own).job
    (span,) = read_json(job / "evidence.json")["manifest"]["audio_check"]
    hidden = read_json(job / "stages" / "audio_check.json")["candidate"]["words"]
    print(
        f"CS-SOLO span {span['orig_start_ms']}-{span['orig_end_ms']} ms, audio file "
        f"{span['audio_start_ms']}-{span['audio_end_ms']} ms, {len(words(hidden))} hidden words"
    )
    assert span["orig_end_ms"] - span["orig_start_ms"] == SPAN_MS
    assert len([w for w in words(hidden) if w not in FILLERS]) >= MIN_WORDS
    # The script's Part 1 answers are long enough, so the span lies in one of them.
    answers = [t for t in truth["turns"] if t["kind"] == "answer" and t["part"] == 1]
    assert any(t["start_ms"] - 500 <= span["orig_start_ms"] < span["orig_end_ms"] <= t["end_ms"] + 500
               for t in answers)  # fmt: skip
    with av.open(str(job / "export" / "for_ai" / "for_ai_audio.m4a")) as container:
        seconds = float(container.duration or 0) / av.time_base
    assert span["audio_end_ms"] <= seconds * 1000
    # The hidden words are the script's words, so a scan for them can find them.
    assert runs_of_three(hidden) & runs_of_three(" ".join(t["text"] for t in truth["turns"]))
    # A run of three words also said outside the span, or in the prompt, may be shown; every other may not.
    # Outside the span means the speech model's words before or after it, and the learner's own lines for
    # the turns that do not overlap it (their difference rows show the learner's words, such as "home town"
    # where the speech model wrote "hometown"). The span's own turn is never taken as outside.
    heard = read_json(job / "stages" / "words.json")["words"]
    before = " ".join(str(w["raw"]) for w in heard if w["start_ms"] < span["orig_start_ms"])
    after = " ".join(str(w["raw"]) for w in heard if w["end_ms"] > span["orig_end_ms"])
    apart = [t["text"] for t in truth["turns"]
             if t["end_ms"] <= span["orig_start_ms"] or t["start_ms"] >= span["orig_end_ms"]]  # fmt: skip
    shown = runs_of_three(before) | runs_of_three(after) | runs_of_three(prompt("grader", "1.0.0"))
    shown |= set().union(*(runs_of_three(text) for text in apart))
    hidden_runs = runs_of_three(hidden) - shown
    # A pointer quotes one word, so words said nowhere outside the span are searched for one by one too.
    outside = " ".join([before, after, prompt("grader", "1.0.0"), *apart])
    only = set(words(hidden)) - set(words(outside)) - FILLERS
    print(f"{len(hidden_runs)} runs of three words and {len(only)} words are said only in the span")
    assert hidden_runs, "every run of the span is also said elsewhere, so this scan checks nothing"
    assert only, "every word of the span is also said elsewhere, so the one-word scan checks nothing"
    texts = {p.name: p.read_text(encoding="utf-8") for p in (job / "export").rglob("*")
             if p.suffix in (".txt", ".md", ".json", ".csv")}  # fmt: skip
    (archive,) = (job / "export").glob("*.zip")
    with zipfile.ZipFile(archive) as z:
        texts |= {f"zip/{name}": z.read(name).decode("utf-8") for name in z.namelist()}
    texts["evidence.json"] = (job / "evidence.json").read_text(encoding="utf-8")
    # for_ai.txt and details/ on disk, the same in the ZIP, and evidence.json
    assert len(texts) >= 6  # noqa: PLR2004
    leaked = {name: sorted(runs_of_three(text) & hidden_runs) for name, text in texts.items()}
    assert {name: found for name, found in leaked.items() if found} == {}
    single = {name: sorted(set(words(text)) & only) for name, text in texts.items()}
    assert {name: found for name, found in single.items() if found} == {}


def test_cs_solo_audio_asked_for_after_the_run_comes_without_running_the_analysis_again(
    built: Callable[[str], tuple[Path, dict[str, Any]]],
) -> None:
    folder, _ = built("CS-SOLO")
    job = facade.analyze(folder / "session.wav", "solo").job
    record = read_json(job / "job.json")
    assert record["settings"]["export"]["audio"] is False  # no audio unless the learner asks
    # The steps get an older time, because a step run again within the same second can write the same record.
    record["stages"] = [{**s, "started_utc": OLD_UTC, "ended_utc": OLD_UTC} for s in record["stages"]]
    write_json(job / "job.json", record)
    before = {s["id"]: s for s in record["stages"]}
    _, files, changed = facade.export(job.name, audio=True)
    after = {s["id"]: s for s in read_json(job / "job.json")["stages"]}
    assert changed and {k for k, s in after.items() if s != before[k]} == {"ai_audio", "evidence", "export"}
    audio = job / "export" / "for_ai" / "for_ai_audio.m4a"
    assert audio in files and (job / "export" / "for_ai" / "audio_step.txt") in files
    with av.open(str(audio)) as container:
        seconds = float(container.duration or 0) / av.time_base
    (made,) = read_json(job / "stages" / "ai_audio.json")["candidates"].values()
    print(f"CS-SOLO audio asked for later: {seconds:.2f} s decoded, {made['duration_ms']} ms reported")
    assert 0 < seconds * 1000 <= 570_000  # the audio for an AI chat is at most 9.5 min
    assert abs(seconds * 1000 - made["duration_ms"]) <= 100  # the step reports what the file holds
    manifest = read_json(job / "evidence.json")["manifest"]
    assert manifest["audio_export"]["produced"] is True
    (span,) = manifest["audio_check"]
    assert span["audio_end_ms"] <= seconds * 1000
    assert "for_ai_audio.m4a" in (job / "export" / "for_ai" / "for_ai.txt").read_text(encoding="utf-8")
    (archive,) = (job / "export").glob("*.zip")
    with zipfile.ZipFile(archive) as z:
        assert not [n for n in z.namelist() if n.endswith(".m4a")]  # audio never goes into the ZIP
