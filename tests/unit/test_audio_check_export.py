"""The audio check in a finished job: which words are withheld where, and what stays in the job folder.

Every expected span, time and text below is worked out by hand from the words and pieces of this file.
"""

from __future__ import annotations

import re
import unicodedata
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fluentfolio.app import facade
from fluentfolio.core.enums import JobState
from fluentfolio.core.errors import InputError
from fluentfolio.core.files import read_json, write_json
from fluentfolio.evidence import build
from fluentfolio.export import answer_check
from fluentfolio.export.audio_check import FILE, WITHHELD
from fluentfolio.ingest import secondary

Row = dict[str, Any]
DURATION = 20_000
SEGMENTS = [(1000, 2000), (2100, 4000), (5200, 9000), (9400, 14000)]
SPOKEN = [
    ("Well,", 1000, 1300, 0), ("I", 1350, 1450, 0), ("think", 1500, 1800, 0), ("um", 1850, 2150, 0),
    ("reading", 2200, 2700, 0), ("is", 2750, 2900, 0), ("really", 2950, 3400, 0), ("useful.", 3450, 3950, 0),
    ("Because", 5200, 5600, 1), ("it", 5650, 5800, 1), ("it", 5850, 6000, 1), ("helps", 6050, 6500, 1),
    ("you", 6550, 6700, 1), ("learn", 6750, 7100, 1), ("new", 7150, 7400, 1),
    ("vocab\u200bulary,", 7450, 8900, 1),  # a zero-width space the hidden words must lose too
    ("and", 9400, 9600, 1), ("also", 9650, 9900, 1), ("ideas", 9950, 10500, 1), ("about", 10550, 10800, 1),
    ("the", 10850, 10950, 1), ("world.", 11000, 11600, 1), ("So", 11700, 11900, 1),
    ("yeah.", 11950, 13900, 1),
]  # fmt: skip
# The audio file: the three stretches of speech with 300 ms between them, as the audio step joins them.
PIECES = [
    {"orig_ms": 1000, "audio_ms": 0, "ms": 3000},
    {"orig_ms": 5200, "audio_ms": 3300, "ms": 3800},
    {"orig_ms": 9400, "audio_ms": 7400, "ms": 4600},
]
MADE = {"candidate": {"file": "for_ai_audio.m4a", "duration_ms": 12_000, "pieces": PIECES}}
# The span starts at the first word: 1.0-9.0 s holds the 16 words from "Well" to "vocabulary" (15 without
# the "um"); "and" starts at 9.4 s. In the audio file 1.0 s is at 0.0 s, and 9.0 s is the end of the second
# piece: 3.3 + 3.8 = 7.1 s.
HIDDEN = "Well, I think um reading is really useful. Because it it helps you learn new vocabulary,"
SPAN = {"orig_start_ms": 1000, "orig_end_ms": 9000, "audio_start_ms": 0, "audio_end_ms": 7100}
# The learner's own transcript: one word changed inside the span (really) and one after it (ideas).
OWN = (
    "Well, I think reading is truly useful.\n"
    "Because it helps you learn new vocabulary, and also thoughts about the world. So yeah.\n"
)


def run(base: Path, synthetic_job: ModuleType, audio: Row | None, **options: Any) -> Path:
    """A finished solo job with the learner's transcript, and the audio file's pieces when `audio` is set."""
    job = synthetic_job.make(base, SPOKEN, SEGMENTS, DURATION, **options)
    source = base / "own.txt"
    source.write_text(OWN, encoding="utf-8")
    secondary.store_transcript(job, source)
    if audio is not None:
        write_json(job / "stages" / "ai_audio.json", {"status": "ok", "candidates": audio})
    assert synthetic_job.finish(base).state is JobState.COMPLETED
    return job


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def runs_of_three(text: str) -> set[tuple[str, ...]]:
    found = words(text)
    return {tuple(found[k : k + 3]) for k in range(len(found) - 2)}


def exported(job: Path) -> dict[str, str]:
    """Every text a learner can send on: the export folder, the ZIP's members and evidence.json."""
    texts = {
        p.relative_to(job).as_posix(): p.read_text(encoding="utf-8")
        for p in (job / "export").rglob("*") if p.suffix in (".txt", ".md", ".json", ".csv")
    }  # fmt: skip
    (archive,) = (job / "export").glob("*.zip")
    with zipfile.ZipFile(archive) as z:
        texts |= {f"zip/{name}": z.read(name).decode("utf-8") for name in z.namelist()}
    return texts | {"evidence.json": (job / "evidence.json").read_text(encoding="utf-8")}


def pointer(evidence: Row, pointer_id: str) -> Row:
    (c,) = evidence["candidates"]
    return next(p for p in c["pointers"] + c["details"]["pointers"] if p["id"] == pointer_id)


def test_the_span_is_the_first_8_seconds_with_12_words_and_its_words_stay_in_the_job(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    job = run(tmp_path, synthetic_job, MADE)
    evidence = read_json(job / "evidence.json")
    assert evidence["manifest"]["audio_check"] == [{"candidate": "candidate", **SPAN}]
    assert read_json(job / FILE) == {"candidate": {**SPAN, "words": HIDDEN}}
    (c,) = evidence["candidates"]
    # The run of hidden words goes on into the second utterance, which gets its own mark; the pause inside
    # the span goes with it, the one after it stays.
    assert [u["text"] for u in c["utterances"]] == [
        WITHHELD, f"[pause 1.2 s] {WITHHELD} [pause 0.4 s] and also ideas about the world. So yeah."
    ]  # fmt: skip
    assert [a["audio_ms"] for a in c["answers"]] == [0]
    text = (job / "export" / "for_ai" / "for_ai.txt").read_text(encoding="utf-8").split("\n")
    assert "AUDIO CHECK: original 00:01.0-00:09.0 = audio file 00:00.0-00:07.1" in text
    assert "ANSWER A01 | unknown | orig 00:01.0 | audio 00:00.0" in text


def test_quotes_and_difference_rows_of_the_span_are_withheld_but_compared_in_full(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    plain = read_json(run(tmp_path / "plain", synthetic_job, None) / "evidence.json")
    checked = read_json(run(tmp_path / "checked", synthetic_job, MADE) / "evidence.json")
    # The comparison reads the words before they were withheld: the same rows and the same word match.
    assert checked["manifest"]["user_transcript"] == plain["manifest"]["user_transcript"]
    assert plain["manifest"]["user_transcript"]["differences_total"] >= 2
    rows = [r["utterance_id"] for r in plain["differences"]]
    assert [r["utterance_id"] for r in checked["differences"]] == rows and set(rows) == {"U001", "U002"}
    assert all(r["ours"] == r["users"] == WITHHELD for r in checked["differences"])  # both hold hidden words
    assert not any(WITHHELD in r["ours"] + r["users"] for r in plain["differences"])
    # The repeated "it" lies in the span: its quote is withheld. Without an audio file it is shown.
    assert [i["text"] for i in pointer(checked, "X01")["items"]] == [WITHHELD]
    assert [i["text"] for i in pointer(plain, "X01")["items"]] != [WITHHELD]


def test_no_exported_file_holds_three_hidden_words_in_a_row(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    hidden = runs_of_three(HIDDEN)
    for name, text in exported(run(tmp_path / "checked", synthetic_job, MADE)).items():
        assert not runs_of_three(text) & hidden, name
    # The same scan finds them in the export of the job without an audio file, so it can fail.
    found = exported(run(tmp_path / "plain", synthetic_job, None))
    assert runs_of_three(found["export/for_ai/for_ai.txt"]) & hidden


def test_music_moves_the_span_and_another_language_too_leaves_none(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    # Music over 1.0-4.0 s: the first span free of it starts at "Because", 5.2-13.2 s, with 15 words (to
    # "So"); 5.2 s is the second piece's start (3.3 s), and 13.2 s lies 3.8 s into the third (7.4 + 3.8).
    job = run(tmp_path / "music", synthetic_job, MADE, music=[(1000, 4000)])
    found = read_json(job / "evidence.json")["manifest"]["audio_check"]
    assert found == [
        {"candidate": "candidate", "orig_start_ms": 5200, "orig_end_ms": 13_200, "audio_start_ms": 3300,
         "audio_end_ms": 11_200}
    ]  # fmt: skip
    # With 5.2-9.0 s in another language as well, no 8-second stretch of the answer is free of both.
    job = run(tmp_path / "both", synthetic_job, MADE, music=[(1000, 4000)], marked=[(5200, 9000)])
    evidence = read_json(job / "evidence.json")
    assert evidence["manifest"]["audio_check"] == [] and read_json(job / FILE) == {}
    assert not any(WITHHELD in u["text"] for u in evidence["candidates"][0]["utterances"])
    text = (job / "export" / "for_ai" / "for_ai.txt").read_text(encoding="utf-8").split("\n")
    assert "AUDIO CHECK: none (no suitable span)" in text


def test_words_over_music_overlapped_or_in_another_language_are_kept_out_of_the_span() -> None:
    found = [
        {"start_ms": 0, "end_ms": 100, "flags": []},
        {"start_ms": 100, "end_ms": 200, "flags": ["music"]},
        {"start_ms": 200, "end_ms": 300, "flags": ["overlap"]},
        {"start_ms": 300, "end_ms": 400, "flags": []},  # its middle lies in the marked speech
        {"start_ms": 400, "end_ms": 500, "flags": ["edge_assigned"]},
    ]
    marked = [{"start_ms": 320, "end_ms": 380, "track": None}]
    assert build._unsure(found, [0, 1, 2, 3, 4], marked) == {1, 2, 3}
    assert build._unsure(found, [0, 4], marked) == set()  # only the candidate's own words are looked at


def test_the_hidden_words_are_given_only_when_asked_and_the_answer_is_checked_against_them(
    tmp_path: Path, synthetic_job: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(tmp_path, synthetic_job, MADE)
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "data"))
    (job_id,) = [p.name for p in (tmp_path / "data" / "jobs").iterdir()]
    assert facade.hidden_words(job_id) == HIDDEN
    answer = tmp_path / "answer.txt"
    for heard, verdict in ((HIDDEN, "PASS"), ("unclear", "FAIL")):
        step0 = f"audio_access: native\naudio_check: {heard}\ninputs: for_ai.txt\n"
        answer.write_text(step0, encoding="utf-8")
        report = facade.check_answer(job_id, answer)
        assert next(r.verdict for r in report.results if r.rule == "AC-11") == verdict
    with pytest.raises(InputError) as caught:
        facade.check_answer(job_id, answer, "person1")
    assert caught.value.code == "E_INPUT_OPTION"
    with pytest.raises(InputError) as caught:
        facade.check_answer(job_id, answer.with_suffix(".docx"))
    assert caught.value.code == "E_INPUT_TEXT"


PLANTED = ("Maria", "Souza", "Abdullah", "Rashid", "王芳", "Anna", "Lukas", "Brown", "Brien", "Trần", "Hồng")
NAMED = (
    # The first line, right after the byte-order mark; possessives and quotes as Word types them.
    "Lukas Brown: Well, I think Brown\u2019s, O\u2019Brien\u2019s, Lukas' and 'Maria' reading"
    " is truly useful.\n"
    "Maria de Souza: Because it helps you learn new vocabulary, Lukas, and also thoughts about the world.\n"
    "王芳： So yeah, Maria.\n"
    "Abdullah al-Rashid: Thanks, Souza and Rashid.\n"
    "Anna： Thanks, 王芳 and Anna.\n"
    "Trần Thị Hồng: Me too, Hồng would say.\n"
    "Liam O\u2019Brien: And O'Brien agrees, Brown.\n"  # a curly apostrophe, as Word types it
)


@pytest.mark.parametrize("codec", ["utf-8", "utf-8-sig", "utf-16", "utf-8-nfd"])
def test_no_planted_name_of_the_learners_transcript_reaches_an_export(
    tmp_path: Path, synthetic_job: ModuleType, codec: str
) -> None:
    job = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, DURATION)
    source = tmp_path / "own.txt"
    # "utf-8-nfd": the accents typed as separate marks, as some Mac and Vietnamese input writes them.
    data = (
        unicodedata.normalize("NFD", NAMED).encode("utf-8") if codec == "utf-8-nfd" else NAMED.encode(codec)
    )
    source.write_bytes(data)
    assert secondary.store_transcript(job, source) == 7  # noqa: PLR2004 - each label is a speaker
    assert synthetic_job.finish(tmp_path).state is JobState.COMPLETED
    texts = exported(job)
    assert any("truly" in text for text in texts.values())  # the transcript's words are in the export
    planted = {form for p in PLANTED for form in (p, unicodedata.normalize("NFD", p))}
    for name, text in texts.items():
        assert not [p for p in planted if p in text], name


def test_an_answer_that_writes_a_name_removed_from_the_learners_transcript_fails_privacy(
    tmp_path: Path, synthetic_job: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, DURATION)
    source = tmp_path / "own.txt"
    source.write_text(f"Ilsabet Varga: {OWN}", encoding="utf-8")  # a made-up speaker label
    secondary.store_transcript(job, source)
    assert synthetic_job.finish(tmp_path).state is JobState.COMPLETED
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "data"))
    answer = tmp_path / "answer.txt"
    step0 = "audio_access: none\naudio_check: not done\ninputs: for_ai.txt\n"
    for said, verdict in (("the candidate", "PASS"), ("Ilsabet", "FAIL"), ("ilsabet varga", "FAIL")):
        answer.write_text(f"{step0}Your results: {said} keeps going.\n", encoding="utf-8")
        report = facade.check_answer(job.name, answer)
        assert next(r.verdict for r in report.results if r.rule == "AC-09") == verdict, said
    # Without a transcript nothing was removed, so the rule does not apply.
    (job / "input" / secondary.NAMES).unlink()
    assert (
        next(r.verdict for r in facade.check_answer(job.name, answer).results if r.rule == "AC-09") == "N/A"
    )


def test_an_answer_that_writes_a_removed_name_with_another_apostrophe_fails_privacy(
    tmp_path: Path, synthetic_job: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, DURATION)
    source = tmp_path / "own.txt"
    source.write_text(f"Ilsabet O\u2019Varga: {OWN}", encoding="utf-8")  # a made-up speaker label
    secondary.store_transcript(job, source)
    assert synthetic_job.finish(tmp_path).state is JobState.COMPLETED
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "data"))
    answer = tmp_path / "answer.txt"
    step0 = "audio_access: none\naudio_check: not done\ninputs: for_ai.txt\n"
    for said, verdict in (("the candidate", "PASS"), ("O\u2019Varga", "FAIL"), ("O'Varga", "FAIL")):
        answer.write_text(f"{step0}Your results: {said} keeps going.\n", encoding="utf-8")
        report = facade.check_answer(job.name, answer)
        assert next(r.verdict for r in report.results if r.rule == "AC-09") == verdict, said


def test_an_answer_needs_a_finished_job_and_in_role_swap_the_person(
    tmp_path: Path, synthetic_job: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, DURATION)
    monkeypatch.setenv("FLUENTFOLIO_HOME", str(tmp_path / "data"))
    answer = tmp_path / "answer.txt"
    answer.write_text("audio_access: none\naudio_check: not done\ninputs: for_ai.txt\n", encoding="utf-8")
    with pytest.raises(InputError) as caught:
        facade.check_answer(job.name, answer)
    assert caught.value.code == "E_JOB_NOT_FINISHED"
    assert synthetic_job.finish(tmp_path).state is JobState.COMPLETED
    evidence = read_json(job / "evidence.json")
    (c,) = evidence["candidates"]
    two = [{**c, "id": "person1", "label": "Person 1"}, {**c, "id": "person2", "label": "Person 2"}]
    write_json(job / "evidence.json", {**evidence, "candidates": two})
    with pytest.raises(InputError) as caught:
        facade.check_answer(job.name, answer)
    assert caught.value.code == "E_INPUT_OPTION" and "person1 or person2" in str(caught.value)
    seen: list[list[str]] = []
    real = answer_check.check_answer
    # Only the named person's evidence reaches the checker, so an answer about the other one fails its format.
    monkeypatch.setattr(answer_check, "check_answer", lambda text, ev, **kw: (
        seen.append([c["id"] for c in ev["candidates"]]) or real(text, ev, **kw)
    ))  # fmt: skip
    assert facade.check_answer(job.name, answer, "person2").results
    assert seen == [["person2"]]
    assert facade.hidden_words(job.name, "person2") is None  # no audio file, so no span
