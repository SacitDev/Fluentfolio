"""The export ZIP: which files go in, their names and date, and that the same files give the same bytes."""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import ModuleType

from fluentfolio.core.enums import JobState
from fluentfolio.core.files import read_json, write_json
from fluentfolio.export import bundle
from fluentfolio.export.bundle import write_zip

FILES = {
    "export/for_ai/for_ai_person2.txt": b"second person\n",
    "export/for_ai/for_ai_person1.txt": b"first person\n",
    "export/for_ai/for_ai_audio_person1.m4a": b"not text",
    "export/details/evidence.json": b'{"schema_version": "1.0"}\n',
    "export/details/README.txt": b"what each file holds\n",
}


def job_folder(tmp_path: Path) -> Path:
    job = tmp_path / "20261014-200000-ab12"
    for name, data in FILES.items():
        (job / name).parent.mkdir(parents=True, exist_ok=True)
        (job / name).write_bytes(data)
    return job


def test_the_zip_holds_the_texts_and_details_in_name_order_and_never_the_audio(tmp_path: Path) -> None:
    job = job_folder(tmp_path)
    assert write_zip(job, list(FILES)) == "export/fluentfolio_20261014-200000-ab12.zip"
    with zipfile.ZipFile(job / "export" / "fluentfolio_20261014-200000-ab12.zip") as archive:
        assert archive.namelist() == [
            "details/README.txt", "details/evidence.json",
            "for_ai/for_ai_person1.txt", "for_ai/for_ai_person2.txt",
        ]  # fmt: skip
        assert {i.date_time for i in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        assert archive.read("for_ai/for_ai_person1.txt") == b"first person\n"
        assert archive.read("details/evidence.json") == FILES["export/details/evidence.json"]
        assert archive.testzip() is None


SPOKEN = [
    ("I", 1000, 1200, 0), ("like", 1250, 1500, 0), ("my", 1550, 1700, 0), ("home", 1750, 2100, 0),
    ("town", 2150, 2500, 0), ("because", 2550, 3000, 0), ("it", 3050, 3200, 0), ("is", 3250, 3400, 0),
    ("quiet.", 3450, 3950, 0),
]  # fmt: skip


def zipped(job: Path, files: list[str]) -> list[str]:
    with zipfile.ZipFile(job / files[-1]) as archive:
        return archive.namelist()


def test_details_hold_the_transcript_a_readme_and_opensmile_values_only_while_it_runs(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    job = synthetic_job.make(tmp_path, SPOKEN, [(1000, 4000)], 12_000)
    assert synthetic_job.finish(tmp_path).state is JobState.COMPLETED
    details = job / "export" / "details"
    line = "- `00:01.0` **U001** the candidate (A01, Part unknown): I like my home town because it is quiet."
    assert line in (details / "transcript.md").read_text(encoding="utf-8").split("\n")
    found = job / "analyzers" / "opensmile.json"
    values = {"candidate": {"functionals": {"all": {"loudness_sma3_amean": 0.5}}}}
    write_json(found, {"status": "ok", "candidates": values})
    files = bundle.run(job)["files"]
    table = details / "egemaps_88.csv"
    assert table.read_text(encoding="utf-8") == "candidate,span,loudness_sma3_amean\ncandidate,all,0.5\n"
    assert "export/details/egemaps_88.csv" in files and "details/egemaps_88.csv" in zipped(job, files)
    assert "details/egemaps_88.csv: " in (details / "README.txt").read_text(encoding="utf-8")
    assert {"details/README.txt", "details/transcript.md"} <= set(zipped(job, files))
    write_json(found, {"status": "disabled", "code": "E_OPENSMILE_CONSENT_MISSING", "candidates": {}})
    files = bundle.run(job)["files"]
    assert not table.exists() and "details/egemaps_88.csv" not in zipped(job, files)
    assert "egemaps_88.csv" not in (details / "README.txt").read_text(encoding="utf-8")


def test_an_audio_file_is_named_in_its_persons_text_and_the_readme_but_never_zipped(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    job = synthetic_job.make(tmp_path, SPOKEN, [(1000, 4000)], 12_000)
    assert synthetic_job.finish(tmp_path).state is JobState.COMPLETED
    made = {"file": "for_ai_audio.m4a", "duration_ms": 4300, "pieces": []}
    write_json(job / "stages" / "ai_audio.json", {"status": "ok", "candidates": {"candidate": made}})
    (job / "export" / "for_ai" / "for_ai_audio.m4a").write_bytes(b"audio")
    files = bundle.run(job)["files"]
    assert "export/for_ai/for_ai_audio.m4a" in files
    assert not any(name.endswith(".m4a") for name in zipped(job, files))
    text = (job / "export" / "for_ai" / "for_ai.txt").read_text(encoding="utf-8").split("\n")
    assert "audio for AI: for_ai_audio.m4a, candidate only, 0:04" in text
    readme = (job / "export" / "details" / "README.txt").read_text(encoding="utf-8")
    assert "for_ai/for_ai_audio.m4a: one person's own speech only" in readme


def test_the_audio_step_message_names_the_file_its_length_and_span_and_goes_with_the_audio(
    tmp_path: Path, synthetic_job: ModuleType
) -> None:
    job = synthetic_job.make(tmp_path, SPOKEN, [(1000, 4000)], 12_000)
    assert synthetic_job.finish(tmp_path).state is JobState.COMPLETED
    made = {"file": "for_ai_audio.m4a", "duration_ms": 12_000, "pieces": []}  # no piece, so no span
    write_json(job / "stages" / "ai_audio.json", {"status": "ok", "candidates": {"candidate": made}})
    files = bundle.run(job)["files"]
    path = job / "export" / "for_ai" / "audio_step.txt"
    assert "export/for_ai/audio_step.txt" in files and "for_ai/audio_step.txt" in zipped(job, files)
    export_id = read_json(job / "evidence.json")["manifest"]["export_id"]
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0] == f"=== FLUENTFOLIO AUDIO STEP v0.2.0 | export {export_id} ==="
    assert lines[1] == (
        f"The attached audio file for_ai_audio.m4a belongs to Fluentfolio export {export_id}. It holds the "
        "candidate only, 0:12 in total."
    )
    assert lines[3] == (
        'Do STEP 0 again for this audio. This export has no audio-check span ("AUDIO CHECK: none" in the '
        'HEADER): write "audio_check: not done" and keep the Pronunciation estimate from the clues in the '
        "data, with its evidence note."
    )
    assert not any("{" in line for line in lines)  # every placeholder is filled
    readme = (job / "export" / "details" / "README.txt").read_text(encoding="utf-8")
    assert "Open for_ai/for_ai.txt, copy all of it" in readme
    assert "for_ai/audio_step.txt: the message to send with that person's audio file" in readme
    # With a span, the message gives its times in the audio file, each cut down to the tenth.
    evidence = read_json(job / "evidence.json")
    span = {"orig_start_ms": 1000, "orig_end_ms": 9000, "audio_start_ms": 1250, "audio_end_ms": 9299}
    evidence["manifest"]["audio_check"] = [{"candidate": "candidate", **span}]
    write_json(job / "evidence.json", evidence)
    bundle.run(job)
    assert path.read_text(encoding="utf-8").split("\n")[3] == (
        "Do STEP 0 again for this audio. For audio_check, write what the candidate says in the audio file "
        "from 00:01.2 to 00:09.2."
    )
    # The audio setting turned off since: the message goes with the audio file.
    write_json(job / "stages" / "ai_audio.json", {"status": "disabled", "candidates": {}})
    assert "export/for_ai/audio_step.txt" not in bundle.run(job)["files"] and not path.exists()


def test_the_same_files_give_the_same_zip_bytes_and_no_temporary_file_is_left(tmp_path: Path) -> None:
    job = job_folder(tmp_path)
    write_zip(job, list(FILES))
    first = (job / "export" / "fluentfolio_20261014-200000-ab12.zip").read_bytes()
    write_zip(job, list(reversed(FILES)))
    assert (job / "export" / "fluentfolio_20261014-200000-ab12.zip").read_bytes() == first
    names = sorted(p.name for p in (job / "export").iterdir())
    assert names == ["details", "fluentfolio_20261014-200000-ab12.zip", "for_ai"]
