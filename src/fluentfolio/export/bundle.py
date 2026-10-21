"""The export stage: per candidate the text for an AI chat, plus details/ and the ZIP, in export/."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

from fluentfolio.core.files import atomic_write, read_json, write_json
from fluentfolio.export import details, for_ai

NAMES = {"candidate": "for_ai.txt", "person1": "for_ai_person1.txt", "person2": "for_ai_person2.txt"}
# The message to send with each person's audio file, made only when the audio file is.
STEPS = {
    "candidate": "audio_step.txt",
    "person1": "audio_step_person1.txt",
    "person2": "audio_step_person2.txt",
}
# The earliest date a ZIP entry can hold; with it and a fixed order the same files give the same ZIP bytes.
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def write_zip(job: Path, files: list[str]) -> str:
    """export/fluentfolio_<job_id>.zip with the for_ai texts and details/; audio never goes in (some chats
    refuse a ZIP that holds audio, and the audio step is the learner's own choice)."""
    members = sorted(
        f for f in files
        if (f.startswith("export/for_ai/") and f.endswith(".txt")) or f.startswith("export/details/")
    )  # fmt: skip
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in members:
            info = zipfile.ZipInfo(name.removeprefix("export/"), ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16  # a plain readable file when unpacked on Linux
            archive.writestr(info, (job / name).read_bytes())
    path = job / "export" / f"fluentfolio_{job.name}.zip"
    atomic_write(path, buffer.getvalue())
    return path.relative_to(job).as_posix()


def run(job: Path) -> dict[str, Any]:
    """Write export/for_ai/, export/details/ and the ZIP from the results."""
    evidence = read_json(job / "evidence.json")
    structure = read_json(job / "stages" / "structure.json")
    found = job / "stages" / "ai_audio.json"
    audio = read_json(found)["candidates"] if found.is_file() else {}  # made only when the learner asks
    out = job / "export"
    files = []
    for c in evidence["candidates"]:
        path = out / "for_ai" / NAMES[c["id"]]
        atomic_write(path, for_ai.render(evidence, c["id"], audio_file=audio.get(c["id"])))
        files.append(path.relative_to(job).as_posix())
        if (made := audio.get(c["id"])) is not None:
            path = out / "for_ai" / STEPS[c["id"]]
            atomic_write(path, for_ai.audio_step(evidence, c["id"], made))
            files.append(path.relative_to(job).as_posix())
    files += [f"export/for_ai/{a['file']}" for a in audio.values()]
    if not evidence["candidates"]:  # roles unavailable: who spoke when, and how to name the candidate
        path = out / "for_ai" / NAMES["candidate"]
        atomic_write(
            path,
            for_ai.render_without_roles(evidence, structure["transcript"], structure["unknown_candidate"]),
        )
        files.append(path.relative_to(job).as_posix())
    # A run again after the roles or the audio setting changed must not leave old texts beside the new.
    for old in [*(out / "for_ai").glob("for_ai*.txt"), *(out / "for_ai").glob("audio_step*.txt")]:
        if old.relative_to(job).as_posix() not in files:
            old.unlink()
    write_json(out / "details" / "evidence.json", evidence)
    runs = [] if evidence["candidates"] else structure["transcript"]  # kept only when roles are unavailable
    atomic_write(out / "details" / "transcript.md", details.transcript(evidence, runs))
    files += ["export/details/evidence.json", "export/details/transcript.md"]
    measures = read_json(job / "analyzers" / "measures.json")
    # Each pitch tool's own values beside the combined ones, so a reader can see where they differ.
    write_json(out / "details" / "acoustics_by_tool.json", {
        "pitch": {cid: c["pitch_by_tool"] for cid, c in measures["candidates"].items()},
        "nc_input": evidence["manifest"]["nc_component_used"],
    })  # fmt: skip
    files.append("export/details/acoustics_by_tool.json")
    found = job / "analyzers" / "opensmile.json"
    csv_path = out / "details" / "egemaps_88.csv"
    if found.is_file() and (table := details.egemaps(read_json(found))) is not None:
        atomic_write(csv_path, table)
        files.append("export/details/egemaps_88.csv")
    else:  # consent taken back since an earlier run: its values must not stay beside the new export
        csv_path.unlink(missing_ok=True)
    atomic_write(out / "details" / "README.txt", details.readme(evidence, files))
    files.append("export/details/README.txt")
    files.append(write_zip(job, files))
    return {"status": "ok", "files": files}
