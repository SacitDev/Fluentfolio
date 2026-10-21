"""The composed solo session through the whole analysis with the real models."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest

from fluentfolio.app import facade
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import source_commit
from fluentfolio.models import catalogue

pytestmark = pytest.mark.slow
TESTS = Path(__file__).resolve().parents[1]
SCHEMA = TESTS.parent / "docs" / "schemas" / "evidence-v1.schema.json"


def test_cs_solo_gives_valid_evidence_that_names_what_ran(
    tmp_path: Path, session_builder: ModuleType, scoring: ModuleType
) -> None:
    truth = session_builder.build(TESTS / "fixtures" / "sessions" / "CS-SOLO.json", tmp_path / "session")
    done = facade.analyze(tmp_path / "session" / "session.wav", "solo")
    assert scoring.gaps(done.job) == []
    evidence = read_json(done.job / "evidence.json")
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(evidence)
    m = evidence["manifest"]
    assert m["stub"] is False and m["commit"] == source_commit() is not None
    assert m["asr"]["revision"] == catalogue.get(m["asr"]["model"]).revision
    assert m["audio"]["duration_ms"] == truth["duration_ms"]
    rows = {t["id"]: t for t in m["tools"]}
    assert all(rows[s]["status"] == "ok" and rows[s]["wall_ms"] > 0 for s in ("media", "vad", "asr"))
    assert rows["asr"]["ran"] == m["asr"]["model"] and rows["asr"]["peak_ram_mb"] > 0
    (c,) = evidence["candidates"]
    # The questions read aloud are split off: one answer per answer turn of the session.
    assert len(c["answers"]) == sum(1 for t in truth["turns"] if t["kind"] == "answer")
    for answer in c["answers"]:
        assert 0 <= answer["start_ms"] < answer["end_ms"] <= truth["duration_ms"] and answer["word_count"] > 0
    text = (done.job / "export" / "for_ai" / "for_ai.txt").read_text(encoding="utf-8")
    first = f"=== FLUENTFOLIO GRADING PROMPT v0.2.0 | evidence schema 1 | export {m['export_id']} ===\n"
    assert text.startswith(first)
    assert text.endswith(f"\nEND OF FLUENTFOLIO FILE {m['export_id']}\n")
    assert sum(line.endswith((" START OF DATA", " END OF DATA")) for line in text.split("\n")) == 2
    assert read_json(done.job / "export" / "details" / "evidence.json") == evidence
