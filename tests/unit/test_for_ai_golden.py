"""The text for an AI chat, byte for byte, from evidence saved from real runs of two composed sessions.

A golden only catches drift; the other export tests prove what each line says. After a deliberate change,
read the new text line by line, then rewrite it with `pytest --update-goldens -k golden`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from fluentfolio.core.files import read_json
from fluentfolio.export.for_ai import render

TESTS = Path(__file__).resolve().parents[1]
FIXTURES = TESTS / "fixtures" / "evidence"
GOLDEN = TESTS / "golden"
SCHEMA = TESTS.parent / "docs" / "schemas" / "evidence-v1.schema.json"
CASES = [
    ("solo", "candidate", "for_ai_solo.txt"),
    ("role_swap", "person1", "for_ai_role_swap_person1.txt"),
    ("role_swap", "person2", "for_ai_role_swap_person2.txt"),
]


def fixed_draw(n: int) -> str:
    """The boundary's random part, fixed: token_hex(n) gives 2n hex digits."""
    return "5eed" * (n // 2)


def audio_of(evidence: dict[str, Any]) -> dict[str, Any] | None:
    """The audio file the run made for this evidence's one candidate, as the export step passes it on."""
    made = evidence["manifest"]["audio_export"]
    return {"file": "for_ai_audio.m4a", "duration_ms": made["duration_ms"]} if made["produced"] else None


@pytest.mark.parametrize(("fixture", "candidate", "golden"), CASES, ids=[c[2] for c in CASES])
def test_the_text_matches_its_golden_byte_for_byte(
    update_goldens: bool, fixture: str, candidate: str, golden: str
) -> None:
    evidence = read_json(FIXTURES / f"{fixture}.json")
    # The evidence holds one audio length, the longest person's, so role-swap texts are rendered without one:
    # their check line says none, while the span's words stay withheld as the evidence holds them.
    audio = audio_of(evidence) if fixture == "solo" else None
    text = render(evidence, candidate, draw=fixed_draw, audio_file=audio).encode("utf-8")
    if update_goldens:
        GOLDEN.mkdir(exist_ok=True)
        (GOLDEN / golden).write_bytes(text)
    assert (GOLDEN / golden).read_bytes() == text


def test_the_saved_evidence_is_valid_and_covers_what_the_goldens_are_for() -> None:
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    solo, swap = (read_json(FIXTURES / f"{name}.json") for name in ("solo", "role_swap"))
    for evidence in (solo, swap):
        validator.validate(evidence)
    assert (solo["manifest"]["mode"], swap["manifest"]["mode"]) == ("solo", "role-swap")
    assert solo["manifest"]["audio_export"]["produced"] and solo["differences"]
    assert [c["id"] for c in swap["candidates"]] == ["person1", "person2"]
    # An audio-check span in each, so the goldens hold the withheld words and the check line.
    assert [s["candidate"] for s in solo["manifest"]["audio_check"]] == ["candidate"]
    assert [s["candidate"] for s in swap["manifest"]["audio_check"]] == ["person1", "person2"]
