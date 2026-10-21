"""The size of the text for an AI chat from real runs: a 15-minute composed interview, the private recordings.

The text has no size limit: nothing is cut to fit, and it states no size. Its estimated tokens (characters
divided by 4) are recorded so the docs can say how long a text gets. The private runs record numbers only,
and every assertion compares plain numbers, so a failure report never shows a path or a word.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fluentfolio.app import facade
from fluentfolio.core.files import read_json
from fluentfolio.export.for_ai import approx_tokens

pytestmark = pytest.mark.slow
FIFTEEN_MIN_MS = 15 * 60_000


def sized(job: Path) -> list[dict[str, Any]]:
    """Each text's characters, estimated tokens and any size or cut lines, as counts."""
    out = []
    for path in sorted((job / "export" / "for_ai").glob("for_ai*.txt")):
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        out.append({
            "chars": len(text), "approx_tokens": approx_tokens(text),
            "size_note": sum(line.startswith("size: about ") for line in lines),
            "cut_notes": sum(line.startswith("cut to fit: ") or line.endswith(" [cut]") for line in lines),
        })  # fmt: skip
    return out


def test_a_15_minute_interview_gives_the_whole_text_with_nothing_cut(
    built: Callable[[str], tuple[Path, dict[str, Any]]],
) -> None:
    folder, truth = built("CS-LONG")
    assert truth["duration_ms"] >= FIFTEEN_MIN_MS
    done = facade.analyze(folder / "session.wav", "interview", partner_consent="cli-flag")
    (text,) = sized(done.job)
    print(f"CS-LONG: {truth['duration_ms']} ms, for_ai.txt {text}")
    assert (text["size_note"], text["cut_notes"]) == (0, 0)


@pytest.mark.private_sample
@pytest.mark.parametrize("index", [0, 1], ids=["sample1", "sample2"])
def test_private_recording_texts_and_their_size(
    index: int, private_recordings: list[Path], private_summary: Callable[[str, dict[str, object]], None]
) -> None:
    done = facade.analyze(private_recordings[index], "interview", partner_consent="cli-flag")
    evidence = read_json(done.job / "evidence.json")
    texts = sized(done.job)
    found = {
        "duration_s": round(evidence["manifest"]["audio"]["duration_ms"] / 1000, 1),
        "candidates": len(evidence["candidates"]), "texts": len(texts),
        **{key: [t[key] for t in texts] for key in ("chars", "approx_tokens", "size_note", "cut_notes")},
    }  # fmt: skip
    private_summary("export_size", {f"sample{index + 1}": found})
    n_texts, n_candidates = len(texts), len(evidence["candidates"])
    assert (n_texts, n_candidates) == (1, 1)
    assert [(t["size_note"], t["cut_notes"]) for t in texts] == [(0, 0)]
