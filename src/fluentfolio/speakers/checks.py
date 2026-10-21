"""Checks on speaker turns: who counts as a person, one or three voices, similar voices, a solo second voice.

Every backend's turns pass through here, so the speakers step writes one shape whichever backend ran.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fluentfolio.core.enums import Mode
from fluentfolio.speakers.ownership import Turn, overlaps, talk_ms

OTHER_PCT = 5  # a voice with less of the talk time than this (percent) is "other", kept out of the evidence
MAX_SPEAKERS = 3  # two-person modes allow 1 to 3 speakers; any further voice is "other" too
# Each embedder has its own cosine scale, so each backend has its own "sound alike" line: the 99th percentile
# of different same-gender speakers of LibriSpeech test-clean, rounded up to 0.05.
# Known limit: measured on read speech; revisit with real recordings.
SIMILAR = {"pyannote-community-1": 0.45, "sherpa-onnx-diarizer": 0.9}
SOLO_SECOND_VOICE_PCT = 5  # a second voice in more than this percent of a solo recording warns
# The child that takes over when speaker separation fails, and the backend it stands in for.
STANDS_IN_FOR = {"speakers-sherpa": "pyannote-community-1"}


def relabel(turns: list[Turn]) -> tuple[list[Turn], dict[str, str]]:
    """Speaker IDs S1, S2, ... in order of first appearance, with the map from the backend's own labels."""
    ids: dict[str, str] = {}
    for _, _, label in sorted(turns):
        ids.setdefault(label, f"S{len(ids) + 1}")
    return sorted((a, b, ids[label]) for a, b, label in turns), ids


def summarize(backend: str, turns: list[Turn], *, mode: str) -> dict[str, Any]:
    """The speakers step's result: the turns, each speaker's talk time and kind, and the mode's warnings."""
    talk = {
        s: talk_ms([(a, b) for a, b, k in turns if k == s]) for s in dict.fromkeys(k for _, _, k in turns)
    }
    total = sum(talk.values())
    ranked = sorted(talk, key=lambda s: talk[s], reverse=True)
    main = {s for s in ranked[:MAX_SPEAKERS] if talk[s] * 100 >= OTHER_PCT * total}  # whole numbers: exact
    warnings = []
    if len(main) < 2:  # noqa: PLR2004 - the two people of a two-person mode
        warnings.append("W_ONE_VOICE")
    elif len(main) == MAX_SPEAKERS:
        warnings.append("W_THIRD_VOICE")
    if len(main) < len(talk):
        warnings.append("W_OTHER_VOICE")
    return {
        "status": "ok",
        "backend": backend,
        "mode": mode,
        "turns": [{"start_ms": a, "end_ms": b, "speaker": k} for a, b, k in turns],
        "speakers": [
            {
                "id": s,
                "speech_ms": talk[s],
                "share": round(talk[s] / total, 3) if total else 0.0,
                "kind": "main" if s in main else "other",
            }
            for s in talk
        ],  # fmt: skip
        "overlap_ms": sum(b - a for a, b in overlaps(turns)),
        "similarity": None,
        "role_confidence_cap": None,
        "warnings": warnings,
    }


def main_pair(result: dict[str, Any]) -> list[str]:
    """The two main speakers with the most talk time, when there are at least two."""
    main = sorted((s for s in result["speakers"] if s["kind"] == "main"), key=lambda s: -s["speech_ms"])
    return [s["id"] for s in main[:2]] if len(main) >= 2 else []  # noqa: PLR2004


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / norm if norm > 0 and np.isfinite(norm) else None


def add_similarity(result: dict[str, Any], value: float | None) -> None:
    """Record how alike the two main voices sound; alike enough, the swap points need a human check."""
    result["similarity"] = None if value is None else round(value, 3)
    if value is not None and value >= SIMILAR[result["backend"]]:
        result["warnings"].append("W_SIMILAR_VOICES")
        result["role_confidence_cap"] = "medium"


def second_voice(backend: str, turns: list[Turn]) -> dict[str, Any]:
    """Solo: the share of speech in voices other than the main one; over 5 % it is a warning only."""
    talk = {s: talk_ms([(a, b) for a, b, k in turns if k == s]) for s in {k for _, _, k in turns}}
    total = sum(talk.values())
    other = total - max(talk.values(), default=0)
    share = other / total if total else 0.0
    return {
        "status": "ok",
        "backend": "single",
        "mode": Mode.SOLO,
        "second_voice": {"checked_by": backend, "voices": len(talk), "share": round(share, 3)},
        "warnings": ["W_SECOND_VOICE_SOLO"] if other * 100 > SOLO_SECOND_VOICE_PCT * total else [],
    }
