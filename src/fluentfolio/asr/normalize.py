"""One word form for every model: punctuation split off, fillers in one spelling, cut-offs flagged.

Nothing else changes: no word is removed, added, corrected or re-cased, so the transcript stays as
verbatim as the model made it. Punctuation comes from the model, never from the speaker.
"""

from __future__ import annotations

# Hesitation sounds in the spellings models use, mapped to one form each. British transcripts write
# "er" and "erm" for the sounds American ones write "uh" and "um".
FILLERS = {
    "um": "um", "umm": "um", "uhm": "um", "erm": "um",
    "uh": "uh", "uhh": "uh", "er": "uh",
    "ah": "ah", "eh": "eh",
    "hmm": "hmm", "hm": "hmm", "mm": "mm", "mmm": "mm",
}  # fmt: skip
# A word broken off is written "wor-". A dash after a whole word is punctuation, so only the hyphen counts.
CUT_MARK = "-"
APOSTROPHES = ("'", "’")


def _inside(c: str) -> bool:
    """Characters that belong to a word: letters, digits and apostrophes ("don't", "students'")."""
    return c.isalnum() or c in APOSTROPHES


def split(raw: str) -> tuple[str, bool]:
    """The word without the punctuation around it, and whether it was cut off."""
    text = raw.strip()
    start, end = 0, len(text)
    while start < end and not _inside(text[start]):
        start += 1
    while end > start and not _inside(text[end - 1]):
        end -= 1
    cut_off = end < len(text) and text[end] == CUT_MARK
    return text[start:end].replace("’", "'"), cut_off


def word(
    raw: str, start_ms: int, end_ms: int, confidence: float | None, model: str
) -> dict[str, object] | None:
    """One output word, or None when the model's text holds no word (punctuation alone)."""
    text, cut_off = split(raw)
    if not text:
        return None
    filler = FILLERS.get(text.lower())
    return {
        "text": filler or text,
        "raw": raw.strip(),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "confidence": None if confidence is None else round(confidence, 4),
        "confidence_reason": None if confidence is not None else "the model gives no confidence",
        "is_filler": filler is not None,
        "cut_off": cut_off,
        "model": model,
        "flags": [],
    }
