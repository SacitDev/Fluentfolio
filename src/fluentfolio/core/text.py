"""Text that leaves Fluentfolio for an AI chat: one line, NFC, nothing invisible that could hide orders."""

from __future__ import annotations

import unicodedata

# Line and paragraph separators end a line in some readers, so they become spaces like newlines do.
BREAKS = ("Zl", "Zp")
FIELD_MARK = "|"  # separates the fields of a line, so it never appears inside one


def clean(text: str) -> str:
    """One NFC line without control, zero-width, direction or tag characters, and with `|` replaced."""
    out = []
    for c in unicodedata.normalize("NFC", text):
        category = unicodedata.category(c)
        if c in "\t\n\r" or category in BREAKS:
            out.append(" ")
        # Cc control, Cf format (zero-width, direction marks, tags), Co private use, Cs, Cn unassigned
        elif category.startswith("C"):
            continue
        else:
            out.append("/" if c == FIELD_MARK else c)
    return " ".join("".join(out).split())
