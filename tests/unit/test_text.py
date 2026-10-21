"""Text bound for an AI chat comes out as one clean line."""

import unicodedata

import pytest

from fluentfolio.core.text import clean


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cafe\u0301", "caf\u00e9"),  # decomposed accent joins its letter (NFC)
        ("zero\u200bwidth", "zerowidth"),
        ("a\u202eb\u202cc", "abc"),  # direction overrides
        ("tag\U000e0041\U000e0042s", "tags"),  # invisible tag characters can spell hidden orders
        ("bell\x07 and\x00 nul", "bell and nul"),
        ("line\none\r\ntwo\tthree", "line one two three"),
        ("sep\u2028arate\u2029d", "sep arate d"),
        ("a | b|c", "a / b/c"),
        ("  spaced   out  ", "spaced out"),
        ("private\ue000use", "privateuse"),
        ("plain words, kept.", "plain words, kept."),
    ],
)
def test_clean(raw: str, expected: str) -> None:
    assert clean(raw) == expected


def test_nothing_invisible_survives_in_any_plane() -> None:
    every = "".join(chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF)
    out = clean(every)
    assert "\n" not in out and "|" not in out
    assert not {unicodedata.category(c) for c in out} & {"Cc", "Cf", "Co", "Cn", "Zl", "Zp"}
