"""The analyzers: our measure groups, what each needs and gives, and which a learner can switch off.

Doctor, Settings and the job plan read this table without loading any tool. Measure IDs are those of the
evidence file; a group that gives only `details/` values is kept out of the text for an AI chat.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Analyzer:
    id: str
    title: str
    gives: str  # in a learner's words, for the Settings page
    needs: tuple[str, ...]  # component IDs; any one of them is enough
    switchable: bool  # an optional tool does the work, so Settings can switch it off


ANALYZERS = (
    Analyzer("fluency", "Fluency", "speech rate, pauses and the length of runs", ("silero-vad",), False),
    Analyzer("disfluency", "Hesitation", "hesitation sounds, repeats and self-corrections", (), False),
    Analyzer("discourse", "Linking words", "linking words and phrases", (), False),
    Analyzer("lexical", "Vocabulary", "word variety and rare words", (), False),
    Analyzer(
        "grammar", "Grammar pointers", "clauses and examples of structures", ("spacy-en-core-web-sm",), True
    ),
    Analyzer(
        "pitch",
        "Pitch",
        "pitch range, phrase endings and flat stretches",
        ("praat", "swiftf0", "opensmile"),
        True,
    ),
    Analyzer("loudness", "Loudness", "loudness and its range", ("opensmile",), True),
    Analyzer("rhythm", "Rhythm", "how vowel lengths vary (kept in the details)", ("mfa",), False),
    Analyzer(
        "pronunciation",
        "Words to check",
        "words whose sounds differ from the dictionary (experimental)",
        ("phone-model", "espeak-ng"),
        True,
    ),
    Analyzer(
        "voice-quality", "Voice quality", "jitter, shimmer and noise (kept in the details)", ("praat",), True
    ),
)


def switchable() -> tuple[str, ...]:
    """The analyzers a job's settings can switch off."""
    return tuple(a.id for a in ANALYZERS if a.switchable)
