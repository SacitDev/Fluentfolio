"""The optional text inputs: a question list, and the learner's own transcript with people's names removed.

The transcript feeds only the list of differences from Fluentfolio's transcript. Names are replaced by
neutral labels as it is taken in. The removed names are kept apart in the job's input folder, only so the
answer check can tell when an AI chat's answer writes one; no export holds them.
"""

from __future__ import annotations

import codecs
import re
import unicodedata
from pathlib import Path

from fluentfolio.core.errors import InputError
from fluentfolio.core.files import atomic_write, read_json, write_json

MAX_BYTES = 2_000_000  # 2 MB in decimal units, as every size the app shows
QUESTION_TYPES = (".txt", ".md")
TRANSCRIPT_TYPES = (".txt", ".md", ".vtt", ".srt")
QUESTIONS = "questions.txt"
TRANSCRIPT = "user_transcript.txt"
NAMES = "removed_names.json"  # in the job's input folder, beside the transcript; never exported
# Word, Google Docs and phones may type "O'Brien" or "Jean-Luc" with a curly apostrophe or a dash.
# Between letters each becomes one form, so a name is the same in its label, the talk and an answer.
APOSTROPHE = re.compile(r"(?<=[^\W\d_])[\u2018\u2019\u00b4`\uff07\u2032](?=[^\W\d_])")
HYPHEN = re.compile(r"(?<=[^\W\d_])[\u2010\u2011\u2012\u2013\u2212\ufe63\uff0d](?=[^\W\d_])")
# The byte-order marks Notepad and Word write before UTF-8 and "Unicode" (UTF-16) text.
BOMS = ((codecs.BOM_UTF8, "utf-8"), (codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be"))
# Lines that carry no words: WEBVTT headers and notes, cue numbers, time ranges, lone timestamps.
TIMING = re.compile(
    r"^(?:WEBVTT\b.*|NOTE\b.*|\d+|[\d:.,]+\s*-->.*|\(?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\)?)$"
)
VOICE_TAG = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>]+)>")
# "Anna Smith: ...", "**Anna** (00:12): ...", "Candidate 1: ..." in meeting transcripts: up to five
# capitalised words, then perhaps a number of one or two digits, and a colon, which may be the full-width
# one of Chinese and Japanese text. A capital may be any letter, as in a Turkish or Polish name, so the words
# are matched as letters here and their capitals checked in _capitalised. A replaced byte (U+FFFD) counts as
# a letter, so "Emilie" saved in another encoding than UTF-8 is still a name.
# Known limit: a time before the name ("[00:03] Anna Smith:") hides the label; revisit if learners'
# transcripts come so.
LETTER = r"(?:[^\W\d_]|\ufffd)"
# Combining marks that no composed letter holds, as Devanagari or Arabic vowel signs, go on a word.
# Known limit: marks above U+FFFF, as in Adlam, are not among them; revisit if a learner needs one.
MARK = "".join(f"\\u{c:04x}" for c in range(0x300, 0x10000) if unicodedata.category(chr(c))[0] == "M")
WORD = rf"{LETTER}[\w'.\ufffd{MARK}-]*"
LABEL = re.compile(
    rf"^(?:\*\*|__)?({WORD}(?:\s+{WORD}){{0,4}}(?:\s+\d{{1,2}})?)(?:\*\*|__)?"
    r"\s*(?:\([\d:.]+\))?\s*[:\uff1a]\s*(.*)$"
)
# Small words inside a name, as in "Maria de Souza", "Ludwig van Beethoven" or "Mohammed bin Salman", or
# before a hyphen, as in "Abdullah al-Rashid". Alone they are words of the talk too ("van", "bin").
PARTICLES = frozenset(
    "al bin binti bint da das de del della den der di do dos du e el ibn la le te ter van von y".split()
)
# A label made only of the words below names a role or a place in the test ("Examiner:", "IELTS Examiner:",
# "Candidate One:", "Speaker 1:", "S2:", "Question:", "Part One:", "Test Taker:", "Cue Card:", "Follow-up
# Questions:"), not a person: it still becomes a neutral label, but nothing is removed from the talk and an
# AI answer may use its words. Only letters, one or two digits and number words make a tag, so "Speaker Li:"
# keeps "Li" as a name.
ROLE_WORD = (
    r"(?:candidate|examiner|examinee|interviewer|interviewee|teacher|tutor|student|learner|partner|me|you|host"
    r"|guest|speaker|person|participant|question|answer|q|topic|part|section|task|test|taker|speaking|cue"
    r"|card|turn|follow|follow-up|warm|warm-up|introduction|intro|interview|practice|prompt|round|discussion"
    r"|note)"
)
# Headings of practice notes and feedback ("Date:", "Time:", "Fluency:", "Model Answer:") are no person
# either, and most of these words are in every graded answer.
HEADING_WORD = (
    r"(?:date|time|theme|subject|session|unit|level|lesson|exercise|feedback|summary|score|band|model|sample"
    r"|example|recording|transcript|comment|vocabulary|fluency|coherence|pronunciation|grammar|lexical"
    r"|resource|accuracy|range|result)"
)
NUMBER_WORD = r"one|two|three|four|five|six|seven|eight|nine|ten|first|second|third"
# These name a role only beside one of the words above; "Long" alone is a given name.
SMALL_WORD = r"(?:the|my|your|an?|ielts|mock|long|up)"
ROLE = re.compile(rf"(?i)(?:{ROLE_WORD}|{HEADING_WORD})(?:'s|s'|s)?")  # "Examiner's Question:", "Questions:"
TAG = re.compile(rf"(?i)[a-z]|\d{{1,2}}|[a-z]\d{{1,2}}|{NUMBER_WORD}")
SMALL = re.compile(rf"(?i){SMALL_WORD}")
# "Examiner Anna:", "The Teacher Anna:": the role in front of a person's name, which is no part of the name.
TITLE = re.compile(rf"(?i)^(?:{SMALL_WORD}\s+)?{ROLE_WORD}\s+")


def read_text(source: Path, allowed: tuple[str, ...]) -> str:
    """Read UTF-8, or UTF-16 with its byte-order mark; bytes that do not decode are replaced, not guessed."""
    if source.suffix.lower() not in allowed:
        raise InputError(
            "E_INPUT_TEXT", f"{source.suffix or 'no extension'} is not one of {', '.join(allowed)}"
        )
    size = source.stat().st_size
    if size > MAX_BYTES:
        raise InputError("E_INPUT_TEXT", f"{size} bytes, over the {MAX_BYTES} bytes (2 MB) a text may have")
    try:
        data = source.read_bytes()
    except OSError as exc:  # another program holds it open, or this user may not read it
        raise InputError("E_INPUT_UNREADABLE", type(exc).__name__) from exc
    bom, codec = next(((b, c) for b, c in BOMS if data.startswith(b)), (b"", "utf-8"))
    text = data[len(bom) :].decode(codec, errors="replace")
    # UTF-16 or UTF-32 without its mark: read as UTF-8, its names would pass as other words.
    if "\x00" in text:
        raise InputError("E_INPUT_TEXT", "not UTF-8 or UTF-16 text")
    # Decomposed text, as some Mac and Vietnamese input writes it, is composed: one name, one form.
    return HYPHEN.sub("-", APOSTROPHE.sub("'", unicodedata.normalize("NFC", text)))


def _label(n: int) -> str:
    return f"Speaker {chr(ord('A') + n)}" if n < 26 else f"Speaker {n + 1}"  # noqa: PLR2004 - letters A-Z


def without_names(text: str) -> tuple[str, list[str]]:
    """Transcript lines as "Speaker A: words", with each name replaced everywhere; and the speakers' names."""
    names: dict[str, str] = {}
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or TIMING.match(line):
            continue
        speaker = None
        if tag := VOICE_TAG.search(line):
            speaker, line = tag.group(1).strip(), VOICE_TAG.sub("", line).replace("</v>", "").strip()
        elif (label := LABEL.match(line)) and _capitalised(label.group(1)):
            speaker, line = label.group(1), label.group(2)
        if speaker:
            line = f"{names.setdefault(speaker, _label(len(names)))}: {line}"
        lines.append(line)
    out = "\n".join(lines)
    parts = _parts(list(names))
    # Known limit: a name goes only as its label writes it, so "anna" in the talk stays, as matching any
    # case would take out every "will" or "an" too; revisit if learners' transcripts show such names.
    # A name goes with its "'s", "'d" or "'ll" and in quotes, but "Don't" stays a word, and so does the
    # "Neill" of "O'Neill".
    for part in sorted(parts, key=len, reverse=True):
        found = rf"(?<!\w)(?<!\w'){re.escape(part)}(?=(?:'(?:s|d|ll)?)?(?![\w']))"
        out = re.sub(found, names[parts[part]], out)
    return out, list(names)


def _capitalised(label: str) -> bool:
    # Known limit: a label with a word in lower case ("anna:", "Anna smith:") is not taken for a name, as
    # a sentence with a colon would be; revisit if learners' transcripts show such labels.
    words = label.split()
    return not words[0][0].islower() and all(
        _name_word(word, inner=i < len(words) - 1) for i, word in enumerate(words[1:], 1)
    )


def _name_word(word: str, inner: bool) -> bool:
    """A word starting with a capital, a digit or a letter without case (Chinese, Arabic), or a particle."""
    head, hyphen, rest = word.partition("-")
    if hyphen and head in PARTICLES:  # "al-Rashid"
        return bool(rest) and not rest[0].islower()
    return not word[0].islower() or (inner and word in PARTICLES)


def _is_role(label: str) -> bool:
    words = label.split()
    roles = [w for w in words if ROLE.fullmatch(w) or TAG.fullmatch(w)]
    return bool(roles) and all(w in roles or SMALL.fullmatch(w) for w in words)


def _parts(speakers: list[str]) -> dict[str, str]:
    """Each person's name and each part of it, with the name it comes from; role labels are left out."""
    # A first or last name may also come up in the talk ("Thanks, Anna"), so each part goes too, and each
    # piece of a hyphenated part ("Jean-Luc", "al-Rashid"). A given name that is also a word, such as "An",
    # goes too, though a sentence that starts with it then loses it. A particle alone stays in the talk.
    return {
        part: name
        for name in speakers
        if not _is_role(name)
        for part in [name, *_pieces(TITLE.sub("", name))]
        if len(part) > 1 and not TAG.fullmatch(part) and part not in PARTICLES
    }


def _pieces(name: str) -> list[str]:
    words = name.split()
    return [*words, *(piece for word in words if "-" in word for piece in word.split("-"))]


def store_questions(job: Path, source: Path) -> None:
    atomic_write(job / "input" / QUESTIONS, read_text(source, QUESTION_TYPES))


def store_transcript(job: Path, source: Path) -> int:
    """Keep the learner's transcript with names removed, and the removed names; returns how many speakers."""
    text, names = without_names(read_text(source, TRANSCRIPT_TYPES))
    atomic_write(job / "input" / TRANSCRIPT, text)
    write_json(job / "input" / NAMES, {"names": sorted(_parts(names))})
    return len(names)


def removed_names(job: Path) -> list[str]:
    """The names taken out of the learner's transcript, with their parts; none if no transcript was given."""
    found = job / "input" / NAMES
    return list(read_json(found)["names"]) if found.is_file() else []
