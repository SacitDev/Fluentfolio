"""The optional text inputs: a question list, and the learner's own transcript taken in without names."""

from __future__ import annotations

import codecs
import unicodedata
from pathlib import Path

import pytest

from fluentfolio.core.errors import InputError
from fluentfolio.ingest import secondary

NAMES = ("Anna", "Smith", "Jonas", "Weber")  # planted names; none may be in the stored transcript
MEET = (
    "**Anna Smith** (00:12): Hello, can you hear me?\n"
    "**Jonas Weber** (00:15): Yes. Thanks, Anna. Shall we start?\n"
    "\n"
    "Anna Smith: Sure. Jonas, do you like your home town?\n"
)
VTT = (
    "WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.500\n<v Anna Smith>Do you like music?</v>\n\n"
    "2\n00:00:04.000 --> 00:00:09.000\n<v.loud Jonas>Yes, Anna, I do.</v>\n"
)
SRT = "1\n00:00:01,000 --> 00:00:03,500\nDo you like music?\n\n2\n00:00:04,000 --> 00:00:09,000\nYes I do.\n"


def test_speaker_names_become_neutral_labels_everywhere() -> None:
    text, speakers = secondary.without_names(MEET)
    assert text.split("\n") == [
        "Speaker A: Hello, can you hear me?",
        "Speaker B: Yes. Thanks, Speaker A. Shall we start?",
        "Speaker A: Sure. Speaker B, do you like your home town?",
    ]
    assert speakers == ["Anna Smith", "Jonas Weber"]
    assert not any(name in text for name in NAMES)


def test_subtitle_files_lose_their_timing_lines_and_voice_names() -> None:
    text, speakers = secondary.without_names(VTT)
    assert text == "Speaker A: Do you like music?\nSpeaker B: Yes, Speaker A, I do."
    assert speakers == ["Anna Smith", "Jonas"]
    assert secondary.without_names(SRT) == ("Do you like music?\nYes I do.", [])


def test_the_stored_transcript_holds_no_planted_name(tmp_path: Path) -> None:
    source = tmp_path / "meeting.md"
    source.write_text(MEET, encoding="utf-8")
    job = tmp_path / "job"
    assert secondary.store_transcript(job, source) == 2  # noqa: PLR2004
    stored = (job / "input" / secondary.TRANSCRIPT).read_text(encoding="utf-8")
    assert stored.startswith("Speaker A: ") and not any(name in stored for name in NAMES)


def test_the_removed_names_are_kept_apart_for_the_answer_check(tmp_path: Path) -> None:
    source = tmp_path / "meeting.md"
    source.write_text(MEET, encoding="utf-8")
    job = tmp_path / "job"
    assert secondary.removed_names(job) == []  # no transcript given: nothing was removed
    secondary.store_transcript(job, source)
    # Each full name and each part of it, as the transcript lost them.
    assert sorted(secondary.removed_names(job)) == [
        "Anna",
        "Anna Smith",
        "Jonas",
        "Jonas Weber",
        "Smith",
        "Weber",
    ]
    source.write_text(SRT, encoding="utf-8")
    secondary.store_transcript(job, source)  # a new transcript with no names leaves none from the last one
    assert secondary.removed_names(job) == []


ROLES = (
    "Examiner: Tell me about your home town.\n"
    "Me: Me? I love it. Anna says it is quiet.\n"
    "Speaker C: Thanks.\n"
    "S2: Sure.\n"
    "Anna Smith: Hello, Candidate.\n"
)


def test_a_role_label_becomes_a_neutral_label_but_is_not_a_name(tmp_path: Path) -> None:
    text, speakers = secondary.without_names(ROLES)
    assert text.split("\n") == [
        "Speaker A: Tell me about your home town.",
        "Speaker B: Me? I love it. Speaker E says it is quiet.",  # "Me" in the talk stays a word
        "Speaker C: Thanks.",
        "Speaker D: Sure.",
        "Speaker E: Hello, Candidate.",
    ]
    assert speakers == ["Examiner", "Me", "Speaker C", "S2", "Anna Smith"]
    source = tmp_path / "meeting.md"
    source.write_text(ROLES, encoding="utf-8")
    job = tmp_path / "job"
    assert secondary.store_transcript(job, source) == 5  # noqa: PLR2004 - every label is a speaker
    # Only the person's name goes to the answer check, so an answer may say "the candidate" or "me".
    assert sorted(secondary.removed_names(job)) == ["Anna", "Anna Smith", "Smith"]


@pytest.mark.parametrize(
    ("label", "names"),
    [
        ("Candidate A", []),
        ("IELTS Examiner", []),  # every answer's last line says "IELTS"
        ("The Examiner", []),
        ("Mock Examiner", []),
        ("Person 2", []),
        ("Candidate One", []),  # the prompt asks for "one practice exercise"
        ("Question", []),  # every answer has the heading "Data for the Fluentfolio answer check"
        ("Answer", []),
        ("Part One", []),  # every answer writes "Part"
        ("Test Taker", []),
        ("Examinee", []),
        ("Topic", []),
        ("Speaker Li", ["Li", "Speaker Li"]),  # one letter or up to two digits make a tag, not a short name
        ("Examiner Anna", ["Anna", "Examiner Anna"]),  # the role word of a person's label is no name alone
        ("An Nguyen", ["An", "An Nguyen", "Nguyen"]),  # a given name that is also a word is still a name
        ("My Linh", ["Linh", "My", "My Linh"]),
        ("An", ["An"]),
        ("Christopher Guest", ["Christopher", "Christopher Guest", "Guest"]),  # a role word after a name
        ("Anna Host", ["Anna", "Anna Host", "Host"]),
        # Headings of the test's parts: every answer's title says "Speaking", and sample answers say "card".
        ("Cue Card", []),
        ("Task Card", []),
        ("Speaking Test", []),
        ("Examiner's Question", []),
        ("Follow-up Questions", []),
        ("Follow Up", []),
        ("Warm-up", []),
        ("Long Turn", []),
        ("Round Two", []),
        ("Introduction", []),
        ("Interview", []),
        ("Practice", []),
        ("Prompt", []),
        ("Long", ["Long"]),  # a given name that only names a heading beside a heading word
        ("Long Nguyen", ["Long", "Long Nguyen", "Nguyen"]),
        # Names whose first letter is not one of A to Z.
        ("Şükrü", ["Şükrü"]),
        ("Émilie", ["Émilie"]),
        ("Łukasz", ["Łukasz"]),
        ("Ömer Faruk", ["Faruk", "Ömer", "Ömer Faruk"]),
        ("Ángel Özlem", ["Ángel", "Ángel Özlem", "Özlem"]),
        # Small words inside a name, names in scripts without capitals, and hyphenated names.
        ("Maria de Souza", ["Maria", "Maria de Souza", "Souza"]),
        ("Ludwig van Beethoven", ["Beethoven", "Ludwig", "Ludwig van Beethoven"]),
        ("Abdullah al-Rashid", ["Abdullah", "Abdullah al-Rashid", "Rashid", "al-Rashid"]),
        ("Mohammed bin Salman Al Saud", ["Al", "Mohammed", "Mohammed bin Salman Al Saud", "Salman", "Saud"]),
        ("Nguyen Van Anh", ["Anh", "Nguyen", "Nguyen Van Anh", "Van"]),  # a capital "Van" is a name
        ("Jean-Luc Picard", ["Jean", "Jean-Luc", "Jean-Luc Picard", "Luc", "Picard"]),
        ("李明", ["李明"]),
        ("王老师", ["王老师"]),
        ("محمد", ["محمد"]),
        # Headings of practice notes, whose words a graded answer writes.
        ("Date", []),
        ("Time", []),
        ("Model Answer", []),
        ("Sample Answer", []),
        ("Fluency", []),
        ("Lexical Resource", []),
        ("Pronunciation", []),
        ("Feedback", []),
        ("Session", []),
    ],
)
def test_a_role_with_a_tag_or_a_small_word_is_no_name(tmp_path: Path, label: str, names: list[str]) -> None:
    source = tmp_path / "meeting.md"
    source.write_text(f"{label}: Hello.\nMe: Hi, nice to meet you.\n", encoding="utf-8")
    secondary.store_transcript(tmp_path / "job", source)
    assert sorted(secondary.removed_names(tmp_path / "job")) == names
    stored = (tmp_path / "job" / "input" / secondary.TRANSCRIPT).read_text(encoding="utf-8")
    # The label is a speaker either way, so its line is stored under a neutral label.
    assert stored.split("\n") == ["Speaker A: Hello.", "Speaker B: Hi, nice to meet you."]


def test_a_short_name_after_speaker_is_taken_out_of_the_talk() -> None:
    text, _ = secondary.without_names("Speaker Li: Hello.\nCandidate A: Thanks, Li.\n")
    assert text.split("\n") == ["Speaker A: Hello.", "Speaker B: Thanks, Speaker A."]


def test_a_given_name_that_is_also_a_word_is_taken_out_of_the_talk() -> None:
    text, _ = secondary.without_names("An Nguyen: Hello.\nCandidate A: Thanks, An.\n")
    assert text.split("\n") == ["Speaker A: Hello.", "Speaker B: Thanks, Speaker A."]


def test_a_name_with_a_letter_outside_a_to_z_is_taken_out_of_the_talk() -> None:
    text, speakers = secondary.without_names("Şükrü Yılmaz: Hello.\nÉmilie: Thanks, Şükrü.\n")
    assert text.split("\n") == ["Speaker A: Hello.", "Speaker B: Thanks, Speaker A."]
    assert speakers == ["Şükrü Yılmaz", "Émilie"]


def test_a_heading_word_in_the_talk_stays_a_word() -> None:
    text, _ = secondary.without_names("Cue Card: Describe a place.\nCandidate A: The Card says a place.\n")
    assert text.split("\n") == ["Speaker A: Describe a place.", "Speaker B: The Card says a place."]


def test_a_particle_or_a_full_width_colon_still_takes_the_name_out_of_the_talk() -> None:
    text, speakers = secondary.without_names(
        "Maria de Souza: Hello.\n"
        "Anna：Thanks, Maria. The van is here, de facto.\n"
        "Abdullah al-Rashid: Hi, Souza and Anna. Rashid here.\n"
        "王芳： 王芳 too.\n"
    )
    assert text.split("\n") == [
        "Speaker A: Hello.",
        "Speaker B: Thanks, Speaker A. The van is here, de facto.",  # a particle alone stays a word
        "Speaker C: Hi, Speaker A and Speaker B. Speaker C here.",
        "Speaker D: Speaker D too.",
    ]
    assert speakers == ["Maria de Souza", "Anna", "Abdullah al-Rashid", "王芳"]


# Word, Google Docs and phones may type the apostrophe or the hyphen of a name in another form.
@pytest.mark.parametrize(
    ("name", "plain", "typed"),
    [
        ("Liam O\u2019Brien", "O'Brien", "O\u2019Brien"),
        ("Niamh O\u2018Neill", "O'Neill", "O\u2018Neill"),
        ("Sean O\u00b4Connor", "O'Connor", "O\u00b4Connor"),
        ("Ciara O`Shea", "O'Shea", "O`Shea"),
        ("Eoin O\uff07Kelly", "O'Kelly", "O\uff07Kelly"),
        ("Jean\u2010Luc Picard", "Jean-Luc", "Jean\u2010Luc"),
        ("Marie\u2011Claire Dubois", "Marie-Claire", "Marie\u2011Claire"),
        ("Anna Smith\u2013Jones", "Smith-Jones", "Smith\u2013Jones"),
    ],
)
def test_a_name_with_another_apostrophe_or_dash_is_taken_out_of_the_talk(
    tmp_path: Path, name: str, plain: str, typed: str
) -> None:
    source = tmp_path / "own.txt"
    source.write_text(
        f"{name}: I lived there.\nCandidate: Yes, {plain} and {typed} told me.\n", encoding="utf-8"
    )
    text, speakers = secondary.without_names(secondary.read_text(source, secondary.TRANSCRIPT_TYPES))
    assert text.split("\n") == [
        "Speaker A: I lived there.",
        "Speaker B: Yes, Speaker A and Speaker A told me.",
    ]
    assert speakers[1:] == ["Candidate"]


# A name goes with an apostrophe form after it or in quotes, however the apostrophe was typed; "Don't"
# stays a word, and a name inside another name stays.
@pytest.mark.parametrize(
    ("talk", "kept"),
    [
        ("Anna's sister and Smith's flat.", "Speaker A's sister and Speaker A's flat."),
        ("Anna\u2019s sister and Smith\u2019s flat.", "Speaker A's sister and Speaker A's flat."),
        ("James' book and Ward's pen.", "Speaker B' book and Speaker B's pen."),
        ("She said 'Anna' and \u2018James\u2019.", "She said 'Speaker A' and \u2018Speaker B\u2019."),
        ("Anna'll come, and Anna'd know.", "Speaker A'll come, and Speaker A'd know."),
        ("O\u2019Brien\u2019s and Marie\u2011Claire\u2019s cars.", "Speaker E's and Speaker F's cars."),
        (
            "Don't wait, Don\u2019t go, O'Neill and Lee's car.",
            "Don't wait, Don't go, O'Neill and Speaker C's car.",
        ),
    ],
)
def test_a_name_with_an_apostrophe_form_after_it_or_in_quotes_is_taken_out(
    tmp_path: Path, talk: str, kept: str
) -> None:
    source = tmp_path / "own.txt"
    people = [
        "Anna Smith",
        "James Ward",
        "Don Lee",
        "Neill Ray",
        "Liam O\u2019Brien",
        "Marie\u2011Claire Dubois",
    ]
    labels = "".join(f"{name}: Hello.\n" for name in people)
    source.write_text(f"{labels}Candidate: {talk}\n", encoding="utf-8")
    text, speakers = secondary.without_names(secondary.read_text(source, secondary.TRANSCRIPT_TYPES))
    assert text.split("\n")[-1] == f"Speaker G: {kept}"
    assert len(speakers) == 7 and speakers[-1] == "Candidate"  # noqa: PLR2004 - six people and the candidate


def test_only_an_apostrophe_or_a_dash_between_letters_changes(tmp_path: Path) -> None:
    source = tmp_path / "own.txt"
    kept = "Candidate: The word \u2018home\u2019, the students\u2019 rooms, 1990\u20132000 \u2013 all good."
    source.write_text(kept, encoding="utf-8")
    assert secondary.read_text(source, secondary.TRANSCRIPT_TYPES) == kept


# Known limit, pinned: a label in lower case is not taken for a name, nor one that ends in a particle,
# nor one after a time.
@pytest.mark.parametrize(
    "line",
    [
        "şükrü: I lived there.",
        "Şükrü yılmaz: I lived there.",
        "anna: I lived.",
        "Anna smith: I lived there.",
        "[00:03] Anna Smith: I lived there.",
        "Maria de: I lived there.",
        "de Souza: I lived there.",
        "So I do: it works.",
    ],
)
def test_a_label_word_must_start_with_a_capital(line: str) -> None:
    assert secondary.without_names(line) == (line, [])


# Known limit, pinned: a name goes only as its label writes it, and a name only said in the talk stays.
def test_a_name_in_lower_case_or_only_said_in_the_talk_stays() -> None:
    text, _ = secondary.without_names(
        "Anna Smith: Who do you trust?\nCandidate: My friend Sarah, and anna.\n"
    )
    assert text.split("\n") == ["Speaker A: Who do you trust?", "Speaker B: My friend Sarah, and anna."]


# Known limit, pinned: rarer forms keep the name, as docs/limitations.md says.
@pytest.mark.parametrize(
    "talk", ["Anna've been there.", "Ask l'Anna.", "The Smiths' house.", "She said 'Anna's'."]
)
def test_a_rarer_form_of_a_name_stays(talk: str) -> None:
    text, _ = secondary.without_names(f"Anna Smith: Hello.\nCandidate: {talk}\n")
    assert text.split("\n") == ["Speaker A: Hello.", f"Speaker B: {talk}"]


# Known limit, pinned: a combining mark above U+FFFF, as in this made-up Adlam name, ends the label word.
def test_a_label_with_a_mark_above_u_ffff_is_not_seen() -> None:
    name = "\U0001e900\U0001e922\U0001e944\U0001e922"
    talk = f"{name}: I lived there.\nCandidate: Yes, {name} did.\n"
    text, speakers = secondary.without_names(talk)
    assert speakers == ["Candidate"]
    assert text.splitlines()[0] == talk.splitlines()[0] and text.count(name) == 2


TALK = "Émilie Brown: Hello.\nCandidate A: Thanks, Émilie.\n"


@pytest.mark.parametrize(
    "data",
    [
        TALK.encode("utf-8-sig"),  # Notepad's "UTF-8 with BOM": the mark must not hide the first name
        TALK.encode("utf-16"),  # Notepad's and Word's "Unicode", with its byte-order mark
        codecs.BOM_UTF16_BE + TALK.encode("utf-16-be"),
        TALK.encode("cp1252"),  # not UTF-8: the replaced byte stays part of the name
    ],
)
def test_a_transcript_in_another_encoding_loses_its_names(tmp_path: Path, data: bytes) -> None:
    source = tmp_path / "meeting.txt"
    source.write_bytes(data)
    secondary.store_transcript(tmp_path / "job", source)
    stored = (tmp_path / "job" / "input" / secondary.TRANSCRIPT).read_text(encoding="utf-8")
    assert stored.split("\n") == ["Speaker A: Hello.", "Speaker B: Thanks, Speaker A."]


VIETNAMESE = unicodedata.normalize(
    "NFD", "Tr\u1ea7n Th\u1ecb H\u1ed3ng"
)  # decomposed, as some Mac input writes it
DEVANAGARI = (
    "\u092a\u094d\u0930\u093f\u092f\u093e \u0936\u0930\u094d\u092e\u093e"  # "Priya Sharma", vowel signs
)
ARABIC = "\u0645\u064f\u062d\u064e\u0645\u0651\u064e\u062f"  # "Muhammad" with its short-vowel marks


@pytest.mark.parametrize(
    "name", [VIETNAMESE, DEVANAGARI, ARABIC], ids=["vietnamese-nfd", "devanagari", "arabic"]
)
def test_a_name_with_combining_marks_is_taken_out_of_the_talk(tmp_path: Path, name: str) -> None:
    source = tmp_path / "meeting.txt"
    composed = unicodedata.normalize("NFC", name)  # the same name typed the other way in the talk
    source.write_text(f"{name}: Hello.\nCandidate A: Thanks, {composed}. And {name}.\n", encoding="utf-8")
    secondary.store_transcript(tmp_path / "job", source)
    stored = (tmp_path / "job" / "input" / secondary.TRANSCRIPT).read_text(encoding="utf-8")
    assert stored.split("\n") == ["Speaker A: Hello.", "Speaker B: Thanks, Speaker A. And Speaker A."]


@pytest.mark.parametrize("codec", ["utf-16-le", "utf-16-be", "utf-32"])
def test_a_text_with_nul_characters_is_refused(tmp_path: Path, codec: str) -> None:
    source = tmp_path / "meeting.txt"
    source.write_bytes(TALK.encode(codec))  # UTF-16 without its mark, or UTF-32: names would slip through
    with pytest.raises(InputError) as caught:
        secondary.store_transcript(tmp_path / "job", source)
    assert caught.value.code == "E_INPUT_TEXT"
    assert not (tmp_path / "job").exists()


@pytest.mark.parametrize(("number", "label"), [("1", True), ("12", True), ("100", False)])  # up to two digits
def test_a_label_may_end_in_a_number(number: str, label: bool) -> None:
    text, speakers = secondary.without_names(f"Candidate {number}: I lived there.\n")
    assert speakers == ([f"Candidate {number}"] if label else [])
    assert text == ("Speaker A: I lived there." if label else f"Candidate {number}: I lived there.")


@pytest.mark.parametrize(("extra", "allowed"), [(-1, True), (0, True), (1, False)])  # around 2 MB
def test_a_text_may_have_at_most_2_mb(tmp_path: Path, extra: int, allowed: bool) -> None:
    source = tmp_path / "questions.txt"
    source.write_bytes(b"a" * (2_000_000 + extra))  # 2 MB, written out so a changed limit fails
    if allowed:
        secondary.store_questions(tmp_path / "job", source)
        assert (tmp_path / "job" / "input" / secondary.QUESTIONS).stat().st_size == source.stat().st_size
    else:
        with pytest.raises(InputError) as caught:
            secondary.store_questions(tmp_path / "job", source)
        assert caught.value.code == "E_INPUT_TEXT"
        assert not (tmp_path / "job").exists()


@pytest.mark.parametrize(
    ("name", "kinds", "allowed"),
    [
        ("list.TXT", secondary.QUESTION_TYPES, True),
        ("list.vtt", secondary.QUESTION_TYPES, False),  # subtitles are a transcript, not a question list
        ("call.srt", secondary.TRANSCRIPT_TYPES, True),
        ("call.docx", secondary.TRANSCRIPT_TYPES, False),
        ("notes", secondary.TRANSCRIPT_TYPES, False),
    ],
)
def test_only_plain_text_types_are_taken(
    tmp_path: Path, name: str, kinds: tuple[str, ...], allowed: bool
) -> None:
    source = tmp_path / name
    source.write_bytes(b"caf\xe9 au lait")  # Latin-1, not UTF-8
    replaced = "caf" + chr(0xFFFD) + " au lait"  # the byte is replaced, never guessed
    if allowed:
        assert secondary.read_text(source, kinds) == replaced
    else:
        with pytest.raises(InputError) as caught:
            secondary.read_text(source, kinds)
        assert caught.value.code == "E_INPUT_TEXT"
