"""The files in export/details/: transcript, README and openSMILE's CSV, against hand-written inputs."""

from __future__ import annotations

from typing import Any

from fluentfolio.export.details import AUDIO, PARTNER, UNTRUSTED, egemaps, readme, transcript


def evidence(mode: str = "interview") -> dict[str, Any]:
    turn = {"end_ms": 900, "addressed_to": "candidate"}
    said = {"answer_id": "A01", "start_ms": 1_250, "text": "I like_it [pause 0.6 s] a lot."}
    return {
        "schema_version": "1.0.0",
        "manifest": {
            "job_id": "20261013-120000-abcd", "export_id": "c0ffee", "tool_version": "0.1.0",
            "mode": mode, "asr": {"model": "parakeet-tdt-0.6b-v2"},
        },
        "candidates": [{
            "id": "candidate", "label": "the candidate",
            "answers": [{"id": "A01", "part": 1}, {"id": "A02", "part": "unknown"}],
            "utterances": [
                {"id": "U002", "answer_id": "A02", "start_ms": 9_000, "text": "Maybe <b>bold</b> *stars*."},
                {"id": "U001", **said},
            ],
        }],
        "examiner_turns": [
            {**turn, "id": "Q02", "start_ms": 8_000, "kind": "question", "text": "And now?"},
            {**turn, "id": "Q01", "start_ms": 500, "kind": "read_aloud", "text": "Do you like it?"},
        ],
        "disclaimer": "Fluentfolio measures a practice recording and gives no band.",
    }  # fmt: skip


HEAD = [
    "# Transcript",
    "",
    "Job 20261013-120000-abcd, export c0ffee. Times are from the original recording. The words are the "
    "speech model's (parakeet-tdt-0.6b-v2), so some may be misheard.",
    "",
]


def test_the_transcript_lists_questions_and_answers_in_time_order_with_markup_escaped() -> None:
    assert transcript(evidence(), []).split("\n") == [
        *HEAD,
        "- `00:00.5` **Q01** question read aloud: Do you like it?",
        "- `00:01.2` **U001** the candidate (A01, Part 1): I like\\_it [pause 0.6 s] a lot.",
        "- `00:08.0` **Q02** examiner: And now?",
        "- `00:09.0` **U002** the candidate (A02, Part unknown): Maybe \\<b\\>bold\\</b\\> \\*stars\\*.",
        "",
    ]


def test_with_no_candidate_the_transcript_shows_who_spoke_when_cleaned() -> None:
    none = {**evidence(), "candidates": []}
    runs = [
        {"speaker": "S1", "start_ms": 0, "text": "Hel\u200blo | there"},
        {"speaker": None, "start_ms": 61_000, "text": "Hi."},
    ]
    assert transcript(none, runs).split("\n")[len(HEAD) :] == [
        "- `00:00.0` **-** S1: Hello / there",
        "- `01:01.0` **-** unknown voice: Hi.",
        "",
    ]


def test_the_readme_names_only_the_files_there_and_the_partner_line_only_with_a_partner() -> None:
    files = ["export/for_ai/for_ai.txt", "export/details/evidence.json", "export/details/transcript.md"]
    lines = readme(evidence("interview"), files).split("\n")
    assert lines[1] == (
        "job 20261013-120000-abcd | export c0ffee | tool 0.1.0 | grading prompt v0.2.0 | "
        "evidence schema 1.0.0"
    )
    assert lines[4].startswith("Open for_ai/for_ai.txt, copy all of it")
    assert UNTRUSTED in lines and PARTNER in lines
    at = lines.index("Files")
    assert lines[at + 1 : at + 6] == [
        "for_ai/for_ai.txt: the grading prompt and the evidence for one person",
        "details/README.txt: this file",
        "details/evidence.json: every measure with its unit, the span it covers, its source tool and how "
        "reliable it is",
        "details/transcript.md: the whole transcript, with times from the original recording",
        "fluentfolio_20261013-120000-abcd.zip: the texts and details/ in one file; it never holds audio",
    ]
    assert "Fluentfolio measures a practice recording and gives no band." in lines
    solo = readme(evidence("solo"), [*files, "export/details/egemaps_88.csv"])
    assert PARTNER not in solo and "details/egemaps_88.csv: openSMILE's 88 voice values" in solo
    assert solo.isascii()


def test_the_readme_of_a_role_swap_names_both_texts() -> None:
    files = [
        "export/for_ai/for_ai_person2.txt", "export/for_ai/for_ai_person1.txt",
        "export/for_ai/for_ai_audio_person1.m4a",
    ]  # fmt: skip
    lines = readme(evidence("role-swap"), files).split("\n")
    # The audio file sorts before the texts, and still the first person's text is the one to open.
    assert lines[4].startswith("Open for_ai/for_ai_person1.txt,")
    assert sum(x.endswith(": the grading prompt and the evidence for one person") for x in lines) == 2
    assert f"for_ai/for_ai_audio_person1.m4a: {AUDIO}" in lines


def test_the_egemaps_csv_has_one_row_per_person_and_span_and_none_without_opensmile() -> None:
    values = {"F0semitoneFrom27.5Hz_sma3nz_amean": 25.5, "loudness_sma3_amean": 0.75}
    found = {"status": "ok", "candidates": {
        "person1": {"functionals": {"part1": values, "all": {k: v * 2 for k, v in values.items()}}},
        "person2": {"functionals": {}},
    }}  # fmt: skip
    assert egemaps(found) == (
        "candidate,span,F0semitoneFrom27.5Hz_sma3nz_amean,loudness_sma3_amean\n"
        "person1,part1,25.5,0.75\n"
        "person1,all,51.0,1.5\n"
    )
    assert egemaps({"status": "disabled", "candidates": {}}) is None
    assert egemaps({"status": "ok", "candidates": {"candidate": {"functionals": {}}}}) is None
