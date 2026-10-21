"""The structure step: roles, questions, Parts, the solo split and each candidate's own speech.

Every session here is made of words with chosen times, so each expected time comes from the test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fluentfolio.analyzers.measures import collect, fluency
from fluentfolio.asr.normalize import word
from fluentfolio.core.errors import InputError
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.time import inside, subtract, total, union
from fluentfolio.speakers.ownership import OVERLAP, overlaps
from fluentfolio.structure import build, overrides, parts, questions, roles

Word = dict[str, Any]
STEP = 400  # each test word starts 400 ms after the one before and lasts 350 ms
PAD = 100  # the voice detector's speech reaches 100 ms past each word
ANSWER = "I really like it because it is quiet and green and the people are kind."  # 15 words, 5950 ms
LONG = " ".join(["I read every evening because it helps me relax and forget my day."] * 10)  # 51 950 ms
QUESTION = "Do you like your home town?"  # 6 words, 2350 ms
CARD = "Now I'm going to give you a topic. You have one minute to prepare. Describe a book you enjoyed."


def said(speaker: str | None, text: str, start: int, step: int = STEP) -> list[Word]:
    """The words of one line, each `step` ms after the last and 50 ms shorter than that."""
    out = []
    for k, raw in enumerate(text.split()):
        w = word(raw, start + k * step, start + k * step + step - 50, None, "test")
        assert w is not None
        out.append({**w, "speaker": speaker, "chunk": 0})
    return out


def talk(*lines: tuple[str | None, str, int]) -> list[Word]:
    """Lines one after another, each starting the given gap (ms) after the previous line ended."""
    words: list[Word] = []
    for speaker, text, gap in lines:
        words += said(speaker, text, words[-1]["end_ms"] + gap if words else gap)
    return words


def job(
    tmp_path: Path, mode: str, words: list[Word], *, swap_style: str | None = None,
    segments: list[tuple[int, int]] | None = None, status: str = "ok", edits: dict[str, Any] | None = None,
    listed: str | None = None, stages: list[dict[str, Any]] | None = None, warnings: list[str] | None = None,
    cap: str | None = None,
) -> Path:  # fmt: skip
    """A job folder whose steps up to the words step are done, with speaker turns from the words."""
    path = tmp_path / "job"
    write_json(
        path / "job.json", {"mode": mode, "swap_style": swap_style, "stages": stages or [], "settings": {}}
    )
    write_json(path / "stages" / "media.json", {"duration_ms": words[-1]["end_ms"] + 2000})
    found = segments or union((max(0, w["start_ms"] - PAD), w["end_ms"] + PAD) for w in words)
    write_json(
        path / "stages" / "vad.json", {"segments": [list(s) for s in found], "speech_ms": total(found)}
    )
    write_json(path / "stages" / "words.json", {"words": words})
    talk_ms: dict[str, int] = {}
    for w in words:
        if w["speaker"]:
            talk_ms[w["speaker"]] = talk_ms.get(w["speaker"], 0) + w["end_ms"] - w["start_ms"]
    write_json(path / "stages" / "speakers.json", {
        "status": status, "backend": "pyannote-community-1", "warnings": warnings or [],
        "role_confidence_cap": cap,
        "speakers": [{"id": s, "kind": "main", "speech_ms": ms} for s, ms in talk_ms.items()],
        "turns": [{"start_ms": r.start_ms, "end_ms": r.end_ms, "speaker": r.speaker}
                  for r in roles.runs(words) if r.speaker],
    })  # fmt: skip
    if edits is not None:
        write_json(path / overrides.FILE, edits)
    if listed is not None:
        (path / "input").mkdir()
        (path / "input" / build.QUESTIONS).write_text(listed, encoding="utf-8")
    return path


def owners(out: dict[str, Any]) -> dict[int, str]:
    """Word index -> the candidate whose answer holds that word."""
    whose = {a["id"]: a["candidate"] for a in out["answers"]}
    return {
        i: whose[u["answer_id"]]
        for u in out["utterances"]
        for i in range(u["first_word"], u["last_word"] + 1)
    }


# Roles and turns


def turns_of(words: list[Word]) -> tuple[list[roles.Turn], list[str]]:
    found = roles.runs(words)
    pair = sorted({str(w["speaker"]) for w in words if w["speaker"]})
    roles.mark_backchannels(found, pair, words)
    return roles.turns(found, pair), pair


def test_a_short_run_beside_the_others_long_turn_is_a_backchannel_and_never_a_turn() -> None:
    words = talk(
        ("S1", QUESTION, 0), ("S2", ANSWER, 600), ("S1", "mm-hmm", 100), ("S2", "and lovely parks", 100)
    )
    found = roles.runs(words)
    roles.mark_backchannels(found, ["S1", "S2"], words)
    # 5950 ms + 1150 ms of the other person beside it; a 1150 ms run of 3 words is short too.
    assert [r.backchannel for r in found] == [False, False, True, False]
    ts = roles.turns(found, ["S1", "S2"])
    assert [(t.speaker, len(t.ids)) for t in ts] == [("S1", 6), ("S2", 18)]  # the answer joins across it


@pytest.mark.parametrize(
    ("line", "step", "other_ms", "expected"),
    [
        ("mm-hmm", STEP, 5000, True),  # 5000 ms of the other person: next to a long turn
        ("mm-hmm", STEP, 4950, False),
        ("you know what I", 240, 5950, True),  # 4 words in 910 ms: under 1 s
        ("you know what I", 270, 5950, False),  # 4 words in 1030 ms: not short
        ("right I see so", STEP, 5950, False),  # 4 words in 1550 ms
        ("yes I see", STEP, 5950, True),  # 3 words in 1150 ms
    ],
)
def test_backchannel_edges(line: str, step: int, other_ms: int, expected: bool) -> None:
    words = said("S2", " ".join(["word"] * (other_ms // STEP)), 0)
    words[-1]["end_ms"] = other_ms  # the other person's run lasts exactly other_ms
    words += said("S1", line, other_ms + 200, step)
    found = roles.runs(words)
    roles.mark_backchannels(found, ["S1", "S2"], words)
    assert found[1].backchannel is expected


def test_a_short_reply_to_a_question_is_an_answer_not_a_backchannel() -> None:
    words = talk(("S1", QUESTION, 0), ("S2", "Yes I do.", 600), ("S1", ANSWER, 600))
    found = roles.runs(words)
    roles.mark_backchannels(found, ["S1", "S2"], words)
    assert [r.backchannel for r in found] == [False, False, False]
    statement = talk(("S1", "That sounds nice.", 0), ("S2", "Yes I do.", 600), ("S1", ANSWER, 600))
    found = roles.runs(statement)
    roles.mark_backchannels(found, ["S1", "S2"], statement)
    assert found[1].backchannel  # after a statement, with 7 s of the other person beside it


def interview_turns() -> tuple[list[roles.Turn], list[Word]]:
    words = talk(("S1", QUESTION, 0), ("S2", ANSWER, 600), ("S1", "Why is that?", 600), ("S2", ANSWER, 600))
    ts, _ = turns_of(words)
    return ts, words


@pytest.mark.parametrize(
    ("talk_ms", "examiner", "confidence"),
    [
        ({"S1": 2000, "S2": 10000}, "S1", "high"),  # both signals agree and the examiner talks 17 %
        ({"S1": 4000, "S2": 6000}, "S1", "high"),  # 40 % exactly
        ({"S1": 4500, "S2": 5500}, "S1", "medium"),  # agree, but the examiner talks 45 %
        ({"S1": 7000, "S2": 3000}, "S1", "low"),  # the one who asks talks more: the signals disagree
        ({"S1": 5000, "S2": 5000}, "S1", "medium"),  # one signal only
    ],
)
def test_interview_roles_and_their_confidence(
    talk_ms: dict[str, int], examiner: str, confidence: str
) -> None:
    ts, words = interview_turns()
    found = roles.interview(ts, words, ["S1", "S2"], talk_ms)
    assert (found.examiner, found.candidate, found.confidence) == (examiner, "S2", confidence)


def test_without_any_question_the_quieter_person_examines_with_medium_or_low_confidence() -> None:
    words = talk(("S1", "That is nice.", 0), ("S2", ANSWER, 600), ("S1", "I see.", 600), ("S2", ANSWER, 600))
    ts, _ = turns_of(words)
    assert roles.interview(ts, words, ["S1", "S2"], {"S1": 1000, "S2": 9000}).confidence == "medium"
    tie = roles.interview(ts, words, ["S1", "S2"], {"S1": 5000, "S2": 5000})
    assert (tie.examiner, tie.confidence) == ("S1", "low")


@pytest.mark.parametrize(
    ("text", "likeness", "kind"),
    [
        ("Why?", 1.0, "question"),
        ("That's interesting. Why do you think so?", 1.0, "question"),
        ("Why do you think so? I see.", 0.5, "question"),
        ("Do you work or study", 1.0, "question"),  # a question opener without a question mark
        ("What I like most is reading.", 0.0, "other"),  # "what i" is no opener
        ("Now, let's talk about music.", 1.0, "prompt"),
        ("Sorry, could you repeat that?", 0.0, "question"),  # a short request to repeat never asks
        (CARD, 1.0, "cue_card"),
    ],
)
def test_how_much_a_turn_asks(text: str, likeness: float, kind: str) -> None:
    words = said("S1", text, 0)
    found = questions.asking(words, list(range(len(words))))
    assert (found.likeness, found.kind) == (likeness, kind)


def test_a_part_cue_names_its_part() -> None:
    def cue(text: str) -> int | None:
        words = said("S1", text, 0)
        return questions.part_cue(words, list(range(len(words))))

    assert cue("Now in part three we will discuss some general questions.") == 3  # noqa: PLR2004
    assert cue("We've been talking about books.") == 3  # noqa: PLR2004
    assert cue("I'm going to give you a topic.") == 2  # noqa: PLR2004
    assert cue("In your country, do people read books?") is None


def swap_words(contrary_after: tuple[int, ...] = (), answer: str = ANSWER) -> list[Word]:
    """S1 examines S2 for 4 exchanges, then S2 examines S1 for 4; a contrary exchange after the given ones."""
    lines: list[tuple[str | None, str, int]] = []
    for k in range(4):
        lines += [("S1", QUESTION, 600), ("S2", ANSWER, 600)]
        if k in contrary_after:
            lines += [("S2", QUESTION, 600), ("S1", answer, 600)]
    for _ in range(4):
        lines += [("S2", QUESTION, 600), ("S1", ANSWER, 600)]
    return talk(*lines)


def test_role_swap_once_finds_one_swap_and_every_answer_owner(tmp_path: Path) -> None:
    words = swap_words()
    out = build.run(job(tmp_path, "role-swap", words, swap_style="once"))
    one, two = out["candidates"]
    assert (one["id"], one["speaker"], one["heard_questions_first"]) == ("person1", "S2", False)
    assert (two["id"], two["speaker"], two["heard_questions_first"]) == ("person2", "S1", True)
    swap = next(w["start_ms"] for w in words[4 * 21 :] if w["speaker"] == "S2")  # S2's first question
    assert one["roles"] == {"confidence": "high", "method": "pyannote", "swap_times_ms": [swap]}
    # The old tool gave 4 of 8 answers to the wrong person; here no word is owned by the wrong one.
    truth = {i: "person1" if w["speaker"] == "S2" else "person2" for i, w in enumerate(words)}
    answered = {i for i in truth if i in owners(out)}
    assert {i: owners(out)[i] for i in answered} == {i: truth[i] for i in answered}
    assert sum(len(said("S", ANSWER, 0)) for _ in range(8)) == len(answered)
    assert [q["addressed_to"] for q in out["questions"]] == ["person1"] * 4 + ["person2"] * 4
    assert out["warnings"] == []


def test_a_long_detected_silence_puts_the_swap_at_the_question_after_an_unstopped_answer(
    tmp_path: Path,
) -> None:
    unstopped = [("S1", QUESTION, 600), ("S2", ANSWER[:-1], 600)]  # S2's last answer has no full stop
    lines = [("S1", QUESTION, 600), ("S2", ANSWER, 600)] * 3 + unstopped
    words = talk(*lines, *[("S2", QUESTION, 600), ("S1", ANSWER, 600)] * 4)
    end, ask = words[4 * 21 - 1]["end_ms"], words[4 * 21]["start_ms"]  # S2's last answer word, then "Do"
    segments = [(0, end - 200), (end + 800, words[-1]["end_ms"] + PAD)]  # 1 s of quiet around the 600 ms gap
    out = build.run(job(tmp_path, "role-swap", words, swap_style="once", segments=segments))
    assert out["candidates"][0]["roles"]["swap_times_ms"] == [ask]
    assert all(owners(out).get(i) == "person1" for i in range(4 * 21 - 15, 4 * 21))


def test_once_overrules_a_lone_contrary_exchange_and_per_question_follows_it(tmp_path: Path) -> None:
    once = build.run(job(tmp_path / "a", "role-swap", swap_words((1,)), swap_style="once"))
    assert len(once["candidates"][0]["roles"]["swap_times_ms"]) == 1 and once["warnings"] == []
    each = build.run(job(tmp_path / "b", "role-swap", swap_words((1,)), swap_style="per-question"))
    assert len(each["candidates"][0]["roles"]["swap_times_ms"]) == 3  # noqa: PLR2004 - away, back, swap
    contrary = each["answers"][2]  # S1 answering S2's question
    assert (contrary["candidate"], each["candidates"][1]["speaker"]) == ("person2", "S1")


def test_three_overruled_stretches_raise_frequent_swaps(tmp_path: Path) -> None:
    two = build.run(job(tmp_path / "a", "role-swap", swap_words((0, 1)), swap_style="once"))
    three = build.run(job(tmp_path / "b", "role-swap", swap_words((0, 1, 2)), swap_style="once"))
    assert two["warnings"] == [] and three["warnings"] == ["W_FREQUENT_SWAPS"]


def test_only_a_turn_that_asks_gets_a_vote_from_the_long_answer_after_it() -> None:
    # A request to repeat and a statement ask for nothing, so the long answers after them are no votes.
    # The question is word 44: 7 + 15 + 7 + 15 words come before it.
    words = talk(
        ("S1", "Sorry, could you say that again please?", 0), ("S2", ANSWER, 600),
        ("S1", "That is very interesting to hear about.", 600), ("S2", ANSWER, 600),
        ("S1", QUESTION, 600), ("S2", ANSWER, 600),
    )  # fmt: skip
    ts, _ = turns_of(words)
    assert [(e.asker, e.ask.ids[0], e.weight) for e in roles.exchanges(ts, words)] == [("S1", 44, 5.95)]


def test_once_undoes_a_phase_under_3_exchanges_and_90_s_but_per_question_keeps_it() -> None:
    lines: list[tuple[str | None, str, int]] = []
    for asker, answerer, n, answer in (("S1", "S2", 4, ANSWER), ("S2", "S1", 2, " ".join([ANSWER] * 5)),
                                       ("S1", "S2", 4, ANSWER), ("S2", "S1", 4, ANSWER)):  # fmt: skip
        lines += [(asker, QUESTION, 600), (answerer, answer, 600)] * n
    words = talk(*lines)
    ts, pair = turns_of(words)
    votes = roles.exchanges(ts, words)
    # An answer turn that ends with the next question counts whole (8.9 s); 30 s at most (30.0).
    assert [round(v.weight, 2) for v in votes[3:7]] == [8.9, 29.95, 30.0, 5.95]
    found, against = roles.phases(votes, pair, "once")
    # Two swaps (10) cost less than the two long answers (59.95): the smoother swaps, and "once" undoes the
    # phase, as it has 2 exchanges in under 90 s. S1's phase then agrees on 53.5 of 113.45.
    assert [(p.examiner, p.first, p.last, p.confidence) for p in found] == [
        ("S1", 0, 9, "low"),
        ("S2", 10, 13, "high"),
    ]
    assert against == 1
    each, _ = roles.phases(votes, pair, "per-question")
    assert [(p.examiner, p.first, p.last) for p in each] == [("S1", 0, 3), ("S2", 4, 5), ("S1", 6, 9),
                                                             ("S2", 10, 13)]  # fmt: skip


def middle_phase(answers: list[str], gap_back: int) -> tuple[list[roles.Exchange], list[str]]:
    """S1 asks 4 times, S2 asks once per given answer, then S1 asks 4 times again.

    `gap_back` is the silence between the last answer to S2 and S1's next question.
    """
    lines = [("S1", QUESTION, 600), ("S2", ANSWER, 600)] * 4
    for answer in answers:
        lines += [("S2", QUESTION, 600), ("S1", answer, 600)]
    lines += [("S1", QUESTION, gap_back), ("S2", ANSWER, 600)]
    lines += [("S1", QUESTION, 600), ("S2", ANSWER, 600)] * 3
    words = talk(*lines)
    ts, pair = turns_of(words)
    return roles.exchanges(ts, words), pair


def test_once_keeps_a_phase_of_exactly_3_exchanges_in_under_90_s() -> None:
    votes, pair = middle_phase([ANSWER] * 3, 600)
    found, _ = roles.phases(votes, pair, "once")
    assert [(p.examiner, p.first, p.last) for p in found] == [("S1", 0, 3), ("S2", 4, 6), ("S1", 7, 10)]


@pytest.mark.parametrize(("gap_back", "kept"), [(2699, False), (2700, True), (2701, True)])
def test_once_keeps_a_phase_of_2_exchanges_from_90_s_on(gap_back: int, kept: bool) -> None:
    # The phase runs from the start of S2's last answer (S2 then asks) to the end of S1's next question:
    # 5950 + 2 x (600 + 2350 + 600 + 35 950) + gap_back + 2350 = 87 300 + gap_back ms.
    votes, pair = middle_phase([" ".join([ANSWER] * 6)] * 2, gap_back)  # 90 words, 35 950 ms each
    found, _ = roles.phases(votes, pair, "once")
    expected = [("S1", 0, 3), ("S2", 4, 5), ("S1", 6, 9)] if kept else [("S1", 0, 9)]
    assert [(p.examiner, p.first, p.last) for p in found] == expected


def test_an_interview_whose_roles_change_for_good_warns(tmp_path: Path) -> None:
    out = build.run(job(tmp_path, "interview", swap_words()))
    assert out["warnings"] == ["W_ROLE_CHANGE"]
    steady = build.run(job(tmp_path / "b", "interview", swap_words()[:84]))  # the first 4 exchanges
    assert steady["warnings"] == []


# A whole interview: Part 1, the topic card, a minute to prepare, the long turn, a follow-up, Part 3


def interview_words(*, go: bool = True) -> list[Word]:
    """With go=False the examiner gives no start signal: the long turn starts 60 s after the card."""
    start = (
        [("S1", "Please start speaking now.", 60_000), ("S2", LONG, 800)] if go else [("S2", LONG, 60_000)]
    )
    return talk(
        ("S1", "Let's talk about your home town. " + QUESTION, 0),
        ("S2", ANSWER, 600),
        ("S1", "Why is that?", 600),
        ("S2", ANSWER, 600),
        ("S1", CARD, 600),
        *start,
        ("S1", "Thank you. Do you read often?", 500),
        ("S2", "Yes quite often.", 600),
        ("S1", "We've been talking about books. Why do people read less now?", 700),
        ("S2", ANSWER, 600),
        ("S1", "How could schools change that?", 600),
        ("S2", ANSWER, 600),
    )


def test_an_interview_gives_roles_questions_answers_and_parts(tmp_path: Path) -> None:
    words = interview_words()
    out = build.run(job(tmp_path, "interview", words))
    assert (out["status"], out["reason"], out["warnings"]) == ("ok", None, [])
    (c,) = out["candidates"]
    assert (c["id"], c["speaker"], c["roles"]["confidence"], c["roles"]["method"]) == (
        "candidate", "S2", "high", "pyannote",
    )  # fmt: skip
    kinds = ["question", "question", "cue_card", "question", "question", "question"]
    assert [q["kind"] for q in out["questions"]] == kinds
    assert {q["addressed_to"] for q in out["questions"]} == {"candidate"}
    assert [q["part_cue"] for q in out["questions"]] == [None, None, 2, None, 3, None]
    answers = out["answers"]
    assert [a["question_id"] for a in answers] == ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"]
    assert [a["part"] for a in answers] == [1, 1, 2, 2, 3, 3]
    assert [a["after_preparation"] for a in answers] == [False, False, True, False, False, False]
    # Every word of S2 is an answer word and no word of S1 is.
    assert set(owners(out)) == {i for i, w in enumerate(words) if w["speaker"] == "S2"}
    assert c["part2"] == {
        "prep_excluded_ms": 60_000,  # the silence between the card and "Please start"
        "long_turn_ms": 51_950 + 2 * PAD,
        "examiner_prompts_n": 1,
        "ended_by": "examiner_prompt",  # the examiner spoke 500 ms after the last word
    }
    q = {x["id"]: x for x in out["questions"]}
    assert [(p["part"], p["start_ms"], p["end_ms"], p["source"], p["confidence"]) for p in out["parts"]] == [
        (1, 0, answers[1]["end_ms"], "auto", "high"),
        (2, q["Q03"]["start_ms"], answers[3]["end_ms"], "auto", "high"),
        (3, q["Q05"]["start_ms"], answers[5]["end_ms"], "auto", "high"),
    ]
    # No other voice comes within 300 ms of an answer, so the candidate's speech is every answer span.
    assert out["speech"] == {"candidate": [[a["start_ms"], a["end_ms"]] for a in answers]}
    first = next(w for w in words if w["speaker"] == "S2")
    assert answers[0]["start_ms"] == first["start_ms"] - PAD  # out to the detected speech, 100 ms before


def test_the_minute_to_prepare_is_not_a_response_time(tmp_path: Path) -> None:
    path = job(tmp_path, "interview", interview_words(go=False))  # the long turn follows the card
    out = build.run(path)
    q = {x["id"]: x for x in out["questions"]}
    long_turn = out["answers"][2]
    assert long_turn["after_preparation"] and long_turn["start_ms"] - q["Q03"]["end_ms"] == 60_000 - PAD
    write_json(path / "stages" / "structure.json", out)
    write_json(path / "stages" / "asr.json", {"provenance": {"transcript_style": "verbatim-best-effort"}})
    values = collect.run(path)["candidates"]["candidate"]["scopes"]["all"]["values"]
    # Five answers start 600 ms after their question; the long turn's minute of silence is left out.
    assert (values["latency_median"]["value"], values["latency_median"]["n"]) == (0.6, 5)
    assert values["latency_over_10s"]["value"] == 0


def test_a_word_without_a_speaker_leaves_the_roles_found(tmp_path: Path) -> None:
    words = interview_words()
    words[1]["speaker"] = None  # one word the separation could not place
    out = build.run(job(tmp_path, "interview", words))
    assert out["status"] == "ok" and out["candidates"][0]["speaker"] == "S2"


def test_a_users_part_edit_replaces_the_parts_found(tmp_path: Path) -> None:
    words = interview_words()
    plain = build.run(job(tmp_path / "a", "interview", words))
    a = plain["answers"]
    edit = {"parts": {"candidate": {"part2": [a[1]["start_ms"], a[3]["end_ms"]]}}}
    out = build.run(job(tmp_path / "b", "interview", words, edits=edit))
    assert [x["part"] for x in out["answers"]] == [1, 2, 2, 2, 3, 3]
    assert {p["source"] for p in out["parts"]} == {"user"} and {p["confidence"] for p in out["parts"]} == {
        "high"
    }


def test_a_question_list_matches_and_names_what_was_not_asked(tmp_path: Path) -> None:
    listed = (
        "Part 1\n1. Do you like your home town?\n2. What do you do at weekends?\n"
        "Part 3\n- Why do people read less now?\n"
    )
    out = build.run(job(tmp_path, "interview", interview_words(), listed=listed))
    assert [q["listed"] for q in out["questions"]] == [0, None, None, None, 2, None]
    assert out["unmatched_questions"] == ["What do you do at weekends?"]
    # The headings label the answers they match; the long turn after the card is Part 2 by its timing.
    assert [a["part"] for a in out["answers"]] == [1, 1, 2, 2, 3, 3]
    assert {p["source"] for p in out["parts"]} == {"question_list"}


# The candidate's own speech (U11, U17)


@pytest.mark.parametrize("split_vad", [False, True])
def test_a_partners_mm_hmm_in_a_pause_leaves_one_pause_not_two(tmp_path: Path, split_vad: bool) -> None:
    words = said("S1", QUESTION, 0) + said("S2", "I like it a lot", 3050)  # ends at 5000
    words += said("S1", "mm-hmm", 5100, 350)  # 5100-5400
    words += said("S2", "because it is quiet and green and the people are kind and friendly", 5600)
    vad = [(0, 2400), (3000, 5000), (5100, 5400), (5600, 10800)] if split_vad else [(0, 2400), (3000, 10800)]
    out = build.run(job(tmp_path, "interview", words, segments=vad))
    (answer,) = out["answers"]
    assert (answer["start_ms"], answer["end_ms"]) == (3000, 10800)
    speech = out["speech"]["candidate"]
    assert speech == [[3000, 5000], [5600, 10800]]
    assert fluency.pauses([(3000, 10800)], [(a, b) for a, b in speech]) == [(5000, 5600)]  # one 0.6 s pause


def test_a_partners_voice_the_speech_model_missed_is_still_cut_from_the_pause(tmp_path: Path) -> None:
    words = said("S1", QUESTION, 0) + said("S2", "I like it a lot", 3050)  # ends at 5000
    words += said("S1", "mm", 5100, 150)  # only the first half of "mm-hmm" was recognised: 5100-5200
    words += said("S2", "because it is quiet and green and the people are kind and friendly", 5600)
    path = job(tmp_path, "interview", words, segments=[(0, 2400), (3000, 10800)])
    found = read_json(path / "stages" / "speakers.json")
    for t in found["turns"]:
        if t["start_ms"] == 5100:  # noqa: PLR2004 - the voice separation heard all of it
            t["end_ms"] = 5450
    write_json(path / "stages" / "speakers.json", found)
    speech = build.run(path)["speech"]["candidate"]
    assert speech == [[3000, 5000], [5600, 10800]]
    assert fluency.pauses([(3000, 10800)], [(a, b) for a, b in speech]) == [(5000, 5600)]


def test_the_other_persons_turn_is_never_the_candidates_pause(tmp_path: Path) -> None:
    words = talk(
        ("S1", QUESTION, 0), ("S2", " ".join([ANSWER] * 2), 600),
        ("S1", "Do you like music?", 600), ("S2", "Yes I do.", 600),
        ("S1", "I love music myself and I listen to it every day on my way to work and back home again in "
               "the evening. Why not?", 600),  # 25 words: 9950 ms
        ("S2", "Not really.", 600),
        ("S1", QUESTION, 600), ("S2", " ".join([ANSWER] * 2), 600),
    )  # fmt: skip
    out = build.run(job(tmp_path, "interview", words))
    assert out["candidates"][0]["speaker"] == "S2"
    assert [a["word_count"] for a in out["answers"]] == [30, 3, 2, 30]  # the short replies are answers
    spans = [(a["start_ms"], a["end_ms"]) for a in out["answers"]]
    speech = [(a, b) for a, b in out["speech"]["candidate"]]
    assert fluency.pauses(spans, speech) == []  # the old tool counted the 9 s turn as a hesitation
    turn = [w for w in words if w["speaker"] == "S1" and w["text"] in ("love", "not")]
    assert not inside(speech, [(turn[0]["start_ms"], turn[1]["end_ms"])])


def test_overlap_and_the_others_words_leave_the_candidates_speech() -> None:
    words = said("S1", QUESTION, 0) + said("S2", ANSWER, 2950)  # S2 speaks 2950-8900
    words += said("S1", "Sorry to stop you there what about your family", 7000)
    words.sort(key=lambda w: w["start_ms"])
    for w in words:
        if 7000 <= (w["start_ms"] + w["end_ms"]) / 2 < 8900:  # noqa: PLR2004 - both talk from 7.0 to 8.9 s
            w["flags"] = [OVERLAP]
    ids = {i for i, w in enumerate(words) if w["speaker"] == "S2"}
    answer = [{"start_ms": 2900, "end_ms": 8950}]
    speech = build._speech(words, answer, ids, [[2850, 10650]], [(7000, 8900)], [])
    # The last S2 word outside the overlap ends at 6900; S1's next word starts at 7000, within 300 ms.
    assert speech == [[2900, 6900]]
    clean = build._speech(words, answer, ids, [[2850, 10650]], [], [])
    assert clean == [[2900, 6900]]  # the 300 ms around S1's words alone gives the same edge here
    # Turns that take turns never overlap: the old tool found 0.73 and 0.49 overlap in such a case.
    assert overlaps([(0, 2350, "S1"), (2950, 8900, "S2")]) == []
    apart = said("S1", QUESTION, 0) + said("S2", ANSWER, 2950)
    spoken = build._speech(apart, answer, set(range(6, 21)), [[2850, 8950]], [], [])
    assert spoken == [[2900, 8950]]  # nothing is cut
    # The other voice in the overlap is often not recognised at all: the overlap alone must still cut it.
    for w in apart:
        if 7000 <= (w["start_ms"] + w["end_ms"]) / 2 < 8900:  # noqa: PLR2004
            w["flags"] = [OVERLAP]
    unheard = build._speech(apart, answer, set(range(6, 21)), [[2850, 8950]], [(7000, 8900)], [])
    assert unheard == [[2900, 7000], [8900, 8950]]


# Solo (U13, U17)


def solo_words(gap: int = 50) -> list[Word]:
    return talk(
        ("S1", "Do you like reading books and magazines in your free time or other things?", 0),
        ("S1", " ".join(["I read every evening because it helps me relax."] * 9), 450),
        ("S1", "What do you read most?", 1500),
        ("S1", ANSWER, 1000),
        ("S1", "Describe a book you enjoyed. You should say what it was.", 1500),
        ("S1", "Who wrote it.", gap),
        ("S1", "And explain why you liked it.", 50),
        ("S1", LONG, 60_000),
        ("S1", "Why do people read less now?", 1500),
        ("S1", " ".join([ANSWER] * 3), 1000),
    )


def test_solo_splits_questions_read_aloud_from_answers_with_no_word_wrong(tmp_path: Path) -> None:
    words = solo_words()
    truth = ["ask"] * 14 + ["answer"] * 81 + ["ask"] * 5 + ["answer"] * 15 + ["ask"] * 20 + ["answer"] * 130
    truth += ["ask"] * 6 + ["answer"] * 45
    assert len(truth) == len(words)
    out = build.run(job(tmp_path, "solo", words))
    roles_found = ["answer" if i in owners(out) else "ask" for i in range(len(words))]
    assert roles_found == truth  # the old tool had half the turns wrong
    assert {q["kind"] for q in out["questions"]} == {"read_aloud"}
    assert [q["cue_card"] for q in out["questions"]] == [False, False, True, False]
    assert [a["part"] for a in out["answers"]] == [1, 1, 2, 3]
    # From the last card line to where the detected speech of the answer starts, 100 ms before its first word.
    assert out["candidates"][0]["part2"]["prep_excluded_ms"] == 60_000 - PAD
    assert out["candidates"][0]["part2"]["examiner_prompts_n"] == 0


@pytest.mark.parametrize(("gap", "card_line"), [(2999, True), (3000, False)])
def test_a_topic_card_line_counts_only_within_3_s(tmp_path: Path, gap: int, card_line: bool) -> None:
    out = build.run(job(tmp_path, "solo", solo_words(gap)))
    who = [w for w in solo_words(gap) if w["text"] == "Who"]
    index = solo_words(gap).index(who[0])
    assert (index not in owners(out)) is card_line


@pytest.mark.parametrize("one_segment", [False, True])
def test_a_question_read_aloud_is_left_out_of_the_candidates_speech(
    tmp_path: Path, one_segment: bool
) -> None:
    words = said("S1", "Do you like reading books and magazines in your free time or other things?", 0)
    words += said("S1", " ".join(["I read every evening because it helps me relax."] * 9), 6000)
    end = words[-1]["end_ms"]
    vad = [(0, 40_000)] if one_segment else [(0, 5600), (5950, 40_000)]
    out = build.run(job(tmp_path, "solo", words, segments=vad))
    assert words[13]["end_ms"] == 5550 and end == 38_350  # noqa: PLR2004 - the question ends at 5.55 s
    (speech,) = out["speech"]["candidate"]
    # 300 ms after the question's last word at the earliest, and never past the answer's speech.
    assert 5850 <= speech[0] <= 6000 and speech[1] == 40_000  # noqa: PLR2004
    (question,) = out["questions"]
    assert (question["kind"], question["end_ms"]) == ("read_aloud", 5550)


def test_with_a_question_list_only_listed_sentences_are_read_aloud() -> None:
    words = said("S1", "Your favourite food. Why do I like it? Because it is spicy and warm.", 0)
    listed = questions.parse_list("What is your favourite food?")
    blocks = questions.solo_split(words, listed)
    assert [(role, len(ids)) for role, ids in blocks] == [("read_aloud", 3), ("answer", 11)]
    plain = questions.solo_split(words)
    assert [role for role, _ in plain] == ["answer", "read_aloud", "answer"]


@pytest.mark.parametrize(("gap", "apart"), [(999, False), (1000, True), (1001, True)])
def test_a_silence_of_1_s_ends_a_sentence_that_has_no_full_stop(gap: int, apart: bool) -> None:
    words = talk(("S1", "My sister and I cook on Sundays", 0), ("S1", "What do you like to cook?", gap))
    blocks = [(role, len(ids)) for role, ids in questions.solo_split(words)]
    assert blocks == ([("answer", 7), ("read_aloud", 6)] if apart else [("read_aloud", 13)])


@pytest.mark.parametrize(("quiet", "apart"), [(999, False), (1000, True), (1001, True)])
def test_a_detected_silence_of_1_s_ends_a_sentence_when_the_word_times_run_into_it(
    quiet: int, apart: bool
) -> None:
    words = talk(("S1", "My sister and I cook on Sundays", 0), ("S1", "What do you like to cook?", 600))
    end = words[6]["end_ms"]  # the speech model stretches "Sundays" 200 ms into the silence
    silences = questions.long_silences([(0, end - 200), (end - 200 + quiet, words[-1]["end_ms"])])
    blocks = [(role, len(ids)) for role, ids in questions.solo_split(words, None, silences)]
    assert blocks == ([("answer", 7), ("read_aloud", 6)] if apart else [("read_aloud", 13)])


def test_a_word_inside_a_detected_silence_starts_the_sentence_after_the_silence() -> None:
    words = talk(
        ("S1", "It was cheaper.", 0), ("S1", "What", 700), ("S1", "do you usually do at the weekend?", 160),
    )  # fmt: skip
    # The detector heard "What" as quiet: one silence covers the gap before it, the word and the gap after it.
    quiet_from, quiet_to = words[2]["end_ms"] - 300, words[3]["end_ms"] + 140
    silences = questions.long_silences([(0, quiet_from), (quiet_to, words[-1]["end_ms"])])
    blocks = [(role, len(ids)) for role, ids in questions.solo_split(words, None, silences)]
    assert blocks == [("answer", 3), ("read_aloud", 8)]


@pytest.mark.parametrize(
    ("lines", "quiet", "split"),
    [
        # "cheaper." ends at 1150, "What" is 1550-1900, "do" starts at 2800. The silence 850-2000 covers all
        # 400 ms of the gap before "What" and 100 of the 900 ms after it: the sentence ends before "What".
        ((("S1", "It was cheaper.", 0), ("S1", "What", 400), ("S1", "do you usually do at weekends?", 900)),
         (850, 2000), [("answer", 3), ("read_aloud", 7)]),
        # "on" ends at 1150, "Sundays" is 2050-2400, "Do" starts at 2800. The silence 1950-2950 covers 100 of
        # the 900 ms gap before "Sundays" and all 400 ms after it: the sentence ends before "Do".
        ((("S1", "I cook on", 0), ("S1", "Sundays", 900), ("S1", "Do you like it?", 400)),
         (1950, 2950), [("answer", 4), ("read_aloud", 4)]),
    ],
)  # fmt: skip
def test_a_silence_ends_the_sentence_at_the_gap_it_covers_most_not_the_widest_gap(
    lines: tuple[tuple[str, str, int], ...], quiet: tuple[int, int], split: list[tuple[str, int]]
) -> None:
    words = talk(*lines)
    silences = questions.long_silences([(0, quiet[0]), (quiet[1], words[-1]["end_ms"])])
    assert [(role, len(ids)) for role, ids in questions.solo_split(words, None, silences)] == split


@pytest.mark.parametrize(
    ("before", "after", "split"),
    [
        ("I usually cook", "how do you say stir fry at weekends.", ["answer"]),
        ("We eat a lot of", "what is it called dumplings.", ["answer"]),
        ("I usually cook", "how do you spend your weekends?", ["answer", "read_aloud"]),
        ("I usually cook", "Describe a meal you enjoyed.", ["answer", "read_aloud"]),
        ("I usually cook.", "How do you spend your weekends", ["answer", "read_aloud"]),
    ],
)
def test_after_an_unstopped_pause_a_question_opener_alone_is_a_word_search(
    before: str, after: str, split: list[str]
) -> None:
    words = talk(("S1", before, 0), ("S1", after, 1300))
    assert [role for role, _ in questions.solo_split(words)] == split
    likeness = questions.asking(words, list(range(len(words)))).likeness
    assert likeness == (1.0 if split[-1] == "read_aloud" else 0.0)


def test_the_solo_structure_cuts_at_the_detected_silences(tmp_path: Path) -> None:
    words = talk(
        ("S1", "My sister and I cook on Sundays", 0), ("S1", "What do you like to cook?", 600),
        ("S1", "I like to cook rice.", 1500),
    )  # fmt: skip
    end = words[6]["end_ms"]
    segments = [(0, end - 200), (end + 800, words[-1]["end_ms"] + PAD)]
    out = build.run(job(tmp_path, "solo", words, segments=segments))
    assert sorted(owners(out)) == [*range(7), *range(13, 18)]


def test_a_detected_silence_cuts_only_between_the_words_it_touches() -> None:
    words = talk(("S1", "I cook", 0), ("S1", "on Sundays What do you like to cook?", 800))
    # 1.05 s of quiet between "cook" and "on", none before "What": the question still takes "on Sundays".
    end = words[1]["end_ms"]
    silences = questions.long_silences([(0, end - 100), (end + 950, words[-1]["end_ms"])])
    blocks = [(role, len(ids)) for role, ids in questions.solo_split(words, None, silences)]
    assert blocks == [("answer", 2), ("read_aloud", 8)]


def test_the_new_examiner_asks_from_the_first_word_after_a_long_detected_silence() -> None:
    words = talk(("S1", "We cook on Sundays", 0), ("S1", "Do you like cooking?", 500))
    ids = list(range(len(words)))
    speech = [(0, words[3]["end_ms"] - 300), (words[4]["start_ms"] + 300, words[-1]["end_ms"])]
    assert questions.ask_from(words, ids) == 0
    assert questions.ask_from(words, ids, questions.long_silences(speech)) == 4  # noqa: PLR2004 - "Do"


def test_a_question_list_reads_headings_and_list_marks() -> None:
    text = (
        "**Part one**\n1) Do you work or study?\n- What do you like about it?\n\n---\n"
        "PART 2:\nDescribe a city.\n"
    )
    assert questions.parse_list(text) == [
        questions.Listed("Do you work or study?", 1),
        questions.Listed("What do you like about it?", 1),
        questions.Listed("Describe a city.", 2),
    ]


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("what is your favourite", 0),  # 4 of its 5 words
        ("is your favourite", 0),  # 3 of 5: a share of 0.6
        ("your favourite", None),  # 2 of 5: 0.4
        ("why", None),  # all of "Why?", but one word is never enough
        ("like music", None),  # 2 of the 4 words of "Do you like music?": 0.5
        ("you like music", 2),  # 3 of 4: 0.75
    ],
)
def test_a_listed_question_matches_at_a_share_of_0_6_and_two_words(heard: str, expected: int | None) -> None:
    listed = questions.parse_list("What is your favourite food?\nWhy?\nDo you like music?")
    assert questions.match(listed, heard.split()) == expected


# Parts


def step(answer_ms: int, *, start: int = 0, card: bool = False, cue: int | None = None,
         listed: int | None = None, silence: int = 0) -> parts.Step:  # fmt: skip
    return parts.Step(start, start, start + answer_ms, card, cue, listed, silence)


def at(*spans: parts.Step) -> list[parts.Step]:
    """Steps placed one after another with 1 s between them."""
    out, t = [], 0
    for s in spans:
        out.append(parts.Step(t, t, t + s.answer_ms, s.cue_card, s.part_cue, s.listed_part, s.silence_ms))
        t += s.answer_ms + 1000
    return out


def test_no_long_answer_or_no_cue_leaves_every_part_unknown() -> None:
    short = parts.detect(at(step(20_000), step(44_999, card=True, silence=60_000)))
    assert (short.labels, short.confidence, short.anchor) == (["unknown", "unknown"], "low", None)
    bare = parts.detect(at(step(20_000), step(60_000, silence=14_999)))
    assert bare.labels == ["unknown", "unknown"]  # a long answer alone is not trusted


@pytest.mark.parametrize(
    ("card", "silence", "confidence"),
    [(True, 15_000, "high"), (True, 0, "medium"), (False, 15_000, "medium")],
)
def test_the_long_turn_after_a_card_or_a_silence_is_part_2(card: bool, silence: int, confidence: str) -> None:
    found = parts.detect(at(step(10_000), step(45_000, card=card, silence=silence), step(20_000)))
    assert (found.labels, found.confidence, found.anchor, found.source) == ([1, 2, 3], confidence, 1, "auto")


def test_short_follow_ups_stay_in_part_2_up_to_two_and_a_cue_ends_them() -> None:
    def labels(*after: parts.Step) -> list[int | str]:
        return parts.detect(at(step(10_000), step(60_000, card=True), *after)).labels

    assert labels(step(14_999), step(5000), step(5000)) == [1, 2, 2, 2, 3]
    assert labels(step(15_000), step(5000)) == [1, 2, 3, 3]
    assert labels(step(5000, cue=3), step(5000)) == [1, 2, 3, 3]
    first = at(step(10_000), step(60_000, card=True))
    for after, label in ((45_000, 2), (45_001, 3)):  # how long after the long turn the short answer starts
        start = first[1].end_ms + after
        assert parts.detect([*first, step(5000, start=start)]).labels == [1, 2, label]


def test_a_cue_moves_a_boundary_by_one_exchange_and_never_breaks_the_order() -> None:
    before = parts.detect(at(step(10_000), step(10_000, cue=2), step(60_000, card=True), step(20_000)))
    assert before.labels == [1, 2, 2, 3]
    # A Part 3 cue in Part 1 (keyword rules alone were not trusted before) leaves the timing's order.
    early = parts.detect(at(step(10_000, cue=3), step(10_000), step(60_000, card=True), step(20_000)))
    assert early.labels == [1, 1, 2, 3]


@pytest.mark.parametrize(("between", "anchored"), [(14_999, True), (15_000, False), (15_001, False)])
def test_a_card_short_exchanges_before_the_long_turn_still_anchors_it(between: int, anchored: bool) -> None:
    # A question about the card or "can I start?" comes between; a full answer means the card was for it.
    found = parts.detect(at(step(10_000), step(5000, card=True), step(between), step(60_000), step(20_000)))
    if anchored:
        assert (found.labels, found.anchor, found.confidence) == ([1, 2, 2, 2, 3], 3, "medium")
    else:
        assert (found.labels, found.anchor) == (["unknown"] * 5, None)


def test_a_question_about_the_card_keeps_the_long_turn_in_part_2(tmp_path: Path) -> None:
    words = talk(
        ("S1", "Let's talk about your home town. " + QUESTION, 0),
        ("S2", ANSWER, 600),
        ("S1", CARD, 600),
        ("S2", "Sorry, can it be any book?", 600),
        ("S1", "Yes, any book.", 600),
        ("S1", "Please begin.", 10_000),  # under 15 s of preparation: only the card can anchor the long turn
        ("S2", LONG, 800),
        ("S1", "We've been talking about books. Why do people read less now?", 500),
        ("S2", ANSWER, 600),
    )
    out = build.run(job(tmp_path, "interview", words))
    answers = out["answers"]
    assert [a["part"] for a in answers] == [1, 2, 2, 3]
    (c,) = out["candidates"]
    long_turn = answers[2]["end_ms"] - answers[2]["start_ms"]
    assert c["part2"] is not None and c["part2"]["long_turn_ms"] == long_turn
    q = {x["id"]: x for x in out["questions"]}
    assert [p["part"] for p in out["parts"]] == [1, 2, 3]
    assert out["parts"][1]["start_ms"] == q["Q02"]["start_ms"]  # Part 2 starts with the card


def test_listed_headings_come_first_and_an_earlier_part_never_returns() -> None:
    found = parts.detect(at(step(5000, listed=1), step(5000), step(5000, listed=2), step(5000, listed=1)))
    assert (found.labels, found.source, found.confidence) == ([1, 1, 2, 2], "question_list", "high")


def test_part_spans_run_from_the_first_question_to_the_last_answer() -> None:
    steps = [
        parts.Step(0, 2000, 8000, False, None, None, 0),
        parts.Step(9000, 10_000, 16_000, False, None, None, 0),
        parts.Step(17_000, 80_000, 140_000, True, None, None, 0),
    ]
    assert parts.spans(steps, [1, 1, 2]) == [(1, 0, 16_000), (2, 17_000, 140_000)]
    assert parts.spans(steps, ["unknown", "unknown", "unknown"]) == []


@pytest.mark.parametrize(
    ("speaker", "gap", "expected"),
    [
        ("", 0, "stopped"),  # no word follows, and the recording goes on for 2 s
        ("S1", 999, "examiner_prompt"),
        ("S1", 1000, "stopped"),
        ("S1", -100, "cut_off"),  # the examiner started before the last word ended
        ("S2", 300, "stopped"),  # the candidate goes on
        (None, 300, "unknown"),  # the next word has no speaker
    ],
)
def test_how_the_long_turn_ended(speaker: str | None, gap: int, expected: str) -> None:
    words = said("S2", "and that is all", 0)
    if speaker != "":
        words += said(speaker, "Thank you.", words[-1]["end_ms"] + gap)
    assert build._ended(words, 3, "S2", words[-1]["end_ms"] + 2000) == expected
    assert build._ended(words[:4], 3, "S2", words[3]["end_ms"] + 499) == "cut_off"  # the recording ended


# Corrections from the review form


def test_a_part_edit_is_checked_and_labels_answers_by_their_start() -> None:
    with pytest.raises(ValueError, match="start before"):
        overrides.PartEdit(part2=[5000, 3000])  # start > end is refused
    with pytest.raises(ValueError, match="Part 3"):
        overrides.PartEdit(part2=[3000, 5000], part3_start_ms=4000)
    edit = overrides.PartEdit(part2=[3000, 5000], part3_start_ms=7000)
    assert [edit.label(t) for t in (2999, 3000, 4999, 5000, 6999, 7000)] == [1, 2, 2, 3, 3, 3]


def test_unreadable_overrides_are_refused(tmp_path: Path) -> None:
    assert overrides.load(tmp_path) == overrides.Overrides()
    for bad in ({"swap_people": "yes"}, {"colour": "red"}, {"parts": {"candidate": {"part2": [9, 1]}}}):
        write_json(tmp_path / overrides.FILE, bad)
        with pytest.raises(InputError) as caught:
            overrides.load(tmp_path)
        assert caught.value.code == "E_OVERRIDES_INVALID"


def test_swapped_people_and_set_swap_times_replace_what_was_found(tmp_path: Path) -> None:
    words = swap_words()
    edits = {"swap_people": True, "swap_times_ms": [30_000, 60_000]}
    out = build.run(job(tmp_path, "role-swap", words, swap_style="once", edits=edits))
    one, two = out["candidates"]
    assert (one["speaker"], one["roles"]["method"], one["roles"]["swap_times_ms"]) == (
        "S1",
        "user",
        [30_000, 60_000],
    )
    assert two["speaker"] == "S2" and {c["roles"]["confidence"] for c in out["candidates"]} == {"high"}


# Roles that cannot be found, and confidence caps


def test_without_speakers_roles_are_unavailable_and_nothing_is_guessed(tmp_path: Path) -> None:
    words = interview_words()
    for w in words:
        w["speaker"] = None
    out = build.run(job(tmp_path, "interview", words, status="failed"))
    assert (out["status"], out["reason"], out["candidates"], out["answers"]) == (
        "partial",
        build.NO_ROLES,
        [],
        [],
    )
    assert out["warnings"] == ["W_ROLES_UNAVAILABLE"] and out["unknown_candidate"] == build.UNKNOWN_CANDIDATE
    assert [t["speaker"] for t in out["transcript"]] == [None]
    assert out["transcript"][0]["text"].startswith("Let's talk about")


def test_one_voice_gives_no_roles_unless_the_user_names_it(tmp_path: Path) -> None:
    words = said("S2", " ".join([ANSWER] * 4), 0)
    out = build.run(job(tmp_path / "a", "interview", words))
    assert out["warnings"] == ["W_ROLES_UNAVAILABLE"]
    told = build.run(job(tmp_path / "b", "interview", words, edits={"one_voice": "candidate"}))
    (c,) = told["candidates"]
    assert (c["speaker"], c["roles"]["method"], len(told["answers"])) == ("S2", "user", 1)


@pytest.mark.parametrize(
    ("stages", "warnings", "cap", "expected"),
    [
        ([], [], None, "high"),
        (
            [{"id": "speakers", "fallback": {"to": "speakers-sherpa", "after": "E_MODEL_MISSING"}}],
            [],
            None,
            "medium",
        ),
        ([{"id": "speakers", "fallback": {"to": "cpu", "after": "E_GPU_OOM"}}], [], None, "high"),
        ([], ["W_THIRD_VOICE"], None, "medium"),
        ([], [], "medium", "medium"),
    ],
)
def test_role_confidence_is_capped_after_a_fallback_or_an_odd_voice_count(
    tmp_path: Path, stages: list[dict[str, Any]], warnings: list[str], cap: str | None, expected: str
) -> None:
    out = build.run(job(tmp_path, "interview", interview_words(), stages=stages, warnings=warnings, cap=cap))
    assert out["candidates"][0]["roles"]["confidence"] == expected


@pytest.mark.parametrize("same_voices", [False, True])
def test_the_same_voices_on_both_tracks_leave_low_roles_and_a_partial_structure(
    tmp_path: Path, same_voices: bool
) -> None:
    path = job(tmp_path, "interview", interview_words())
    vad = read_json(path / "stages" / "vad.json")
    write_json(path / "stages" / "vad.json", {**vad, "warnings": ["W_TRACK_BLEED"] if same_voices else []})
    out = build.run(path)
    assert out["candidates"][0]["roles"]["confidence"] == ("low" if same_voices else "high")
    assert out["status"] == ("partial" if same_voices else "ok")
    assert (out["reason"] == build.SAME_VOICES) is same_voices


# Time spans


def test_time_spans_merge_cut_and_keep() -> None:
    assert union([(5, 9), (0, 3), (3, 4), (8, 12)]) == [(0, 4), (5, 12)]
    assert inside([(0, 10), (20, 30)], [(5, 25)]) == [(5, 10), (20, 25)]
    assert subtract([(0, 10)], [(2, 3), (5, 7), (9, 20)]) == [(0, 2), (3, 5), (7, 9)]
    assert subtract([(0, 10)], [(10, 12), (-5, 0)]) == [(0, 10)]
    assert total([(0, 4), (2, 6), (10, 11)]) == 7  # noqa: PLR2004
    assert subtract([(0, 10)], []) == [(0, 10)] and inside([(0, 10)], []) == []
