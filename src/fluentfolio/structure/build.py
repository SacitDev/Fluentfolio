"""The structure step: the candidates, what was said to them, their answers and Parts (stages/structure.json).

Solo splits one voice into questions read aloud and answers; interview and role-swap find who asks and who
answers. Each candidate's speech spans hold only their own voice, the input of every candidate measure:
other voices with 300 ms around them, overlap, and questions read aloud are left out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fluentfolio.core.enums import Mode
from fluentfolio.core.files import read_json
from fluentfolio.core.time import inside, subtract, union
from fluentfolio.speakers.ownership import OVERLAP, overlaps
from fluentfolio.structure import overrides, parts, questions, roles
from fluentfolio.structure.questions import Listed, Word

Row = dict[str, Any]
PARTNER_MS = 300  # another voice this close is not the candidate's speech, so "mm-hmm" leaves one pause
CUT_OFF_MS = 500  # a long turn that ends this close to the end of the recording was cut off
STEPPED_IN_MS = 1000  # the examiner speaking this soon after the long turn ended it
LEVELS = ("low", "medium", "high")
LABELS = {"candidate": "the candidate", "person1": "Person 1", "person2": "Person 2"}
SOURCES = {"pyannote-community-1": "pyannote", "sherpa-onnx-diarizer": "sherpa", "tracks": "tracks"}
QUESTIONS = "questions.txt"  # the optional question list, in the job's input folder
NO_ROLES = (
    "Roles are unavailable: only one voice was found, or the voices could not be told apart, so no speech "
    "counts as a candidate's answer and no candidate measure was made (roles_unavailable)."
)
UNKNOWN_CANDIDATE = "The candidate is not known. Set the roles in the review form, or run again as solo."
SAME_VOICES = (
    "Both tracks carry the same voices, so who said what is uncertain and roles are low confidence. "
    "Run again with the mixed layout."
)


@dataclass
class Block:
    """Words said to a candidate ("ask": a question, a prompt, a question read aloud) or their answer."""

    role: str
    candidate: str
    speaker: str
    ids: list[int]
    kind: str = "read_aloud"

    def start(self, words: list[Word]) -> int:
        return int(words[self.ids[0]]["start_ms"])

    def end(self, words: list[Word]) -> int:
        return max(int(words[i]["end_ms"]) for i in self.ids)


@dataclass
class People:
    blocks: list[Block]
    candidates: dict[str, Row]  # candidate ID -> speaker, confidence, swap times, heard_questions_first
    warnings: list[str]


def _pair(found: Row, edits: overrides.Overrides) -> list[str]:
    """The two people: the main voices with the most talk time, less any the user marked as another voice."""
    main = [
        s for s in found.get("speakers", []) if s["kind"] == "main" and s["id"] not in edits.other_speakers
    ]
    return [s["id"] for s in sorted(main, key=lambda s: -s["speech_ms"])[:2]]


def _interview(words: list[Word], ts: list[roles.Turn], pair: list[str], found: Row, swap: bool) -> People:
    talk = {s["id"]: s["speech_ms"] for s in found["speakers"]}
    r = roles.interview(ts, words, pair, talk)
    examiner, candidate = (r.candidate, r.examiner) if swap else (r.examiner, r.candidate)
    changed = len(roles.phases(roles.exchanges(ts, words), pair, "once")[0]) > 1
    blocks = [
        Block("ask", "candidate", t.speaker, t.ids, questions.asking(words, t.ids).kind)
        if t.speaker == examiner else Block("answer", "candidate", t.speaker, t.ids)
        for t in ts
    ]  # fmt: skip
    info: Row = {
        "speaker": candidate,
        "confidence": r.confidence,
        "swap_times_ms": [],
        "heard_questions_first": None,
    }
    return People(blocks, {"candidate": info}, ["W_ROLE_CHANGE"] if changed else [])


def _cut(words: list[Word], ids: list[int], times: list[int]) -> list[list[int]]:
    """A turn's words cut at each swap time that falls inside it: an answer, then the first question."""
    out = [[ids[0]]]
    for i in ids[1:]:
        if any(words[out[-1][-1]]["start_ms"] < x <= words[i]["start_ms"] for x in times):
            out.append([i])
        else:
            out[-1].append(i)
    return out


def _role_swap(
    words: list[Word],
    ts: list[roles.Turn],
    pair: list[str],
    found: Row,
    style: str,
    edits: overrides.Overrides,
    silences: list[questions.Span],
) -> People:
    votes = roles.exchanges(ts, words)
    found_phases, against = roles.phases(votes, pair, style)
    talk = {s["id"]: s["speech_ms"] for s in found["speakers"]}
    first = found_phases[0].examiner if found_phases else min(pair, key=lambda s: talk[s])
    if edits.swap_times_ms is not None:  # the user's swap times replace the ones found
        times = sorted(edits.swap_times_ms)
        examiners = [
            first if k % 2 == 0 else next(s for s in pair if s != first) for k in range(len(times) + 1)
        ]
        levels = ["high"] * len(examiners)
    else:  # a swap happens where the new examiner's first question starts, often inside their turn
        times = [
            words[questions.ask_from(words, votes[p.first].ask.ids, silences)]["start_ms"]
            for p in found_phases[1:]
        ]
        examiners = [p.examiner for p in found_phases] or [first]
        levels = [p.confidence for p in found_phases] or ["low"]
    if edits.swap_people:
        examiners = [next(s for s in pair if s != e) for e in examiners]
    person = {next(s for s in pair if s != examiners[0]): "person1"}
    person[next(s for s in pair if s not in person)] = "person2"
    blocks = []
    for t in ts:
        for ids in _cut(words, t.ids, times):
            examiner = examiners[sum(1 for x in times if x <= words[ids[0]]["start_ms"])]
            candidate = person[next(s for s in pair if s != examiner)]
            if t.speaker == examiner:
                blocks.append(Block("ask", candidate, t.speaker, ids, questions.asking(words, ids).kind))
            else:
                blocks.append(Block("answer", candidate, t.speaker, ids))
    confidence = min(levels, key=LEVELS.index)
    candidates = {
        cid: {"speaker": s, "confidence": confidence, "swap_times_ms": times,
              "heard_questions_first": cid == "person2"}
        for s, cid in sorted(person.items(), key=lambda kv: kv[1])
    }  # fmt: skip
    return People(blocks, candidates, ["W_FREQUENT_SWAPS"] if against >= roles.FREQUENT else [])


def _people(
    words: list[Word], record: Row, found: Row, edits: overrides.Overrides, silences: list[questions.Span]
) -> People | None:
    """Two-person modes: the blocks and candidates, or None when roles are unavailable (never guessed)."""
    if found.get("status") not in ("ok", "partial") or all(w["speaker"] is None for w in words):
        return None
    pair = _pair(found, edits)
    if len(pair) == 1 and edits.one_voice == "candidate":
        ids = [i for i, w in enumerate(words) if w["speaker"] == pair[0]]
        info: Row = {
            "speaker": pair[0],
            "confidence": "high",
            "swap_times_ms": [],
            "heard_questions_first": None,
        }
        return People([Block("answer", "candidate", pair[0], ids)] if ids else [], {"candidate": info}, [])
    if len(pair) < 2:  # noqa: PLR2004 - the two people
        return None
    rs = roles.runs(words)
    roles.mark_backchannels(rs, pair, words)
    ts = roles.turns(rs, pair)
    if record["mode"] == Mode.INTERVIEW:
        return _interview(words, ts, pair, found, edits.swap_people)
    return _role_swap(words, ts, pair, found, record.get("swap_style") or "once", edits, silences)


def _silence(words: list[Word], a: int, b: int) -> int:
    """The longest stretch without any word between a and b (ms)."""
    longest, reach = 0, a
    for w in words:
        if w["end_ms"] <= a or w["start_ms"] >= b:
            continue
        longest = max(longest, w["start_ms"] - reach)
        reach = max(reach, w["end_ms"])
    return max(longest, b - reach)


def _utterances(words: list[Word], ids: list[int], answer_id: str, speaker: str, n: int) -> list[Row]:
    """An answer's words cut wherever its speech-model chunk changes or another voice's word comes between."""
    groups: list[list[int]] = []
    for i in ids:
        if groups and i == groups[-1][-1] + 1 and words[i]["chunk"] == words[i - 1]["chunk"]:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [
        {"id": f"U{n + k + 1:03d}", "answer_id": answer_id, "speaker": speaker, "first_word": g[0],
         "last_word": g[-1], "start_ms": words[g[0]]["start_ms"],
         "end_ms": max(words[i]["end_ms"] for i in g)}
        for k, g in enumerate(groups)
    ]  # fmt: skip


def _ended(words: list[Word], last: int, speaker: str, duration_ms: int) -> str:
    """How a long turn ended, from the word after its last word."""
    end = words[last]["end_ms"]
    if last + 1 == len(words):
        return "cut_off" if duration_ms - end < CUT_OFF_MS else "stopped"
    nxt = words[last + 1]
    if nxt["speaker"] is None:
        return "unknown"
    if nxt["speaker"] == speaker or nxt["start_ms"] - end >= STEPPED_IN_MS:
        return "stopped"
    return "cut_off" if nxt["start_ms"] < end else "examiner_prompt"


def _widen(start: int, end: int, segments: list[list[int]], lo: int, hi: int) -> tuple[int, int]:
    """Stretch an answer to the edges of the detected speech around its first and last word, within lo-hi.

    Word times from the speech model often end before the voice does.
    """
    for a, b in segments:
        if a <= start <= b:
            start = max(min(a, start), lo)
        if a <= end <= b:
            end = min(max(b, end), hi)
    return start, end


def _assemble(
    words: list[Word], blocks: list[Block], listed: list[Listed] | None, segments: list[list[int]]
) -> Row:
    """Questions (what was said to each candidate), answers, utterances and one Part step per answer."""
    asked: list[Row] = []
    answers: list[Row] = []
    utterances: list[Row] = []
    steps: dict[str, list[parts.Step]] = {}
    pending: dict[str, list[Row]] = {}
    last_end: dict[str, int] = {}
    owned: dict[str, set[int]] = {}  # each candidate's answer words
    ordered = sorted(blocks, key=lambda b: b.start(words))
    for n, b in enumerate(ordered):
        start, end = b.start(words), b.end(words)
        if b.role == "ask":
            hit = questions.match(listed, questions.lower(words, b.ids)) if listed else None
            q = {
                "id": f"Q{len(asked) + 1:02d}", "start_ms": start, "end_ms": end, "kind": b.kind,
                "text": " ".join(words[i]["raw"] for i in b.ids), "speaker": b.speaker,
                "addressed_to": b.candidate, "listed": hit,
                "cue_card": questions.holds(words, b.ids, "cue_card_words.txt"),
                "part_cue": questions.part_cue(words, b.ids),
            }  # fmt: skip
            asked.append(q)
            pending.setdefault(b.candidate, []).append(q)
            continue
        before = pending.pop(b.candidate, [])
        lo = ordered[n - 1].end(words) if n else 0
        hi = ordered[n + 1].start(words) if n + 1 < len(ordered) else max([end, *(s[1] for s in segments)])
        start, end = _widen(start, end, segments, lo, hi)
        answer = {
            "id": f"A{len(answers) + 1:02d}", "candidate": b.candidate,
            "question_id": before[-1]["id"] if before else None, "part": "unknown", "start_ms": start,
            "end_ms": end, "word_count": len(b.ids), "after_preparation": False, "last_word": b.ids[-1],
        }  # fmt: skip
        answers.append(answer)
        owned.setdefault(b.candidate, set()).update(b.ids)
        utterances += _utterances(words, b.ids, str(answer["id"]), b.speaker, len(utterances))
        listed_part = (
            listed[before[-1]["listed"]].part
            if listed and before and before[-1]["listed"] is not None
            else None
        )
        first = before[0]["start_ms"] if before else start
        since = last_end.get(b.candidate, first)
        cue = next((q["part_cue"] for q in before if q["part_cue"]), None)
        card = any(q["cue_card"] for q in before)
        step = parts.Step(first, start, end, card, cue, listed_part, _silence(words, since, start))
        steps.setdefault(b.candidate, []).append(step)
        last_end[b.candidate] = end
    hits = {q["listed"] for q in asked}
    unmatched = [q.text for k, q in enumerate(listed or []) if k not in hits]
    return {"questions": asked, "answers": answers, "utterances": utterances,
            "unmatched_questions": unmatched, "steps": steps, "owned": owned}  # fmt: skip


def _parts(
    cid: str, info: Row, out: Row, words: list[Word], duration_ms: int, edit: overrides.PartEdit | None
) -> None:
    """Label the candidate's answers with Parts, and add the Part spans and the Part 2 record."""
    mine = [a for a in out["answers"] if a["candidate"] == cid]
    steps = out["steps"].get(cid, [])
    if edit is not None:
        labels: list[int | str] = [edit.label(a["start_ms"]) for a in mine]
        two = [k for k, label in enumerate(labels) if label == 2]
        found = parts.Parts(
            labels, "user", "high", max(two, key=lambda k: steps[k].answer_ms) if two else None, 0
        )
    else:
        found = parts.detect(steps)
    for a, label in zip(mine, found.labels, strict=True):
        a["part"] = label
    out["parts"] += [
        {"candidate": cid, "part": p, "start_ms": a, "end_ms": b, "source": found.source,
         "confidence": found.confidence}
        for p, a, b in parts.spans(steps, found.labels)
    ]  # fmt: skip
    info["part2"] = None
    if found.anchor is not None:
        k = found.anchor
        prep = steps[k].silence_ms if steps[k].silence_ms >= parts.PREP_MS else 0
        # The minute to prepare is neither a pause nor a response time.
        mine[k]["after_preparation"] = prep > 0
        info["part2"] = {
            "prep_excluded_ms": prep, "long_turn_ms": steps[k].answer_ms,
            "examiner_prompts_n": found.follow_ups,
            "ended_by": _ended(words, mine[k]["last_word"], info["speaker"], duration_ms),
        }  # fmt: skip


def _speech(
    words: list[Word], answers: list[Row], ids: set[int], base: list[list[int]], shared: list[Any],
    partner: list[tuple[int, int]],
) -> list[list[int]]:  # fmt: skip
    """The candidate's own speech in their answers: VAD speech less other voices (and 300 ms) and overlap."""
    own = [(words[i]["start_ms"], words[i]["end_ms"]) for i in ids if OVERLAP not in words[i]["flags"]]
    # The other voices' turns count as well as their words: the speech model often misses a short "mm-hmm".
    heard = [(w["start_ms"], w["end_ms"]) for i, w in enumerate(words) if i not in ids]
    others = [(a - PARTNER_MS, b + PARTNER_MS) for a, b in [*heard, *partner]]
    cut = union([*subtract(others, own), *shared])
    return [list(s) for s in subtract(inside(base, [(a["start_ms"], a["end_ms"]) for a in answers]), cut)]


def run(job: Path) -> dict[str, Any]:
    """Structure for the declared mode; roles that cannot be found leave the job complete, with gaps."""
    record = read_json(job / "job.json")
    vad = read_json(job / "stages" / "vad.json")
    words = read_json(job / "stages" / "words.json")["words"]
    duration_ms = read_json(job / "stages" / "media.json")["duration_ms"]
    path = job / "stages" / "speakers.json"
    found = read_json(path) if path.is_file() else {}
    edits = overrides.load(job)
    listed_file = job / "input" / QUESTIONS
    listed = questions.parse_list(listed_file.read_text(encoding="utf-8")) if listed_file.is_file() else None
    silences = questions.long_silences(vad["segments"])
    if record["mode"] == Mode.SOLO:
        blocks = [
            Block("ask" if role == "read_aloud" else "answer", "candidate", "S1", ids)
            for role, ids in questions.solo_split(words, listed, silences)
        ]  # fmt: skip
        info: Row = {
            "speaker": "S1",
            "confidence": "high",
            "swap_times_ms": [],
            "heard_questions_first": None,
        }
        people: People | None = People(blocks, {"candidate": info}, [])
        speakers = [{"id": "S1", "speech_ms": vad["speech_ms"], "source": "single"}]
        method = "single"
    else:
        people = _people(words, record, found, edits, silences)
        speakers = [
            {"id": s["id"], "speech_ms": s["speech_ms"], "source": SOURCES[found["backend"]]}
            for s in found.get("speakers", [])
        ]  # fmt: skip
        user = edits.swap_people or edits.swap_times_ms is not None or edits.one_voice is not None
        method = "user" if user else SOURCES.get(found.get("backend", ""), "single")
    if people is None:
        transcript = [
            {"speaker": r.speaker, "start_ms": r.start_ms, "end_ms": r.end_ms,
             "text": " ".join(words[i]["raw"] for i in r.ids)}
            for r in roles.runs(words)
        ]  # fmt: skip
        return {
            "status": "partial", "reason": NO_ROLES, "speakers": speakers, "candidates": [], "parts": [],
            "questions": [], "answers": [], "utterances": [], "speech": {}, "unmatched_questions": [],
            "warnings": ["W_ROLES_UNAVAILABLE"], "transcript": transcript,
            "unknown_candidate": UNKNOWN_CANDIDATE,
        }  # fmt: skip
    out = _assemble(words, people.blocks, listed, vad["segments"])
    out["parts"] = []
    capped = _capped(record, found)
    # One microphone on both channels: each track holds both people, so the owner of each word is a guess.
    same_voices = "W_TRACK_BLEED" in vad.get("warnings", [])
    turns = [(t["start_ms"], t["end_ms"], t["speaker"]) for t in found.get("turns", [])]
    shared = overlaps(turns)
    candidates = []
    for cid, info in people.candidates.items():
        _parts(
            cid, info, out, words, duration_ms, next((e for k, e in edits.parts.items() if k == cid), None)
        )
        confidence = info["confidence"]
        if same_voices:
            confidence = "low"
        elif capped and confidence == "high":
            confidence = "medium"
        candidates.append({
            "id": cid, "label": LABELS[cid], "speaker": info["speaker"],
            "heard_questions_first": info["heard_questions_first"], "part2": info["part2"],
            "roles": {"confidence": confidence, "method": method, "swap_times_ms": info["swap_times_ms"]},
        })  # fmt: skip
    owned = out.pop("owned")
    del out["steps"]
    out["speech"] = {}
    for c in candidates:
        mine = [a for a in out["answers"] if a["candidate"] == c["id"]]
        track = next((t for t in vad.get("tracks", []) if f"S{t['track']}" == c["speaker"]), None)
        base = track["segments"] if track else vad["segments"]
        partner = [(a, b) for a, b, s in turns if s != c["speaker"]]
        out["speech"][c["id"]] = _speech(words, mine, owned.get(c["id"], set()), base, shared, partner)
    return {
        "status": "partial" if same_voices else "ok", "reason": SAME_VOICES if same_voices else None,
        "speakers": speakers, "candidates": candidates, "warnings": people.warnings, **out,
    }  # fmt: skip


def _capped(record: Row, found: Row) -> bool:
    """Role confidence is at most medium after a speaker fallback, similar voices, or one or three voices."""
    fell_back = any(
        s["id"] == "speakers" and s.get("fallback") and s["fallback"]["to"] != "cpu" for s in record["stages"]
    )
    odd = {"W_ONE_VOICE", "W_THIRD_VOICE"} & set(found.get("warnings", []))
    return bool(fell_back or found.get("role_confidence_cap") == "medium" or odd)
