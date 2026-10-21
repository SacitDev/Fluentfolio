"""How the pages write a job, a step, a measure and the transcript: plain functions from records to rows."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, get_args

from fluentfolio.analyzers import registry
from fluentfolio.core.errors import InputError, catalogue
from fluentfolio.core.text import clean
from fluentfolio.evidence.labels import COVERAGE, NONE_FLAGGED, NOT_CHECKED
from fluentfolio.settings.model import Section, Settings
from fluentfolio.web.render import clock, length_text, upfirst

MODES = {"solo": "Solo", "interview": "Interview", "role-swap": "Role swap"}
# The small picture beside each mode's sentence: who speaks when, as (lane, colour, start, length) on a line
# 100 long. Lane 0 is the learner, lane 1 the other person; "read" is a question the learner reads aloud.
DIAGRAMS = {
    "solo": [(0, "read", 0, 8), (0, "spk-1", 10, 32), (0, "read", 46, 8), (0, "spk-1", 56, 44)],
    "interview": [(1, "spk-2", 0, 8), (0, "spk-1", 10, 32), (1, "spk-2", 46, 8), (0, "spk-1", 56, 44)],
    "role-swap": [(1, "spk-2", 0, 8), (0, "spk-1", 10, 36), (0, "spk-1", 50, 8), (1, "spk-2", 60, 40)],
}
FIRST_WORDS = 12  # how many of a track's first words the review form shows beside its preview
LAYOUTS = {
    "mixed": "One recording of everyone",
    "per-track": "One track per person",
    "two-files": "Two files",
}
# The New page's per-run choice; its values are the review policies in Settings.
REVIEW_RUN = {"when-unsure": "Only if the analysis is unsure", "always": "Yes", "never": "No"}
# The mode a mismatch warning offers to run the roles again as; a third voice and frequent swaps are
# corrected in the review form instead, as their hints say.
RERUN_FOR = {"W_SECOND_VOICE_SOLO": "interview", "W_ONE_VOICE": "solo", "W_ROLE_CHANGE": "role-swap"}


@dataclass(frozen=True)
class Note:
    """A warning of the run in plain words, with the run in another mode it offers, if any."""

    text: str
    mode: str | None = None
    offer: str = ""  # what that run does and how long it takes, shown before its button
    consent: bool = False  # the new mode has two people and the job has no agreement on record


def rerun_offer(first: str, estimate_ms: int | None, where: str) -> str:
    """What running again in another mode does: the steps it repeats and about how long they take."""
    took = f", about {max(1, round(estimate_ms / 60000))} min on the {where.upper()}" if estimate_ms else ""
    return f'Runs again from "{STEPS[first]}"{took}. The words are not written down again.'


DEVICES = {"cpu": "processor", "gpu": "graphics card"}
STATES = {
    "queued": "Waiting", "running": "Running", "completed": "Finished",
    "completed_with_gaps": "Finished, with gaps", "failed": "Failed", "cancelled": "Cancelled",
    "interrupted": "Stopped",
}  # fmt: skip
# The tint of each state's tag; the word beside it always says the state too.
TONES = {
    "queued": "run", "running": "run", "completed": "ok", "completed_with_gaps": "warn", "failed": "bad",
    "cancelled": "", "interrupted": "",
}  # fmt: skip


def state(record: Any) -> tuple[str, str]:
    """A job's state word and tint; a job that a delete left half done is partly deleted, not failed."""
    if (record.error or {}).get("code") == "E_DELETE_INCOMPLETE":
        return "Partly deleted", ""
    return STATES[record.state], TONES[record.state]


RESULT_TABS = {
    "overview": "Overview", "measures": "Measures", "transcript": "Transcript", "words": "Words to check",
    "export": "Copy for AI",
}  # fmt: skip


def result_tabs(job_id: str) -> list[tuple[str, str, str]]:
    """The tabs of a finished analysis, each with its own link; Copy for AI is a page of its own."""
    base = f"/jobs/{job_id}"
    links = {"overview": base, "export": f"{base}/export"}
    return [(key, text, links.get(key, f"{base}?tab={key}")) for key, text in RESULT_TABS.items()]


STEP_DONE = {"ok": "Done", "partial": "Done, with gaps", "failed": "Failed"}  # every other status: Skipped
STEPS = {
    "media": "Reading the recording", "vad": "Finding speech", "language": "Checking the language",
    "asr": "Writing down the words", "speakers": "Telling the voices apart",
    "words": "Giving each word its speaker", "structure": "Finding questions, answers and Parts",
    "candidate_audio": "Cutting out the candidate's audio",
    "ai_audio": "Making the audio file for an AI chat", "phones": "Listening to the sounds",
    "alignment": "Timing each word", "praat": "Measuring pitch and voice (Praat)",
    "swiftf0": "Measuring pitch (SwiftF0)", "opensmile": "Measuring loudness (openSMILE)",
    "nlp": "Finding sentence structures", "measures": "Working out the measures",
    "evidence": "Writing the evidence", "export": "Writing the text for an AI chat",
}  # fmt: skip
TOOL_WORDS = {
    "ok": "done", "partial": "done, with gaps", "skipped": "skipped", "unavailable": "not available",
    "failed": "failed", "disabled": "switched off",
}  # fmt: skip
AGREEMENT = {
    "agree": "tools agree", "approximate": "tools differ a little", "single_tool": "one tool only",
    "unreliable": "unreliable", "too_little_speech": "too little speech",
}  # fmt: skip
NO_NUMBER = ("unreliable", "too_little_speech")
DECIMALS = {"count": 0, "Hz": 0, "ratio": 2}  # times, rates and shares to 0.1
PART_SCOPES = ("part1", "part2", "part3")  # a measure's scope, beside "all"
LIST_ID = re.compile(r" \(list X\d+\)")  # list IDs serve the AI chat's text; the page names its lists
ASKING = {"read_aloud": "question read aloud", "cue_card": "giving the task card", "question": "asking"}
MARKS = re.compile(r"(\[[^\]]*\])")  # the transcript's bracket marks: [?word], [non-English], [pause 1.2 s]
NON_ENGLISH = "[non-English]"


def about(ms: int) -> str:
    """A time estimate as a learner reads it."""
    return "less than a minute" if ms < 60000 else f"about {round(ms / 60000)} min"  # noqa: PLR2004


def memory_note(short: tuple[str, str, str] | None) -> str:
    """The upload's warning when a step needs more memory than is free now, or "" when there is enough."""
    if short is None:
        return ""
    step, need_gb, free_gb = short
    return (
        f'The step "{STEPS[step]}" may need about {need_gb} GB of memory for a recording this long, and '
        f"{free_gb} GB is free now. Close other programs before you start, or the analysis may stop."
    )


def when(utc: str) -> str:
    """A UTC stamp in this computer's own time, as a learner reads a date: "15 Oct 2026, 14:05"."""
    local = datetime.fromisoformat(utc.replace("Z", "+00:00")).astimezone()
    return f"{local.day} {local:%b %Y}, {local:%H:%M}"


def title(record: Any) -> str:
    """A job's name: its recording's own name, or a plain word when it has none."""
    return str(record.input.get("display_name") or "Recording")


def steps(plan: Sequence[str], record: Any) -> list[tuple[str, str]]:
    """Each step of a job with its word: Waiting, Running, Done, Skipped or Failed."""
    found = {s.id: s for s in record.stages}
    rows = []
    for step in plan:
        done = found.get(step)
        if done is None or done.status is None:
            word = "Running" if record.stage == step and record.state == "running" else "Waiting"
        else:
            word = STEP_DONE.get(str(done.status), "Skipped")
        rows.append((STEPS.get(step, step), word))
    return rows


def elapsed_ms(record: Any) -> int:
    """The time the job's steps have taken, the running one so far included; time spent waiting is not."""
    total = 0
    for s in record.stages:
        if s.wall_ms is not None:
            total += s.wall_ms
        elif s.started_utc and record.state == "running":
            started = datetime.fromisoformat(s.started_utc.replace("Z", "+00:00"))
            total += max(0, int((datetime.now(UTC) - started).total_seconds() * 1000))
    return total


def ms_of(text: str) -> int:
    """A time typed as m:ss or h:mm:ss, in ms."""
    parts = text.strip().split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts) or int(parts[-1]) >= 60:  # noqa: PLR2004
        raise InputError("E_REVIEW_INVALID", "a time is not written as m:ss")
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + int(p)
    return seconds * 1000


def plain(text: str) -> str:
    """A Doctor line without its command-line sentences: the page gives a link to the page that fixes it."""
    return " ".join(s for s in re.split(r"(?<=\.)\s+", text) if "`" not in s)


def number(measure: dict[str, Any]) -> str:
    """A value with its unit, or the reason it has none; an unreliable value shows its label, not a number."""
    if measure["agreement"] in NO_NUMBER:  # the more exact reason, also when the value is missing
        return upfirst(AGREEMENT[measure["agreement"]])
    if measure["missing"] is not None:
        return "Not available"
    unit = "" if measure["unit"] == "count" else f" {measure['unit']}"
    shown = f"{measure['value']:.{DECIMALS.get(measure['unit'], 1)}f}{unit}"
    return f"about {shown}" if measure["agreement"] == "approximate" else shown


@dataclass(frozen=True)
class Row:
    name: str
    cells: list[str]  # the whole test, then each Part
    reliability: str  # "" when no column has a number
    how: str


def measures(
    candidate: dict[str, Any], definitions: dict[str, str], tools: dict[str, str]
) -> tuple[list[Row], list[str]]:
    """The measures table, one row per measure, and its scope columns: the whole test and the Parts found.

    `tools` are the tools' names by ID, for the weights of a combined value ("Praat 0.5, SwiftF0 0.5").
    """
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for m in candidate["measures"]:
        by_id.setdefault(m["id"], {})[m["scope"]] = m
    parts = [p for p in PART_SCOPES if any(p in scopes for scopes in by_id.values())]
    rows = []
    for scopes in by_id.values():
        whole = scopes.get("all") or next(iter(scopes.values()))
        how = [LIST_ID.sub("", definitions.get(whole["id"], ""))]
        if len(whole["sources"]) > 1:  # a combined value names its tools and weights, and whether they agree
            weights = ", ".join(
                f"{tools.get(s['tool'], s['tool'])} {s['weight']:g}" for s in whole["sources"]
            )
            how.append(f"{weights}; {AGREEMENT.get(whole['agreement'] or '', 'no label')}")
        if whole["missing"] is not None:
            how.append(upfirst(whole["missing"]["text"]))  # a sentence of its own after the definition
        cells = [number(scopes[s]) if s in scopes else "" for s in ("all", *parts)]
        # With no number in any column, a reliability word would vouch for something not measured.
        valued = any(m["missing"] is None and m["agreement"] not in NO_NUMBER for m in scopes.values())
        reliability = whole["reliability"] if valued else ""
        rows.append(Row(upfirst(whole["name"]), cells, reliability, ". ".join(h for h in how if h)))
    return rows, ["Whole test", *(f"Part {p.removeprefix('part')}" for p in parts)]


@dataclass(frozen=True)
class Pointers:
    summary: str
    note: str  # what was looked for, or why nothing was
    items: list[dict[str, Any]]
    empty: str  # in place of an empty list
    shown: int
    total: int


def pointers(candidate: dict[str, Any]) -> list[Pointers]:
    """Each pointer list: a check that did not run and a note with no list never read as a count of 0."""
    found = []
    cover = next((p["note"] for p in candidate["pointers"] if p["id"] == COVERAGE.id), "")
    for p in candidate["pointers"]:
        name, note = upfirst(p["name"]), p["note"] or ""
        if p["id"] == COVERAGE.id:
            found.append(Pointers(name, note, [], "", 0, 0))
        elif note.startswith(NOT_CHECKED):  # a bare "Not checked." has its reason in the coverage note
            why = cover if note == f"{NOT_CHECKED}." and cover else note
            found.append(Pointers(f"{name}: not checked", why, [], "", 0, 0))
        else:
            empty = "" if p["total"] else "None found in these answers. This tool can miss problems."
            note = note.removeprefix(NONE_FLAGGED)  # the empty line already says none were found
            found.append(Pointers(f"{name}: {p['total']}", note, p["items"], empty, p["shown"], p["total"]))
    return found


@dataclass(frozen=True)
class Line:
    start_ms: int
    who: str
    role: str
    part: str
    text: str
    voice: str  # the colour class of who spoke, by person as on the timeline strip; the words say it too

    @property
    def pieces(self) -> list[tuple[str, str]]:
        """The text cut at its bracket marks: an unsure word or other-language speech is highlighted, as the
        page's help line says; a pause or a withheld stretch is a quiet aside."""
        found = []
        for piece in MARKS.split(self.text):
            if piece.startswith("[?") or piece == NON_ENGLISH:
                found.append(("mark", piece))
            elif piece.startswith("["):
                found.append(("aside", piece))
            elif piece:
                found.append(("text", piece))
        return found


def _part(spans: list[dict[str, Any]], at_ms: int) -> str:
    found = next((s["part"] for s in spans if s["start_ms"] <= at_ms < s["end_ms"]), "unknown")
    return "" if found == "unknown" else f"Part {found}"


def transcript(evidence: dict[str, Any], runs: list[dict[str, Any]]) -> list[Line]:
    """Every turn in time order, each with who spoke, their role and the Part, as words, not colour alone.

    `runs` is who spoke when, for a job where no candidate was found.
    """
    people = {c["id"]: c for c in evidence["candidates"]}
    voice = {cid: f"spk-{k + 1}" for k, cid in enumerate(people)}
    lines = []
    for r in runs:  # the speech model's own text, so it is cleaned here as the evidence text was
        who = f"Voice {r['speaker']}" if r["speaker"] else "Unknown voice"
        lines.append(Line(r["start_ms"], who, "", "", clean(r["text"]), ""))
    for c in people.values():
        part_of = {a["id"]: a["part"] for a in c["answers"]}
        for u in c["utterances"]:
            part = part_of.get(u["answer_id"], "unknown")
            shown = "" if part == "unknown" else f"Part {part}"
            lines.append(
                Line(u["start_ms"], upfirst(c["label"]), "answering", shown, u["text"], voice[c["id"]])
            )
    for t in evidence["examiner_turns"]:
        asked = people.get(t["addressed_to"])
        spans = asked["parts"] if asked else []
        role = ASKING.get(t["kind"], "asking")
        if t["kind"] == "read_aloud":
            who, by = "The candidate", voice.get(t["addressed_to"], "")
        elif len(people) > 1:  # in role-swap the other person asks
            other = next(c for c in people.values() if c["id"] != t["addressed_to"])
            who, by = upfirst(other["label"]), voice[other["id"]]
        else:
            who, by = "The examiner", "spk-2"
        lines.append(Line(t["start_ms"], who, role, _part(spans, t["start_ms"]), t["text"], by))
    return sorted(lines, key=lambda line: line.start_ms)


def voices(evidence: dict[str, Any], saved: dict[str, Any]) -> list[tuple[str, str]]:
    """Each voice found, whose it is and how long it spoke, for the review form's "someone else" boxes.

    With a track per person, voice Sn is track n and is named so. The evidence does not say which other voice
    an interview took as the examiner, except with a track per person, where both voices are the two people.
    """
    whose = {c["speaker"]: c["label"] for c in evidence["candidates"]}
    tracked = evidence["manifest"]["layout"] != "mixed"
    interview = evidence["manifest"]["mode"] == "interview"
    found = []
    for s in evidence["speakers"]:
        if s["id"] in whose:
            who = whose[s["id"]]
        elif interview and tracked and s["id"] not in saved.get("other_speakers", []):
            who = "the examiner"
        else:
            who = "not the candidate" if interview else "neither person"
        name = f"Track {s['id'][1:]}" if tracked else f"Voice {s['id']}"
        found.append((s["id"], f"{name} ({who}, {length_text(s['speech_ms'])} of speech)"))
    return found


def part_times(candidate: dict[str, Any]) -> tuple[str, str, str]:
    """Part 2's start and end and Part 3's start as the review form shows them; empty where none was found."""
    found = {p["part"]: p for p in candidate["parts"]}
    two, three = found.get(2), found.get(3)
    return (
        clock(two["start_ms"]) if two else "", clock(two["end_ms"]) if two else "",
        clock(three["start_ms"]) if three else "",
    )  # fmt: skip


@dataclass(frozen=True)
class Track:
    """One track of a recording with one track per person, as the review form shows it."""

    number: int
    who: str  # whose track the analysis made it: "the candidate", "the examiner", "Person 1"...
    person: bool  # one of the two people, so swapping the people moves it
    start_ms: int | None  # where its first word starts; None when no word was found on it
    words: str  # its first words


def tracks(evidence: dict[str, Any], words: list[dict[str, Any]], saved: dict[str, Any]) -> list[Track]:
    """Each track of a recording with one track per person: whose it is now, and its first words."""
    if evidence["manifest"]["layout"] == "mixed" or not evidence["candidates"]:
        return []
    whose = {c["speaker"]: c["label"] for c in evidence["candidates"]}
    others = set(saved.get("other_speakers", []))
    found = []
    for n in (1, 2):  # such a recording has two tracks, and track n is voice Sn
        speaker = f"S{n}"
        if speaker in whose:
            who, person = whose[speaker], True
        elif speaker in others:
            who, person = "someone else", False
        elif evidence["manifest"]["mode"] == "interview":
            who, person = "the examiner", True
        else:
            who, person = "neither person", False
        mine = [w for w in words if w.get("track") == n]
        own = mine[:FIRST_WORDS]
        start = own[0]["start_ms"] if own else None
        more = "..." if len(mine) > FIRST_WORDS else ""  # so a cut sentence does not read as broken text
        found.append(Track(n, who, person, start, clean(" ".join(w["raw"] for w in own)) + more))
    return found


def review_edits(
    sent: dict[str, list[str]], evidence: dict[str, Any], saved: dict[str, Any]
) -> dict[str, Any]:
    """The review form's corrections on top of the ones saved before.

    The page shows the analysis made with the saved corrections, so a time is taken only where the learner
    changed it from the one shown, and a time left as shown keeps the saved or found value to the millisecond;
    an unchanged form is not a new correction. A swap turns the people around from how the page shows them.
    """

    def one(key: str) -> str:
        return (sent.get(key) or [""])[0].strip()

    swap = one("swap_people") == "yes" or one("tracks") == "swap"
    edits: dict[str, Any] = {
        "swap_people": saved.get("swap_people", False) != swap, "other_speakers": sent.get("other", []),
        "parts": {k: dict(v) for k, v in saved.get("parts", {}).items()},
    }  # fmt: skip
    if one("one_voice") == "candidate":
        edits["one_voice"] = "candidate"
    if "swap_times_ms" in saved:
        edits["swap_times_ms"] = saved["swap_times_ms"]
    people = evidence["candidates"]
    if "swap_times" in sent and people:
        typed = [t.strip() for t in one("swap_times").split(",") if t.strip()]
        if typed != [clock(t) for t in people[0]["roles"]["swap_times_ms"]]:
            edits["swap_times_ms"] = [ms_of(t) for t in typed]  # none left: the roles never change
    for c in people:
        shown = part_times(c)
        start, end, third = (one(f"{c['id']}.{k}") for k in ("part2_start", "part2_end", "part3"))
        edit = edits["parts"].get(c["id"], {})
        if (start, end) != shown[:2]:
            edit["part2"] = [ms_of(start), ms_of(end)]
        if third and third != shown[2]:
            edit["part3_start_ms"] = ms_of(third)
        if edit:
            edits["parts"][c["id"]] = edit
    return edits


def tools(evidence: dict[str, Any]) -> list[str]:
    """What each step did, in plain words: which tool ran, or why none did."""
    out = []
    for t in evidence["manifest"]["tools"]:
        line = f"{STEPS.get(t['id'], t['id'])}: {TOOL_WORDS[t['status']]}"
        if t["ran"] and t["status"] in ("ok", "partial"):
            line += f" ({t['ran']}{', instead of ' + t['fallback_for'] if t['fallback_for'] else ''})"
        if t["reason"] and t["status"] != "ok":
            line += f". {t['reason']}"
        out.append(line)
    return out


@dataclass(frozen=True)
class Control:
    key: str  # section.name, as `settings set` takes it
    title: str
    help: str
    kind: str  # choice, switch or number
    value: str  # true or false for a switch
    options: list[tuple[str, str]]  # a choice's values and labels
    least: int | None  # a number's smallest value


def settings_form(settings: Settings, tab: str, models: dict[str, str]) -> list[tuple[str, list[Control]]]:
    """The Settings page's groups, drawn from the settings model and the analyzer registry.

    Every section of the tab gives a group and every field a control, so a new setting needs no page code;
    the speech models are the ones this version offers, by name.
    """
    groups = []
    for name, info in Settings.model_fields.items():
        section = info.annotation
        if not (isinstance(section, type) and issubclass(section, Section)) or section.tab != tab:
            continue
        values = getattr(settings, name).model_dump(mode="json")
        controls = []
        for field, spec in section.model_fields.items():
            value, labels = values[field], (spec.json_schema_extra or {}).get("labels", {})  # type: ignore[union-attr]
            kind, options = (
                ("switch" if isinstance(value, bool) else "number" if isinstance(value, int) else "choice"),
                [],
            )
            if kind == "choice":
                allowed = list(models) if f"{name}.{field}" == "asr.model" else get_args(spec.annotation)
                allowed = allowed or [str(v) for v in spec.annotation]  # type: ignore[union-attr]  # an enum
                options = [(v, models.get(v) or labels.get(v) or upfirst(v)) for v in allowed]
            least = next((m.ge for m in spec.metadata if hasattr(m, "ge")), None)
            shown = str(value).lower() if kind == "switch" else str(value)
            about = spec.description or ""
            controls.append(
                Control(f"{name}.{field}", spec.title or field, about, kind, shown, options, least)
            )
        groups.append((str(section.model_config.get("title") or name), controls))
    if tab == "basic":  # each analyzer that an optional tool runs can be switched off
        switches = [
            Control(f"analyzers.{a.id}", a.title, upfirst(a.gives) + ".", "switch",
                    str(settings.analyzer_on(a.id)).lower(), [], None)
            for a in registry.ANALYZERS if a.switchable
        ]  # fmt: skip
        groups.append(("What to measure", switches))
    return groups


COMPONENT_WORDS = {
    "ready": "Ready", "needs-download": "Needs download", "needs-token": "Needs your Hugging Face token",
    "needs-consent": "Not installed: licence not accepted", "self-test-failed": "Failed its self-test",
    "not-available": "Not available",
}  # fmt: skip
COMPONENT_TONES = {
    "ready": "ok", "needs-download": "", "needs-token": "warn", "needs-consent": "warn",
    "self-test-failed": "bad", "not-available": "",
}  # fmt: skip
INSTALLABLE = ("needs-download", "self-test-failed")  # what Install selected can fetch or fetch again
# The Components page groups the parts by what they do, the speech models first.
PART_GROUPS = {
    "asr": "Writing down the words", "language": "Checking the language", "vad": "Finding speech",
    "diarizer": "Telling the voices apart", "aligner": "Timing each word",
    "phones": "Listening to the sounds of words", "acoustic": "Measuring pitch, voice and loudness",
    "nlp": "Finding sentence structures",
}  # fmt: skip
HELP_TABS = {
    "start": "Start here", "how": "How it works", "record": "Recording tips", "parts": "The parts",
    "about": "About",
}  # fmt: skip
# The Help page's plain line for each part; a test checks that every offered part has one.
PART_ABOUT = {
    "parakeet-tdt-0.6b-v2": (
        "Speech recognition, the default: writes down what was said, with the moment of each word. "
        'It keeps hesitations such as "um".'
    ),
    "parakeet-tdt-0.6b-v3": (
        "Speech recognition, a newer version of the default. On the English tests it made a few more "
        "mistakes."
    ),
    "whisper-large-v3-turbo": (
        "Speech recognition you can choose instead. It writes smooth text that drops many hesitations, "
        "so it is marked as not for grammar evidence."
    ),
    "whisper-tiny": (
        "Language check: marks speech that is not English. That speech stays in the transcript, marked."
    ),
    "silero-vad": "Finds where someone is speaking and where the pauses are.",
    "pyannote-community-1": (
        "Tells two voices apart. It made fewer mistakes than sherpa-onnx on three of four test "
        "sessions. It needs your own free Hugging Face token."
    ),
    "sherpa-onnx-diarizer": (
        "Tells two voices apart without any account. It also checks a solo recording for a second voice."
    ),
    "mfa": (
        "Finds when each word and sound starts and ends. Experimental: it did not pass its word-timing "
        "check, so it is off unless you choose it."
    ),
    "ctc-aligner": "Finds when each word starts and ends, with the phone recogniser. It is used by default.",
    "phone-model": (
        "Listens to the sounds in each word. The word timing uses it, and so does the experimental list of "
        "words to check."
    ),
    "espeak-ng": (
        "Knows how English words are usually pronounced, in US and UK forms. The word timing and the words "
        "to check use it."
    ),
    "pronunciation-dictionary": (
        "A list of English words with their US pronunciations. The words to check use it to leave out words "
        "whose expected sounds would only be a guess, such as names."
    ),
    "praat": "Measures the pitch of your voice, how your phrases end, and flat stretches.",
    "swiftf0": "A second pitch tool. Fluentfolio compares the tools and says how well they agree.",
    "opensmile": (
        "Measures loudness, and gives a third pitch track. Its licence allows non-commercial use only, so it "
        "runs only after you accept that licence."
    ),
    "spacy-en-core-web-sm": (
        "Reads the grammar of the transcript and points to examples of structures you used, such as "
        "conditionals. It never counts errors."
    ),
}
LICENCE_WORDS = {
    "permissive": "Free to use", "cc-by": "Free to use, with credit", "gpl": "Free to use (GPL)",
    "nc": "Non-commercial use only",
}  # fmt: skip


def task_result(kind: str, reply: dict[str, Any] | None, titles: dict[str, str]) -> tuple[bool, list[str]]:
    """Whether a finished long action worked, and what it did, one plain line each."""
    if reply is None:
        return False, ["It stopped before the end. Try again."]
    data, problem = reply.get("data") or {}, reply.get("error")
    lines = []
    if kind == "install":
        for part in data.get("parts", []):
            lines.append(f"{titles.get(part['id'], part['id'])}: {COMPONENT_WORDS[part['status']].lower()}.")
    elif kind == "selftest" and data:
        off = [p for p in data["components"] if p["status"] != "ok"]
        # A reason may be a catalogue message that already ends in a full stop.
        why = {p["id"]: (p["reason"] or "").rstrip(".") for p in off}
        if data["passed"]:
            limits = [f"{titles.get(i, i)} is off because {r or 'it did not pass'}" for i, r in why.items()]
            lines.append("Ready, with limits: " + "; ".join(limits) + "." if off else "Ready.")
        else:
            lines += [f"{titles.get(i, i)} did not pass: {r}." for i, r in why.items() if r]
    if problem:
        # The self-test's failed parts are named above by title; the message's detail repeats them as IDs.
        plain = kind == "selftest" and data
        lines.append(catalogue()[problem["code"]].message if plain else problem["message"])
    return bool(reply.get("ok")), lines
