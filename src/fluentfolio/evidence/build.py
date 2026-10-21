"""The evidence stage: evidence.json from the job record and the stage files, checked by the schema models."""

from __future__ import annotations

import secrets
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from fluentfolio import __version__
from fluentfolio.analyzers.measures import pronunciation
from fluentfolio.analyzers.measures.fluency import VIEWS_MS, place, runs
from fluentfolio.analyzers.measures.values import rounded
from fluentfolio.core.errors import catalogue as codes
from fluentfolio.core.files import read_json, write_json
from fluentfolio.core.paths import source_commit
from fluentfolio.core.text import clean
from fluentfolio.evidence.labels import COVERAGE, LABELS, NONE_FLAGGED, NOT_CHECKED, POINTERS, reliability
from fluentfolio.evidence.models import SCHEMA_VERSION, Evidence
from fluentfolio.export import audio_check, differences
from fluentfolio.export.for_ai import PROMPT_VERSION
from fluentfolio.ingest import secondary
from fluentfolio.models import catalogue
from fluentfolio.speakers.checks import STANDS_IN_FOR
from fluentfolio.speakers.ownership import MUSIC, OVERLAP
from fluentfolio.structure import overrides

Row = dict[str, Any]
OURS = "fluentfolio"
DISCLAIMER = (
    "Fluentfolio measures a practice recording and gives no band. A band from an AI chat is an unofficial "
    "estimate. Fluentfolio is not affiliated with or endorsed by the British Council, IDP IELTS or Cambridge "
    "University Press & Assessment."
)
NO_THRESHOLD = "No word is marked unsure: the confidence threshold for this speech model is not set yet."
MUSIC_ONLY = (
    "The confidence threshold for this speech model is not set yet, so a word is marked unsure ([?word]) "
    "only when it was heard over background music."
)
NON_ENGLISH = "[non-English]"
FEW_FILLERS = "Fillers are under-counted with this model."
NOT_RUN = "did not run"
MFA_EXPERIMENTAL = "experimental on this system, where it did not pass its word-timing check"
LIGHT_REASON = "off for this light run of a long recording"
LIGHT_STEPS = ("phones", "opensmile")  # with the MFA aligner, the parts a light run switches off
# Both lists are found by word lists, not by meaning, so the reader is told what a listed word proves.
LINKING_NOTE = (
    'Found by matching a word list, so a word counts even where it does not link ideas, such as "first" '
    "used as an adverb. Read each use in the transcript."
)
RARE_NOTE = (
    "Content words outside the 5,000 most common English words (wordfreq's list), each at its first use; "
    "names found by the grammar parser left out. A frequency count, not a judgement of level or of "
    "correct use."
)
LICENCE_ORDER = ("permissive", "cc-by", "gpl", "nc")  # from the least to the most restrictive


@dataclass(frozen=True)
class Producer:
    name: str
    version: str
    licence_class: str


def _tools(asr: Row) -> dict[str, Producer]:
    """What produced each kind of value: the voice detector, the speech model, the app's code and wordfreq."""
    vad = catalogue.get("silero-vad")
    model = catalogue.get(asr["model"])
    return {
        "media": Producer("pyav", metadata.version("av"), "gpl"),  # its wheel carries LGPL FFmpeg libraries
        "vad": Producer(vad.id, f"{vad.package} {metadata.version(str(vad.package))}", vad.licence_class),
        "asr": Producer(model.id, asr["revision"], model.licence_class),
        OURS: Producer(OURS, __version__, "permissive"),
        "ctc": Producer("ctc-aligner", __version__, "permissive"),
        "wordfreq": Producer("wordfreq", metadata.version("wordfreq"), "gpl"),  # its word lists are CC BY-SA
    }


def _display(word: Row) -> str:
    """A word as the transcript shows it: one spelling for hesitation sounds, the model's own for the rest.

    A word heard over background music is shown as unsure: `[?word]`.
    """
    text = str(word["text"] if word["is_filler"] else word["raw"])
    return f"[?{text}]" if MUSIC in word["flags"] else text


def _foreign(word: Row, marked: list[Row]) -> bool:
    """True when the word's middle lies in speech the language check marked, on the word's own track."""
    middle = (word["start_ms"] + word["end_ms"]) // 2
    return any(s["start_ms"] <= middle < s["end_ms"] and s.get("track") == word.get("track") for s in marked)


def _unsure(words: list[Row], ids: list[int], marked: list[Row]) -> set[int]:
    """The words an audio-check span must not touch: heard over music, overlapped, or in another language.

    A span over them would ask the chat for words that no one can check against the transcript.
    """
    return {
        i for i in ids
        if MUSIC in words[i]["flags"] or OVERLAP in words[i]["flags"] or _foreign(words[i], marked)
    }  # fmt: skip


def pause_mark(ms: int) -> str:
    """`[pause x.x s]` rounded half up, but never up to a counted length (0.5 s, 1 s) it did not reach."""
    # Otherwise a 960 ms pause reads "1.0 s" in the transcript but is not among the pauses of 1 s or more.
    shown = rounded(ms / 1000, 1)
    for line in VIEWS_MS:
        if ms < line <= shown * 1000:
            shown = (line - 100) / 1000
    return f"[pause {shown:.1f} s]"


def _texts(
    utterances: list[Row], words: list[Row], pauses: list[tuple[int, int]], marked: list[Row],
    hidden: frozenset[int] = frozenset(),
) -> dict[str, str]:  # fmt: skip
    """Utterance texts with a `[pause x.x s]` mark before the word that follows each silent pause.

    A `[non-English]` mark goes before each run of words in speech the language check marked. The words
    of the audio-check span (`hidden`) become one withheld mark, the pauses between them included.
    """
    marks: dict[int, list[str]] = {}
    for (a, b), k in zip(pauses, place(pauses, [w["start_ms"] for w in words]), strict=True):
        marks.setdefault(k, []).append(pause_mark(b - a))
    firsts = {u["first_word"] for u in utterances}
    texts = {}
    for u in utterances:
        parts: list[str] = []
        before = False
        for i in range(u["first_word"], u["last_word"] + 1):
            if i in hidden:
                if i - 1 not in hidden or i == u["first_word"]:  # a run that goes on from the last utterance
                    parts += [*marks.get(i, []), audio_check.WITHHELD]
                before = False
                continue
            parts += marks.get(i, [])
            now = _foreign(words[i], marked)
            if now and not before:
                parts.append(NON_ENGLISH)
            before = now
            parts.append(_display(words[i]))
        if u["last_word"] + 1 not in firsts:  # a pause after the last word of the answer stays with it
            parts += marks.get(u["last_word"] + 1, [])
        texts[u["id"]] = clean(" ".join(parts))
    return texts


def _speech_ms(speech: list[tuple[int, int]], answers: list[Row]) -> int:
    """Speech inside the answers as speaking time counts it, so the header and that measure agree."""
    turns = [(a["start_ms"], a["end_ms"]) for a in answers]
    return sum(e - s for turn in runs(turns, speech) for s, e in turn)


@dataclass(frozen=True)
class Context:
    mode: str
    role_confidence: str
    style: str
    unsure_pct: float | None
    audio: str  # the recording's audio-quality class
    tools: dict[str, Producer]


def _measure(key: str, value: Row, scope: str, ctx: Context, fused: Row | None = None) -> Row:
    label = LABELS[key]
    tool = ctx.tools[label.source]
    level = reliability(key, ctx.role_confidence, ctx.style, ctx.unsure_pct, ctx.audio)
    sources = [{"tool": tool.name, "version": tool.version, "weight": 1.0}]
    licence = tool.licence_class
    extra: Row = {}
    if fused is not None:  # pitch combined from several tools: each with its weight, the strictest licence
        if fused["sources"]:
            sources = fused["sources"]
            licence = max((ctx.tools[s["tool"]].licence_class for s in sources), key=LICENCE_ORDER.index)
        extra = {name: fused[name] for name in ("agreement", "spread_st", "nc_input")}
    return {
        "id": label.id, "name": label.name, "unit": value["unit"], "scope": scope, "value": value["value"],
        "kind": label.kind, "reliability": level,
        "missing": value["missing"], "sources": sources, **extra,
        "licence_class": licence, "valid_in_mode": ctx.mode in label.modes,
        "context_only": label.kind == "context",
        "n": value["n"], "note": value["note"],
    }  # fmt: skip


def _pointers(key: str, items: list[Row], note: str | None = None) -> Row:
    label = POINTERS[key]
    total = len(items)
    found = [
        {"utterance_id": p["utterance_id"], "first": p["at"][0], "last": p["at"][1], "text": p["text"]}
        for p in items
    ]
    shown = total if label.cap is None else min(label.cap, total)
    return {"id": label.id, "name": label.name, "items": found, "shown": shown, "total": total, "note": note}


def _pronunciation_notes(checked: Row, flagged: int) -> tuple[str, str]:
    """The line under the words to check, and the coverage line: what was checked and how far to trust it."""
    if checked["missing"] is not None:
        return f"{NOT_CHECKED}.", f"{NOT_CHECKED}: {checked['missing']['text']}."
    listed = NONE_FLAGGED if flagged == 0 else ""
    return (
        f"{listed}Patterns: {checked['patterns']}.",
        f"{checked['coverage']} {pronunciation.ACCENT} {pronunciation.VALIDATION}",
    )


def _clauses(scope: str, found: Row) -> str:
    name = "all answers" if scope == "all" else "Part " + scope.removeprefix("part")
    per_unit = found["clauses_per_unit"]
    units = "no clause" if per_unit is None else f"{per_unit} clauses per unit"
    sub = found["subordinate"]
    return f"{name}: {units}, {sub} subordinate clause{'' if sub == 1 else 's'}"


def _grammar_note(found: Row) -> str:
    """The line under the structure examples: clauses per Part, how many of each structure, a caution."""
    if found["missing"] is not None:
        return f"{NOT_CHECKED}: {found['missing']['text']}."
    per = "; ".join(_clauses(scope, row) for scope, row in found["scopes"].items())
    counts = ", ".join(f"{kind} {n}" for kind, n in found["counts"].items())
    return (
        f"{per}. A unit is a main clause with the clauses that depend on it. Found: {counts}. From spaCy's "
        "parser, which is trained on written text: read each example in the transcript. Not an error count."
    )


def _candidate(
    c: Row, structure: Row, results: Row, words: list[Row], speech: list[tuple[int, int]], ctx: Context,
    marked: list[Row], pieces: list[Row],
) -> tuple[Row, audio_check.Span | None, dict[str, str]]:  # fmt: skip
    """One candidate's answers, utterances, measures and pointers; values with no descriptor go to details.

    Also the audio-check span in the candidate's audio file (`pieces`), and the utterance texts with its
    words still in them, for the comparison with the learner's own transcript.
    """
    answers = [a for a in structure["answers"] if a["candidate"] == c["id"]]
    ids = {a["id"] for a in answers}
    utterances = [u for u in structure["utterances"] if u["answer_id"] in ids]
    pauses = [(a, b) for a, b in results["scopes"]["all"]["pauses"]]
    mine = [i for u in utterances for i in range(u["first_word"], u["last_word"] + 1)]
    span = audio_check.choose(answers, utterances, words, _unsure(words, mine, marked), pieces)
    hidden = frozenset(span.hidden if span else ())
    plain = _texts(utterances, words, pauses, marked)
    texts = _texts(utterances, words, pauses, marked, hidden) if span else plain
    within = audio_check.utterances_of(span, utterances) if span else set()
    quoted = {t for i in hidden if not words[i]["is_filler"] for t in audio_check.tokens(words[i]["raw"])}
    measures: list[Row] = []
    details: list[Row] = []
    for scope, found in results["scopes"].items():
        for key, value in found["values"].items():
            row = _measure(key, value, scope, ctx, found.get("fusion", {}).get(key))
            (details if LABELS[key].details_only else measures).append(row)
    # Voice quality is measured per answer, so its scope is the answer's ID.
    details += [_measure(key, value, aid, ctx) for aid, found in results["answers"].items()
                for key, value in found.items()]  # fmt: skip
    pointers: list[Row] = []
    detail_pointers: list[Row] = []
    flagged = len(results["pointers"]["words_to_check"])
    under_list, coverage = _pronunciation_notes(results["pronunciation"], flagged)
    notes = {
        "words_to_check": under_list, "structures": _grammar_note(results["grammar"]),
        "discourse_markers": LINKING_NOTE, "rare_words": RARE_NOTE,
    }  # fmt: skip
    for key, items in results["pointers"].items():
        listed = _pointers(key, items, notes.get(key))
        listed["items"] = audio_check.withhold_items(listed["items"], within, quoted)
        (detail_pointers if POINTERS[key].details_only else pointers).append(listed)
    pointers.append({"id": COVERAGE.id, "name": COVERAGE.name, "items": [], "shown": 0, "total": 0,
                     "note": coverage})  # fmt: skip
    part_fields = ("part", "start_ms", "end_ms", "source", "confidence")
    answer_fields = ("id", "question_id", "part", "start_ms", "end_ms", "word_count")
    return {
        "id": c["id"], "label": c["label"], "speaker": c["speaker"],
        "heard_questions_first": c["heard_questions_first"], "roles": c["roles"],
        "parts": [{k: p[k] for k in part_fields} for p in structure["parts"] if p["candidate"] == c["id"]],
        "speech_ms": _speech_ms(speech, answers), "part2": c["part2"],
        "answers": [
            {**{k: a[k] for k in answer_fields},
             "audio_ms": audio_check.answer_start(pieces, a["start_ms"], a["end_ms"])}
            for a in answers
        ],
        "utterances": [
            {"id": u["id"], "answer_id": u["answer_id"], "start_ms": u["start_ms"], "end_ms": u["end_ms"],
             "text": texts[u["id"]]}
            for u in utterances
        ],
        "measures": measures,
        "pointers": pointers,
        "details": {"measures": details, "pointers": detail_pointers},
    }, span, plain  # fmt: skip


def _speaker_tool(speakers: Row) -> Producer | None:
    """What told the speakers apart: a separation model, the tracks, or the solo second-voice check."""
    name = speakers.get("second_voice", {}).get("checked_by") or speakers.get("backend")
    if name == "tracks":
        return Producer(name, __version__, "permissive")
    if name not in catalogue.components():
        return None
    return Producer(name, speakers.get("revision") or speakers["version"], catalogue.get(name).licence_class)


def _aligned(
    alignment: Row, tools: dict[str, Producer], light: bool = False
) -> tuple[Producer | None, str, str | None, str | None]:
    """What gave the word times: the aligner used most, "N of M aligned", what it stood in for, and why."""
    methods = Counter(u["method"] for u in alignment["utterances"])
    total = len(alignment["utterances"])
    coverage = (
        f"{methods['mfa']} of {total} utterances aligned by MFA; {methods['ctc']} by the CTC aligner "
        f"(limited); {methods['asr']} keep the speech model's times"
    )
    mfa = alignment["mfa"]
    experimental = mfa.get("experimental", False)
    if methods["mfa"]:
        return tools["mfa"], coverage, None, f"MFA is {MFA_EXPERIMENTAL}" if experimental else None
    ran = tools["ctc"] if methods["ctc"] else tools["asr"]
    if mfa["status"] == "disabled" and light:
        return ran, coverage, None, f"MFA is {LIGHT_REASON}"
    if mfa["status"] in ("disabled", "skipped"):
        off = mfa["status"] == "disabled" and experimental
        return ran, coverage, None, f"MFA is off; it is {MFA_EXPERIMENTAL}" if off else None
    return ran, coverage, "mfa", f"MFA gave no usable times ({mfa.get('code') or 'its check failed'})"


def _analyzer(job: Path, step: str, component: str) -> Producer:
    """A tool step's component with the version the step reported, or "did not run"."""
    path = job / "analyzers" / f"{step}.json"
    found = read_json(path).get("provenance") if path.is_file() else None
    return Producer(component, found["version"] if found else NOT_RUN, catalogue.get(component).licence_class)


def _tool_rows(record: Row, structure: Row, alignment: Row, tools: dict[str, Producer]) -> list[Row]:
    """One row per finished step, with what ran, where, how long and how much memory it took."""
    light = bool(record.get("light_run"))
    which = {"media": "media", "vad": "vad", "language": "language", "asr": "asr", "speakers": "speakers",
             "words": OURS, "structure": OURS, "candidate_audio": OURS, "ai_audio": OURS, "phones": "phones",
             "praat": "praat",
             "swiftf0": "swiftf0", "opensmile": "opensmile", "nlp": "nlp", "measures": OURS}  # fmt: skip
    rows = []
    for rec in record["stages"]:
        if rec.get("ended_utc") is None:
            continue
        tool = tools.get(which.get(rec["id"], ""))
        reason = structure.get("reason") if rec["id"] == "structure" else None
        replaced = coverage = None
        if (fallback := rec.get("fallback")) and fallback["to"] == "cpu":
            reason = f"ran on the CPU after {fallback['after']} on the GPU"
        elif fallback:  # another backend took over; the row names the one that ran
            replaced = STANDS_IN_FOR[fallback["to"]]
            reason = f"{replaced} could not run ({fallback['after']}), so another backend ran instead"
        if rec["id"] == "alignment" and alignment:
            tool, coverage, replaced, reason = _aligned(alignment, tools, light)
        elif light and rec["id"] in LIGHT_STEPS and rec["status"] == "disabled":
            reason = LIGHT_REASON
        rows.append({
            "id": rec["id"], "status": rec["status"], "reason_code": rec.get("error_code"), "reason": reason,
            "coverage": coverage, "version": tool.version if tool else None,
            "ran": tool.name if tool else None, "fallback_for": replaced, "device": rec.get("device"),
            "wall_ms": rec.get("wall_ms"),
            "peak_ram_mb": rec.get("peak_ram_mb"), "peak_vram_mb": rec.get("peak_vram_mb"),
            "licence_class": tool.licence_class if tool else None,
        })  # fmt: skip
    return rows


def _warnings(record: Row, *results: Row) -> list[Row]:
    found = [code for r in results for code in r.get("warnings", [])]
    if record.get("light_run"):
        found.append("W_LIGHT_RUN")
    for rec in record["stages"]:
        if fallback := rec.get("fallback"):
            found.append("W_CPU_FALLBACK" if fallback["to"] == "cpu" else "W_SPEAKERS_FALLBACK")
    return [{"code": code, "text": codes()[code].message} for code in dict.fromkeys(found)]


def _limitations(structure: Row, provenance: Row, language: Row, music_ms: int) -> list[str]:
    """What a reader must keep in mind: missing roles, transcript limits, marks, music and questions."""
    limitations = [structure["reason"]] if structure.get("reason") else []
    model = catalogue.get(provenance["model"])
    if model.note:
        limitations.append(f"Transcript by {model.title}: {model.note}.")
    if provenance["filler_capture"] == "lower_bound":
        limitations.append(FEW_FILLERS)
    if foreign_ms := language.get("non_english_ms"):
        limitations.append(
            f"{rounded(foreign_ms / 1000, 1):.1f} s of speech is marked {NON_ENGLISH} by the language check. "
            "It stays in the transcript and the measures; nothing was translated or removed."
        )
    if music_ms:
        # Without this line a pitch value missing for "too little voiced speech" reads as a short answer.
        limitations.append(
            f"The background-music check (a heuristic) flagged {rounded(music_ms / 1000, 1):.1f} s of the "
            "recording. Pitch, voice and loudness use only the speech outside it, so a value there can be "
            "missing because of the music, not because an answer was short."
        )
    if unmatched := structure["unmatched_questions"]:
        limitations.append(f"From your question list, not asked or not recognised: {' | '.join(unmatched)}")
    return [*limitations, MUSIC_ONLY if music_ms else NO_THRESHOLD]


def _differences(
    users: Path, candidates: list[Row], turns: list[Row], texts: dict[str, str]
) -> tuple[list[Row], float | None]:
    """The learner's transcript against the whole session in time order, questions included.

    Each utterance is compared by its text before the audio-check words were withheld (`texts`), so the
    learner's own words for the span do not all show up as differences.
    """
    timed = [(t["start_ms"], differences.Piece(None, None, t["text"])) for t in turns] + [
        (u["start_ms"], differences.Piece(u["id"], u["answer_id"], texts[u["id"]]))
        for c in candidates
        for u in c["utterances"]
    ]
    session = [piece for _, piece in sorted(timed, key=lambda t: t[0])]
    return differences.compare(session, users.read_text(encoding="utf-8"))


def _producers(
    job: Path, provenance: Row, speakers: Row, language: Row, alignment: Row
) -> dict[str, Producer]:
    """Every tool that gave values in this job, with the version it reported, or "did not run"."""
    tools = _tools(provenance)
    if speaker_tool := _speaker_tool(speakers):
        tools["speakers"] = speaker_tool
    if checked := language.get("provenance"):
        tools["language"] = Producer(
            checked["model"], checked["revision"], catalogue.get(checked["model"]).licence_class
        )
    found = job / "stages" / "phones.json"
    if (phones := read_json(found).get("provenance") if found.is_file() else None) is not None:
        tools["phones"] = Producer(
            phones["model"], phones["revision"], catalogue.get(phones["model"]).licence_class
        )
    version = alignment.get("mfa", {}).get("version")
    tools["mfa"] = Producer("mfa", version or NOT_RUN, catalogue.get("mfa").licence_class)
    tools |= {name: _analyzer(job, name, name) for name in ("praat", "swiftf0", "opensmile")}
    tools["nlp"] = _analyzer(job, "nlp", "spacy-en-core-web-sm")
    return tools


def build(job: Path) -> Row:
    """The evidence of a finished analysis, validated: times inside the recording, one owner per answer."""
    return _assemble(job)[0]


def _assemble(job: Path) -> tuple[Row, Row]:
    """The evidence, and per candidate the audio-check span with its hidden words, which stay in the job."""
    record = read_json(job / "job.json")
    media = read_json(job / "stages" / "media.json")
    vad = read_json(job / "stages" / "vad.json")
    asr = read_json(job / "stages" / "asr.json")
    structure = read_json(job / "stages" / "structure.json")
    words = read_json(job / "stages" / "words.json")["words"]  # the speech model's words, with their flags
    music = "W_BACKGROUND_MUSIC" in media["warnings"]
    measures = read_json(job / "analyzers" / "measures.json")
    found = job / "stages" / "speakers.json"
    speakers = read_json(found) if found.is_file() else {}
    found = job / "stages" / "language.json"
    language = read_json(found) if found.is_file() else {}
    marked = [s for s in language.get("segments", []) if s["marked"]]
    provenance = asr["provenance"]
    found = job / "stages" / "alignment.json"
    alignment = read_json(found) if found.is_file() else {}
    tools = _producers(job, provenance, speakers, language, alignment)
    stub = any(r.get("stub", False) for r in (media, vad, asr, structure, measures))
    found = job / "stages" / "ai_audio.json"
    audio = read_json(found)["candidates"] if found.is_file() else {}
    candidates = []
    plain: dict[str, str] = {}
    checks: Row = {}
    within: set[str] = set()  # the utterances that hold a hidden word
    for c in structure["candidates"]:
        ctx = Context(record["mode"], c["roles"]["confidence"], provenance["transcript_style"], None,
                      media["quality"]["class"], tools)  # fmt: skip
        results = measures["candidates"][c["id"]]
        speech = [(a, b) for a, b in structure["speech"][c["id"]]]  # the candidate's own voice only
        pieces = audio.get(c["id"], {}).get("pieces", [])
        row, span, texts = _candidate(c, structure, results, words, speech, ctx, marked, pieces)
        candidates.append(row)
        plain |= texts
        if span:
            within |= audio_check.utterances_of(span, structure["utterances"])
            checks[c["id"]] = {
                "orig_start_ms": span.orig_start_ms, "orig_end_ms": span.orig_end_ms,
                "audio_start_ms": span.audio_start_ms, "audio_end_ms": span.audio_end_ms,
                "words": audio_check.hidden_text(span, words),
            }  # fmt: skip
    # By ID and name: two labels share M07 (with and without the parser), and only the one used is defined.
    used = {
        (m["id"], m["name"]): m["unit"] for c in candidates for m in c["measures"] + c["details"]["measures"]
    }
    definitions = [
        {
            "measure_id": label.id,
            "unit": used[label.id, label.name],
            "definition": label.definition,
            "caveat": label.caveat,
            "how": label.how,
        }
        for label in LABELS.values()
        if (label.id, label.name) in used
    ]
    turn_fields = ("id", "start_ms", "end_ms", "kind", "addressed_to")
    examiner_turns = [
        {**{k: q[k] for k in turn_fields}, "text": clean(q["text"])} for q in structure["questions"]
    ]
    # The learner's transcript is stored only with names replaced (ingest.secondary).
    users = job / "input" / secondary.TRANSCRIPT
    ingested = users.is_file()
    rows, match = _differences(users, candidates, examiner_turns, plain) if ingested else ([], None)
    # The learner's side of a row holds the span's words too, so both sides of its rows are withheld.
    rows = audio_check.withhold_rows(rows, within)
    shown = sum(
        min(differences.MAX_ROWS, sum(r["utterance_id"] in ids for r in rows))
        for ids in ({u["id"] for u in c["utterances"]} for c in candidates)
    )
    audio_export = {
        "produced": bool(audio), "candidate_only": True, "partner_voice_included": False,
        # Role-swap has a file per person, each under the limit; the longer one is given.
        "duration_ms": max((a["duration_ms"] for a in audio.values()), default=0),
    }  # fmt: skip
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "manifest": {
            "export_id": secrets.token_hex(3), "job_id": record["job_id"], "tool_version": __version__,
            "prompt_version": PROMPT_VERSION, "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python_assigns_band": False, "stub": stub, "commit": source_commit(), "mode": record["mode"],
            "layout": record["layout"], "swap_style": record["swap_style"],
            "asr": {
                **{k: provenance[k] for k in ("model", "revision", "transcript_style", "filler_capture")},
                "punctuation": provenance["punctuation"], "confidence_threshold": None,
            },
            "unsure_word_share_pct": None,
            "audio": {
                "duration_ms": media["duration_ms"], "quality_class": media["quality"]["class"],
                "snr_db": media["quality"]["snr_db"], "clipping_pct": media["quality"]["clipping_pct"],
                "music_warning": music,
            },
            "device": record["device"], "audio_export": audio_export,
            "audio_check": [
                {"candidate": cid, **{k: v for k, v in check.items() if k != "words"}}
                for cid, check in checks.items()
            ],
            "user_transcript": {"status": "ingested" if ingested else "none", "names_removed": ingested,
                                "word_match_pct": match, "differences_shown": shown,
                                "differences_total": len(rows)},
            "tools": _tool_rows(record, structure, alignment, tools),
            "warnings": _warnings(record, media, vad, asr, speakers, structure),
            # openSMILE is the one non-commercial component; its values are in the evidence when it ran.
            "nc_component_used": tools["opensmile"].version != NOT_RUN,
            "overrides_applied": (job / overrides.FILE).is_file(),
        },
        "speakers": structure["speakers"],
        "candidates": candidates,
        "examiner_turns": examiner_turns,
        "differences": rows,
        "definitions": definitions,
        "limitations": _limitations(
            structure, provenance, language, sum(b - a for a, b in media["music_spans"])
        ),
        "disclaimer": DISCLAIMER,
    }  # fmt: skip
    return Evidence.model_validate(evidence).model_dump(mode="json"), checks


def run(job: Path) -> dict[str, Any]:
    """Write evidence.json, and the hidden words of the audio check beside the stage files (never exported).

    The words file is written on every run, empty when no span was found, so none from an earlier run stays.
    """
    evidence, checks = _assemble(job)
    write_json(job / audio_check.FILE, checks)
    write_json(job / "evidence.json", evidence)
    return {"status": "ok"}
