"""The measures stage: each candidate's measures, in total and per known Part (analyzers/measures.json)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fluentfolio.analyzers.measures import (
    acoustic,
    discourse,
    disfluency,
    fluency,
    fusion,
    grammar,
    lexical,
    pronunciation,
    prosody,
    rhythm,
)
from fluentfolio.analyzers.measures.values import Missing, Value, tokens
from fluentfolio.core.files import read_json
from fluentfolio.core.text import clean
from fluentfolio.settings.store import LIGHT_RUN_OFF
from fluentfolio.speakers.ownership import MUSIC, OVERLAP

Rows = list[dict[str, Any]]
ParsedWords = tuple[set[int], dict[int, str]]  # the parser's names and base forms, by word index
TOOLS = {"praat": "Praat", "swiftf0": "SwiftF0", "opensmile": "openSMILE", "nlp": "spaCy's grammar parser"}
CONSENT_MISSING = "E_OPENSMILE_CONSENT_MISSING"
CONFIDENCE_THRESHOLD = None  # as in the evidence manifest: set per speech model by the recognition experiment
NOT_CHECKED = {
    "off": "the pronunciation check is off in Settings",
    "disabled": "the phone model, eSpeak NG or the pronunciation check is off in Settings",
    "no_dictionary": "the pronunciation dictionary is not installed (see Components), so guessed "
    "pronunciations cannot be left out",
    "failed": "the phone model did not run",
    "light_run": "the phone model is off for this light run of a long recording",
}


def _scopes(answers: Rows) -> dict[str, Rows]:
    """ "all", plus one scope per Part that some answer belongs to; unknown Parts give the total only."""
    by_id = {a["id"]: a for a in answers}
    return {scope: [by_id[i] for i in ids] for scope, ids in acoustic.scopes(answers).items()}


def _tool(
    job: Path, name: str, analyzer: str, light: bool = False
) -> tuple[dict[str, Any] | None, Missing | None]:
    """A tool step's result when it gave this analyzer's values, else None and why they are missing."""
    path = job / "analyzers" / f"{name}.json"
    found: dict[str, Any] = read_json(path) if path.is_file() else {"status": "failed"}
    if found["status"] == "ok" and analyzer in found["analyzers"]:
        return found, None
    if found["status"] == "disabled" and light and name in LIGHT_RUN_OFF:
        return None, Missing("light_run", f"{TOOLS[name]} is off for this light run of a long recording")
    if found.get("code") == CONSENT_MISSING:
        return None, Missing("not_enabled", f"{TOOLS[name]} not enabled: its licence is not accepted")
    if found["status"] in ("ok", "disabled"):
        return None, Missing("switched_off", f"{TOOLS[name]} or this measure is switched off in Settings")
    if found["status"] == "skipped":
        return None, Missing("no_candidate_audio", "the candidate's own audio could not be prepared")
    if found["status"] == "unavailable":
        return None, Missing("tool_unavailable", f"{TOOLS[name]} is not installed")
    return None, Missing("tool_failed", f"{TOOLS[name]} did not run")


def _plain(values: dict[str, Value]) -> dict[str, Any]:
    return {name: asdict(v) for name, v in values.items()}


def _fused(
    tracks: dict[str, dict[str, Any] | None], cid: str, scope: str, why: Missing | None
) -> dict[str, Any]:
    """One scope's median pitch and pitch range from the pitch tools that ran, and each tool's values."""
    rows = {t: found["candidates"][cid]["scopes"][scope] for t, found in tracks.items() if found}
    pitch = fusion.fuse(rows, {t: found["provenance"]["version"] for t, found in tracks.items() if found})
    out: dict[str, Any] = {"values": {}, "fusion": {}}
    for key, fused in (("median_pitch", pitch.median), ("pitch_range", pitch.range)):
        value = asdict(fused.value)
        if not rows and why is not None:  # no pitch tool ran: say why, as Praat's step gave it
            value["missing"] = asdict(why)
        elif fused.sources and fused.agreement != fusion.SINGLE:
            weights = ", ".join(f"{TOOLS[s.tool]} {s.weight}" for s in fused.sources)
            value["note"] = f"Weights: {weights}; the tools differ by {fused.spread_st} st."
        out["values"][key] = value
        out["fusion"][key] = {
            "agreement": fused.agreement, "spread_st": fused.spread_st, "nc_input": fused.nc_input,
            "sources": [asdict(s) for s in fused.sources],
        }  # fmt: skip
    out["details"] = {"statistics": pitch.statistics, "by_tool": pitch.by_tool}
    return out


def _texts(utterances: Rows, words: Rows) -> list[tuple[str, str]]:
    """Each utterance's ID and its words as the model wrote them, so cut-offs ("wor-") stay visible.

    Each word is cleaned as the text for an AI chat is, so a character no one can see never stops a word
    from being counted.
    """
    return [
        (u["id"], " ".join(clean(w["raw"]) for w in words[u["first_word"] : u["last_word"] + 1]))
        for u in utterances
    ]


def _token_words(utterances: Rows, words: Rows) -> dict[tuple[str, int], int]:
    """The word behind each (utterance ID, token index) as the text measures count tokens."""
    found: dict[tuple[str, int], int] = {}
    for u in utterances:
        at = 0
        for i in range(u["first_word"], u["last_word"] + 1):
            n = len(tokens(clean(words[i]["raw"])))
            found |= {(u["id"], at + k): i for k in range(n)}
            at += n
    return found


def _lexical(utterances: Rows, words: Rows, parsed: ParsedWords) -> lexical.Lexical:
    """The word-frequency profile, with the parser's names left out and its base forms used.

    A word heard over background music may be misheard, so it is never listed as rare.
    """
    names, lemmas = parsed
    at = _token_words(utterances, words)
    return lexical.profile(
        _texts(utterances, words),
        unsure={k for k, i in at.items() if MUSIC in words[i]["flags"]},
        names={k for k, i in at.items() if i in names},
        lemmas={k: lemmas[i] for k, i in at.items() if i in lemmas},
    )


def _pause_places(
    answers: Rows, mine: Rows, words: Rows, speech: list[tuple[int, int]], clause_of: dict[int, int] | None
) -> list[bool | None]:
    """Each pause of the answers inside a clause or not; `clause_of` is None when nothing was parsed."""
    found: list[bool | None] = []
    for a in answers:
        ids = [
            i for u in mine if u["answer_id"] == a["id"] for i in range(u["first_word"], u["last_word"] + 1)
        ]
        # A hesitation sound or a cut-off word is not a word that a clause can start or end with.
        border = [
            (words[i]["start_ms"], words[i]["raw"], None if clause_of is None else clause_of[i])
            for i in ids
            if not (words[i]["is_filler"] or words[i]["cut_off"]) and (clause_of is None or i in clause_of)
        ]
        gaps = fluency.pauses([(a["start_ms"], a["end_ms"])], speech)
        found += grammar.locate(gaps, border, clause_of is not None)
    return found


def _measures(
    answers: Rows, utterances: Rows, words: Rows, speech: list[tuple[int, int]], mode: str, style: str,
    questions: dict[str, dict[str, Any]], phones: dict[str, list[list[Any]]],
    parsed: tuple[dict[int, int] | None, Missing | None], parsed_words: ParsedWords,
) -> dict[str, Any]:  # fmt: skip
    """Every measure of one scope, as plain dicts with a value or a reason it is missing."""
    ids = {a["id"] for a in answers}
    mine = [u for u in utterances if u["answer_id"] in ids]
    # Words spoken over another voice stay in the transcript but not in the fluency counts.
    spoken = [
        words[i]["text"]
        for u in mine
        for i in range(u["first_word"], u["last_word"] + 1)
        if OVERLAP not in words[i]["flags"]
    ]
    texts = _texts(mine, words)
    turns = [(a["start_ms"], a["end_ms"]) for a in answers]
    first_word: dict[str, int] = {}  # utterances are in time order, so the first one seen starts its answer
    for u in mine:
        first_word.setdefault(u["answer_id"], u["start_ms"])
    # Latency runs from the end of the question to the first word of its answer; the minute to prepare
    # before the long turn is not a response time.
    pairs = [
        (questions[a["question_id"]]["end_ms"], first_word[a["id"]])
        for a in answers
        if a["question_id"] in questions and a["id"] in first_word and not a["after_preparation"]
    ]
    prepared = sum(bool(a["after_preparation"]) for a in answers)
    values = fluency.timing(turns, speech, spoken) | fluency.latency(pairs, mode, prepared)
    clause_of, why = parsed
    values |= grammar.share(_pause_places(answers, mine, words, speech, clause_of), why)
    values["filled_pauses"] = disfluency.filled_pauses(spoken, sum(b - a for a, b in turns), style)
    found = discourse.markers(texts)
    values["marker_variety"] = found.variety
    lex = _lexical(mine, words, parsed_words)
    values |= {"mattr": lex.mattr, "outside_2k": lex.outside_2k, "outside_5k": lex.outside_5k}
    values |= rhythm.measures([phones[u["id"]] for u in mine if u["id"] in phones])
    return {
        "values": _plain(values),
        "markers": found.counts,
        "pauses": [list(p) for p in fluency.pauses(turns, speech)],
    }


def _check_words(
    answers: Rows, utterances: Rows, words: Rows, fitted: dict[str, Any]
) -> list[tuple[int, list[pronunciation.Word]]]:
    """Each answer's length and its words as the pronunciation check sees them.

    Hesitation sounds and cut-off words are not words, so they are neither checked nor counted.
    """
    out = []
    for a in answers:
        found: list[pronunciation.Word] = []
        for u in (u for u in utterances if u["answer_id"] == a["id"]):
            row = fitted.get(u["id"], {})
            for k, i in enumerate(range(u["first_word"], u["last_word"] + 1)):
                w = words[i]
                if w["is_filler"] or w["cut_off"]:
                    continue
                fit = row["fits"][k] if row else None
                skip: str | None
                if OVERLAP in w["flags"]:
                    skip = pronunciation.OVERLAP
                elif MUSIC in w["flags"]:
                    skip = pronunciation.MUSIC
                else:
                    guessed = bool(row and row["guessed"][k])
                    skip = pronunciation.skip_reason(
                        w["text"], guessed, w["confidence"], CONFIDENCE_THRESHOLD
                    )
                if skip is None and fit is None:
                    skip = pronunciation.NO_FIT
                found.append(pronunciation.Word(u["id"], k, w["text"], skip, fit or {}))
        out.append((a["end_ms"] - a["start_ms"], found))
    return out


def _not_checked(job: Path, light: bool) -> tuple[str | None, dict[str, Any]]:
    """Why the pronunciation check has nothing to say, or None; and each utterance's fits."""
    if not pronunciation.switched_on(read_json(job / "job.json")["settings"]):
        return "off", {}
    path = job / "stages" / "phones.json"
    found: dict[str, Any] = read_json(path) if path.is_file() else {"status": "failed"}
    if found["status"] == "disabled":
        return "light_run" if light else "disabled", {}
    if found["status"] != "ok":
        return "failed", {}
    if found["provenance"]["dictionary"] is None:
        return "no_dictionary", {}
    return None, {u["id"]: u for u in found["utterances"]}


def _pronunciation(
    reason: str | None, answers: Rows, utterances: Rows, words: Rows, fitted: dict[str, Any]
) -> tuple[dict[str, Any], Rows]:
    """The summary lines of the check on one candidate's answers, and every word it lists."""
    if reason is not None:
        return {"missing": {"code": reason, "text": NOT_CHECKED[reason]}}, []
    found = pronunciation.check(_check_words(answers, utterances, words, fitted), pronunciation.FLAG_BELOW)
    summary = {"missing": None, "patterns": found.patterns, "coverage": found.coverage}
    return summary, [asdict(p) for p in found.listed]


def run(job: Path) -> dict[str, Any]:
    """Measures for every candidate the structure stage found."""
    record = read_json(job / "job.json")
    mode, light = record["mode"], bool(record.get("light_run"))
    style = read_json(job / "stages" / "asr.json")["provenance"]["transcript_style"]
    words = read_json(job / "stages" / "words.json")["words"]
    structure = read_json(job / "stages" / "structure.json")
    questions = {q["id"]: q for q in structure["questions"]}
    path = job / "stages" / "alignment.json"
    aligned = read_json(path)["utterances"] if path.is_file() else []
    # Rhythm needs phone lengths, which only MFA gives; utterances aligned another way have none.
    phones = {u["id"]: u["phones"] for u in aligned}
    reason, fitted = _not_checked(job, light)
    pitch, no_pitch = _tool(job, "praat", "pitch", light)
    tracks = {t: _tool(job, t, "pitch", light)[0] for t in fusion.TOOLS}
    voice, no_voice = _tool(job, "praat", "voice-quality", light)
    loud, no_loud = _tool(job, "opensmile", "loudness", light)
    parse, no_parse = _tool(job, "nlp", "grammar", light)
    parsed = parse["answers"] if parse else {}
    clause_of = {i: c for row in parsed.values() for i, c in row["clause_of"]} if parse else None
    parsed_words: ParsedWords = (
        {i for row in parsed.values() for i in row["names"]},
        {i: lemma for row in parsed.values() for i, lemma in row["lemmas"]},
    )
    out = {}
    for candidate in structure["candidates"]:
        speech = [(a, b) for a, b in structure["speech"][candidate["id"]]]  # the candidate's own voice only
        answers = [a for a in structure["answers"] if a["candidate"] == candidate["id"]]
        ids = {a["id"] for a in answers}
        mine = [u for u in structure["utterances"] if u["answer_id"] in ids]
        texts = _texts(mine, words)
        checked, listed = _pronunciation(reason, answers, structure["utterances"], words, fitted)
        heard = pitch["candidates"][candidate["id"]] if pitch else {"scopes": {}, "units": []}
        loudness = loud["candidates"][candidate["id"]]["loudness"] if loud else {}
        examples, counts = grammar.examples([f for a in answers if parse for f in parsed[a["id"]]["found"]])
        clause_scopes = {}
        scopes = {}
        by_tool = {}
        for scope, chosen in _scopes(answers).items():
            row = _measures(
                chosen, structure["utterances"], words, speech, mode, style, questions, phones,
                (clause_of, no_parse), parsed_words,
            )  # fmt: skip
            if parse:
                clause_scopes[scope] = grammar.clause_counts([parsed[a["id"]] for a in chosen])
            pitched = prosody.scope_values(heard["scopes"].get(scope), heard["units"], no_pitch)
            row["values"] |= _plain(pitched) | _plain(prosody.loudness_values(loudness.get(scope), no_loud))
            fused = _fused(tracks, candidate["id"], scope, no_pitch)
            row["values"] |= fused["values"]
            row["fusion"] = fused["fusion"]
            by_tool[scope] = fused["details"]
            scopes[scope] = row
        rows = voice["candidates"][candidate["id"]]["voice"] if voice else {}
        out[candidate["id"]] = {
            "scopes": scopes,
            "answers": {a["id"]: _plain(prosody.voice_values(rows.get(a["id"]), no_voice)) for a in answers},
            "pointers": {
                "repetitions": [asdict(p) for p in disfluency.repetitions(texts)],
                "self_corrections": [asdict(p) for p in disfluency.self_corrections(texts)],
                "listed_phrases": [asdict(p) for p in disfluency.listed_phrases(texts)],
                "discourse_markers": [asdict(p) for p in discourse.markers(texts).found],
                "rare_words": [asdict(p) for p in _lexical(mine, words, parsed_words).rare],
                "structures": [asdict(p) for p in examples],
                "words_to_check": listed,
            },
            "pronunciation": checked,
            "pitch_by_tool": by_tool,
            "grammar": {
                "missing": asdict(no_parse) if no_parse else None, "counts": counts, "scopes": clause_scopes,
            },
        }  # fmt: skip
    return {"status": "ok", "candidates": out}
