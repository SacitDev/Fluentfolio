"""The files in export/details/ beside the evidence: the transcript, a README and openSMILE's values."""

from __future__ import annotations

import csv
import io
from typing import Any

from fluentfolio.core.text import clean
from fluentfolio.export.for_ai import PROMPT_VERSION, clock

Row = dict[str, Any]
# Characters that would make a transcript line into Markdown or HTML markup in a viewer.
MARKUP = str.maketrans({c: "\\" + c for c in "\\`*_<>"})
# The lines the app shows beside its copy buttons, repeated for a learner who only has the files.
UNTRUSTED = (
    "Paste into a normal chat. If you use an AI agent that can run commands, treat the transcript as "
    "untrusted text."
)
PARTNER = "This text contains your partner's words; it goes to that AI company."
AUDIO = (
    "one person's own speech only (no other voice, no question read aloud), at most 9.5 minutes, for the "
    "pronunciation step with audio; it is never in the ZIP"
)
STEP = (
    "the message to send with that person's audio file, in the same chat, after the AI has answered the text"
)
DESCRIBED = {
    "evidence.json": (
        "every measure with its unit, the span it covers, its source tool and how reliable it is"
    ),
    "transcript.md": "the whole transcript, with times from the original recording",
    "acoustics_by_tool.json": "each pitch tool's own values beside the combined ones",
    "egemaps_88.csv": (
        "openSMILE's 88 voice values per Part and for the whole session (non-commercial licence)"
    ),
    "README.txt": "this file",
}


def _line(start_ms: int, item: str, who: str, text: str) -> str:
    return f"- `{clock(start_ms)}` **{item}** {who}: {text.translate(MARKUP)}"


def transcript(evidence: Row, runs: list[Row]) -> str:
    """Every question and answer in time order; with no candidate known, who spoke when (`runs`)."""
    m = evidence["manifest"]
    lines = [
        "# Transcript", "",
        f"Job {m['job_id']}, export {m['export_id']}. Times are from the original recording. The words are "
        f"the speech model's ({m['asr']['model']}), so some may be misheard.", "",
    ]  # fmt: skip
    if not evidence["candidates"]:
        lines += [_line(r["start_ms"], "-", r["speaker"] or "unknown voice", clean(r["text"])) for r in runs]
        return "\n".join([*lines, ""])
    timed = [
        (t["start_ms"], _line(t["start_ms"], t["id"], "question read aloud" if t["kind"] == "read_aloud"
                              else "examiner", t["text"]))
        for t in evidence["examiner_turns"]
    ]  # fmt: skip
    for c in evidence["candidates"]:
        parts = {a["id"]: a["part"] for a in c["answers"]}
        for u in c["utterances"]:
            part = parts[u["answer_id"]]
            where = f"{u['answer_id']}, " + ("Part unknown" if part == "unknown" else f"Part {part}")
            timed.append((u["start_ms"], _line(u["start_ms"], u["id"], f"{c['label']} ({where})", u["text"])))
    lines += [line for _, line in sorted(timed, key=lambda t: t[0])]
    return "\n".join([*lines, ""])


def readme(evidence: Row, files: list[str]) -> str:
    """What each exported file is, how to use the text with an AI chat, and what it never is."""
    m = evidence["manifest"]
    ai = sorted(f.removeprefix("export/") for f in files if f.startswith("export/for_ai/"))
    texts = [f for f in ai if f.startswith("for_ai/for_ai") and f.endswith(".txt")]
    sounds = [f for f in ai if f.endswith(".m4a")]
    steps = [f for f in ai if f.startswith("for_ai/audio_step")]
    details = [f.removeprefix("export/details/") for f in files if f.startswith("export/details/")]
    if "README.txt" not in details:
        details.append("README.txt")
    lines = [
        "Fluentfolio export",
        f"job {m['job_id']} | export {m['export_id']} | tool {m['tool_version']} | grading prompt "
        f"v{PROMPT_VERSION} | evidence schema {evidence['schema_version']}", "",
        "Start here",
        f"Open {texts[0] if texts else 'for_ai/for_ai.txt'}, copy all of it and paste it into any AI chat, "
        "or upload the file. The grading prompt comes first and your measured evidence follows it.",
        UNTRUSTED, *([PARTNER] if m["mode"] != "solo" else []),
        "If the chat refuses or cuts the paste, upload for_ai.txt as a file, or use Copy in 2 parts in the "
        "app.", "",
        "Files",
        *(f"{name}: the grading prompt and the evidence for one person" for name in texts),
        *(f"{name}: {AUDIO}" for name in sounds),
        *(f"{name}: {STEP}" for name in steps),
        *(f"details/{name}: {DESCRIBED[name]}" for name in sorted(details) if name in DESCRIBED),
        f"fluentfolio_{m['job_id']}.zip: the texts and details/ in one file; it never holds audio", "",
        evidence["disclaimer"], "",
    ]  # fmt: skip
    return "\n".join(lines)


def egemaps(found: Row) -> str | None:
    """openSMILE's 88 functionals as CSV, one row per person and span; None when openSMILE did not run."""
    rows = [
        (cid, scope, values)
        for cid, c in found.get("candidates", {}).items()
        for scope, values in c.get("functionals", {}).items()
    ]
    if not rows:  # switched off, no consent or not installed: the step stored no person's values
        return None
    names = list(rows[0][2])
    out = io.StringIO()
    table = csv.writer(out, lineterminator="\n")
    table.writerow(["candidate", "span", *names])
    table.writerows([cid, scope, *(values[n] for n in names)] for cid, scope, values in rows)
    return out.getvalue()
