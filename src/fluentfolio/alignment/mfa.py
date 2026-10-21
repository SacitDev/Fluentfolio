"""Word and phone times from MFA: the corpus and dictionary of one session, its command, and its result.

MFA runs once per session on the candidates' utterances, one speaker folder per candidate, so its speaker
adaptation sees each voice apart. Each utterance then takes MFA's times if the guard passes, else the CTC
aligner's, else the speech model's own.
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from fluentfolio.alignment import guard
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import mfa_root
from fluentfolio.models import catalogue
from fluentfolio.models.catalogue import SYSTEM

RATE = 16000
PAD_MS = 200  # audio kept around an utterance's words, never more than half the gap to a neighbour word
MAX_CUT_MS = 30000
IS_WINDOWS = sys.platform == "win32"
DICTIONARY = "english_us_arpa.dict"
ACOUSTIC = "english_us_arpa_acoustic.zip"
G2P = "english_us_arpa_g2p.zip"
EXTRA = {"mm": "M", "mhm": "M HH M"}  # hesitation sounds the dictionary lacks, in its phone set
SILENCE = ("", "sil", "sp", "spn")
ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen"
).split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
LARGE = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"))
ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
            "nine": "ninth", "twelve": "twelfth"}  # fmt: skip
PIECE = re.compile(r"\d+(?:[.,]\d+)*(?:st|nd|rd|th)?|[a-z]+(?:'[a-z]+)*")
INTERVAL = re.compile(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"')

Row = dict[str, Any]
Interval = tuple[int, int, str]  # start and end in ms in the recording, and the label


def number_words(n: int) -> list[str]:
    """A whole number in words, as a speaker of English reads it: 2010 is "two thousand ten"."""
    if n < 20:  # noqa: PLR2004
        return [ONES[n]]
    if n < 100:  # noqa: PLR2004
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    if n < 1000:  # noqa: PLR2004
        return [ONES[n // 100], "hundred"] + (number_words(n % 100) if n % 100 else [])
    size, name = next((s, w) for s, w in LARGE if n >= s)
    return [*number_words(n // size), name] + (number_words(n % size) if n % size else [])


def _ordinal(word: str) -> str:
    if word in ORDINALS:
        return ORDINALS[word]
    return word[:-1] + "ieth" if word.endswith("y") else word + "th"


def _spoken(digits: str) -> list[str]:
    """Digits as words: "1,000" one thousand, "3.5" three point five, "21st" twenty first."""
    ordinal = digits[-2:] in ("st", "nd", "rd", "th")
    digits = digits[:-2] if ordinal else digits
    whole, _, rest = digits.replace(",", "").partition(".")
    words = number_words(int(whole))
    if rest:
        words += ["point", *(ONES[int(d)] for d in rest if d.isdigit())]
    elif ordinal:
        words[-1] = _ordinal(words[-1])
    return words


def tokens(text: str) -> list[str]:
    """The words the aligners get for one transcript word: lower-case ASCII letters, numbers spelt out.

    A hyphen or another mark splits a word ("well-known" is two); accents are dropped ("cafe").
    """
    plain = unicodedata.normalize("NFKD", text.replace("’", "'")).encode("ascii", "ignore").decode().lower()
    out: list[str] = []
    for piece in PIECE.findall(plain):
        out += _spoken(piece) if piece[0].isdigit() else [piece]
    return out


@dataclass(frozen=True)
class Cut:
    """One candidate utterance as both aligners see it: its audio window and the words in it."""

    id: str
    speaker: str  # the candidate, which names MFA's speaker folder
    wav: str  # the input file it comes from: audio16k.wav, or a person's own track
    start_ms: int
    end_ms: int
    words: list[int]  # the transcript word indices
    tokens: list[list[str]]  # per word, the words the aligners get; none for a cut-off word


def cuts(job: Path) -> list[Cut]:
    """Every candidate utterance, with its words padded by up to 200 ms of audio on each side."""
    words = read_json(job / "stages" / "words.json")["words"]
    structure = read_json(job / "stages" / "structure.json")
    duration = int(read_json(job / "stages" / "media.json")["duration_ms"])
    owner = {a["id"]: a["candidate"] for a in structure["answers"]}
    found = []
    for u in structure["utterances"]:
        first, last = u["first_word"], u["last_word"]
        start = min(words[i]["start_ms"] for i in range(first, last + 1))
        end = max(words[i]["end_ms"] for i in range(first, last + 1))
        before = words[first - 1]["end_ms"] if first > 0 else 0
        after = words[last + 1]["start_ms"] if last + 1 < len(words) else duration
        pad = min(PAD_MS, max(0, (MAX_CUT_MS - (end - start)) // 2))
        track = words[first].get("track")
        found.append(Cut(
            u["id"], owner[u["answer_id"]], "audio16k.wav" if track is None else f"track_{track}.wav",
            max(0, start - min(pad, max(0, (start - before) // 2))),
            min(duration, end + min(pad, max(0, (after - end) // 2))),
            list(range(first, last + 1)),
            [[] if words[i]["cut_off"] else tokens(words[i]["text"]) for i in range(first, last + 1)],
        ))  # fmt: skip
    return found


def mini_dictionary(full: Path, needed: set[str]) -> tuple[list[str], set[str]]:
    """The dictionary lines of the words said, plus hesitation sounds it lacks; and the words it lacks.

    MFA guesses a pronunciation for each word it lacks (G2P), so such a word is never a word to check.
    """
    lines, found = [], set()
    with full.open(encoding="utf-8") as source:
        for line in source:
            if line.split("\t", 1)[0] in needed:
                lines.append(line.rstrip("\r\n"))
                found.add(line.split("\t", 1)[0])
    for word, phones in EXTRA.items():
        if word in needed and word not in found:
            lines.append(f"{word}\t{phones}")
            found.add(word)
    return lines, needed - found


@dataclass(frozen=True)
class Program:
    """MFA's command line, its folders for PATH, its own variables, where it runs and what it leaves."""

    argv: list[str]
    path: list[Path]
    env: dict[str, str]
    cwd: Path
    scratch: Path


def env_dir(app: Path) -> Path:
    return app / "envs" / "mfa"


def python(env: Path) -> Path:
    return env / "python.exe" if IS_WINDOWS else env / "bin" / "python"


def env_path(env: Path) -> list[Path]:
    """The env's folders for PATH, where its programs find their libraries."""
    if not IS_WINDOWS:
        return [env / "bin"]
    library = env / "Library"
    return [
        env,
        library / "mingw-w64" / "bin",
        library / "usr" / "bin",
        library / "bin",
        env / "Scripts",
        env / "bin",
    ]


def program(app: Path, models: Path, job_id: str) -> Program:
    """MFA through its env's own python, never a launcher script, with speaker adaptation on."""
    env = env_dir(app)
    root = mfa_root(app, job_id)
    folder = models / "mfa"
    # MFA adds fresh random noise to the audio before its features by default, which moves a word by a frame
    # now and then between two runs; without it the same recording always gives the same times.
    argv = [
        str(python(env)), "-I", "-m", "montreal_forced_aligner", "align", str(root / "corpus"),
        str(root / "dictionary.dict"), str(folder / ACOUSTIC), str(root / "out"),
        "--g2p_model_path", str(folder / G2P), "--clean", "--overwrite", "--num_jobs", "2",
        "--no_use_postgres", "--dither", "0",
    ]  # fmt: skip
    # The profile folders point inside the job's MFA folder, so MFA never writes to the user's own profile.
    home = root / "home"
    variables = {"MFA_ROOT_DIR": str(root / "root"), "HOME": str(home), "USERPROFILE": str(home)}
    if IS_WINDOWS:
        variables |= {
            "APPDATA": str(home / "AppData" / "Roaming"),
            "LOCALAPPDATA": str(home / "AppData" / "Local"),
        }
    return Program(argv, env_path(env), variables, root, root)


def version(app: Path) -> str | None:
    """The MFA version installed in the env, from the env's own package record."""
    found = sorted((env_dir(app) / "conda-meta").glob("montreal-forced-aligner-*.json"))
    return found[0].name.removeprefix("montreal-forced-aligner-").split("-")[0] if found else None


def experimental() -> bool:
    """MFA did not pass its word-timing check on this system, so its times are called experimental here."""
    return SYSTEM in catalogue.get("mfa").experimental_on


def prepare(job: Path, app: Path, models: Path) -> Program | Row:
    """Write the corpus and the dictionary and return MFA's command, or why MFA does not run."""
    record = read_json(job / "job.json")
    if not catalogue.switched_on(record["settings"].get("components", {}), "mfa", SYSTEM):
        return {"status": "disabled"}
    tool = program(app, models, record["job_id"])
    files = [Path(tool.argv[0]), *(models / "mfa" / name for name in (DICTIONARY, ACOUSTIC, G2P))]
    if not all(f.is_file() for f in files):
        return {"status": "unavailable", "code": "E_COMPONENT_UNAVAILABLE"}
    found = [c for c in cuts(job) if any(c.tokens)]
    if not found:
        return {"status": "skipped"}
    corpus = tool.scratch / "corpus"
    for c in found:
        folder = corpus / c.speaker
        folder.mkdir(parents=True, exist_ok=True)
        audio, _ = sf.read(
            str(job / "input" / c.wav), start=c.start_ms * RATE // 1000, stop=c.end_ms * RATE // 1000,
            dtype="int16",
        )  # fmt: skip
        sf.write(str(folder / f"{c.id}.wav"), audio, RATE, subtype="PCM_16")
        (folder / f"{c.id}.lab").write_text(" ".join(t for w in c.tokens for t in w) + "\n", encoding="utf-8")
    lines, _ = mini_dictionary(models / "mfa" / DICTIONARY, {t for c in found for w in c.tokens for t in w})
    (tool.scratch / "dictionary.dict").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tool.scratch / "root").mkdir(parents=True, exist_ok=True)
    return tool


def textgrid(path: Path, offset_ms: int) -> dict[str, list[Interval]]:
    """The words and phones tiers of a long-format TextGrid, in ms in the recording."""
    text = path.read_text(encoding="utf-8")
    tiers = {}
    for name in ("words", "phones"):
        head = f'name = "{name}"'
        if head not in text:
            raise ValueError(f"no {name} tier")
        body = text.split(head, 1)[1].split("item [", 1)[0]
        tiers[name] = [
            (offset_ms + round(float(a) * 1000), offset_ms + round(float(b) * 1000), label)
            for a, b, label in INTERVAL.findall(body)
        ]
    return tiers


def match(expected: list[list[str]], found: list[Interval]) -> list[tuple[int, int] | None] | None:
    """Each transcript word's span from MFA's word intervals, or None when the two word lists differ."""
    said = [w for w in found if w[2] not in SILENCE]
    spans: list[tuple[int, int] | None] = []
    k = 0
    for word in expected:
        start = end = None
        for token in word:
            if k == len(said):
                return None
            a, b, label = said[k]
            if label.lower() not in (token, "<unk>"):
                return None
            start, end, k = a if start is None else start, b, k + 1
        spans.append(None if start is None or end is None else (start, end))
    return spans if k == len(said) else None


def analysis(out: Path) -> dict[str, float | None]:
    """MFA's phone-duration deviation per utterance file, from its alignment_analysis.csv."""
    path = out / "alignment_analysis.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as rows:
        return {
            Path(r["file"]).stem: float(r["phone_duration_deviation"])
            if r["phone_duration_deviation"]
            else None
            for r in csv.DictReader(rows)
        }


def _mfa_words(
    c: Cut,
    out: Path,
    deviations: dict[str, float | None],
    asr: list[tuple[int, int]],
    speech: list[list[int]],
) -> tuple[list[tuple[int, int] | None], list[Interval]] | str:
    """MFA's word spans and phones for one utterance, or the reason they are not used."""
    grid = out / c.speaker / f"{c.id}.TextGrid"
    if not grid.is_file():
        return "no_textgrid"  # MFA can exit 0 without one
    try:
        tiers = textgrid(grid, c.start_ms)
    except ValueError:  # a tier missing, or text that is not UTF-8
        return "bad_textgrid"
    spans = match(c.tokens, tiers["words"])
    if spans is None:
        return "words_differ"
    if c.id not in deviations:
        return "no_analysis"
    phones = [p for p in tiers["phones"] if p[2] not in SILENCE]
    timed = [(s, a) for s, a in zip(spans, asr, strict=True) if s is not None]
    unplaced = [a for s, a in zip(spans, asr, strict=True) if s is None]
    failed = guard.check(
        deviations[c.id], phones, [s for s, _ in timed], [a for _, a in timed], speech,
        (c.start_ms, c.end_ms), unplaced,
    )  # fmt: skip
    return failed or (spans, phones)


def finish(job: Path, app: Path, models: Path, ran: Row) -> Row:
    """Each utterance's word times and the method that gave them; MFA's phones where its times are used."""
    words = read_json(job / "stages" / "words.json")["words"]
    structure = read_json(job / "stages" / "structure.json")
    path = job / "stages" / "phones.json"
    found = read_json(path) if path.is_file() else {}
    ctc = {u["id"]: u["words"] for u in found.get("utterances", []) if u["status"] == "ok"}
    out = mfa_root(app, read_json(job / "job.json")["job_id"]) / "out"
    deviations = analysis(out) if ran["status"] == "ok" else {}
    found_cuts = cuts(job)
    g2p: set[str] = set()
    if ran["status"] == "ok":
        needed = {t for c in found_cuts for w in c.tokens for t in w}
        g2p = mini_dictionary(models / "mfa" / DICTIONARY, needed)[1]
    rows = []
    for c in found_cuts:
        asr = [(words[i]["start_ms"], words[i]["end_ms"]) for i in c.words]
        result = "not_run" if ran["status"] != "ok" or not any(c.tokens) else (
            _mfa_words(c, out, deviations, asr, structure["speech"][c.speaker])
        )  # fmt: skip
        phones: list[Interval] = []
        if not isinstance(result, str):
            method, (spans, phones), reason = "mfa", result, None
        elif c.id in ctc:
            method, spans, reason = "ctc", [None if s is None else (s[0], s[1]) for s in ctc[c.id]], result
        else:
            method, spans, reason = "asr", [None] * len(c.words), result
        rows.append({
            "id": c.id, "candidate": c.speaker, "method": method, "reason": reason,
            "words": [
                {"i": i, "start_ms": (s or a)[0], "end_ms": (s or a)[1],
                 "method": "unaligned" if not t else method if s else "asr", "g2p": any(x in g2p for x in t)}
                for i, s, a, t in zip(c.words, spans, asr, c.tokens, strict=True)
            ],
            "phones": [list(p) for p in phones],
        })  # fmt: skip
    aligned = sum(r["method"] == "mfa" for r in rows)
    # MFA switched on but not aligning every utterance it was given leaves a gap in the job.
    wanted = sum(any(c.tokens) for c in found_cuts)
    return {
        "status": "ok" if ran["status"] in ("disabled", "skipped") or aligned == wanted else "partial",
        "mfa": {
            **ran,
            "version": version(app) if ran["status"] in ("ok", "failed") else None,
            "aligned": aligned,
            "utterances": len(rows),
            "experimental": experimental(),
        },  # fmt: skip
        "guard": {"t1": guard.T1, "t2": guard.T2, "provisional": True},
        "utterances": rows,
    }
