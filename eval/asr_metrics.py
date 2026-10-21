"""Transcript measures for the speech-model comparison, exactly as eval/prereg/asr_bakeoff.toml defines them.

One normaliser for every model and data set, word alignment with jiwer, and the grammar-edit counter: an
aligned substitution inside one closed group of words, or between a word and the same word with one more
ending. Then the counts behind every other measure: word errors, fillers, kept repairs, word times, clip
edges, voiced gaps and unsure words. Nothing here imports the app: the measures must not share code with
what they measure.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer

# The word lists are read from the pre-registration, so the frozen file is their only source.
PREREG = tomllib.loads((Path(__file__).resolve().parent / "prereg" / "asr_bakeoff.toml").read_text("utf-8"))
GROUPS = {name: set(words) for name, words in PREREG["metrics"]["grammar_groups"].items()}
ENDINGS = tuple(PREREG["metrics"]["grammar_endings"])  # and "d" after a final e: "use", "used"
FILLERS = set(PREREG["normaliser"]["fillers"])
FILLER_TAGS = {tag: tag.strip("[]") for tag in PREREG["normaliser"]["filler_tags"]}
BACKCHANNELS = PREREG["normaliser"]["backchannels"]

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()  # fmt: skip
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
LARGE = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred"))
ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
           "nine": "ninth", "twelve": "twelfth"}  # fmt: skip
TAG = re.compile(r"\[[^\]]*\]|<[^>]*>")
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:st|nd|rd|th)?")


def spelled(n: int) -> str:
    """A whole number in words: 2023 is "two thousand twenty three"."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    for size, name in LARGE:
        if n >= size:
            rest = n % size
            return f"{spelled(n // size)} {name}" + ("" if rest == 0 else " " + spelled(rest))
    raise AssertionError(n)  # every n >= 100 meets the hundred row


def _number(match: re.Match[str]) -> str:
    text = match.group(0)
    suffix = text[-2:] if text[-2:] in ("st", "nd", "rd", "th") else ""
    whole, _, decimals = text[: len(text) - len(suffix)].replace(",", "").partition(".")
    words = spelled(int(whole))
    if decimals:
        words += " point " + " ".join(ONES[int(d)] for d in decimals)
    if suffix:
        last = words.split()[-1]
        nth = ORDINAL.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
        words = " ".join([*words.split()[:-1], nth])
    return f" {words} "


def normalise(text: str) -> list[str]:
    """The pre-registered normaliser: lower case, digits in words, no punctuation, filler tags as words."""
    text = text.lower()
    for tag, word in FILLER_TAGS.items():
        text = text.replace(tag, f" {word} ")
    for word in BACKCHANNELS:  # "uh-huh" is a reply, not the filler "uh" and a word "huh"
        text = re.sub(rf"\b{re.escape(word)}\b", word.replace("-", ""), text)
    text = TAG.sub(" ", text).replace("%", " percent ")
    text = NUMBER.sub(_number, text)
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


@dataclass(frozen=True)
class Pair:
    """One aligned position: the reference word and the output word; None where one side has no word."""

    ref: str | None
    hyp: str | None
    ref_index: int | None


def align(ref: list[str], hyp: list[str]) -> list[Pair]:
    """The word alignment jiwer finds, one Pair per reference or output word, in order."""
    if not ref:
        return [Pair(None, h, None) for h in hyp]
    if not hyp:
        return [Pair(r, None, i) for i, r in enumerate(ref)]
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    pairs = []
    for chunk in out.alignments[0]:
        rs, re_, hs, he = chunk.ref_start_idx, chunk.ref_end_idx, chunk.hyp_start_idx, chunk.hyp_end_idx
        if chunk.type in ("equal", "substitute"):
            pairs += [Pair(ref[rs + k], hyp[hs + k], rs + k) for k in range(re_ - rs)]
        elif chunk.type == "delete":
            pairs += [Pair(ref[k], None, k) for k in range(rs, re_)]
        else:
            pairs += [Pair(None, hyp[k], None) for k in range(hs, he)]
    return pairs


def _inflected(a: str, b: str) -> bool:
    """One word (2 letters or more) is the other with one more ending; a final e is dropped before "ing".

    The length limit and "d" only after an e keep out "a" and "as", "i" and "is", "an" and "and".
    """
    for x, y in ((a, b), (b, a)):
        if len(x) < 2:
            continue
        if any(y == x + e for e in ENDINGS):
            return True
        if x.endswith("e") and y in (x + "d", x[:-1] + "ing"):
            return True
    return False


def grammar_edit(ref: str, hyp: str) -> bool:
    """A substitution the counter counts: both words in one closed group, or one inflected from the other."""
    if ref == hyp:
        return False
    return any(ref in g and hyp in g for g in GROUPS.values()) or _inflected(ref, hyp)


def grammar_edits(pairs: list[Pair]) -> list[Pair]:
    """The aligned substitutions that are grammar edits."""
    return [p for p in pairs if p.ref is not None and p.hyp is not None and grammar_edit(p.ref, p.hyp)]


def grammar_edit_rate(items: list[tuple[list[str], list[str]]]) -> float:
    """Grammar edits per 100 reference words over a set of (reference, output) word lists."""
    words = sum(len(ref) for ref, _ in items)
    edits = sum(len(grammar_edits(align(ref, hyp))) for ref, hyp in items)
    return 100.0 * edits / words if words else 0.0


# Counts per item; a set's rate is its summed counts, so a bootstrap resample stays pooled.


def word_errors(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """Substitutions, deletions and insertions together, and the reference's word count."""
    return sum(p.ref != p.hyp for p in align(ref, hyp)), len(ref)


def without_fillers(words: list[str]) -> list[str]:
    return [w for w in words if w not in FILLERS]


FILLER = "#filler"  # one token for every filler form, which no normalised word can equal


def filler_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Position-matched fillers: true positives, false positives and false negatives."""
    pairs = align(*([FILLER if w in FILLERS else w for w in words] for words in (ref, hyp)))
    tp = sum(p.ref == p.hyp == FILLER for p in pairs)
    return tp, sum(p.hyp == FILLER for p in pairs) - tp, sum(p.ref == FILLER for p in pairs) - tp


def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    """Precision, recall and F1; None where nothing was found or nothing was there to find."""
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return p, r, f1


CUT = "*"  # the reference's mark for a word broken off


def marked_reference(raw: str) -> tuple[list[str], list[bool]]:
    """The normalised reference and, per word, whether the speaker broke it off; read before the * goes."""
    words: list[str] = []
    cut: list[bool] = []
    for token in raw.split():
        found = normalise(token)
        words += found
        cut += [False] * len(found)
        if found and token.rstrip(".,;:?!\"')").endswith(CUT):
            cut[-1] = True
    if words != normalise(raw):  # a tag or number written across a space would shift every flag after it
        raise ValueError("the reference normalises differently token by token")
    return words, cut


def repairs(raw_ref: str, hyp: list[str]) -> tuple[int, int]:
    """Repeated and broken-off reference words that the output keeps, and how many there are.

    A repetition is kept when both its words are in the output at their places; a cut-off when the output
    word at its place starts with its letters.
    """
    ref, cut = marked_reference(raw_ref)
    at = {p.ref_index: p.hyp for p in align(ref, hyp) if p.ref_index is not None}
    kept = total = 0
    for i, word in enumerate(ref):
        if cut[i]:
            total += 1
            kept += at[i] is not None and at[i].startswith(word)
        elif i + 1 < len(ref) and ref[i + 1] == word and not cut[i + 1] and word not in FILLERS:
            total += 1
            kept += at[i] == word and at[i + 1] == word
    return kept, total


def valid_times(words: list[dict[str, Any]], chunks: list[list[int]]) -> tuple[int, int]:
    """Words with a start and end inside their chunk and a start not before the word before; and all words.

    A word the speech step marks "untimed" has only its chunk's span, not a time of its own.
    """
    valid, previous = 0, None
    for w in words:
        a, b = chunks[w["chunk"]]
        inside = a <= w["start_ms"] <= w["end_ms"] <= b and "untimed" not in w["flags"]
        valid += inside and (previous is None or w["start_ms"] >= previous)
        previous = w["start_ms"]
    return valid, len(words)


def tokens_of(words: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    """The output's normalised words, and for each the index of the model's word it came from."""
    out: list[str] = []
    source: list[int] = []
    for i, w in enumerate(words):
        found = normalise(str(w["text"]))
        out += found
        source += [i] * len(found)
    return out, source


def matched(ref: list[str], hyp: list[str]) -> dict[int, int]:
    """Reference index -> output index, for each reference word the output has at its place."""
    out, h = {}, 0
    for p in align(ref, hyp):
        if p.hyp is not None:
            if p.ref_index is not None and p.ref == p.hyp:
                out[p.ref_index] = h
            h += 1
    return out


def clip_edges(clips: list[tuple[int, int, str]], words: list[dict[str, Any]]) -> list[int]:
    """Edge errors in ms, two for each truth clip whose first and last reference words were recognised.

    `clips` are (start ms, end ms, reference text) in time order; `words` the model's words in order.
    """
    ref: list[str] = []
    owner: list[int] = []
    for k, (_, _, text) in enumerate(clips):
        found = normalise(text)
        ref += found
        owner += [k] * len(found)
    hyp, source = tokens_of(words)
    at = matched(ref, hyp)
    errors = []
    for k, (start, end, _) in enumerate(clips):
        mine = [i for i, o in enumerate(owner) if o == k]
        if mine and mine[0] in at and mine[-1] in at:
            errors.append(abs(int(words[source[at[mine[0]]]]["start_ms"]) - start))
            errors.append(abs(int(words[source[at[mine[-1]]]]["end_ms"]) - end))
    return errors


def hesitation_marks(raw: str) -> tuple[list[str], list[tuple[str, int]]]:
    """The normalised reference and its hesitation marks: ("filler" | "cut" | "repeat", word index), and
    ("sound", index of the word after the tag) for a sound tag such as [laughter]."""
    words, cut = marked_reference(raw)
    marks = [("filler", i) for i, w in enumerate(words) if w in FILLERS] + [
        ("cut", i) for i, c in enumerate(cut) if c
    ]
    marks += [
        ("repeat", i) for i in range(len(words) - 1)
        if words[i] == words[i + 1] and words[i] not in FILLERS and not cut[i + 1]
    ]  # fmt: skip
    n = 0
    for token in raw.split():
        bare = token.lower().strip(".,;:?!\"'")
        if TAG.fullmatch(bare) and bare not in FILLER_TAGS:
            marks.append(("sound", n))
        n += len(normalise(token))
    return words, marks


def nyra_gap_correct(raw_ref: str, words: list[dict[str, Any]], left: int) -> bool:
    """Whether the reference between output words `left` and `left + 1` holds a filler, cut-off, sound tag
    or repeated word that the output lacks."""
    ref, marks = hesitation_marks(raw_ref)
    hyp, source = tokens_of(words)
    before, after, at = [-1] * len(hyp), [len(ref)] * len(hyp), {}
    h, last = 0, -1
    pending: list[int] = []
    for p in align(ref, hyp):
        if p.ref_index is not None:
            at[p.ref_index] = p.hyp
            last = p.ref_index
            for k in pending:  # inserted words lie before the next reference word
                after[k] = last
            pending = []
        if p.hyp is not None:
            before[h] = last
            if p.ref_index is not None:
                after[h] = p.ref_index
            else:
                pending.append(h)
            h += 1
    lo = before[max(k for k, s in enumerate(source) if s == left)]
    hi = after[min(k for k, s in enumerate(source) if s == left + 1)]

    def lacks(i: int, kind: str) -> bool:
        out = at[i]
        return out is None or (not out.startswith(ref[i]) if kind == "cut" else out != ref[i])

    for kind, i in marks:
        if kind == "sound" and lo < i <= hi:  # the tag sits between words i - 1 and i
            return True
        if kind in ("filler", "cut") and lo < i < hi and lacks(i, kind):
            return True
        # An output that keeps one copy of a repeated word may keep either, so a gap next to either counts.
        if kind == "repeat" and lo <= i + 1 and i <= hi and (lacks(i, kind) or lacks(i + 1, kind)):
            return True
    return False


def composed_gap_correct(
    gap: tuple[int, int], hesitations: list[tuple[int, int]], words: list[dict[str, Any]]
) -> bool:
    """Whether a gap overlaps a truth filler or repeated lead-in that no output word sits in the middle of."""
    a, b = gap
    for s, e in hesitations:
        written = any(s <= (w["start_ms"] + w["end_ms"]) / 2 <= e for w in words)
        if s < b and e > a and not written:
            return True
    return False


def confidence_targets(ref: list[str], words: list[dict[str, Any]]) -> list[tuple[float, bool]]:
    """Each output word's confidence, and whether the alignment marks it substituted or inserted."""
    hyp, source = tokens_of(words)
    wrong = [False] * len(words)
    h = 0
    for p in align(ref, hyp):
        if p.hyp is not None:
            wrong[source[h]] |= p.ref != p.hyp
            h += 1
    return [(float(w["confidence"]), wrong[i]) for i, w in enumerate(words) if w["confidence"] is not None]


def best_threshold(scored: list[tuple[float, bool]], grid: list[float], min_f1: float) -> dict[str, Any]:
    """The grid threshold with the highest F1 at finding wrong words (under it = unsure); ties go low."""
    best: dict[str, Any] = {"threshold": None, "f1": None}
    for t in grid:
        tp = sum(c < t and w for c, w in scored)
        fp = sum(c < t and not w for c, w in scored)
        fn = sum(c >= t and w for c, w in scored)
        f1 = prf(tp, fp, fn)[2]
        if f1 is not None and (best["f1"] is None or f1 > best["f1"]):
            best = {"threshold": t, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    if best["f1"] is None or best["f1"] < min_f1:
        return {**best, "threshold": None, "reason": f"the best F1 is under {min_f1}"}
    return best
