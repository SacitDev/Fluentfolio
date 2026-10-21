"""The nlp step: spaCy's English parser finds each answer's clauses and a few sentence structures.

The model is the en_core_web_sm wheel, unpacked into the models folder and loaded by path, so installing
it never changes the app's own packages. It runs on the CPU in a process of its own. The parser is trained
on written text, so what it finds is a pointer to read in the transcript, never an error count.
"""

from __future__ import annotations

import tomllib
import warnings
from bisect import bisect_right
from collections.abc import Callable
from functools import cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fluentfolio.analyzers.measures import acoustic
from fluentfolio.core.errors import SetupError
from fluentfolio.core.files import read_json
from fluentfolio.core.paths import models_dir
from fluentfolio.models import catalogue
from fluentfolio.models.download import require, unpacked

if TYPE_CHECKING:
    from spacy.tokens import Doc, Token

COMPONENT = "spacy-en-core-web-sm"
VERBS = ("VERB", "AUX")
Row = dict[str, Any]


@cache
def rules() -> dict[str, list[str]]:
    """The closed set of labels and words in the package's grammar.toml."""
    text = resources.files("fluentfolio.data").joinpath("grammar.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)


def model_folder(models: Path) -> Path:
    """The unpacked pipeline: the folder inside the wheel that holds its config.cfg."""
    component = catalogue.get(COMPONENT)
    require(component, models)
    found = sorted(unpacked(component, models).glob("*/*/config.cfg"))
    if not found:
        raise SetupError("E_MODEL_MISSING", COMPONENT)
    return found[0].parent


def _finite(t: Token) -> bool:
    finite = rules()["finite"]
    return t.tag_ in finite or any(c.dep_ in ("aux", "auxpass") and c.tag_ in finite for c in t.children)


def clauses(doc: Doc) -> tuple[set[int], set[int]]:
    """The token positions of the main clauses' heads and of the finite subordinate clauses' heads."""
    r = rules()
    main, sub = set(), set()
    for t in doc:
        joined = t.dep_ in r["joined"] and any(c.dep_ in r["subjects"] for c in t.children)
        if t.dep_ == "ROOT" or (joined and t.pos_ in VERBS):
            main.add(t.i)
        elif t.dep_ in r["subordinate"] and t.pos_ in VERBS and _finite(t):
            sub.add(t.i)
    return main, sub


def clause_of(t: Token, heads: set[int]) -> int:
    """The clause a token belongs to: the nearest clause head above it, itself included."""
    while t.i not in heads and t.head.i != t.i:
        t = t.head
    return int(t.i)


def _speculation(t: Token) -> bool:
    r = rules()
    if t.lower_ in r["speculation"]:
        return True
    verb = t.head
    perfect = any(c.lemma_ == "have" and c.dep_ == "aux" and c.i > t.i for c in verb.children)
    return t.lower_ in r["speculation_before_be_or_have"] and (verb.lemma_ == "be" or perfect)


def _conditional(t: Token) -> bool:
    r = rules()
    if not (t.lower_ in r["conditional"] and t.dep_ == "mark" and t.head.dep_ == "advcl"):
        return False
    return not any(c.dep_ == "mark" and c.lower_ in r["comparison"] for c in t.head.children)


def _passive(t: Token) -> bool:
    # The small model also marks "'s" in "he's been wanting" as a passive auxiliary.
    return t.dep_ == "auxpass" and t.head.tag_ in rules()["participle"] and t.head.lemma_ != "be"


def structures(doc: Doc) -> list[tuple[str, int, int, int]]:
    """Each structure found: its type, the token that shows it, and the first and last token of its words."""
    found = []
    for t in doc:
        if _conditional(t):
            found.append(("conditional", t.i, t.head.left_edge.i, t.head.right_edge.i))
        if t.dep_ == "relcl":
            found.append(("relative clause", t.i, t.left_edge.i, t.right_edge.i))
        if _passive(t):
            found.append(("passive", t.i, min(t.i, t.head.i), max(t.i, t.head.i)))
        if t.tag_ == "MD" and _speculation(t):
            found.append(("modal of speculation", t.i, min(t.i, t.head.i), max(t.i, t.head.i)))
    return found


def parse(nlp: Callable[[str], Doc], ids: list[int], words: list[Row], utterances: list[Row]) -> Row:
    """One answer: the clause of each word, the clause counts and the structures with where they are.

    Also the words tagged as names, and each word's base form (lemma) where it differs, both read from the
    word's first token, for the word-frequency profile. Hesitation sounds and cut-off words are left out of
    the parsed text, so they neither confuse the parser nor belong to a clause.
    """
    kept = [i for i in ids if not words[i]["is_filler"] and not words[i]["cut_off"]]
    starts, text = [], ""
    for i in kept:
        starts.append(len(text))
        text += words[i]["raw"] + " "
    doc = nlp(text)

    def word(token: int) -> int:
        return kept[bisect_right(starts, doc[token].idx) - 1]

    main, sub = clauses(doc)
    heads = main | sub
    clause: dict[int, int] = {}
    leading: dict[int, Token] = {}  # each word's first token
    for t in doc:
        if not (t.is_punct or t.is_space):
            clause.setdefault(word(t.i), clause_of(t, heads))
            leading.setdefault(word(t.i), t)
    found = []
    for kind, key, first, last in structures(doc):
        u = next(u for u in utterances if u["first_word"] <= word(key) <= u["last_word"])
        a, b = max(word(first), u["first_word"]), min(word(last), u["last_word"])  # within one utterance
        found.append({
            "type": kind, "utterance_id": u["id"], "at": [a - u["first_word"], b - u["first_word"]],
            "text": " ".join(words[j]["raw"] for j in range(a, b + 1)),
        })  # fmt: skip
    return {
        "clause_of": [[i, c] for i, c in sorted(clause.items())],
        "units": len(main), "clauses": len(heads), "subordinate": len(sub), "found": found,
        "names": [i for i, t in sorted(leading.items()) if t.pos_ == "PROPN"],
        "lemmas": [[i, t.lemma_.lower()] for i, t in sorted(leading.items()) if t.lemma_.lower() != t.lower_],
    }  # fmt: skip


def run(job: Path, progress: Callable[[int], None]) -> Row:
    """The parse of every answer in the recording."""
    settings = read_json(job / "job.json")["settings"]
    if not acoustic.switched_on(settings, COMPONENT, "grammar"):
        return {"status": "disabled", "answers": {}}
    folder = model_folder(models_dir())
    import spacy

    # Known limit: the model declares spaCy 3.7, which cannot run with numpy 2; the tests check its
    # parses with this spaCy instead. Revisit if a model for this spaCy is published on Hugging Face.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\[W095\]")
        nlp = spacy.load(folder)
    words = read_json(job / "stages" / "words.json")["words"]
    structure = read_json(job / "stages" / "structure.json")
    out = {}
    for n, a in enumerate(structure["answers"]):
        mine = [u for u in structure["utterances"] if u["answer_id"] == a["id"]]
        ids = [i for u in mine for i in range(u["first_word"], u["last_word"] + 1)]
        out[a["id"]] = parse(nlp, ids, words, mine)
        progress(100 * (n + 1) // len(structure["answers"]))
    version = f"spaCy {spacy.__version__}, en_core_web_sm {nlp.meta['version']}"
    return {
        "status": "ok", "provenance": {"tool": COMPONENT, "version": version}, "analyzers": ["grammar"],
        "answers": out,
    }  # fmt: skip
