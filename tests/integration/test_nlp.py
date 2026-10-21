"""The grammar parser with the real spaCy model, on sentences whose structures were worked out by hand."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fluentfolio.analyzers import nlp
from fluentfolio.app import facade
from fluentfolio.core.enums import JobState
from fluentfolio.core.errors import AppError
from fluentfolio.core.files import read_json
from fluentfolio.jobs import loop
from fluentfolio.models import catalogue, download

pytestmark = pytest.mark.slow

# Sentence, the structures in it, main clauses, subordinate clauses.
KNOWN = [
    ("If it rains tomorrow, I will stay at home.", ["conditional"], 1, 1),
    ("The book that I bought last week was written by my aunt.", ["relative clause", "passive"], 1, 1),
    ("It might rain later today.", ["modal of speculation"], 1, 0),
    ("They must have forgotten about the meeting.", ["modal of speculation"], 1, 0),
    ("You could ask your teacher for help.", [], 1, 0),
    ("I went home and my sister cooked dinner.", [], 2, 0),
    ("I want to travel because it helps me learn.", [], 1, 1),
    ("He pulled up a window as if the air were heavy.", [], 1, 1),
    ("He's been wanting to visit Spain for years.", [], 1, 0),
    ("The bridge has been built across the river.", ["passive"], 1, 0),
]
# The solo answer of the evidence tests: speech 1.0-4.0 s, 5.2-9.0 s and 9.4-14.0 s.
SEGMENTS = [(1000, 2000), (2100, 4000), (5200, 9000), (9400, 14000)]
SPOKEN = [
    ("Well,", 1000, 1300, 0), ("I", 1350, 1450, 0), ("think", 1500, 1800, 0), ("um", 1850, 2150, 0),
    ("reading", 2200, 2700, 0), ("is", 2750, 2900, 0), ("really", 2950, 3400, 0), ("useful.", 3450, 3950, 0),
    ("Because", 5200, 5600, 1), ("it", 5650, 5800, 1), ("it", 5850, 6000, 1), ("helps", 6050, 6500, 1),
    ("you", 6550, 6700, 1), ("learn", 6750, 7100, 1), ("new", 7150, 7400, 1),
    ("vocabulary,", 7450, 8900, 1),
    ("and", 9400, 9600, 1), ("also", 9650, 9900, 1), ("ideas", 9950, 10500, 1), ("about", 10550, 10800, 1),
    ("the", 10850, 10950, 1), ("world.", 11000, 11600, 1), ("So", 11700, 11900, 1),
    ("yeah.", 11950, 13900, 1),
]  # fmt: skip


@pytest.fixture(scope="module")
def parser(app_root: Path) -> Any:
    models = app_root / "models"
    try:
        folder = nlp.model_folder(models)
    except AppError:
        # Unpacked only when missing: unpacking again rewrites files that a parallel test worker may be
        # reading. A whole wheel is not fetched again.
        download.download(catalogue.get(nlp.COMPONENT), models)
        folder = nlp.model_folder(models)
    import spacy

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\[W095\]")
        return spacy.load(folder)


def one_sentence(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    words = [{"raw": w, "is_filler": False, "cut_off": False} for w in text.split()]
    return words, [{"id": "U001", "first_word": 0, "last_word": len(words) - 1}]


@pytest.mark.parametrize(("text", "kinds", "units", "sub"), KNOWN)
def test_the_model_gives_the_labels_the_rules_read(
    parser: Any, text: str, kinds: list[str], units: int, sub: int
) -> None:
    words, utterances = one_sentence(text)
    found = nlp.parse(parser, list(range(len(words))), words, utterances)
    print(text, [(f["type"], f["text"]) for f in found["found"]], found["units"], found["subordinate"])
    assert [f["type"] for f in found["found"]] == kinds
    assert (found["units"], found["subordinate"]) == (units, sub)


def test_the_clauses_of_a_conditional_sentence(parser: Any) -> None:
    words, utterances = one_sentence("If it rains tomorrow, I will stay at home.")
    found = nlp.parse(parser, list(range(len(words))), words, utterances)
    clause = dict(found["clause_of"])
    assert len({clause[i] for i in range(4)}) == 1 and len({clause[i] for i in range(4, 9)}) == 1
    assert clause[3] != clause[4]  # "tomorrow," and "I": the boundary between the two clauses
    (conditional,) = found["found"]
    assert (conditional["at"], conditional["text"]) == ([0, 3], "If it rains tomorrow,")


def test_the_parser_step_places_the_pauses_of_an_answer(
    tmp_path: Path, synthetic_job: ModuleType, app_root: Path
) -> None:
    job = synthetic_job.make(tmp_path, SPOKEN, SEGMENTS, 20_000)
    roots = loop.Roots(app=app_root, data=tmp_path / "data", models=app_root / "models")
    stages = [s for s in facade.plan("solo") if s.id == "nlp" or s.id not in synthetic_job.LEFT_OUT]
    record = loop.run_job(roots, synthetic_job.JOB_ID, stages, audio_min=5, no_progress_s=120)
    assert record.state is JobState.COMPLETED, record
    step = read_json(job / "analyzers" / "nlp.json")
    assert (step["status"], step["analyzers"]) == ("ok", ["grammar"])
    version = step["provenance"]["version"]
    assert version.startswith("spaCy 3.8.") and version.endswith("en_core_web_sm 3.7.1")
    evidence = read_json(job / "evidence.json")
    (c,) = evidence["candidates"]
    (m07,) = [m for m in c["measures"] if m["id"] == "M07"]
    # "useful." ends one sentence and "Because" starts another; "vocabulary," and "and also ideas" are one
    # clause, so 1 of the 2 pauses is inside a clause.
    assert (m07["name"], m07["value"], m07["n"]) == ("pauses inside a clause", 50.0, 2)
    (row,) = [t for t in evidence["manifest"]["tools"] if t["id"] == "nlp"]
    assert (row["status"], row["ran"], row["licence_class"]) == ("ok", "spacy-en-core-web-sm", "permissive")
    # Three units ("I think", "it helps", "yeah") and one subordinate clause ("reading is useful").
    x05 = next(p for p in c["pointers"] if p["id"] == "X05")
    assert x05["items"] == [] and x05["note"].startswith(
        "all answers: 1.3 clauses per unit, 1 subordinate clause."
    )
