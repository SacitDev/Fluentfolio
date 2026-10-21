"""Grammar pointers and pause places: the rules on hand-made parses, where the test sets every label."""

from __future__ import annotations

from typing import Any

import pytest
from spacy.lang.en import English
from spacy.tokens import Doc

from fluentfolio.analyzers import nlp
from fluentfolio.analyzers.measures import grammar
from fluentfolio.analyzers.measures.values import Missing

# One token: text, part of speech, tag, label, head (its position in the sentence), lemma.
Token = tuple[str, str, str, str, int, str]
VOCAB = English().vocab  # English word attributes such as lower case, and no trained model


def doc(tokens: list[Token], spaces: list[bool] | None = None) -> Doc:
    words, pos, tags, deps, heads, lemmas = (list(column) for column in zip(*tokens, strict=True))
    return Doc(VOCAB, words=words, spaces=spaces, pos=pos, tags=tags, deps=deps, heads=heads, lemmas=lemmas)


# "If it rains I stay": a conditional clause inside the main clause.
IF_RAINS = [
    ("If", "SCONJ", "IN", "mark", 2, "if"), ("it", "PRON", "PRP", "nsubj", 2, "it"),
    ("rains", "VERB", "VBZ", "advcl", 4, "rain"), ("I", "PRON", "PRP", "nsubj", 4, "I"),
    ("stay", "VERB", "VBP", "ROOT", 4, "stay"),
]  # fmt: skip
# "The book that I bought was written by her": a relative clause and a passive.
BOOK = [
    ("The", "DET", "DT", "det", 1, "the"), ("book", "NOUN", "NN", "nsubjpass", 6, "book"),
    ("that", "PRON", "WDT", "dobj", 4, "that"), ("I", "PRON", "PRP", "nsubj", 4, "I"),
    ("bought", "VERB", "VBD", "relcl", 1, "buy"), ("was", "AUX", "VBD", "auxpass", 6, "be"),
    ("written", "VERB", "VBN", "ROOT", 6, "write"), ("by", "ADP", "IN", "agent", 6, "by"),
    ("her", "PRON", "PRP", "pobj", 7, "she"),
]  # fmt: skip
# "I want to go": the infinitive stays part of the main clause.
WANT = [
    ("I", "PRON", "PRP", "nsubj", 1, "I"), ("want", "VERB", "VBP", "ROOT", 1, "want"),
    ("to", "PART", "TO", "aux", 3, "to"), ("go", "VERB", "VB", "xcomp", 1, "go"),
]  # fmt: skip
# "I wonder if it works": "if" opens a reported question here, not a condition.
WONDER = [
    ("I", "PRON", "PRP", "nsubj", 1, "I"), ("wonder", "VERB", "VBP", "ROOT", 1, "wonder"),
    ("if", "SCONJ", "IN", "mark", 4, "if"), ("it", "PRON", "PRP", "nsubj", 4, "it"),
    ("works", "VERB", "VBZ", "ccomp", 1, "work"),
]  # fmt: skip
# "He left as if it rained": "as if" compares, it sets no condition.
AS_IF = [
    ("He", "PRON", "PRP", "nsubj", 1, "he"), ("left", "VERB", "VBD", "ROOT", 1, "leave"),
    ("as", "SCONJ", "IN", "mark", 5, "as"), ("if", "SCONJ", "IN", "mark", 5, "if"),
    ("it", "PRON", "PRP", "nsubj", 5, "it"), ("rained", "VERB", "VBD", "advcl", 1, "rain"),
]  # fmt: skip
# "It has been built": a passive in the perfect.
BEEN_BUILT = [
    ("It", "PRON", "PRP", "nsubjpass", 3, "it"), ("has", "AUX", "VBZ", "aux", 3, "have"),
    ("been", "AUX", "VBN", "auxpass", 3, "be"), ("built", "VERB", "VBN", "ROOT", 3, "build"),
]  # fmt: skip
# Two labels the small English model gives in speech: "'s" marked as a passive auxiliary in
# "He's been wanting" (an -ing form) and in "It's been on" (a form of "be" itself).
BEEN_WANTING = [
    ("He", "PRON", "PRP", "nsubjpass", 3, "he"), ("'s", "AUX", "VBZ", "auxpass", 3, "be"),
    ("been", "AUX", "VBN", "aux", 3, "be"), ("wanting", "VERB", "VBG", "ROOT", 3, "want"),
]  # fmt: skip
BEEN_ON = [
    ("It", "PRON", "PRP", "nsubjpass", 2, "it"), ("'s", "AUX", "VBZ", "auxpass", 2, "be"),
    ("been", "AUX", "VBN", "ROOT", 2, "be"), ("on", "ADP", "IN", "prep", 2, "on"),
]  # fmt: skip


def joined(subject: bool) -> list[Token]:
    """ "I came and she left" or "I came and left": a second main clause only with its own subject."""
    first = [("I", "PRON", "PRP", "nsubj", 1, "I"), ("came", "VERB", "VBD", "ROOT", 1, "come"),
             ("and", "CCONJ", "CC", "cc", 1, "and")]  # fmt: skip
    if subject:
        return [
            *first,
            ("she", "PRON", "PRP", "nsubj", 4, "she"),
            ("left", "VERB", "VBD", "conj", 1, "leave"),
        ]
    return [*first, ("left", "VERB", "VBD", "conj", 1, "leave")]


def modal(word: str, verb: str, perfect: bool = False) -> list[Token]:
    """ "It <modal> <verb>" or "It <modal> have <verb>"."""
    head = 3 if perfect else 2
    tokens = [("It", "PRON", "PRP", "nsubj", head, "it"), (word, "AUX", "MD", "aux", head, word)]
    if perfect:
        tokens.append(("have", "AUX", "VB", "aux", 3, "have"))
    lemma = {"been": "be", "gone": "go"}.get(verb, verb)
    return [
        *tokens,
        (verb, "VERB" if lemma != "be" else "AUX", "VBN" if perfect else "VB", "ROOT", head, lemma),
    ]


def kinds(tokens: list[Token]) -> list[str]:
    return [kind for kind, *_ in nlp.structures(doc(tokens))]


@pytest.mark.parametrize(
    ("tokens", "main", "sub"),
    [
        (IF_RAINS, {4}, {2}),
        (BOOK, {6}, {4}),
        (WANT, {1}, set()),
        (WONDER, {1}, {4}),
        (joined(subject=True), {1, 4}, set()),
        (joined(subject=False), {1}, set()),
    ],
)
def test_main_and_finite_subordinate_clauses(tokens: list[Token], main: set[int], sub: set[int]) -> None:
    assert nlp.clauses(doc(tokens)) == (main, sub)


def test_each_word_belongs_to_its_nearest_clause() -> None:
    d = doc(BOOK)
    heads = set().union(*nlp.clauses(d))
    assert [nlp.clause_of(t, heads) for t in d] == [6, 6, 4, 4, 4, 6, 6, 6, 6]
    d = doc(WANT)  # "to go" has no subject or tense of its own, so it is in the main clause
    assert [nlp.clause_of(t, set().union(*nlp.clauses(d))) for t in d] == [1, 1, 1, 1]


def test_structures_and_the_words_they_cover() -> None:
    assert nlp.structures(doc(IF_RAINS)) == [("conditional", 0, 0, 2)]
    assert nlp.structures(doc(BOOK)) == [("relative clause", 4, 2, 4), ("passive", 5, 5, 6)]
    assert kinds(WONDER) == [] and kinds(WANT) == []


@pytest.mark.parametrize(
    ("tokens", "found"),
    [(AS_IF, []), (BEEN_BUILT, [("passive", 2, 2, 3)]), (BEEN_WANTING, []), (BEEN_ON, [])],
)
def test_a_passive_needs_a_past_participle_and_as_if_is_no_condition(
    tokens: list[Token], found: list[tuple[str, int, int, int]]
) -> None:
    assert nlp.structures(doc(tokens)) == found


@pytest.mark.parametrize(
    ("tokens", "found"),
    [
        (modal("might", "rain"), True),
        (modal("may", "rain"), True),
        (modal("could", "be"), True),
        (modal("must", "be"), True),
        (modal("could", "gone", perfect=True), True),
        (modal("must", "been", perfect=True), True),
        (modal("could", "swim"), False),
        (modal("must", "go"), False),
        (modal("should", "be"), False),
    ],
)
def test_modals_of_speculation(tokens: list[Token], found: bool) -> None:
    assert kinds(tokens) == (["modal of speculation"] if found else [])


def word_rows(raws: list[str], fillers: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    return [{"raw": r, "is_filler": i in fillers, "cut_off": r.endswith("-")} for i, r in enumerate(raws)]


def test_a_parse_maps_tokens_back_to_the_transcript_words() -> None:
    # "If it um rains, I stay." The hesitation sound is left out of the parsed text; the comma and the full
    # stop are tokens of their own inside the words "rains," and "stay.".
    words = word_rows(["If", "it", "um", "rains,", "I", "stay."], fillers=(2,))
    tokens = [
        ("If", "SCONJ", "IN", "mark", 2, "if"), ("it", "PRON", "PRP", "nsubj", 2, "it"),
        ("rains", "VERB", "VBZ", "advcl", 5, "rain"), (",", "PUNCT", ",", "punct", 5, ","),
        ("I", "PRON", "PRP", "nsubj", 5, "I"), ("stay", "VERB", "VBP", "ROOT", 5, "stay"),
        (".", "PUNCT", ".", "punct", 5, "."),
    ]  # fmt: skip
    built = doc(tokens, spaces=[True, True, False, True, True, False, True])

    def parser(text: str) -> Doc:
        assert text == built.text == "If it rains, I stay. "
        return built

    one = [{"id": "U001", "first_word": 0, "last_word": 5}]
    found = nlp.parse(parser, list(range(6)), words, one)
    assert found["clause_of"] == [[0, 2], [1, 2], [3, 2], [4, 5], [5, 5]]  # no clause for the hesitation
    assert (found["units"], found["clauses"], found["subordinate"]) == (1, 2, 1)
    assert found["found"] == [{"type": "conditional", "utterance_id": "U001", "at": [0, 3],
                               "text": "If it um rains,"}]  # fmt: skip
    # Word 3, "rains,", has the base form "rain"; "I" differs from its lemma only in case. No name.
    assert (found["names"], found["lemmas"]) == ([], [[3, "rain"]])
    # A structure that runs on into the next utterance is listed in the utterance where it starts.
    two = [{"id": "U001", "first_word": 0, "last_word": 2}, {"id": "U002", "first_word": 3, "last_word": 5}]
    (example,) = nlp.parse(parser, list(range(6)), words, two)["found"]
    assert (example["utterance_id"], example["at"], example["text"]) == ("U001", [0, 2], "If it um")


def test_a_parse_gives_the_words_tagged_as_names_by_their_first_token() -> None:
    # "Anna's aunt lives in New York." Word 0 is split into "Anna" and "'s"; "New York" is two words.
    words = word_rows(["Anna's", "aunt", "lives", "in", "New", "York."])
    tokens = [
        ("Anna", "PROPN", "NNP", "poss", 2, "Anna"), ("'s", "PART", "POS", "case", 0, "'s"),
        ("aunt", "NOUN", "NN", "nsubj", 3, "aunt"), ("lives", "VERB", "VBZ", "ROOT", 3, "live"),
        ("in", "ADP", "IN", "prep", 3, "in"), ("New", "PROPN", "NNP", "compound", 6, "New"),
        ("York", "PROPN", "NNP", "pobj", 4, "York"), (".", "PUNCT", ".", "punct", 3, "."),
    ]  # fmt: skip
    built = doc(tokens, spaces=[False, True, True, True, True, True, False, True])
    one = [{"id": "U001", "first_word": 0, "last_word": 5}]
    found = nlp.parse(lambda text: built, list(range(6)), words, one)
    assert (found["names"], found["lemmas"]) == ([0, 4, 5], [[2, "live"]])


def test_pauses_inside_and_between_clauses() -> None:
    # Words start at 0, 1000, 2000 and 3000 ms; the first two are one clause, the last two another.
    words = [(0, "I", 1), (1000, "think,", 1), (2000, "it", 3), (3000, "works.", 3)]
    pauses = [(600, 900), (1600, 1900), (2500, 2600), (3500, 3800), (-400, -100)]
    assert grammar.locate(pauses, words, parsed=True) == [True, False, True, None, None]
    # Without a parse, only a sentence end in the punctuation makes a boundary; the comma does not.
    plain = [(s, text, None) for s, text, _ in words]
    assert grammar.locate(pauses, plain, parsed=False) == [True, True, True, None, None]
    # A word that starts at the pause's middle comes after the pause; one that starts before it, before.
    assert grammar.locate([(1000, 2999)], words, parsed=True) == [False]  # middle 1999.5
    assert grammar.locate([(1000, 3000)], words, parsed=True) == [False]  # middle 2000: "it" is after
    assert grammar.locate([(1000, 3001)], words, parsed=True) == [True]  # middle 2000.5: "it" is before


def test_the_share_of_pauses_inside_a_clause() -> None:
    found = grammar.share([True, False, True, None], None)["pauses_in_clause"]
    assert (found.value, found.unit, found.n) == (66.7, "%", 3)
    assert found.note == "3 of 4 pauses have a word on both sides; only these count."
    assert grammar.share([True, False], None)["pauses_in_clause"].note is None
    why = Missing("tool_failed", "spaCy's grammar parser did not run")
    plain = grammar.share([True, False], why)
    assert list(plain) == ["internal_pauses"] and plain["internal_pauses"].value == 50.0
    assert "No clause boundaries (spaCy's grammar parser did not run)" in str(plain["internal_pauses"].note)
    none = grammar.share([None], None)["pauses_in_clause"]
    assert (none.value, none.missing and none.missing.code, none.n) == (None, "no_pauses", 0)


def test_clause_counts_round_half_up_and_need_a_unit() -> None:
    rows = [{"units": 2, "clauses": 2, "subordinate": 0}, {"units": 2, "clauses": 3, "subordinate": 1}]
    assert grammar.clause_counts(rows) == {"clauses_per_unit": 1.3, "subordinate": 1}  # 5 / 4 = 1.25
    assert grammar.clause_counts([{"units": 0, "clauses": 0, "subordinate": 0}])["clauses_per_unit"] is None


def test_two_examples_of_each_structure_and_every_count() -> None:
    found = [{"type": t, "utterance_id": f"U00{k}", "at": [0, 1], "text": "x y"}
             for k, t in enumerate(["passive", "conditional", "passive", "passive"], start=1)]  # fmt: skip
    listed, counts = grammar.examples(found)
    assert [(p.utterance_id, p.text) for p in listed] == [
        ("U001", "passive: x y"), ("U002", "conditional: x y"), ("U003", "passive: x y"),
    ]  # fmt: skip
    assert counts == {"conditional": 1, "relative clause": 0, "passive": 3, "modal of speculation": 0}
