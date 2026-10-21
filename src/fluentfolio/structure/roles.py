"""Who asks and who answers: speaker runs, backchannels, turns, and the roles in interview and role-swap.

Roles come from what each person does (asking, then a long answer from the other) and how much each
talks; a name or a voice is never used. Every result carries a confidence word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

from fluentfolio.structure.questions import Word, asking

BACKCHANNEL_MS = 1000  # a run under 1 s, or of 3 words or fewer, beside the other person's long turn
BACKCHANNEL_WORDS = 3
LONG_MS = 5000  # an answer of 5 s or more; also the other person's speech around a backchannel
HIGH_SHARE = (
    0.4  # an examiner who talks 40 % of the time or less, with both signals agreeing: high confidence
)
ANSWER_CAP_S = 30  # in a vote, a long answer counts as 30 s at most, so one long turn cannot outvote a phase
# The cost of a swap in vote weight (question-likeness x s): one short answer's worth. A larger cost can
# outweigh a whole phase of short answers; "once" relies on its 3-exchange or 90 s floor instead.
PENALTY = {"once": 5.0, "per-question": 1.0}
MIN_EXCHANGES = 3  # "once or per Part": a phase needs 3 exchanges or 90 s
MIN_PHASE_MS = 90000
FREQUENT = 3  # this many stretches of votes overruled by the smoothing raise W_FREQUENT_SWAPS
SURE, FAIR = 0.8, 0.6  # the share of a phase's vote weight that agrees with it: high, medium, else low


@dataclass
class Run:
    speaker: str | None
    ids: list[int]
    start_ms: int
    end_ms: int
    backchannel: bool = False

    @property
    def ms(self) -> int:
        return self.end_ms - self.start_ms


def runs(words: list[Word]) -> list[Run]:
    """Consecutive words of one speaker, in time order."""
    out: list[Run] = []
    for i, w in enumerate(words):
        if out and out[-1].speaker == w["speaker"]:
            out[-1].ids.append(i)
            out[-1].end_ms = max(out[-1].end_ms, w["end_ms"])
        else:
            out.append(Run(w["speaker"], [i], w["start_ms"], w["end_ms"]))
    return out


def mark_backchannels(found: list[Run], pair: list[str], words: list[Word]) -> None:
    """A short run inside or next to 5 s or more of the other person's speech is a backchannel ("mm-hmm").

    A short question ("Why?") and a short reply right after the other person's question ("Yes, I do.") are
    part of the talk, not backchannels.
    """
    for k, r in enumerate(found):
        if r.speaker not in pair or (r.ms >= BACKCHANNEL_MS and len(r.ids) > BACKCHANNEL_WORDS):
            continue
        prev = found[k - 1] if k else None
        replies = (
            prev is not None
            and prev.speaker in pair
            and prev.speaker != r.speaker
            and asking(words, prev.ids).likeness == 1
        )
        if replies or asking(words, r.ids).likeness == 1:
            continue
        beside = [found[j] for j in (k - 1, k + 1) if 0 <= j < len(found)]
        r.backchannel = sum(n.ms for n in beside if n.speaker in pair and n.speaker != r.speaker) >= LONG_MS


@dataclass
class Turn:
    speaker: str
    parts: list[Run] = field(default_factory=list)

    @property
    def ids(self) -> list[int]:
        return [i for r in self.parts for i in r.ids]

    @property
    def start_ms(self) -> int:
        return self.parts[0].start_ms

    @property
    def end_ms(self) -> int:
        return max(r.end_ms for r in self.parts)

    @property
    def ms(self) -> int:
        return self.end_ms - self.start_ms


def turns(found: list[Run], pair: list[str]) -> list[Turn]:
    """The two people's runs without backchannels, joined while the same person keeps the floor."""
    out: list[Turn] = []
    for r in found:
        if r.speaker not in pair or r.backchannel:
            continue
        if not out or out[-1].speaker != r.speaker:
            out.append(Turn(str(r.speaker)))
        out[-1].parts.append(r)
    return out


@dataclass(frozen=True)
class Roles:
    examiner: str
    candidate: str
    confidence: str


def interview(found: list[Turn], words: list[Word], pair: list[str], talk: dict[str, int]) -> Roles:
    """One assignment for the session: the examiner asks more of the questions that get a long answer."""
    own = dict.fromkeys(pair, 0)
    asked = dict.fromkeys(pair, 0)
    for k, t in enumerate(found):
        own[t.speaker] += 1
        nxt = found[k + 1] if k + 1 < len(found) else None
        if nxt and nxt.ms >= LONG_MS and asking(words, t.ids).likeness > 0:
            asked[t.speaker] += 1
    share = {s: asked[s] / own[s] if own[s] else 0.0 for s in pair}
    a, b = pair
    by_asking = a if share[a] > share[b] else b if share[b] > share[a] else None
    by_talk = a if talk[a] < talk[b] else b if talk[b] < talk[a] else None
    examiner = by_asking or by_talk or a
    if by_asking and by_talk:
        quiet = talk[examiner] <= HIGH_SHARE * (talk[a] + talk[b])
        confidence = "low" if by_asking != by_talk else "high" if quiet else "medium"
    else:
        confidence = "medium" if by_asking or by_talk else "low"
    return Roles(examiner, b if examiner == a else a, confidence)


@dataclass(frozen=True)
class Exchange:
    asker: str
    ask: Turn
    answer: Turn
    weight: float


def exchanges(found: list[Turn], words: list[Word]) -> list[Exchange]:
    """Each asking turn followed by the other person's answer of 5 s or more, with its vote weight."""
    out = []
    for t, nxt in pairwise(found):
        likeness = asking(words, t.ids).likeness
        if likeness > 0 and nxt.ms >= LONG_MS:
            out.append(Exchange(t.speaker, t, nxt, likeness * min(nxt.ms / 1000, ANSWER_CAP_S)))
    return out


def _smooth(votes: list[Exchange], pair: list[str], penalty: float) -> list[str]:
    """Viterbi over who examines: a vote against the state costs its weight, a swap costs the penalty."""
    other = {pair[0]: pair[1], pair[1]: pair[0]}
    cost = dict.fromkeys(pair, 0.0)
    back = []
    for e in votes:
        came = {s: s if cost[s] <= cost[other[s]] + penalty else other[s] for s in pair}
        cost = {
            s: cost[came[s]] + (penalty if came[s] != s else 0.0) + (0.0 if e.asker == s else e.weight)
            for s in pair
        }
        back.append(came)
    state = min(pair, key=lambda s: cost[s])
    path = []
    for came in reversed(back):
        path.append(state)
        state = came[state]
    return path[::-1]


@dataclass
class Phase:
    examiner: str
    first: int  # exchange indexes
    last: int
    confidence: str = "low"


def _group(path: list[str]) -> list[Phase]:
    out: list[Phase] = []
    for k, s in enumerate(path):
        if out and out[-1].examiner == s:
            out[-1].last = k
        else:
            out.append(Phase(s, k, k))
    return out


def phases(votes: list[Exchange], pair: list[str], style: str) -> tuple[list[Phase], int]:
    """The smoothed roles as phases with a confidence each, and how many stretches of votes were overruled.

    "once" (once or per Part): a swap needs 3 exchanges or 90 s, else it is undone, weakest first.
    "per-question": a small swap cost and no such floor.
    """
    path = _smooth(votes, pair, PENALTY[style])
    found = _group(path)
    while style == "once" and len(found) > 1:
        weak = [
            p for p in found
            if p.last - p.first + 1 < MIN_EXCHANGES
            and votes[p.last].answer.end_ms - votes[p.first].ask.start_ms < MIN_PHASE_MS
        ]  # fmt: skip
        if not weak:
            break
        p = min(weak, key=lambda p: sum(e.weight for e in votes[p.first : p.last + 1]))
        flipped = next(s for s in pair if s != p.examiner)
        path[p.first : p.last + 1] = [flipped] * (p.last - p.first + 1)
        found = _group(path)
    for p in found:
        mine = votes[p.first : p.last + 1]
        share = sum(e.weight for e in mine if e.asker == p.examiner) / sum(e.weight for e in mine)
        p.confidence = "high" if share >= SURE else "medium" if share >= FAIR else "low"
    against = sum(
        1 for k, e in enumerate(votes) if e.asker != path[k] and (k == 0 or votes[k - 1].asker == path[k - 1])
    )
    return found, against
