"""Spans of time in whole milliseconds: merge them, cut some out, keep what lies inside others."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

Span = tuple[int, int]  # start and end in ms, start < end


def union(spans: Iterable[Sequence[int]]) -> list[Span]:
    """Spans merged where they overlap or touch, in time order."""
    out: list[list[int]] = []
    for a, b in sorted((int(a), int(b)) for a, b in spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def inside(spans: Iterable[Sequence[int]], within: Iterable[Sequence[int]]) -> list[Span]:
    """The parts of `spans` that lie inside `within`."""
    keep = union(within)
    out = []
    for a, b in union(spans):
        out += [(max(a, s), min(b, e)) for s, e in keep if s < b and e > a]
    return out


def subtract(spans: Iterable[Sequence[int]], cuts: Iterable[Sequence[int]]) -> list[Span]:
    """The parts of `spans` that no cut covers."""
    holes = union(cuts)
    out = []
    for a, b in union(spans):
        for s, e in holes:
            if e <= a or s >= b:
                continue
            if s > a:
                out.append((a, s))
            a = max(a, e)
        if a < b:
            out.append((a, b))
    return out


def total(spans: Iterable[Sequence[int]]) -> int:
    """Time covered by at least one span."""
    return sum(b - a for a, b in union(spans))
