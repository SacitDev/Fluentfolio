"""The experiments' bootstrap, against answers known without it: exact small cases and the normal theory."""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

# eval/ is not a package; a script that imports this sibling finds it registered under its own name.
_SPEC = importlib.util.spec_from_file_location(
    "stats", Path(__file__).resolve().parents[2] / "eval" / "stats.py"
)
assert _SPEC is not None and _SPEC.loader is not None
stats = sys.modules.setdefault("stats", importlib.util.module_from_spec(_SPEC))
_SPEC.loader.exec_module(stats)


def mean(units: list[float]) -> float:
    return sum(units) / len(units)


def test_two_units_give_the_three_possible_means() -> None:
    # Resampling [0, 1] gives a mean of 0, 0.5 or 1 with chances 1/4, 1/2 and 1/4, so with 2,000 draws the
    # 2.5th and 97.5th percentiles are 0 and 1.
    assert stats.bootstrap([0.0, 1.0], mean, 2000, 20261013) == (0.5, 0.0, 1.0)


def test_the_interval_of_a_mean_has_the_width_normal_theory_gives() -> None:
    units = list(np.random.default_rng(7).normal(10.0, 2.0, 400))
    point, low, high = stats.bootstrap(units, mean, 2000, 20261013)
    # The standard error of a mean of 400 values is the sample SD over 20; the interval spans +-1.96 of them.
    expected = 2 * 1.96 * statistics.stdev(units) / 20
    assert point == pytest.approx(statistics.mean(units))
    assert high - low == pytest.approx(expected, rel=0.08) and low < point < high


def test_the_same_seed_gives_the_same_interval_and_another_seed_another() -> None:
    units = [1.0, 4.0, 2.0, 8.0, 5.0, 7.0]
    assert stats.bootstrap(units, mean, 500, 1) == stats.bootstrap(units, mean, 500, 1)
    assert stats.bootstrap(units, mean, 500, 1) != stats.bootstrap(units, mean, 500, 2)


def test_a_pooled_statistic_stays_pooled_inside_each_resample() -> None:
    # Units are (errors, words); the pooled rate of a resample is its errors over its words, never a mean
    # of rates. Units (1, 10) and (9, 10): every resample of two has 20 words, so the rates are 0.1, 0.5, 0.9.
    def rate(units: list[tuple[int, int]]) -> float:
        return sum(e for e, _ in units) / sum(w for _, w in units)

    assert stats.bootstrap([(1, 10), (9, 10)], rate, 2000, 3) == (0.5, 0.1, 0.9)


def test_a_paired_difference_uses_the_same_resample_for_both_sides() -> None:
    # Each unit holds both models' values; the second is always the first plus 2, so every resample's
    # difference is exactly -2 however much the units differ.
    units = [(x, x + 2.0) for x in (0.0, 5.0, 11.0, 30.0)]
    first = lambda us: mean([a for a, _ in us])  # noqa: E731
    second = lambda us: mean([b for _, b in us])  # noqa: E731
    point, low, high = stats.paired(units, first, second, 2000, 20261013)
    assert (point, low, high) == pytest.approx((-2.0, -2.0, -2.0))
    assert not stats.excludes_zero(-0.5, 0.5)
    assert stats.excludes_zero(-2.0, -1.0) and stats.excludes_zero(0.1, 3.0)


def test_no_units_is_an_error() -> None:
    with pytest.raises(ValueError, match="no units"):
        stats.bootstrap([], mean, 10, 1)
