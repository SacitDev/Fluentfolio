"""Bootstrap intervals for the experiments: whole units (speakers, clips or sessions) resampled with a seed.

A unit is whatever the pre-registration names; a statistic is computed over a list of units, so a pooled
rate stays pooled inside each resample. A paired difference computes both statistics on the same resample.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

LEVELS = (2.5, 97.5)  # a 95 % percentile interval, numpy's default linear interpolation between draws


def bootstrap[T](
    units: Sequence[T], statistic: Callable[[list[T]], float], resamples: int, seed: int
) -> tuple[float, float, float]:
    """The statistic on all units, and the 95 % percentile interval of it over resampled units."""
    if not units:
        raise ValueError("no units to resample")
    draws = np.random.default_rng(seed).integers(0, len(units), size=(resamples, len(units)))
    found = [statistic([units[i] for i in row]) for row in draws]
    low, high = np.percentile(found, LEVELS)
    return statistic(list(units)), float(low), float(high)


def paired[T](
    units: Sequence[T], first: Callable[[list[T]], float], second: Callable[[list[T]], float], resamples: int,
    seed: int,
) -> tuple[float, float, float]:  # fmt: skip
    """The difference first minus second on all units, and its interval with both on the same resample."""
    return bootstrap(units, lambda chosen: first(chosen) - second(chosen), resamples, seed)


def excludes_zero(low: float, high: float) -> bool:
    return low > 0 or high < 0
