"""Statistics used by claims and reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def slowdown_pct(x: Sequence[float], base: Sequence[float]) -> float:
    """median(x) / median(base) - 1, as a percentage."""
    return (np.median(x) / np.median(base) - 1) * 100


def slowdown_ci(x: Sequence[float], base: Sequence[float], resamples: int = 2000,
                seed: int = 0xCE) -> tuple[float, float, float]:
    """Slowdown % with a percentile-bootstrap 95% CI (independent resampling of both samples)."""
    rng = np.random.default_rng(seed)
    xa, ba = np.asarray(x), np.asarray(base)
    point = (np.median(xa) / np.median(ba) - 1) * 100
    boots = [(np.median(rng.choice(xa, xa.size)) / np.median(rng.choice(ba, ba.size)) - 1) * 100
             for _ in range(resamples)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of ys on xs."""
    return float(np.polyfit(xs, ys, 1)[0])
