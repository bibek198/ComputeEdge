"""Tests for computeedge_analysis.stats."""

from __future__ import annotations

import math

from computeedge_analysis.stats import linear_slope, slowdown_ci, slowdown_pct
from hypothesis import given
from hypothesis import strategies as st


def test_slowdown_pct_basic():
    x = [150.0, 150.0, 150.0]
    base = [100.0, 100.0, 100.0]
    assert math.isclose(slowdown_pct(x, base), 50.0)


def test_slowdown_ci_bounds():
    x = [120.0, 125.0, 130.0, 122.0, 128.0]
    base = [100.0, 100.0, 100.0, 100.0, 100.0]
    pt, lo, hi = slowdown_ci(x, base, resamples=500, seed=42)
    assert lo <= pt <= hi
    assert 15.0 <= pt <= 35.0


@given(
    st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_linear_slope_hypothesis(slope, intercept):
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    ys = [slope * x + intercept for x in xs]
    fitted = linear_slope(xs, ys)
    assert math.isclose(fitted, slope, rel_tol=1e-5, abs_tol=1e-5)
