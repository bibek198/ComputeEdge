"""Tests for computeedge_analysis.units."""

from __future__ import annotations

import math

from computeedge_analysis.units import fmt_compact, size_bytes
from hypothesis import given
from hypothesis import strategies as st


def test_size_bytes_standard():
    assert size_bytes("16K") == 16 * 1024
    assert size_bytes("256K") == 256 * 1024
    assert size_bytes("4M") == 4 * 1024 * 1024
    assert size_bytes("64M") == 64 * 1024 * 1024
    assert size_bytes("1G") == 1024 * 1024 * 1024
    assert size_bytes("1024") == 1024


@given(st.integers(min_value=1, max_value=10000))
def test_size_bytes_hypothesis(n):
    assert size_bytes(f"{n}K") == n * 1024
    assert size_bytes(f"{n}M") == n * 1024 * 1024
    assert size_bytes(f"{n}G") == n * 1024 * 1024 * 1024
    assert size_bytes(str(n)) == n


def test_fmt_compact():
    assert fmt_compact("hello") == "hello"
    assert fmt_compact(float("nan")) == "—"
    assert fmt_compact(math.nan) == "—"
    assert fmt_compact(0.0) == "0"
    assert fmt_compact(12.345) == "12.3"
    assert fmt_compact(99.9) == "99.9"
    assert fmt_compact(100.0) == "100"
    assert fmt_compact(1500) == "1,500"
    assert fmt_compact(1234567.0) == "1,234,567"
