"""Size and number formatting shared by claims, reports and figures."""

from __future__ import annotations

_MULT = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def size_bytes(label: str) -> int:
    """'64M' -> 67108864 (binary units K, M, G); a plain number is returned as bytes."""
    return int(label[:-1]) * _MULT[label[-1]] if label and label[-1] in _MULT else int(label)


def fmt_compact(x: object) -> str:
    """Compact number for tables: strings unchanged, NaN as an em dash, 3 significant digits below 100."""
    if isinstance(x, str):
        return x
    if x != x:  # NaN
        return "—"
    return f"{x:.3g}" if abs(x) < 100 else f"{x:,.0f}"
