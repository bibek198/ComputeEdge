"""Repository locations used by the analysis (see data/README.md for the data layout)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
SESSIONS = DATA / "sessions"
ORIG = SESSIONS / "orig"  # the original M4-M6 runs, a frozen session
DERIVED = DATA / "derived"  # regenerable reports and comparisons
EXCLUSIONS = DATA / "exclusions.json"
PAPER_NUMBERS = REPO / "paper" / "generated" / "numbers.tex"
PAPER_FIGURES = REPO / "paper" / "figures"
