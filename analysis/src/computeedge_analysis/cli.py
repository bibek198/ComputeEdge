"""Command-line entry points.

  ce-claims  [--allow-dirty] [--out DIR]          evaluate every claim in every session
  ce-report  {baseline,characterization,interaction} [--allow-dirty]
  ce-figures [--session orig|sN]                  paper figures

Also runnable as `python -m computeedge_analysis.cli <command> ...`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def claims_main(argv: list[str] | None = None) -> int:
    """ce-claims: evaluates the claims and writes the report and paper numbers."""
    from .claims import run_claims

    ap = argparse.ArgumentParser(prog="ce-claims", description="Evaluate every claim in every session.")
    ap.add_argument("--allow-dirty", action="store_true", help="include runs from uncommitted (-dirty) builds")
    ap.add_argument("--out", type=Path, help="report directory (default: data/derived/reports/confirm-<stamp>)")
    a = ap.parse_args(argv)
    return run_claims(a.allow_dirty, a.out)


def report_main(argv: list[str] | None = None) -> int:
    """ce-report: regenerates a milestone report from the original runs."""
    from .reports import REPORTS

    ap = argparse.ArgumentParser(prog="ce-report", description="Regenerate a report from the original runs.")
    ap.add_argument("report", choices=sorted(REPORTS))
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    return REPORTS[a.report](a.allow_dirty)


def figures_main(argv: list[str] | None = None) -> int:
    """ce-figures: paper figures from one session's raw data."""
    from .figures.paper import make_figures

    ap = argparse.ArgumentParser(prog="ce-figures", description="Paper figures (single-column PDFs).")
    ap.add_argument("--session", default="orig", help="orig, or a confirmation session index such as s0")
    a = ap.parse_args(argv)
    return make_figures(a.session)


COMMANDS = {"claims": claims_main, "report": report_main, "figures": figures_main}


def main(argv: list[str] | None = None) -> int:
    """`python -m computeedge_analysis.cli <command> ...`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
