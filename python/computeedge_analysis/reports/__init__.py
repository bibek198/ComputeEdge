"""Milestone and comparison reports generated from measurement runs."""

from .baseline import run_baseline_report
from .characterization import run_characterization_report
from .interaction import run_interaction_report

REPORTS = {
    "baseline": run_baseline_report,
    "characterization": run_characterization_report,
    "interaction": run_interaction_report,
}

__all__ = [
    "REPORTS",
    "run_baseline_report",
    "run_characterization_report",
    "run_interaction_report",
]
