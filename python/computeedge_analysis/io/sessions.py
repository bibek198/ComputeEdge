"""Measurement sessions: the frozen original runs ("orig") and the confirmation sessions.

A session is a set of run directories. Lookups go to the newest run of the session
that contains the requested benchmark, as in the original claims checker.
"""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..records import MetricSummaryRow, SummaryRow, TimeseriesRow
from .exclusions import excluded_runs
from .runs import benchmarks_in, is_run, metric_samples, metric_summary_rows, orig_runs, summary_rows, timeseries_rows


class Session:
    """The runs of one measurement session, with lookup by benchmark and parameters.

    Where several runs contain the same benchmark, the newest one is used.
    """

    def __init__(self, name: str, runs: list[Path]):
        """Creates a session called `name` from its run directories (any order)."""
        self.name = name
        self.runs = sorted(runs, reverse=True)  # newest first

    def run_with(self, benchmark: str) -> Path | None:
        """The newest run of this session that contains `benchmark`."""
        for run in self.runs:
            if benchmark in benchmarks_in(run):
                return run
        return None

    def summary(self, benchmark: str, **params: object) -> SummaryRow | None:
        """The first summary row of `benchmark` matching `params`."""
        run = self.run_with(benchmark)
        if not run:
            return None
        return next((r for r in summary_rows(run) if r.benchmark == benchmark and r.matches(**params)), None)

    def summaries(self, benchmark: str) -> list[SummaryRow]:
        """All summary rows of `benchmark`, in file order."""
        run = self.run_with(benchmark)
        return [r for r in summary_rows(run) if r.benchmark == benchmark] if run else []

    def median_ns(self, benchmark: str, **params: object) -> float | None:
        """Median ns per call of the matching configuration."""
        r = self.summary(benchmark, **params)
        return r.median_ns if r else None

    def metric(self, benchmark: str, metric: str, **params: object) -> float | None:
        """Median of a per-call metric for the matching configuration (metrics_summary.csv)."""
        run = self.run_with(benchmark)
        if not run:
            return None
        for r in metric_summary_rows(run):
            if r.benchmark == benchmark and r.metric == metric and r.matches(**params):
                return r.median
        return None

    def metric_rows(self, benchmark: str, metric: str) -> list[MetricSummaryRow]:
        """All metrics_summary rows of `benchmark` for `metric`, in file order."""
        run = self.run_with(benchmark)
        if not run:
            return []
        return [r for r in metric_summary_rows(run) if r.benchmark == benchmark and r.metric == metric]

    def samples(self, benchmark: str, metric: str, **params: object) -> list[float]:
        """Every measured value of a metric for the matching configuration, in file order."""
        run = self.run_with(benchmark)
        if not run:
            return []
        return [r.value for r in metric_samples(run)
                if r.benchmark == benchmark and r.metric == metric and r.matches(**params)]

    def timeseries(self) -> tuple[list[TimeseriesRow], str] | None:
        """The thermal_drift time series and its heater load, or None."""
        run = self.run_with("thermal_drift")
        if not run:
            return None
        return list(timeseries_rows(run)), self.summary("thermal_drift").params["load"]


def load_sessions(allow_dirty: bool = False, data_dir: Path | None = None) -> list[Session]:
    """The original session followed by every confirmation session, in name order, excluded runs left out.

    Runs from uncommitted builds (`-dirty`) are left out unless `allow_dirty`. A
    session with no remaining runs is omitted.
    """
    data = data_dir or paths.DATA
    excluded = excluded_runs(data)

    def clean(p: Path) -> bool:
        return (allow_dirty or not p.name.endswith("-dirty")) and p.resolve() not in excluded

    sessions = [Session("orig", [r for r in orig_runs(data) if clean(r)])]
    root = data / "sessions"
    if root.exists():
        for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "orig"):
            runs = [r for r in d.iterdir() if is_run(r) and clean(r)]
            if runs:
                sessions.append(Session(d.name, runs))
    return sessions


def label(s: Session) -> str:
    """'orig' or the sN index from a session directory name (<stamp>_sN_<sha>)."""
    m = [part for part in s.name.split("_") if part.startswith("s") and part[1:].isdigit()]
    return m[0] if m else s.name
