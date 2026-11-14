"""Read-only access to one run directory (one benchmark process; see data/README.md).

Two views of the same files are offered:

- typed records (`summary_rows`, `metric_summary_rows`, `metric_samples`,
  `timeseries_rows`), used by the claims and figures;
- the plain-dict view of the original milestone report scripts (`read_csv`,
  `metric_table`, `pick`), kept so the ported reports compute exactly what the
  scripts did.

Nothing here writes to a run directory.
"""

from __future__ import annotations

import csv
import json
from functools import cache
from pathlib import Path

from .. import paths
from ..errors import DataError
from ..records import MetricSample, MetricSummaryRow, SummaryRow, TimeseriesRow

# metrics_summary.csv columns converted to float by `metric_table` (as in the original ce_plot.metric_table).
_METRIC_TABLE_FLOATS = ("median", "p25", "p75", "iqr", "ci95_lo", "ci95_hi", "mean", "min", "max")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV file as dicts.

    Raises:
        DataError: the file cannot be read.
    """
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError as e:
        raise DataError(f"cannot read {path}: {e}") from e


@cache
def summary_rows(run: Path) -> tuple[SummaryRow, ...]:
    """The run's summary.csv, in file order."""
    return tuple(SummaryRow.parse(r) for r in read_csv(run / "summary.csv"))


@cache
def metric_summary_rows(run: Path) -> tuple[MetricSummaryRow, ...]:
    """The run's metrics_summary.csv in file order (empty if the run has no metrics)."""
    path = run / "metrics_summary.csv"
    return tuple(MetricSummaryRow.parse(r) for r in read_csv(path)) if path.exists() else ()


@cache
def metric_samples(run: Path) -> tuple[MetricSample, ...]:
    """The run's metrics.csv in file order (empty if the run has no metrics)."""
    path = run / "metrics.csv"
    return tuple(MetricSample.parse(r) for r in read_csv(path)) if path.exists() else ()


@cache
def timeseries_rows(run: Path) -> tuple[TimeseriesRow, ...]:
    """The run's timeseries.csv in file order (empty if absent)."""
    path = run / "timeseries.csv"
    return tuple(TimeseriesRow.parse(r) for r in read_csv(path)) if path.exists() else ()


def env(run: Path) -> dict:
    """The run's env.json.

    Raises:
        DataError: the file is missing or is not valid JSON.
    """
    try:
        return json.loads((run / "env.json").read_text())
    except (OSError, ValueError) as e:
        raise DataError(f"cannot read {run / 'env.json'}: {e}") from e


def benchmarks_in(run: Path) -> set[str]:
    """Benchmark names present in the run's summary."""
    return {r.benchmark for r in summary_rows(run)}


def is_run(path: Path) -> bool:
    """Whether `path` is a run directory (has a summary.csv)."""
    return (path / "summary.csv").exists()


def orig_runs(data_dir: Path | None = None) -> list[Path]:
    """The original runs (`<data>/sessions/orig/`), newest first."""
    orig = (data_dir or paths.DATA) / "sessions" / "orig"
    return sorted((p for p in orig.iterdir() if is_run(p)), reverse=True)


def newest_orig_run_with(benchmark: str, allow_dirty: bool = False, data_dir: Path | None = None) -> Path | None:
    """The newest original run containing `benchmark` (used by the milestone reports)."""
    for run in orig_runs(data_dir):
        if not allow_dirty and run.name.endswith("-dirty"):
            continue
        if benchmark in benchmarks_in(run):
            return run
    return None


def metric_table(run: Path, benchmark: str) -> list[dict]:
    """metrics_summary.csv rows of one benchmark as dicts, statistics converted to float.

    Parameter columns stay strings (empty where a parameter does not apply), exactly as the
    original report scripts saw them.
    """
    out = []
    for r in read_csv(run / "metrics_summary.csv"):
        if r["benchmark"] != benchmark:
            continue
        for k in _METRIC_TABLE_FLOATS:
            r[k] = float(r[k])
        out.append(r)
    return out


def pick(rows: list[dict], metric: str, **params: object) -> dict | None:
    """The first row of `metric` whose parameters equal str(value) for every given parameter."""
    for r in rows:
        if r["metric"] == metric and all(r.get(k) == str(v) for k, v in params.items()):
            return r
    return None


def run_source(run: Path, repo: Path | None = None) -> dict:
    """Provenance entry for a report's sources.json: run path, git SHA, thermal state and power source."""
    e = env(run)
    return {"run": str(run.relative_to(repo or paths.REPO)), "git_sha": e.get("git_sha"),
            "thermal_state": e.get("thermal_state"), "power_source": e.get("power_source")}
