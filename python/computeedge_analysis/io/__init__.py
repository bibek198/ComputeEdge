"""Read-only access to the measurement evidence in data/ (see data/README.md).

  runs        one run directory (one benchmark process): typed records and the report view
  sessions    sets of runs: the frozen original session and the confirmation sessions
  exclusions  runs left out of analysis, with their evidence

Nothing in this package writes to data/.
"""

from ..errors import DataError
from .exclusions import excluded_runs
from .runs import (
    benchmarks_in,
    env,
    is_run,
    metric_samples,
    metric_summary_rows,
    metric_table,
    newest_orig_run_with,
    orig_runs,
    pick,
    read_csv,
    run_source,
    summary_rows,
    timeseries_rows,
)
from .sessions import Session, label, load_sessions

__all__ = [
    "DataError",
    "Session",
    "benchmarks_in",
    "env",
    "excluded_runs",
    "is_run",
    "label",
    "load_sessions",
    "metric_samples",
    "metric_summary_rows",
    "metric_table",
    "newest_orig_run_with",
    "orig_runs",
    "pick",
    "read_csv",
    "run_source",
    "summary_rows",
    "timeseries_rows",
]
