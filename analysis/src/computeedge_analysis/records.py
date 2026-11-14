"""Typed records for the benchmark output files (schema v1, docs/data_format.md).

Every CSV row is parsed once into a frozen record: statistics become floats and
the remaining columns become string parameters. Empty parameter cells (a
parameter that does not apply to a row) are omitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# Statistics columns of summary.csv (everything else, apart from `benchmark`, is a parameter).
SUMMARY_STATS = ("n_samples", "inner_iterations", "median_ns", "p25_ns", "p75_ns", "iqr_ns", "ci95_lo_ns",
                 "ci95_hi_ns", "mean_ns", "stddev_ns", "min_ns", "max_ns", "gflops_median")
SUMMARY_TEXT = ("thermal_before", "thermal_after")
# Statistics columns of metrics_summary.csv.
METRIC_STATS = ("n_samples", "median", "p25", "p75", "iqr", "ci95_lo", "ci95_hi", "mean", "min", "max", "stddev")


def _params(row: Mapping[str, str], exclude: set[str]) -> Mapping[str, str]:
    return MappingProxyType({k: v for k, v in row.items() if k not in exclude and v not in ("", None)})


def _matches(params: Mapping[str, str], wanted: Mapping[str, object]) -> bool:
    return all(params.get(k) == str(v) for k, v in wanted.items())


@dataclass(frozen=True)
class SummaryRow:
    """One benchmark configuration's timing summary (a summary.csv row)."""

    benchmark: str
    params: Mapping[str, str]
    stats: Mapping[str, float]
    thermal_before: str
    thermal_after: str

    @property
    def median_ns(self) -> float:
        """Median time per call in nanoseconds."""
        return self.stats["median_ns"]

    def matches(self, **wanted: object) -> bool:
        """Whether every given parameter equals str(value)."""
        return _matches(self.params, wanted)

    @classmethod
    def parse(cls, row: Mapping[str, str]) -> SummaryRow:
        """Builds a record from a csv.DictReader row."""
        stats = {k: float(row[k]) for k in SUMMARY_STATS if row.get(k) not in (None, "")}
        exclude = {"benchmark", *SUMMARY_STATS, *SUMMARY_TEXT}
        return cls(row["benchmark"], _params(row, exclude), MappingProxyType(stats),
                   row.get("thermal_before", ""), row.get("thermal_after", ""))


@dataclass(frozen=True)
class MetricSummaryRow:
    """Summary statistics of one per-call metric for one configuration (metrics_summary.csv)."""

    benchmark: str
    metric: str
    params: Mapping[str, str]
    median: float
    ci95_lo: float
    ci95_hi: float
    stats: Mapping[str, float] = field(repr=False)

    def matches(self, **wanted: object) -> bool:
        """Whether every given parameter equals str(value)."""
        return _matches(self.params, wanted)

    @classmethod
    def parse(cls, row: Mapping[str, str]) -> MetricSummaryRow:
        """Builds a record from a csv.DictReader row."""
        stats = {k: float(row[k]) for k in METRIC_STATS if row.get(k) not in (None, "")}
        exclude = {"benchmark", "metric", *METRIC_STATS}
        return cls(row["benchmark"], row["metric"], _params(row, exclude), stats["median"], stats["ci95_lo"],
                   stats["ci95_hi"], MappingProxyType(stats))


@dataclass(frozen=True)
class MetricSample:
    """One measured call's value of one metric (metrics.csv, long format)."""

    benchmark: str
    metric: str
    params: Mapping[str, str]
    rep: int
    value: float

    def matches(self, **wanted: object) -> bool:
        """Whether every given parameter equals str(value)."""
        return _matches(self.params, wanted)

    @classmethod
    def parse(cls, row: Mapping[str, str]) -> MetricSample:
        """Builds a record from a csv.DictReader row."""
        exclude = {"benchmark", "metric", "rep", "value"}
        return cls(row["benchmark"], row["metric"], _params(row, exclude), int(row["rep"]), float(row["value"]))


@dataclass(frozen=True)
class TimeseriesRow:
    """One call of a sustained-load run (timeseries.csv, thermal_drift only)."""

    t_s: float
    cycle: int
    workload: str
    ns: float
    thermal: str
    on_ac: bool | None  # None in runs recorded before the column existed

    @classmethod
    def parse(cls, row: Mapping[str, str]) -> TimeseriesRow:
        """Builds a record from a csv.DictReader row."""
        on_ac = row.get("on_ac")
        return cls(float(row["t_s"]), int(row["cycle"]), row["workload"], float(row["ns"]), row["thermal"],
                   None if on_ac in (None, "") else on_ac == "1")
