"""Tests for computeedge_analysis.records."""

from __future__ import annotations

from computeedge_analysis.records import (
    MetricSample,
    MetricSummaryRow,
    SummaryRow,
    TimeseriesRow,
)


def test_summary_row_parse():
    raw = {
        "benchmark": "matmul",
        "impl": "accelerate",
        "mode": "e2e",
        "n": "128",
        "n_samples": "30",
        "inner_iterations": "1",
        "median_ns": "12345.0",
        "ci95_lo_ns": "12000.0",
        "ci95_hi_ns": "12500.0",
        "gflops_median": "34.5",
        "thermal_before": "nominal",
        "thermal_after": "nominal",
    }
    row = SummaryRow.parse(raw)
    assert row.benchmark == "matmul"
    assert row.median_ns == 12345.0
    assert row.thermal_before == "nominal"
    assert row.params["impl"] == "accelerate"
    assert row.params["n"] == "128"
    assert row.matches(impl="accelerate", n=128)
    assert not row.matches(impl="metal")


def test_metric_summary_row_parse():
    raw = {
        "benchmark": "dispatch_empty",
        "metric": "total_ns",
        "command_buffers": "1",
        "encoders": "1",
        "dispatches": "1",
        "median": "85000.0",
        "ci95_lo": "84000.0",
        "ci95_hi": "86000.0",
    }
    row = MetricSummaryRow.parse(raw)
    assert row.benchmark == "dispatch_empty"
    assert row.metric == "total_ns"
    assert row.median == 85000.0
    assert row.matches(command_buffers=1, encoders=1)


def test_metric_sample_parse():
    raw = {
        "benchmark": "cache_interference",
        "metric": "cpu_ns",
        "pass": "read",
        "ws": "1M",
        "rep": "5",
        "value": "45678.0",
    }
    sample = MetricSample.parse(raw)
    assert sample.benchmark == "cache_interference"
    assert sample.metric == "cpu_ns"
    assert sample.rep == 5
    assert sample.value == 45678.0
    assert sample.matches(ws="1M")


def test_timeseries_row_parse():
    raw = {
        "t_s": "12.5",
        "cycle": "3",
        "workload": "cpu_1024",
        "ns": "1500000.0",
        "thermal": "nominal",
        "on_ac": "1",
    }
    row = TimeseriesRow.parse(raw)
    assert row.t_s == 12.5
    assert row.cycle == 3
    assert row.workload == "cpu_1024"
    assert row.ns == 1500000.0
    assert row.thermal == "nominal"
    assert row.on_ac is True
