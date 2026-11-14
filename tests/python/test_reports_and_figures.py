"""Tests for computeedge_analysis.reports and computeedge_analysis.figures."""

from __future__ import annotations

from pathlib import Path

from computeedge_analysis.figures import make_figures
from computeedge_analysis.reports import (
    run_baseline_report,
    run_characterization_report,
    run_interaction_report,
)


def test_baseline_report_generation(tmp_path: Path):
    out_dir = tmp_path / "baseline"
    ret = run_baseline_report(allow_dirty=False, out_dir=out_dir)
    assert ret == 0
    assert (out_dir / "comparison.md").exists()
    assert (out_dir / "comparison.csv").exists()
    assert (out_dir / "latency.png").exists()
    assert (out_dir / "throughput.png").exists()
    assert (out_dir / "sources.json").exists()


def test_characterization_report_generation(tmp_path: Path):
    out_dir = tmp_path / "characterization"
    ret = run_characterization_report(allow_dirty=False, out_dir=out_dir)
    assert ret == 0
    assert (out_dir / "report.md").exists()
    assert (out_dir / "sources.json").exists()
    assert (out_dir / "dispatch_scaling.png").exists()


def test_interaction_report_generation(tmp_path: Path):
    out_dir = tmp_path / "interaction"
    ret = run_interaction_report(allow_dirty=False, out_dir=out_dir)
    assert ret == 0
    assert (out_dir / "report.md").exists()
    assert (out_dir / "sources.json").exists()


def test_make_figures(tmp_path: Path):
    out_dir = tmp_path / "figures"
    ret = make_figures("orig", out_dir=out_dir)
    assert ret == 0
    assert (out_dir / "round_trip.pdf").exists()
    assert (out_dir / "first_use.pdf").exists()
    assert (out_dir / "thermal.pdf").exists()
    assert (out_dir / "sources.json").exists()
