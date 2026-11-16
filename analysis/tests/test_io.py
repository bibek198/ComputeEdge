"""Tests for computeedge_analysis.io."""

from __future__ import annotations

import json
from pathlib import Path

from computeedge_analysis.io import (
    excluded_runs,
    is_run,
    label,
    load_sessions,
    orig_runs,
)


def test_load_sessions_real_data():
    sessions = load_sessions(allow_dirty=False)
    assert len(sessions) >= 2
    orig_sess = sessions[0]
    assert orig_sess.name == "orig"
    assert label(orig_sess) == "orig"

    s0_sess = sessions[1]
    assert "s0" in s0_sess.name
    assert label(s0_sess) == "s0"


def test_orig_runs_present():
    runs = orig_runs()
    assert len(runs) >= 15
    for r in runs:
        assert is_run(r)


def test_session_methods_on_orig():
    sessions = load_sessions(allow_dirty=False)
    orig = sessions[0]

    assert orig.run_with("matmul") is not None
    assert orig.run_with("non_existent_benchmark") is None

    med = orig.median_ns("matmul", impl="accelerate", mode="kernel", n=128)
    assert med is not None and med > 0

    met = orig.metric("dispatch_empty", "total_ns", experiment="cmdbufs", command_buffers=1)
    assert met is not None and met > 0

    samples = orig.samples("cache_interference", "cpu_ns", **{"pass": "read", "ws": "1M", "phase": "gpu_tiny"})
    assert len(samples) > 0


def test_exclusions_parsing(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    exc_file = data_dir / "exclusions.json"
    exc_file.write_text(json.dumps({
        "excluded_runs": [
            {"run": "sessions/s0/bad_run", "benchmark": "thermal_drift", "reason": "test"}
        ]
    }))
    bad_run = (data_dir / "sessions" / "s0" / "bad_run").resolve()
    assert bad_run in excluded_runs(data_dir)


def test_exclusions_parsing_confirmation_and_nonexistent(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    exc_file = data_dir / "exclusions.json"
    exc_file.write_text(json.dumps({
        "excluded_runs": [
            {"run": "sessions/confirmation/s2/run_thermal", "benchmark": "thermal_drift", "reason": "sleep"},
            {"run": "sessions/nonexistent/path", "benchmark": "benchmark", "reason": "missing"}
        ]
    }))
    runs = excluded_runs(data_dir)
    assert (data_dir / "sessions" / "confirmation" / "s2" / "run_thermal").resolve() in runs
    assert (data_dir / "sessions" / "nonexistent" / "path").resolve() in runs


def test_exclusions_malformed_raises_data_error(tmp_path: Path):
    import pytest
    from computeedge_analysis.errors import DataError

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    exc_file = data_dir / "exclusions.json"

    # Malformed JSON
    exc_file.write_text("{ not json")
    with pytest.raises(DataError):
        excluded_runs(data_dir)

    # Missing excluded_runs key
    exc_file.write_text(json.dumps({"wrong_key": []}))
    with pytest.raises(DataError):
        excluded_runs(data_dir)


def test_load_sessions_filters_excluded_run_only(tmp_path: Path):
    data_dir = tmp_path / "data"
    conf_dir = data_dir / "sessions" / "confirmation" / "20261115_test_sess"
    conf_dir.mkdir(parents=True)
    (conf_dir / "session.json").write_text(json.dumps({"state": "completed", "status": "complete"}))

    run1 = conf_dir / "run_keep"
    run1.mkdir()
    (run1 / "env.json").write_text(json.dumps({}))
    (run1 / "summary.csv").write_text("benchmark,median_ns\nkeep_bench,100\n")

    run2 = conf_dir / "run_exclude"
    run2.mkdir()
    (run2 / "env.json").write_text(json.dumps({}))
    (run2 / "summary.csv").write_text("benchmark,median_ns\nexclude_bench,200\n")

    exc_file = data_dir / "exclusions.json"
    exc_file.write_text(json.dumps({
        "excluded_runs": [{
            "run": "sessions/confirmation/20261115_test_sess/run_exclude",
            "benchmark": "exclude_bench",
            "reason": "testing",
        }]
    }))

    sessions = load_sessions(allow_dirty=False, data_dir=data_dir)
    test_sess = [s for s in sessions if "20261115_test_sess" in s.name][0]
    run_names = [r.name for r in test_sess.runs]
    assert "run_keep" in run_names
    assert "run_exclude" not in run_names


def test_session_state_filtering(tmp_path: Path):
    data_dir = tmp_path / "data"
    sessions_dir = data_dir / "sessions"
    sessions_dir.mkdir(parents=True)

    # Completed session
    s_good = sessions_dir / "20261114_s1_good"
    s_good.mkdir()
    (s_good / "session.json").write_text(json.dumps({"state": "completed", "status": "complete"}))
    run_good = s_good / "20261114_run"
    run_good.mkdir()
    (run_good / "env.json").write_text(json.dumps({}))
    (run_good / "summary.csv").write_text("benchmark,median_ns\n")

    # Partial session marked by state
    s_partial = sessions_dir / "20261114_s2_partial"
    s_partial.mkdir()
    (s_partial / "session.json").write_text(json.dumps({"state": "partial", "status": "aborted"}))
    run_part = s_partial / "20261114_run"
    run_part.mkdir()
    (run_part / "env.json").write_text(json.dumps({}))
    (run_part / "summary.csv").write_text("benchmark,median_ns\n")

    # Partial session marked by PARTIAL marker file
    s_marker = sessions_dir / "20261114_s3_marker"
    s_marker.mkdir()
    (s_marker / "PARTIAL").write_text("Interrupted")
    run_marker = s_marker / "20261114_run"
    run_marker.mkdir()
    (run_marker / "env.json").write_text(json.dumps({}))
    (run_marker / "summary.csv").write_text("benchmark,median_ns\n")

    # Failed session marked by state
    s_failed = sessions_dir / "20261114_s4_failed"
    s_failed.mkdir()
    (s_failed / "session.json").write_text(json.dumps({"state": "failed", "status": "failed (exit code 1)"}))
    run_failed = s_failed / "20261114_run"
    run_failed.mkdir()
    (run_failed / "env.json").write_text(json.dumps({}))
    (run_failed / "summary.csv").write_text("benchmark,median_ns\n")

    loaded = load_sessions(allow_dirty=True, data_dir=data_dir)
    names = [s.name for s in loaded]
    assert "orig" in names
    assert "20261114_s1_good" in names
    assert "20261114_s2_partial" not in names
    assert "20261114_s3_marker" not in names
    assert "20261114_s4_failed" not in names

