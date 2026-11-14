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
