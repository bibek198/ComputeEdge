#!/usr/bin/env python3
"""Records the output schema of benchmark runs, for refactoring equivalence checks.

Given result directories (one benchmark process each), writes a JSON document
describing, per benchmark name: the CSV headers of every file it wrote, its
parameter keys, the set of values each parameter took, and its metric names.
Timings are deliberately excluded: the snapshot captures *what* is measured and
*how it is labelled*, which must not change during a behaviour-preserving
refactor.

Usage:
    schema_snapshot.py OUT.json RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Columns that are values, not schema (they vary run to run).
VALUE_COLUMNS = {
    "rep", "inner_iterations", "ns_per_call", "thermal", "value", "n_samples", "median_ns", "p25_ns", "p75_ns",
    "iqr_ns", "ci95_lo_ns", "ci95_hi_ns", "mean_ns", "stddev_ns", "min_ns", "max_ns", "gflops_median",
    "thermal_before", "thermal_after", "metric", "median", "p25", "p75", "iqr", "ci95_lo", "ci95_hi", "mean",
    "min", "max", "stddev",
}
# Parameter values that legitimately vary between processes (none today; listed for clarity).
VOLATILE_PARAMS: set[str] = set()


def snapshot(run_dirs: list[Path]) -> dict:
    """Builds the schema description for the given run directories."""
    out: dict[str, dict] = {}
    for run in run_dirs:
        for csv_path in sorted(run.glob("*.csv")):
            if csv_path.name == "timeseries.csv":  # handled below
                continue
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                for row in reader:
                    bench = row.get("benchmark", "(none)")
                    entry = out.setdefault(bench, {"files": {}, "params": {}, "metrics": set()})
                    entry["files"].setdefault(csv_path.name, header)
                    for key, val in row.items():
                        if key in ("benchmark",) or key in VALUE_COLUMNS or val in (None, ""):
                            continue
                        if key not in VOLATILE_PARAMS:
                            entry["params"].setdefault(key, set()).add(val)
                    if "metric" in row and row["metric"]:
                        entry["metrics"].add(row["metric"])
        env = run / "env.json"
        if env.exists():
            keys = sorted(json.loads(env.read_text()).keys())
            out.setdefault("(env.json)", {"files": {}, "params": {}, "metrics": set()})["files"]["env.json"] = keys
        ts = run / "timeseries.csv"
        if ts.exists():
            with open(ts, newline="") as f:
                rows = list(csv.DictReader(f))
            entry = out.setdefault("(timeseries)", {"files": {}, "params": {}, "metrics": set()})
            entry["files"]["timeseries.csv"] = list(rows[0].keys()) if rows else []
            entry["params"].setdefault("workload", set()).update(r["workload"] for r in rows)
    return {b: {"files": e["files"], "params": {k: sorted(v) for k, v in sorted(e["params"].items())},
                "metrics": sorted(e["metrics"])} for b, e in sorted(out.items())}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(sys.argv[1])
    runs = [Path(p) for p in sys.argv[2:]]
    missing = [str(r) for r in runs if not r.is_dir()]
    if missing:
        print(f"error: not directories: {', '.join(missing)}", file=sys.stderr)
        return 1
    out.write_text(json.dumps(snapshot(runs), indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(runs)} run dirs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
