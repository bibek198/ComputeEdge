"""M4 baseline comparison: builds one table and two figures from committed benchmark results.

Ported from benchmarks/analysis/compare_baselines.py.
Series (all FP32 square SGEMM, one synchronous call per measurement):
  CE CPU · Accelerate   cpu_matmul   impl=accelerate mode=e2e  (allocates output)
  MLX · CPU stream      mlx_matmul   stream=cpu                (allocates output)
  CE Metal · MPS        metal_matmul kernel=mps storage=nocopy (pre-allocated output)
  CE Metal · tiled16    metal_matmul kernel=tiled16 storage=nocopy
  MLX · GPU stream      mlx_matmul   stream=gpu
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

from .. import paths, style
from ..io import is_run, read_csv

# (label, benchmark, param filter, colour, marker, linestyle). CPU solid, GPU dashed.
SERIES = [
    ("CE CPU · Accelerate", "matmul", {"impl": "accelerate", "mode": "e2e"}, "#2a78d6", "o", "-"),
    ("MLX · CPU stream", "matmul_mlx", {"stream": "cpu"}, "#eb6834", "s", "-"),
    ("CE Metal · MPS", "matmul_metal", {"kernel": "mps", "storage": "nocopy"}, "#1baf7a", "D", "--"),
    ("CE Metal · tiled16", "matmul_metal", {"kernel": "tiled16", "storage": "nocopy"}, "#eda100", "^", "--"),
    ("MLX · GPU stream", "matmul_mlx", {"stream": "gpu"}, "#e87ba4", "v", "--"),
]


def newest_runs(allow_dirty: bool, data_dir: Path | None = None) -> dict[str, Path]:
    """Newest run directory containing each benchmark name in orig."""
    orig = (data_dir or paths.DATA) / "sessions" / "orig"
    chosen: dict[str, Path] = {}
    for run in sorted((p for p in orig.iterdir() if is_run(p)), reverse=True):
        if run.parent.name == "comparisons" or (not allow_dirty and run.name.endswith("-dirty")):
            continue
        for name in {r["benchmark"] for r in read_csv(run / "summary.csv")}:
            chosen.setdefault(name, run)
    return chosen


def collect(runs: dict[str, Path]) -> dict[str, dict[int, dict]]:
    """Gathers latency, throughput and CIs for each benchmark configuration."""
    data: dict[str, dict[int, dict]] = {}
    for label, bench, filt, *_ in SERIES:
        run = runs.get(bench)
        if run is None:
            continue
        for row in read_csv(run / "summary.csv"):
            if row["benchmark"] != bench or any(row.get(k) != v for k, v in filt.items()):
                continue
            if row.get("m") != row.get("n") or row.get("n") != row.get("k"):
                continue
            n = int(row["n"])
            data.setdefault(label, {})[n] = {
                "median_us": float(row["median_ns"]) / 1e3,
                "ci_lo_us": float(row["ci95_lo_ns"]) / 1e3,
                "ci_hi_us": float(row["ci95_hi_ns"]) / 1e3,
                "gflops": 2.0 * n**3 / float(row["median_ns"]),
                "thermal": f"{row['thermal_before']}->{row['thermal_after']}",
                "run": run.name,
            }
    return data


def fmt_us(v: float) -> str:
    """Formats microseconds with adaptive units."""
    return f"{v:,.1f}" if v < 1000 else f"{v / 1000:,.2f} ms"


def write_table(data: dict, sizes: list[int], out: Path) -> str:
    """Writes comparison.csv and generates the markdown table."""
    labels = [s[0] for s in SERIES if s[0] in data]
    lines = [
        "| n | " + " | ".join(labels) + " | fastest (ties: overlapping 95% CIs) |",
        "|---:|" + "---:|" * len(labels) + "---|",
    ]
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "series", "median_us", "ci95_lo_us", "ci95_hi_us", "gflops", "thermal", "run"])
        for n in sizes:
            cells = []
            for lab in labels:
                d = data[lab].get(n)
                if d is None:
                    cells.append("—")
                    continue
                w.writerow([
                    n,
                    lab,
                    d["median_us"],
                    d["ci_lo_us"],
                    d["ci_hi_us"],
                    d["gflops"],
                    d["thermal"],
                    d["run"],
                ])
                unit = "µs" if d["median_us"] < 1000 else ""
                cells.append(f"{fmt_us(d['median_us'])}{' ' + unit if unit else ''} ({d['gflops']:,.0f})")
            present = [(lab, data[lab][n]) for lab in labels if n in data[lab]]
            best = min(present, key=lambda x: x[1]["median_us"])
            tied = [lab for lab, d in present if d["ci_lo_us"] <= best[1]["ci_hi_us"]]
            fastest = " ≈ ".join(f"**{lab}**" if lab == best[0] else lab for lab in tied)
            lines.append(f"| {n} | " + " | ".join(cells) + f" | {fastest} |")
    return "\n".join(lines)


def plot_baseline(
    data: dict,
    sizes: list[int],
    metric: str,
    ylabel: str,
    chart_title: str,
    path: Path,
    logy: bool,
) -> None:
    """Plots latency or throughput across matrix sizes."""
    fig, (ax,) = style.new_axes(figsize=(8.0, 4.8))
    ax.set_xscale("log", base=2)
    style.labels(ax, "Matrix size n (n×n · n×n, FP32)", ylabel)
    if logy:
        ax.set_yscale("log")
    for label, _, _, colour, marker, ls in SERIES:
        pts = data.get(label)
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[n][metric] for n in xs]
        ax.plot(
            xs,
            ys,
            color=colour,
            linewidth=2,
            linestyle=ls,
            marker=marker,
            markersize=6,
            markeredgecolor=style.SURFACE,
            markeredgewidth=1.5,
            label=label,
            zorder=3,
        )
        if metric == "median_us":
            lo = [pts[n]["ci_lo_us"] for n in xs]
            hi = [pts[n]["ci_hi_us"] for n in xs]
            ax.vlines(xs, lo, hi, color=colour, linewidth=1, zorder=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlim(sizes[0] / 1.25, sizes[-1] * 1.25)
    if not logy:
        ax.set_ylim(bottom=0)
    style.legend(ax, loc="upper left")
    style.title(ax, chart_title, "Solid = CPU, dashed = GPU. Median of 30 synchronous calls; whiskers = 95% CI.")
    style.save(fig, path)


def _shown(path: Path) -> Path:
    try:
        return path.relative_to(paths.REPO)
    except ValueError:
        return path


def run_baseline_report(
    allow_dirty: bool = False,
    out_dir: Path | None = None,
    data_dir: Path | None = None,
) -> int:
    """Runs the M4 baseline comparison report."""
    data_root = data_dir or paths.DATA
    runs = newest_runs(allow_dirty, data_root)
    missing = [b for b in ("matmul", "matmul_metal", "matmul_mlx") if b not in runs]
    if missing:
        print(f"error: no results for {', '.join(missing)}", file=sys.stderr)
        return 1

    data = collect(runs)
    sizes = sorted({n for pts in data.values() for n in pts})
    out = out_dir or paths.DERIVED / "comparisons" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    envs = {b: json.loads((d / "env.json").read_text()) for b, d in runs.items()}
    (out / "sources.json").write_text(
        json.dumps(
            {
                b: {
                    "run": str(d.relative_to(paths.REPO)),
                    "git_sha": envs[b].get("git_sha"),
                    "thermal_state": envs[b].get("thermal_state"),
                    "power_source": envs[b].get("power_source"),
                }
                for b, d in runs.items()
            },
            indent=2,
        )
        + "\n"
    )

    table = write_table(data, sizes, out)
    plot_baseline(
        data,
        sizes,
        "median_us",
        "Latency per call (µs, log scale)",
        "SGEMM latency per synchronous call — lower is better",
        out / "latency.png",
        logy=True,
    )
    plot_baseline(
        data,
        sizes,
        "gflops",
        "Throughput (GFLOP/s)",
        "SGEMM throughput — higher is better",
        out / "throughput.png",
        logy=False,
    )

    env = next(iter(envs.values()))
    md = [
        "# M4 baseline comparison: FP32 SGEMM",
        "",
        f"Machine: {env.get('chip')} ({env.get('model')}), macOS {env.get('os_version')} ({env.get('os_build')}), "
        f"{env.get('power_source')}. Sources: `sources.json`.",
        "",
        "Cells give the median latency per synchronous call, with GFLOP/s in parentheses. "
        "The bootstrap 95% CIs are in `comparison.csv`.",
        "",
        table,
        "",
        "![Latency](latency.png)",
        "",
        "![Throughput](throughput.png)",
        "",
        "## What each series measures",
        "",
        "- **CE CPU · Accelerate**: `matmul()` with Accelerate `cblas_sgemm`, including output allocation. C++.",
        "- **MLX · CPU / GPU stream**: `mx.eval(mx.matmul(a, b, stream=…))` from Python. This includes graph "
        "construction, output allocation from MLX's buffer cache, dispatch, synchronisation and Python overhead.",
        "- **CE Metal · MPS / tiled16**: `metal::matmul_into()` with NoCopy storage into a pre-allocated output, "
        "including encode, commit and waitUntilCompleted. C++.",
        "",
        "These are end-to-end per-call costs as each API is used. They are not kernel-only times; see the "
        "`metrics_summary.csv` of the Metal run for the phase breakdown.",
    ]
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    print(table)
    print(f"\nWritten to {_shown(out)}")
    return 0
