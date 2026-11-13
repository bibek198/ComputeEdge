#!/usr/bin/env python3
"""M4 baseline comparison: builds one table and two figures from committed
benchmark results (summary.csv files under benchmarks/results/).

Series (all FP32 square SGEMM, one synchronous call per measurement):
  CE CPU · Accelerate   cpu_matmul   impl=accelerate mode=e2e  (allocates output)
  MLX · CPU stream      mlx_matmul   stream=cpu                (allocates output)
  CE Metal · MPS        metal_matmul kernel=mps storage=nocopy (pre-allocated output)
  CE Metal · tiled16    metal_matmul kernel=tiled16 storage=nocopy
  MLX · GPU stream      mlx_matmul   stream=gpu

By default the newest clean (non "-dirty") run of each benchmark is used; pass
--runs to choose explicitly. Writes benchmarks/results/comparisons/<stamp>/ with
comparison.md, comparison.csv, latency.png, throughput.png and sources.json.

Usage: .venv/bin/python benchmarks/analysis/compare_baselines.py [--runs DIR ...] [--allow-dirty]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "benchmarks" / "results"

# Reference palette (dataviz skill), light mode; validated: all checks pass,
# contrast WARN on slots 3-5 -> direct labels + distinct markers + table view.
SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# (label, benchmark, param filter, colour, marker, linestyle). CPU solid, GPU dashed.
SERIES = [
    ("CE CPU · Accelerate", "matmul", {"impl": "accelerate", "mode": "e2e"}, "#2a78d6", "o", "-"),
    ("MLX · CPU stream", "matmul_mlx", {"stream": "cpu"}, "#eb6834", "s", "-"),
    ("CE Metal · MPS", "matmul_metal", {"kernel": "mps", "storage": "nocopy"}, "#1baf7a", "D", "--"),
    ("CE Metal · tiled16", "matmul_metal", {"kernel": "tiled16", "storage": "nocopy"}, "#eda100", "^", "--"),
    ("MLX · GPU stream", "matmul_mlx", {"stream": "gpu"}, "#e87ba4", "v", "--"),
]


def load_summary(run: Path) -> list[dict]:
    with open(run / "summary.csv", newline="") as f:
        return list(csv.DictReader(f))


def newest_runs(allow_dirty: bool) -> dict[str, Path]:
    """Newest run directory containing each benchmark name."""
    chosen: dict[str, Path] = {}
    for run in sorted((p for p in RESULTS.iterdir() if (p / "summary.csv").exists()), reverse=True):
        if run.parent.name == "comparisons" or (not allow_dirty and run.name.endswith("-dirty")):
            continue
        for name in {r["benchmark"] for r in load_summary(run)}:
            chosen.setdefault(name, run)
    return chosen


def collect(runs: dict[str, Path]) -> dict[str, dict[int, dict]]:
    data: dict[str, dict[int, dict]] = {}
    for label, bench, filt, *_ in SERIES:
        run = runs.get(bench)
        if run is None:
            continue
        for row in load_summary(run):
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
    return f"{v:,.1f}" if v < 1000 else f"{v / 1000:,.2f} ms"


def write_table(data: dict, sizes: list[int], out: Path) -> str:
    labels = [s[0] for s in SERIES if s[0] in data]
    lines = ["| n | " + " | ".join(labels) + " | fastest (ties: overlapping 95% CIs) |",
             "|---:|" + "---:|" * len(labels) + "---|"]
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
                w.writerow([n, lab, d["median_us"], d["ci_lo_us"], d["ci_hi_us"], d["gflops"], d["thermal"], d["run"]])
                unit = "µs" if d["median_us"] < 1000 else ""
                cells.append(f"{fmt_us(d['median_us'])}{' ' + unit if unit else ''} ({d['gflops']:,.0f})")
            present = [(lab, data[lab][n]) for lab in labels if n in data[lab]]
            best = min(present, key=lambda x: x[1]["median_us"])
            tied = [lab for lab, d in present if d["ci_lo_us"] <= best[1]["ci_hi_us"]]
            fastest = " ≈ ".join(f"**{lab}**" if lab == best[0] else lab for lab in tied)
            lines.append(f"| {n} | " + " | ".join(cells) + f" | {fastest} |")
    return "\n".join(lines)


def style_axes(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_2, labelsize=9)
    ax.set_xlabel("Matrix size n (n×n · n×n, FP32)", color=TEXT_2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)


def plot(data: dict, sizes: list[int], metric: str, ylabel: str, title: str, path: Path, logy: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax, ylabel)
    if logy:
        ax.set_yscale("log")
    for label, _, _, colour, marker, ls in SERIES:
        pts = data.get(label)
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[n][metric] for n in xs]
        ax.plot(xs, ys, color=colour, linewidth=2, linestyle=ls, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        if metric == "median_us":  # CI whiskers (often thinner than the marker)
            lo = [pts[n]["ci_lo_us"] for n in xs]
            hi = [pts[n]["ci_hi_us"] for n in xs]
            ax.vlines(xs, lo, hi, color=colour, linewidth=1, zorder=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlim(sizes[0] / 1.25, sizes[-1] * 1.25)
    if not logy:
        ax.set_ylim(bottom=0)
    # Five series: identity comes from the legend + distinct markers (+ line style:
    # solid = CPU, dashed = GPU); exact values are in comparison.md / .csv.
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=TEXT, handlelength=3)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=22)
    ax.text(0, 1.015, "Solid = CPU, dashed = GPU. Median of 30 synchronous calls; whiskers = 95% CI.",
            transform=ax.transAxes, color=TEXT_2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", help="result directories (default: newest clean run per benchmark)")
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args()

    if a.runs:
        runs = {}
        for d in map(Path, a.runs):
            for name in {r["benchmark"] for r in load_summary(d)}:
                runs[name] = d
    else:
        runs = newest_runs(a.allow_dirty)
    missing = [b for b in ("matmul", "matmul_metal", "matmul_mlx") if b not in runs]
    if missing:
        print(f"error: no results for {', '.join(missing)}", file=sys.stderr)
        return 1

    data = collect(runs)
    sizes = sorted({n for pts in data.values() for n in pts})
    out = RESULTS / "comparisons" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)

    envs = {b: json.loads((d / "env.json").read_text()) for b, d in runs.items()}
    (out / "sources.json").write_text(json.dumps(
        {b: {"run": str(d.relative_to(REPO)), "git_sha": envs[b].get("git_sha"),
             "thermal_state": envs[b].get("thermal_state"), "power_source": envs[b].get("power_source")}
         for b, d in runs.items()}, indent=2) + "\n")

    table = write_table(data, sizes, out)
    plot(data, sizes, "median_us", "Latency per call (µs, log scale)",
         "SGEMM latency per synchronous call — lower is better", out / "latency.png", logy=True)
    plot(data, sizes, "gflops", "Throughput (GFLOP/s)",
         "SGEMM throughput — higher is better", out / "throughput.png", logy=False)

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
    print(f"\nWritten to {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
