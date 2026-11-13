#!/usr/bin/env python3
"""M5 characterization report (Tracks A and B): tables and figures from the
dispatch_overhead, sync_methods and storage_modes results.

Uses the newest clean run of each benchmark. For the synchronization-method
comparison, every clean run is shown side by side, because method order within a
run is a confound (see the report text). Writes
benchmarks/results/reports/m5-<stamp>/ with report.md and the PNG figures.

Usage: .venv/bin/python benchmarks/analysis/m5_report.py [--allow-dirty]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

import ce_plot as P

KINDS_CREATE = ["shared", "private", "nocopy", "shared_copy"]
KINDS_GPU = ["shared", "private", "nocopy"]
KINDS_TOUCH = ["host", "shared", "nocopy"]
GPU_FIRST_COLS = [("shared", "fresh"), ("shared", "reused"), ("private", "fresh"), ("private", "reused"),
                  ("nocopy", "fresh"), ("nocopy", "rewrap"), ("nocopy", "reused")]


def us(ns: float) -> str:
    v = ns / 1e3
    return f"{v:,.1f}" if v < 10_000 else f"{v / 1e3:,.1f} ms"


def size_bytes(label: str) -> int:
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(label[:-1]) * mult[label[-1]] if label[-1] in mult else int(label)


def ordered(values, key=None):
    return sorted(set(values), key=key)


# ---------------------------------------------------------------------------
# Track A
# ---------------------------------------------------------------------------

def dispatch_section(run: Path, out: Path) -> list[str]:
    rows = P.metric_table(run, "dispatch_empty")
    cold = P.metric_table(run, "dispatch_cold")
    exps = [("dispatches", "dispatches", "dispatches per encoder (1 encoder, 1 command buffer)"),
            ("encoders", "encoders", "encoders per command buffer (1 dispatch each)"),
            ("cmdbufs", "command_buffers", "command buffers, committed back to back (1 encoder, 1 dispatch each)")]

    fig, (ax,) = P.new_axes()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    fits = []
    for i, (exp, param, desc) in enumerate(exps):
        pts = sorted((int(r[param]), r) for r in rows if r["experiment"] == exp and r["metric"] == "total_ns")
        if not pts:
            continue
        xs = [n for n, _ in pts]
        P.series(ax, xs, [r["median"] / 1e3 for _, r in pts], i, desc.split(" (")[0],
                 [r["ci95_lo"] / 1e3 for _, r in pts], [r["ci95_hi"] / 1e3 for _, r in pts])
        fit = {"experiment": exp, "unit": desc}
        for metric in ("total_ns", "encode_ns", "commit_ns", "gpu_ns"):
            m = {int(r[param]): r["median"] for r in rows if r["experiment"] == exp and r["metric"] == metric}
            slope, intercept = np.polyfit(list(m), list(m.values()), 1)
            fit[metric] = (slope, intercept)
        fits.append(fit)
    P.labels(ax, "Count per submission (log scale)", "Latency per submission (µs, log scale)")
    P.title(ax, "Empty-kernel submission cost — lower is better",
            "Median of 30 synchronous submissions (waitUntilCompleted); whiskers = 95% CI.")
    P.legend(ax)
    P.save(fig, out / "dispatch_scaling.png")

    md = ["## Track A1–A2: dispatch overhead", "",
          "Empty kernels (one thread, no memory access), so all time is runtime overhead.", "",
          "![Dispatch scaling](dispatch_scaling.png)", "",
          "Marginal cost per added unit (least-squares slope over the medians):", "",
          "| added unit | total µs/unit | encode µs/unit | commit µs/unit | GPU µs/unit | fitted fixed cost (total, µs) |",
          "|---|---:|---:|---:|---:|---:|"]
    for f in fits:
        md.append(f"| {f['unit']} | {f['total_ns'][0] / 1e3:.2f} | {f['encode_ns'][0] / 1e3:.3f} | "
                  f"{f['commit_ns'][0] / 1e3:.3f} | {f['gpu_ns'][0] / 1e3:.3f} | {f['total_ns'][1] / 1e3:.0f} |")
    base = [r for r in rows if r["metric"] == "total_ns" and r["command_buffers"] == "1" and r["encoders"] == "1"
            and r["dispatches"] == "1"]
    md += ["", "Breakdown of a single empty round trip (1 command buffer, 1 encoder, 1 dispatch), medians in µs. "
           "The same configuration is measured once per sweep, so the rows also show within-run drift:", "",
           "| sweep | total | encode | commit | queue | GPU | wake |", "|---|---:|---:|---:|---:|---:|---:|"]
    for b in base:
        get = lambda m: P.pick(rows, m, experiment=b["experiment"], command_buffers=1, encoders=1, dispatches=1)
        md.append(f"| {b['experiment']} | " + " | ".join(
            us(get(m)["median"]) for m in ("total_ns", "encode_ns", "commit_ns", "queue_ns", "gpu_ns", "wake_ns")) + " |")
    if cold:
        c = {r["metric"]: r["median"] for r in cold}
        md += ["", f"Cold start (first empty dispatch in the process, one sample): compiling the probe library and "
               f"pipeline took {us(c['compile_ns'])} µs (OS shader cache state not controlled), then the first round "
               f"trip took {us(c['total_ns'])} µs, of which {us(c['queue_ns'])} µs was queueing."]
    return md


def sync_section(runs: list[Path], idle_run: Path | None, out: Path) -> list[str]:
    md = ["", "## Track A3: synchronization methods", "",
          "Wake = CPU resumes − GPUEndTime; with event methods the event is signalled inside the command "
          "buffer, so wake can be slightly negative. Median wake latency in µs, one column per run "
          "(runs differ in method order):", ""]
    tables = [P.metric_table(r, "sync_method") for r in runs]
    orders = []
    for run in runs:
        seen = []
        for r in P.read_csv(run / "summary.csv"):
            if r["benchmark"] == "sync_method" and r["sync"] not in seen:
                seen.append(r["sync"])
        orders.append(seen)
    methods = orders[0]
    works = ordered((r["work"] for r in tables[0]), key=lambda w: 0 if w == "empty" else size_bytes(w))
    for metric, what in (("wake_ns", "wake latency"), ("total_ns", "total round trip")):
        md += [f"**{what}** (µs):", "",
               "| work | method | " + " | ".join(f"run {i + 1}" for i in range(len(runs))) + " |",
               "|---|---|" + "---:|" * len(runs)]
        for w in works:
            for m in methods:
                cells = []
                for t in tables:
                    r = P.pick(t, metric, sync=m, work=w)
                    cells.append(f"{us(r['median'])} [{us(r['ci95_lo'])}, {us(r['ci95_hi'])}]" if r else "—")
                md.append(f"| {w} | {m} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["Method order per run: " + "; ".join(f"run {i + 1}: {', '.join(o)}" for i, o in enumerate(orders)), ""]

    if idle_run:
        rows = P.metric_table(idle_run, "sync_idle")
        gaps = ordered((r["gap_ms"] for r in rows), key=float)
        comps = [("encode_ns", "encode"), ("commit_ns", "commit"), ("queue_ns", "queue (commit → GPU start)"),
                 ("gpu_ns", "GPU execution"), ("wake_ns", "wake (GPU end → CPU)")]
        fig, (ax,) = P.new_axes(figsize=(8.0, 4.4))
        ys = np.arange(len(gaps))
        left = np.zeros(len(gaps))
        for i, (metric, label) in enumerate(comps):
            vals = np.array([max(P.pick(rows, metric, sync="wait_until_completed", gap_ms=g)["median"], 0) / 1e3
                             for g in gaps])
            ax.barh(ys, vals, left=left, height=0.6, color=P.CATEGORICAL[i], edgecolor=P.SURFACE, linewidth=2,
                    label=label, zorder=3)
            left += vals
        for y, total in zip(ys, left):
            ax.text(total + 8, y, f"{total:.0f} µs", va="center", color=P.TEXT_2, fontsize=8.5)
        ax.set_yticks(ys, [f"{g} ms" for g in gaps])
        ax.invert_yaxis()
        ax.set_xlim(0, left.max() * 1.15)
        ax.grid(axis="y", visible=False)
        P.labels(ax, "Median time per phase (µs)", "GPU idle before the call")
        P.title(ax, "An idle GPU makes the next empty round trip slower",
                "Empty kernel, waitUntilCompleted. Bars stack phase medians (approximately the median total).")
        P.legend(ax, loc="lower right")
        P.save(fig, out / "idle_gap.png")

        md += ["## Track A4: idle gap before a submission", "",
               "Untimed sleep before every call; empty kernel. Medians in µs.", "", "![Idle gap](idle_gap.png)", "",
               "| sync | idle gap | total | encode | commit | queue | GPU | wake |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in ordered(r["sync"] for r in rows):
            for g in gaps:
                get = lambda m: P.pick(rows, m, sync=s, gap_ms=g)
                md.append(f"| {s} | {g} ms | " + " | ".join(
                    us(get(m)["median"]) for m in ("total_ns", "encode_ns", "commit_ns", "queue_ns", "gpu_ns",
                                                   "wake_ns")) + " |")
    return md


# ---------------------------------------------------------------------------
# Track B
# ---------------------------------------------------------------------------

def storage_section(runs: dict[str, Path], out: Path) -> list[str]:
    """`runs` maps each storage_* benchmark to the newest run containing it."""
    md = ["", "## Track B: storage modes", ""]

    # Creation cost.
    rows = P.metric_table(runs["storage_create"], "storage_create")
    sizes = ordered((r["size"] for r in rows), key=size_bytes)
    xs = [size_bytes(s) for s in sizes]
    fig, (ax,) = P.new_axes()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    for i, k in enumerate(KINDS_CREATE):
        pts = [P.pick(rows, "create_ns", kind=k, size=s) for s in sizes]
        if all(pts):
            P.series(ax, xs, [p["median"] / 1e3 for p in pts], i, k, [p["ci95_lo"] / 1e3 for p in pts],
                     [p["ci95_hi"] / 1e3 for p in pts])
    ax.set_xticks(xs, sizes)
    P.labels(ax, "Buffer size (log scale)", "Create time (µs, log scale)")
    P.title(ax, "MTLBuffer creation cost by storage kind — lower is better",
            "nocopy wraps resident host memory; shared_copy copies it. Median of 30; whiskers = 95% CI.")
    P.legend(ax)
    P.save(fig, out / "storage_create.png")
    md += ["### B1: buffer creation and release", "", "![Creation](storage_create.png)", "",
           "Median create / release time in µs:", "",
           "| size | " + " | ".join(KINDS_CREATE) + " |", "|---:|" + "---:|" * len(KINDS_CREATE)]
    for s in sizes:
        cells = []
        for k in KINDS_CREATE:
            c, r = P.pick(rows, "create_ns", kind=k, size=s), P.pick(rows, "release_ns", kind=k, size=s)
            cells.append(f"{us(c['median'])} / {us(r['median'])}" if c and r else "—")
        md.append(f"| {s} | " + " | ".join(cells) + " |")

    # CPU first touch.
    rows = P.metric_table(runs["storage_cpu_touch"], "storage_cpu_touch") if "storage_cpu_touch" in runs else []
    if rows:
        sizes_t = ordered((r["size"] for r in rows), key=size_bytes)
        md += ["", "### B2: CPU first touch and warm bandwidth", "",
               "GB/s (bytes / median time). First write = memset of freshly created memory (page faults); "
               "warm = the same memory again.", "",
               "| size | kind | first write | warm write | warm read |", "|---:|---|---:|---:|---:|"]
        for s in sizes_t:
            b = size_bytes(s)
            for k in KINDS_TOUCH:
                vals = [P.pick(rows, m, kind=k, size=s) for m in ("first_write_ns", "warm_write_ns", "warm_read_ns")]
                if all(vals):
                    md.append(f"| {s} | {k} | " + " | ".join(f"{b / v['median']:.1f}" for v in vals) + " |")

    # GPU first use.
    rows = P.metric_table(runs["storage_gpu_first"], "storage_gpu_first") if "storage_gpu_first" in runs else []
    if rows:
        sizes_g = ordered((r["size"] for r in rows), key=size_bytes)
        xs = [size_bytes(s) for s in sizes_g]
        fig, (ax,) = P.new_axes()
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        for i, k in enumerate(KINDS_GPU):
            for buf, ls in (("fresh", "-"), ("rewrap", ":"), ("reused", "--")):
                pts = [P.pick(rows, "total_ns", kind=k, size=s, buffer=buf) for s in sizes_g]
                if all(pts):
                    P.series(ax, xs, [p["median"] / 1e3 for p in pts], i, f"{k} · {buf}",
                             [p["ci95_lo"] / 1e3 for p in pts], [p["ci95_hi"] / 1e3 for p in pts], linestyle=ls)
        ax.set_xticks(xs, sizes_g)
        P.labels(ax, "Size of the bound (untouched) buffer (log scale)", "Round trip (µs, log scale)")
        P.title(ax, "First GPU use of a buffer: fresh vs reused — lower is better",
                "Kernel never touches the buffer. Solid = fresh buffer, dotted = new NoCopy wrapper over the same "
                "memory, dashed = reused.")
        P.legend(ax)
        P.save(fig, out / "gpu_first_use.png")
        md += ["", "### B3: first GPU use (residency) vs reuse", "", "![First use](gpu_first_use.png)", "",
               "Median total / queue in µs:", "",
               "| size | " + " | ".join(f"{k} {b}" for k, b in GPU_FIRST_COLS) + " |",
               "|---:|" + "---:|" * len(GPU_FIRST_COLS)]
        for s in sizes_g:
            cells = []
            for k, buf in GPU_FIRST_COLS:
                t = P.pick(rows, "total_ns", kind=k, size=s, buffer=buf)
                q = P.pick(rows, "queue_ns", kind=k, size=s, buffer=buf)
                cells.append(f"{us(t['median'])} / {us(q['median'])}" if t and q else "—")
            md.append(f"| {s} | " + " | ".join(cells) + " |")

    # GPU bandwidth.
    rows = P.metric_table(runs["storage_gpu_bw"], "storage_gpu_bw") if "storage_gpu_bw" in runs else []
    if rows:
        sizes_b = ordered((r["size"] for r in rows), key=size_bytes)
        xs = [size_bytes(s) for s in sizes_b]
        streams = ["read", "write", "copy"]
        fig, axes = P.new_axes(figsize=(10.0, 4.0), ncols=3, sharey=True)
        bw = {}
        for j, st in enumerate(streams):
            ax = axes[j]
            ax.set_xscale("log", base=2)
            for i, k in enumerate(KINDS_GPU):
                pts = [P.pick(rows, "gpu_ns", kind=k, size=s, stream=st) for s in sizes_b]
                if not all(pts):
                    continue
                moved = [int(p["bytes_moved"]) for p in pts]
                gbs = [m / p["median"] for m, p in zip(moved, pts)]
                bw[(st, k)] = dict(zip(sizes_b, gbs))
                P.series(ax, xs, gbs, i, k)
            ax.set_xticks(xs[::2], sizes_b[::2])
            ax.set_ylim(bottom=0)
            ax.set_title(st, loc="left", color=P.TEXT, fontsize=10)
            P.labels(ax, "Buffer size", "GB/s (bytes moved / GPU time)" if j == 0 else "")
        P.legend(axes[0], loc="lower right")
        fig.suptitle("GPU streaming bandwidth by storage kind — higher is better", x=0.01, ha="left",
                     color=P.TEXT, fontsize=12)
        P.save(fig, out / "gpu_bandwidth.png")
        md += ["", "### B4: GPU streaming bandwidth", "", "![Bandwidth](gpu_bandwidth.png)", "",
               "GB/s from GPUStart–GPUEnd (copy counts read + write):", "",
               "| size | " + " | ".join(f"{st} {k}" for st in streams for k in KINDS_GPU) + " |",
               "|---:|" + "---:|" * (len(streams) * len(KINDS_GPU))]
        for s in sizes_b:
            md.append(f"| {s} | " + " | ".join(
                f"{bw[(st, k)][s]:.1f}" if (st, k) in bw else "—" for st in streams for k in KINDS_GPU) + " |")
    return md


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args()

    dispatch = P.newest_run_with("dispatch_empty", a.allow_dirty)
    idle = P.newest_run_with("sync_idle", a.allow_dirty)
    storage = {b: r for b in ("storage_create", "storage_cpu_touch", "storage_gpu_first", "storage_gpu_bw")
               if (r := P.newest_run_with(b, a.allow_dirty))}
    sync_runs = sorted(r for r in P.run_dirs()
                       if (a.allow_dirty or not r.name.endswith("-dirty"))
                       and any(x["benchmark"] == "sync_method" for x in P.read_csv(r / "summary.csv")))
    if not (dispatch and sync_runs and "storage_create" in storage):
        print("error: missing dispatch_overhead, sync_methods or storage_modes results", file=sys.stderr)
        return 1

    out = P.RESULTS / "reports" / f"m5-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    used = {"dispatch": dispatch, "idle": idle, **storage, **{f"sync_{i + 1}": r for i, r in enumerate(sync_runs)}}
    sources = {}
    for name, run in used.items():
        if run is None:
            continue
        env = json.loads((run / "env.json").read_text())
        sources[name] = {"run": str(run.relative_to(P.REPO)), "git_sha": env.get("git_sha"),
                         "thermal_state": env.get("thermal_state"), "power_source": env.get("power_source")}
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    env = json.loads((dispatch / "env.json").read_text())

    md = ["# M5 characterization: dispatch, synchronization and storage modes", "",
          f"Machine: {env.get('chip')} ({env.get('model')}), macOS {env.get('os_version')} ({env.get('os_build')}), "
          f"{env.get('power_source')}. Runs: `sources.json`. Every number is a median of 30 calls "
          "unless stated; CIs are bootstrap 95% CIs of the median.", ""]
    md += dispatch_section(dispatch, out)
    md += sync_section(sync_runs, idle, out)
    md += storage_section(storage, out)
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(f"Written to {out.relative_to(P.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
