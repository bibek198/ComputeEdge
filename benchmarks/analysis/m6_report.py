#!/usr/bin/env python3
"""M6 report: cache interference (RQ3), thermal drift (RQ4) and the follow-up
experiments on the open M5 findings. Uses the newest clean run of each
benchmark and writes data/derived/reports/m6-<stamp>/ (report.md, PNGs,
sources.json).

Usage: .venv/bin/python benchmarks/analysis/m6_report.py [--allow-dirty]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import ce_plot as P

PHASES_VS_TINY = [("gpu_write", "GPU writes G"), ("gpu_read", "GPU reads G"), ("cpu_read", "CPU reads G")]


def size_bytes(label: str) -> int:
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(label[:-1]) * mult[label[-1]] if label[-1] in mult else int(label)


class Samples:
    """Index of metrics.csv: (benchmark, metric, frozenset(params)) -> values."""

    def __init__(self, run: Path):
        self.idx: dict = defaultdict(list)
        with open(run / "metrics.csv", newline="") as f:
            for r in csv.DictReader(f):
                params = frozenset((k, v) for k, v in r.items() if k not in ("benchmark", "rep", "metric", "value") and v)
                self.idx[(r["benchmark"], r["metric"], params)].append(float(r["value"]))

    def get(self, benchmark: str, metric: str, **params) -> list[float]:
        want = {(k, str(v)) for k, v in params.items()}
        for (b, m, p), vals in self.idx.items():
            if b == benchmark and m == metric and want <= p:
                return vals
        return []


def pct(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.0f}% [{t[1]:+.0f}, {t[2]:+.0f}]"


def ci_excludes_zero(t) -> bool:
    return t[1] > 0 or t[2] < 0


# ---------------------------------------------------------------------------
# RQ3: cache interference
# ---------------------------------------------------------------------------

def cache_section(run: Path, out: Path) -> list[str]:
    S = Samples(run)
    B = "cache_interference"
    rows = P.metric_table(run, B)
    passes = []
    for r in rows:
        key = (r["pass"], r["ws"])
        if key not in passes:
            passes.append(key)
    gs = sorted({r["g"] for r in rows if r["g"] not in ("-",) and r["phase"] in ("gpu_write",)}, key=size_bytes)

    md = ["## RQ3: does GPU work slow the CPU's next pass?", "",
          "Timed CPU pass after an untimed interference phase. Slowdowns are median CPU-pass time vs the control "
          "named in each column, with bootstrap 95% CIs. **Bold** = CI excludes zero.", "",
          "Controls: *tiny* = same submit-and-poll CPU path with negligible GPU memory traffic and the same "
          "duration; *spin* = plain CPU spin for the same duration; *short* = 300 µs spin.", ""]

    def cell(t):
        s = pct(t)
        return f"**{s}**" if ci_excludes_zero(t) else s

    md += ["| pass | W | G | tiny vs spin | GPU writes G vs tiny | GPU reads G vs tiny | CPU reads G vs tiny | "
           "GPU concurrent vs short | overlap |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fig_data = defaultdict(dict)  # phase -> {ws: slowdown} at the middle G, read pass
    mid_g = gs[len(gs) // 2] if gs else None
    for p, ws in passes:
        short = S.get(B, "cpu_ns", **{"pass": p, "ws": ws, "phase": "idle_short"})
        for g in gs:
            q = lambda ph: S.get(B, "cpu_ns", **{"pass": p, "ws": ws, "phase": ph, "g": g})
            tiny, spin = q("gpu_tiny"), q("idle_spin")
            if not tiny or not spin:
                continue
            cells = [cell(P.slowdown_ci(tiny, spin))]
            for ph, _ in PHASES_VS_TINY:
                t = P.slowdown_ci(q(ph), tiny)
                cells.append(cell(t))
                if p == "read" and g == mid_g:
                    fig_data[ph][ws] = t
            conc = q("gpu_concurrent")
            t = P.slowdown_ci(conc, short)
            cells.append(cell(t))
            if p == "read" and g == mid_g:
                fig_data["gpu_concurrent"][ws] = t
            ov = S.get(B, "overlap", **{"pass": p, "ws": ws, "phase": "gpu_concurrent", "g": g})
            cells.append(f"{100 * np.mean(ov):.0f}%" if ov else "—")
            md.append(f"| {p} | {ws} | {g} | " + " | ".join(cells) + " |")

    md += ["", "Handoff (read pass only): the GPU writes the CPU's own W buffer, then the CPU reads it.", "",
           "| W | CPU pass after control (µs) | after GPU wrote it (µs) | slowdown vs tiny |", "|---:|---:|---:|---:|"]
    for p, ws in passes:
        same = S.get(B, "cpu_ns", **{"pass": p, "ws": ws, "phase": "gpu_same"})
        tiny = S.get(B, "cpu_ns", **{"pass": p, "ws": ws, "phase": "gpu_tiny_same"})
        if same and tiny:
            t = P.slowdown_ci(same, tiny)
            fig_data["gpu_same"][ws] = t
            md.append(f"| {ws} | {np.median(tiny) / 1e3:.1f} | {np.median(same) / 1e3:.1f} | {cell(t)} |")

    # Figure: read pass, slowdown vs W for each phase (at the middle G).
    wss = [ws for p, ws in passes if p == "read"]
    labels = {"gpu_write": f"GPU writes {mid_g} (other buffer)", "gpu_read": f"GPU reads {mid_g} (other buffer)",
              "cpu_read": f"CPU reads {mid_g} (other buffer)", "gpu_concurrent": f"GPU writing {mid_g} concurrently",
              "gpu_same": "GPU wrote the CPU's own data (handoff)"}
    fig, (ax,) = P.new_axes(figsize=(8.0, 4.6))
    x = np.arange(len(wss))
    phases = [ph for ph in ("gpu_write", "gpu_read", "cpu_read", "gpu_concurrent", "gpu_same") if fig_data.get(ph)]
    width = 0.8 / max(len(phases), 1)
    for i, ph in enumerate(phases):
        vals = [fig_data[ph].get(ws, (np.nan, np.nan, np.nan)) for ws in wss]
        mid = np.array([v[0] for v in vals])
        err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
        ax.bar(x + (i - (len(phases) - 1) / 2) * width, mid, width * 0.9, color=P.CATEGORICAL[i], label=labels[ph],
               zorder=3, yerr=err, ecolor=P.TEXT_2, capsize=2, error_kw={"linewidth": 0.8})
    ax.axhline(0, color=P.AXIS, linewidth=1)
    ax.set_xticks(x, [f"W = {w}" for w in wss])
    ax.grid(axis="x", visible=False)
    P.labels(ax, "CPU working set (read pass)", "CPU pass slowdown vs control (%)")
    P.title(ax, "What slows the CPU's next pass over its own data",
            "Median slowdown vs the tiny-GPU-work control (concurrent: vs 300 µs spin); whiskers = 95% CI.")
    P.legend(ax)
    P.save(fig, out / "cache_interference.png")
    md += ["", "![Cache interference](cache_interference.png)"]
    return md


# ---------------------------------------------------------------------------
# RQ4: thermal drift
# ---------------------------------------------------------------------------

STATE_SHADE = {"fair": "#efeee9", "serious": "#dcdad2", "critical": "#c3c2b7"}


def thermal_section(run: Path, out: Path) -> list[str]:
    ts = P.read_csv(run / "timeseries.csv")
    load = next(r["load"] for r in P.read_csv(run / "summary.csv"))
    by = defaultdict(list)
    for r in ts:
        by[r["workload"]].append((float(r["t_s"]), float(r["ns"]), r["thermal"]))
    flops = {"cpu_1024": 2 * 1024**3, "gpu_1024": 2 * 1024**3, "cpu_2048": 2 * 2048**3, "gpu_2048": 2 * 2048**3}

    def rolling(points, window_s=30.0):
        t = np.array([p[0] for p in points])
        v = np.array([p[1] for p in points])
        out_t, out_v = [], []
        for i in range(len(t)):
            m = (t > t[i] - window_s) & (t <= t[i])
            out_t.append(t[i] / 60)
            out_v.append(np.median(v[m]))
        return np.array(out_t), np.array(out_v)

    # Thermal-state spans (from any workload's samples, in time order).
    states = sorted(((float(r["t_s"]), r["thermal"]) for r in ts))
    spans, cur, s0 = [], states[0][1], states[0][0]
    for t, st in states[1:]:
        if st != cur:
            spans.append((s0, t, cur))
            cur, s0 = st, t
    spans.append((s0, states[-1][0], cur))

    fig, axes = P.new_axes(figsize=(8.5, 6.6), ncols=1)
    plt = P.plt
    fig.clf()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    for ax in (ax1, ax2):
        ax.set_facecolor(P.SURFACE)
        ax.grid(True, color=P.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(P.AXIS)
        ax.tick_params(colors=P.MUTED, labelcolor=P.TEXT_2, labelsize=9)
        for a, b, st in spans:
            if st in STATE_SHADE:
                ax.axvspan(a / 60, b / 60, color=STATE_SHADE[st], zorder=0, linewidth=0)
    for i, w in enumerate(["cpu_1024", "cpu_2048", "gpu_1024", "gpu_2048"]):
        t, v = rolling(by[w])
        ax1.plot(t, flops[w] / v, color=P.CATEGORICAL[i], linewidth=2, linestyle="-" if w.startswith("cpu") else "--",
                 label=w.replace("_", " · ") + "²", zorder=3)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("GFLOP/s (30 s rolling median)", color=P.TEXT_2, fontsize=10)
    ax1.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=4, labelcolor=P.TEXT)
    ax1.set_title(f"Sustained load ({load}) on a fanless MacBook Air M4", loc="left", color=P.TEXT, fontsize=12, pad=22)
    shaded = sorted({st for _, _, st in spans if st in STATE_SHADE}, key=list(STATE_SHADE).index)
    ax1.text(0, 1.03, "Solid = CPU (Accelerate), dashed = GPU (MPS). Shading = macOS thermal state: "
             + (", ".join(f"{st} ({'light' if st == 'fair' else 'darker'})" for st in shaded) or "nominal throughout"),
             transform=ax1.transAxes, color=P.TEXT_2, fontsize=8.5)
    for i, n in enumerate(["1024", "2048"]):
        tc, vc = rolling(by[f"cpu_{n}"])
        tg, vg = rolling(by[f"gpu_{n}"])
        k = min(len(vc), len(vg))
        ax2.plot(tc[:k], vc[:k] / vg[:k], color=P.CATEGORICAL[i], linewidth=2, label=f"{n}²", zorder=3)
    ax2.axhline(1.0, color=P.TEXT_2, linewidth=1, linestyle=":")
    ax2.text(1.0, 1.0, "CPU = GPU ", transform=ax2.get_yaxis_transform(), color=P.TEXT_2, fontsize=8,
             ha="right", va="top")
    ax2.set_ylabel("CPU time / GPU time\n(>1: GPU faster)", color=P.TEXT_2, fontsize=10)
    ax2.set_xlabel("Minutes under load", color=P.TEXT_2, fontsize=10)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=P.TEXT)
    P.save(fig, out / "thermal_drift.png")

    # Per-minute table and state durations.
    minutes = int(max(float(r["t_s"]) for r in ts) // 60) + 1
    md = ["", "## RQ4: thermal drift under sustained load", "",
          f"Load: `{load}`; one synchronous call per workload per ~1 s cycle. Medians per minute; the ratio column "
          "is CPU time / GPU time (>1 means the GPU is faster, 1 = crossover).", "",
          "![Thermal drift](thermal_drift.png)", "",
          "| minute | thermal (most common) | CPU 1024² ms | GPU 1024² ms | ratio 1024² | CPU 2048² ms | GPU 2048² ms | "
          "ratio 2048² | empty GPU round trip µs |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in range(minutes):
        def med(w):
            v = [p[1] for p in by[w] if int(p[0] // 60) == m]
            return np.median(v) if v else np.nan
        sts = [p[2] for p in by["cpu_2048"] if int(p[0] // 60) == m]
        common = max(set(sts), key=sts.count) if sts else "—"
        c1, g1, c2, g2, e = (med(w) for w in ("cpu_1024", "gpu_1024", "cpu_2048", "gpu_2048", "gpu_empty"))
        md.append(f"| {m} | {common} | {c1 / 1e6:.2f} | {g1 / 1e6:.2f} | {c1 / g1:.2f} | {c2 / 1e6:.2f} | "
                  f"{g2 / 1e6:.2f} | {c2 / g2:.2f} | {e / 1e3:.0f} |")
    total = states[-1][0] - states[0][0]
    dur = defaultdict(float)
    for a, b, st in spans:
        dur[st] += b - a
    md += ["", "Time in each thermal state: " + ", ".join(f"{st} {d / 60:.1f} min ({100 * d / max(total, 1):.0f}%)"
                                                         for st, d in dur.items())]
    first = {st: a for a, b, st in reversed(spans)}
    if any(st != "nominal" for st in first):
        md.append("First entry: " + ", ".join(f"{st} at {a / 60:.1f} min" for st, a in sorted(first.items(),
                                                                                        key=lambda x: x[1])))
    return md


# ---------------------------------------------------------------------------
# Follow-ups
# ---------------------------------------------------------------------------

def followups_section(run: Path, out: Path) -> list[str]:
    md = ["", "## Follow-ups on open M5 findings", ""]
    rows = P.metric_table(run, "followup_wake")
    if rows:
        sizes = sorted({r["size"] for r in rows}, key=size_bytes)
        md += ["### Finding 7: does wake latency grow with bytes written (cache writeback)?", "",
               "Spin on MTLSharedEvent. Medians in µs.", "",
               "| size | read: GPU time | read: wake | write: GPU time | write: wake |", "|---:|---:|---:|---:|---:|"]
        for s in sizes:
            g = lambda st, m: P.pick(rows, m, stream=st, size=s)
            md.append(f"| {s} | {g('read', 'gpu_ns')['median'] / 1e3:.0f} | {g('read', 'wake_ns')['median'] / 1e3:.0f} | "
                      f"{g('write', 'gpu_ns')['median'] / 1e3:.0f} | {g('write', 'wake_ns')['median'] / 1e3:.0f} |")
    rows = P.metric_table(run, "followup_cpu_write")
    if rows:
        b = size_bytes(rows[0]["size"])
        md += ["", "### Finding 14: CPU write rate by memory source (warm, 64 MiB)", "", "GB/s from medians.", "",
               "| source | memset | store loop | read |", "|---|---:|---:|---:|"]
        for src in dict.fromkeys(r["source"] for r in rows):
            g = lambda m: b / P.pick(rows, m, source=src)["median"]
            md.append(f"| {src} | {g('memset_ns'):.1f} | {g('store_loop_ns'):.1f} | {g('read_ns'):.1f} |")
    rows = P.metric_table(run, "followup_cpu_write_fresh")
    if rows:
        b = size_bytes(rows[0]["size"])
        md += ["", "Successive memsets into newly created memory (GB/s; memalign/nocopy reuse allocator pages):", "",
               "| source | write 1 | write 2 | write 3 | write 4 |", "|---|---:|---:|---:|---:|"]
        for src in dict.fromkeys(r["source"] for r in rows):
            md.append(f"| {src} | " + " | ".join(
                f"{b / P.pick(rows, f'write{i}_ns', source=src)['median']:.1f}" for i in range(1, 5)) + " |")
    rows = P.metric_table(run, "followup_gpu_write")
    if rows:
        sizes = sorted({r["size"] for r in rows}, key=size_bytes)
        md += ["", "### Finding 15: GPU write bandwidth, one buffer vs 256 MiB chunks", "",
               "GB/s = bytes / GPU time (two-pass: per pass).", "",
               "| size | one buffer | one buffer, 2nd pass in same CB | as 256 MiB buffers |", "|---:|---:|---:|---:|"]
        xs, one, chunks = [], [], []
        for s in sizes:
            b = size_bytes(s)
            o = P.pick(rows, "gpu_ns", layout="one_buffer", size=s)
            two = P.pick(rows, "two_pass_gpu_ns", layout="one_buffer", size=s)
            c = P.pick(rows, "gpu_ns", layout="256M_chunks", size=s)
            xs.append(b)
            one.append(b / o["median"])
            chunks.append(b / c["median"] if c else np.nan)
            md.append(f"| {s} | {b / o['median']:.1f} | {2 * b / two['median']:.1f} | "
                      + (f"{b / c['median']:.1f}" if c else "—") + " |")
        fig, (ax,) = P.new_axes(figsize=(8.0, 4.4))
        ax.set_xscale("log", base=2)
        P.series(ax, xs, one, 0, "one buffer")
        cx = [x for x, c in zip(xs, chunks) if not np.isnan(c)]
        P.series(ax, cx, [c for c in chunks if not np.isnan(c)], 1, "same bytes as 256 MiB buffers")
        ax.set_xticks(xs, sizes, fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(0, max(one + [c for c in chunks if not np.isnan(c)]) * 1.15)
        P.labels(ax, "Bytes written per pass", "GPU write bandwidth (GB/s)")
        P.title(ax, "GPU writes slow down with buffer size, not with total bytes",
                "Private buffers, float4 grid-stride write kernel; median of 30.")
        P.legend(ax, loc="lower left")
        P.save(fig, out / "gpu_write_size.png")
        md += ["", "![GPU write vs buffer size](gpu_write_size.png)"]
    return md


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args()
    runs = {b: P.newest_run_with(b, a.allow_dirty) for b in ("cache_interference", "thermal_drift", "followup_wake")}
    if not any(runs.values()):
        print("error: no M6 results", file=sys.stderr)
        return 1
    out = P.DERIVED / "reports" / f"m6-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    sources = {}
    for name, run in runs.items():
        if run:
            env = json.loads((run / "env.json").read_text())
            sources[name] = {"run": str(run.relative_to(P.REPO)), "git_sha": env.get("git_sha"),
                             "thermal_state": env.get("thermal_state"), "power_source": env.get("power_source")}
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    env = json.loads((next(r for r in runs.values() if r) / "env.json").read_text())
    md = ["# M6: cache interference, thermal drift, and follow-ups", "",
          f"Machine: {env.get('chip')} ({env.get('model')}), macOS {env.get('os_version')} ({env.get('os_build')}), "
          f"{env.get('power_source')}. Runs: `sources.json`.", ""]
    if runs["cache_interference"]:
        md += cache_section(runs["cache_interference"], out)
    if runs["thermal_drift"]:
        md += thermal_section(runs["thermal_drift"], out)
    if runs["followup_wake"]:
        md += followups_section(runs["followup_wake"], out)
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(f"Written to {out.relative_to(P.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
