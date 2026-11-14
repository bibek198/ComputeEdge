"""M6 report: cache interference (RQ3), thermal drift (RQ4) and follow-up experiments.

Ported from benchmarks/analysis/m6_report.py.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import paths, style
from ..io import metric_table, newest_orig_run_with, pick, read_csv
from ..stats import slowdown_ci
from ..units import size_bytes

PHASES_VS_TINY = [("gpu_write", "GPU writes G"), ("gpu_read", "GPU reads G"), ("cpu_read", "CPU reads G")]
STATE_SHADE = {"fair": "#efeee9", "serious": "#dcdad2", "critical": "#c3c2b7"}


class Samples:
    """Index of metrics.csv: (benchmark, metric, frozenset(params)) -> values."""

    def __init__(self, run: Path):
        """Builds index from run's metrics.csv."""
        self.idx: dict = defaultdict(list)
        with open(run / "metrics.csv", newline="") as f:
            for r in csv.DictReader(f):
                params = frozenset(
                    (k, v) for k, v in r.items() if k not in ("benchmark", "rep", "metric", "value") and v
                )
                self.idx[(r["benchmark"], r["metric"], params)].append(float(r["value"]))

    def get(self, benchmark: str, metric: str, **params: object) -> list[float]:
        """Gets samples matching benchmark, metric and parameters."""
        want = {(k, str(v)) for k, v in params.items()}
        for (b, m, p), vals in self.idx.items():
            if b == benchmark and m == metric and want <= p:
                return vals
        return []


def pct(t: tuple[float, float, float]) -> str:
    """Formats point estimate and bootstrap 95% CI."""
    return f"{t[0]:+.0f}% [{t[1]:+.0f}, {t[2]:+.0f}]"


def ci_excludes_zero(t: tuple[float, float, float]) -> bool:
    """Checks whether 95% CI excludes zero."""
    return t[1] > 0 or t[2] < 0


def cache_section(run: Path, out: Path) -> list[str]:
    """Generates RQ3 cache interference tables and figure."""
    s_idx = Samples(run)
    bench = "cache_interference"
    rows = metric_table(run, bench)
    passes = []
    for r in rows:
        key = (r["pass"], r["ws"])
        if key not in passes:
            passes.append(key)
    gs = sorted({r["g"] for r in rows if r["g"] not in ("-",) and r["phase"] in ("gpu_write",)}, key=size_bytes)

    md = [
        "## RQ3: does GPU work slow the CPU's next pass?",
        "",
        "Timed CPU pass after an untimed interference phase. Slowdowns are median CPU-pass time vs the control "
        "named in each column, with bootstrap 95% CIs. **Bold** = CI excludes zero.",
        "",
        "Controls: *tiny* = same submit-and-poll CPU path with negligible GPU memory traffic and the same "
        "duration; *spin* = plain CPU spin for the same duration; *short* = 300 µs spin.",
        "",
    ]

    def cell(t):
        s = pct(t)
        return f"**{s}**" if ci_excludes_zero(t) else s

    md += [
        "| pass | W | G | tiny vs spin | GPU writes G vs tiny | GPU reads G vs tiny | CPU reads G vs tiny | "
        "GPU concurrent vs short | overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fig_data = defaultdict(dict)
    mid_g = gs[len(gs) // 2] if gs else None
    for p, ws in passes:
        short = s_idx.get(bench, "cpu_ns", **{"pass": p, "ws": ws, "phase": "idle_short"})
        for g in gs:
            tiny = s_idx.get(bench, "cpu_ns", **{"pass": p, "ws": ws, "phase": "gpu_tiny", "g": g})
            spin = s_idx.get(bench, "cpu_ns", **{"pass": p, "ws": ws, "phase": "idle_spin", "g": g})
            if not tiny or not spin:
                continue
            cells = [cell(slowdown_ci(tiny, spin))]
            for ph, _ in PHASES_VS_TINY:
                q_ph = s_idx.get(bench, "cpu_ns", **{"pass": p, "ws": ws, "phase": ph, "g": g})
                t = slowdown_ci(q_ph, tiny)
                cells.append(cell(t))
                if p == "read" and g == mid_g:
                    fig_data[ph][ws] = t
            conc = s_idx.get(bench, "cpu_ns", **{"pass": p, "ws": ws, "phase": "gpu_concurrent", "g": g})
            t = slowdown_ci(conc, short)
            cells.append(cell(t))
            if p == "read" and g == mid_g:
                fig_data["gpu_concurrent"][ws] = t
            ov = s_idx.get(bench, "overlap", **{"pass": p, "ws": ws, "phase": "gpu_concurrent", "g": g})
            cells.append(f"{100 * np.mean(ov):.0f}%" if ov else "—")
            md.append(f"| {p} | {ws} | {g} | " + " | ".join(cells) + " |")

    md += [
        "",
        "Handoff (read pass only): the GPU writes the CPU's own W buffer, then the CPU reads it.",
        "",
        "| W | CPU pass after control (µs) | after GPU wrote it (µs) | slowdown vs tiny |",
        "|---:|---:|---:|---:|",
    ]
    for p, ws in passes:
        same = s_idx.get(bench, "cpu_ns", **{"pass": p, "ws": ws, "phase": "gpu_same"})
        tiny = s_idx.get(bench, "cpu_ns", **{"pass": p, "ws": ws, "phase": "gpu_tiny_same"})
        if same and tiny:
            t = slowdown_ci(same, tiny)
            fig_data["gpu_same"][ws] = t
            md.append(f"| {ws} | {np.median(tiny) / 1e3:.1f} | {np.median(same) / 1e3:.1f} | {cell(t)} |")

    # Figure: read pass, slowdown vs W for each phase (at the middle G).
    wss = [ws for p, ws in passes if p == "read"]
    labels = {
        "gpu_write": f"GPU writes {mid_g} (other buffer)",
        "gpu_read": f"GPU reads {mid_g} (other buffer)",
        "cpu_read": f"CPU reads {mid_g} (other buffer)",
        "gpu_concurrent": f"GPU writing {mid_g} concurrently",
        "gpu_same": "GPU wrote the CPU's own data (handoff)",
    }
    fig, (ax,) = style.new_axes(figsize=(8.0, 4.6))
    x = np.arange(len(wss))
    phases = [ph for ph in ("gpu_write", "gpu_read", "cpu_read", "gpu_concurrent", "gpu_same") if fig_data.get(ph)]
    width = 0.8 / max(len(phases), 1)
    for i, ph in enumerate(phases):
        vals = [fig_data[ph].get(ws, (np.nan, np.nan, np.nan)) for ws in wss]
        mid = np.array([v[0] for v in vals])
        err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
        ax.bar(
            x + (i - (len(phases) - 1) / 2) * width,
            mid,
            width * 0.9,
            color=style.CATEGORICAL[i],
            label=labels[ph],
            zorder=3,
            yerr=err,
            ecolor=style.TEXT_2,
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
    ax.axhline(0, color=style.AXIS, linewidth=1)
    ax.set_xticks(x, [f"W = {w}" for w in wss])
    ax.grid(axis="x", visible=False)
    style.labels(ax, "CPU working set (read pass)", "CPU pass slowdown vs control (%)")
    style.title(
        ax,
        "What slows the CPU's next pass over its own data",
        "Median slowdown vs the tiny-GPU-work control (concurrent: vs 300 µs spin); whiskers = 95% CI.",
    )
    style.legend(ax)
    style.save(fig, out / "cache_interference.png")
    md += ["", "![Cache interference](cache_interference.png)"]
    return md


def thermal_section(run: Path, out: Path) -> list[str]:
    """Generates RQ4 thermal drift figure and tables."""
    ts = read_csv(run / "timeseries.csv")
    load = next(r["load"] for r in read_csv(run / "summary.csv"))
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

    states = sorted((float(r["t_s"]), r["thermal"]) for r in ts)
    spans, cur, s0 = [], states[0][1], states[0][0]
    for t, st in states[1:]:
        if st != cur:
            spans.append((s0, t, cur))
            cur, s0 = st, t
    spans.append((s0, states[-1][0], cur))

    plt = style.plt
    fig = plt.figure(figsize=(8.5, 6.6), dpi=200)
    fig.patch.set_facecolor(style.SURFACE)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    for ax in (ax1, ax2):
        style.style_axes(ax)
        for a, b, st in spans:
            if st in STATE_SHADE:
                ax.axvspan(a / 60, b / 60, color=STATE_SHADE[st], zorder=0, linewidth=0)
    for i, w in enumerate(["cpu_1024", "cpu_2048", "gpu_1024", "gpu_2048"]):
        t, v = rolling(by[w])
        ax1.plot(
            t,
            flops[w] / v,
            color=style.CATEGORICAL[i],
            linewidth=2,
            linestyle="-" if w.startswith("cpu") else "--",
            label=w.replace("_", " · ") + "²",
            zorder=3,
        )
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("GFLOP/s (30 s rolling median)", color=style.TEXT_2, fontsize=10)
    ax1.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=4, labelcolor=style.TEXT)
    ax1.set_title(
        f"Sustained load ({load}) on a fanless MacBook Air M4",
        loc="left",
        color=style.TEXT,
        fontsize=12,
        pad=22,
    )
    shaded = sorted({st for _, _, st in spans if st in STATE_SHADE}, key=list(STATE_SHADE).index)
    ax1.text(
        0,
        1.03,
        "Solid = CPU (Accelerate), dashed = GPU (MPS). Shading = macOS thermal state: "
        + (", ".join(f"{st} ({'light' if st == 'fair' else 'darker'})" for st in shaded) or "nominal throughout"),
        transform=ax1.transAxes,
        color=style.TEXT_2,
        fontsize=8.5,
    )
    for i, n in enumerate(["1024", "2048"]):
        tc, vc = rolling(by[f"cpu_{n}"])
        tg, vg = rolling(by[f"gpu_{n}"])
        k = min(len(vc), len(vg))
        ax2.plot(tc[:k], vc[:k] / vg[:k], color=style.CATEGORICAL[i], linewidth=2, label=f"{n}²", zorder=3)
    ax2.axhline(1.0, color=style.TEXT_2, linewidth=1, linestyle=":")
    ax2.text(
        1.0,
        1.0,
        "CPU = GPU ",
        transform=ax2.get_yaxis_transform(),
        color=style.TEXT_2,
        fontsize=8,
        ha="right",
        va="top",
    )
    ax2.set_ylabel("CPU time / GPU time\n(>1: GPU faster)", color=style.TEXT_2, fontsize=10)
    ax2.set_xlabel("Minutes under load", color=style.TEXT_2, fontsize=10)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=style.TEXT)
    style.save(fig, out / "thermal_drift.png")

    minutes = int(max(float(r["t_s"]) for r in ts) // 60) + 1
    md = [
        "",
        "## RQ4: thermal drift under sustained load",
        "",
        f"Load: `{load}`; one synchronous call per workload per ~1 s cycle. Medians per minute; the ratio column "
        "is CPU time / GPU time (>1 means the GPU is faster, 1 = crossover).",
        "",
        "![Thermal drift](thermal_drift.png)",
        "",
        "| minute | thermal (most common) | CPU 1024² ms | GPU 1024² ms | ratio 1024² | CPU 2048² ms | GPU 2048² ms | "
        "ratio 2048² | empty GPU round trip µs |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in range(minutes):
        def med(w, min_idx=m):
            v = [p[1] for p in by[w] if int(p[0] // 60) == min_idx]
            return np.median(v) if v else np.nan

        sts = [p[2] for p in by["cpu_2048"] if int(p[0] // 60) == m]
        common = max(set(sts), key=sts.count) if sts else "—"
        c1, g1, c2, g2, e = (med(w) for w in ("cpu_1024", "gpu_1024", "cpu_2048", "gpu_2048", "gpu_empty"))
        md.append(
            f"| {m} | {common} | {c1 / 1e6:.2f} | {g1 / 1e6:.2f} | {c1 / g1:.2f} | {c2 / 1e6:.2f} | "
            f"{g2 / 1e6:.2f} | {c2 / g2:.2f} | {e / 1e3:.0f} |"
        )
    total = states[-1][0] - states[0][0]
    dur = defaultdict(float)
    for a, b, st in spans:
        dur[st] += b - a
    md += [
        "",
        "Time in each thermal state: "
        + ", ".join(f"{st} {d / 60:.1f} min ({100 * d / max(total, 1):.0f}%)" for st, d in dur.items()),
    ]
    first = {st: a for a, b, st in reversed(spans)}
    if any(st != "nominal" for st in first):
        md.append(
            "First entry: "
            + ", ".join(f"{st} at {a / 60:.1f} min" for st, a in sorted(first.items(), key=lambda x: x[1]))
        )
    return md


def followups_section(run: Path, out: Path) -> list[str]:
    """Generates follow-up experiment tables and figure."""
    md = ["", "## Follow-ups on open M5 findings", ""]
    rows = metric_table(run, "followup_wake")
    if rows:
        sizes = sorted({r["size"] for r in rows}, key=size_bytes)
        md += [
            "### Finding 7: does wake latency grow with bytes written (cache writeback)?",
            "",
            "Spin on MTLSharedEvent. Medians in µs.",
            "",
            "| size | read: GPU time | read: wake | write: GPU time | write: wake |",
            "|---:|---:|---:|---:|---:|",
        ]
        for s in sizes:
            read_gpu = pick(rows, "gpu_ns", stream="read", size=s)["median"] / 1e3
            read_wake = pick(rows, "wake_ns", stream="read", size=s)["median"] / 1e3
            write_gpu = pick(rows, "gpu_ns", stream="write", size=s)["median"] / 1e3
            write_wake = pick(rows, "wake_ns", stream="write", size=s)["median"] / 1e3
            md.append(f"| {s} | {read_gpu:.0f} | {read_wake:.0f} | {write_gpu:.0f} | {write_wake:.0f} |")
    rows = metric_table(run, "followup_cpu_write")
    if rows:
        b = size_bytes(rows[0]["size"])
        md += [
            "",
            "### Finding 14: CPU write rate by memory source (warm, 64 MiB)",
            "",
            "GB/s from medians.",
            "",
            "| source | memset | store loop | read |",
            "|---|---:|---:|---:|",
        ]
        for src in dict.fromkeys(r["source"] for r in rows):
            m_ns = b / pick(rows, "memset_ns", source=src)["median"]
            s_ns = b / pick(rows, "store_loop_ns", source=src)["median"]
            r_ns = b / pick(rows, "read_ns", source=src)["median"]
            md.append(f"| {src} | {m_ns:.1f} | {s_ns:.1f} | {r_ns:.1f} |")
    rows = metric_table(run, "followup_cpu_write_fresh")
    if rows:
        b = size_bytes(rows[0]["size"])
        md += [
            "",
            "Successive memsets into newly created memory (GB/s; memalign/nocopy reuse allocator pages):",
            "",
            "| source | write 1 | write 2 | write 3 | write 4 |",
            "|---|---:|---:|---:|---:|",
        ]
        for src in dict.fromkeys(r["source"] for r in rows):
            md.append(
                f"| {src} | "
                + " | ".join(f"{b / pick(rows, f'write{i}_ns', source=src)['median']:.1f}" for i in range(1, 5))
                + " |"
            )
    rows = metric_table(run, "followup_gpu_write")
    if rows:
        sizes = sorted({r["size"] for r in rows}, key=size_bytes)
        md += [
            "",
            "### Finding 15: GPU write bandwidth, one buffer vs 256 MiB chunks",
            "",
            "GB/s = bytes / GPU time (two-pass: per pass).",
            "",
            "| size | one buffer | one buffer, 2nd pass in same CB | as 256 MiB buffers |",
            "|---:|---:|---:|---:|",
        ]
        xs, one, chunks = [], [], []
        for s in sizes:
            b = size_bytes(s)
            o = pick(rows, "gpu_ns", layout="one_buffer", size=s)
            two = pick(rows, "two_pass_gpu_ns", layout="one_buffer", size=s)
            c = pick(rows, "gpu_ns", layout="256M_chunks", size=s)
            xs.append(b)
            one.append(b / o["median"])
            chunks.append(b / c["median"] if c else np.nan)
            md.append(
                f"| {s} | {b / o['median']:.1f} | {2 * b / two['median']:.1f} | "
                + (f"{b / c['median']:.1f}" if c else "—")
                + " |"
            )
        fig, (ax,) = style.new_axes(figsize=(8.0, 4.4))
        ax.set_xscale("log", base=2)
        style.series(ax, xs, one, 0, "one buffer")
        cx = [x for x, c in zip(xs, chunks, strict=True) if not np.isnan(c)]
        style.series(ax, cx, [c for c in chunks if not np.isnan(c)], 1, "same bytes as 256 MiB buffers")
        ax.set_xticks(xs, sizes, fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(0, max(one + [c for c in chunks if not np.isnan(c)]) * 1.15)
        style.labels(ax, "Bytes written per pass", "GPU write bandwidth (GB/s)")
        style.title(
            ax,
            "GPU writes slow down with buffer size, not with total bytes",
            "Private buffers, float4 grid-stride write kernel; median of 30.",
        )
        style.legend(ax, loc="lower left")
        style.save(fig, out / "gpu_write_size.png")
        md += ["", "![GPU write vs buffer size](gpu_write_size.png)"]
    return md


def _shown(path: Path) -> Path:
    try:
        return path.relative_to(paths.REPO)
    except ValueError:
        return path


def run_interaction_report(
    allow_dirty: bool = False,
    out_dir: Path | None = None,
    data_dir: Path | None = None,
) -> int:
    """Runs the M6 report on cache interference, thermal drift and follow-ups."""
    data_root = data_dir or paths.DATA
    runs = {
        b: newest_orig_run_with(b, allow_dirty, data_root)
        for b in ("cache_interference", "thermal_drift", "followup_wake")
    }
    if not any(runs.values()):
        print("error: no M6 results", file=sys.stderr)
        return 1
    out = out_dir or paths.DERIVED / "reports" / f"m6-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, run in runs.items():
        if run:
            env = json.loads((run / "env.json").read_text())
            sources[name] = {
                "run": str(run.relative_to(paths.REPO)),
                "git_sha": env.get("git_sha"),
                "thermal_state": env.get("thermal_state"),
                "power_source": env.get("power_source"),
            }
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    env = json.loads((next(r for r in runs.values() if r) / "env.json").read_text())
    md = [
        "# M6: cache interference, thermal drift, and follow-ups",
        "",
        f"Machine: {env.get('chip')} ({env.get('model')}), macOS {env.get('os_version')} ({env.get('os_build')}), "
        f"{env.get('power_source')}. Runs: `sources.json`.",
        "",
    ]
    if runs["cache_interference"]:
        md += cache_section(runs["cache_interference"], out)
    if runs["thermal_drift"]:
        md += thermal_section(runs["thermal_drift"], out)
    if runs["followup_wake"]:
        md += followups_section(runs["followup_wake"], out)
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(f"Written to {_shown(out)}")
    return 0
