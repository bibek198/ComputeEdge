"""Paper figures (single-column PDFs) from one session's raw results.

Ported from paper/make_figures.py; draws:
  - round_trip.pdf: breakdown of dispatch/sync overhead across idle gaps (sync_idle)
  - first_use.pdf: first GPU use / residency vs reuse across buffer sizes (storage_gpu_first)
  - thermal.pdf: CPU vs GPU crossover drift under sustained load (thermal_drift)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import paths, style
from ..claims.registry import need
from ..io import Session, label, load_sessions

W = 3.45  # IEEE column width, inches


def _axes(h: float):
    plt = style.plt
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
    })
    fig, ax = plt.subplots(figsize=(W, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color=style.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style.AXIS)
    ax.tick_params(colors=style.MUTED, labelcolor=style.TEXT_2)
    return fig, ax


def _save(fig, path: Path):
    fig.tight_layout(pad=0.3)
    fig.savefig(path, facecolor="white")
    style.plt.close(fig)


def round_trip(s: Session, out_dir: Path) -> None:
    """Draws round_trip.pdf: median time per phase across GPU idle gaps."""
    gaps = ["0", "0.1", "1", "10", "100"]
    comps = [
        ("encode_ns", "encode"),
        ("commit_ns", "commit"),
        ("queue_ns", "queue"),
        ("gpu_ns", "GPU exec"),
        ("wake_ns", "wake"),
    ]
    fig, ax = _axes(1.75)
    y = np.arange(len(gaps))
    left = np.zeros(len(gaps))
    for i, (m, lbl) in enumerate(comps):
        v = np.array([
            max(need(s.metric("sync_idle", m, sync="wait_until_completed", gap_ms=g)), 0) / 1e3
            for g in gaps
        ])
        ax.barh(
            y,
            v,
            left=left,
            height=0.62,
            color=style.CATEGORICAL[i],
            edgecolor="white",
            linewidth=1,
            label=lbl,
            zorder=3,
        )
        left += v
    for yy, tot in zip(y, left, strict=True):
        ax.text(tot + 6, yy, f"{tot:.0f}", va="center", fontsize=6.5, color=style.TEXT_2)
    ax.set_yticks(y, [f"{g} ms" for g in gaps])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 1.12)
    ax.set_xlabel("Median time per phase (µs)")
    ax.set_ylabel("GPU idle before call")
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2)
    _save(fig, out_dir / "round_trip.pdf")


def first_use(s: Session, out_dir: Path) -> None:
    """Draws first_use.pdf: first-touch and rewrap overhead across buffer sizes."""
    sizes = ["16K", "256K", "4M", "64M", "256M", "1G"]
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    xs = [int(z[:-1]) * mult[z[-1]] for z in sizes]
    series_spec = [
        ("nocopy", "reused", "same MTLBuffer reused", "--"),
        ("nocopy", "rewrap", "new no-copy wrapper, resident memory", ":"),
        ("shared", "fresh", "new Shared buffer", "-"),
    ]
    fig, ax = _axes(1.85)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    for i, (kind, buf, lbl, ls) in enumerate(series_spec):
        ys = [
            need(s.metric("storage_gpu_first", "total_ns", kind=kind, size=z, buffer=buf)) / 1e3
            for z in sizes
        ]
        ax.plot(
            xs,
            ys,
            color=style.CATEGORICAL[i],
            linestyle=ls,
            linewidth=1.6,
            marker=style.MARKERS[i],
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=lbl,
            zorder=3,
        )
    ax.set_xticks(xs, sizes)
    ax.minorticks_off()
    ax.set_xlabel("Size of the bound (untouched) buffer")
    ax.set_ylabel("Round trip (µs)")
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    _save(fig, out_dir / "first_use.pdf")


def thermal(s: Session, out_dir: Path) -> str:
    """Draws thermal.pdf: CPU/GPU crossover ratio over minutes of sustained load."""
    ts_rows, load = need(s.timeseries())
    by = defaultdict(list)
    for r in ts_rows:
        by[r.workload].append((r.t_s, r.ns, r.thermal))

    def rolling(points, window=30.0):
        t = np.array([p[0] for p in points])
        v = np.array([p[1] for p in points])
        return t / 60, np.array([np.median(v[(t > ti - window) & (t <= ti)]) for ti in t])

    fig, ax = _axes(1.75)
    states = sorted((r.t_s, r.thermal) for r in ts_rows)
    start = next((t for t, st in states if st != "nominal"), None)
    if start is not None:
        ax.axvspan(start / 60, states[-1][0] / 60, color="#efeee9", zorder=0, linewidth=0)
        ax.text(
            start / 60 + 0.15,
            0.97,
            "thermal state: fair",
            transform=ax.get_xaxis_transform(),
            fontsize=6.3,
            color=style.TEXT_2,
            va="top",
        )
    for i, n in enumerate(["1024", "2048"]):
        tc, vc = rolling(by[f"cpu_{n}"])
        _, vg = rolling(by[f"gpu_{n}"])
        k = min(len(vc), len(vg))
        ax.plot(tc[:k], vc[:k] / vg[:k], color=style.CATEGORICAL[i], linewidth=1.5, label=f"SGEMM {n}²", zorder=3)
    ax.axhline(1, color=style.TEXT_2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Minutes under sustained load")
    ax.set_ylabel("CPU time / GPU time")
    ax.legend(loc="center right", frameon=False)
    _save(fig, out_dir / "thermal.pdf")
    return load


def make_figures(
    session_name: str = "orig",
    out_dir: Path | None = None,
    data_dir: Path | None = None,
) -> int:
    """Builds paper figures (PDFs) from the selected session's raw data."""
    fig_dir = out_dir or paths.PAPER_FIGURES
    data = data_dir or paths.DATA
    sessions = {label(s): s for s in load_sessions(False, data)}
    if session_name not in sessions:
        raise ValueError(f"unknown session {session_name}; available: {', '.join(sessions)}")
    s = sessions[session_name]
    fig_dir.mkdir(parents=True, exist_ok=True)
    round_trip(s, fig_dir)
    first_use(s, fig_dir)
    load = thermal(s, fig_dir)
    used = {
        b: str(need(s.run_with(b)).relative_to(paths.REPO))
        for b in ("sync_idle", "storage_gpu_first", "thermal_drift")
    }
    (fig_dir / "sources.json").write_text(
        json.dumps({"session": session_name, "thermal_load": load, "runs": used}, indent=2) + "\n"
    )
    print(f"figures from session {session_name}: {', '.join(sorted(p.name for p in fig_dir.glob('*.pdf')))}")
    return 0
