#!/usr/bin/env python3
"""Paper figures (single-column PDFs) from one session's raw results.

    .venv/bin/python paper/make_figures.py [--session NAME]

--session is "orig" (the original M4-M6 runs, default) or a confirmation session
index such as "s0" (data/sessions/<stamp>_s0_<sha>/). The thermal figure uses the newest mixed-load
thermal run in the chosen session. Sources are recorded in figures/sources.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmarks" / "analysis"))
import ce_plot as P  # noqa: E402
import confirm_claims as C  # noqa: E402

plt = P.plt
FIG = HERE / "figures"
W = 3.45  # IEEE column width, inches

plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.8, "pdf.fonttype": 42, "font.family": "sans-serif"})


def axes(h: float):
    fig, ax = plt.subplots(figsize=(W, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color=P.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(P.AXIS)
    ax.tick_params(colors=P.MUTED, labelcolor=P.TEXT_2)
    return fig, ax


def save(fig, name: str):
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


def round_trip(s: C.Session):
    gaps = ["0", "0.1", "1", "10", "100"]
    comps = [("encode_ns", "encode"), ("commit_ns", "commit"), ("queue_ns", "queue"), ("gpu_ns", "GPU exec"),
             ("wake_ns", "wake")]
    fig, ax = axes(1.75)
    y = np.arange(len(gaps))
    left = np.zeros(len(gaps))
    for i, (m, label) in enumerate(comps):
        v = np.array([max(C.need(s.metric("sync_idle", m, sync="wait_until_completed", gap_ms=g)), 0) / 1e3
                      for g in gaps])
        ax.barh(y, v, left=left, height=0.62, color=P.CATEGORICAL[i], edgecolor="white", linewidth=1, label=label,
                zorder=3)
        left += v
    for yy, tot in zip(y, left):
        ax.text(tot + 6, yy, f"{tot:.0f}", va="center", fontsize=6.5, color=P.TEXT_2)
    ax.set_yticks(y, [f"{g} ms" for g in gaps])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 1.12)
    ax.set_xlabel("Median time per phase (µs)")
    ax.set_ylabel("GPU idle before call")
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2)
    save(fig, "round_trip.pdf")


def first_use(s: C.Session):
    sizes = ["16K", "256K", "4M", "64M", "256M", "1G"]
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    xs = [int(z[:-1]) * mult[z[-1]] for z in sizes]
    series = [("nocopy", "reused", "same MTLBuffer reused", "--"),
              ("nocopy", "rewrap", "new no-copy wrapper, resident memory", ":"),
              ("shared", "fresh", "new Shared buffer", "-")]
    fig, ax = axes(1.85)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    for i, (kind, buf, label, ls) in enumerate(series):
        ys = [C.need(s.metric("storage_gpu_first", "total_ns", kind=kind, size=z, buffer=buf)) / 1e3 for z in sizes]
        ax.plot(xs, ys, color=P.CATEGORICAL[i], linestyle=ls, linewidth=1.6, marker=P.MARKERS[i], markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
    ax.set_xticks(xs, sizes)
    ax.minorticks_off()
    ax.set_xlabel("Size of the bound (untouched) buffer")
    ax.set_ylabel("Round trip (µs)")
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    save(fig, "first_use.pdf")


def thermal(s: C.Session):
    ts, load = C.need(s.timeseries())
    by = defaultdict(list)
    for r in ts:
        by[r["workload"]].append((float(r["t_s"]), float(r["ns"]), r["thermal"]))

    def rolling(points, window=30.0):
        t = np.array([p[0] for p in points])
        v = np.array([p[1] for p in points])
        return t / 60, np.array([np.median(v[(t > ti - window) & (t <= ti)]) for ti in t])

    fig, ax = axes(1.75)
    states = sorted((float(r["t_s"]), r["thermal"]) for r in ts)
    start = next((t for t, st in states if st != "nominal"), None)
    if start is not None:
        ax.axvspan(start / 60, states[-1][0] / 60, color="#efeee9", zorder=0, linewidth=0)
        ax.text(start / 60 + 0.15, 0.97, "thermal state: fair", transform=ax.get_xaxis_transform(), fontsize=6.3,
                color=P.TEXT_2, va="top")
    for i, n in enumerate(["1024", "2048"]):
        tc, vc = rolling(by[f"cpu_{n}"])
        _, vg = rolling(by[f"gpu_{n}"])
        k = min(len(vc), len(vg))
        ax.plot(tc[:k], vc[:k] / vg[:k], color=P.CATEGORICAL[i], linewidth=1.5, label=f"SGEMM {n}²", zorder=3)
    ax.axhline(1, color=P.TEXT_2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Minutes under sustained load")
    ax.set_ylabel("CPU time / GPU time")
    ax.legend(loc="center right", frameon=False)
    save(fig, "thermal.pdf")
    return load


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="orig", help="orig, or a confirmation session index such as s0")
    a = ap.parse_args()
    sessions = {C.label(s): s for s in C.load_sessions(False)}
    if a.session not in sessions:
        print(f"unknown session {a.session}; have: {', '.join(sessions)}", file=sys.stderr)
        return 1
    s = sessions[a.session]
    FIG.mkdir(exist_ok=True)
    round_trip(s)
    first_use(s)
    load = thermal(s)
    used = {b: str(s.run_with(b).relative_to(P.REPO)) for b in ("sync_idle", "storage_gpu_first", "thermal_drift")}
    (FIG / "sources.json").write_text(json.dumps({"session": a.session, "thermal_load": load, "runs": used},
                                                 indent=2) + "\n")
    print(f"figures from session {a.session}: {', '.join(sorted(p.name for p in FIG.glob('*.pdf')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
