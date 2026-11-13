"""Shared helpers for the analysis scripts: result loading and chart styling.

Palette: the reference dataviz palette (light mode), validated with
validate_palette.js; categorical slots are assigned in fixed order. Contrast is
WARN on slots 3-5, so every chart carries a legend plus distinct markers and
every figure has a table alongside it in the report.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
# The original M4-M6 runs (the "orig" session). Report helpers that pick "the
# newest run of a benchmark" search only here.
RESULTS = DATA / "sessions" / "orig"
DERIVED = DATA / "derived"

SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "D", "^", "v"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_dirs() -> list[Path]:
    """Result directories, newest first (comparisons excluded)."""
    return sorted((p for p in RESULTS.iterdir() if (p / "summary.csv").exists() and p.name != "comparisons"),
                  reverse=True)


def newest_run_with(benchmark: str, allow_dirty: bool = False) -> Path | None:
    for run in run_dirs():
        if not allow_dirty and run.name.endswith("-dirty"):
            continue
        if any(r["benchmark"] == benchmark for r in read_csv(run / "summary.csv")):
            return run
    return None


def metric_table(run: Path, benchmark: str) -> list[dict]:
    """metrics_summary.csv rows for one benchmark, values converted to float."""
    out = []
    for r in read_csv(run / "metrics_summary.csv"):
        if r["benchmark"] != benchmark:
            continue
        for k in ("median", "p25", "p75", "iqr", "ci95_lo", "ci95_hi", "mean", "min", "max"):
            r[k] = float(r[k])
        out.append(r)
    return out


def pick(rows: list[dict], metric: str, **params) -> dict | None:
    for r in rows:
        if r["metric"] == metric and all(r.get(k) == str(v) for k, v in params.items()):
            return r
    return None


def new_axes(figsize=(8.0, 4.8), ncols=1, sharey=False):
    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=200, sharey=sharey, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    axes = list(axes[0])
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=TEXT_2, labelsize=9)
    return fig, axes


def labels(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)


def title(fig_or_ax, text: str, subtitle: str | None = None) -> None:
    ax = fig_or_ax
    ax.set_title(text, loc="left", color=TEXT, fontsize=12, pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=TEXT_2, fontsize=8.5)


def series(ax, xs, ys, i: int, label: str, lo=None, hi=None, linestyle="-") -> None:
    colour = CATEGORICAL[i % len(CATEGORICAL)]
    ax.plot(xs, ys, color=colour, linewidth=2, linestyle=linestyle, marker=MARKERS[i % len(MARKERS)], markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    if lo is not None and hi is not None:
        ax.vlines(xs, lo, hi, color=colour, linewidth=1, zorder=2)


def fit_log_limits(ax, xs, ys, pad=1.6) -> None:
    """Tight limits for log-log axes (autoscaling leaves large empty margins)."""
    ys = [y for y in ys if y > 0]
    ax.set_xlim(min(xs) / pad, max(xs) * pad)
    ax.set_ylim(min(ys) / pad, max(ys) * pad)


def legend(ax, loc="upper left") -> None:
    ax.legend(loc=loc, frameon=False, fontsize=8.5, labelcolor=TEXT, handlelength=3)


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def metric_samples(run: Path, benchmark: str, metric: str, **params) -> list[float]:
    """Raw per-sample values of one metric for the result matching `params` (metrics.csv)."""
    out = []
    for r in read_csv(run / "metrics.csv"):
        if r["benchmark"] == benchmark and r["metric"] == metric and all(r.get(k) == str(v) for k, v in params.items()):
            out.append(float(r["value"]))
    return out


def slowdown_ci(x: list[float], base: list[float], resamples: int = 2000, seed: int = 0xCE) -> tuple[float, float, float]:
    """median(x)/median(base) - 1 as a percentage, with a percentile-bootstrap 95% CI
    (independent resampling of both samples)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    xa, ba = np.asarray(x), np.asarray(base)
    point = (np.median(xa) / np.median(ba) - 1) * 100
    boots = [(np.median(rng.choice(xa, xa.size)) / np.median(rng.choice(ba, ba.size)) - 1) * 100
             for _ in range(resamples)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)
