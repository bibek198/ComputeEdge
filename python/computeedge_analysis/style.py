"""Chart styling shared by reports and figures.

Palette: the reference categorical palette, light mode, assigned in fixed order.
Contrast is marginal on slots 3-5, so every chart carries a legend and distinct
markers, and every figure has a table alongside it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "D", "^", "v"]
# Background bands for macOS thermal states (neutral greys; status colours are reserved).
STATE_SHADE = {"fair": "#efeee9", "serious": "#dcdad2", "critical": "#c3c2b7"}


def style_axes(ax, labelsize: float = 9) -> None:
    """Applies the house style to one axes: surface, grid behind data, recessive spines and ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_2, labelsize=labelsize)


def new_axes(figsize=(8.0, 4.8), ncols=1, sharey=False):
    """A styled figure with `ncols` axes in one row; returns (fig, [axes])."""
    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=200, sharey=sharey, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    axes = list(axes[0])
    for ax in axes:
        style_axes(ax)
    return fig, axes


def labels(ax, xlabel: str, ylabel: str) -> None:
    """Axis labels in secondary text colour."""
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)


def title(ax, text: str, subtitle: str | None = None) -> None:
    """Left-aligned title with an optional one-line subtitle."""
    ax.set_title(text, loc="left", color=TEXT, fontsize=12, pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=TEXT_2, fontsize=8.5)


def series(ax, xs, ys, i: int, label: str, lo=None, hi=None, linestyle="-") -> None:
    """Line series `i` (fixed palette order) with markers and optional CI whiskers."""
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
    """Frameless legend in primary text colour."""
    ax.legend(loc=loc, frameon=False, fontsize=8.5, labelcolor=TEXT, handlelength=3)


def save(fig, path: Path) -> None:
    """Tight layout, write, close."""
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
