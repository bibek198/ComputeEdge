#!/usr/bin/env python3
"""Checks every quantitative claim in research/findings.md against every session.

Session "orig" is the original M4-M6 measurement (the committed top-level runs);
s0, s1, ... are confirmation sessions from scripts/run_session.sh
(benchmarks/results/sessions/<id>/). Each claim has an explicit pass criterion that
is fixed here *before* the confirmation data is looked at, and it is evaluated
independently per session. A claim is

  confirmed      passes in every session (needs >= 3 sessions to be paper-ready)
  inconsistent   passes in some sessions only
  refuted        fails in every session

Outputs benchmarks/results/reports/confirm-<stamp>/{claims.md, claims.csv, claims.json}
and paper/generated/numbers.tex (LaTeX macros: the cross-session median of each
headline number, so the paper never carries hand-copied values).

Usage: .venv/bin/python benchmarks/analysis/confirm_claims.py [--allow-dirty]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

import ce_plot as P

SESSIONS = P.RESULTS / "sessions"
PAPER_NUMBERS = P.REPO / "paper" / "generated" / "numbers.tex"


# ---------------------------------------------------------------------------
# Data access for one session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, name: str, runs: list[Path]):
        self.name = name
        self.runs = sorted(runs, reverse=True)  # newest first
        self._csv: dict[Path, list[dict]] = {}

    def _rows(self, path: Path) -> list[dict]:
        if path not in self._csv:
            self._csv[path] = P.read_csv(path) if path.exists() else []
        return self._csv[path]

    def run_with(self, benchmark: str) -> Path | None:
        for run in self.runs:
            if any(r["benchmark"] == benchmark for r in self._rows(run / "summary.csv")):
                return run
        return None

    @staticmethod
    def _match(row: dict, params: dict) -> bool:
        return all(row.get(k) == str(v) for k, v in params.items())

    def summary(self, benchmark: str, **params) -> dict | None:
        run = self.run_with(benchmark)
        if not run:
            return None
        for r in self._rows(run / "summary.csv"):
            if r["benchmark"] == benchmark and self._match(r, params):
                return r
        return None

    def median_ns(self, benchmark: str, **params) -> float | None:
        r = self.summary(benchmark, **params)
        return float(r["median_ns"]) if r else None

    def metric(self, benchmark: str, metric: str, **params) -> float | None:
        """Median of a per-call metric (metrics_summary.csv)."""
        run = self.run_with(benchmark)
        if not run:
            return None
        for r in self._rows(run / "metrics_summary.csv"):
            if r["benchmark"] == benchmark and r["metric"] == metric and self._match(r, params):
                return float(r["median"])
        return None

    def samples(self, benchmark: str, metric: str, **params) -> list[float]:
        run = self.run_with(benchmark)
        if not run:
            return []
        return [float(r["value"]) for r in self._rows(run / "metrics.csv")
                if r["benchmark"] == benchmark and r["metric"] == metric and self._match(r, params)]

    def timeseries(self) -> tuple[list[dict], str] | None:
        run = self.run_with("thermal_drift")
        if not run:
            return None
        load = self.summary("thermal_drift")["load"]
        return self._rows(run / "timeseries.csv"), load


class Missing(Exception):
    pass


def need(v):
    if v is None or (isinstance(v, list) and not v):
        raise Missing()
    return v


# ---------------------------------------------------------------------------
# Claims. Each returns (values: dict[str, float], passed: bool, note: str).
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    id: str
    finding: str
    text: str
    criterion: str
    fn: Callable[[Session], tuple[dict, bool, str]]
    headline: dict = field(default_factory=dict)  # value key -> LaTeX macro name
    # Sessions whose values feed the headline macros (e.g. only mixed-load thermal runs).
    headline_if: Callable[[dict], bool] | None = None


CLAIMS: list[Claim] = []


def claim(id, finding, text, criterion, headline=None, headline_if=None):
    def deco(fn):
        CLAIMS.append(Claim(id, finding, text, criterion, fn, headline or {}, headline_if))
        return fn
    return deco


@claim("C1", "1", "CPU (Accelerate) beats the GPU (MPS) below ~1024², the GPU wins above",
       "Like-for-like (both into pre-allocated outputs): CPU/GPU time < 1 at 128² and 512², > 1 at 2048² and 4096²",
       {"ratio_1024": "CrossRatioOneK", "ratio_128": "CrossRatioSmall", "ratio_2048": "CrossRatioTwoK"})
def c1(s: Session):
    v = {}
    for n in (128, 512, 1024, 2048, 4096):
        cpu = need(s.median_ns("matmul", impl="accelerate", mode="kernel", n=n))
        gpu = need(s.median_ns("matmul_metal", kernel="mps", storage="nocopy", n=n))
        v[f"ratio_{n}"] = cpu / gpu
    ok = v["ratio_128"] < 1 and v["ratio_512"] < 1 and v["ratio_2048"] > 1 and v["ratio_4096"] > 1
    return v, ok, "ratio = CPU time / GPU time"


@claim("C2", "2", "ComputeEdge's MPS path costs the same as MLX's GPU stream at large sizes",
       "|ComputeEdge MPS / MLX GPU - 1| < 10% at 4096²", {"ratio": "MlxParityRatio"})
def c2(s: Session):
    ce = need(s.median_ns("matmul_metal", kernel="mps", storage="nocopy", n=4096))
    mlx = need(s.median_ns("matmul_mlx", stream="gpu", n=4096))
    r = ce / mlx
    return {"ratio": r}, abs(r - 1) < 0.10, ""


@claim("C3", "4", "An empty GPU round trip costs ~85-120 µs; CPU-side encode + commit is a small share",
       "Empty round trip (1 CB, 1 encoder, 1 dispatch) median in [50, 250] µs and encode+commit < 10% of it",
       {"total_us": "RoundTripUs", "cpu_share_pct": "RoundTripCpuSharePct", "queue_us": "RoundTripQueueUs",
        "wake_us": "RoundTripWakeUs"})
def c3(s: Session):
    p = dict(command_buffers=1, encoders=1, dispatches=1, experiment="cmdbufs")
    total = need(s.metric("dispatch_empty", "total_ns", **p))
    enc = need(s.metric("dispatch_empty", "encode_ns", **p))
    com = need(s.metric("dispatch_empty", "commit_ns", **p))
    v = {"total_us": total / 1e3, "cpu_share_pct": 100 * (enc + com) / total,
         "queue_us": need(s.metric("dispatch_empty", "queue_ns", **p)) / 1e3,
         "wake_us": need(s.metric("dispatch_empty", "wake_ns", **p)) / 1e3}
    return v, 50 <= v["total_us"] <= 250 and v["cpu_share_pct"] < 10, ""


def slope(s: Session, exp: str, param: str, metric="total_ns") -> float:
    pts = [(int(r[param]), float(r["median"])) for r in P.read_csv(need(s.run_with("dispatch_empty")) / "metrics_summary.csv")
           if r["benchmark"] == "dispatch_empty" and r["experiment"] == exp and r["metric"] == metric]
    need(pts)
    xs, ys = zip(*pts)
    return float(np.polyfit(xs, ys, 1)[0])


@claim("C4", "5", "An extra command buffer costs far more than an extra dispatch",
       "Marginal total cost per command buffer > 20x the marginal cost per dispatch",
       {"per_cb_us": "PerCommandBufferUs", "per_dispatch_us": "PerDispatchUs", "ratio": "CbOverDispatch"})
def c4(s: Session):
    cb = slope(s, "cmdbufs", "command_buffers")
    d = slope(s, "dispatches", "dispatches")
    return {"per_cb_us": cb / 1e3, "per_dispatch_us": d / 1e3, "ratio": cb / d}, cb / d > 20, ""


@claim("C5", "6", "Even busy-polling, the CPU sees GPU completion no sooner than ~39 µs after GPUEndTime",
       "Median wake with spin-on-event, empty kernel >= 25 µs", {"wake_us": "SpinWakeFloorUs"})
def c5(s: Session):
    w = need(s.metric("sync_method", "wake_ns", sync="spin_event", work="empty")) / 1e3
    return {"wake_us": w}, w >= 25, ""


@claim("C6", "8", "An idle GPU makes the next round trip several times slower",
       "Round trip after 100 ms idle >= 3x back-to-back (waitUntilCompleted) and after 10 ms > after 0 ms",
       {"ratio_100ms": "IdleHundredMsRatio", "back_to_back_us": "IdleZeroUs", "after_100ms_us": "IdleHundredMsUs"})
def c6(s: Session):
    g = lambda gap: need(s.metric("sync_idle", "total_ns", sync="wait_until_completed", gap_ms=gap))
    z, ten, hundred = g("0"), g("10"), g("100")
    v = {"ratio_100ms": hundred / z, "back_to_back_us": z / 1e3, "after_100ms_us": hundred / 1e3,
         "ratio_10ms": ten / z}
    return v, v["ratio_100ms"] >= 3 and ten > z, ""


@claim("C7", "11", "A new MTLBuffer over already-resident memory pays a size-proportional cost on first GPU use",
       "NoCopy rewrap / reused round trip >= 3x at 64 MiB and >= 20x at 1 GiB",
       {"rewrap_64m_us": "RewrapSixtyFourMUs", "rewrap_1g_ms": "RewrapOneGMs", "reused_1g_us": "ReusedOneGUs"})
def c7(s: Session):
    f = lambda size, buf: need(s.metric("storage_gpu_first", "total_ns", kind="nocopy", size=size, buffer=buf))
    r64, u64, r1g, u1g = f("64M", "rewrap"), f("64M", "reused"), f("1G", "rewrap"), f("1G", "reused")
    v = {"rewrap_64m_us": r64 / 1e3, "rewrap_1g_ms": r1g / 1e6, "reused_1g_us": u1g / 1e3,
         "ratio_64m": r64 / u64, "ratio_1g": r1g / u1g}
    return v, v["ratio_64m"] >= 3 and v["ratio_1g"] >= 20, ""


@claim("C8", "10", "Storage mode does not change GPU streaming bandwidth",
       "At 64 MiB and 256 MiB, read and write GB/s of shared/private/nocopy all within ±10% of their mean",
       {"max_dev_pct": "StorageModeMaxDevPct"})
def c8(s: Session):
    worst = 0.0
    for size, b in (("64M", 64 << 20), ("256M", 256 << 20)):
        for stream in ("read", "write"):
            bw = [b / need(s.metric("storage_gpu_bw", "gpu_ns", kind=k, size=size, stream=stream))
                  for k in ("shared", "private", "nocopy")]
            m = statistics.mean(bw)
            worst = max(worst, max(abs(x / m - 1) for x in bw) * 100)
    return {"max_dev_pct": worst}, worst <= 10, ""


def slowdown(s: Session, phase: str, control: str, **p) -> float:
    x = need(s.samples("cache_interference", "cpu_ns", phase=phase, **p))
    c = need(s.samples("cache_interference", "cpu_ns", phase=control, **p))
    return (np.median(x) / np.median(c) - 1) * 100


@claim("C9", "16", "GPU traffic on other buffers does not evict the CPU's cached working set",
       "|slowdown| of the read pass after GPU writes/reads G (vs tiny-GPU control) < 10% for every W <= 12 MiB and G",
       {"max_abs_pct": "NoEvictMaxAbsPct"})
def c9(s: Session):
    worst = 0.0
    for ws in ("1M", "4M", "12M"):
        for g in ("16M", "64M", "256M"):
            for ph in ("gpu_write", "gpu_read"):
                worst = max(worst, abs(slowdown(s, ph, "gpu_tiny", **{"pass": "read", "ws": ws, "g": g})))
    return {"max_abs_pct": worst}, worst < 10, ""


@claim("C10", "17", "GPU-to-CPU handoff of cache-resident data costs ~45% of a cached pass",
       "Read-pass slowdown after the GPU wrote the CPU's own W (vs tiny control) >= 25% for W in 1-12 MiB",
       {"min_pct": "HandoffMinPct", "max_pct": "HandoffMaxPct"})
def c10(s: Session):
    v = [slowdown(s, "gpu_same", "gpu_tiny_same", **{"pass": "read", "ws": ws}) for ws in ("1M", "4M", "12M")]
    return {"min_pct": min(v), "max_pct": max(v)}, min(v) >= 25, ""


@claim("C11", "18", "Concurrent GPU streaming slows CPU work",
       "Median over valid cells (GPU overlapped >= 90% of samples) of concurrent-vs-short slowdown > +10%",
       {"median_pct": "ConcurrentMedianPct", "sgemm_median_pct": "ConcurrentSgemmPct"})
def c11(s: Session):
    vals, sgemm = [], []
    run = need(s.run_with("cache_interference"))
    cells = {(r["pass"], r["ws"], r["g"]) for r in P.read_csv(run / "summary.csv") if r["phase"] == "gpu_concurrent"}
    for pas, ws, g in cells:
        ov = s.samples("cache_interference", "overlap", **{"pass": pas, "ws": ws, "phase": "gpu_concurrent", "g": g})
        if not ov or np.mean(ov) < 0.9:
            continue
        x = need(s.samples("cache_interference", "cpu_ns", **{"pass": pas, "ws": ws, "phase": "gpu_concurrent", "g": g}))
        c = need(s.samples("cache_interference", "cpu_ns", **{"pass": pas, "ws": ws, "phase": "idle_short"}))
        sd = (np.median(x) / np.median(c) - 1) * 100
        (sgemm if pas == "sgemm" else vals).append(sd)
    need(vals)
    med = float(np.median(vals + sgemm))
    return {"median_pct": med, "sgemm_median_pct": float(np.median(sgemm)) if sgemm else float("nan"),
            "valid_cells": len(vals) + len(sgemm)}, med > 10, ""


def thermal_minutes(s: Session):
    ts, load = need(s.timeseries())
    per = defaultdict(lambda: defaultdict(list))
    states = defaultdict(list)
    for r in ts:
        m = int(float(r["t_s"]) // 60)
        per[r["workload"]][m].append(float(r["ns"]))
        if r["workload"] == "cpu_2048":
            states[m].append(r["thermal"])
    med = {w: {m: float(np.median(v)) for m, v in d.items()} for w, d in per.items()}
    return med, states, load


@claim("C12", "20", "Sustained load costs throughput on the heated unit(s)",
       "Throughput in the last 5 minutes vs minute 0 drops >= 10% for the unit(s) the heater loads (mixed: both)",
       {"cpu_loss_pct": "ThermalCpuLossPct", "gpu_loss_pct": "ThermalGpuLossPct"},
       headline_if=lambda v: v.get("load") == "mixed")
def c12(s: Session):
    med, _, load = thermal_minutes(s)
    last = max(med["cpu_2048"])
    tail = range(max(0, last - 4), last + 1)
    loss = lambda w: (1 - med[w][0] / float(np.median([med[w][m] for m in tail]))) * 100  # time up = throughput down
    v = {"cpu_loss_pct": loss("cpu_2048"), "gpu_loss_pct": loss("gpu_2048"), "load": load}
    heated = {"mixed": ["cpu_loss_pct", "gpu_loss_pct"], "cpu": ["cpu_loss_pct"], "gpu": ["gpu_loss_pct"],
              "none": []}[load]
    return v, all(v[k] >= 10 for k in heated), f"load={load}"


@claim("C13", "21", "The CPU/GPU crossover drifts under sustained load, invisibly to macOS thermal state",
       "CPU/GPU time ratio at 1024² changes by >= 10% between minute 0 and the last 5 minutes",
       {"ratio_start": "DriftRatioStart", "ratio_end": "DriftRatioEnd"},
       headline_if=lambda v: v.get("load") == "mixed")
def c13(s: Session):
    med, states, load = thermal_minutes(s)
    last = max(med["cpu_1024"])
    tail = range(max(0, last - 4), last + 1)
    r0 = med["cpu_1024"][0] / med["gpu_1024"][0]
    r1 = float(np.median([med["cpu_1024"][m] / med["gpu_1024"][m] for m in tail]))
    distinct = {max(set(v), key=v.count) for v in states.values()}
    v = {"ratio_start": r0, "ratio_end": r1, "change_pct": (r1 / r0 - 1) * 100, "load": load,
         "thermal_states_seen": len(distinct)}
    return v, abs(r1 / r0 - 1) >= 0.10, f"load={load}"


@claim("C14", "23", "A just-created Shared buffer is slow to write for its first two passes, then matches host memory",
       "Shared write 2 < 0.8x write 4, and Shared write 4 within 10% of posix_memalign write 4",
       {"w1": "FreshWriteOneGBs", "w2": "FreshWriteTwoGBs", "w4": "FreshWriteFourGBs"})
def c14(s: Session):
    b = 64 << 20
    g = lambda src, i: b / need(s.metric("followup_cpu_write_fresh", f"write{i}_ns", source=src))
    v = {"w1": g("shared", 1), "w2": g("shared", 2), "w4": g("shared", 4), "host_w4": g("memalign", 4)}
    return v, v["w2"] < 0.8 * v["w4"] and abs(v["w4"] / v["host_w4"] - 1) < 0.10, "GB/s"


@claim("C15", "24", "GPU write bandwidth falls with the size of one buffer, not with total bytes",
       "1 GiB as one buffer < 0.6x the bandwidth of the same bytes as 256 MiB buffers",
       {"one_gbs": "WriteOneBufferGBs", "chunks_gbs": "WriteChunksGBs"})
def c15(s: Session):
    b = 1 << 30
    one = b / need(s.metric("followup_gpu_write", "gpu_ns", layout="one_buffer", size="1G"))
    ch = b / need(s.metric("followup_gpu_write", "gpu_ns", layout="256M_chunks", size="1G"))
    return {"one_gbs": one, "chunks_gbs": ch}, one < 0.6 * ch, "GB/s"


@claim("C16", "22", "Wake latency after long GPU work does not depend on whether it wrote or read",
       "|wake(write) / wake(read) - 1| < 20% at 64 MiB and 256 MiB", {"max_diff_pct": "WakeRwMaxDiffPct"})
def c16(s: Session):
    worst = 0.0
    for size in ("64M", "256M"):
        r = need(s.metric("followup_wake", "wake_ns", stream="read", size=size))
        w = need(s.metric("followup_wake", "wake_ns", stream="write", size=size))
        worst = max(worst, abs(w / r - 1) * 100)
    return {"max_diff_pct": worst}, worst < 20, ""


@claim("C17", "16 (scope)", "GPU traffic displaces SLC-resident CPU data (reproduces arXiv 2606.27098 on M4, with clock controls)",
       "Registered 2026-11-14 before any data: SLC-probe pass after the GPU writes 256 MiB is >= 15% slower than "
       "after the tiny-GPU control", {"slowdown_256m_pct": "SlcSlowdownPct"})
def c17(s: Session):
    v = {f"slowdown_{g.lower()}_pct": slowdown(s, "gpu_write", "gpu_tiny", **{"pass": "slc", "ws": "16M@8K", "g": g})
         for g in ("16M", "64M", "256M")}
    v["tiny_vs_spin_256m_pct"] = slowdown(s, "gpu_tiny", "idle_spin", **{"pass": "slc", "ws": "16M@8K", "g": "256M"})
    return v, v["slowdown_256m_pct"] >= 15, ""


# ---------------------------------------------------------------------------
# Sessions, evaluation, output
# ---------------------------------------------------------------------------

EXCLUSIONS = P.RESULTS / "exclusions.json"


def excluded_runs() -> set[Path]:
    """Runs listed in benchmarks/results/exclusions.json (kept on disk, left out of analysis)."""
    if not EXCLUSIONS.exists():
        return set()
    return {(P.RESULTS / e["run"]).resolve() for e in json.loads(EXCLUSIONS.read_text())["excluded_runs"]}


def load_sessions(allow_dirty: bool) -> list[Session]:
    excluded = excluded_runs()
    clean = lambda p: (allow_dirty or not p.name.endswith("-dirty")) and p.resolve() not in excluded
    original = [r for r in P.run_dirs() if clean(r)]
    sessions = [Session("orig", original)]
    if SESSIONS.exists():
        for d in sorted(p for p in SESSIONS.iterdir() if p.is_dir()):
            runs = [r for r in d.iterdir() if (r / "summary.csv").exists() and clean(r)]
            if runs:
                sessions.append(Session(d.name, runs))
    return sessions


def label(s: Session) -> str:
    """'orig' or the sN index from a session directory name (<stamp>_sN_<sha>)."""
    m = [part for part in s.name.split("_") if part.startswith("s") and part[1:].isdigit()]
    return m[0] if m else s.name


def fmt(x) -> str:
    if isinstance(x, str):
        return x
    if x != x:  # NaN
        return "—"
    return f"{x:.3g}" if abs(x) < 100 else f"{x:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    sessions = load_sessions(a.allow_dirty)

    results = {}  # claim id -> session name -> (values, passed, note) or None
    for c in CLAIMS:
        results[c.id] = {}
        for s in sessions:
            try:
                results[c.id][s.name] = c.fn(s)
            except Missing:
                results[c.id][s.name] = None

    out = P.RESULTS / "reports" / f"confirm-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    md = ["# Claim confirmation across sessions", "",
          f"{len(sessions)} session(s): " + ", ".join(f"`{label(s)}` = `{s.name}`" for s in sessions) + ".",
          "Criteria are fixed in `benchmarks/analysis/confirm_claims.py`. A claim is paper-ready when it is "
          "**confirmed in >= 3 sessions**. Runs in `benchmarks/results/exclusions.json` are left out "
          f"({len(excluded_runs())} excluded).", "",
          "| claim | finding | statement | criterion | " + " | ".join(label(x) for x in sessions)
          + " | verdict |", "|---|---:|---|---|" + "---|" * len(sessions) + "---|"]
    rows_csv, verdicts, json_out, macros_verdict = [], {}, {}, []
    for c in CLAIMS:
        cells, outcomes = [], []
        for s in sessions:
            r = results[c.id][s.name]
            if r is None:
                cells.append("n/a")
                continue
            v, ok, note = r
            outcomes.append(ok)
            key = next(iter(c.headline), next(iter(v)))
            cells.append(("✅ " if ok else "❌ ") + fmt(v[key]))
            for k, val in v.items():
                rows_csv.append([c.id, s.name, k, val, ok])
        n_ok = sum(outcomes)
        if not outcomes:
            verdict = "no data"
        elif n_ok == len(outcomes):
            verdict = f"confirmed ({n_ok}/{len(outcomes)})"
        elif n_ok == 0:
            verdict = f"refuted (0/{len(outcomes)})"
        else:
            verdict = f"inconsistent ({n_ok}/{len(outcomes)})"
        verdicts[c.id] = verdict
        # \verdictof{C1} in the paper (digits are not allowed in LaTeX command names)
        macros_verdict.append(f"\\expandafter\\newcommand\\csname verdict{c.id}\\endcsname{{{verdict}}}")
        md.append(f"| {c.id} | {c.finding} | {c.text} | {c.criterion} | " + " | ".join(cells) + f" | **{verdict}** |")
        json_out[c.id] = {"finding": c.finding, "text": c.text, "criterion": c.criterion, "verdict": verdict,
                          "sessions": {s.name: (None if results[c.id][s.name] is None else
                                                {"values": results[c.id][s.name][0],
                                                 "passed": results[c.id][s.name][1]}) for s in sessions}}

    # Cross-session spread of headline values.
    md += ["", "## Headline values across sessions", "",
           "| claim | value | " + " | ".join(label(x) for x in sessions) + " | median | spread (max/min) |",
           "|---|---|" + "---:|" * len(sessions) + "---:|---:|"]
    macros = []
    for c in CLAIMS:
        for key, macro in c.headline.items():
            vals = []
            cells = []
            for s in sessions:
                r = results[c.id][s.name]
                x = r[0].get(key) if r else None
                cells.append(fmt(x) if isinstance(x, (int, float)) else "—")
                if c.headline_if and r and not c.headline_if(r[0]):
                    continue
                if isinstance(x, (int, float)) and x == x:
                    vals.append(float(x))
            if not vals:
                continue
            med = float(np.median(vals))
            spread = (max(vals) / min(vals)) if min(vals) > 0 else float("nan")
            md.append(f"| {c.id} | {key} | " + " | ".join(cells) + f" | {fmt(med)} | {fmt(spread)} |")
            digits = 0 if abs(med) >= 100 else (1 if abs(med) >= 10 else 2)
            macros.append(f"\\newcommand{{\\{macro}}}{{{med:,.{digits}f}}}  % {c.id} {key}, median of {len(vals)} "
                          f"session(s)")
            for suffix, x in (("Min", min(vals)), ("Max", max(vals))):
                d = 0 if abs(x) >= 100 else (1 if abs(x) >= 10 else 2)
                macros.append(f"\\newcommand{{\\{macro}{suffix}}}{{{x:,.{d}f}}}")
    (out / "claims.md").write_text("\n".join(md) + "\n")
    with open(out / "claims.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["claim", "session", "value", "x", "passed"])
        w.writerows(rows_csv)
    (out / "claims.json").write_text(json.dumps(json_out, indent=2, default=str) + "\n")

    PAPER_NUMBERS.parent.mkdir(parents=True, exist_ok=True)
    PAPER_NUMBERS.write_text(
        "% Generated by benchmarks/analysis/confirm_claims.py -- do not edit.\n"
        f"% Source: {out.relative_to(P.REPO)}; {len(sessions)} session(s).\n"
        f"\\newcommand{{\\NumSessions}}{{{len(sessions)}}}\n"
        "\\newcommand{\\verdictof}[1]{\\csname verdict#1\\endcsname}\n" + "\n".join(macros + macros_verdict) + "\n")

    for c in CLAIMS:
        print(f"{c.id:4s} {verdicts[c.id]:22s} {c.text}")
    print(f"\nWritten to {out.relative_to(P.REPO)} and {PAPER_NUMBERS.relative_to(P.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
