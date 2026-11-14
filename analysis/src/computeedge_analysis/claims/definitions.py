"""The claims C1-C17 (research/findings.md).

Criteria are pre-registered: do not change one after its confirmation data exists;
add a new claim instead. Claim bodies are moved verbatim from the original
benchmarks/analysis/confirm_claims.py (pre-refactor tag), adapted only to the typed
records of computeedge_analysis.io.
"""

# ruff: noqa: E731, E501 - claim bodies are kept verbatim from their registration

from __future__ import annotations

import statistics
from collections import defaultdict

import numpy as np

from ..io import Session
from .registry import claim, need


@claim("C1", "1", "CPU (Accelerate) beats the GPU (MPS) below ~1024², the GPU wins above",
       "Like-for-like (both into pre-allocated outputs): CPU/GPU time < 1 at 128² and 512², > 1 at 2048² and 4096²",
       {"ratio_1024": "CrossRatioOneK", "ratio_128": "CrossRatioSmall", "ratio_2048": "CrossRatioTwoK"})
def c1(s: Session):
    """Claim C1 (see its criterion)."""
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
    """Claim C2 (see its criterion)."""
    ce = need(s.median_ns("matmul_metal", kernel="mps", storage="nocopy", n=4096))
    mlx = need(s.median_ns("matmul_mlx", stream="gpu", n=4096))
    r = ce / mlx
    return {"ratio": r}, abs(r - 1) < 0.10, ""


@claim("C3", "4", "An empty GPU round trip costs ~85-120 µs; CPU-side encode + commit is a small share",
       "Empty round trip (1 CB, 1 encoder, 1 dispatch) median in [50, 250] µs and encode+commit < 10% of it",
       {"total_us": "RoundTripUs", "cpu_share_pct": "RoundTripCpuSharePct", "queue_us": "RoundTripQueueUs",
        "wake_us": "RoundTripWakeUs"})
def c3(s: Session):
    """Claim C3 (see its criterion)."""
    p = dict(command_buffers=1, encoders=1, dispatches=1, experiment="cmdbufs")
    total = need(s.metric("dispatch_empty", "total_ns", **p))
    enc = need(s.metric("dispatch_empty", "encode_ns", **p))
    com = need(s.metric("dispatch_empty", "commit_ns", **p))
    v = {"total_us": total / 1e3, "cpu_share_pct": 100 * (enc + com) / total,
         "queue_us": need(s.metric("dispatch_empty", "queue_ns", **p)) / 1e3,
         "wake_us": need(s.metric("dispatch_empty", "wake_ns", **p)) / 1e3}
    return v, 50 <= v["total_us"] <= 250 and v["cpu_share_pct"] < 10, ""


def slope(s: Session, exp: str, param: str, metric="total_ns") -> float:
    """Least-squares marginal cost per unit of `param` in one dispatch_empty sweep."""
    need(s.run_with("dispatch_empty"))
    pts = [(int(r.params[param]), r.median) for r in s.metric_rows("dispatch_empty", metric)
           if r.params.get("experiment") == exp]
    need(pts)
    xs, ys = zip(*pts, strict=True)
    return float(np.polyfit(xs, ys, 1)[0])


@claim("C4", "5", "An extra command buffer costs far more than an extra dispatch",
       "Marginal total cost per command buffer > 20x the marginal cost per dispatch",
       {"per_cb_us": "PerCommandBufferUs", "per_dispatch_us": "PerDispatchUs", "ratio": "CbOverDispatch"})
def c4(s: Session):
    """Claim C4 (see its criterion)."""
    cb = slope(s, "cmdbufs", "command_buffers")
    d = slope(s, "dispatches", "dispatches")
    return {"per_cb_us": cb / 1e3, "per_dispatch_us": d / 1e3, "ratio": cb / d}, cb / d > 20, ""


@claim("C5", "6", "Even busy-polling, the CPU sees GPU completion no sooner than ~39 µs after GPUEndTime",
       "Median wake with spin-on-event, empty kernel >= 25 µs", {"wake_us": "SpinWakeFloorUs"})
def c5(s: Session):
    """Claim C5 (see its criterion)."""
    w = need(s.metric("sync_method", "wake_ns", sync="spin_event", work="empty")) / 1e3
    return {"wake_us": w}, w >= 25, ""


@claim("C6", "8", "An idle GPU makes the next round trip several times slower",
       "Round trip after 100 ms idle >= 3x back-to-back (waitUntilCompleted) and after 10 ms > after 0 ms",
       {"ratio_100ms": "IdleHundredMsRatio", "back_to_back_us": "IdleZeroUs", "after_100ms_us": "IdleHundredMsUs"})
def c6(s: Session):
    """Claim C6 (see its criterion)."""
    g = lambda gap: need(s.metric("sync_idle", "total_ns", sync="wait_until_completed", gap_ms=gap))
    z, ten, hundred = g("0"), g("10"), g("100")
    v = {"ratio_100ms": hundred / z, "back_to_back_us": z / 1e3, "after_100ms_us": hundred / 1e3,
         "ratio_10ms": ten / z}
    return v, v["ratio_100ms"] >= 3 and ten > z, ""


@claim("C7", "11", "A new MTLBuffer over already-resident memory pays a size-proportional cost on first GPU use",
       "NoCopy rewrap / reused round trip >= 3x at 64 MiB and >= 20x at 1 GiB",
       {"rewrap_64m_us": "RewrapSixtyFourMUs", "rewrap_1g_ms": "RewrapOneGMs", "reused_1g_us": "ReusedOneGUs"})
def c7(s: Session):
    """Claim C7 (see its criterion)."""
    f = lambda size, buf: need(s.metric("storage_gpu_first", "total_ns", kind="nocopy", size=size, buffer=buf))
    r64, u64, r1g, u1g = f("64M", "rewrap"), f("64M", "reused"), f("1G", "rewrap"), f("1G", "reused")
    v = {"rewrap_64m_us": r64 / 1e3, "rewrap_1g_ms": r1g / 1e6, "reused_1g_us": u1g / 1e3,
         "ratio_64m": r64 / u64, "ratio_1g": r1g / u1g}
    return v, v["ratio_64m"] >= 3 and v["ratio_1g"] >= 20, ""


@claim("C8", "10", "Storage mode does not change GPU streaming bandwidth",
       "At 64 MiB and 256 MiB, read and write GB/s of shared/private/nocopy all within ±10% of their mean",
       {"max_dev_pct": "StorageModeMaxDevPct"})
def c8(s: Session):
    """Claim C8 (see its criterion)."""
    worst = 0.0
    for size, b in (("64M", 64 << 20), ("256M", 256 << 20)):
        for stream in ("read", "write"):
            bw = [b / need(s.metric("storage_gpu_bw", "gpu_ns", kind=k, size=size, stream=stream))
                  for k in ("shared", "private", "nocopy")]
            m = statistics.mean(bw)
            worst = max(worst, max(abs(x / m - 1) for x in bw) * 100)
    return {"max_dev_pct": worst}, worst <= 10, ""


def slowdown(s: Session, phase: str, control: str, **p) -> float:
    """Median CPU-pass slowdown (%) of `phase` vs `control` in cache_interference."""
    x = need(s.samples("cache_interference", "cpu_ns", phase=phase, **p))
    c = need(s.samples("cache_interference", "cpu_ns", phase=control, **p))
    return (np.median(x) / np.median(c) - 1) * 100


@claim("C9", "16", "GPU traffic on other buffers does not evict the CPU's cached working set",
       "|slowdown| of the read pass after GPU writes/reads G (vs tiny-GPU control) < 10% for every W <= 12 MiB and G",
       {"max_abs_pct": "NoEvictMaxAbsPct"})
def c9(s: Session):
    """Claim C9 (see its criterion)."""
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
    """Claim C10 (see its criterion)."""
    v = [slowdown(s, "gpu_same", "gpu_tiny_same", **{"pass": "read", "ws": ws}) for ws in ("1M", "4M", "12M")]
    return {"min_pct": min(v), "max_pct": max(v)}, min(v) >= 25, ""


@claim("C11", "18", "Concurrent GPU streaming slows CPU work",
       "Median over valid cells (GPU overlapped >= 90% of samples) of concurrent-vs-short slowdown > +10%",
       {"median_pct": "ConcurrentMedianPct", "sgemm_median_pct": "ConcurrentSgemmPct"})
def c11(s: Session):
    """Claim C11 (see its criterion)."""
    vals, sgemm = [], []
    need(s.run_with("cache_interference"))
    cells = {(r.params["pass"], r.params["ws"], r.params["g"]) for r in s.summaries("cache_interference")
             if r.params.get("phase") == "gpu_concurrent"}
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
    """Per-workload, per-minute medians, per-minute thermal states and the heater load."""
    ts, load = need(s.timeseries())
    per = defaultdict(lambda: defaultdict(list))
    states = defaultdict(list)
    for r in ts:
        m = int(r.t_s // 60)
        per[r.workload][m].append(r.ns)
        if r.workload == "cpu_2048":
            states[m].append(r.thermal)
    med = {w: {m: float(np.median(v)) for m, v in d.items()} for w, d in per.items()}
    return med, states, load


@claim("C12", "20", "Sustained load costs throughput on the heated unit(s)",
       "Throughput in the last 5 minutes vs minute 0 drops >= 10% for the unit(s) the heater loads (mixed: both)",
       {"cpu_loss_pct": "ThermalCpuLossPct", "gpu_loss_pct": "ThermalGpuLossPct"},
       headline_if=lambda v: v.get("load") == "mixed")
def c12(s: Session):
    """Claim C12 (see its criterion)."""
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
    """Claim C13 (see its criterion)."""
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
    """Claim C14 (see its criterion)."""
    b = 64 << 20
    g = lambda src, i: b / need(s.metric("followup_cpu_write_fresh", f"write{i}_ns", source=src))
    v = {"w1": g("shared", 1), "w2": g("shared", 2), "w4": g("shared", 4), "host_w4": g("memalign", 4)}
    return v, v["w2"] < 0.8 * v["w4"] and abs(v["w4"] / v["host_w4"] - 1) < 0.10, "GB/s"


@claim("C15", "24", "GPU write bandwidth falls with the size of one buffer, not with total bytes",
       "1 GiB as one buffer < 0.6x the bandwidth of the same bytes as 256 MiB buffers",
       {"one_gbs": "WriteOneBufferGBs", "chunks_gbs": "WriteChunksGBs"})
def c15(s: Session):
    """Claim C15 (see its criterion)."""
    b = 1 << 30
    one = b / need(s.metric("followup_gpu_write", "gpu_ns", layout="one_buffer", size="1G"))
    ch = b / need(s.metric("followup_gpu_write", "gpu_ns", layout="256M_chunks", size="1G"))
    return {"one_gbs": one, "chunks_gbs": ch}, one < 0.6 * ch, "GB/s"


@claim("C16", "22", "Wake latency after long GPU work does not depend on whether it wrote or read",
       "|wake(write) / wake(read) - 1| < 20% at 64 MiB and 256 MiB", {"max_diff_pct": "WakeRwMaxDiffPct"})
def c16(s: Session):
    """Claim C16 (see its criterion)."""
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
    """Claim C17 (see its criterion)."""
    v = {f"slowdown_{g.lower()}_pct": slowdown(s, "gpu_write", "gpu_tiny", **{"pass": "slc", "ws": "16M@8K", "g": g})
         for g in ("16M", "64M", "256M")}
    v["tiny_vs_spin_256m_pct"] = slowdown(s, "gpu_tiny", "idle_spin", **{"pass": "slc", "ws": "16M@8K", "g": "256M"})
    return v, v["slowdown_256m_pct"] >= 15, ""
