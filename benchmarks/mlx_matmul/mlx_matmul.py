#!/usr/bin/env python3
"""MLX matmul baseline, measured with the same protocol and output format as the
C++ harness (src/benchmark/benchmark.cpp):

  * process-level GPU pre-warm, then per-benchmark warm-up of >= N calls AND >= T ms
  * calls per sample auto-calibrated so one sample lasts >= --min-sample-us
  * median, IQR and percentile-bootstrap 95% CI of the median (2000 resamples)
  * env.json from build/release/benchmarks/ce_env (+ MLX/Python versions),
    samples.csv tagged with thermal state, summary.csv with identical columns

One measured call = mx.eval(mx.matmul(a, b, stream=s)): graph construction,
output allocation from MLX's buffer cache, dispatch and synchronisation, i.e. what
a user of MLX's Python API pays per matmul. Inputs are materialised beforehand.
Note this includes Python interpreter overhead (a few µs), which the C++
benchmarks do not have.

Usage:
  .venv/bin/python benchmarks/mlx_matmul/mlx_matmul.py [--sizes 128,512,1024,2048,4096]
      [--streams gpu,cpu] [--warmup 10] [--min-warmup-ms 200] [--prewarm-ms 1000]
      [--reps 30] [--min-sample-us 50] [--out data/scratch] [--cooldown]
      [--no-write] [--ce-env build/release/benchmarks/ce_env]
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Thermal state via the Objective-C runtime (NSProcessInfo.thermalState), cheap
# enough to read after every sample like the C++ harness does.
# ---------------------------------------------------------------------------
_objc = ctypes.CDLL(ctypes.util.find_library("objc"))
ctypes.CDLL(ctypes.util.find_library("Foundation"))
_objc.objc_getClass.restype = ctypes.c_void_p
_objc.objc_getClass.argtypes = [ctypes.c_char_p]
_objc.sel_registerName.restype = ctypes.c_void_p
_objc.sel_registerName.argtypes = [ctypes.c_char_p]
_msg_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", _objc))
_msg_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", _objc))
_PROCESS_INFO = _msg_ptr(_objc.objc_getClass(b"NSProcessInfo"), _objc.sel_registerName(b"processInfo"))
_SEL_THERMAL = _objc.sel_registerName(b"thermalState")
_THERMAL_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def thermal_state() -> str:
    return _THERMAL_NAMES.get(_msg_long(_PROCESS_INFO, _SEL_THERMAL), "unknown")


def wait_for_nominal(timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    while thermal_state() != "nominal" and time.monotonic() < deadline:
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Statistics (same definitions as src/benchmark/stats.cpp)
# ---------------------------------------------------------------------------
def summarize(values: list[float], resamples: int = 2000, seed: int = 0xCE) -> dict:
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    s = {
        "n_samples": n,
        "median": float(np.quantile(v, 0.5)),
        "p25": float(np.quantile(v, 0.25)),
        "p75": float(np.quantile(v, 0.75)),
        "mean": float(v.mean()),
        "stddev": float(v.std(ddof=1)) if n > 1 else 0.0,
        "min": float(v[0]),
        "max": float(v[-1]),
    }
    s["iqr"] = s["p75"] - s["p25"]
    if n > 1 and resamples > 0:
        rng = np.random.default_rng(seed)
        medians = np.median(rng.choice(v, size=(resamples, n), replace=True), axis=1)
        s["ci95_lo"] = float(np.quantile(medians, 0.025))
        s["ci95_hi"] = float(np.quantile(medians, 0.975))
    else:
        s["ci95_lo"] = s["ci95_hi"] = s["median"]
    return s


# ---------------------------------------------------------------------------
# Runner (mirrors bench::run)
# ---------------------------------------------------------------------------
def run(fn, warmup: int, min_warmup_ns: int, reps: int, min_sample_ns: int) -> dict:
    thermal_before = thermal_state()
    start = time.perf_counter_ns()
    i = 0
    while i < warmup or time.perf_counter_ns() - start < min_warmup_ns:
        fn()
        i += 1

    inner = 1
    while inner < (1 << 20):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        if time.perf_counter_ns() - t0 >= min_sample_ns:
            break
        inner *= 2

    samples, thermal = [], []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter_ns() - t0) / inner)
        thermal.append(thermal_state())
    return {
        "samples": samples,
        "thermal": thermal,
        "inner": inner,
        "thermal_before": thermal_before,
        "thermal_after": thermal_state(),
    }


def record_environment(ce_env: Path) -> dict:
    if not ce_env.exists():
        sys.exit(f"error: {ce_env} not found; build the release preset first")
    env = json.loads(subprocess.run([str(ce_env)], check=True, capture_output=True, text=True).stdout)
    # ce_env reports the C++ build; these fields describe this process instead.
    env["framework"] = "mlx"
    env["mlx_version"] = mx.__version__
    env["numpy_version"] = np.__version__
    env["python_version"] = platform.python_version()
    env["mlx_default_device"] = str(mx.default_device())
    env["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    env["thermal_state"] = thermal_state()
    return env


def write_results(root: Path, env: dict, results: list[dict]) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = root / f"{stamp}_{env.get('git_sha') or 'unknown'}"
    out, i = base, 1
    while out.exists():
        out, i = Path(f"{base}_{i}"), i + 1
    out.mkdir(parents=True)

    (out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    keys = sorted({k for r in results for k in r["params"]})
    with open(out / "samples.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["benchmark", *keys, "rep", "inner_iterations", "ns_per_call", "thermal"])
        for r in results:
            for rep, (ns, th) in enumerate(zip(r["samples"], r["thermal"])):
                w.writerow([r["name"], *[r["params"].get(k, "") for k in keys], rep, r["inner"], ns, th])
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["benchmark", *keys, "n_samples", "inner_iterations", "median_ns", "p25_ns", "p75_ns", "iqr_ns",
                    "ci95_lo_ns", "ci95_hi_ns", "mean_ns", "stddev_ns", "min_ns", "max_ns", "gflops_median",
                    "thermal_before", "thermal_after"])
        for r in results:
            s = r["summary"]
            gflops = r["flops"] / s["median"] if r["flops"] and s["median"] > 0 else 0
            w.writerow([r["name"], *[r["params"].get(k, "") for k in keys], s["n_samples"], r["inner"], s["median"],
                        s["p25"], s["p75"], s["iqr"], s["ci95_lo"], s["ci95_hi"], s["mean"], s["stddev"], s["min"],
                        s["max"], gflops, r["thermal_before"], r["thermal_after"]])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="128,512,1024,2048,4096")
    p.add_argument("--streams", default="gpu,cpu")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--min-warmup-ms", type=float, default=200)
    p.add_argument("--prewarm-ms", type=float, default=1000)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--min-sample-us", type=float, default=50)
    p.add_argument("--out", default=str(REPO / "data" / "scratch"))
    p.add_argument("--cooldown", action="store_true")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--ce-env", default=str(REPO / "build" / "release" / "benchmarks" / "ce_env"))
    a = p.parse_args()

    sizes = [int(x) for x in a.sizes.split(",") if x]
    streams = {"gpu": mx.gpu, "cpu": mx.cpu}
    chosen = [s for s in a.streams.split(",") if s]
    for s in chosen:
        if s not in streams:
            sys.exit(f"error: unknown stream {s}")

    env = record_environment(Path(a.ce_env))
    print("ComputeEdge MLX matmul benchmark")
    print(f"  {env['chip']} ({env['model']}), macOS {env['os_version']} ({env['os_build']}), "
          f"MLX {mx.__version__}, Python {platform.python_version()}")
    print(f"  power: {env['power_source']}{', LOW POWER MODE' if env['low_power_mode'] else ''}, "
          f"thermal: {env['thermal_state']}, git {env['git_sha']}")
    if env["power_source"] != "AC Power":
        print("  WARNING: not on AC power.")

    # Process-level GPU pre-warm (excluded from results), as in metal_matmul.
    wa, wb = mx.random.normal((256, 256)), mx.random.normal((256, 256))
    mx.eval(wa, wb)
    until, calls = time.perf_counter_ns() + int(a.prewarm_ms * 1e6), 0
    while time.perf_counter_ns() < until:
        mx.eval(mx.matmul(wa, wb, stream=mx.gpu))
        calls += 1
    print(f"GPU pre-warm: {calls} dispatches over {a.prewarm_ms:g} ms\n")

    results = []
    rng = np.random.default_rng(1)
    for n in sizes:
        an = rng.uniform(-1, 1, (n, n)).astype(np.float32)
        bn = rng.uniform(-1, 1, (n, n)).astype(np.float32)
        A, B = mx.array(an), mx.array(bn)
        mx.eval(A, B)
        ref = None
        if n <= 2048:
            # NumPy 2.x on Accelerate can raise spurious FP warnings in matmul; suppress
            # them but require a finite reference.
            with np.errstate(all="ignore"):
                ref = an.astype(np.float64) @ bn.astype(np.float64)
            if not np.all(np.isfinite(ref)):
                print(f"reference matmul produced non-finite values at n={n}", file=sys.stderr)
                return 1
        flops = 2.0 * n * n * n

        for sname in chosen:
            stream = streams[sname]
            if ref is not None:  # correctness gate (same tolerance as the C++ benchmarks)
                got = np.array(mx.matmul(A, B, stream=stream))
                err = np.max(np.abs(got - ref) / (np.abs(ref) + 1.0)) / np.sqrt(n)
                if err > 1e-5:
                    print(f"correctness check FAILED at n={n} stream={sname} (scaled error {err:.3g})",
                          file=sys.stderr)
                    return 1
            if a.cooldown:
                wait_for_nominal()

            def fn(A=A, B=B, stream=stream):
                mx.eval(mx.matmul(A, B, stream=stream))

            r = run(fn, a.warmup, int(a.min_warmup_ms * 1e6), a.reps, int(a.min_sample_us * 1e3))
            r.update(name="matmul_mlx", flops=flops, summary=summarize(r["samples"]),
                     params={"framework": "mlx", "stream": sname, "m": str(n), "n": str(n), "k": str(n),
                             "dtype": "f32"})
            s = r["summary"]
            print(f"matmul_mlx stream={sname} n={n}  median {s['median'] / 1e3:.3f} us  IQR {s['iqr'] / 1e3:.3f} us  "
                  f"CI95 [{s['ci95_lo'] / 1e3:.3f}, {s['ci95_hi'] / 1e3:.3f}] us  {flops / s['median']:.1f} GFLOP/s  "
                  f"[{r['thermal_before']}->{r['thermal_after']}, x{r['inner']}]", flush=True)
            results.append(r)

    if not a.no_write:
        out = write_results(Path(a.out), env, results)
        print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
