# Getting Started

## Requirements

- An Apple Silicon Mac. Development and measurement use a MacBook Air M4.
- Xcode Command Line Tools. Full Xcode is not required, because Metal shaders are compiled at runtime.
- CMake ≥ 3.24 and Ninja: `brew install cmake ninja`

## Build and test

Two presets are defined in `CMakePresets.json`. Everything they produce goes under `build/`, which git ignores.

```sh
cmake --preset release && cmake --build --preset release && ctest --preset release   # build/release
cmake --preset debug   && cmake --build --preset debug   && ctest --preset debug     # build/debug, ASan + UBSan
```

Use `release` for benchmarks and `debug` while developing. Deleting `build/` is always safe, because it is fully regenerated.

The test suite covers:
- Tensor, allocator and instrumentation unit tests.
- CPU matmul (naive and Accelerate) against a double-precision reference.
- Metal matmul in every storage mode against Accelerate.
- Timing invariants: GPU timestamps fall inside the CPU timeline.
- Two layering checks: the core library links without Apple frameworks, and public headers contain no Objective-C.

## Using the library

```cpp
#include "computeedge/computeedge.h"
#include "computeedge/metal.h"

using namespace computeedge;

Tensor A({2, 2}, {1, 2, 3, 4});
Tensor B({2, 2}, {5, 6, 7, 8});

Tensor C_cpu = matmul(A, B);                                     // Accelerate (default)
Tensor C_ref = matmul(A, B, MatmulImpl::Naive);                  // reference

metal::Timing t;
Tensor C_gpu = metal::matmul(A, B, {metal::StorageMode::NoCopy}, &t);
// t.encode_ns, t.commit_ns, t.wait_ns, t.gpu_ns(), t.queue_ns(), t.wake_ns(), ...
```

Link against `computeedge`, which brings in the core, CPU and Metal libraries. `computeedge_core` alone has no Apple
framework dependencies.

## Event log

```cpp
ScopedEventLogging logging;            // enable for this scope
Tensor C = metal::matmul(A, B);
EventLog::global().write_csv("events.csv");
```

Each CSV row is `t_ns,type,id,exec_id,bytes,backend,label`, with all timestamps on one clock, GPU ones included.
`./build/release/examples/matrix_multiplication events.csv` writes a small example log.

## Benchmarks

Run benchmarks from a **Release** build, **on AC power**, and **after re-running `cmake --preset release`** once your changes are
committed. The git SHA is captured at configure time, so this keeps results tagged with the right revision.

| Binary | What it measures |
|---|---|
| `build/release/benchmarks/cpu_matmul` | Accelerate vs naive SGEMM, compute-only and including output allocation |
| `build/release/benchmarks/metal_matmul` | Metal SGEMM per storage mode (NoCopy / Shared / Private), with per-phase breakdown and cold start |
| `build/release/benchmarks/event_log_overhead` | Cost of `emit()` and `ScopedTimer` with the log disabled and enabled |
| `benchmarks/mlx_matmul/mlx_matmul.py` | MLX matmul on the CPU and GPU streams, using the same protocol and output format |
| `benchmarks/analysis/compare_baselines.py` | Comparison table and figures from the newest clean results |
| `build/release/benchmarks/dispatch_overhead` | M5 Track A: empty-kernel round trips vs dispatches, encoders and command buffers per submission, plus cold start |
| `build/release/benchmarks/sync_methods` | M5 Track A: five ways to wait for the GPU, and the cost of an idle gap before a submission |
| `build/release/benchmarks/storage_modes` | M5 Track B: buffer creation, CPU first touch, first GPU use (fresh / rewrap / reused), GPU bandwidth per storage kind |
| `benchmarks/analysis/m5_report.py` | M5 report (tables and figures) from the newest clean runs; findings are in `research/findings.md` |
| `build/release/benchmarks/cache_interference` | M6 RQ3: CPU pass slowdown after/during GPU work, against equal-duration controls |
| `build/release/benchmarks/thermal_drift` | M6 RQ4: 15 min sustained load; CPU and GPU SGEMM over time, crossover drift (`--minutes`, `--load`) |
| `build/release/benchmarks/followups` | M6: experiments testing the open M5 findings (wake latency, CPU writes, GPU write size) |
| `benchmarks/analysis/m6_report.py` | M6 report from the newest clean runs |

The Metal benchmark's `--kernels tiled16,mps` option runs our kernel and Apple's MPS GEMM through the same code path.

### Running the full baseline set (M4)

```sh
python3 -m venv .venv && .venv/bin/pip install -r benchmarks/requirements.txt   # once
scripts/run_baselines.sh
```

The script rebuilds the release preset (which refreshes the git SHA), runs the CPU, Metal and MLX benchmarks with
cooldown, and writes `benchmarks/results/comparisons/<timestamp>/` containing `comparison.md`, `comparison.csv`,
`latency.png`, `throughput.png` and `sources.json`. Commit first so results are not tagged `-dirty`.

### Running the M5 characterization

```sh
build/release/benchmarks/dispatch_overhead --cooldown
build/release/benchmarks/sync_methods --cooldown
build/release/benchmarks/storage_modes --cooldown        # ~5 min; up to ~3.6 GB of memory at 1 GiB
.venv/bin/python benchmarks/analysis/m5_report.py         # -> benchmarks/results/reports/m5-<timestamp>/
```

M6: `cache_interference --cooldown`, `followups --cooldown`, `thermal_drift --minutes 15` (starts by waiting for
thermal state nominal), then `.venv/bin/python benchmarks/analysis/m6_report.py`. For power and frequency during the
thermal run, start `sudo powermetrics --samplers cpu_power,gpu_power,thermal -i 1000` in another terminal.

The low-level probes behind these benchmarks (`include/computeedge/metal_probes.h`) can also be called directly.

Common flags are `--sizes`, `--reps`, `--warmup`, `--min-warmup-ms`, `--cooldown` (wait for thermal state
`nominal`), `--out DIR` and `--no-write`. Each binary's source file header lists all of its flags.

Each run writes `benchmarks/results/<timestamp>_<git-sha>/`:

| File | Contents |
|---|---|
| `env.json` | Chip, GPU, model, macOS build, power source, Low Power Mode, thermal state, compiler and flags, build type, git SHA |
| `samples.csv` | Every measured repetition, tagged with thermal state |
| `summary.csv` | Median, IQR, bootstrap 95% CI of the median, mean, min/max, GFLOP/s |
| `metrics.csv`, `metrics_summary.csv` | Per-call phase breakdown (Metal benchmarks) |

The measurement protocol (warm-up, GPU pre-warm, repetitions, thermal handling) is described in
[plan_and_architecture.md §7.2](plan_and_architecture.md).
