# Getting Started

## Requirements

- An Apple Silicon Mac. Development and measurement use a MacBook Air M4.
- Xcode Command Line Tools. Full Xcode is not required, because Metal shaders are compiled at runtime.
- CMake ≥ 3.24 and Ninja: `brew install cmake ninja`

## Build and test

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

For development, build with AddressSanitizer and UBSan:

```sh
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCE_SANITIZE=ON
cmake --build build-asan && ctest --test-dir build-asan --output-on-failure
```

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
`./build/examples/matrix_multiplication events.csv` writes a small example log.

## Benchmarks

Run benchmarks from a **Release** build, **on AC power**, and **after re-running `cmake`** once your changes are
committed. The git SHA is captured at configure time, so this keeps results tagged with the right revision.

| Binary | What it measures |
|---|---|
| `build/benchmarks/cpu_matmul` | Accelerate vs naive SGEMM, compute-only and including output allocation |
| `build/benchmarks/metal_matmul` | Metal SGEMM per storage mode (NoCopy / Shared / Private), with per-phase breakdown and cold start |
| `build/benchmarks/event_log_overhead` | Cost of `emit()` and `ScopedTimer` with the log disabled and enabled |

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
