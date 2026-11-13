# ComputeEdge

An instrumented C++ tensor engine used to measure, and then exploit, the real costs of CPU–GPU execution on
Apple Silicon's unified memory.

- **Platform:** Apple Silicon (MacBook Air M4)
- **Stack:** C++20, Objective-C++, Apple Metal, Accelerate
- **Status:** M1–M3 done (tensor core, instrumentation, Accelerate baseline, benchmark harness, Metal backend); M4 (baselines vs MLX) next

## Quick start

```sh
brew install cmake ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build && ctest --test-dir build --output-on-failure
./build/examples/matrix_multiplication            # 19 22 / 43 50
./build/benchmarks/metal_matmul --sizes 128,1024  # per-phase GPU cost breakdown
```

See [docs/getting_started.md](docs/getting_started.md) for the library API, the event log and benchmarks.

## Layout

```text
include/computeedge/   public headers (no Objective-C): tensor, ops, metal, allocator, instrumentation, benchmark
src/core/              tensor, 16 KB-aligned host allocator, event log, validation   → computeedge_core
src/cpu/               matmul: Accelerate cblas_sgemm (baseline), naive (reference)  → computeedge_cpu
src/metal/             Metal backend + runtime-compiled kernels                       → computeedge_metal
src/benchmark/         harness: runner, statistics, environment recorder              → computeedge_bench
benchmarks/            baseline/ and overhead/ benchmarks, committed results/
tests/                 GoogleTest suites + layering checks
docs/, research/       plan & architecture, getting started, related work
```

## Documents

- [Project plan & architecture](docs/plan_and_architecture.md): goals, research questions, architecture diagrams, methodology, roadmap
- [Getting started](docs/getting_started.md)
- [Related work & novelty assessment](research/related_work.md)
