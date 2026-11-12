# ComputeEdge

An instrumented C++ tensor engine used to measure, and then exploit, the real costs of CPU–GPU execution on
Apple Silicon's unified memory.

- **Platform:** Apple Silicon (MacBook Air M4)
- **Stack:** C++20, Objective-C++, Apple Metal, Accelerate
- **Status:** M1–M2 done (tensor core, instrumentation, Accelerate baseline, benchmark harness); M3 (Metal backend) next

## Build

Requires Apple Silicon, Xcode Command Line Tools, CMake ≥ 3.24 and Ninja (`brew install cmake ninja`).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

./build/examples/matrix_multiplication events.csv   # prints 19 22 / 43 50 and writes the event log
```

Add `-DCE_SANITIZE=ON` (with `-DCMAKE_BUILD_TYPE=Debug`) for an AddressSanitizer + UBSan build.

## Benchmarks

```sh
./build/benchmarks/cpu_matmul                       # Accelerate + naive, 128²…4096², kernel and end-to-end
./build/benchmarks/cpu_matmul --sizes 128,256 --impls accelerate --reps 50 --cooldown
```

Each run writes `benchmarks/results/<timestamp>_<git-sha>/` containing `env.json` (chip, macOS build, power
source, thermal state, compiler, build type, git SHA), `samples.csv` (every repetition, tagged with thermal state)
and `summary.csv` (median, IQR, bootstrap 95% CI, GFLOP/s). Run on AC power, from a Release build, after
re-running `cmake` so the recorded git SHA is current. `--cooldown` waits for thermal state `nominal` before each
benchmark.

## Layout

```text
include/computeedge/   public headers: tensor, ops, allocator, instrumentation, scoped_timer, benchmark
src/core/              tensor, 16 KB-aligned host allocator, event log
src/cpu/               matmul: Accelerate cblas_sgemm (baseline) and naive (reference)
src/benchmark/         harness: runner, statistics, environment recorder (Objective-C++), CSV/JSON output
benchmarks/            benchmark executables and committed results
tests/                 GoogleTest suites
examples/              matrix_multiplication
```

## Documents

- [Project plan & architecture](docs/plan_and_architecture.md): goals, research questions, architecture diagrams, methodology, roadmap
- [Related work & novelty assessment](research/related_work.md)
