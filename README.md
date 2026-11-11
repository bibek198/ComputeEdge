# ComputeEdge

An instrumented C++ tensor engine used to measure, and then exploit, the real costs of CPU–GPU execution on
Apple Silicon's unified memory.

- **Platform:** Apple Silicon (MacBook Air M4)
- **Stack:** C++20, Objective-C++, Apple Metal, Accelerate
- **Status:** M1 done (tensor core, instrumented allocator, event log, ScopedTimer); M2 (Accelerate baseline + benchmark harness) next

## Build

Requires Apple Silicon, Xcode Command Line Tools, CMake ≥ 3.24 and Ninja (`brew install cmake ninja`).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

./build/examples/matrix_multiplication events.csv   # prints 19 22 / 43 50 and writes the event log
```

Add `-DCE_SANITIZE=ON` (with `-DCMAKE_BUILD_TYPE=Debug`) for an AddressSanitizer + UBSan build.

## Layout

```text
include/computeedge/   public headers: tensor, allocator, instrumentation, scoped_timer, error
src/core/              tensor, 16 KB-aligned host allocator, event log
src/cpu/               naive matmul (correctness reference)
tests/                 GoogleTest suites
examples/              matrix_multiplication
```

## Documents

- [Project plan & architecture](docs/plan_and_architecture.md): goals, research questions, architecture diagrams, methodology, roadmap
- [Related work & novelty assessment](research/related_work.md)
