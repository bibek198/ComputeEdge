# ADR 0003: Benchmark Presentation vs. Measurement Separation

## Status
Accepted

## Context
Initial benchmark implementations intermingled CLI option parsing, measurement execution loops, hardware state checks, console printing (`std::cout`), and CSV result persistence in single files. This tight coupling made it difficult to unit-test option parsing and error contracts without executing physical hardware sweeps, and prohibited silent or headless automated benchmarking.

## Decision
1. **Three-Module Structure per Benchmark:**
   Every benchmark directory under `benchmarks/<name>/` is decomposed into three isolated translation units:
   - `options.{h,cpp}`: CLI parameter parsing, common flags configuration, default assignments, and early option validation.
   - `experiments.{h,cpp}`: Pure measurement logic. Operates through abstract `Reporter` interfaces rather than directly printing to stdout.
   - `main.cpp`: Entry-point wrapper wiring CLI options, creating concrete `ConsoleReporter` and `ResultWriter` instances, invoking experiments, and returning exit codes.
2. **Abstract Reporter Interface:**
   - Introduced `computeedge::bench::harness::Reporter` with decoupled callbacks (`banner`, `step`, `result`, `detail`, `error`).
   - `SilentReporter` suppresses output during unit tests and automated sweeps.
   - `ConsoleReporter` reproduces the exact human-readable CLI terminal formatting of the original baselines.
3. **No stdout Pollution in Core Libraries:**
   - Removed direct `std::cout` emissions from `src/` and `include/`. Primitives like `Tensor::print()` were converted to `to_string()` and `operator<<`.
4. **Preservation of Measurement Invariants:**
   - Loop execution order, synchronization barriers, warm-up iterations, and pre-warm workloads (including the critical D14 distinction: 64 MiB Private write stream for probes vs. 256² Shared matrix multiplication loop for `metal_matmul`) were preserved identically.

## Consequences
- CLI flags and validation logic are unit-testable via GoogleTest death tests without hardware execution.
- Experiments can run headlessly and emit structured outputs without interleaved terminal formatting.
- Exact measurement parity with pre-refactoring baselines is strictly maintained.
