# ComputeEdge

An instrumented C++20 tensor engine and characterization framework used to measure, and then exploit, the microarchitectural overheads of CPU–GPU execution on Apple Silicon unified memory.

- **Platform:** Apple Silicon (tested on Apple Silicon MacBook Air M4)
- **Stack:** C++20, Objective-C++, Apple Metal, Accelerate, CMake, Python 3.10+, NumPy, Matplotlib
- **Status:** Complete refactored architecture with modular CMake targets, presentation/measurement harness decoupling, robust error handling, typed Python analysis package, and automated equivalence verification across all 17 pre-registered research claims.

---

## Quick Start

### Prerequisites

```sh
brew install cmake ninja clang-format shellcheck
python3 -m venv .venv
source .venv/bin/activate
pip install -e . -r benchmarks/adapters/mlx/requirements.txt pytest hypothesis ruff==0.8.4
```

### Build & Run Tests

```sh
# Release build & test suite (207 C++ unit and contract tests)
cmake --preset release && cmake --build --preset release && ctest --preset release --output-on-failure

# Debug build with ASan + UBSan
cmake --preset debug && cmake --build --preset debug && ctest --preset debug --output-on-failure

# Python analysis test suite (22 pytest & Hypothesis property tests)
pytest

# Code style and linters (clang-format, ruff, shellcheck)
./tools/lint.sh
```

### Run Benchmarks & Analysis

```sh
# Run CPU SGEMM baseline
./build/release/benchmarks/cpu_matmul --sizes 128,1024

# Run Metal SGEMM with GPU phase timing breakdown
./build/release/benchmarks/metal_matmul --sizes 128,1024 --kernels mps,tiled16

# Verify pre-registered research claims C1–C17
ce-claims

# Generate milestone reports and paper figures
ce-report baseline
ce-figures
```

---

## Repository Layout

```text
cmake/                       Modular CMake infrastructure
  ComputeEdgeOptions.cmake   Build toggles (warnings, sanitizers, MLX adapter, benchmarks, tests)
  CompilerWarnings.cmake     Compiler warning flags for Clang / AppleClang
  Dependencies.cmake         External dependencies (Accelerate, GoogleTest via FetchContent)
  Metal.cmake                Metal SDK detection, MSL compiler, metallib generation
  Packaging.cmake            CPack / packaging configuration
  Sanitizers.cmake           ASan / UBSan configuration
include/computeedge/         Modular public C++ headers (clean C++20, zero Objective-C)
  core/                      Host allocator (16 KB page-aligned), EventLog, Tensor primitives
  cpu/                       Accelerate BLAS wrapper and naive reference GEMM
  metal/                     Metal device backend, buffer abstractions, runtime kernel compiler, probes
  bench/                     Benchmark runner, environment capture, ResultWriter, window statistics
src/                         Implementation sources
  core/                      Core memory management, event logging, and tensor operators
  cpu/                       Accelerate and naive matmul implementations
  metal/                     Metal backend runtime and low-level GPU probe dispatch
  bench/                     Benchmark framework, result writer, and thermal windowing
benchmarks/                  Decoupled benchmark suite (options, experiments, main)
  common/harness/            Shared harness: CLI parsing, console reporting, GPU helpers, run_main wrapper
  workloads/                 Independent experimental workloads
    gemm/                    cpu_matmul, metal_matmul
    overhead/                dispatch_overhead, sync_methods
    memory/                  storage_modes, followups
    interference/            cache_interference
    thermal/                 thermal_drift
  adapters/mlx/              MLX GEMM comparison baseline (runner script & requirements)
analysis/                    Python analysis subsystem (computeedge_analysis package)
  src/computeedge_analysis/  Typed records, io, stats, units, style, claims C1–C17, reports, figures, CLI
  tests/                     Pytest unit and Hypothesis property tests
configs/                     Session specifications and benchmark parameters (session.toml)
data/                        Raw experimental evidence and manifests
  sessions/orig/             Original baseline research sessions (137 files, permanently immutable)
  sessions/confirmation/     Confirmation benchmark sessions
  derived/                   Derived milestone reports and comparative analyses
  scratch/                   Default output directory for ad-hoc benchmark runs (git-ignored)
tests/                       Structured C++ test suites
  unit/                      Unit tests (core, cpu, metal, bench, harness)
  integration/benchmarks/    Integration and CLI option contract tests
  architecture/              Architecture and link hygiene enforcement (core_only_check)
research/                    Research documentation and hypothesis registry
  hypotheses/                Pre-registered claims C1–C17 criteria and registration records
  findings/                  Microarchitectural characterization findings on Apple Silicon
  related_work/              Context and literature comparison
  limitations/               Threats to validity and microarchitectural constraints
docs/                        Documentation, architecture, and verification
  adr/                       Architecture Decision Records (ADRs 0001–0005)
  plan_and_architecture.md   System goals, methodology, and architectural roadmap
  refactoring_verification.md Comprehensive phase-by-phase refactoring verification record
tools/                       Developer tooling and automated verification
  lint.sh                    Unified linter (clang-format, ruff, shellcheck)
  verification/              Equivalence gates, baseline hash verifiers, and schema capture
```

---

## Architectural Highlights

- **Modular CMake Target Hierarchy:** Modern namespaced targets (`computeedge::core`, `computeedge::cpu`, `computeedge::metal`, `computeedge::bench`, `computeedge::harness`, `computeedge::harness_gpu`). Strict link hygiene ensures non-GPU benchmarks (such as `cpu_matmul`) link zero Metal dependencies.
- **Decoupled Presentation & Measurement:** Every benchmark binary is cleanly split into `options.{h,cpp}`, `experiments.{h,cpp}`, and `main.cpp`. Experiments execute through abstract `Reporter` interfaces, keeping core and benchmark libraries free of stdout side effects.
- **Robust Error Handling:** Standardized `run_main()` wrapper catching `UsageError` (exit code 2), `InvalidRunError` (exit code 3), and `computeedge::Error` / `std::exception` (exit code 1). Automatically flushes partial results with a `PARTIAL` marker file on failure. Bounded timeouts (`10s`) prevent GPU synchronization hangs.
- **Raw Evidence Immutability:** Historical research data (137 baseline files) is pinned with SHA-256 hashes (`docs/verification/baseline/BASELINE.sha256`) and write-protected. All ad-hoc executions default to `data/scratch/`.
- **Pre-Registered Claims Verification:** Pre-registered claims C1–C17 are evaluated against multi-session data via `ce-claims`, guaranteeing exact parity of claim verdicts, numerical outputs, and publication-ready LaTeX tables (`numbers.tex`).

For details on key decisions, see the [Architecture Decision Records](docs/adr/README.md).

---

## Automated Equivalence Gate

To verify complete equivalence of raw data hashes, output schema, execution traces, claim verdicts, and generated LaTeX numbers against the pre-refactoring baseline:

```sh
./tools/verification/check_equivalence.sh
```

---

## Documentation Links

- [Getting Started](docs/getting_started.md): Detailed API examples, benchmark workflows, and options guide.
- [Architecture Decision Records](docs/adr/README.md): Detailed rationale for architectural choices.
- [Project Plan & Architecture](docs/plan_and_architecture.md): Scientific goals, methodology, and research questions.
- [Refactoring Verification Log](docs/refactoring_verification.md): Phase-by-phase verification gate audit trail.
- [Research Hypotheses](research/hypotheses/README.md): Pre-registered claims C1–C17 and criteria.
- [Research Findings](research/findings/README.md): Characterization findings on Apple Silicon unified memory.
- [Related Work](research/related_work/README.md): Context and literature comparison.
- [Threats to Validity & Limitations](research/limitations/README.md): Experimental validity and hardware boundaries.
