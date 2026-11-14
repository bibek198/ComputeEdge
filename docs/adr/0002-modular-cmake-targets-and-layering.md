# ADR 0002: Modular CMake Target Hierarchy and Header Layering

## Status
Accepted

## Context
ComputeEdge previously relied on monolithic targets and global compiler options (`add_compile_options()`, `add_link_options()`). This caused tight coupling across components, made Objective-C++ flags bleed into pure C++ units, and allowed unnecessary framework dependencies (e.g. Metal frameworks linked into purely CPU benchmarks).

## Decision
1. **Target-Scoped Compile Options:**
   - Removed all directory-level `add_compile_options()` and `add_link_options()`.
   - Introduced an `INTERFACE` target `computeedge_compile_options` carrying strict warnings (`-Wall`, `-Wextra`, `-Wpedantic`, `-Wconversion`, `-Wsign-conversion`, `-Wshadow`, etc.), `-Werror` (configurable via `CE_WARNINGS_AS_ERRORS`), and sanitizers (ASan, UBSan).
2. **Modern Namespace Aliases:**
   - Replaced flat target names with namespaced aliases:
     - `computeedge::core`: Memory management, host allocator, event log, tensor primitives (zero Apple framework dependencies).
     - `computeedge::cpu`: Accelerated CPU GEMM via Apple Accelerate framework and reference naive implementations.
     - `computeedge::metal`: Apple Metal backend, runtime kernel compilation, command queue management, and probe dispatch.
     - `computeedge::bench`: Benchmark runner core, execution environments, result writers, and window statistics.
     - `computeedge::harness`: Common benchmark harness CLI parsing, console reporting, correctness checking, and host memory helpers.
     - `computeedge::harness_gpu`: Shared GPU pre-warm workloads and synchronization helpers.
     - `computeedge::computeedge`: Umbrella interface target combining core, CPU, and Metal.
3. **Automatic ARC Configuration:**
   - Objective-C++ sources (`*.mm`) automatically receive `-fobjc-arc` via generator expressions (`$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>`) without requiring manual per-source flags.
4. **Header Self-Containment:**
   - Implemented `computeedge_header_checks` target compiling each public header independently to guarantee absence of hidden transitive header dependencies.
5. **Strict Link Hygiene:**
   - CPU-only targets (such as `cpu_matmul`) link only `computeedge::harness` and `computeedge::cpu`, completely free of Metal dependencies.

## Consequences
- Clean layering boundaries between host core, CPU accelerators, and Metal runtime.
- Fast compilation times and isolated rebuilds.
- Elimination of framework bloat in non-GPU binaries.
