# Refactoring verification record

This document proves that the 2026-11 restructuring of ComputeEdge did not change what is measured, how results are
labelled, or any raw data, and it records every intentional behaviour change. The plan has two passes:

- **Pass 1 (phases 1–5)** is behaviour-preserving restructuring.
- **Pass 2 (phases 6–8)** makes correctness and safety changes, one per commit, each with its own test.

Each phase appends its gate results below. Snapshots live in `docs/verification/`.

## Baseline (Phase 0a)

| Item | Value |
|---|---|
| Baseline commit | `1ddffcd` (tag `pre-refactor`) |
| Raw-data manifest | `benchmarks/results/MANIFEST.sha256`: 137 files; SHA-256 `4fe583df62e7760a17ccc9160b265929723f0d2b7fe9a0fec7b0053ca5c75601` |
| Retained manifest copy | `docs/verification/baseline/BASELINE.sha256` (identical bytes; never regenerated) |
| Claim verdicts | `docs/verification/baseline/claims.json` (SHA-256 `e214dda0…96d5`): C1–C8, C10, C11, C14–C16 confirmed (2/2); C12, C13 confirmed (1/1); C9 inconsistent (1/2); C17 no data |
| `numbers.tex` | `docs/verification/baseline/numbers.normalized.tex` (SHA-256 `5df8aec7…b70a`) |
| Output schema | `docs/verification/schema_pre_refactor.json`, from `tools/verification/capture_schema.sh` against the `pre-refactor` build |

**Normalization of `numbers.tex`.** The generator writes one provenance line, `% Source: <report dir>; N session(s).`,
and the report directory name contains a timestamp. Two runs of identical code therefore differ in that line only.
The equivalence gate compares the file with that line removed (`grep -v '^% Source:'`); every other byte must match.

**How the baselines were produced.** Claims and `numbers.tex` came from `benchmarks/analysis/confirm_claims.py`, run in a
detached worktree at `pre-refactor`, so no tracked file in the main tree was modified. The schema came from the
release build of `pre-refactor` (`cmake --preset release && cmake --build --preset release`), running
`tools/verification/capture_schema.sh docs/verification/schema_pre_refactor.json`.

**Build and test commands (baseline):**
```sh
cmake --preset release && cmake --build --preset release && ctest --preset release   # 161 tests
cmake --preset debug   && cmake --build --preset debug   && ctest --preset debug     # ASan + UBSan
```

## Instrumentation (Phase 0b)

- **Hook.** `bench::trace()` and `bench::trace_enabled()` (`include/computeedge/benchmark.h`) are inert unless
  `CE_TRACE` names a file. When enabled, they record *what* executes, never timings: each `run` / `run_with_metrics`
  call with its name, parameters, warm-up / calibration / repetition policy and setup flag; each pre-warm workload;
  the correctness gates; the cache-calibration steps; and the thermal heater and workload set.
- **Inert when disabled.** The schema snapshot with the hook compiled in and disabled is byte-identical to
  `schema_pre_refactor.json`, and is still identical with tracing on. The unit test `Trace.InertWithoutEnvironmentVariable`
  pins the disabled behaviour.
- **Baseline traces.** `docs/verification/trace_pre_refactor/<binary>.trace` (9 binaries, 152 lines), captured with
  `tools/verification/capture_schema.sh <json> build/release docs/verification/trace_pre_refactor`. They pin the
  intentionally different pre-warm workloads (finding D14): `metal_matmul` uses `metal_matmul n=256 storage=shared`,
  while the probe benchmarks use `gpu_stream_write kind=private bytes=67108864`.
- **Tag.** `pre-refactor-instrumented`.

## Gate results

| Phase | Gate | Result |
|---|---|---|
| 0a | Baseline captured | PASS |
| 0b | Hook inert when disabled (schema byte-identical; 162 tests pass) | PASS |
| 0b | Baseline execution traces captured | PASS |
