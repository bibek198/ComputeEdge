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

## Tooling (Phase 1)

- **Formatting.** `.clang-format` (Google base, 120 columns, short functions on one line, as in the existing code) and
  `.editorconfig`. The formatting pass was whitespace-only: all 44 C++/ObjC++ files are identical to their
  pre-format versions with whitespace removed. Include sorting followed in a separate commit; each changed file
  has the same multiset of lines as before. `.metal` files and `kernel_sources.h.in` are not formatted.
- **Static analysis.** `.clang-tidy` enforces the naming convention plus bugprone / performance checks.
  `ruff.toml` covers Python, and `shellcheck` covers shell. `tools/lint.sh` runs them all.
- **CI.** `.github/workflows/ci.yml` builds and tests release and debug (ASan/UBSan) with `-DCE_WARNINGS_AS_ERRORS=ON`
  (project targets only, never googletest). It also verifies the data against the retained baseline manifest
  (mandatory) and runs lint. Committed, not pushed. Verified locally: both presets build with `-Werror` and pass 162 tests.
- **Lint baseline** (report-only until each reaches zero; `tools/lint.sh` lists the blocking checks):

  | Check | Findings at Phase 1 | Fixed in |
  |---|---:|---|
  | format | 0 (blocking) | — |
  | clang-tidy | 74 (44 naming, 11 exception-escape, 2 empty-catch, 2 use-after-move, misc. performance) | naming: phases 2–4; error paths: phase 6 |
  | ruff | 144 (69 missing docstrings, lambdas, loop-variable capture, line length, naming) | phase 5 |
  | shellcheck | 1 (`run_session.sh`) | phase 8 |

## Layout and data move (Phase 2)

- **2a, C++ modules.** `include/computeedge/{core,cpu,metal,bench}/` with `src/` and `tests/` mirroring them.
  `benchmark.h`/`.cpp` were split by section into `bench/{stats,environment,runner,result_writer,trace}`, and
  `ops.h` into `cpu/matmul.h` plus `core/validate.h`. The tests were split by suite into `*_test.cpp`. Code was moved
  verbatim. All 162 tests pass in both presets, and every public header compiles on its own.
- **2b, binaries.** `benchmarks/<binary>/main.cpp`, `benchmarks/harness/`, `benchmarks/mlx_matmul/`,
  `tools/ce_env/` and `scripts/`. Code was moved verbatim. Built binaries keep their paths.
- **2c, data.** `benchmarks/results/` became `data/`, with `sessions/orig/` (15 original runs), `sessions/<id>/`,
  `derived/` and `exclusions.json`, all moved with `git mv`.
  - **Manifest:** the old file with its paths rewritten, not re-hashed. Its first 137 hash fields are identical to
    the retained baseline. Three entries were appended for a committed claims report
    (`confirm-20261114-011725`) that predates the refactor and had never been added.
- **The one intentional default change.** The default `--out` of every benchmark (C++ and MLX) moved from
  `benchmarks/results`, which no longer exists, to `data/scratch/`, which is git-ignored. Ad-hoc runs can no longer
  enter the original session. Schema and trace snapshots pass `--out` explicitly, so they are unaffected.
- **Incident, caught and reverted before commit.** A bulk replacement of path strings in comments and docs also
  edited two committed report files under `data/derived/reports/` and the generated `paper/generated/numbers.tex`.
  - Detection: the manifest check and a byte comparison.
  - Fix: the files were restored from the git index / HEAD, and both are byte-identical to `pre-refactor`
    (verified with `cmp`).
  - Prevention: every later text replacement excludes `data/` and `paper/generated/`. Phase 8 makes the data
    zone write-protected for analysis code.

## Gate results

| Phase | Gate | Result |
|---|---|---|
| 0a | Baseline captured | PASS |
| 0b | Hook inert when disabled (schema byte-identical; 162 tests pass) | PASS |
| 0b | Baseline execution traces captured | PASS |
| 1 | Formatting is whitespace-only (token check over 44 files) | PASS |
| 1 | Include sorting is reordering-only | PASS |
| 1 | Both presets build with `-Werror`; 162 tests pass | PASS |
| 1 | Equivalence gate after formatting (data, schema, traces, claims, numbers) | PASS |
| 2a | Equivalence gate; 162 tests in both presets; headers self-contained | PASS |
| 2b | Equivalence gate | PASS |
| 2c | Baseline hashes at new paths (137/137); manifest hash column unchanged; equivalence gate | PASS |
