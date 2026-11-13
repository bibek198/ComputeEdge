# Research data

Every measurement behind ComputeEdge's findings and paper. **These files are the primary data: do not edit
or delete them.**

## Rules

- **Immutable.** A committed result file is never modified. A run that turns out to be wrong (a bug, the wrong
  build) is superseded by a new run, and the reason is recorded in `research/findings.md`. Its files stay.
- **Checked.** `MANIFEST.sha256` holds the SHA-256 of every file. `ctest` runs
  `scripts/results_manifest.sh check`, which fails if any listed file has changed or disappeared.
  After committing new runs, run `scripts/results_manifest.sh add`; existing entries are never rewritten.
  The pre-refactor manifest is retained, unmodified, in `docs/verification/baseline/BASELINE.sha256`.
- **Byte-exact in git.** `.gitattributes` marks this tree `-text`, so git never rewrites line endings.
- **Provenance.** Every run directory is `<YYYYmmdd-HHMMSS>_<git sha>` (`-dirty` if the tree had uncommitted
  changes) and carries `env.json`: chip, OS build, power source, Low Power Mode, thermal state, compiler, flags,
  build type and git SHA. The code that produced a run is `git checkout <sha>`.

- **Excluded, not deleted.** A run invalidated by something outside the experiment (power source change, a
  concurrent heavy app) is listed in `exclusions.json` with the evidence, and left out of every analysis. Its
  files stay. Never exclude a run because its result is inconvenient.

## Layout

| Path | Contents |
|---|---|
| `sessions/orig/<stamp>_<sha>/` | The original M4–M6 runs (the "orig" session). One directory per benchmark process: `env.json`, `samples.csv`, `summary.csv`, and when present `metrics.csv`, `metrics_summary.csv`, `timeseries.csv` |
| `sessions/<stamp>_s<N>_<sha>/` | Confirmation sessions (`scripts/run_session.sh`): `session.json`, `conditions.csv`, `logs/`, plus one run directory per benchmark |
| `derived/comparisons/`, `derived/reports/` | Derived tables and figures (regenerable from the runs; kept because the docs cite them) |
| `exclusions.json` | Runs left out of analysis, each with its reason and evidence |
| `scratch/` | Default `--out` for ad-hoc benchmark runs. Git-ignored and never read by the analysis |

## File schemas

| File | Columns |
|---|---|
| `samples.csv` | benchmark, parameters…, `rep`, `inner_iterations`, `ns_per_call`, `thermal` (one row per measured repetition) |
| `summary.csv` | benchmark, parameters…, `n_samples`, `inner_iterations`, `median_ns`, `p25_ns`, `p75_ns`, `iqr_ns`, `ci95_lo_ns`, `ci95_hi_ns` (bootstrap of the median, 2000 resamples, seed 0xCE), `mean_ns`, `stddev_ns`, `min_ns`, `max_ns`, `gflops_median`, `thermal_before`, `thermal_after` |
| `metrics.csv` | benchmark, parameters…, `rep`, `metric`, `value` (long format; per-call phase breakdown, ns unless the name says otherwise) |
| `metrics_summary.csv` | per (benchmark, parameters, metric): the same statistics as `summary.csv` |
| `timeseries.csv` | thermal_drift only: `t_s`, `cycle`, `workload`, `ns`, `thermal` (every call, in order) |

Times are nanoseconds from `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`. GPU timestamps (`MTLCommandBuffer
GPUStartTime/GPUEndTime`) share that timebase. The meaning of each phase metric is documented in
`include/computeedge/metal.h` (`Timing`) and `include/computeedge/metal_probes.h`.

## Superseded runs

| Run | Reason |
|---|---|
| (storage_modes, first attempt at 9458c64's parent) | Not committed: the process was killed at 1 GiB by a probe memory leak (fixed in 9458c64) and wrote nothing. |
