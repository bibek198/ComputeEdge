# ADR 0004: Python Analysis Package Architecture and Pre-Registered Claims

## Status
Accepted

## Context
Analysis and report generation were originally carried out by multiple monolithic scripts under `benchmarks/analysis/`. These scripts mixed CSV file reading, ad-hoc string parsing, statistical calculations, Matplotlib plotting, LaTeX macro generation, and claim evaluation with duplicate logic across files.

A robust, modular, and type-safe Python package was required to guarantee reproducibility of all pre-registered scientific claims (C1–C17) across both historical and new hardware sessions.

## Decision
1. **Package Layout (`computeedge_analysis`):**
   Established a modern Python package structure under `python/computeedge_analysis/`:
   - `records.py`: Immutable, typed dataclasses (`SummaryRow`, `MetricSummaryRow`, `MetricSample`, `TimeseriesRow`) using `slots=True` and `frozen=True`.
   - `io/`: Dedicated readers (`sessions.py`, `runs.py`, `exclusions.py`) enforcing schema validation, handling missing optional fields gracefully, and providing cached session readers.
   - `stats.py`: Verified statistical functions (`slowdown_pct`, `slowdown_ci` using 2000-iteration percentile-bootstrap with fixed seeds, `linear_slope`).
   - `units.py`: Unit conversion and human-readable formatters (`size_bytes`, `fmt_compact`).
   - `style.py`: Standardized publication light-mode palette and formatting helpers for paper figures.
   - `claims/`: Split into modular responsibilities:
     - `registry.py`: Declarative registration of claim metadata and evaluators.
     - `definitions.py`: Evaluator definitions for pre-registered claims C1–C17 preserving verbatim scientific criteria.
     - `evaluate.py`: Multi-session evaluation engine isolating per-claim exceptions.
   - `reports/`: Milestone reports (`baseline.py`, `characterization.py`, `interaction.py`).
   - `figures/`: Vector and raster paper figure generation (`paper.py`).
   - `cli.py`: Unified CLI entry points (`ce-claims`, `ce-report`, `ce-figures`).
2. **Pre-Registered Claims Invariance:**
   - Pre-registered criteria for claims C1 through C17 are strictly frozen.
   - All claim verdicts and values must identically match `docs/verification/baseline/claims.json`.
   - Generated LaTeX definitions must match `docs/verification/baseline/numbers.normalized.tex` under comment-stripped normalization.
3. **Automated Python Testing:**
   - Comprehensive unit and Hypothesis property tests under `tests/python/` executed via `pytest`.
   - Zero-finding linting enforced by `ruff`.

## Consequences
- Single source of truth for all research data transformations, metrics, and claim evaluations.
- Full isolation of claim evaluation failures: a defect in evaluating one claim does not crash multi-session reports.
- Seamless command-line interface for paper figure and report generation.
