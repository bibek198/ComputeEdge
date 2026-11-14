# ADR 0001: Raw Experimental Evidence Immutability and Manifest Verification

## Status
Accepted

## Context
ComputeEdge is a scientific research instrument measuring microarchitectural phenomena (GPU dispatch latency, unified memory bandwidth, SLC contention, thermal drift) on Apple Silicon. The conclusions and publications rely on raw experimental datasets captured during controlled hardware sessions.

During codebase refactoring, reorganizing directory structures or re-running analysis scripts risked inadvertently overwriting, deleting, or altering raw measurements. A foundational research invariant was required to guarantee historical evidence immutability.

## Decision
1. **Immutable Raw Baseline:** The original 137 raw experimental measurement files are permanently frozen. Their SHA-256 hashes are captured in `docs/verification/baseline/BASELINE.sha256` and never regenerated.
2. **Dedicated Data Roots:**
   - Historical sessions reside under `data/sessions/orig/`.
   - Confirmation sessions reside under `data/sessions/`.
   - Ad-hoc and scratch benchmark runs default strictly to `data/scratch/` (git-ignored) and never write into historical session trees.
3. **Automated Manifests and Write Protection:**
   - Every completed session directory generates a `manifest.sha256`.
   - Completed session directories are marked write-protected (`chmod a-w`).
   - `scripts/results_manifest.sh check` and `tools/verification/verify_baseline_hashes.py` enforce hash integrity in CI and local equivalence gates.
4. **Append-Only Policy:** Defective runs are never edited or rewritten in-place; they are superseded by new runs with documented provenance.

## Consequences
- Guaranteed reproducibility of research findings across refactorings and future development.
- In-place modifications to data directories are immediately rejected by gate verification scripts.
- Analysis pipelines operate strictly in read-only mode over raw sessions.
