#!/usr/bin/env bash
# Refactoring equivalence gate. Compares the committed HEAD against the pre-refactor
# baseline in docs/verification/ and exits non-zero on any difference:
#
#   1. raw data   every file in the retained baseline manifest still hashes to its
#                 recorded value, at its (possibly moved) location
#   2. schema     output schema of every benchmark binary is unchanged
#   3. trace      execution traces (what each benchmark runs) are unchanged
#   4. analysis   claim verdicts/values identical; numbers.tex identical apart from
#                 the timestamped provenance line
#
#   tools/verification/check_equivalence.sh [--skip-build]
#
# Requires a clean tree. Analysis runs in a temporary worktree of HEAD, so no
# tracked file is modified.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo"
base="$repo/docs/verification"
skip_build=0
[[ "${1:-}" == "--skip-build" ]] && skip_build=1

# The binaries come from build/release, so the tree must match HEAD exactly.
[[ -z "$(git status --porcelain)" ]] || { echo "error: commit first; the gate checks HEAD and its build" >&2; exit 2; }
work="$(mktemp -d)"
cleanup() { git -C "$repo" worktree remove --force "$work/wt" >/dev/null 2>&1 || true; rm -rf "$work"; }
trap cleanup EXIT
git worktree add -q --detach "$work/wt" HEAD
wt="$work/wt"
fail=0
pass() { echo "PASS  $1"; }
bad()  { echo "FAIL  $1"; fail=1; }

# ---- 1. raw data -------------------------------------------------------------
# The data tree moved during the refactor; locate it, then verify the retained
# baseline hashes against the files at their current paths.
if python3 "$repo/tools/verification/verify_baseline_hashes.py" "$base/baseline/BASELINE.sha256" "$wt"; then
  pass "raw data: all baseline files present with identical hashes"
else
  bad "raw data hashes"
fi

# ---- 2+3. schema and traces ---------------------------------------------------
build="$repo/build/release"
if [[ $skip_build -eq 0 ]]; then
  (cmake --preset release >/dev/null && cmake --build --preset release >/dev/null) || { bad "build"; exit 1; }
fi
capture="$wt/tools/verification/capture_schema.sh"
[[ -x "$capture" ]] || capture="$repo/tools/verification/capture_schema.sh"
(cd "$repo" && CE_PYTHON="$repo/.venv/bin/python" "$capture" "$work/schema.json" "$build" "$work/traces" >/dev/null)
if cmp -s "$base/schema_pre_refactor.json" "$work/schema.json"; then pass "schema identical"
else bad "schema differs:"; diff "$base/schema_pre_refactor.json" "$work/schema.json" | head -40; fi
if diff -r "$base/trace_pre_refactor" "$work/traces" >/dev/null; then pass "execution traces identical"
else bad "execution traces differ:"; diff -r "$base/trace_pre_refactor" "$work/traces" | head -40; fi

# ---- 4. analysis ----------------------------------------------------------------
py="$repo/.venv/bin/python"
claims_json=""; numbers=""
if [[ -f "$wt/benchmarks/analysis/confirm_claims.py" ]]; then           # pre-Phase-5 layout
  (cd "$wt" && "$py" benchmarks/analysis/confirm_claims.py >/dev/null)
  claims_json="$(find "$wt/benchmarks/results/reports" -maxdepth 1 -type d -name 'confirm-*' | sort | tail -1)/claims.json"
  numbers="$wt/paper/generated/numbers.tex"
elif [[ -f "$wt/python/pyproject.toml" || -d "$wt/python/computeedge_analysis" ]]; then   # package layout
  (cd "$wt" && PYTHONPATH="$wt/python" "$py" -m computeedge_analysis.cli claims --out "$work/claims" >/dev/null)
  claims_json="$work/claims/claims.json"
  numbers="$wt/paper/generated/numbers.tex"
else
  bad "analysis: cannot find the claims checker"; claims_json=/dev/null; numbers=/dev/null
fi
if cmp -s "$base/baseline/claims.json" "$claims_json"; then pass "claim verdicts and values identical"
else bad "claims differ:"; diff "$base/baseline/claims.json" "$claims_json" | head -30; fi
if grep -v '^% Source:' "$numbers" | cmp -s "$base/baseline/numbers.normalized.tex" -; then pass "numbers.tex identical (normalized)"
else bad "numbers.tex differs:"; grep -v '^% Source:' "$numbers" | diff "$base/baseline/numbers.normalized.tex" - | head -30; fi

if [[ $fail -eq 0 ]]; then
  echo "ALL EQUIVALENCE GATES PASS ($(git rev-parse --short HEAD))"
else
  echo "EQUIVALENCE GATES FAILED"
  exit 1
fi
