#!/usr/bin/env bash
# Runs every benchmark binary with minimal sweeps and records its output schema
# (tools/verification/schema_snapshot.py). Used to prove that a refactor did not
# change what any benchmark measures or how results are labelled.
#
#   tools/verification/capture_schema.sh OUT.json [BUILD_DIR] [TRACE_DIR]
#
# BUILD_DIR defaults to build/release. With TRACE_DIR, each binary also writes its
# execution trace (CE_TRACE, see bench::trace) to TRACE_DIR/<binary>.trace. Results go to a temporary directory that is
# removed afterwards; nothing is written to the data tree.
set -euo pipefail
cd "$(dirname "$0")/../.."

out_json="$1"
build="${2:-build/release}"
trace_dir="${3:-}"
if [[ -n "$trace_dir" ]]; then
  trace_dir="$(cd "$(dirname "$trace_dir")" && pwd)/$(basename "$trace_dir")"
  rm -rf "$trace_dir" && mkdir -p "$trace_dir"
fi
B="$build/benchmarks"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

quick=(--reps 3 --warmup 1 --min-warmup-ms 0)
run() {  # name, then the command
  local name="$1"; shift
  echo "  $name"
  local trace=""
  [[ -n "$trace_dir" ]] && trace="$trace_dir/$name.trace"
  CE_TRACE="$trace" "$@" > "$tmp/$name.log" 2>&1 || { echo "error: $name failed; see log:" >&2; cat "$tmp/$name.log" >&2; exit 1; }
}

mkdir -p "$tmp/runs"
o=(--out "$tmp/runs")
run cpu_matmul         "$B/cpu_matmul" "${quick[@]}" "${o[@]}" --sizes 64
run metal_matmul       "$B/metal_matmul" "${quick[@]}" "${o[@]}" --sizes 64 --prewarm-ms 0
run event_log_overhead "$B/event_log_overhead" --reps 3 "${o[@]}"
run dispatch_overhead  "$B/dispatch_overhead" "${quick[@]}" "${o[@]}" --prewarm-ms 0 --dispatches 1,4 --encoders 1,4 --cmdbufs 1,2
run sync_methods       "$B/sync_methods" "${quick[@]}" "${o[@]}" --prewarm-ms 0 --work empty,4M --gaps-ms 0,1
run storage_modes      "$B/storage_modes" "${quick[@]}" "${o[@]}" --prewarm-ms 0 --sizes 16K,256K
run cache_interference "$B/cache_interference" "${quick[@]}" "${o[@]}" --prewarm-ms 0 --working-sets 1M --gpu-sizes 16M
run followups          "$B/followups" "${quick[@]}" "${o[@]}" --prewarm-ms 0
run thermal_drift      "$B/thermal_drift" "${o[@]}" --prewarm-ms 0 --minutes 0.05 --no-cooldown
if [[ -x .venv/bin/python ]]; then
  run mlx_matmul .venv/bin/python benchmarks/mlx/mlx_matmul.py --sizes 64 --reps 3 --warmup 1 --min-warmup-ms 0 \
    --prewarm-ms 0 --out "$tmp/runs" --ce-env "$build/benchmarks/ce_env"
fi
"$B/ce_env" > "$tmp/ce_env.json"
mkdir -p "$tmp/runs/ce_env" && cp "$tmp/ce_env.json" "$tmp/runs/ce_env/env.json"

python3 tools/verification/schema_snapshot.py "$out_json" "$tmp"/runs/*/
