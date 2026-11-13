#!/usr/bin/env bash
# M4: run every baseline in one session, then build the comparison table and figures.
#
#   benchmarks/run_baselines.sh [extra args passed to every benchmark, e.g. --sizes 128,1024]
#
# Requirements: AC power, a committed working tree (so results carry a clean git SHA),
# and the Python environment: python3 -m venv .venv && .venv/bin/pip install -r benchmarks/requirements.txt
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain)" ]]; then
  echo "warning: working tree has uncommitted changes; results will be tagged -dirty" >&2
fi
if ! pmset -g batt | head -1 | grep -q "AC Power"; then
  echo "warning: not on AC power" >&2
fi
[[ -x .venv/bin/python ]] || { echo "error: .venv missing (see header)" >&2; exit 1; }

cmake --preset release >/dev/null          # refreshes the recorded git SHA
cmake --build --preset release >/dev/null

./build/release/benchmarks/cpu_matmul --cooldown --impls accelerate "$@"
./build/release/benchmarks/metal_matmul --cooldown "$@"
.venv/bin/python benchmarks/mlx/mlx_matmul.py --cooldown "$@"
.venv/bin/python benchmarks/analysis/compare_baselines.py
