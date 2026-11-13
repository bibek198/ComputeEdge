#!/usr/bin/env bash
# One confirmation session: re-runs every experiment behind the paper's claims
# (M4 crossover, M5 Tracks A/B, M6 RQ3/RQ4 and follow-ups) into
# benchmarks/results/sessions/<session>/, then checks the claims.
#
#   benchmarks/run_session.sh [--thermal-load mixed|cpu|gpu] [--thermal-minutes 15] [--allow-dirty]
#
# Run 3-5 sessions on different days, each starting from a cool machine (lid open,
# on AC, Low Power Mode off, other apps closed). Takes about an hour; the 15-minute
# thermal run is always last because it heats the machine.
#
# Guards: the session stops if AC power is lost before a benchmark; thermal_drift
# stops itself if the power source changes mid-run. conditions.csv records power
# and total CPU use of other processes around every benchmark. Avoid video calls
# and heavy apps during a session: they are not detectable here.
#
# Confounds this controls for, session to session:
#   - benchmark order is shuffled (seeded by the session index, recorded in session.json)
#   - sweep orders (sizes, counts, methods) are reversed on odd-numbered sessions
#   - the thermal heater load rotates mixed -> cpu -> gpu unless --thermal-load is given
set -euo pipefail
cd "$(dirname "$0")/.."

thermal_load=""
thermal_minutes=15
allow_dirty=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --thermal-load) thermal_load="$2"; shift 2 ;;
    --thermal-minutes) thermal_minutes="$2"; shift 2 ;;
    --allow-dirty) allow_dirty=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$(git status --porcelain)" && $allow_dirty -eq 0 ]]; then
  echo "error: uncommitted changes; commit first so results carry a clean git SHA (or --allow-dirty)" >&2
  exit 1
fi
pmset -g batt | head -1 | grep -q "AC Power" || echo "warning: not on AC power" >&2
[[ -x .venv/bin/python ]] || { echo "error: .venv missing (python3 -m venv .venv && .venv/bin/pip install -r benchmarks/requirements.txt)" >&2; exit 1; }

cmake --preset release >/dev/null   # refreshes the recorded git SHA
cmake --build --preset release >/dev/null

sessions_dir=benchmarks/results/sessions
mkdir -p "$sessions_dir"
index=$(find "$sessions_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
session="$(date +%Y%m%d-%H%M%S)_s${index}_$(git rev-parse --short=12 HEAD)"
out="$sessions_dir/$session"
mkdir -p "$out/logs"
loads=(mixed cpu gpu)
[[ -n "$thermal_load" ]] || thermal_load=${loads[$((index % 3))]}
reverse=$((index % 2))

# rev "a,b,c" -> "c,b,a" on odd sessions
ord() { if [[ $reverse -eq 1 ]]; then echo "$1" | tr ',' '\n' | tail -r | paste -sd, -; else echo "$1"; fi; }
B=build/release/benchmarks

methods=$(.venv/bin/python -c "import random; m=['wait_until_completed','event_wait','completion_handler','spin_event','spin_status']; random.Random($index).shuffle(m); print(','.join(m))")
declare -a names cmds
names+=(cpu_matmul);   cmds+=("$B/cpu_matmul --cooldown --impls accelerate --modes e2e,kernel --sizes $(ord 128,512,1024,2048,4096) --out $out")
names+=(metal_matmul); cmds+=("$B/metal_matmul --cooldown --kernels mps --storage $(ord nocopy,shared) --sizes $(ord 128,512,1024,2048,4096) --out $out")
names+=(mlx_matmul);   cmds+=(".venv/bin/python benchmarks/mlx/mlx_matmul.py --cooldown --sizes $(ord 128,512,1024,2048,4096) --out $out")
names+=(dispatch);     cmds+=("$B/dispatch_overhead --cooldown --dispatches $(ord 1,4,16,64,256,1024) --encoders $(ord 1,4,16,64,256) --cmdbufs $(ord 1,2,4,8,16,32,64) --out $out")
names+=(sync);         cmds+=("$B/sync_methods --cooldown --methods $methods --gaps-ms $(ord 0,0.1,1,10,100) --out $out")
names+=(storage);      cmds+=("$B/storage_modes --cooldown --experiments create,gpu_first,gpu_bw --sizes $(ord 16K,256K,4M,64M,256M,1G) --out $out")
names+=(cache);        cmds+=("$B/cache_interference --cooldown --working-sets $(ord 1M,4M,12M,64M) --gpu-sizes $(ord 16M,64M,256M) --out $out")
names+=(followups);    cmds+=("$B/followups --cooldown --out $out")

order=$(.venv/bin/python -c "import random; o=list(range(${#names[@]})); random.Random(1000+$index).shuffle(o); print(' '.join(map(str,o)))")
run_order=()
for i in $order; do run_order+=("${names[$i]}"); done
run_order+=(thermal_drift)

# Conditions before/after every benchmark: power source and total CPU use of all
# other processes (no process names are recorded). A session that loses AC power
# stops; its results stay on disk but are marked in session.json.
echo "benchmark,phase,time_local,power,other_cpu_pct" > "$out/conditions.csv"
power() { pmset -g batt | head -1 | sed -E "s/.*'(.*)'.*/\1/"; }
other_cpu() { ps -A -o %cpu= | awk '{s += $1} END {printf "%.0f", s}'; }
note() { echo "$1,$2,$(date +%H:%M:%S),$(power),$(other_cpu)" >> "$out/conditions.csv"; }
status="complete"
check_ac() {
  if [[ "$(power)" != "AC Power" ]]; then
    echo "error: not on AC power before $1; stopping the session" >&2
    status="aborted: lost AC power before $1"
    return 1
  fi
}

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "session $session: order ${run_order[*]}; reversed sweeps=$reverse; thermal load=$thermal_load"
for i in $order; do
  check_ac "${names[$i]}" || break
  note "${names[$i]}" start
  echo "[$(date +%H:%M:%S)] ${names[$i]}"
  eval "${cmds[$i]}" > "$out/logs/${names[$i]}.log" 2>&1
  note "${names[$i]}" end
done
if [[ "$status" == "complete" ]] && check_ac thermal_drift; then
  note thermal_drift start
  echo "[$(date +%H:%M:%S)] thermal_drift ($thermal_minutes min, $thermal_load)"
  if ! $B/thermal_drift --minutes "$thermal_minutes" --load "$thermal_load" --out "$out" \
      > "$out/logs/thermal_drift.log" 2>&1; then
    status="thermal_drift invalid: power source changed during the run"
    echo "warning: $status" >&2
  fi
  note thermal_drift end
fi

cat > "$out/session.json" <<JSON
{
  "session": "$session",
  "index": $index,
  "git_sha": "$(git rev-parse HEAD)",
  "started_utc": "$started",
  "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "order": "$(echo "${run_order[*]}")",
  "reversed_sweeps": $reverse,
  "sync_method_order": "$methods",
  "thermal_load": "$thermal_load",
  "thermal_minutes": $thermal_minutes,
  "status": "$status",
  "uptime": "$(uptime | sed 's/.*up \([^,]*\),.*/\1/')",
  "process_count": $(ps -A | wc -l | tr -d ' ')
}
JSON
echo "session written to $out"
.venv/bin/python benchmarks/analysis/confirm_claims.py
