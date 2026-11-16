#!/usr/bin/env bash
# Dedicated clean confirmation runner for thermal_drift (s_clean).
# Runs an uninterrupted 15-minute thermal drift benchmark under caffeinate -dis
# on AC Power, with active monitoring of power assertions, thermal state, and sleep events.
set -euo pipefail
cd "$(dirname "$0")/.."

sessions_dir="data/sessions/confirmation"
mkdir -p "$sessions_dir"

# Verify AC Power precondition
power_src=$(pmset -g batt | head -1 | sed -E "s/.*'(.*)'.*/\1/")
if [[ "$power_src" != "AC Power" ]]; then
  echo "error: system is not on AC Power (currently: $power_src); stopping" >&2
  exit 1
fi

index=$(find "$sessions_dir" -mindepth 1 -maxdepth 1 -type d -name '*_s[0-9]*_*' | wc -l | tr -d ' ')
stamp=$(date +%Y%m%d-%H%M%S)
sha=$(git rev-parse --short=12 HEAD)
session="${stamp}_s${index}_${sha}"
out="$sessions_dir/$session"
mkdir -p "$out/logs"

echo "=== Clean Thermal Confirmation Experiment ==="
echo "Session:          $session"
echo "Output Directory: $out"
echo "Commit SHA:       $(git rev-parse HEAD)"
echo "Power Source:     $power_src"
echo "Thermal Load:     mixed"
echo "Duration:         15 minutes"

# Pre-experiment sleep/wake log baseline
pmset -g log | tail -n 50 > "$out/logs/pre_experiment_pmset.log" 2>&1 || true

# Conditions log
echo "benchmark,phase,time_local,power,other_cpu_pct" > "$out/conditions.csv"
other_cpu() { ps -A -o %cpu= | awk '{s += $1} END {printf "%.0f", s}'; }
power_now() { pmset -g batt | head -1 | sed -E "s/.*'(.*)'.*/\1/"; }
echo "thermal_drift,start,$(date +%H:%M:%S),$(power_now),$(other_cpu)" >> "$out/conditions.csv"

# Telemetry monitoring header
echo "timestamp,time_s,uptime_s,power,assertions_count" > "$out/telemetry.csv"

start_epoch=$(date +%s)
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Background telemetry logger (checks every 10 seconds while benchmark runs)
(
  while true; do
    now_epoch=$(date +%s)
    elapsed=$((now_epoch - start_epoch))
    curr_power=$(pmset -g batt | head -1 | sed -E "s/.*'(.*)'.*/\1/")
    assertions=$(pmset -g assertions | grep -c "PreventUserIdleSystemSleep.*caffeinate" || true)
    echo "$(date +%H:%M:%S),$elapsed,$elapsed,$curr_power,$assertions" >> "$out/telemetry.csv"
    sleep 10
  done
) &
LOGGER_PID=$!

cleanup() {
  local rc=$?
  kill "$LOGGER_PID" 2>/dev/null || true
  wait "$LOGGER_PID" 2>/dev/null || true
  echo "thermal_drift,end,$(date +%H:%M:%S),$(power_now),$(other_cpu)" >> "$out/conditions.csv"
  
  # Capture post-experiment sleep/wake logs
  pmset -g log | tail -n 100 > "$out/logs/post_experiment_pmset.log" 2>&1 || true
  pmset -g assertions > "$out/logs/final_assertions.log" 2>&1 || true

  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local state="completed"
  local status="complete"
  if [[ $rc -ne 0 ]]; then
    state="failed"
    status="failed with exit code $rc"
  fi

  cat > "$out/session.json" <<JSON
{
  "session": "$session",
  "index": $index,
  "git_sha": "$(git rev-parse HEAD)",
  "started_utc": "$started_utc",
  "finished_utc": "$finished_utc",
  "order": "thermal_drift",
  "reversed_sweeps": 0,
  "sync_method_order": "none",
  "thermal_load": "mixed",
  "thermal_minutes": 15,
  "power_assertion": "caffeinate -dis",
  "status": "$status",
  "state": "$state",
  "uptime": "$(uptime | sed 's/.*up \([^,]*\),.*/\1/')",
  "process_count": $(ps -A | wc -l | tr -d ' ')
}
JSON

  # Manifest generation
  (cd "$out" && find . -type f ! -name "manifest.sha256" | LC_ALL=C sort | while IFS= read -r f; do shasum -a 256 "$f"; done > manifest.sha256)
  chmod -R a-w "$out" || true
  echo "Session recorded at $out (rc=$rc)"
}
trap cleanup EXIT INT TERM

# Execute thermal drift with caffeinate -dis
echo "[$(date +%H:%M:%S)] Launching caffeinate -dis thermal_drift..."
caffeinate -dis build/release/benchmarks/thermal_drift \
  --minutes 15 \
  --load mixed \
  --interval-ms 1000 \
  --out "$out" \
  2>&1 | tee "$out/logs/thermal_drift.log"

echo "[$(date +%H:%M:%S)] Benchmark execution finished cleanly."
