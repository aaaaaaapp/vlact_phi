#!/usr/bin/env bash
set -euo pipefail

# Persistent continuation guard. It waits for the current pure-Pi repeats,
# writes a completion marker, and keeps a periodic heartbeat so a detached
# process never silently becomes idle.
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
LOG=${1:-$ROOT/logs/experiment_chain_node3_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG"
echo "[$(date -Is)] chain started" | tee "$LOG/chain.log"
while pgrep -f '[r]un_matched_pure_pi_repeats_node3.sh' >/dev/null || pgrep -f '[e]val_policy_fixed_seeds.py' >/dev/null; do
  echo "[$(date -Is)] active eval processes present" >> "$LOG/chain.log"
  sleep 300
done
echo "[$(date -Is)] pure repeats complete; manifests are ready for aggregation" | tee -a "$LOG/chain.log"
touch "$LOG/PURE_REPEATS_COMPLETE"
# Keep the guard alive with a heartbeat for post-hoc monitoring.
while true; do
  echo "[$(date -Is)] chain idle after completed queue; no unfinished authorized phase" >> "$LOG/chain.log"
  sleep 1800
done
