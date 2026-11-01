#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
SIM=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
PY=/mnt/jfzn/liujn/robotwin_env/bin/python
OUT="$ROOT/logs/tanh02_pure_full8_node2_20261029"
TASKS=(click_alarmclock beat_block_hammer open_microwave shake_bottle move_playingcard_away pick_dual_bottles handover_block place_object_basket)

while :; do
  completed=0
  for task in "${TASKS[@]}"; do
    for start in 0 20 40 60 80; do
      for mode in pure tanh; do
        path_file="$OUT/$task/$mode/$start/result_path.txt"
        [[ -s "$path_file" ]] || continue
        read -r result_dir <"$path_file"
        if [[ "$result_dir" != /* ]]; then result_dir="$SIM/$result_dir"; fi
        [[ -s "$result_dir/_fixed_seeds.json" ]] && completed=$((completed+1))
      done
    done
  done
  echo "[$(date -Is)] completed_chunks=$completed/80" >>"$OUT/workers/finalize.log"
  if ((completed == 80)); then break; fi
  sleep 30
done
cd "$SIM"
"$PY" "$ROOT/scripts/physical_tokens/summarize_tanh02_pure_full8.py" "$OUT" >"$OUT/workers/final_summary.log"
echo "[$(date -Is)] ALL_DONE 80/80" >>"$OUT/workers/finalize.log"
