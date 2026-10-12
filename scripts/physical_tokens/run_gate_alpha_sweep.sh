#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
SRC=$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000
SEEDS='[100100001,100100002,100100003,100100004,100100005,100100006,100100007,100100008,100100009,100100010]'
PIDS=()
for spec in '0 0.05 7010' '1 0.10 7011' '2 0.20 7012' '3 0.50 7013'; do
  read -r gpu alpha port <<< "$spec"; tag=${alpha/./p}; run=qwenpi_v4_p64_m32_gate_tanh${tag}_derived
  out=$ROOT/results/Checkpoints/$run
  mkdir -p "$out/checkpoints"
  "$PY" "$ROOT/scripts/physical_tokens/bound_gate_checkpoint.py" --input "$SRC/checkpoints/steps_20000_pytorch_model.pt" --output "$out/checkpoints/steps_20000_pytorch_model.pt" --mode tanh --alpha-max "$alpha"
  cp "$SRC/config.yaml" "$out/config.yaml"; cp "$SRC/dataset_statistics.json" "$out/dataset_statistics.json"
  GPU_ID=$gpu PORT=$port TASK=click_bell TAG=gate_tanh_${tag}_sweep CKPT="$out/checkpoints/steps_20000_pytorch_model.pt" SEEDS="$SEEDS" N=10 \
    nohup setsid bash "$ROOT/scripts/physical_tokens/run_one_last_four_rollout.sh" >"$ROOT/logs/gate_tanh_${tag}_sweep_driver.log" 2>&1 </dev/null & PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
