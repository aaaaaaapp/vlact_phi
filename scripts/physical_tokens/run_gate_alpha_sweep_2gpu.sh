#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
SRC=$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000
SEEDS='[100100001,100100002,100100003,100100004,100100005,100100006,100100007,100100008,100100009,100100010]'
run_pair(){ local a=$1 b=$2; local pids=(); for spec in "0 $a 7010" "1 $b 7011"; do read -r gpu alpha port <<< "$spec"; tag=${alpha/./p}; run=qwenpi_v4_p64_m32_gate_tanh${tag}_derived; out=$ROOT/results/Checkpoints/$run; mkdir -p "$out/checkpoints"; "$PY" "$ROOT/scripts/physical_tokens/bound_gate_checkpoint.py" --input "$SRC/checkpoints/steps_20000_pytorch_model.pt" --output "$out/checkpoints/steps_20000_pytorch_model.pt" --mode tanh --alpha-max "$alpha"; cp "$SRC/config.yaml" "$out/config.yaml"; cp "$SRC/dataset_statistics.json" "$out/dataset_statistics.json"; GPU_ID=$gpu PORT=$port TASK=click_bell TAG=gate_tanh_${tag}_2gpu CKPT="$out/checkpoints/steps_20000_pytorch_model.pt" SEEDS="$SEEDS" N=10 nohup setsid bash "$ROOT/scripts/physical_tokens/run_one_last_four_rollout.sh" >"$ROOT/logs/gate_tanh_${tag}_2gpu_driver.log" 2>&1 </dev/null & pids+=("$!"); done; for p in "${pids[@]}"; do wait "$p"; done; }
run_pair 0.05 0.10
run_pair 0.20 0.50
