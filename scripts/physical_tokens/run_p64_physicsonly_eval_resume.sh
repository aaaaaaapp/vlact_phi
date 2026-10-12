#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
cd "$repo"
python_bin=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
ckpt="$repo/results/Checkpoints/qwenpi_v4_p64_physicsonly_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
cache="$repo/artifacts/token_caches/robotwin_p64_predicted_action4_physics64_full_step20000.npz"
out="$repo/artifacts/evaluations/p64_physicsonly_64"
config="$repo/examples/Robotwin/train_files/qwenpi_v4_p64_m32_eval.yaml"
run_group() { local gpu="$1"; shift; for task in "$@"; do
  echo "[$(date -Is)] gpu=$gpu task=$task" >&2
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/physical_tokens/evaluate_joint_token_predictions.py --config "$config" --checkpoint "$ckpt" --token-cache "$cache" --task-name "$task" --num-samples 64 --device cuda:0 --output "$out/${task}.json" > "$out/${task}.log" 2>&1
done; }
run_group 0 beat_block_hammer move_playingcard_away pick_dual_bottles & p0=$!
run_group 1 handover_block open_microwave place_object_basket & p1=$!
run_group 4 shake_bottle & p4=$!
wait "$p0"; wait "$p1"; wait "$p4"
