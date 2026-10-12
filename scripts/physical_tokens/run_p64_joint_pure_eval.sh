#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/jfzn/liujn/code/VLAct-PhysToken; cd "$repo"
py=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
ckpt="$repo/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
config="$repo/examples/Robotwin/train_files/qwenpi_v4_p64_m32_eval.yaml"
out="$repo/artifacts/evaluations/p64_joint_pure_64"; mkdir -p "$out"
tasks=(beat_block_hammer click_alarmclock handover_block move_playingcard_away open_microwave pick_dual_bottles place_object_basket shake_bottle)
pids=(); for i in "${!tasks[@]}"; do t="${tasks[$i]}"; CUDA_VISIBLE_DEVICES="$i" "$py" scripts/physical_tokens/evaluate_qwenpi_oracle.py --config "$config" --checkpoint "$ckpt" --mode e0 --task-name "$t" --num-samples 64 --seed 42 --device cuda:0 --output "$out/$t.json" > "$out/$t.log" 2>&1 & pids+=("$!"); done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done; exit "$status"
