#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/jfzn/liujn/code/VLAct-PhysToken; cd "$repo"
py=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
tasks=(beat_block_hammer click_alarmclock handover_block move_playingcard_away open_microwave pick_dual_bottles place_object_basket shake_bottle)
run_panel() { local tag="$1"; local config="$2"; local ckpt="$3"; local out="$4"; mkdir -p "$out"; local ps=(); for i in "${!tasks[@]}"; do t="${tasks[$i]}"; CUDA_VISIBLE_DEVICES="$i" "$py" scripts/physical_tokens/evaluate_qwenpi_oracle.py --config "$config" --checkpoint "$ckpt" --mode e0 --task-name "$t" --num-samples 64 --seed 42 --device cuda:0 --output "$out/$t.json" > "$out/$t.log" 2>&1 & ps+=("$!"); done; local s=0; for p in "${ps[@]}"; do wait "$p" || s=1; done; return "$s"; }
run_panel p64_joint_e0 "$repo/examples/Robotwin/train_files/qwenpi_v4_p64_m32_eval.yaml" "$repo/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt" "$repo/artifacts/evaluations/p64_joint_e0_64"
run_panel d50_pure_e0 "$repo/results/Checkpoints/qwenpi_v4_d50_frozen_policy_predicted_seed42_steps500/config.yaml" "$repo/results/Checkpoints/qwenpi_v4_d50_frozen_policy_predicted_seed42_steps500/checkpoints/steps_500_pytorch_model.pt" "$repo/artifacts/evaluations/d50_pure_e0_64"
