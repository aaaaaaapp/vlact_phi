#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
data_root=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin
study="${repo}/artifacts/robotwin_full50_study"
prepared="${study}/clean_randomized_32x64_v1"
tmp_root=/tmp/robotwin_randomized_32x64_v1
run_id=qwenpi_v4_dualar_32x64_clean_randomized20k_m8_ga4_v1
python=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
action_ckpt="${study}/action_tokenizers/ema_32x525_decay095_dead100_bs128_steps20000/step-00020000.pt"
physics_ckpt="${study}/physics_tokenizers/ema_t64_k2048_steps20000/step-00020000.pt"
visual_ckpt=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt
clean_cache="${study}/dual_ar_32action_64physics_targets.npz"
combined="${prepared}/clean_randomized_dual_targets.npz"
task_list="${repo}/docs/robotwin_50task_list.txt"
log_dir="${study}/node002_logs"
status="${log_dir}/${run_id}.status"

if [[ "$(hostname)" != *node002* ]]; then
  echo 'Run only on node002' >&2
  exit 2
fi
export PATH="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin:${PATH}"
export PYTHONPATH="${repo}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
cd "${repo}"
mapfile -t tasks < "${task_list}"
if [[ "${#tasks[@]}" -ne 50 ]]; then
  echo 'Expected 50 task names' >&2
  exit 2
fi
for required in "${action_ckpt}" "${physics_ckpt}" "${visual_ckpt}" "${clean_cache}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing input ${required}" >&2
    exit 2
  fi
done
mkdir -p "${prepared}/action_ids" "${prepared}/physics_ids" "${prepared}/dual" \
  "${prepared}/logs" "${tmp_root}/actions" "${tmp_root}/features" "${log_dir}"

worker() {
  local shard="$1" gpu task i action_raw action_ids feature physics_ids dual task_log
  gpu="$((4 + shard))"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  for i in "${!tasks[@]}"; do
    if (( i % 4 != shard )); then continue; fi
    task="${tasks[$i]}"
    action_raw="${tmp_root}/actions/${task}.npz"
    action_ids="${prepared}/action_ids/${task}.npz"
    feature="${tmp_root}/features/${task}.npz"
    physics_ids="${prepared}/physics_ids/${task}.npz"
    dual="${prepared}/dual/${task}.npz"
    task_log="${prepared}/logs/${task}.log"
    if [[ -s "${dual}" && -s "${dual}.manifest.json" ]]; then
      echo "READY ${task}"; continue
    fi
    echo "START ${task} GPU ${gpu} $(date -Is)" | tee -a "${task_log}"
    if [[ ! -s "${action_ids}" || ! -s "${action_ids}.manifest.json" ]]; then
      if [[ ! -s "${action_raw}" || ! -s "${action_raw}.manifest.json" ]]; then
        "${python}" -u scripts/physical_tokens/export_robotwin_action_cache.py \
          --data-root "${data_root}" --data-name "Randomized/${task}" \
          --robot-type robotwin_wrap32 --output "${action_raw}" --stride 4 >>"${task_log}" 2>&1
      fi
      "${python}" scripts/physical_tokens/export_action_token_ids.py \
        --input "${action_raw}" --checkpoint "${action_ckpt}" --output "${action_ids}" \
        --batch-size 512 --num-workers 0 --device cuda:0 >>"${task_log}" 2>&1
    fi
    if [[ ! -s "${feature}" || ! -s "${feature}.manifest.json" ]]; then
      "${python}" -u scripts/physical_tokens/export_robotwin_spatiotemporal_features.py \
        --data-root "${data_root}" --data-name "Randomized/${task}" \
        --robot-type robotwin_wrap32 --camera video.cam_high \
        --offsets 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32 \
        --spatial-grid-size 4 --base-vlm ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
        --pretrained-checkpoint "${visual_ckpt}" \
        --pretrained-checkpoint-sha256 5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d \
        --output "${feature}" --stride 4 --batch-size 16 --device cuda:0 >>"${task_log}" 2>&1
    fi
    if [[ ! -s "${physics_ids}" || ! -s "${physics_ids}.manifest.json" ]]; then
      "${python}" scripts/physical_tokens/export_spatiotemporal_physics_token_ids.py \
        --input "${feature}" --checkpoint "${physics_ckpt}" --output "${physics_ids}" \
        --batch-size 128 --num-workers 0 --device cuda:0 >>"${task_log}" 2>&1
    fi
    "${python}" scripts/physical_tokens/merge_robotwin_clean_randomized_dual_targets.py join-task \
      --task "${task}" --action "${action_ids}" --physics "${physics_ids}" \
      --feature "${feature}" --output "${dual}" >>"${task_log}" 2>&1
    if [[ ! -s "${dual}" || ! -s "${dual}.manifest.json" ]]; then
      echo "Invalid dual target output for ${task}" >&2; return 1
    fi
    # These two generated caches are reproducible; retain their manifests in the task log.
    rm -f -- "${action_raw}" "${action_raw}.manifest.json" "${feature}" "${feature}.manifest.json"
    echo "READY ${task} $(date -Is)" | tee -a "${task_log}"
  done
}

if [[ "${1:-}" == worker ]]; then
  worker "${2:?shard required}"
  exit 0
fi
if [[ "${1:-}" != orchestrate ]]; then
  echo 'Usage: run_node002_robotwin_clean_randomized_20k.sh orchestrate|worker SHARD' >&2
  exit 2
fi
if [[ -e "${repo}/results/Checkpoints/${run_id}" ]]; then
  echo "Refusing to reuse training output ${run_id}" >&2
  exit 2
fi
printf 'PREPARING %s\n' "$(date -Is)" > "${status}"
finish() {
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then printf 'FAILED %s %s\n' "${rc}" "$(date -Is)" > "${status}"; fi
}
trap finish EXIT
for gpu in 4 5 6 7; do
  free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 60000 ]]; then
    echo "GPU ${gpu} has only ${free_mib:-unknown} MiB free" >&2; exit 2
  fi
done
pids=()
for shard in 0 1 2 3; do
  bash "$0" worker "${shard}" >"${prepared}/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one randomized cache worker failed; see ${prepared}/logs" >&2
  exit 1
fi
if [[ ! -s "${combined}" || ! -s "${combined}.manifest.json" ]]; then
  "${python}" scripts/physical_tokens/merge_robotwin_clean_randomized_dual_targets.py combine \
    --clean "${clean_cache}" --random-dir "${prepared}/dual" --task-list "${task_list}" \
    --output "${combined}"
fi
for gpu in 4 5 6 7; do
  free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 60000 ]]; then
    echo "GPU ${gpu} has only ${free_mib:-unknown} MiB free before training" >&2; exit 2
  fi
done
printf 'TRAINING %s\n' "$(date -Is)" > "${status}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29717 \
  starVLA/training/train_starvla.py \
  --config_yaml scripts/physical_tokens/qwenpi_v4_dualar_32x64_concat_20k.yaml \
  --run_id "${run_id}" \
  --datasets.vla_data.data_mix robotwin_clean_randomized_wrap_32 \
  --datasets.vla_data.discrete_token_cache "${combined}"
printf 'COMPLETE %s\n' "$(date -Is)" > "${status}"
