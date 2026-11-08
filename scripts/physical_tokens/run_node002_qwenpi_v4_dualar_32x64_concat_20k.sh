#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
run_id=qwenpi_v4_dualar_32x64_fresh40k_concat_m8_ga4_gc0_steps20000_v1
if [[ "$(hostname)" != *node002* ]]; then
  echo "This launcher must run on node002" >&2
  exit 2
fi
for gpu in 4 5 6 7; do
  free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 60000 ]]; then
    echo "GPU ${gpu} has only ${free_mib:-unknown} MiB free; refusing shared-GPU training" >&2
    exit 2
  fi
done
if [[ -e "${repo}/results/Checkpoints/${run_id}" ]]; then
  echo "Run directory already exists; refusing to overwrite ${run_id}" >&2
  exit 2
fi

export PATH="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin:${PATH}"
export PYTHONPATH="${repo}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd "${repo}"

config=scripts/physical_tokens/qwenpi_v4_dualar_32x64_concat_20k.yaml
source_checkpoint=results/Checkpoints/qwen_dual_ar_32a64p_full50_seed42_steps40000_strict_tied_heads_uniform_physics_init_lora_r32_robotwin_ft_acp_2gpu_fresh_m16_ga4_gc0_correctlr_gradnorm_v2/checkpoints/steps_40000_pytorch_model.pt
for required in "${config}" "${source_checkpoint}" \
  artifacts/robotwin_full50_study/dual_ar_32action_64physics_targets.npz \
  /mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

mkdir -p artifacts/robotwin_full50_study/node002_logs
log="artifacts/robotwin_full50_study/node002_logs/${run_id}.log"
memory_log="artifacts/robotwin_full50_study/node002_logs/${run_id}_gpu_memory.csv"
status_file="artifacts/robotwin_full50_study/node002_logs/${run_id}.status"
printf 'RUNNING %s\n' "$(date -Is)" > "${status_file}"
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 10 > "${memory_log}" &
monitor_pid=$!
finish() {
  local rc=$?
  kill "${monitor_pid}" 2>/dev/null || true
  printf 'EXIT %s %s\n' "${rc}" "$(date -Is)" > "${status_file}"
}
trap finish EXIT

echo "node002 GPUs 4-7: 4 x microbatch 8 x accumulation 4 = global batch 128; 20000 steps; Qwen checkpointing off"
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29716 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config}" 2>&1 | tee -a "${log}"
