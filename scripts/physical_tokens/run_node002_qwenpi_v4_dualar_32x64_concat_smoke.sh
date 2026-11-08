#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
if [[ "$(hostname)" != *node002* ]]; then
  echo "This smoke launcher must run on node002" >&2
  exit 2
fi
export PATH="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin:${PATH}"
export PYTHONPATH="${repo}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd "${repo}"

config=scripts/physical_tokens/qwenpi_v4_dualar_32x64_concat_smoke.yaml
source_checkpoint=results/Checkpoints/qwen_dual_ar_32a64p_full50_seed42_steps40000_strict_tied_heads_uniform_physics_init_lora_r32_robotwin_ft_acp_2gpu_fresh_m16_ga4_gc0_correctlr_gradnorm_v2/checkpoints/steps_40000_pytorch_model.pt
for required in "${config}" "${source_checkpoint}" \
  artifacts/robotwin_full50_study/dual_ar_32action_64physics_targets.npz \
  /mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

num_processes=4
micro_batch="${NODE002_MICRO_BATCH_SIZE:-16}"
if (( micro_batch < 1 || micro_batch > 32 || 32 % micro_batch != 0 )); then
  echo "Micro batch must divide 32 and be between 1 and 32" >&2
  exit 2
fi
grad_accum=$((32 / micro_batch))
max_steps="${NODE002_MAX_STEPS:-1}"
if (( max_steps < 1 )); then
  echo "NODE002_MAX_STEPS must be positive" >&2
  exit 2
fi
gradient_checkpointing="${NODE002_GRADIENT_CHECKPOINTING:-1}"
if [[ "${gradient_checkpointing}" != 0 && "${gradient_checkpointing}" != 1 ]]; then
  echo "NODE002_GRADIENT_CHECKPOINTING must be 0 or 1" >&2
  exit 2
fi
run_id="qwenpi_v4_dualar_32x64_fresh40k_concat_m${micro_batch}_ga${grad_accum}_gc${gradient_checkpointing}_steps${max_steps}_smoke"
if [[ -e "results/Checkpoints/${run_id}" ]]; then
  echo "Run directory already exists: results/Checkpoints/${run_id}" >&2
  exit 2
fi

for gpu in 4 5 6 7; do
  free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 60000 ]]; then
    echo "GPU ${gpu} has only ${free_mib:-unknown} MiB free; refusing shared-GPU smoke" >&2
    exit 2
  fi
done
global_batch=$((num_processes * micro_batch * grad_accum))
if [[ "${global_batch}" -ne 128 ]]; then
  echo "Global batch mismatch: ${global_batch}" >&2
  exit 2
fi
echo "node002 GPUs 4-7: 4 x microbatch ${micro_batch} x accumulation ${grad_accum} = global batch 128; Qwen checkpointing ${gradient_checkpointing}"

mkdir -p artifacts/robotwin_full50_study/node002_logs
log="artifacts/robotwin_full50_study/node002_logs/${run_id}.log"
memory_log="artifacts/robotwin_full50_study/node002_logs/${run_id}_gpu_memory.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > "${memory_log}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port 29714 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config}" \
  --run_id "${run_id}" \
  --datasets.vla_data.per_device_batch_size "${micro_batch}" \
  --trainer.gradient_accumulation_steps "${grad_accum}" \
  --trainer.max_train_steps "${max_steps}" \
  --trainer.enable_gradient_checkpointing "${gradient_checkpointing}" \
  --trainer.save_interval "${max_steps}" \
  --trainer.eval_interval "$((max_steps + 1))" 2>&1 | tee -a "${log}"
