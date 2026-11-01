#!/usr/bin/env bash
# One A100-80GB, global batch 128; vary micro batch and checkpointing for tuning.
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
export PATH="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin:${PATH}"
export PYTHONPATH="${repo}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:/mnt/jfzn/liujn/code/VLAct-main/.acp_deps:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONDONTWRITEBYTECODE=1
export DUAL_AR_ACTION_TOKEN_COUNT=32
export DUAL_AR_CONFIG=scripts/physical_tokens/qwen_strict_lora_r32_32action_64physics_full50_robotwin_finetune.yaml
export DUAL_AR_LORA=1
export DUAL_AR_LORA_RANK=32
export DUAL_AR_LORA_ALPHA=32
export DUAL_AR_UNIFORM_PHYSICS_INIT=1
export DUAL_AR_NUM_PROCESSES=1
export DUAL_AR_MICRO_BATCH_SIZE="${DUAL_AR_MICRO_BATCH_SIZE:-16}"
export DUAL_AR_GRADIENT_CHECKPOINTING="${DUAL_AR_GRADIENT_CHECKPOINTING:-0}"
if ! [[ "${DUAL_AR_MICRO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( 128 % DUAL_AR_MICRO_BATCH_SIZE != 0 )); then
  echo "Single-GPU micro batch must be a positive divisor of 128" >&2
  exit 2
fi
if [[ "${DUAL_AR_GRADIENT_CHECKPOINTING}" != "0" && "${DUAL_AR_GRADIENT_CHECKPOINTING}" != "1" ]]; then
  echo "DUAL_AR_GRADIENT_CHECKPOINTING must be 0 or 1" >&2
  exit 2
fi
export DUAL_AR_GRAD_ACCUM_STEPS="$((128 / DUAL_AR_MICRO_BATCH_SIZE))"
export DUAL_AR_GLOBAL_BATCH_SIZE=128
export DUAL_AR_MAX_STEPS="${DUAL_AR_MAX_STEPS:-20000}"
export DUAL_AR_WARMUP_STEPS="${DUAL_AR_WARMUP_STEPS:-500}"
export DUAL_AR_SAVE_INTERVAL="${DUAL_AR_SAVE_INTERVAL:-1000}"
export DUAL_AR_MAIN_PROCESS_PORT="${DUAL_AR_MAIN_PROCESS_PORT:-29685}"
export DUAL_AR_RUN_TAG="${DUAL_AR_RUN_TAG:-robotwin_ft_acp_1gpu_bs128_r32_32action_m${DUAL_AR_MICRO_BATCH_SIZE}_gc${DUAL_AR_GRADIENT_CHECKPOINTING}_v1}"
export DUAL_AR_EVAL_ALL_TASKS="${DUAL_AR_EVAL_ALL_TASKS:-1}"
export DUAL_AR_EVAL_SAMPLES_PER_TASK="${DUAL_AR_EVAL_SAMPLES_PER_TASK:-16}"
export DUAL_AR_EVAL_FREE_SAMPLES="${DUAL_AR_EVAL_FREE_SAMPLES:-1}"
export DUAL_AR_EVAL_GPU=0

cd "${repo}"
mkdir -p artifacts/robotwin_full50_study/acp_logs
log_file="${DUAL_AR_LOG_FILE:-artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m${DUAL_AR_MICRO_BATCH_SIZE}_gc${DUAL_AR_GRADIENT_CHECKPOINTING}_bs128_r32_v1.log}"
echo "ACP one A100-80GB: micro ${DUAL_AR_MICRO_BATCH_SIZE} x accumulation ${DUAL_AR_GRAD_ACCUM_STEPS} = global batch 128; gradient checkpointing ${DUAL_AR_GRADIENT_CHECKPOINTING}; max steps ${DUAL_AR_MAX_STEPS}"
echo "Python: $(command -v python)"
echo "Persistent log: ${repo}/${log_file}"
bash scripts/physical_tokens/run_full50_dual_autoregressive_reasoner.sh 2>&1 | tee -a "${log_file}"
