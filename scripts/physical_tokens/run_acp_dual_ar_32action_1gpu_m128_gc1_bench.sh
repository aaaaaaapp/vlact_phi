#!/usr/bin/env bash
# Short ACP benchmark: one A100-80GB, 128 x 1 = global batch 128, checkpointing on.
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
export DUAL_AR_MICRO_BATCH_SIZE=128
export DUAL_AR_GRADIENT_CHECKPOINTING=1
export DUAL_AR_MAX_STEPS=20
export DUAL_AR_SAVE_INTERVAL=1000
export DUAL_AR_EVAL_ALL_TASKS=0
export DUAL_AR_EVAL_SAMPLES_PER_TASK=1
export DUAL_AR_EVAL_FREE_SAMPLES=0
export DUAL_AR_MAIN_PROCESS_PORT=29689

memory_log="${repo}/artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m128_gc1_bs128_gpu_memory.csv"
mkdir -p "$(dirname "${memory_log}")"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 1 > "${memory_log}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

bash "${repo}/scripts/physical_tokens/run_acp_dual_ar_32action_1gpu_tuned_20k.sh"
