#!/usr/bin/env bash
# Four-GPU RoboTwin VLA training on node2.
# Effective global batch = 4 GPUs x PER_DEVICE_BATCH_SIZE x GRADIENT_ACCUMULATION_STEPS.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/jfzn/liujn/miniconda3}"
ENV_NAME="${ENV_NAME:-vlact}"
DATA_ROOT="${DATA_ROOT:-/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin}"
BASE_VLM="${BASE_VLM:-/mnt/jfzn/liujn/code/VLAct-main/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
CONFIG_YAML="${CONFIG_YAML:-examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml}"
DATA_MIX="${DATA_MIX:-robotwin_clean_randomized_50}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${REPO_ROOT}/results/Checkpoints}"
RUN_ID="${RUN_ID:-robotwin_randomized_qwen3oft_bs128_150k_node2_nogc_m16_ga2}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-150000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
WANDB_MODE="${WANDB_MODE:-disabled}"
MASTER_PORT="${MASTER_PORT:-29512}"

if [[ "${1:-}" == "--smoke" ]]; then
  MAX_TRAIN_STEPS="${SMOKE_STEPS:-4}"
  SAVE_INTERVAL="${MAX_TRAIN_STEPS}"
  EVAL_INTERVAL="${MAX_TRAIN_STEPS}"
  RUN_ID="${RUN_ID}_smoke"
fi

cd "${REPO_ROOT}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MASTER_PORT WANDB_MODE TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

if [[ "${NUM_GPUS}" != "4" ]]; then echo "NUM_GPUS must be 4; got ${NUM_GPUS}" >&2; exit 2; fi
IFS=',' read -r -a _GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#_GPU_ARRAY[@]}" != "4" ]]; then echo "GPU_IDS must contain four IDs; got ${GPU_IDS}" >&2; exit 2; fi
for required_path in "${DATA_ROOT}" "${CONFIG_YAML}" "${BASE_VLM}"; do
  if [[ ! -e "${required_path}" ]]; then echo "Required path does not exist: ${required_path}" >&2; exit 3; fi
done
for required_dataset in "adjust_bottle" "Randomized/adjust_bottle"; do
  if [[ ! -d "${DATA_ROOT}/${required_dataset}" ]]; then
    echo "Required Robotwin dataset split is missing: ${DATA_ROOT}/${required_dataset}" >&2
    echo "Prepare the full Clean + Randomized LeRobot tree before using DATA_MIX=${DATA_MIX}." >&2
    exit 3
  fi
done
if [[ -n "${PRETRAINED_CHECKPOINT}" && ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "PRETRAINED_CHECKPOINT does not exist: ${PRETRAINED_CHECKPOINT}" >&2; exit 3
fi
GLOBAL_BATCH=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${GLOBAL_BATCH}" != "128" ]]; then echo "Global batch must be 128; got ${GLOBAL_BATCH}" >&2; exit 2; fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "${BASH_SOURCE[0]}" "${OUTPUT_DIR}/"
echo "Starting RoboTwin VLA: GPUs=${GPU_IDS}, per_device=${PER_DEVICE_BATCH_SIZE}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, global_batch=${GLOBAL_BATCH}"
echo "Data=${DATA_ROOT}  BaseVLM=${BASE_VLM}  Run=${OUTPUT_DIR}"
EXTRA_ARGS=()
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then EXTRA_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"); fi

exec accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" --main_process_port "${MASTER_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" --framework.name QwenOFT \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.enable_gradient_checkpointing false \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" --trainer.eval_interval "${EVAL_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" --run_id "${RUN_ID}" \
  --wandb_project starVLA_Robotwin --wandb_entity "${WANDB_ENTITY:-}" \
  "${EXTRA_ARGS[@]}"
