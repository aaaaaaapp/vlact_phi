#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/jfzn/liujn/code/VLAct-PhysToken
PYTHON="${PHIZERO_PREP_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
DATA_ROOT=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin
EXPERIMENT_DIR="${REPO}/artifacts/robotwin_full50_study/phizero_fsq64_fps16"
OUTPUT_DIR="${EXPERIMENT_DIR}/tokenizer_clips_clean_randomized_fps16"

cd "${REPO}"
exec "${PYTHON}" scripts/physical_tokens/prepare_phizero_robotwin_fps16.py \
  --clean-root "${DATA_ROOT}" \
  --randomized-root "${DATA_ROOT}/Randomized" \
  --tasks docs/robotwin_50task_list.txt \
  --split docs/robotwin_tokenizer_episode_split.json \
  --output-dir "${OUTPUT_DIR}" \
  --workers "${ROBOTWIN_PREP_WORKERS:-4}"
