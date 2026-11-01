#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
max_wait_minutes="${ROBOTWIN_PHYSICS_MERGE_WAIT_MINUTES:-360}"

if [[ "$(hostname)" != *node002* ]]; then
  echo "This runner must execute on node002, which owns its local feature array" >&2
  exit 2
fi
for ((minute=0; minute<max_wait_minutes; minute++)); do
  if [[ -s "${cache}" && -s "${cache}.manifest.json" ]]; then
    break
  fi
  if (( minute % 10 == 0 )); then
    echo "$(date -Is) waiting for merged physics cache (${minute}/${max_wait_minutes} min)"
  fi
  sleep 60
done
if [[ ! -s "${cache}" || ! -s "${cache}.manifest.json" ]]; then
  echo "Timed out waiting for merged physics cache" >&2
  exit 2
fi
"${python_bin}" -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8")); assert m["num_tasks"]==50 and m["exact_action_key_alignment"] is True' "${cache}.manifest.json"

echo "$(date -Is) preparing node002-local read-only feature array"
frame_array="$(bash scripts/physical_tokens/prepare_full50_physics_frame_array.sh)"
export ROBOTWIN_FULL50_FRAME_FEATURES_NPY="${frame_array}"
echo "$(date -Is) array=${frame_array}"

echo "$(date -Is) starting three EMA/FSQ physical baselines"
ROBOTWIN_PHYSICS_TRAIN_GPUS=0,1,2 \
  bash scripts/physical_tokens/run_full50_physics_tokenizer_baselines.sh
echo "$(date -Is) starting six paired FSQ/iFSQ physical runs"
ROBOTWIN_PHYSICS_IFSQ_GPUS=0,1,2,3,4,5 \
  bash scripts/physical_tokens/run_full50_physics_ifsq_study.sh
echo "$(date -Is) evaluating all nine physical checkpoints"
ROBOTWIN_PHYSICS_EVAL_GPU=6 \
  bash scripts/physical_tokens/run_full50_physics_evaluation.sh
echo "$(date -Is) complete"
