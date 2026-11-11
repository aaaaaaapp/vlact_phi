#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/mnt/jfzn/liujn/code/VLAct-PhysToken
PYTHON=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
ACCELERATE=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/accelerate
EXPERIMENT_DIR="${REPO}/artifacts/robotwin_full50_study/phizero_fsq64_fps16"
TARGET_CACHE="${EXPERIMENT_DIR}/dual_ar_32action_64physics_k25000_clean_randomized_padcap8_short2_targets.npz"
TARGET_MANIFEST="${TARGET_CACHE}.manifest.json"
CONFIG="${REPO}/scripts/physical_tokens/qwen_dual_ar_32a64p_phizero_fsq64_robotwin_10k.yaml"
TOKENIZER_CKPT="${EXPERIMENT_DIR}/tokenizer_fsq64_gbs16_steps10k/run_20261109_180333/step-10000.safetensors"
ACTION_CKPT="${EXPERIMENT_DIR}/action_tokenizers/ema_32x525_decay095_dead100_bs128_steps20000/step-00020000.pt"
BASE_CKPT=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
DATA_ROOT=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin
RAM_CLIPS=/dev/shm/liujn/vlact_robotwin_phizero_fsq64/tokenizer_clips_clean_randomized_fps16
NODE4=10.120.154.142
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="qwen_dual_ar_32a64p_phizero_fsq64_k25000_robotwin_clean_randomized_fps16_padcap8_short2_node2_${STAMP}"
LOG_DIR="${REPO}/artifacts/robotwin_full50_study/node002_logs"
LOG="${LOG_DIR}/${RUN_ID}_orchestrator.log"
STATUS="${LOG_DIR}/${RUN_ID}.status"
LOCK="${EXPERIMENT_DIR}/.robotwin_fsq64_dual_ar_node2.lock"
GPU_IDS="${ROBOTWIN_DUAL_AR_GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="$(awk -F, '{print NF}' <<<"${GPU_IDS}")"
TARGET_GLOBAL_BATCH="${ROBOTWIN_DUAL_AR_GLOBAL_BATCH:-128}"
MAIN_PROCESS_PORT="${ROBOTWIN_DUAL_AR_MAIN_PROCESS_PORT:-29781}"

if [[ "$(hostname -s)" != *node002* ]]; then
  echo "Run this watcher on node2; current host is $(hostname -s)." >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1
exec 9>"${LOCK}"
flock -n 9 || { echo "Another RoboTwin FSQ64 AR watcher holds ${LOCK}."; exit 3; }
write_status() {
  local code=$?
  if (( code == 0 )); then
    printf 'COMPLETE %s\n' "$(date -Is)" > "${STATUS}"
  else
    printf 'FAILED exit_code=%s %s\n' "${code}" "$(date -Is)" > "${STATUS}"
  fi
}
trap write_status EXIT

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "$(date -Is) watcher started on $(hostname -s)"
echo "target_cache=${TARGET_CACHE} run_id=${RUN_ID} gpu_ids=${GPU_IDS} target_global_batch=${TARGET_GLOBAL_BATCH}"

exporter_count_local() {
  ps -C python -o args= | grep -F 'scripts/physical_tokens/export_phizero_robotwin_dual_ar_targets.py' | wc -l || true
}

exporter_count_node4() {
  ssh -i /mnt/jfzn/liujn/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes \
    -o ConnectTimeout=8 "liujn@${NODE4}" \
    "ps -C python -o args= | grep -F 'scripts/physical_tokens/export_phizero_robotwin_dual_ar_targets.py' | wc -l" \
    2>/dev/null || true
}

quiet_polls=0
while [[ ! -s "${TARGET_CACHE}" || ! -s "${TARGET_MANIFEST}" ]]; do
  local_running="$(exporter_count_local | tr -d '[:space:]')"
  remote_running="$(exporter_count_node4 | tr -d '[:space:]')"
  [[ "${local_running}" =~ ^[0-9]+$ ]] || local_running=0
  if [[ "${remote_running}" =~ ^[0-9]+$ ]]; then
    remote_known=1
  else
    remote_known=0
  fi
  echo "$(date -Is) waiting for target export; node2_python=${local_running} node4_python=${remote_running:-unreachable}"
  if [[ "${local_running}" == 0 && "${remote_known}" == 1 && "${remote_running}" == 0 ]]; then
    quiet_polls=$((quiet_polls + 1))
  else
    quiet_polls=0
  fi
  if (( quiet_polls >= 5 )); then
    echo "Export processes disappeared on both nodes without producing a complete cache; refusing to train." >&2
    exit 1
  fi
  sleep 60
done

# The exporter writes the NPZ and sidecar atomically. A short stability check
# guards against observing a shared-filesystem update while it is in flight.
first_size="$(stat -c %s "${TARGET_CACHE}")"
sleep 30
second_size="$(stat -c %s "${TARGET_CACHE}")"
if [[ "${first_size}" != "${second_size}" ]]; then
  echo "Target archive is still changing (${first_size} -> ${second_size} bytes); waiting for it to settle."
  while true; do
    sleep 30
    stable_size="$(stat -c %s "${TARGET_CACHE}")"
    [[ "${stable_size}" == "${second_size}" ]] && break
    second_size="${stable_size}"
  done
fi

echo "$(date -Is) validating target cache and training inputs"
"${PYTHON}" - "${REPO}" "${TARGET_CACHE}" "${TARGET_MANIFEST}" "${TOKENIZER_CKPT}" "${ACTION_CKPT}" "${BASE_CKPT}" "${CONFIG}" "${DATA_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

repo, cache_path, manifest_path, tokenizer_path, action_path, base_path, config_path, data_root = map(Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for path in (cache_path, manifest_path, tokenizer_path, action_path, base_path, config_path, data_root):
    if not path.exists():
        raise SystemExit(f"Missing required training input: {path}")

report = json.loads(manifest_path.read_text())
digest = sha256(cache_path)
if report.get("output_sha256") != digest:
    raise SystemExit("Target-cache SHA256 does not match its sidecar manifest")
if report.get("phi_zero_checkpoint_sha256") != sha256(tokenizer_path):
    raise SystemExit("Target cache was encoded with a different PhiZero checkpoint")
if report.get("action_tokenizer_checkpoint_sha256") != sha256(action_path):
    raise SystemExit("Target cache was encoded with a different action-tokenizer checkpoint")
if report.get("kind") != "robotwin_clean_randomized_phizero_fsq64_dual_ar_targets":
    raise SystemExit(f"Unexpected target-cache kind: {report.get('kind')}")
if report.get("frame_protocol") != {
    "clean_source_fps": 50, "randomized_source_fps": 15, "target_fps": 16,
    "num_frames": 33, "max_padded_frames": 8,
    "short_starts": {"clean": [0, 20], "randomized": [0, 6]},
}:
    raise SystemExit(f"Unexpected frame protocol: {report.get('frame_protocol')}")

tasks = [line.strip() for line in (repo / "docs/robotwin_50task_list.txt").read_text().splitlines() if line.strip()]
expected_names = tasks + [f"Randomized/{task}" for task in tasks]
with np.load(cache_path, allow_pickle=False) as archive:
    arrays = {key: np.asarray(archive[key]) for key in archive.files}
required = {
    "action_token_ids", "physics_token_ids", "task_ids", "task_names", "episode_ids",
    "base_indices", "source_fps", "real_frame_counts", "padded_frame_counts",
    "forced_short_episode", "episode_is_short",
}
if required - arrays.keys():
    raise SystemExit(f"Target cache missing fields: {sorted(required - arrays.keys())}")
n = len(arrays["episode_ids"])
if n == 0 or arrays["action_token_ids"].shape != (n, 32) or arrays["physics_token_ids"].shape != (n, 64):
    raise SystemExit("Target cache has an empty or incorrectly shaped token array")
if arrays["task_names"].astype(str).tolist() != expected_names or len(tasks) != 50:
    raise SystemExit("Target-cache task table does not match the canonical 50 Clean + 50 Randomized tasks")
task_ids = arrays["task_ids"].astype(np.int64)
if not np.array_equal(np.unique(task_ids), np.arange(100)):
    raise SystemExit("Target cache must cover all 100 domain-qualified tasks")
if arrays["action_token_ids"].min() < 0 or arrays["action_token_ids"].max() >= 525:
    raise SystemExit("Action token ID outside [0,524]")
if arrays["physics_token_ids"].min() < 0 or arrays["physics_token_ids"].max() >= 25000:
    raise SystemExit("Physics token ID outside [0,24999]")
if not np.all(arrays["source_fps"][task_ids < 50] == 50) or not np.all(arrays["source_fps"][task_ids >= 50] == 15):
    raise SystemExit("Source FPS values do not agree with Clean/Randomized task IDs")
if not np.all(arrays["real_frame_counts"] + arrays["padded_frame_counts"] == 33):
    raise SystemExit("Real and padded frame counts do not sum to 33")
regular = ~arrays["forced_short_episode"].astype(bool)
if np.any(arrays["padded_frame_counts"][regular] > 8):
    raise SystemExit("A regular target exceeds the 8-frame terminal-padding cap")
short = arrays["forced_short_episode"].astype(bool)
if np.any(short & ~arrays["episode_is_short"].astype(bool)):
    raise SystemExit("Forced-short targets are not marked as short episodes")
if np.any(short & (task_ids < 50) & ~np.isin(arrays["base_indices"], [0, 20])):
    raise SystemExit("Clean forced-short target has an unexpected start index")
if np.any(short & (task_ids >= 50) & ~np.isin(arrays["base_indices"], [0, 6])):
    raise SystemExit("Randomized forced-short target has an unexpected start index")
keys = np.stack((task_ids, arrays["episode_ids"], arrays["base_indices"]), axis=1)
if len(np.unique(keys, axis=0)) != n:
    raise SystemExit("Target cache contains duplicate domain/task/episode/base keys")
if report.get("count") != n or report.get("action_shape") != [n, 32] or report.get("physics_shape") != [n, 64]:
    raise SystemExit("Target-cache dimensions do not match the sidecar report")

cfg = OmegaConf.load(config_path)
if cfg.framework.name != "QwenDualAutoregressiveReasoner":
    raise SystemExit(f"Unexpected framework: {cfg.framework.name}")
if int(cfg.framework.action_token_reasoner.token_count) != 32 or int(cfg.framework.action_token_reasoner.vocab_size) != 525:
    raise SystemExit("Action-token configuration does not match the cache")
if int(cfg.framework.dual_autoregressive_reasoner.physics_token_count) != 64 or int(cfg.framework.dual_autoregressive_reasoner.physics_vocab_size) != 25000:
    raise SystemExit("Physics-token configuration does not match the cache")
if not bool(cfg.framework.dual_autoregressive_reasoner.separate_heads) or not bool(cfg.framework.dual_autoregressive_reasoner.strict_modality_tying):
    raise SystemExit("Expected separate, modality-tied action/physics heads")
if Path(str(cfg.datasets.vla_data.discrete_token_cache)).resolve() != cache_path.resolve():
    raise SystemExit("Training config points to a different discrete-token cache")
if Path(str(cfg.trainer.pretrained_checkpoint)).resolve() != base_path.resolve():
    raise SystemExit("Training config points to a different pretrained VLM checkpoint")
if cfg.datasets.vla_data.data_mix != "robotwin_clean_randomized_wrap_32":
    raise SystemExit("Training config does not use the Clean + Randomized mixture")
if int(cfg.trainer.max_train_steps) != 10000:
    raise SystemExit("Training config must use the agreed 10k-step budget")
print(json.dumps({
    "cache_rows": n,
    "clean_rows": int(np.sum(task_ids < 50)),
    "randomized_rows": int(np.sum(task_ids >= 50)),
    "action_shape": list(arrays["action_token_ids"].shape),
    "physics_shape": list(arrays["physics_token_ids"].shape),
    "padded_rows": int(np.sum(arrays["padded_frame_counts"] > 0)),
    "forced_short_rows": int(np.sum(short)),
    "target_sha256": digest,
}, indent=2))
PY

# The exporter no longer needs the temporary decoded-clip RAM cache once both
# rank groups have exited and the immutable target archive has passed checks.
cleanup_waits=0
while true; do
  local_running="$(exporter_count_local | tr -d '[:space:]')"
  remote_running="$(exporter_count_node4 | tr -d '[:space:]')"
  [[ "${local_running}" =~ ^[0-9]+$ ]] || local_running=1
  if [[ "${remote_running}" =~ ^[0-9]+$ ]] && [[ "${local_running}" == 0 && "${remote_running}" == 0 ]]; then
    break
  fi
  cleanup_waits=$((cleanup_waits + 1))
  if (( cleanup_waits >= 10 )); then
    echo "Exporter output is complete but a rank is slow to exit; continuing without deleting its RAM cache."
    break
  fi
  echo "$(date -Is) waiting for exporter shutdown before releasing RAM clips; node2=${local_running} node4=${remote_running:-unreachable}"
  sleep 30
done
if [[ "${local_running:-1}" == 0 && -d "${RAM_CLIPS}" ]]; then
  echo "Removing completed node2 RAM clip cache: ${RAM_CLIPS}"
  rm -rf -- "${RAM_CLIPS}"
fi
if [[ "${remote_running:-unknown}" == 0 ]]; then
  ssh -i /mnt/jfzn/liujn/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes \
    -o ConnectTimeout=8 "liujn@${NODE4}" \
    "if [[ -d '${RAM_CLIPS}' ]]; then rm -rf -- '${RAM_CLIPS}'; fi" \
    || echo "Could not release node4 RAM clips now; leaving that cache untouched."
else
  echo "Leaving node4 RAM clips untouched because its exporter has not been confirmed idle."
fi

IFS=, read -r -a gpu_array <<<"${GPU_IDS}"
if (( ${#gpu_array[@]} != NUM_PROCESSES )); then
  echo "Could not determine GPU count from GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi
if (( TARGET_GLOBAL_BATCH % (NUM_PROCESSES * 8) != 0 )); then
  echo "Global batch ${TARGET_GLOBAL_BATCH} is not divisible by ${NUM_PROCESSES} GPUs x microbatch 8." >&2
  exit 2
fi

launch_stage() {
  local tag="$1" steps="$2" micro="$3" accumulation="$4" warmup="$5"
  local run_id="${RUN_ID}_${tag}"
  local log_path="${LOG_DIR}/${run_id}.log"
  local save_interval="${steps}"
  local log_frequency=10
  if (( steps <= 50 )); then log_frequency=1; fi
  if [[ -e "${REPO}/results/Checkpoints/${run_id}" ]]; then
    echo "Refusing to overwrite existing run directory: results/Checkpoints/${run_id}" >&2
    return 2
  fi
  echo "$(date -Is) launching ${tag}: GPUs=${GPU_IDS} microbatch=${micro} accumulation=${accumulation} global_batch=$((NUM_PROCESSES * micro * accumulation)) gradient_checkpointing=false steps=${steps}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.free \
    --format=csv -l 10 > "${log_path%.log}_gpu_memory.csv" &
  local monitor_pid=$!
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE}" launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG}" \
    --trainer.max_train_steps "${steps}" \
    --trainer.gradient_accumulation_steps "${accumulation}" \
    --datasets.vla_data.per_device_batch_size "${micro}" \
    --trainer.num_warmup_steps "${warmup}" \
    --trainer.eval_interval "$((steps + 1))" \
    --trainer.save_interval "${save_interval}" \
    --trainer.enable_gradient_checkpointing false \
    --trainer.logging_frequency "${log_frequency}" \
    --run_id "${run_id}" 2>&1 | tee "${log_path}"
  local train_status=${PIPESTATUS[0]}
  set -e
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  if (( train_status != 0 )); then
    echo "$(date -Is) ${tag} exited with status ${train_status}" >&2
    return "${train_status}"
  fi
  local checkpoint="${REPO}/results/Checkpoints/${run_id}/final_model/pytorch_model.pt"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "${tag} ended without a final trainable checkpoint: ${checkpoint}" >&2
    return 1
  fi
  echo "$(date -Is) ${tag} succeeded; checkpoint=${checkpoint}"
}

cd "${REPO}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

# A 50-step full-path smoke tests K=25k logits, data joins, checkpoint loading,
# DDP, optimizer updates and checkpoint saving before the expensive 10k run.
# Start at microbatch 16 on each 80-GiB node2 GPU; back off only on an actual
# OOM while preserving the global batch of 128.
smoke_micro=0
for candidate_micro in 16 8 4 2; do
  divisor=$((NUM_PROCESSES * candidate_micro))
  if (( TARGET_GLOBAL_BATCH % divisor != 0 )); then
    continue
  fi
  candidate_accum=$((TARGET_GLOBAL_BATCH / divisor))
  set +e
  launch_stage smoke_m"${candidate_micro}"_ga"${candidate_accum}" 50 \
    "${candidate_micro}" "${candidate_accum}" 5
  smoke_status=$?
  set -e
  if (( smoke_status == 0 )); then
    smoke_micro="${candidate_micro}"
    smoke_accum="${candidate_accum}"
    break
  fi
  smoke_log="${LOG_DIR}/${RUN_ID}_smoke_m${candidate_micro}_ga${candidate_accum}.log"
  if ! rg -q 'CUDA out of memory|OutOfMemoryError|out of memory' "${smoke_log}"; then
    echo "Smoke failed for a non-OOM reason; refusing to launch the 10k run." >&2
    exit "${smoke_status}"
  fi
  echo "$(date -Is) microbatch ${candidate_micro} smoke OOM; trying a smaller microbatch on the same GPUs."
done
if (( smoke_micro == 0 )); then
  echo "No smoke-test microbatch fit while preserving global batch ${TARGET_GLOBAL_BATCH}; not launching the 10k run." >&2
  exit 1
fi

launch_stage train10k 10000 "${smoke_micro}" "${smoke_accum}" 500

FINAL_CHECKPOINT="${REPO}/results/Checkpoints/${RUN_ID}_train10k/final_model/pytorch_model.pt"
EVAL_GPU="${ROBOTWIN_DUAL_AR_EVAL_GPU:-1}"
echo "$(date -Is) starting held-out teacher-forced/free-generation diagnostic on GPU ${EVAL_GPU}"
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" \
  scripts/physical_tokens/evaluate_dual_autoregressive_reasoner.py \
  --config "${CONFIG}" \
  --checkpoint "${FINAL_CHECKPOINT}" \
  --split docs/robotwin_tokenizer_episode_split.json \
  --tasks adjust_bottle Randomized/adjust_bottle \
  --samples-per-task 2 \
  --free-samples 2 \
  --output "${REPO}/results/Checkpoints/${RUN_ID}_train10k/heldout_diagnostic.json" \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}_heldout_diagnostic.log"

echo "$(date -Is) pipeline finished successfully; 10k checkpoint=${FINAL_CHECKPOINT}"
