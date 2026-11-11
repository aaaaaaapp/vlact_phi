#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/jfzn/liujn/code/VLAct-PhysToken
PHIZERO_REPO=/mnt/jfzn/liujn/code/PhiZero-main
PYTHON="${PHIZERO_TRAIN_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
EXPERIMENT_DIR="${REPO}/artifacts/robotwin_full50_study/phizero_fsq64_fps16"
DATA_DIR="${EXPERIMENT_DIR}/tokenizer_clips_clean_randomized_fps16"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${EXPERIMENT_DIR}/tokenizer_fsq64_gbs16_steps10k/run_${STAMP}"
LOG="${RUN_DIR}/training.log"
LOCK="${EXPERIMENT_DIR}/.tokenizer_fsq64_fps16.lock"
GPU_IDS="${NODE2_GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NODE2_NUM_PROCESSES:-8}"
GLOBAL_BATCH="${PHIZERO_GLOBAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${PHIZERO_GRADIENT_ACCUMULATION_STEPS:-1}"

if [[ "$(hostname -s)" != *node002* ]]; then
  echo "Run this script on node2; current host is $(hostname -s)." >&2
  exit 2
fi
for path in \
  "${DATA_DIR}/train.csv" \
  "${DATA_DIR}/val.csv" \
  "${DATA_DIR}/manifest.jsonl" \
  "${DATA_DIR}/preparation_report.json" \
  /mnt/jfzn/liujn/models/phizero_model/tokenizer/tokenizer.safetensors \
  /mnt/jfzn/liujn/models/phizero_model/wan2.2-vae/Wan2.2_VAE.safetensors \
  /mnt/jfzn/liujn/models/phizero_model/wan2.2-ti2v-5b/diffusion_pytorch_model-00001-of-00003.safetensors \
  /mnt/jfzn/liujn/models/phizero_model/wan2.2-ti2v-5b/diffusion_pytorch_model-00002-of-00003.safetensors \
  /mnt/jfzn/liujn/models/phizero_model/wan2.2-ti2v-5b/diffusion_pytorch_model-00003-of-00003.safetensors; do
  [[ -s "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 2; }
done

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG}") 2>&1
exec 9>"${LOCK}"
flock -n 9 || { echo "Another 16 FPS PhiZero tokenizer run holds ${LOCK}."; exit 3; }

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:${PHIZERO_REPO}/DiffSynth-Studio${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${DIFFSYNTH_ATTENTION_IMPLEMENTATION:-flash_attention_2}"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

"${PYTHON}" - "${DATA_DIR}" "${GLOBAL_BATCH}" "${GRAD_ACCUM}" "${NUM_PROCESSES}" <<'PY'
import csv, json, math, sys, torch
from pathlib import Path
data_dir, global_batch, grad_accum, nproc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
csv_path = data_dir / "train.csv"
with open(csv_path, newline="") as f:
    csv_rows = [row["video"] for row in csv.DictReader(f)]
manifest = [json.loads(line) for line in (data_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
train = [row for row in manifest if row["split"] == "train"]
val = [row for row in manifest if row["split"] == "val"]
if not train or not val:
    raise SystemExit(f"Expected nonempty train/val splits, got {len(train)} / {len(val)}")
domains = {row["source_domain"] for row in manifest}
if domains != {"clean", "randomized"}:
    raise SystemExit(f"Expected both clean and randomized data, got domains={domains}")
expected_fps = {"clean": 50, "randomized": 15}
for split_name, split_rows in (("train", train), ("val", val)):
    with (data_dir / f"{split_name}.csv").open(newline="") as f:
        csv_split = [row["video"] for row in csv.DictReader(f)]
    if csv_split != [row["video"] for row in split_rows]:
        raise SystemExit(f"{split_name}.csv does not exactly match its manifest rows")
    if {row["source_domain"] for row in split_rows} != domains:
        raise SystemExit(f"{split_name} split does not contain both domains")
for row in manifest:
    if (row["source_fps"], row["target_fps"], row["num_frames"]) != (expected_fps[row["source_domain"]], 16, 33):
        raise SystemExit(f"Unexpected fps/frame protocol: {row}")
    if row["source_fps_metadata"] != row["source_fps"] or row["max_padded_frames"] != 8:
        raise SystemExit(f"Source FPS or pad-cap metadata mismatch: {row}")
    if row["real_frame_count"] + row["padded_frame_count"] != 33:
        raise SystemExit(f"Invalid real/padded frame counts: {row}")
    if row["forced_short_episode"]:
        stride = 20 if row["source_domain"] == "clean" else 6
        full_span_frames = 101 if row["source_domain"] == "clean" else 31
        if (not row["episode_is_short"] or row["episode_source_frames"] >= full_span_frames
                or row["base_index"] not in (0, stride)):
            raise SystemExit(f"Invalid short-episode exception: {row}")
    elif row["padded_frame_count"] > 8:
        raise SystemExit(f"Normal window exceeds the 8-frame cap: {row}")
    if not (data_dir / row["video"]).is_file():
        raise SystemExit(f"Missing training clip: {row['video']}")
clips = len(train)
if torch.cuda.device_count() != nproc:
    raise SystemExit(f"CUDA_VISIBLE_DEVICES exposes {torch.cuda.device_count()} GPUs; expected {nproc}")
if global_batch % grad_accum:
    raise SystemExit("global batch must be divisible by gradient accumulation")
micro_global = global_batch // grad_accum
local_micro = math.ceil(micro_global / nproc)
if local_micro > 1 and micro_global % nproc:
    raise SystemExit("When per-rank microbatch exceeds one, microbatch must divide evenly across processes")
steps_per_epoch = math.ceil(math.ceil(clips / micro_global) / grad_accum)
domain_counts = {domain: sum(row["source_domain"] == domain for row in train) for domain in sorted(domains)}
print(f"cuda={torch.cuda.device_count()} train_clips={clips} domain_counts={domain_counts} "
      f"global_batch={global_batch} "
      f"grad_accum={grad_accum} per_rank_microbatch={local_micro} "
      f"optimizer_steps_per_epoch={steps_per_epoch}", flush=True)
PY

echo "$(date -Is) starting PhiZero FSQ64 16 FPS training on $(hostname -s)"
echo "run_dir=${RUN_DIR} GPUs=${CUDA_VISIBLE_DEVICES} global_batch=${GLOBAL_BATCH} grad_accum=${GRAD_ACCUM}"

cd "${REPO}"
"${PYTHON}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --multi_gpu \
  scripts/physical_tokens/train_phizero_global_batch.py \
  --csv "${DATA_DIR}/train.csv" \
  --init-checkpoint /mnt/jfzn/liujn/models/phizero_model/tokenizer/tokenizer.safetensors \
  --output-dir "${RUN_DIR}" \
  --dit-path \
    /mnt/jfzn/liujn/models/phizero_model/wan2.2-ti2v-5b/diffusion_pytorch_model-00001-of-00003.safetensors \
    /mnt/jfzn/liujn/models/phizero_model/wan2.2-ti2v-5b/diffusion_pytorch_model-00002-of-00003.safetensors \
    /mnt/jfzn/liujn/models/phizero_model/wan2.2-ti2v-5b/diffusion_pytorch_model-00003-of-00003.safetensors \
  --vae-path /mnt/jfzn/liujn/models/phizero_model/wan2.2-vae/Wan2.2_VAE.safetensors \
  --num-frames 33 \
  --height 512 \
  --width 896 \
  --global-batch-size "${GLOBAL_BATCH}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --epochs 10 \
  --max-train-steps 10000 \
  --queries-per-transition 8 \
  --num-workers "${PHIZERO_NUM_WORKERS:-4}" \
  --warmup-steps 500 \
  --learning-rate 2e-5 \
  --min-learning-rate 1e-6 \
  --save-every 5000 \
  --seed 42 \
  --fsq-entropy-weight 0.05 \
  --levels 8,5,5,5,5,5 &
TRAIN_PID=$!

"${PYTHON}" scripts/physical_tokens/log_phizero_run_wandb_offline.py \
  --log "${LOG}" \
  --run-dir "${RUN_DIR}" \
  --train-pid "${TRAIN_PID}" \
  --name "phizero-fsq64-clean-randomized-fps16-gbs16-ga${GRAD_ACCUM}-${STAMP}" \
  >> "${RUN_DIR}/wandb_offline_mirror.log" 2>&1 &
LOGGER_PID=$!

set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
wait "${LOGGER_PID}"
LOGGER_STATUS=$?
set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Training exited with status ${TRAIN_STATUS}; offline W&B logger status ${LOGGER_STATUS}." >&2
  exit "${TRAIN_STATUS}"
fi
if [[ "${LOGGER_STATUS}" -ne 0 ]]; then
  echo "Training completed, but offline W&B logger exited with status ${LOGGER_STATUS}." >&2
  exit "${LOGGER_STATUS}"
fi
