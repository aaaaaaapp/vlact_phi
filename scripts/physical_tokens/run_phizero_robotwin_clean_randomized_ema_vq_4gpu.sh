#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
phizero_repo=/mnt/jfzn/liujn/code/PhiZero-main
python="${PHIZERO_TRAIN_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
data_root="${repo}/artifacts/robotwin_full50_study/phizero_robotwin_videos_windows1s_clean_randomized"
run_root="${repo}/artifacts/robotwin_full50_study/phizero_clean_randomized_windows1s_ema_vq2048_fresh_decoder_gbs32_3epochs"
output="${run_root}/run_$(date +%Y%m%d_%H%M%S)"
log="${output}/training.log"

mkdir -p "${output}"
exec > >(tee -a "${log}") 2>&1

export PATH="$(dirname "${python}"):${PATH}"
export PYTHONPATH="/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:${phizero_repo}/DiffSynth-Studio${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${DIFFSYNTH_ATTENTION_IMPLEMENTATION:-flash_attention_2}"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

csv="${data_root}/train.csv"
checkpoint=/mnt/jfzn/liujn/models/phizero_model/tokenizer/tokenizer.safetensors
vae=/mnt/jfzn/liujn/models/phizero_model/wan2.2-vae/Wan2.2_VAE.safetensors
dit=(
  /mnt/jfzn/liujn/models/phizero_model/wan2.2-ti2v-5b/diffusion_pytorch_model-00001-of-00003.safetensors
  /mnt/jfzn/liujn/models/phizero_model/wan2.2-ti2v-5b/diffusion_pytorch_model-00002-of-00003.safetensors
  /mnt/jfzn/liujn/models/phizero_model/wan2.2-ti2v-5b/diffusion_pytorch_model-00003-of-00003.safetensors
)

for path in "${csv}" "${checkpoint}" "${vae}" "${dit[@]}"; do
  [[ -s "${path}" ]] || { echo "Missing required training input: ${path}" >&2; exit 2; }
done

echo "$(date -Is) ACP PhiZero fresh EMA-VQ run starting on $(hostname)"
echo "dataset=${csv} output=${output} global_batch=32 (microbatch=4 global, grad_accum=8) epochs=3 frames=33"
echo "init=fresh tokenizer/conditioner/LoRA; frozen pretrained Wan2.2 DiT + trainable LoRA"
"${python}" - <<'PY'
import torch
import accelerate
print(f"torch={torch.__version__} cuda={torch.version.cuda} visible_gpus={torch.cuda.device_count()}")
print(f"accelerate={accelerate.__version__}")
if torch.cuda.device_count() != 4:
    raise SystemExit("ACP worker must expose exactly four GPUs")
PY

cd "${repo}"
"${python}" -m accelerate.commands.launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --multi_gpu \
  scripts/physical_tokens/train_phizero_global_batch.py \
  --csv "${csv}" \
  --init-checkpoint "${checkpoint}" \
  --output-dir "${output}" \
  --dit-path "${dit[@]}" \
  --vae-path "${vae}" \
  --num-frames 33 \
  --global-batch-size 32 \
  --gradient-accumulation-steps 8 \
  --epochs 3 \
  --num-workers "${PHIZERO_NUM_WORKERS:-4}" \
  --warmup-steps 500 \
  --learning-rate 2e-5 \
  --min-learning-rate 1e-6 \
  --save-every 0 \
  --seed 43 \
  --quantizer ema_vq \
  --codebook-size 2048 \
  --vq-ema-decay 0.99 \
  --dead-code-threshold 0.1 \
  --model-ema-decay 0.9999 \
  --fresh-phi-zero-components
