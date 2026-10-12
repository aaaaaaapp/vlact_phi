#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ "${mode}" != "e0" && "${mode}" != "e1" && "${mode}" != "e2" && "${mode}" != "b2" && "${mode}" != "b3" && "${mode}" != "b4" ]]; then
  echo "Usage: $0 {e0|e1|e2|b2|b3|b4} [max_train_steps] [per_device_batch_size]" >&2
  exit 2
fi

max_train_steps="${2:-20}"
per_device_batch_size="${3:-1}"
num_processes="${NUM_PROCESSES:-8}"
main_process_port="${MAIN_PROCESS_PORT:-29620}"
config_yaml="examples/Robotwin/train_files/qwenpi_physical_tokens_click_alarmclock.yaml"
shared_pretrained_checkpoint="/mnt/jfzn/liujn/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt"
node_local_pretrained_checkpoint="${NODE_LOCAL_PRETRAINED_CHECKPOINT:-/tmp/vlact_qwen3_pretrain_steps100000.pt}"
if [[ -f "${node_local_pretrained_checkpoint}" ]]; then
  default_pretrained_checkpoint="${node_local_pretrained_checkpoint}"
else
  default_pretrained_checkpoint="${shared_pretrained_checkpoint}"
fi
pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-${default_pretrained_checkpoint}}"
default_warmup_steps=$((max_train_steps / 10))
if (( default_warmup_steps < 2 )); then
  default_warmup_steps=2
elif (( default_warmup_steps > 200 )); then
  default_warmup_steps=200
fi
num_warmup_steps="${NUM_WARMUP_STEPS:-${default_warmup_steps}}"
eval_interval="${EVAL_INTERVAL:-$((max_train_steps + 1))}"

if [[ "${mode}" == "e0" ]]; then
  conditioning_enabled=false
else
  conditioning_enabled=true
fi

case "${mode}" in
  e2)
    default_token_cache="./artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz"
    default_action_vocab_size=128
    default_max_action_tokens=4
    default_physics_vocab_size=0
    ;;
  b2)
    default_token_cache="./artifacts/token_caches/click_alarmclock_oracle_physics_ema_4x128_step1000.npz"
    default_action_vocab_size=128
    default_max_action_tokens=4
    default_physics_vocab_size=128
    ;;
  b3)
    default_token_cache="./artifacts/token_caches/click_alarmclock_pred_action4_oracle_physics4.npz"
    default_action_vocab_size=128
    default_max_action_tokens=4
    default_physics_vocab_size=128
    ;;
  b4)
    default_token_cache="./artifacts/token_caches/click_alarmclock_pred_action4_shuffled_physics4_seed42.npz"
    default_action_vocab_size=128
    default_max_action_tokens=4
    default_physics_vocab_size=128
    ;;
  *)
    default_token_cache="./artifacts/token_caches/click_alarmclock_clean_ema_step5000.npz"
    default_action_vocab_size=512
    default_max_action_tokens=8
    default_physics_vocab_size=0
    ;;
esac

token_cache="${TOKEN_CACHE:-${ACTION_TOKEN_CACHE:-${default_token_cache}}}"
action_vocab_size="${ACTION_VOCAB_SIZE:-${default_action_vocab_size}}"
max_action_tokens="${MAX_ACTION_TOKENS:-${default_max_action_tokens}}"
physics_vocab_size="${PHYSICS_VOCAB_SIZE:-${default_physics_vocab_size}}"
max_physics_tokens="${MAX_PHYSICS_TOKENS:-4}"
action_gate_init="${ACTION_GATE_INIT:-0.1}"
physics_gate_init="${PHYSICS_GATE_INIT:-0.1}"
action_condition_dropout="${ACTION_CONDITION_DROPOUT:-0.0}"
physics_condition_dropout="${PHYSICS_CONDITION_DROPOUT:-0.0}"
physics_fusion_mode="${PHYSICS_FUSION_MODE:-concat}"
physics_fusion_dim="${PHYSICS_FUSION_DIM:-256}"
physics_fusion_heads="${PHYSICS_FUSION_HEADS:-8}"
dit_injection_layers="${DIT_INJECTION_LAYERS:-all}"
dit_attention_dropout="${DIT_ATTENTION_DROPOUT:-0.0}"
dit_freeze_base_action_model="${DIT_FREEZE_BASE_ACTION_MODEL:-false}"
dit_action_tokens_only="${DIT_ACTION_TOKENS_ONLY:-false}"
dit_physics_adapter_type="${DIT_PHYSICS_ADAPTER_TYPE:-cross_attention}"
dit_physics_memory_pooling="${DIT_PHYSICS_MEMORY_POOLING:-none}"
physics_tokens_per_transition="${PHYSICS_TOKENS_PER_TRANSITION:-16}"
physics_condition_enabled="${PHYSICS_CONDITION_ENABLED:-true}"
physics_usefulness_weight="${PHYSICS_USEFULNESS_WEIGHT:-0.0}"
physics_usefulness_margin="${PHYSICS_USEFULNESS_MARGIN:-0.01}"
variant_suffix="${VARIANT_SUFFIX:-}"
action_init_checkpoint="${ACTION_INIT_CHECKPOINT:-}"
physics_tokenizer_init_checkpoint="${PHYSICS_TOKENIZER_INIT_CHECKPOINT:-}"
physics_tokenizer_init_sha256="${PHYSICS_TOKENIZER_INIT_SHA256:-}"
freeze_modules="${FREEZE_MODULES:-qwen_vl_interface}"
base_learning_rate="${BASE_LEARNING_RATE:-1.0e-5}"
action_model_learning_rate="${ACTION_MODEL_LEARNING_RATE:-1.0e-4}"
discrete_conditioner_learning_rate="${DISCRETE_CONDITIONER_LEARNING_RATE:-1.0e-4}"

action_init_args=()
if [[ -n "${action_init_checkpoint}" ]]; then
  action_init_args+=(--trainer.action_init_checkpoint "${action_init_checkpoint}")
fi
physics_tokenizer_init_args=()
if [[ -n "${physics_tokenizer_init_checkpoint}" ]]; then
  if [[ ! -s "${physics_tokenizer_init_checkpoint}" ]]; then
    echo "Physics tokenizer initialization checkpoint is missing: ${physics_tokenizer_init_checkpoint}" >&2
    exit 2
  fi
  if [[ ! "${physics_tokenizer_init_sha256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "PHYSICS_TOKENIZER_INIT_SHA256 must be set to a 64-digit SHA256" >&2
    exit 2
  fi
  physics_tokenizer_init_args+=(
    --trainer.physics_tokenizer_init_checkpoint "${physics_tokenizer_init_checkpoint}"
    --trainer.physics_tokenizer_init_sha256 "${physics_tokenizer_init_sha256}"
  )
fi

run_id="qwenpi_physical_tokens_click_alarmclock_${mode}${variant_suffix}_seed42_steps${max_train_steps}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_process_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.discrete_conditioning.enabled "${conditioning_enabled}" \
  --framework.discrete_conditioning.action_vocab_size "${action_vocab_size}" \
  --framework.discrete_conditioning.max_action_tokens "${max_action_tokens}" \
  --framework.discrete_conditioning.physics_vocab_size "${physics_vocab_size}" \
  --framework.discrete_conditioning.max_physics_tokens "${max_physics_tokens}" \
  --framework.discrete_conditioning.action_gate_init "${action_gate_init}" \
  --framework.discrete_conditioning.physics_gate_init "${physics_gate_init}" \
  --framework.discrete_conditioning.action_condition_dropout "${action_condition_dropout}" \
  --framework.discrete_conditioning.physics_condition_dropout "${physics_condition_dropout}" \
  --framework.discrete_conditioning.physics_fusion_mode "${physics_fusion_mode}" \
  --framework.discrete_conditioning.physics_fusion_dim "${physics_fusion_dim}" \
  --framework.discrete_conditioning.physics_fusion_heads "${physics_fusion_heads}" \
  --framework.discrete_conditioning.dit_injection_layers "${dit_injection_layers}" \
  --framework.discrete_conditioning.dit_attention_dropout "${dit_attention_dropout}" \
  --framework.discrete_conditioning.dit_freeze_base_action_model "${dit_freeze_base_action_model}" \
  --framework.discrete_conditioning.dit_action_tokens_only "${dit_action_tokens_only}" \
  --framework.discrete_conditioning.dit_physics_adapter_type "${dit_physics_adapter_type}" \
  --framework.discrete_conditioning.dit_physics_memory_pooling "${dit_physics_memory_pooling}" \
  --framework.discrete_conditioning.physics_tokens_per_transition "${physics_tokens_per_transition}" \
  --framework.discrete_conditioning.physics_condition_enabled "${physics_condition_enabled}" \
  --framework.discrete_conditioning.physics_usefulness_weight "${physics_usefulness_weight}" \
  --framework.discrete_conditioning.physics_usefulness_margin "${physics_usefulness_margin}" \
  --datasets.vla_data.discrete_token_cache "${token_cache}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.num_workers 2 \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  "${action_init_args[@]}" \
  "${physics_tokenizer_init_args[@]}" \
  --trainer.freeze_modules "${freeze_modules}" \
  --trainer.learning_rate.base "${base_learning_rate}" \
  --trainer.learning_rate.action_model "${action_model_learning_rate}" \
  --trainer.learning_rate.discrete_conditioner "${discrete_conditioner_learning_rate}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.num_warmup_steps "${num_warmup_steps}" \
  --trainer.eval_interval "${eval_interval}" \
  --trainer.save_interval "${max_train_steps}" \
  --trainer.logging_frequency 1 \
  --run_id "${run_id}"
