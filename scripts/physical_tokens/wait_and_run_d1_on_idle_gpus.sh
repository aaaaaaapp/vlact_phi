#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

poll_seconds="${D1_IDLE_POLL_SECONDS:-60}"
required_checks="${D1_IDLE_CONSECUTIVE_CHECKS:-3}"
max_checks="${D1_WATCH_MAX_CHECKS:-0}"
lock_path="${D1_WATCH_LOCK:-logs/d1_idle_watcher.lock}"
memory_threshold="${D1_IDLE_MEMORY_THRESHOLD_MIB:-256}"

if [[ "${poll_seconds}" -lt 1 || "${required_checks}" -lt 1 || "${max_checks}" -lt 0 ]] \
  || [[ ! "${memory_threshold}" =~ ^[0-9]+$ ]]; then
  echo "poll/check settings must be positive (max checks may be zero for unlimited)" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "D1 watcher failed: nvidia-smi is unavailable" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "D1 watcher failed: flock is unavailable" >&2
  exit 2
fi

mkdir -p "$(dirname "${lock_path}")"
exec 9>"${lock_path}"
if ! flock -n 9; then
  echo "Another D1 watcher/run holds ${lock_path}; exiting." >&2
  exit 4
fi

checks=0
consecutive_idle=0
while true; do
  checks=$((checks + 1))
  compute_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  busy_memory="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v threshold="${memory_threshold}" '$2 + 0 > threshold {print}')"
  if [[ -z "${compute_apps}" && -z "${busy_memory}" ]]; then
    consecutive_idle=$((consecutive_idle + 1))
    echo "$(date --iso-8601=seconds) GPUs idle (${consecutive_idle}/${required_checks})"
  else
    consecutive_idle=0
    app_count=0
    if [[ -n "${compute_apps}" ]]; then
      app_count="$(wc -l <<<"${compute_apps}")"
    fi
    memory_gpu_count=0
    if [[ -n "${busy_memory}" ]]; then
      memory_gpu_count="$(wc -l <<<"${busy_memory}")"
    fi
    echo "$(date --iso-8601=seconds) waiting: ${app_count} compute app row(s), ${memory_gpu_count} GPU(s) above ${memory_threshold} MiB"
  fi

  if [[ "${consecutive_idle}" -ge "${required_checks}" ]]; then
    echo "$(date --iso-8601=seconds) idle window confirmed; starting D1"
    exec bash scripts/physical_tokens/run_d1_multitask_tokenizer.sh
  fi
  if [[ "${max_checks}" -gt 0 && "${checks}" -ge "${max_checks}" ]]; then
    echo "D1 watcher reached max checks without an idle window." >&2
    exit 3
  fi
  sleep "${poll_seconds}"
done
