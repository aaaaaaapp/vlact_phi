#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NODES_ENV="${SCRIPT_DIR}/../setup/nodes.env"

if [[ -f "${NODES_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${NODES_ENV}"
fi

CLUSTER_USER="${CLUSTER_USER:-$(id -un)}"
NODE001_IP="${NODE001_IP:-10.120.154.139}"
NODE002_IP="${NODE002_IP:-10.120.154.140}"
NODE003_IP="${NODE003_IP:-10.120.154.141}"
NODE004_IP="${NODE004_IP:-10.120.154.142}"
SSH_TIMEOUT="${SSH_TIMEOUT:-8}"

usage() {
  printf 'Usage: %s\n' "$(basename "$0")"
  printf 'Checks per-GPU memory and utilization on node1-node4. Run it from node1.\n'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ "$#" -gt 0 ]]; then
  usage >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'ERROR: nvidia-smi was not found on node1.\n' >&2
  exit 1
fi
if ! command -v ssh >/dev/null 2>&1 || ! command -v timeout >/dev/null 2>&1; then
  printf 'ERROR: this script requires ssh and timeout.\n' >&2
  exit 1
fi

QUERY='nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits'
printf 'GPU usage snapshot: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
printf '%-6s %-22s %-4s %-28s %12s %12s %12s %10s %10s %8s %10s\n' \
  NODE HOST GPU MODEL 'MEM_USED(MB)' 'MEM_TOTAL(MB)' 'MEM_FREE(MB)' GPU_UTIL MEM_UTIL 'TEMP(C)' 'POWER(W)'

failed=0
for node_num in 1 2 3 4; do
  case "$node_num" in
    1) ip="$NODE001_IP" ;;
    2) ip="$NODE002_IP" ;;
    3) ip="$NODE003_IP" ;;
    4) ip="$NODE004_IP" ;;
  esac

  if [[ "$node_num" -eq 1 ]]; then
    if ! output="$($QUERY 2>&1)"; then
      printf 'ERROR: node1 (%s): %s\n' "$ip" "$output" >&2
      failed=1
      continue
    fi
    host="$(hostname -s 2>/dev/null || hostname)"
  else
    if ! output="$(timeout "${SSH_TIMEOUT}s" ssh -q \
      -o BatchMode=yes \
      -o ConnectTimeout="$SSH_TIMEOUT" \
      -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=3 \
      -o ServerAliveCountMax=1 \
      "${CLUSTER_USER}@${ip}" "$QUERY" 2>&1)"; then
      printf 'ERROR: node%s (%s) SSH/query failed: %s\n' "$node_num" "$ip" "$output" >&2
      failed=1
      continue
    fi
    host="node$(printf '%03d' "$node_num")"
  fi

  rows="$(printf '%s\n' "$output" | awk -F, '
    NF >= 9 && $1 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
      for (i = 1; i <= 9; i++) {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
      }
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9
    }')"

  if [[ -z "$rows" ]]; then
    printf 'ERROR: node%s (%s) returned no GPU rows.\n' "$node_num" "$ip" >&2
    failed=1
    continue
  fi

  while IFS=$'\t' read -r gpu model used total free gpu_util mem_util temp power; do
    printf '%-6s %-22s %-4s %-28s %12s %12s %12s %9s%% %9s%% %8s %9s\n' \
      "node${node_num}" "$host" "$gpu" "$model" "$used" "$total" "$free" \
      "$gpu_util" "$mem_util" "$temp" "$power"
  done <<< "$rows"
done

exit "$failed"
