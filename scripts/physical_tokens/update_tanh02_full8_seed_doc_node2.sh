#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
RUN_TAG=${RUN_TAG:-tanh02_pure_full8_node2_20261029}
COUNT=${COUNT:-100}
PY=/mnt/jfzn/liujn/robotwin_env/bin/python
RUN_DIR="$ROOT/logs/$RUN_TAG"

[[ "$(hostname)" == *node002* ]] || { echo "Run only on node002" >&2; exit 2; }
for _ in $(seq 1 2880); do
  if "$PY" "$ROOT/scripts/physical_tokens/publish_tanh02_full8_seeds.py" \
      "$RUN_DIR" "$ROOT/docs" --count "$COUNT"; then
    exit 0
  fi
  sleep 30
done
echo "Seed document update timed out before eight tasks completed" >&2
exit 1
