#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
if [[ ! -s "${cache}" || ! -s "${cache}.manifest.json" ]]; then
  echo "Full 50-task merged physics cache and manifest are required" >&2
  exit 2
fi

source_sha="$(sha256sum "${cache}" | cut -d' ' -f1)"
output="${ROBOTWIN_FRAME_ARRAY_OUTPUT:-/tmp/robotwin_full50_frame_features_${source_sha:0:16}.npy}"
validate() {
  "${python_bin}" -c 'import json,numpy as np,sys; m=json.load(open(sys.argv[2], encoding="utf-8")); x=np.load(sys.argv[1], mmap_mode="r", allow_pickle=False); expected=tuple(m["frame_feature_shape"]); assert m["num_tasks"]==50 and x.shape==expected and x.dtype==np.dtype(m["frame_feature_dtype"]), (x.shape,x.dtype,expected)' "$1" "${cache}.manifest.json"
}
if [[ -s "${output}" ]]; then
  validate "${output}"
  echo "${output}"
  exit 0
fi
if [[ -e "${output}" ]]; then
  echo "Refusing incomplete destination: ${output}" >&2
  exit 2
fi

temporary="${output}.tmp.$$"
trap 'if [[ -e "${temporary}" ]]; then rm -f -- "${temporary}"; fi' EXIT
unzip -p "${cache}" frame_features.npy >"${temporary}"
validate "${temporary}"
mv -- "${temporary}" "${output}"
trap - EXIT
echo "${output}"
