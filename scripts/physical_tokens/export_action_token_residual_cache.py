#!/usr/bin/env python3
"""Build full-horizon residual actions after decoding predicted action tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_action_tokenizer import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--predicted-token-cache", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    action_path = args.action_cache.resolve()
    token_path = args.predicted_token_cache.resolve()
    checkpoint_path = args.action_tokenizer_checkpoint.resolve()
    with np.load(action_path, allow_pickle=False) as data:
        actions = np.asarray(data["actions"], dtype=np.float32)
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "actions"}
    with np.load(token_path, allow_pickle=False) as data:
        key_names = ["episode_ids", "base_indices"]
        if "task_ids" in arrays:
            key_names.insert(0, "task_ids")
        for key in key_names:
            if not np.array_equal(data[key], arrays[key]):
                raise ValueError(f"predicted token cache is not key-aligned: {key}")
        token_ids = np.asarray(data["action_token_ids"], dtype=np.int64)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    decoded = []
    with torch.inference_mode():
        for start in range(0, len(token_ids), args.batch_size):
            ids = torch.as_tensor(
                token_ids[start : start + args.batch_size], dtype=torch.long, device=device
            )
            decoded.append(model.decode_tokens(ids, gripper_dims=(12, 13)).cpu().numpy())
    decoded = np.concatenate(decoded).astype(np.float32)
    if decoded.shape != actions.shape:
        raise ValueError(f"decoded/action shapes differ: {decoded.shape} vs {actions.shape}")
    residuals = actions - decoded
    residuals[..., :12] = np.remainder(residuals[..., :12] + np.pi, 2.0 * np.pi) - np.pi
    output_arrays = {**arrays, "actions": residuals.astype(np.float32)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output_arrays)
    os.replace(temporary, args.output)
    report = {
        "schema_version": 1,
        "kind": "predicted_action_token_full_horizon_residuals",
        "action_cache": str(action_path),
        "action_cache_sha256": sha256_file(action_path),
        "predicted_token_cache": str(token_path),
        "predicted_token_cache_sha256": sha256_file(token_path),
        "action_tokenizer_checkpoint": str(checkpoint_path),
        "action_tokenizer_checkpoint_sha256": sha256_file(checkpoint_path),
        "shape": list(residuals.shape),
        "definition": "wrapped joint and continuous gripper residual after predicted 4-token decode",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
