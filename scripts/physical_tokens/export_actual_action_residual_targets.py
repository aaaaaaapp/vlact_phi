#!/usr/bin/env python3
"""Export actual action residuals relative to cached predicted action tokens."""

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

from evaluate_action_tokenizer import build_model as build_action_tokenizer
from train_frozen_physics_residual_decoder import exact_keys, load


DECODER_INDICES = np.asarray(list(range(2, 32, 2)) + [31], dtype=np.int64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actual_residual_targets(actions: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    if actions.shape != decoded.shape or actions.ndim != 3 or actions.shape[-1] != 14:
        raise ValueError("actions and decoded actions must have equal [N,T,14] shapes")
    residual = np.asarray(actions - decoded, dtype=np.float32)
    residual[..., :12] = np.remainder(residual[..., :12] + np.pi, 2.0 * np.pi) - np.pi
    return residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--predicted-action-token-cache", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    actions = load(args.action_cache.resolve())
    predicted = load(args.predicted_action_token_cache.resolve())
    if exact_keys(actions) != exact_keys(predicted):
        raise ValueError("action and predicted-token caches are not exactly key-aligned")
    action_values = np.asarray(actions["actions"], dtype=np.float32)
    if action_values.ndim != 3 or action_values.shape[1] <= int(DECODER_INDICES.max()):
        raise ValueError("action cache does not cover the decoder horizon")

    state = torch.load(
        args.action_tokenizer_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    model = build_action_tokenizer(state["model_config"])
    model.load_state_dict(state["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    decoded_batches = []
    token_ids = np.asarray(predicted["action_token_ids"], dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(token_ids), args.batch_size):
            ids = torch.as_tensor(
                token_ids[start : start + args.batch_size], dtype=torch.long, device=device
            )
            decoded_batches.append(
                model.decode_tokens(ids, gripper_dims=(12, 13)).float().cpu().numpy()
            )
    decoded = np.concatenate(decoded_batches)
    selected_actions = action_values[:, DECODER_INDICES]
    selected_decoded = decoded[:, DECODER_INDICES]
    residual = actual_residual_targets(selected_actions, selected_decoded)
    targets = np.concatenate(
        (np.zeros((len(residual), 1, 14), dtype=np.float32), residual), axis=1
    )
    selected_valid = np.asarray(actions["action_valid_mask"], dtype=bool)[:, DECODER_INDICES]
    target_valid = np.concatenate(
        (np.ones((len(residual), 1), dtype=bool), selected_valid), axis=1
    )
    arrays = {
        "task_ids": np.asarray(actions["task_ids"], dtype=np.int64),
        "task_names": np.asarray(actions["task_names"]),
        "episode_ids": np.asarray(actions["episode_ids"], dtype=np.int64),
        "base_indices": np.asarray(actions["base_indices"], dtype=np.int64),
        "offsets": np.concatenate((np.asarray([0]), DECODER_INDICES)),
        "action_targets": targets,
        "target_valid_mask": target_valid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    manifest = {
        "schema_version": 1,
        "kind": "actual_action_residual_from_predicted_action_tokens",
        "num_samples": len(targets),
        "num_transitions": int(residual.shape[1]),
        "action_cache": str(args.action_cache.resolve()),
        "action_cache_sha256": sha256_file(args.action_cache.resolve()),
        "predicted_action_token_cache": str(args.predicted_action_token_cache.resolve()),
        "predicted_action_token_cache_sha256": sha256_file(
            args.predicted_action_token_cache.resolve()
        ),
        "action_tokenizer_checkpoint": str(args.action_tokenizer_checkpoint.resolve()),
        "action_tokenizer_checkpoint_sha256": sha256_file(
            args.action_tokenizer_checkpoint.resolve()
        ),
        "joint_residual": "wrapped_to_minus_pi_pi",
        "gripper_residual": "direct_difference",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
