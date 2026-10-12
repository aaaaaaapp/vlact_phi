#!/usr/bin/env python3
"""Test whether a frozen action auxiliary decoder depends on aligned token IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_spatiotemporal_physics_token_ids import build_model
from train_spatiotemporal_physical_tokenizer import (
    SpatiotemporalFeatureDataset,
    parse_ints,
    split_complete_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--action-target-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-episodes", default="5,20,25,39,49")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def shuffle_within_episodes(
    ids: torch.Tensor, task_ids: np.ndarray, episodes: np.ndarray, seed: int
) -> torch.Tensor:
    result = ids.clone()
    rng = np.random.default_rng(seed)
    for task, episode in sorted(
        set(zip(task_ids.astype(int).tolist(), episodes.astype(int).tolist()))
    ):
        rows = np.flatnonzero((task_ids == task) & (episodes == episode))
        if len(rows) > 1:
            shift = int(rng.integers(1, len(rows)))
            result[rows] = ids[torch.as_tensor(np.roll(rows, shift))]
    return result


def predict(model, ids: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size].to(device)
            quantized = model.quantizer.decode(batch_ids).reshape(
                len(batch_ids), model.num_transitions, model.queries_per_transition, -1
            )
            predictions.append(model.predict_action_aux(quantized).cpu())
    return torch.cat(predictions)


def metrics(
    prediction: torch.Tensor, actions: torch.Tensor, target_mode: str
) -> dict[str, float]:
    if target_mode == "absolute":
        period = 2.0 * torch.pi
        joint_target = torch.remainder(
            actions[:, 1:, :12] - actions[:, :1, :12] + torch.pi, period
        ) - torch.pi
        joint_error = torch.remainder(
            prediction[..., :12] - joint_target + torch.pi, period
        ) - torch.pi
    elif target_mode == "action_token_residual":
        joint_target = actions[:, 1:, :12]
        joint_error = prediction[..., :12] - joint_target
    else:
        raise ValueError(f"unsupported action auxiliary target mode: {target_mode}")
    velocity_error = (prediction[:, 1:, :12] - prediction[:, :-1, :12]) - (
        joint_target[:, 1:] - joint_target[:, :-1]
    )
    gripper_target = actions[:, 1:, 12:14]
    result = {
        "joint_wrapped_mae_rad": float(joint_error.abs().mean()),
        "joint_smooth_l1": float(F.smooth_l1_loss(prediction[..., :12], joint_target)),
        "velocity_mae": float(velocity_error.abs().mean()),
    }
    if target_mode == "absolute":
        result["gripper_loss"] = float(
            F.binary_cross_entropy_with_logits(prediction[..., 12:14], gripper_target)
        )
        result["gripper_accuracy"] = float(
            ((prediction[..., 12:14] >= 0) == (gripper_target >= 0.5)).float().mean()
        )
    else:
        result["gripper_loss"] = float(
            F.smooth_l1_loss(prediction[..., 12:14], gripper_target)
        )
        result["gripper_mae"] = float(
            (prediction[..., 12:14] - gripper_target).abs().mean()
        )
    return result


def main() -> None:
    args = parse_args()
    dataset = SpatiotemporalFeatureDataset(args.input, args.action_target_cache)
    _, validation = split_complete_indices(
        dataset, parse_ints(args.validation_episodes), validation_episodes_per_task=0
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.action_aux_head is None:
        raise ValueError("checkpoint has no action auxiliary decoder")
    device = torch.device(args.device)
    model.to(device).eval()
    loader = DataLoader(Subset(dataset, validation), batch_size=args.batch_size, shuffle=False)
    ids, actions = [], []
    with torch.inference_mode():
        for batch in loader:
            token_ids, _, _ = model.encode(
                batch["features"].to(device), batch["valid_mask"].to(device)
            )
            ids.append(token_ids.cpu())
            actions.append(batch["action_targets"])
    ids = torch.cat(ids)
    actions = torch.cat(actions)
    episodes = dataset.episode_ids[np.asarray(validation)]
    task_ids = dataset.task_ids[np.asarray(validation)]
    shuffled_ids = shuffle_within_episodes(ids, task_ids, episodes, args.seed)
    target_mode = model.action_aux_target_mode
    aligned = metrics(predict(model, ids, device, args.batch_size), actions, target_mode)
    shuffled = metrics(
        predict(model, shuffled_ids, device, args.batch_size), actions, target_mode
    )
    zero = metrics(torch.zeros_like(actions[:, 1:]), actions, target_mode)
    report = {
        "schema_version": 1,
        "kind": "frozen_action_aux_token_correspondence",
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_episodes": sorted(set(int(value) for value in episodes)),
        "validation_complete_clips": len(validation),
        "shuffle_seed": args.seed,
        "target_mode": target_mode,
        "aligned": aligned,
        "episode_shuffled": shuffled,
        "zero_residual": zero,
        "aligned_relative_improvement": {
            key: (shuffled[key] - aligned[key]) / shuffled[key]
            for key in ("joint_wrapped_mae_rad", "joint_smooth_l1", "velocity_mae", "gripper_loss")
        },
        "aligned_relative_improvement_vs_zero": {
            key: (zero[key] - aligned[key]) / zero[key]
            for key in ("joint_wrapped_mae_rad", "joint_smooth_l1", "velocity_mae", "gripper_loss")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
