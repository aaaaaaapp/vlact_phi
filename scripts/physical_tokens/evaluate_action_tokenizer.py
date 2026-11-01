#!/usr/bin/env python3
"""Evaluate an action tokenizer checkpoint on an episode-held-out cache split."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.tokenizers import ActionChunkTokenizer


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gripper-dims", default="12,13")
    parser.add_argument("--angular-dims", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


class CacheDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path) -> None:
        self.data = np.load(path, mmap_mode="r", allow_pickle=False)
        self.actions = self.data["actions"]
        self.valid = self.data["action_valid_mask"]
        self.episodes = self.data["episode_ids"]
        self.task_ids = self.data["task_ids"] if "task_ids" in self.data.files else np.zeros(len(self.actions), dtype=np.int64)
        self.task_names = (
            tuple(str(value) for value in self.data["task_names"].tolist())
            if "task_names" in self.data.files else ("single_task",)
        )

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.array(self.actions[index], dtype=np.float32)),
            torch.from_numpy(np.array(self.valid[index], dtype=np.bool_)),
            torch.tensor(int(self.task_ids[index]), dtype=torch.long),
        )


def summarize_code_counts(counts: torch.Tensor) -> dict[str, float | int]:
    probabilities = counts.float() / counts.sum().clamp_min(1)
    entropy = -(probabilities * (probabilities + 1e-10).log()).sum()
    return {
        "active_codes": int((counts > 0).sum().cpu()),
        "active_ratio": float((counts > 0).float().mean().cpu()),
        "entropy_nats": float(entropy.cpu()),
        "normalized_entropy": float((entropy / math.log(len(counts))).cpu()),
        "top10_fraction": float(
            (counts.topk(min(10, len(counts))).values.sum() / counts.sum().clamp_min(1)).cpu()
        ),
        "perplexity": float(entropy.exp().cpu()),
        "total_tokens": int(counts.sum().cpu()),
    }


def held_out_indices(
    episodes: np.ndarray, fraction: float, seed: int, split_manifest: Path | None = None
) -> list[int]:
    if split_manifest is not None:
        split = json.loads(split_manifest.read_text(encoding="utf-8"))
        selected = set(map(int, split["validation_episode_ids"]))
        return [index for index, episode in enumerate(episodes) if episode.item() in selected]
    unique = np.unique(episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = max(1, round(len(unique) * fraction))
    selected = set(unique[:count].tolist())
    return [index for index, episode in enumerate(episodes) if episode.item() in selected]


def build_model(config: dict) -> ActionChunkTokenizer:
    return ActionChunkTokenizer(
        action_dim=config["action_dim"],
        horizon=config["horizon"],
        num_tokens=config["num_tokens"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        quantizer=config["quantizer"],
        codebook_size=config["codebook_size"],
        fsq_levels=tuple(config["fsq_levels"]),
        ema_decay=config["ema_decay"],
        commitment_weight=config["commitment_weight"],
        dead_code_steps=config["dead_code_steps"],
    )


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    input_path = args.input or Path(checkpoint["input"]["path"])
    dataset = CacheDataset(input_path)
    validation_indices = held_out_indices(
        dataset.episodes, args.validation_fraction, args.seed, args.split_manifest
    )
    loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    angular_dims = torch.tensor(parse_ints(args.angular_dims), device=device)
    gripper_dims = torch.tensor(parse_ints(args.gripper_dims), device=device)
    code_counts = torch.zeros(model.codebook_size, dtype=torch.long, device=device)
    task_counts = torch.zeros(len(dataset.task_names), model.codebook_size, dtype=torch.long, device=device)
    position_counts = torch.zeros(model.num_tokens, model.codebook_size, dtype=torch.long, device=device)
    joint_error_sum = torch.zeros((), device=device)
    joint_count = torch.zeros((), device=device)
    velocity_error_sum = torch.zeros((), device=device)
    velocity_count = torch.zeros((), device=device)
    gripper_error_sum = torch.zeros((), device=device)
    gripper_correct = torch.zeros((), device=device)
    gripper_count = torch.zeros((), device=device)
    task_sums = torch.zeros(len(dataset.task_names), 6, device=device)
    reconstruction_sum = torch.zeros((), device=device)
    reconstruction_squared_sum = torch.zeros((), device=device)
    reconstruction_count = torch.zeros((), device=device)
    task_reconstruction_sums = torch.zeros(len(dataset.task_names), 3, device=device)

    for actions, valid, task_ids in loader:
        actions = actions.to(device)
        valid = valid.to(device)
        task_ids = task_ids.to(device)
        output = model(actions, gripper_dims=gripper_dims.tolist())
        prediction = output["reconstruction"]
        ids = output["token_ids"]
        reconstruction_mask = valid.unsqueeze(-1).expand_as(prediction)
        reconstruction_difference = prediction - actions
        reconstruction_smooth_l1 = F.smooth_l1_loss(prediction, actions, reduction="none")
        reconstruction_sum += reconstruction_smooth_l1.masked_select(reconstruction_mask).sum()
        reconstruction_squared_sum += reconstruction_difference.square().masked_select(reconstruction_mask).sum()
        reconstruction_count += reconstruction_mask.sum()
        code_counts += torch.bincount(ids.flatten(), minlength=model.codebook_size)
        for task_id in task_ids.unique().tolist():
            task_counts[task_id] += torch.bincount(ids[task_ids == task_id].flatten(), minlength=model.codebook_size)
        for position in range(model.num_tokens):
            position_counts[position] += torch.bincount(ids[:, position], minlength=model.codebook_size)

        joint_difference = prediction.index_select(-1, angular_dims) - actions.index_select(-1, angular_dims)
        joint_difference = torch.remainder(joint_difference + math.pi, 2 * math.pi) - math.pi
        joint_mask = valid.unsqueeze(-1).expand_as(joint_difference)
        joint_error_sum += joint_difference.abs().masked_select(joint_mask).sum()
        joint_count += joint_mask.sum()

        predicted_velocity = prediction[:, 1:].index_select(-1, angular_dims) - prediction[:, :-1].index_select(-1, angular_dims)
        target_velocity = actions[:, 1:].index_select(-1, angular_dims) - actions[:, :-1].index_select(-1, angular_dims)
        velocity_mask = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1).expand_as(predicted_velocity)
        velocity_error_sum += (predicted_velocity - target_velocity).abs().masked_select(velocity_mask).sum()
        velocity_count += velocity_mask.sum()

        predicted_gripper = prediction.index_select(-1, gripper_dims)
        target_gripper = actions.index_select(-1, gripper_dims)
        gripper_mask = valid.unsqueeze(-1).expand_as(predicted_gripper)
        gripper_error_sum += (predicted_gripper - target_gripper).abs().masked_select(gripper_mask).sum()
        gripper_correct += ((predicted_gripper >= 0.5) == (target_gripper >= 0.5)).masked_select(gripper_mask).sum()
        gripper_count += gripper_mask.sum()
        for task_id in task_ids.unique().tolist():
            selected = task_ids == task_id
            selected_reconstruction = reconstruction_mask[selected]
            task_reconstruction_sums[task_id, 0] += reconstruction_smooth_l1[selected].masked_select(selected_reconstruction).sum()
            task_reconstruction_sums[task_id, 1] += reconstruction_difference[selected].square().masked_select(selected_reconstruction).sum()
            task_reconstruction_sums[task_id, 2] += selected_reconstruction.sum()
            selected_joint = joint_mask[selected]
            selected_velocity = velocity_mask[selected]
            selected_gripper = gripper_mask[selected]
            task_sums[task_id, 0] += joint_difference[selected].abs().masked_select(selected_joint).sum()
            task_sums[task_id, 1] += selected_joint.sum()
            task_sums[task_id, 2] += (predicted_velocity[selected] - target_velocity[selected]).abs().masked_select(selected_velocity).sum()
            task_sums[task_id, 3] += selected_velocity.sum()
            task_sums[task_id, 4] += ((predicted_gripper[selected] >= 0.5) == (target_gripper[selected] >= 0.5)).masked_select(selected_gripper).sum()
            task_sums[task_id, 5] += selected_gripper.sum()

    joint_mae_rad = joint_error_sum / joint_count.clamp_min(1)
    task_metrics = {
        name: {
            "reconstruction_smooth_l1": float((task_reconstruction_sums[task_id, 0] / task_reconstruction_sums[task_id, 2].clamp_min(1)).cpu()),
            "reconstruction_rmse": float(torch.sqrt(task_reconstruction_sums[task_id, 1] / task_reconstruction_sums[task_id, 2].clamp_min(1)).cpu()),
            "joint_wrapped_mae_deg": float(torch.rad2deg(task_sums[task_id, 0] / task_sums[task_id, 1].clamp_min(1)).cpu()),
            "joint_velocity_mae_rad": float((task_sums[task_id, 2] / task_sums[task_id, 3].clamp_min(1)).cpu()),
            "gripper_accuracy": float((task_sums[task_id, 4] / task_sums[task_id, 5].clamp_min(1)).cpu()),
        }
        for task_id, name in enumerate(dataset.task_names)
    }
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "input": str(input_path.resolve()),
        "validation_episodes": len(
            set(dataset.episodes[validation_indices].astype(int).tolist())
        ),
        "validation_task_episodes": len(
            set(
                zip(
                    dataset.task_ids[validation_indices].astype(int).tolist(),
                    dataset.episodes[validation_indices].astype(int).tolist(),
                )
            )
        ),
        "validation_chunks": len(validation_indices),
        "reconstruction_smooth_l1": float((reconstruction_sum / reconstruction_count.clamp_min(1)).cpu()),
        "reconstruction_rmse": float(torch.sqrt(reconstruction_squared_sum / reconstruction_count.clamp_min(1)).cpu()),
        "joint_wrapped_mae_rad": float(joint_mae_rad.cpu()),
        "joint_wrapped_mae_deg": float(torch.rad2deg(joint_mae_rad).cpu()),
        "joint_velocity_mae_rad": float((velocity_error_sum / velocity_count.clamp_min(1)).cpu()),
        "gripper_mae": float((gripper_error_sum / gripper_count.clamp_min(1)).cpu()),
        "gripper_accuracy": float((gripper_correct / gripper_count.clamp_min(1)).cpu()),
        **summarize_code_counts(code_counts),
        "task_code_usage": {
            name: summarize_code_counts(task_counts[task_id])
            for task_id, name in enumerate(dataset.task_names)
        },
        "task_metrics": task_metrics,
        "macro_metrics": {
            key: float(np.mean([entry[key] for entry in task_metrics.values()]))
            for key in ("reconstruction_smooth_l1", "reconstruction_rmse", "joint_wrapped_mae_deg", "joint_velocity_mae_rad", "gripper_accuracy")
        },
        "position_code_usage": [summarize_code_counts(counts) for counts in position_counts],
    }


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    output = args.output or args.checkpoint.with_name(args.checkpoint.stem + "-evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
