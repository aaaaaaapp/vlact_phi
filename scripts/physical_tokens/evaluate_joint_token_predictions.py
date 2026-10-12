#!/usr/bin/env python3
"""Evaluate joint action/physics token predictions on a fixed held-out panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    from .evaluate_qwenpi_oracle import (
        build_validation_dataset,
        load_model,
        materialize_sample,
        select_panel,
        sha256_file,
    )
except ImportError:
    from evaluate_qwenpi_oracle import (
        build_validation_dataset,
        load_model,
        materialize_sample,
        select_panel,
        sha256_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stream_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    matches = predicted == target
    return {
        "token_accuracy": float(matches.mean()),
        "sequence_exact_accuracy": float(matches.all(axis=1).mean()),
        "per_position_accuracy": matches.mean(axis=0).astype(float).tolist(),
        "predicted_unique_codes": int(np.unique(predicted).size),
        "target_unique_codes": int(np.unique(target).size),
        "predicted_code_histogram": np.bincount(predicted.reshape(-1)).astype(int).tolist(),
        "target_code_histogram": np.bincount(target.reshape(-1)).astype(int).tolist(),
    }


def train_position_majority_baseline(
    cache_path: Path,
    task_name: str,
    validation_episodes: list[int],
    action_targets: np.ndarray,
    physics_targets: np.ndarray,
) -> dict:
    """Evaluate a no-input per-position prior fitted only on training episodes."""
    with np.load(cache_path, allow_pickle=False) as cache:
        task_names = cache["task_names"].astype(str)
        task_matches = np.flatnonzero(task_names == task_name)
        if len(task_matches) != 1:
            raise ValueError(
                f"task {task_name!r} did not resolve uniquely in token cache"
            )
        train_mask = (cache["task_ids"] == int(task_matches[0])) & ~np.isin(
            cache["episode_ids"], validation_episodes
        )
        if not train_mask.any():
            raise ValueError(f"token cache has no training rows for task {task_name!r}")
        train_action = cache["action_token_ids"][train_mask]
        train_physics = cache["physics_token_ids"][train_mask]

    def evaluate(train_ids: np.ndarray, targets: np.ndarray) -> dict:
        if train_ids.shape[1] != targets.shape[1]:
            raise ValueError("majority baseline position count does not match targets")
        majority_ids = np.asarray(
            [np.bincount(train_ids[:, index]).argmax() for index in range(train_ids.shape[1])],
            dtype=np.int64,
        )
        predictions = np.broadcast_to(majority_ids, targets.shape)
        metrics = stream_metrics(predictions, targets)
        metrics["majority_token_ids"] = majority_ids.tolist()
        return metrics

    return {
        "fit_split": "task-local cache rows excluding validation_episode_ids",
        "action": evaluate(train_action, action_targets),
        "physics": evaluate(train_physics, physics_targets),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    cache_path = args.token_cache.resolve()
    config = OmegaConf.load(config_path)
    if str(config.framework.name) != "QwenPIv4JointTokens":
        raise ValueError("joint token evaluation requires QwenPIv4JointTokens")
    config.datasets.vla_data.discrete_token_cache = str(cache_path)
    config.framework.joint_token_prediction.condition_source = "predicted"
    validation_episodes = [
        int(value) for value in config.datasets.vla_data.exclude_episode_ids
    ]
    dataset = build_validation_dataset(config, validation_episodes, args.task_name)
    panel = select_panel(dataset, args.num_samples, args.seed)
    device = torch.device(args.device)
    model = load_model(config, checkpoint_path, "b3", device)

    action_predictions = []
    action_targets = []
    physics_predictions = []
    physics_targets = []
    for ordinal, panel_item in enumerate(panel):
        sample = materialize_sample(dataset, panel_item, args.seed + ordinal)
        prediction = model.predict_token_ids([sample])
        action_predictions.append(prediction["action_token_ids"][0])
        physics_predictions.append(prediction["physics_token_ids"][0])
        action_targets.append(np.asarray(sample["action_token_ids"], dtype=np.int64))
        physics_targets.append(np.asarray(sample["physics_token_ids"], dtype=np.int64))

    action_predictions = np.stack(action_predictions)
    action_targets = np.stack(action_targets)
    physics_predictions = np.stack(physics_predictions)
    physics_targets = np.stack(physics_targets)
    result = {
        "schema_version": 1,
        "task_name": args.task_name,
        "num_samples": len(panel),
        "seed": args.seed,
        "sample_ids": panel,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "token_cache": str(cache_path),
        "token_cache_sha256": sha256_file(cache_path),
        "action": stream_metrics(action_predictions, action_targets),
        "physics": stream_metrics(physics_predictions, physics_targets),
        "train_position_majority_baseline": train_position_majority_baseline(
            cache_path,
            args.task_name,
            validation_episodes,
            action_targets,
            physics_targets,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
