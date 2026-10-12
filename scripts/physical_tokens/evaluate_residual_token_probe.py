#!/usr/bin/env python3
"""Evaluate predicted residual IDs and their incremental continuous-action value."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_action_token_reasoner import (
    load_reasoner,
    load_tokenizer,
    token_accuracy_metrics,
)
from evaluate_qwenpi_oracle import (
    MetricAccumulator,
    build_validation_dataset,
    load_validation_episodes,
    materialize_sample,
    select_panel,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--residual-token-cache", type=Path, required=True)
    parser.add_argument("--predicted-action-token-cache", type=Path, required=True)
    parser.add_argument("--residual-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=380)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    return parser.parse_args()


def keyed_ids(path: Path) -> dict[tuple[int, int], np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        ids = np.asarray(data["action_token_ids"], dtype=np.int64)
        keys = zip(data["episode_ids"], data["base_indices"])
        result = {
            (int(episode), int(base)): ids[row]
            for row, (episode, base) in enumerate(keys)
        }
    if len(result) != len(ids):
        raise ValueError(f"duplicate keys in {path}")
    return result


def add_residual(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    result = base + residual
    result[..., :12] = np.remainder(result[..., :12] + np.pi, 2.0 * np.pi) - np.pi
    result[..., 12:14] = np.clip(result[..., 12:14], 0.0, 1.0)
    return result


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    residual_cache_path = args.residual_token_cache.resolve()
    config = OmegaConf.load(config_path)
    config.framework.action_token_probe.token_count = args.token_count
    config.framework.action_token_probe.vocab_size = args.vocab_size
    config.datasets.vla_data.discrete_token_cache = str(residual_cache_path)
    validation_episodes, _ = load_validation_episodes(
        config_path, args.split_manifest.resolve()
    )
    dataset = build_validation_dataset(config, validation_episodes)
    panel = select_panel(dataset, args.num_samples, args.seed)
    samples = [
        materialize_sample(dataset, item, args.seed + ordinal)
        for ordinal, item in enumerate(panel)
    ]
    targets = np.stack([sample["action_token_ids"] for sample in samples])
    device = torch.device(args.device)
    reasoner = load_reasoner(config, checkpoint_path, device)
    predictions = []
    for start in range(0, len(samples), args.batch_size):
        predictions.append(reasoner.predict_token_ids(samples[start : start + args.batch_size]))
    predictions = np.concatenate(predictions)

    residual_tokenizer = load_tokenizer(args.residual_tokenizer_checkpoint.resolve(), device)
    action_tokenizer = load_tokenizer(args.action_tokenizer_checkpoint.resolve(), device)
    predicted_action_ids = keyed_ids(args.predicted_action_token_cache)
    keys = [(int(item["episode_id"]), int(item["base_index"])) for item in panel]
    base_ids = np.stack([predicted_action_ids[key] for key in keys])
    base_actions = action_tokenizer.decode_tokens(
        torch.as_tensor(base_ids, device=device), gripper_dims=(12, 13)
    ).float().cpu().numpy()
    oracle_residual = residual_tokenizer.decode_tokens(
        torch.as_tensor(targets, device=device)
    ).float().cpu().numpy()
    predicted_residual = residual_tokenizer.decode_tokens(
        torch.as_tensor(predictions, device=device)
    ).float().cpu().numpy()
    predicted_joint_residual = predicted_residual.copy()
    predicted_joint_residual[..., 12:14] = 0.0
    predicted_gripper_residual = np.zeros_like(predicted_residual)
    predicted_gripper_residual[..., 12:14] = predicted_residual[..., 12:14]
    trajectories = {
        "predicted_action_tokens_only": base_actions,
        "plus_oracle_residual_tokens": add_residual(base_actions, oracle_residual),
        "plus_predicted_residual_tokens": add_residual(base_actions, predicted_residual),
        "plus_predicted_joint_residual_only": add_residual(
            base_actions, predicted_joint_residual
        ),
        "plus_predicted_gripper_residual_only": add_residual(
            base_actions, predicted_gripper_residual
        ),
        "plus_half_scaled_predicted_residual": add_residual(
            base_actions, 0.5 * predicted_residual
        ),
    }
    decoded_metrics = {}
    for name, values in trajectories.items():
        accumulator = MetricAccumulator()
        for index, sample in enumerate(samples):
            accumulator.update(
                values[index],
                sample["action"],
                sample.get("action_valid_mask"),
                sample.get("action_dim_valid_mask"),
            )
        decoded_metrics[name] = accumulator.compute()
    baseline = decoded_metrics["predicted_action_tokens_only"]["action_mae"]
    for name in trajectories:
        if name == "predicted_action_tokens_only":
            continue
        value = decoded_metrics[name]["action_mae"]
        decoded_metrics[name]["action_mae_relative_improvement_vs_action_tokens"] = (
            baseline - value
        ) / baseline
    return {
        "schema_version": 1,
        "kind": "observation_predicted_action_residual_token_probe",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "residual_token_cache": str(residual_cache_path),
        "residual_tokenizer_checkpoint": str(args.residual_tokenizer_checkpoint.resolve()),
        "validation_episode_ids": validation_episodes,
        "num_samples": len(samples),
        "token_count": args.token_count,
        "vocab_size": args.vocab_size,
        "token_metrics": token_accuracy_metrics(predictions, targets, args.vocab_size),
        "decoded_action_metrics": decoded_metrics,
        "sample_ids": panel,
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
