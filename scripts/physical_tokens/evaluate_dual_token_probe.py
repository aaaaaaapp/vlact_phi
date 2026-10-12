#!/usr/bin/env python3
"""Evaluate joint action/physics token predictions and decoded action utility."""

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

from evaluate_action_token_reasoner import load_reasoner, load_tokenizer, token_accuracy_metrics
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
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--physics-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=380)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def add_residual(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    result = base + residual
    result[..., :12] = np.remainder(result[..., :12] + np.pi, 2.0 * np.pi) - np.pi
    result[..., 12:14] = np.clip(result[..., 12:14], 0.0, 1.0)
    return result


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    token_cache_path = args.token_cache.resolve()
    config = OmegaConf.load(config_path)
    # A frozen teacher is a training-only constraint and is not part of inference.
    config.framework.dual_token_probe.enable_action_teacher = False
    config.datasets.vla_data.discrete_token_cache = str(token_cache_path)
    validation_episodes, _ = load_validation_episodes(
        config_path, args.split_manifest.resolve()
    )
    dataset = build_validation_dataset(config, validation_episodes)
    panel = select_panel(dataset, args.num_samples, args.seed)
    samples = [
        materialize_sample(dataset, item, args.seed + ordinal)
        for ordinal, item in enumerate(panel)
    ]
    action_targets = np.stack([sample["action_token_ids"] for sample in samples])
    physics_targets = np.stack([sample["physics_token_ids"] for sample in samples])
    device = torch.device(args.device)
    model = load_reasoner(config, checkpoint_path, device)
    action_predictions, physics_predictions = [], []
    for start in range(0, len(samples), args.batch_size):
        output = model.predict_token_ids(samples[start : start + args.batch_size])
        action_predictions.append(output["action_token_ids"])
        physics_predictions.append(output["physics_token_ids"])
    action_predictions = np.concatenate(action_predictions)
    physics_predictions = np.concatenate(physics_predictions)

    action_tokenizer = load_tokenizer(args.action_tokenizer_checkpoint.resolve(), device)
    physics_tokenizer = load_tokenizer(args.physics_tokenizer_checkpoint.resolve(), device)
    fixed_base = action_tokenizer.decode_tokens(
        torch.as_tensor(action_targets, device=device), gripper_dims=(12, 13)
    ).float().cpu().numpy()
    dual_base = action_tokenizer.decode_tokens(
        torch.as_tensor(action_predictions, device=device), gripper_dims=(12, 13)
    ).float().cpu().numpy()
    oracle_residual = physics_tokenizer.decode_tokens(
        torch.as_tensor(physics_targets, device=device)
    ).float().cpu().numpy()
    predicted_residual = physics_tokenizer.decode_tokens(
        torch.as_tensor(physics_predictions, device=device)
    ).float().cpu().numpy()
    trajectories = {
        "fixed_teacher_action_tokens_only": fixed_base,
        "dual_predicted_action_tokens_only": dual_base,
        "dual_action_plus_oracle_physics": add_residual(dual_base, oracle_residual),
        "dual_action_plus_dual_predicted_physics": add_residual(
            dual_base, predicted_residual
        ),
        "fixed_teacher_action_plus_dual_predicted_physics": add_residual(
            fixed_base, predicted_residual
        ),
        "fixed_teacher_action_plus_oracle_physics": add_residual(
            fixed_base, oracle_residual
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
    dual_baseline = decoded_metrics["dual_predicted_action_tokens_only"]["action_mae"]
    fixed_baseline = decoded_metrics["fixed_teacher_action_tokens_only"]["action_mae"]
    combined = decoded_metrics["dual_action_plus_dual_predicted_physics"]
    combined["action_mae_relative_improvement_vs_dual_action_only"] = (
        dual_baseline - combined["action_mae"]
    ) / dual_baseline
    combined["action_mae_relative_improvement_vs_fixed_teacher_action"] = (
        fixed_baseline - combined["action_mae"]
    ) / fixed_baseline
    return {
        "schema_version": 1,
        "kind": "joint_action_physics_token_probe",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "token_cache": str(token_cache_path),
        "token_cache_sha256": sha256_file(token_cache_path),
        "validation_episode_ids": validation_episodes,
        "num_samples": len(samples),
        "action_token_metrics": token_accuracy_metrics(
            action_predictions, action_targets, model.action_vocab_size
        ),
        "physics_token_metrics": token_accuracy_metrics(
            physics_predictions, physics_targets, model.physics_vocab_size
        ),
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
