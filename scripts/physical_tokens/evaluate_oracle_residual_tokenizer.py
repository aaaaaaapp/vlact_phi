#!/usr/bin/env python3
"""Measure incremental action value of oracle residual-token reconstructions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_action_token_reasoner import load_tokenizer
from evaluate_qwenpi_oracle import MetricAccumulator, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--predicted-action-token-cache", type=Path, required=True)
    parser.add_argument("--residual-token-cache", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-episodes", default="5,20,25,39,49")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_ids(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return (
            np.asarray(data["action_token_ids"], dtype=np.int64),
            np.asarray(data["episode_ids"], dtype=np.int64),
            np.asarray(data["base_indices"], dtype=np.int64),
        )


def add_residual(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    result = base + residual
    result[..., :12] = np.remainder(result[..., :12] + np.pi, 2.0 * np.pi) - np.pi
    result[..., 12:14] = np.clip(result[..., 12:14], 0.0, 1.0)
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    with np.load(args.action_cache.resolve(), allow_pickle=False) as data:
        actions = np.asarray(data["actions"], dtype=np.float32)
        valid = np.asarray(data["action_valid_mask"], dtype=np.bool_)
        dim_mask = np.asarray(data["action_dim_mask"], dtype=np.bool_)
        episodes = np.asarray(data["episode_ids"], dtype=np.int64)
        bases = np.asarray(data["base_indices"], dtype=np.int64)
    action_ids, action_episodes, action_bases = load_ids(args.predicted_action_token_cache)
    residual_ids, residual_episodes, residual_bases = load_ids(args.residual_token_cache)
    for name, candidate in (
        ("action episodes", action_episodes),
        ("action bases", action_bases),
        ("residual episodes", residual_episodes),
        ("residual bases", residual_bases),
    ):
        reference = episodes if "episodes" in name else bases
        if not np.array_equal(candidate, reference):
            raise ValueError(f"cache key mismatch: {name}")
    validation_episodes = tuple(int(value) for value in args.validation_episodes.split(","))
    selected = np.flatnonzero(np.isin(episodes, validation_episodes))
    device = torch.device(args.device)
    action_tokenizer = load_tokenizer(args.action_tokenizer_checkpoint.resolve(), device)
    residual_tokenizer = load_tokenizer(args.residual_tokenizer_checkpoint.resolve(), device)
    base_batches, residual_batches = [], []
    for start in range(0, len(selected), args.batch_size):
        rows = selected[start : start + args.batch_size]
        base_batches.append(
            action_tokenizer.decode_tokens(
                torch.as_tensor(action_ids[rows], device=device), gripper_dims=(12, 13)
            ).float().cpu().numpy()
        )
        residual_batches.append(
            residual_tokenizer.decode_tokens(
                torch.as_tensor(residual_ids[rows], device=device)
            ).float().cpu().numpy()
        )
    base_actions = np.concatenate(base_batches)
    combined_actions = add_residual(base_actions, np.concatenate(residual_batches))
    metrics = {}
    for name, predictions in (
        ("predicted_action_tokens_only", base_actions),
        ("plus_oracle_residual_tokens", combined_actions),
    ):
        accumulator = MetricAccumulator()
        for local, row in enumerate(selected):
            accumulator.update(predictions[local], actions[row], valid[row], dim_mask)
        metrics[name] = accumulator.compute()
    baseline = metrics["predicted_action_tokens_only"]["action_mae"]
    candidate = metrics["plus_oracle_residual_tokens"]["action_mae"]
    metrics["plus_oracle_residual_tokens"]["action_mae_relative_improvement_vs_action_tokens"] = (
        baseline - candidate
    ) / baseline
    report = {
        "schema_version": 1,
        "kind": "oracle_residual_tokenizer_action_utility",
        "residual_tokenizer_checkpoint": str(args.residual_tokenizer_checkpoint.resolve()),
        "residual_tokenizer_checkpoint_sha256": sha256_file(
            args.residual_tokenizer_checkpoint.resolve()
        ),
        "validation_episodes": list(validation_episodes),
        "validation_samples": len(selected),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
