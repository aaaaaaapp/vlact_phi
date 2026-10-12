#!/usr/bin/env python3
"""Evaluate an action-token probe with an equal held-out panel per task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_action_token_reasoner import (
    distribution_stats,
    load_reasoner,
    load_tokenizer,
    token_accuracy_metrics,
)
from evaluate_qwenpi_oracle import MetricAccumulator, materialize_sample, select_panel, sha256_file
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def decoded_metrics(
    ids: np.ndarray, samples: list[dict], tokenizer, device: torch.device
) -> dict:
    actions = tokenizer.decode_tokens(
        torch.as_tensor(ids, dtype=torch.long, device=device), gripper_dims=(12, 13)
    ).float().cpu().numpy()
    accumulator = MetricAccumulator()
    for action, sample in zip(actions, samples):
        accumulator.update(
            action,
            sample["action"],
            sample.get("action_valid_mask"),
            sample.get("action_dim_valid_mask"),
        )
    return accumulator.compute()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.datasets.vla_data.discrete_token_cache = str(args.token_cache.resolve())
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    validation_episodes = list(map(int, split["validation_episode_ids"]))
    training_episodes = set(map(int, split["training_episode_ids"]))
    with np.load(args.token_cache, allow_pickle=False) as cache:
        task_names = [str(value) for value in cache["task_names"].tolist()]
        cache_task_ids = np.asarray(cache["task_ids"], dtype=np.int64)
        cache_episode_ids = np.asarray(cache["episode_ids"], dtype=np.int64)
        cached_targets = np.asarray(cache["action_token_ids"], dtype=np.int64)
    device = torch.device(args.device)
    model = load_reasoner(config, args.checkpoint.resolve(), device)
    tokenizer = load_tokenizer(args.action_tokenizer_checkpoint.resolve(), device)
    vocab_size = int(config.framework.action_token_probe.vocab_size)
    task_results = {}
    all_predictions, all_targets = [], []

    for task_id, task_name in enumerate(task_names):
        data_cfg = OmegaConf.create(OmegaConf.to_container(config.datasets.vla_data, resolve=True))
        data_cfg.exclude_episode_ids = None
        data_cfg.include_episode_ids = validation_episodes
        dataset = make_LeRobotSingleDataset(
            Path(str(data_cfg.data_root_dir)),
            task_name,
            "robotwin_wrap32",
            data_cfg=data_cfg,
        )
        panel = select_panel(dataset, args.samples_per_task, args.seed + task_id)
        samples = [
            materialize_sample(dataset, item, args.seed + task_id * 10000 + index)
            for index, item in enumerate(panel)
        ]
        targets = np.stack([sample["action_token_ids"] for sample in samples]).astype(np.int64)
        predicted = []
        for start in range(0, len(samples), args.batch_size):
            predicted.append(model.predict_token_ids(samples[start : start + args.batch_size]))
        predictions = np.concatenate(predicted).astype(np.int64)
        train_mask = (cache_task_ids == task_id) & np.isin(
            cache_episode_ids, list(training_episodes)
        )
        train_ids = cached_targets[train_mask]
        modal_ids = np.stack(
            [
                np.bincount(train_ids[:, position], minlength=vocab_size).argmax()
                for position in range(train_ids.shape[1])
            ]
        )
        modal_predictions = np.broadcast_to(modal_ids, targets.shape).copy()
        task_results[task_name] = {
            "num_samples": len(samples),
            "token_metrics": token_accuracy_metrics(predictions, targets, vocab_size),
            "predicted_action": decoded_metrics(predictions, samples, tokenizer, device),
            "oracle_action": decoded_metrics(targets, samples, tokenizer, device),
            "modal_token_baseline": {
                "ids": modal_ids.tolist(),
                "token_metrics": token_accuracy_metrics(modal_predictions, targets, vocab_size),
                "decoded_action": decoded_metrics(modal_predictions, samples, tokenizer, device),
            },
        }
        all_predictions.append(predictions)
        all_targets.append(targets)

    metric_names = ("accuracy", "exact_sequence_accuracy")
    macro_token = {
        name: float(np.mean([value["token_metrics"][name] for value in task_results.values()]))
        for name in metric_names
    }
    macro_action_mae = float(
        np.mean([value["predicted_action"]["action_mae"] for value in task_results.values()])
    )
    macro_oracle_mae = float(
        np.mean([value["oracle_action"]["action_mae"] for value in task_results.values()])
    )
    macro_modal_mae = float(
        np.mean(
            [value["modal_token_baseline"]["decoded_action"]["action_mae"] for value in task_results.values()]
        )
    )
    all_predictions_array = np.concatenate(all_predictions)
    all_targets_array = np.concatenate(all_targets)
    report = {
        "schema_version": 1,
        "kind": "multitask_action_token_probe_evaluation",
        "inputs": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "token_cache": str(args.token_cache.resolve()),
            "token_cache_sha256": sha256_file(args.token_cache.resolve()),
            "action_tokenizer_checkpoint": str(args.action_tokenizer_checkpoint.resolve()),
            "action_tokenizer_checkpoint_sha256": sha256_file(
                args.action_tokenizer_checkpoint.resolve()
            ),
        },
        "validation_episode_ids": validation_episodes,
        "samples_per_task": args.samples_per_task,
        "num_tasks": len(task_names),
        "macro": {
            **macro_token,
            "predicted_action_mae": macro_action_mae,
            "oracle_action_mae": macro_oracle_mae,
            "modal_token_action_mae": macro_modal_mae,
            "predicted_action_relative_improvement_vs_modal": (
                macro_modal_mae - macro_action_mae
            )
            / macro_modal_mae,
        },
        "global_token_metrics": token_accuracy_metrics(
            all_predictions_array, all_targets_array, vocab_size
        ),
        "global_prediction_distribution": distribution_stats(
            all_predictions_array, vocab_size
        ),
        "tasks": task_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
