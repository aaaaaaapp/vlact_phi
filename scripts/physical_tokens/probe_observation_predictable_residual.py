#!/usr/bin/env python3
"""Test whether frozen current-image features predict useful action residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_action_token_reasoner import load_tokenizer
from evaluate_qwenpi_oracle import MetricAccumulator
from probe_physics_action_relevance import fit_ridge, predict, standardize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-feature-cache", type=Path, required=True)
    parser.add_argument("--residual-action-cache", type=Path, required=True)
    parser.add_argument("--action-token-cache", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--history-offsets", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def keyed_alignment(reference: dict, candidate: dict, name: str) -> None:
    keys = ["episode_ids", "base_indices"]
    if "task_ids" in reference:
        keys.insert(0, "task_ids")
    for key in keys:
        if key not in candidate:
            raise ValueError(f"{name} is missing key: {key}")
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{name} is not key-aligned: {key}")


def action_one_hot(ids: np.ndarray, vocab_size: int = 128) -> np.ndarray:
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError("action IDs outside vocabulary")
    return np.eye(vocab_size, dtype=np.float32)[ids].reshape(len(ids), -1)


def project_visual(
    features: np.ndarray,
    task_ids: np.ndarray,
    episode_ids: np.ndarray,
    base_indices: np.ndarray,
    history_offsets: list[int],
    output_dim: int,
    seed: int,
) -> np.ndarray:
    if not history_offsets or history_offsets[0] != 0 or any(x < 0 for x in history_offsets):
        raise ValueError("history offsets must start at zero and be non-negative")
    pooled = np.asarray(features, dtype=np.float32).mean(axis=1)
    row_by_key = {
        (int(task), int(episode), int(base)): row
        for row, (task, episode, base) in enumerate(zip(task_ids, episode_ids, base_indices))
    }
    temporal = []
    for row, (task, episode, base) in enumerate(zip(task_ids, episode_ids, base_indices)):
        current = pooled[row]
        parts = [current]
        for offset in history_offsets[1:]:
            prior = max(0, int(base) - offset)
            while (int(task), int(episode), prior) not in row_by_key and prior > 0:
                prior -= 1
            key = (int(task), int(episode), prior)
            if key not in row_by_key:
                raise ValueError(
                    f"no causal history row for task {task}, episode {episode}, base {base}"
                )
            parts.append(current - pooled[row_by_key[key]])
        temporal.append(np.concatenate(parts))
    temporal = np.stack(temporal)
    generator = np.random.default_rng(seed)
    projection = generator.standard_normal(
        (temporal.shape[1], output_dim), dtype=np.float32
    ) / np.sqrt(output_dim)
    return temporal @ projection


def episode_shuffle(
    values: np.ndarray, task_ids: np.ndarray, episode_ids: np.ndarray, seed: int
) -> np.ndarray:
    result = np.empty_like(values)
    generator = np.random.default_rng(seed)
    for task in np.unique(task_ids):
        task_mask = task_ids == task
        episodes = np.unique(episode_ids[task_mask])
        shuffled = generator.permutation(episodes)
        for target_episode, source_episode in zip(episodes, shuffled):
            target_rows = np.flatnonzero(task_mask & (episode_ids == target_episode))
            source_rows = np.flatnonzero(task_mask & (episode_ids == source_episode))
            result[target_rows] = values[np.resize(source_rows, len(target_rows))]
    return result


def task_one_hot(task_ids: np.ndarray) -> np.ndarray:
    unique = np.unique(task_ids)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError(f"task IDs must be contiguous from zero, got {unique.tolist()}")
    return np.eye(len(unique), dtype=np.float32)[task_ids]


def per_task_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    dim_mask: np.ndarray,
    task_ids: np.ndarray,
) -> tuple[dict[str, dict[str, float]], float]:
    results = {}
    maes = []
    for task in np.unique(task_ids):
        mask = task_ids == task
        value = metrics(prediction[mask], target[mask], valid_mask[mask], dim_mask)
        results[str(int(task))] = value
        maes.append(value["action_mae"])
    return results, float(np.mean(maes))


def add_residual(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    result = base + residual
    result[..., :12] = np.remainder(result[..., :12] + np.pi, 2 * np.pi) - np.pi
    result[..., 12:14] = np.clip(result[..., 12:14], 0.0, 1.0)
    return result


def metrics(
    prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray, dim_mask: np.ndarray
) -> dict[str, float]:
    accumulator = MetricAccumulator()
    for index in range(len(prediction)):
        accumulator.update(prediction[index], target[index], valid_mask[index], dim_mask)
    return accumulator.compute()


def main() -> None:
    args = parse_args()
    visual_path = args.visual_feature_cache.resolve()
    residual_path = args.residual_action_cache.resolve()
    token_path = args.action_token_cache.resolve()
    tokenizer_path = args.action_tokenizer_checkpoint.resolve()
    visual = load(visual_path)
    residual = load(residual_path)
    tokens = load(token_path)
    keyed_alignment(residual, visual, "visual feature cache")
    keyed_alignment(residual, tokens, "action token cache")
    task_ids = residual.get("task_ids", np.zeros(len(residual["episode_ids"]), dtype=np.int64)).astype(np.int64)
    episode_ids = residual["episode_ids"].astype(np.int64)
    residual_targets = residual["actions"].astype(np.float32)
    valid_mask = residual["action_valid_mask"].astype(bool)
    dim_mask = residual["action_dim_mask"].astype(bool)
    action_ids = tokens["action_token_ids"].astype(np.int64)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(tokenizer_path, device)
    with torch.inference_mode():
        base_actions = tokenizer.decode_tokens(
            torch.as_tensor(action_ids, device=device), gripper_dims=(12, 13)
        ).float().cpu().numpy()
    true_actions = add_residual(base_actions, residual_targets)

    visual_rows = visual["frame_features"]
    if "source_frame_indices" in visual:
        indices = visual["source_frame_indices"]
        if indices.ndim != 2 or indices.shape[1] < 1:
            raise ValueError("source_frame_indices must be [N,T]")
        visual_rows = visual_rows[indices[:, 0]]
    if len(visual_rows) != len(residual_targets):
        raise ValueError("visual sample rows do not match residual rows")
    projected = project_visual(
        visual_rows,
        task_ids,
        episode_ids,
        residual["base_indices"],
        args.history_offsets,
        args.projection_dim,
        args.seed,
    )
    shuffled = episode_shuffle(projected, task_ids, episode_ids, args.seed)
    action_features = action_one_hot(action_ids)
    context_features = task_one_hot(task_ids)
    feature_sets = {
        "aligned_current_visual": np.concatenate(
            (context_features, action_features, projected), axis=1
        ),
        "episode_shuffled_current_visual": np.concatenate(
            (context_features, action_features, shuffled), axis=1
        ),
    }
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    validation_episodes = set(map(int, split["validation_episode_ids"]))
    training_episodes = sorted(set(map(int, split["training_episode_ids"])))
    calibration_episodes = set(training_episodes[-5:])
    fit_episodes = set(training_episodes[:-5])
    fit_mask = np.isin(episode_ids, list(fit_episodes))
    calibration_mask = np.isin(episode_ids, list(calibration_episodes))
    train_mask = np.isin(episode_ids, training_episodes)
    validation_mask = np.isin(episode_ids, list(validation_episodes))
    flat_residual = residual_targets.reshape(len(residual_targets), -1)
    alphas = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
    results = {}
    for name, features in feature_sets.items():
        fit_x, calibration_x = standardize(features[fit_mask], features[calibration_mask])
        candidates = {}
        for alpha in alphas:
            model = fit_ridge(fit_x, flat_residual[fit_mask], alpha, args.device)
            predicted_residual = predict(calibration_x, model).reshape(
                -1, *residual_targets.shape[1:]
            )
            predicted_action = add_residual(base_actions[calibration_mask], predicted_residual)
            candidate_metrics = metrics(
                predicted_action,
                true_actions[calibration_mask],
                valid_mask[calibration_mask],
                dim_mask,
            )
            per_task, macro_mae = per_task_metrics(
                predicted_action,
                true_actions[calibration_mask],
                valid_mask[calibration_mask],
                dim_mask,
                task_ids[calibration_mask],
            )
            candidate_metrics["macro_task_action_mae"] = macro_mae
            candidate_metrics["per_task"] = per_task
            candidates[str(alpha)] = candidate_metrics
        selected_alpha = min(
            alphas, key=lambda value: candidates[str(value)]["macro_task_action_mae"]
        )
        train_x, validation_x = standardize(features[train_mask], features[validation_mask])
        model = fit_ridge(train_x, flat_residual[train_mask], selected_alpha, args.device)
        predicted_residual = predict(validation_x, model).reshape(
            -1, *residual_targets.shape[1:]
        )
        predicted_action = add_residual(base_actions[validation_mask], predicted_residual)
        validation_metrics = metrics(
            predicted_action,
            true_actions[validation_mask],
            valid_mask[validation_mask],
            dim_mask,
        )
        validation_per_task, validation_macro_mae = per_task_metrics(
            predicted_action,
            true_actions[validation_mask],
            valid_mask[validation_mask],
            dim_mask,
            task_ids[validation_mask],
        )
        validation_metrics["macro_task_action_mae"] = validation_macro_mae
        validation_metrics["per_task"] = validation_per_task
        results[name] = {
            "feature_dim": int(features.shape[1]),
            "selected_alpha": selected_alpha,
            "calibration": candidates,
            "validation": validation_metrics,
        }
    baseline = metrics(
        base_actions[validation_mask],
        true_actions[validation_mask],
        valid_mask[validation_mask],
        dim_mask,
    )
    baseline_per_task, baseline_macro_mae = per_task_metrics(
        base_actions[validation_mask],
        true_actions[validation_mask],
        valid_mask[validation_mask],
        dim_mask,
        task_ids[validation_mask],
    )
    baseline["macro_task_action_mae"] = baseline_macro_mae
    baseline["per_task"] = baseline_per_task
    for result in results.values():
        result["validation"]["action_mae_relative_improvement_vs_zero_residual"] = (
            baseline["action_mae"] - result["validation"]["action_mae"]
        ) / baseline["action_mae"]
        result["validation"]["macro_task_action_mae_relative_improvement_vs_zero_residual"] = (
            baseline_macro_mae - result["validation"]["macro_task_action_mae"]
        ) / baseline_macro_mae
    aligned_mae = results["aligned_current_visual"]["validation"]["macro_task_action_mae"]
    shuffled_mae = results["episode_shuffled_current_visual"]["validation"]["macro_task_action_mae"]
    report = {
        "schema_version": 1,
        "kind": "observation_predictable_continuous_residual_probe",
        "inputs": {
            "visual_feature_cache": str(visual_path),
            "visual_feature_cache_sha256": sha256_file(visual_path),
            "residual_action_cache": str(residual_path),
            "residual_action_cache_sha256": sha256_file(residual_path),
            "action_token_cache": str(token_path),
            "action_token_cache_sha256": sha256_file(token_path),
            "action_tokenizer_checkpoint": str(tokenizer_path),
            "action_tokenizer_checkpoint_sha256": sha256_file(tokenizer_path),
        },
        "projection": {
            "kind": "seeded_gaussian_current_plus_causal_deltas",
            "dim": args.projection_dim,
            "seed": args.seed,
            "history_offsets": args.history_offsets,
        },
        "split": {
            "fit_episodes": sorted(fit_episodes),
            "calibration_episodes": sorted(calibration_episodes),
            "validation_episodes": sorted(validation_episodes),
        },
        "num_samples": {
            "fit": int(fit_mask.sum()),
            "calibration": int(calibration_mask.sum()),
            "train_refit": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
        },
        "zero_residual_validation": baseline,
        "results": results,
        "aligned_vs_shuffled_action_mae_relative_improvement": (
            shuffled_mae - aligned_mae
        ) / shuffled_mae,
        "gate_metric": "macro_task_action_mae",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
