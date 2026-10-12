#!/usr/bin/env python3
"""Measure action information added by physics IDs beyond state/action IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from export_spatiotemporal_physics_token_ids import build_model as build_physics_tokenizer
from evaluate_action_tokenizer import build_model as build_action_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-cache", type=Path, required=True)
    parser.add_argument("--shuffled-cache", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-vocab-size", type=int, default=128)
    parser.add_argument("--physics-vocab-size", type=int, default=128)
    parser.add_argument("--tokens-per-transition", type=int, default=16)
    parser.add_argument(
        "--physics-representation",
        choices=(
            "transition_histogram",
            "positional_count_sketch",
            "aux_decoder_prediction",
            "residual_token_decoder",
        ),
        default="transition_histogram",
    )
    parser.add_argument("--physics-tokenizer-checkpoint", type=Path, default=None)
    parser.add_argument("--residual-token-count", type=int, default=16)
    parser.add_argument("--physics-sketch-dim", type=int, default=2048)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_episode(dataset_dir: Path, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    matches = sorted(dataset_dir.glob(f"data/**/episode_{episode_id:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError(f"expected one parquet for episode {episode_id}, got {matches}")
    table = pq.read_table(matches[0], columns=["observation.state", "action"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    action[:, :12] = np.remainder(action[:, :12] + np.pi, 2.0 * np.pi) - np.pi
    action[:, 12:14] = (action[:, 12:14] > 0.49).astype(np.float32)
    return state, action


def build_targets(
    episode_ids: np.ndarray,
    base_indices: np.ndarray,
    dataset_dir: Path,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = {
        int(episode): load_episode(dataset_dir, int(episode))
        for episode in np.unique(episode_ids)
    }
    states, targets, masks = [], [], []
    for episode, base in zip(episode_ids, base_indices):
        state, action = episodes[int(episode)]
        base = int(base)
        if not 0 <= base < len(action):
            raise ValueError(f"base index {base} outside episode {episode}")
        indices = np.minimum(np.arange(base, base + horizon), len(action) - 1)
        states.append(state[base])
        targets.append(action[indices])
        masks.append(np.arange(base, base + horizon) < len(action))
    return np.stack(states), np.stack(targets), np.stack(masks)


def one_hot_action(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError("action token outside configured vocabulary")
    eye = np.eye(vocab_size, dtype=np.float32)
    return eye[ids].reshape(len(ids), -1)


def transition_histograms(
    ids: np.ndarray, vocab_size: int, tokens_per_transition: int
) -> np.ndarray:
    if ids.shape[1] % tokens_per_transition:
        raise ValueError("physics token count is not transition-aligned")
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError("physics token outside configured vocabulary")
    transitions = ids.reshape(len(ids), -1, tokens_per_transition)
    eye = np.eye(vocab_size, dtype=np.float32)
    return eye[transitions].mean(axis=2).reshape(len(ids), -1)


def positional_count_sketch(
    ids: np.ndarray, vocab_size: int, output_dim: int
) -> np.ndarray:
    """Hash position-specific categorical IDs without erasing spatial identity."""
    if output_dim < 1:
        raise ValueError("physics sketch dimension must be positive")
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError("physics token outside configured vocabulary")
    positions = np.arange(ids.shape[1], dtype=np.int64)[None, :]
    categories = positions * vocab_size + ids.astype(np.int64)
    buckets = (categories * 2654435761 + 1013904223) % output_dim
    signs = np.where(
        ((categories * 2246822519 + 3266489917) & 1) == 0, 1.0, -1.0
    ).astype(np.float32)
    features = np.zeros((len(ids), output_dim), dtype=np.float32)
    rows = np.broadcast_to(np.arange(len(ids))[:, None], ids.shape)
    np.add.at(features, (rows, buckets), signs)
    return features / np.sqrt(ids.shape[1])


def auxiliary_decoder_features(
    ids: np.ndarray, checkpoint_path: Path, device: str, batch_size: int = 256
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_physics_tokenizer(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.action_aux_head is None:
        raise ValueError("physics tokenizer checkpoint has no action auxiliary decoder")
    target_device = torch.device(device)
    model.to(target_device).eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch_ids = torch.as_tensor(
                ids[start : start + batch_size], dtype=torch.long, device=target_device
            )
            quantized = model.quantizer.decode(batch_ids).reshape(
                len(batch_ids), model.num_transitions, model.queries_per_transition, -1
            )
            predictions.append(model.predict_action_aux(quantized).cpu().numpy())
    return np.concatenate(predictions, axis=0).reshape(len(ids), -1).astype(np.float32)


def residual_token_decoder_features(
    ids: np.ndarray,
    checkpoint_path: Path,
    residual_token_count: int,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_action_tokenizer(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.num_tokens != residual_token_count:
        raise ValueError("residual token count does not match tokenizer checkpoint")
    target_device = torch.device(device)
    model.to(target_device).eval()
    residual_ids = ids[:, -residual_token_count:]
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(residual_ids), batch_size):
            batch_ids = torch.as_tensor(
                residual_ids[start : start + batch_size],
                dtype=torch.long,
                device=target_device,
            )
            predictions.append(model.decode_tokens(batch_ids).cpu().numpy())
    return np.concatenate(predictions, axis=0).reshape(len(ids), -1).astype(np.float32)


def standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return tuple((values - mean) / scale for values in (train, *others))


def fit_ridge(
    x: np.ndarray, y: np.ndarray, alpha: float, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
    bias = y_tensor.mean(dim=0, keepdim=True)
    centered = y_tensor - bias
    gram = x_tensor.T @ x_tensor
    gram.diagonal().add_(alpha)
    weights = torch.linalg.solve(gram, x_tensor.T @ centered)
    return weights, bias


def predict(x: np.ndarray, model: tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
    weights, bias = model
    values = torch.as_tensor(x, dtype=torch.float32, device=weights.device) @ weights + bias
    return values.detach().cpu().numpy()


def masked_metrics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    valid = np.broadcast_to(mask[:, :, None], target.shape)
    difference = prediction - target
    joint_difference = np.remainder(difference[:, :, :12] + np.pi, 2.0 * np.pi) - np.pi
    joint_valid = valid[:, :, :12]
    gripper_valid = valid[:, :, 12:14]
    return {
        "action_mae": float(np.abs(difference)[valid].mean()),
        "action_rmse": float(np.sqrt(np.square(difference)[valid].mean())),
        "joint_wrapped_mae_rad": float(np.abs(joint_difference)[joint_valid].mean()),
        "gripper_accuracy": float(
            ((prediction[:, :, 12:14] >= 0.5) == (target[:, :, 12:14] >= 0.5))[
                gripper_valid
            ].mean()
        ),
    }


def main() -> None:
    args = parse_args()
    aligned = load_cache(args.aligned_cache.resolve())
    shuffled = load_cache(args.shuffled_cache.resolve())
    for key in ("episode_ids", "base_indices", "action_token_ids"):
        if not np.array_equal(aligned[key], shuffled[key]):
            raise ValueError(f"aligned and shuffled cache differ in {key}")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    validation_episodes = {int(value) for value in manifest["validation_episode_ids"]}
    all_episodes = sorted({int(value) for value in aligned["episode_ids"]})
    train_episodes = [value for value in all_episodes if value not in validation_episodes]
    calibration_episodes = set(train_episodes[-5:])
    fit_episodes = set(train_episodes[:-5])

    states, targets, masks = build_targets(
        aligned["episode_ids"],
        aligned["base_indices"],
        args.dataset_dir.resolve(),
        args.action_horizon,
    )
    action_features = one_hot_action(aligned["action_token_ids"], args.action_vocab_size)
    base_features = np.concatenate((states, action_features), axis=1)
    if args.physics_representation == "transition_histogram":
        build_physics_features = lambda values: transition_histograms(
            values, args.physics_vocab_size, args.tokens_per_transition
        )
    elif args.physics_representation == "positional_count_sketch":
        build_physics_features = lambda values: positional_count_sketch(
            values, args.physics_vocab_size, args.physics_sketch_dim
        )
    elif args.physics_representation == "aux_decoder_prediction":
        if args.physics_tokenizer_checkpoint is None:
            raise ValueError(
                "aux_decoder_prediction requires --physics-tokenizer-checkpoint"
            )
        checkpoint_path = args.physics_tokenizer_checkpoint.resolve()
        build_physics_features = lambda values: auxiliary_decoder_features(
            values, checkpoint_path, args.device
        )
    else:
        if args.physics_tokenizer_checkpoint is None:
            raise ValueError(
                "residual_token_decoder requires --physics-tokenizer-checkpoint"
            )
        checkpoint_path = args.physics_tokenizer_checkpoint.resolve()
        build_physics_features = lambda values: residual_token_decoder_features(
            values,
            checkpoint_path,
            args.residual_token_count,
            args.device,
        )
    aligned_physics = build_physics_features(aligned["physics_token_ids"])
    shuffled_physics = build_physics_features(shuffled["physics_token_ids"])
    physics_feature_sets = {
        "plus_aligned_physics": aligned_physics,
        "plus_episode_shuffled_physics": shuffled_physics,
    }
    episode_ids = aligned["episode_ids"]
    fit_mask = np.isin(episode_ids, list(fit_episodes))
    calibration_mask = np.isin(episode_ids, list(calibration_episodes))
    train_mask = ~np.isin(episode_ids, list(validation_episodes))
    validation_mask = ~train_mask
    flat_targets = targets.reshape(len(targets), -1)
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)
    fit_base_x, calibration_base_x = standardize(
        base_features[fit_mask], base_features[calibration_mask]
    )
    base_candidates = {}
    for alpha in alphas:
        model = fit_ridge(fit_base_x, flat_targets[fit_mask], alpha, args.device)
        prediction = predict(calibration_base_x, model).reshape(
            -1, args.action_horizon, targets.shape[-1]
        )
        base_candidates[str(alpha)] = masked_metrics(
            prediction, targets[calibration_mask], masks[calibration_mask]
        )
    base_alpha = min(
        alphas, key=lambda value: base_candidates[str(value)]["action_mae"]
    )
    inner_base_model = fit_ridge(
        fit_base_x, flat_targets[fit_mask], base_alpha, args.device
    )
    fit_base_prediction = predict(fit_base_x, inner_base_model)
    calibration_base_prediction = predict(calibration_base_x, inner_base_model)

    train_base_x, validation_base_x = standardize(
        base_features[train_mask], base_features[validation_mask]
    )
    final_base_model = fit_ridge(
        train_base_x, flat_targets[train_mask], base_alpha, args.device
    )
    train_base_prediction = predict(train_base_x, final_base_model)
    validation_base_prediction = predict(validation_base_x, final_base_model)
    validation_base_trajectory = validation_base_prediction.reshape(
        -1, args.action_horizon, targets.shape[-1]
    )
    results = {
        "state_action_tokens": {
            "feature_dim": int(base_features.shape[1]),
            "selected_alpha": base_alpha,
            "calibration": base_candidates,
            "validation": masked_metrics(
                validation_base_trajectory,
                targets[validation_mask],
                masks[validation_mask],
            ),
        }
    }
    for name, physics_features in physics_feature_sets.items():
        fit_physics_x, calibration_physics_x = standardize(
            physics_features[fit_mask], physics_features[calibration_mask]
        )
        fit_residual = flat_targets[fit_mask] - fit_base_prediction
        candidates = {}
        for alpha in alphas:
            residual_model = fit_ridge(
                fit_physics_x, fit_residual, alpha, args.device
            )
            prediction = calibration_base_prediction + predict(
                calibration_physics_x, residual_model
            )
            candidates[str(alpha)] = masked_metrics(
                prediction.reshape(-1, args.action_horizon, targets.shape[-1]),
                targets[calibration_mask],
                masks[calibration_mask],
            )
        selected_alpha = min(
            alphas, key=lambda value: candidates[str(value)]["action_mae"]
        )
        train_physics_x, validation_physics_x = standardize(
            physics_features[train_mask], physics_features[validation_mask]
        )
        train_residual = flat_targets[train_mask] - train_base_prediction
        residual_model = fit_ridge(
            train_physics_x, train_residual, selected_alpha, args.device
        )
        prediction = validation_base_prediction + predict(
            validation_physics_x, residual_model
        )
        results[name] = {
            "feature_dim": int(base_features.shape[1] + physics_features.shape[1]),
            "physics_residual_feature_dim": int(physics_features.shape[1]),
            "selected_alpha": selected_alpha,
            "calibration": candidates,
            "validation": masked_metrics(
                prediction.reshape(-1, args.action_horizon, targets.shape[-1]),
                targets[validation_mask],
                masks[validation_mask],
            ),
        }

    baseline = results["state_action_tokens"]["validation"]
    for name in ("plus_aligned_physics", "plus_episode_shuffled_physics"):
        metrics = results[name]["validation"]
        metrics["action_mae_relative_improvement_vs_baseline"] = (
            baseline["action_mae"] - metrics["action_mae"]
        ) / baseline["action_mae"]
    aligned_metrics = results["plus_aligned_physics"]["validation"]
    shuffled_metrics = results["plus_episode_shuffled_physics"]["validation"]
    report = {
        "schema_version": 1,
        "kind": "physics_action_relevance_residual_ridge_probe",
        "representation": (
            "fixed state/action-ID ridge baseline + physics residual ridge "
            f"({args.physics_representation})"
        ),
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
        "results": results,
        "aligned_minus_shuffled_action_mae_relative_improvement": (
            shuffled_metrics["action_mae"] - aligned_metrics["action_mae"]
        ) / shuffled_metrics["action_mae"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
