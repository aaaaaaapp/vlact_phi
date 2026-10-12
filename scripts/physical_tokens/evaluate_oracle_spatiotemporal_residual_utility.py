#!/usr/bin/env python3
"""Test whether oracle spatiotemporal IDs improve predicted-action residuals."""

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
from evaluate_qwenpi_oracle import sha256_file
from export_spatiotemporal_physics_token_ids import build_model as build_physics_tokenizer
from probe_physics_action_relevance import build_targets, masked_metrics


DECODER_INDICES = np.asarray(list(range(2, 32, 2)) + [31], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-action-cache", type=Path, required=True)
    parser.add_argument("--physics-cache", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--physics-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-episode", type=int, default=6)
    parser.add_argument("--validation-episodes", default="7,8")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def keys(arrays: dict[str, np.ndarray]) -> list[tuple[int, int, int]]:
    return [
        tuple(map(int, value))
        for value in zip(arrays["task_ids"], arrays["episode_ids"], arrays["base_indices"])
    ]


@torch.inference_mode()
def decode_action(ids: np.ndarray, model, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(ids), batch_size):
        batch = torch.as_tensor(ids[start : start + batch_size], dtype=torch.long, device=device)
        outputs.append(model.decode_tokens(batch, gripper_dims=(12, 13)).float().cpu().numpy())
    return np.concatenate(outputs)


@torch.inference_mode()
def decode_residual(ids: np.ndarray, model, device: torch.device, batch_size: int) -> np.ndarray:
    if model.action_aux_head is None:
        raise ValueError("physics tokenizer has no action auxiliary decoder")
    outputs = []
    for start in range(0, len(ids), batch_size):
        batch = torch.as_tensor(ids[start : start + batch_size], dtype=torch.long, device=device)
        quantized = model.quantizer.decode(batch).reshape(
            len(batch), model.num_transitions, model.queries_per_transition, -1
        )
        outputs.append(model.predict_action_aux(quantized).float().cpu().numpy())
    return np.concatenate(outputs)


def add_residual(base: np.ndarray, residual: np.ndarray, scale: float) -> np.ndarray:
    result = base[:, DECODER_INDICES].copy() + scale * residual
    result[..., :12] = np.remainder(result[..., :12] + np.pi, 2.0 * np.pi) - np.pi
    result[..., 12:14] = np.clip(result[..., 12:14], 0.0, 1.0)
    return result


def evaluate(
    base: np.ndarray,
    residual: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> dict[str, float]:
    return masked_metrics(
        add_residual(base, residual, scale), target[:, DECODER_INDICES], mask[:, DECODER_INDICES]
    )


def macro(tasks: dict, section: str, metric: str = "action_mae") -> float:
    return float(np.mean([task[section][metric] for task in tasks.values()]))


def main() -> None:
    args = parse_args()
    action_path = args.predicted_action_cache.resolve()
    physics_path = args.physics_cache.resolve()
    action = load(action_path)
    physics = load(physics_path)
    if not np.array_equal(action["task_names"], physics["task_names"]):
        raise ValueError("task_names differ between caches")
    action_rows = {key: row for row, key in enumerate(keys(action))}
    physics_keys = keys(physics)
    if len(action_rows) != len(keys(action)) or len(set(physics_keys)) != len(physics_keys):
        raise ValueError("cache contains duplicate qualified keys")
    missing = [key for key in physics_keys if key not in action_rows]
    if missing:
        raise ValueError(f"predicted action cache is missing {len(missing)} keys")
    action_ids = action["action_token_ids"][
        np.asarray([action_rows[key] for key in physics_keys], dtype=np.int64)
    ]

    device = torch.device(args.device)
    action_tokenizer = load_tokenizer(args.action_tokenizer_checkpoint.resolve(), device)
    physics_state = torch.load(
        args.physics_tokenizer_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    physics_tokenizer = build_physics_tokenizer(physics_state["model_config"])
    physics_tokenizer.load_state_dict(physics_state["model"], strict=True)
    physics_tokenizer.to(device).eval()
    decoded_action = decode_action(action_ids, action_tokenizer, device, args.batch_size)
    decoded_residual = decode_residual(
        physics["physics_token_ids"], physics_tokenizer, device, args.batch_size
    )

    validation_episodes = {
        int(value) for value in args.validation_episodes.split(",") if value
    }
    scales = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5)
    task_results = {}
    for task_id, task_name in enumerate(physics["task_names"].tolist()):
        task_mask = physics["task_ids"] == task_id
        episode_ids = physics["episode_ids"][task_mask]
        _, targets, masks = build_targets(
            episode_ids,
            physics["base_indices"][task_mask],
            args.dataset_root.resolve() / str(task_name),
            32,
        )
        base = decoded_action[task_mask]
        residual = decoded_residual[task_mask]
        calibration = episode_ids == args.calibration_episode
        validation = np.isin(episode_ids, list(validation_episodes))
        if not calibration.any() or not validation.any():
            raise ValueError(f"task {task_name} lacks calibration or validation samples")
        calibration_metrics = {
            str(scale): evaluate(
                base[calibration], residual[calibration], targets[calibration], masks[calibration], scale
            )
            for scale in scales
        }
        selected_scale = min(
            scales, key=lambda value: calibration_metrics[str(value)]["action_mae"]
        )
        rng = np.random.default_rng(args.seed + task_id)
        shuffled = residual[validation][rng.permutation(int(validation.sum()))]
        task_results[str(task_name)] = {
            "num_calibration_samples": int(calibration.sum()),
            "num_validation_samples": int(validation.sum()),
            "selected_scale": selected_scale,
            "calibration": calibration_metrics,
            "action_only": evaluate(
                base[validation], residual[validation], targets[validation], masks[validation], 0.0
            ),
            "oracle_physics_scale1": evaluate(
                base[validation], residual[validation], targets[validation], masks[validation], 1.0
            ),
            "oracle_physics_calibrated": evaluate(
                base[validation],
                residual[validation],
                targets[validation],
                masks[validation],
                selected_scale,
            ),
            "shuffled_oracle_physics_calibrated": evaluate(
                base[validation], shuffled, targets[validation], masks[validation], selected_scale
            ),
        }

    action_only = macro(task_results, "action_only")
    scale1 = macro(task_results, "oracle_physics_scale1")
    calibrated = macro(task_results, "oracle_physics_calibrated")
    shuffled = macro(task_results, "shuffled_oracle_physics_calibrated")
    report = {
        "schema_version": 1,
        "kind": "d23_oracle_spatiotemporal_residual_utility",
        "inputs": {
            "predicted_action_cache": str(action_path),
            "predicted_action_cache_sha256": sha256_file(action_path),
            "physics_cache": str(physics_path),
            "physics_cache_sha256": sha256_file(physics_path),
            "physics_tokenizer_checkpoint": str(args.physics_tokenizer_checkpoint.resolve()),
            "physics_tokenizer_checkpoint_sha256": sha256_file(
                args.physics_tokenizer_checkpoint.resolve()
            ),
        },
        "calibration_episode": args.calibration_episode,
        "validation_episode_ids": sorted(validation_episodes),
        "macro": {
            "action_only_mae": action_only,
            "oracle_physics_scale1_mae": scale1,
            "oracle_physics_calibrated_mae": calibrated,
            "shuffled_oracle_physics_calibrated_mae": shuffled,
            "scale1_improvement_vs_action_only": (action_only - scale1) / action_only,
            "calibrated_improvement_vs_action_only": (action_only - calibrated) / action_only,
            "calibrated_improvement_vs_shuffled": (shuffled - calibrated) / shuffled,
        },
        "tasks": task_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
