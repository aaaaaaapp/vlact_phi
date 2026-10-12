#!/usr/bin/env python3
"""Test whether oracle physics IDs add value beyond decoded action tokens."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_action_tokenizer import build_model as build_action_tokenizer
from export_spatiotemporal_physics_token_ids import build_model as build_physics_tokenizer
from train_frozen_physics_residual_decoder import exact_keys, load, loss_terms


DECODER_INDICES = np.asarray(list(range(2, 32, 2)) + [31], dtype=np.int64)
FEATURE_INDEX_FIELDS = (
    "task_ids",
    "episode_ids",
    "base_indices",
    "future_valid_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-cache", type=Path, required=True)
    parser.add_argument("--action-token-cache", type=Path, required=True)
    parser.add_argument("--action-target-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--physics-checkpoint", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-episodes", default="7,8")
    parser.add_argument(
        "--excluded-episodes",
        default="",
        help="Episodes excluded from both training and internal checkpoint selection",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def join_action_ids(physics: dict[str, np.ndarray], actions: dict[str, np.ndarray]) -> np.ndarray:
    action_keys = exact_keys(actions)
    if len(set(action_keys)) != len(action_keys):
        raise ValueError("duplicate qualified keys in action-token cache")
    lookup = {key: index for index, key in enumerate(action_keys)}
    missing = [key for key in exact_keys(physics) if key not in lookup]
    if missing:
        raise ValueError(f"action-token cache is missing {len(missing)} physics keys")
    return np.asarray(
        [actions["action_token_ids"][lookup[key]] for key in exact_keys(physics)],
        dtype=np.int64,
    )


def load_feature_index(path: Path) -> dict[str, np.ndarray]:
    """Load only split metadata, not the multi-gigabyte visual feature payload."""
    with np.load(path, allow_pickle=False) as archive:
        missing = set(FEATURE_INDEX_FIELDS) - set(archive.files)
        if missing:
            raise ValueError(f"feature cache is missing fields: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in FEATURE_INDEX_FIELDS}


def make_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 14),
    )


def shuffled_rows(task_ids: np.ndarray, episode_ids: np.ndarray, seed: int) -> np.ndarray:
    result = np.arange(len(task_ids))
    rng = np.random.default_rng(seed)
    for task, episode in sorted(set(zip(task_ids.tolist(), episode_ids.tolist()))):
        rows = np.flatnonzero((task_ids == task) & (episode_ids == episode))
        if len(rows) > 1:
            result[rows] = np.roll(rows, int(rng.integers(1, len(rows))))
    return result


@torch.inference_mode()
def evaluate(
    physics_head: nn.Module,
    action_head: nn.Module,
    physics_features: torch.Tensor,
    action_features: torch.Tensor,
    targets: torch.Tensor,
    task_ids: np.ndarray,
    episode_ids: np.ndarray,
    indices: list[int],
    batch_size: int,
    seed: int,
) -> dict:
    shuffle = shuffled_rows(task_ids, episode_ids, seed)
    variants = {"aligned": [], "episode_shuffled": [], "action_only": [], "zero": []}
    target_batches = []
    for start in range(0, len(indices), batch_size):
        rows = np.asarray(indices[start : start + batch_size])
        action = action_features[rows]
        variants["aligned"].append(
            physics_head(torch.cat((physics_features[rows], action), dim=-1))
        )
        variants["episode_shuffled"].append(
            physics_head(torch.cat((physics_features[shuffle[rows]], action), dim=-1))
        )
        variants["action_only"].append(action_head(action))
        variants["zero"].append(torch.zeros_like(targets[rows]))
        target_batches.append(targets[rows])
    predictions = {name: torch.cat(values) for name, values in variants.items()}
    evaluation_targets = torch.cat(target_batches)
    evaluation_tasks = task_ids[np.asarray(indices)]
    result = {}
    for name, prediction in predictions.items():
        task_metrics = {}
        for task in sorted(set(evaluation_tasks.tolist())):
            mask = torch.as_tensor(evaluation_tasks == task, device=prediction.device)
            terms = loss_terms(prediction[mask], evaluation_targets[mask])
            task_metrics[str(task)] = {
                **{key: float(value) for key, value in terms.items()},
                "joint_mae": float(
                    (prediction[mask, ..., :12] - evaluation_targets[mask, ..., :12])
                    .abs()
                    .mean()
                ),
                "num_samples": int(mask.sum()),
            }
        macro = {
            key: float(np.mean([metrics[key] for metrics in task_metrics.values()]))
            for key in ("loss", "joint_smooth_l1", "velocity_smooth_l1", "gripper_smooth_l1", "joint_mae")
        }
        result[name] = {"macro": macro, "tasks": task_metrics}
    aligned = result["aligned"]["macro"]
    result["aligned_relative_improvement"] = {
        baseline: {
            metric: (result[baseline]["macro"][metric] - aligned[metric])
            / max(result[baseline]["macro"][metric], 1e-12)
            for metric in ("loss", "joint_mae")
        }
        for baseline in ("episode_shuffled", "action_only", "zero")
    }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    physics = load(args.physics_cache.resolve())
    action_tokens = load(args.action_token_cache.resolve())
    action_targets = load(args.action_target_cache.resolve())
    features = load_feature_index(args.feature_cache.resolve())
    if not (
        exact_keys(physics) == exact_keys(action_targets) == exact_keys(features)
    ):
        raise ValueError("physics, target, and feature caches are not exactly key-aligned")
    joined_action_ids = join_action_ids(physics, action_tokens)
    targets = torch.as_tensor(
        np.asarray(action_targets["action_targets"][:, 1:], dtype=np.float32), device=device
    )

    physics_state = torch.load(args.physics_checkpoint.resolve(), map_location="cpu", weights_only=False)
    physics_model = build_physics_tokenizer(physics_state["model_config"])
    physics_model.load_state_dict(physics_state["model"], strict=True)
    physics_model.to(device).eval()
    action_state = torch.load(
        args.action_tokenizer_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    action_model = build_action_tokenizer(action_state["model_config"])
    action_model.load_state_dict(action_state["model"], strict=True)
    action_model.to(device).eval()
    with torch.inference_mode():
        physics_ids = torch.as_tensor(physics["physics_token_ids"], dtype=torch.long, device=device)
        physics_batches = []
        action_batches = []
        for start in range(0, len(physics_ids), args.batch_size):
            ids = physics_ids[start : start + args.batch_size]
            quantized = physics_model.quantizer.decode(ids).reshape(
                len(ids), physics_model.num_transitions, physics_model.queries_per_transition, -1
            )
            physics_batches.append(quantized.mean(dim=2))
            action_ids = torch.as_tensor(
                joined_action_ids[start : start + args.batch_size], dtype=torch.long, device=device
            )
            decoded = action_model.decode_tokens(action_ids, gripper_dims=(12, 13))
            action_batches.append(decoded[:, DECODER_INDICES])
        physics_features = torch.cat(physics_batches)
        action_features = torch.cat(action_batches)

    complete = np.asarray(features["future_valid_mask"], dtype=bool).all(axis=1)
    if "target_valid_mask" in action_targets:
        complete &= np.asarray(action_targets["target_valid_mask"][:, 1:], dtype=bool).all(
            axis=1
        )
    validation_episodes = {int(value) for value in args.validation_episodes.split(",") if value}
    excluded_episodes = {int(value) for value in args.excluded_episodes.split(",") if value}
    if validation_episodes & excluded_episodes:
        raise ValueError("validation and excluded episodes must be disjoint")
    validation = np.flatnonzero(
        complete & np.isin(physics["episode_ids"], list(validation_episodes))
    ).tolist()
    training = np.flatnonzero(
        complete
        & ~np.isin(
            physics["episode_ids"], list(validation_episodes | excluded_episodes)
        )
    ).tolist()
    train_by_task = {
        task: np.asarray([row for row in training if int(physics["task_ids"][row]) == task])
        for task in sorted(set(physics["task_ids"][training].tolist()))
    }
    physics_head = make_head(physics_features.shape[-1] + action_features.shape[-1], args.hidden_dim).to(device)
    action_head = make_head(action_features.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(physics_head.parameters()) + list(action_head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.02
    )
    rng = np.random.default_rng(args.seed)
    best_score = None
    best_action_score = None
    best_states = None
    best_action_state = None
    history = []
    for step in range(1, args.steps + 1):
        sampled_tasks = rng.choice(list(train_by_task), size=args.batch_size, replace=True)
        rows = np.asarray([rng.choice(train_by_task[int(task)]) for task in sampled_tasks])
        physics_prediction = physics_head(
            torch.cat((physics_features[rows], action_features[rows]), dim=-1)
        )
        action_prediction = action_head(action_features[rows])
        physics_loss = loss_terms(physics_prediction, targets[rows])["loss"]
        action_loss = loss_terms(action_prediction, targets[rows])["loss"]
        optimizer.zero_grad(set_to_none=True)
        (physics_loss + action_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(physics_head.parameters()) + list(action_head.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            physics_head.eval()
            action_head.eval()
            result = evaluate(
                physics_head,
                action_head,
                physics_features,
                action_features,
                targets,
                physics["task_ids"],
                physics["episode_ids"],
                validation,
                args.batch_size,
                args.seed,
            )
            score = result["aligned"]["macro"]["loss"]
            history.append({"step": step, "evaluation": result})
            if best_score is None or score < best_score:
                best_score = score
                best_states = {
                    "physics_head": {k: v.detach().cpu().clone() for k, v in physics_head.state_dict().items()},
                    "action_head": {k: v.detach().cpu().clone() for k, v in action_head.state_dict().items()},
                    "step": step,
                }
            action_score = result["action_only"]["macro"]["loss"]
            if best_action_score is None or action_score < best_action_score:
                best_action_score = action_score
                best_action_state = {
                    k: v.detach().cpu().clone() for k, v in action_head.state_dict().items()
                }
            print(json.dumps({
                "step": step,
                "aligned": result["aligned"]["macro"],
                "action_only": result["action_only"]["macro"],
                "improvement": result["aligned_relative_improvement"],
            }))
            physics_head.train()
            action_head.train()

    if best_states is None or best_action_state is None:
        raise RuntimeError("no decoder state selected")
    physics_head.load_state_dict(best_states["physics_head"])
    action_head.load_state_dict(best_action_state)
    physics_head.eval()
    action_head.eval()
    final = evaluate(
        physics_head,
        action_head,
        physics_features,
        action_features,
        targets,
        physics["task_ids"],
        physics["episode_ids"],
        validation,
        args.batch_size,
        args.seed,
    )
    report = {
        "schema_version": 1,
        "kind": "action_conditioned_oracle_physics_residual_probe",
        "best_step": best_states["step"],
        "steps": args.steps,
        "train_samples": len(training),
        "validation_samples": len(validation),
        "validation_episodes": sorted(validation_episodes),
        "excluded_episodes": sorted(excluded_episodes),
        "physics_pooling": "mean",
        "action_condition": "decoded_D17_action_tokens_at_decoder_indices",
        "final": final,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {**report, "physics_head": best_states["physics_head"], "action_head": best_states["action_head"]},
        args.output,
    )
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
