#!/usr/bin/env python3
"""Retrain only the residual decoder on frozen oracle physics-token latents."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_spatiotemporal_physics_token_ids import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--action-target-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-episodes", default="7,8")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decoder-pooling", choices=("mean", "flatten"), default="mean")
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def exact_keys(arrays: dict[str, np.ndarray]) -> list[tuple[int, int, int]]:
    return [
        tuple(map(int, values))
        for values in zip(arrays["task_ids"], arrays["episode_ids"], arrays["base_indices"])
    ]


def loss_terms(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    joint = F.smooth_l1_loss(prediction[..., :12], target[..., :12])
    velocity = F.smooth_l1_loss(
        prediction[:, 1:, :12] - prediction[:, :-1, :12],
        target[:, 1:, :12] - target[:, :-1, :12],
    )
    gripper = F.smooth_l1_loss(prediction[..., 12:14], target[..., 12:14])
    return {
        "loss": joint + 0.25 * velocity + 0.1 * gripper,
        "joint_smooth_l1": joint,
        "velocity_smooth_l1": velocity,
        "gripper_smooth_l1": gripper,
    }


@torch.inference_mode()
def evaluate(head, latents, targets, task_ids, indices, batch_size) -> dict:
    task_metrics = {}
    for task in sorted(set(int(task_ids[index]) for index in indices)):
        rows = np.asarray([index for index in indices if int(task_ids[index]) == task])
        totals = {name: 0.0 for name in ("loss", "joint_smooth_l1", "velocity_smooth_l1", "gripper_smooth_l1")}
        joint_absolute = 0.0
        count = 0
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            prediction = head(latents[batch_rows])
            target = targets[batch_rows]
            terms = loss_terms(prediction, target)
            size = len(batch_rows)
            for name, value in terms.items():
                totals[name] += float(value) * size
            joint_absolute += float((prediction[..., :12] - target[..., :12]).abs().mean()) * size
            count += size
        task_metrics[str(task)] = {
            **{name: value / count for name, value in totals.items()},
            "joint_mae": joint_absolute / count,
            "num_samples": count,
        }
    names = next(iter(task_metrics.values())).keys()
    macro = {
        name: float(np.mean([metrics[name] for metrics in task_metrics.values()]))
        for name in names
        if name != "num_samples"
    }
    return {"macro": macro, "tasks": task_metrics}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    physics = load(args.physics_cache.resolve())
    features = load(args.feature_cache.resolve())
    actions = load(args.action_target_cache.resolve())
    if exact_keys(physics) != exact_keys(features) or exact_keys(physics) != exact_keys(actions):
        raise ValueError("physics, feature, and action target caches are not exactly key-aligned")
    if features["offsets"].tolist() != list(range(0, 33, 2)):
        raise ValueError("unexpected feature offsets")
    targets_np = np.asarray(actions["action_targets"][:, 1:], dtype=np.float32)
    complete = np.asarray(features["future_valid_mask"], dtype=bool).all(axis=1)
    validation_episodes = {
        int(value) for value in args.validation_episodes.split(",") if value
    }
    validation = np.flatnonzero(
        complete & np.isin(physics["episode_ids"], list(validation_episodes))
    ).tolist()
    training = np.flatnonzero(
        complete & ~np.isin(physics["episode_ids"], list(validation_episodes))
    ).tolist()
    if not training or not validation:
        raise ValueError("empty training or validation split")

    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.action_aux_head is None or model.action_aux_target_mode != "action_token_residual":
        raise ValueError("checkpoint is not a residual auxiliary tokenizer")
    if args.decoder_pooling != model.action_aux_pooling:
        if args.decoder_pooling != "flatten":
            raise ValueError("only mean-to-flatten decoder replacement is supported")
        input_dim = checkpoint["model_config"]["latent_dim"] * model.queries_per_transition
        model.action_aux_pooling = "flatten"
        model.action_aux_head = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, checkpoint["model_config"]["hidden_dim"]),
            torch.nn.GELU(),
            torch.nn.Linear(checkpoint["model_config"]["hidden_dim"], model.action_aux_dim),
        )
    model.to(device).eval()
    with torch.inference_mode():
        ids = torch.as_tensor(physics["physics_token_ids"], dtype=torch.long, device=device)
        latent_batches = []
        for start in range(0, len(ids), args.batch_size):
            batch = ids[start : start + args.batch_size]
            quantized = model.quantizer.decode(batch).reshape(
                len(batch), model.num_transitions, model.queries_per_transition, -1
            )
            latent_batches.append(quantized)
        latents = model.action_aux_features(torch.cat(latent_batches))
    targets = torch.as_tensor(targets_np, device=device)
    head = model.action_aux_head.train()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.02
    )
    rng = np.random.default_rng(args.seed)
    train_by_task = {
        task: np.asarray([index for index in training if int(physics["task_ids"][index]) == task])
        for task in sorted(set(int(physics["task_ids"][index]) for index in training))
    }
    history = []
    best = None
    best_state = None
    for step in range(1, args.steps + 1):
        sampled_tasks = rng.choice(list(train_by_task), size=args.batch_size, replace=True)
        rows = np.asarray(
            [rng.choice(train_by_task[int(task)]) for task in sampled_tasks], dtype=np.int64
        )
        prediction = head(latents[rows])
        terms = loss_terms(prediction, targets[rows])
        optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            head.eval()
            result = evaluate(
                head, latents, targets, physics["task_ids"], validation, args.batch_size
            )
            entry = {
                "step": step,
                "train_loss": float(terms["loss"].detach()),
                "learning_rate": scheduler.get_last_lr()[0],
                **result,
            }
            history.append(entry)
            score = result["macro"]["loss"]
            if best is None or score < best:
                best = score
                best_state = {
                    name: value.detach().cpu().clone() for name, value in head.state_dict().items()
                }
            print(json.dumps({"step": step, "train_loss": entry["train_loss"], "validation_macro": result["macro"]}))
            head.train()
    if best_state is None:
        raise RuntimeError("no decoder checkpoint selected")
    head.load_state_dict(best_state)
    model.cpu()
    output_state = dict(checkpoint)
    output_state["model"] = model.state_dict()
    output_state["model_config"] = dict(checkpoint["model_config"])
    output_state["model_config"]["action_aux_pooling"] = args.decoder_pooling
    output_state["residual_decoder_retraining"] = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_episodes": sorted(validation_episodes),
        "balanced_task_sampling": True,
        "decoder_pooling": args.decoder_pooling,
        "best_validation_macro_loss": best,
        "train_complete_clips": len(training),
        "validation_complete_clips": len(validation),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    torch.save(output_state, temporary)
    os.replace(temporary, args.output)
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(output_state["residual_decoder_retraining"], indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
