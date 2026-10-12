#!/usr/bin/env python3
"""Train the PhiZero-aligned tokenizer on deduplicated spatial feature clips."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.tokenizers import SpatiotemporalPhysicalLanguageTokenizer


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


class SpatiotemporalFeatureDataset(Dataset):
    def __init__(self, path: Path, action_target_cache: Path | None = None) -> None:
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as data:
            required = {
                "frame_features",
                "source_frame_indices",
                "future_valid_mask",
                "episode_ids",
                "base_indices",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            self.frame_features = np.asarray(data["frame_features"])
            self.source_indices = np.asarray(data["source_frame_indices"], dtype=np.int64)
            self.valid_mask = np.asarray(data["future_valid_mask"], dtype=np.bool_)
            self.episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
            self.base_indices = np.asarray(data["base_indices"], dtype=np.int64)
            self.offsets = np.asarray(data["offsets"], dtype=np.int64)
            self.task_ids = (
                np.asarray(data["task_ids"], dtype=np.int64)
                if "task_ids" in data.files
                else np.zeros_like(self.episode_ids)
            )
            self.task_names = (
                tuple(str(value) for value in np.asarray(data["task_names"]).tolist())
                if "task_names" in data.files
                else ("single_task",)
            )
        self.action_targets = None
        if action_target_cache is not None:
            with np.load(action_target_cache.resolve(), allow_pickle=False) as data:
                for key, expected in (
                    ("task_ids", self.task_ids),
                    ("episode_ids", self.episode_ids),
                    ("base_indices", self.base_indices),
                    ("offsets", self.offsets),
                ):
                    if key not in data.files or not np.array_equal(data[key], expected):
                        raise ValueError(f"action target cache is not key-aligned: {key}")
                if "action_targets" not in data.files:
                    raise ValueError("action target cache is missing action_targets")
                self.action_targets = np.asarray(data["action_targets"], dtype=np.float32)
                expected = (len(self.episode_ids), len(self.offsets), 14)
                if self.action_targets.shape != expected:
                    raise ValueError(
                        f"action_targets must have shape {expected}, got "
                        f"{self.action_targets.shape}"
                    )
        if self.frame_features.ndim != 3:
            raise ValueError("frame_features must be [num_unique_frames, spatial_tokens, dim]")
        if self.source_indices.shape != self.valid_mask.shape:
            raise ValueError("source_frame_indices and future_valid_mask shapes differ")
        if self.source_indices.min() < 0 or self.source_indices.max() >= len(self.frame_features):
            raise ValueError("source frame index outside frame_features")
        if len(self.episode_ids) != len(self.source_indices):
            raise ValueError("sample key arrays do not match source_frame_indices")
        if self.task_ids.shape != self.episode_ids.shape:
            raise ValueError("task_ids and episode_ids shapes differ")
        if self.task_ids.min() < 0 or self.task_ids.max() >= len(self.task_names):
            raise ValueError("task ID is outside task_names")

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        features = np.array(self.frame_features[self.source_indices[index]], dtype=np.float32)
        result = {
            "features": torch.from_numpy(features),
            "valid_mask": torch.from_numpy(np.array(self.valid_mask[index], dtype=np.bool_)),
        }
        if self.action_targets is not None:
            result["action_targets"] = torch.from_numpy(
                np.array(self.action_targets[index], dtype=np.float32)
            )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-target-cache", type=Path, default=None)
    parser.add_argument("--action-aux-weight", type=float, default=0.0)
    parser.add_argument("--action-velocity-weight", type=float, default=0.25)
    parser.add_argument("--action-gripper-weight", type=float, default=0.1)
    parser.add_argument(
        "--action-aux-target-mode",
        choices=("absolute", "action_token_residual"),
        default="absolute",
    )
    parser.add_argument("--quantizer", choices=("ema_vq", "fsq"), default="ema_vq")
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--fsq-levels", default="4,4,8")
    parser.add_argument("--queries-per-transition", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--dead-code-steps", type=int, default=200)
    parser.add_argument(
        "--token-routing",
        choices=("global_qformer", "spatial_aligned"),
        default="global_qformer",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--normalization-samples", type=int, default=512)
    parser.add_argument("--validation-episodes", default="5,20,25,39,49")
    parser.add_argument(
        "--validation-episodes-per-task",
        type=int,
        default=0,
        help="Use the last N episode IDs of every task for validation.",
    )
    parser.add_argument(
        "--max-train-clips-per-task",
        type=int,
        default=0,
        help="Deterministically cap complete training clips per task; 0 keeps all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_distributed() -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device, rank, world_size


def split_complete_indices(
    dataset: SpatiotemporalFeatureDataset,
    validation_episodes: tuple[int, ...],
    validation_episodes_per_task: int = 0,
) -> tuple[list[int], list[int]]:
    if validation_episodes_per_task < 0:
        raise ValueError("validation_episodes_per_task must be non-negative")
    if validation_episodes_per_task:
        validation_keys = set()
        for task_id in sorted(set(int(value) for value in dataset.task_ids)):
            episodes = sorted(
                set(
                    int(episode)
                    for episode, sample_task in zip(dataset.episode_ids, dataset.task_ids)
                    if int(sample_task) == task_id
                )
            )
            if len(episodes) <= validation_episodes_per_task:
                raise ValueError(
                    f"task {task_id} has {len(episodes)} episodes, not enough for "
                    f"{validation_episodes_per_task} validation episodes plus training"
                )
            validation_keys.update(
                (task_id, episode) for episode in episodes[-validation_episodes_per_task:]
            )
    else:
        validation_set = set(validation_episodes)
        validation_keys = {
            (int(task_id), int(episode))
            for task_id, episode in zip(dataset.task_ids, dataset.episode_ids)
            if int(episode) in validation_set
        }
    complete = dataset.valid_mask.all(axis=1)
    training = [
        index
        for index, (task_id, episode) in enumerate(zip(dataset.task_ids, dataset.episode_ids))
        if complete[index] and (int(task_id), int(episode)) not in validation_keys
    ]
    validation = [
        index
        for index, (task_id, episode) in enumerate(zip(dataset.task_ids, dataset.episode_ids))
        if complete[index] and (int(task_id), int(episode)) in validation_keys
    ]
    if not training or not validation:
        raise ValueError("complete-clip training and validation splits must both be non-empty")
    return training, validation


def cap_indices_per_task(
    dataset: SpatiotemporalFeatureDataset,
    indices: list[int],
    maximum: int,
    seed: int,
) -> list[int]:
    if maximum < 0:
        raise ValueError("max_train_clips_per_task must be non-negative")
    if maximum == 0:
        return indices
    rng = np.random.default_rng(seed)
    selected = []
    for task_id in sorted(set(int(dataset.task_ids[index]) for index in indices)):
        task_indices = np.asarray(
            [index for index in indices if int(dataset.task_ids[index]) == task_id],
            dtype=np.int64,
        )
        if len(task_indices) > maximum:
            task_indices = rng.choice(task_indices, size=maximum, replace=False)
        selected.extend(int(index) for index in task_indices)
    return sorted(selected)


def shard_indices_without_padding(
    indices: list[int], rank: int, world_size: int
) -> list[int]:
    """Partition evaluation indices exactly; unlike DistributedSampler, add no duplicates."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size define an invalid distributed shard")
    return indices[rank::world_size]


def feature_statistics(
    dataset: SpatiotemporalFeatureDataset,
    indices: list[int],
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        np.asarray(indices, dtype=np.int64),
        size=min(len(indices), max_samples),
        replace=False,
    ).tolist()
    feature_dim = dataset.frame_features.shape[-1]
    total = np.zeros(feature_dim, dtype=np.float64)
    squared = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    for index in selected:
        features = dataset.frame_features[dataset.source_indices[index]].astype(np.float32)
        deltas = features[1:] - features[:-1]
        flat = deltas.reshape(-1, feature_dim)
        total += flat.sum(axis=0, dtype=np.float64)
        squared += np.square(flat, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(flat)
    mean = total / max(count, 1)
    variance = np.maximum(squared / max(count, 1) - np.square(mean), 1e-10)
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(variance).astype(np.float32))


def reduce_totals(totals: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    if world_size > 1:
        for value in totals.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    count = totals.pop("count").clamp_min(1)
    return {key: float((value / count).cpu()) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, world_size: int) -> dict[str, float]:
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    totals = {
        key: torch.zeros((), device=device)
        for key in (
            "loss",
            "reconstruction_loss",
            "cosine_loss",
            "quantized_mse",
            "continuous_mse",
            "no_motion_mse",
            "temporal_reverse_token_change",
            "action_aux_loss",
            "action_joint_loss",
            "action_velocity_loss",
            "action_gripper_loss",
            "count",
        )
    }
    token_counts = torch.zeros(raw_model.codebook_size, device=device)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = raw_model(**batch)
        continuous_reconstruction = raw_model.reconstruct_continuous(
            batch["features"], batch["valid_mask"]
        )
        target = batch["features"][:, 1:] - batch["features"][:, :-1]
        sample_count = torch.tensor(float(len(target)), device=device)
        shuffled = batch["features"].clone()
        shuffled[:, 1:] = torch.flip(shuffled[:, 1:], dims=(1,))
        shuffled_ids, _, _ = raw_model.encode(shuffled, batch["valid_mask"])
        for key in totals:
            if key in output:
                totals[key] += output[key].detach() * sample_count
        totals["quantized_mse"] += (
            (output["reconstruction"] - target).square().mean() * sample_count
        )
        totals["continuous_mse"] += (
            (continuous_reconstruction - target).square().mean() * sample_count
        )
        totals["no_motion_mse"] += target.square().mean() * sample_count
        totals["temporal_reverse_token_change"] += (
            (output["token_ids"] != shuffled_ids).float().mean() * sample_count
        )
        token_counts += torch.bincount(
            output["token_ids"].reshape(-1), minlength=raw_model.codebook_size
        )
        totals["count"] += sample_count
    metrics = reduce_totals(totals, world_size)
    if world_size > 1:
        dist.all_reduce(token_counts, op=dist.ReduceOp.SUM)
    probabilities = token_counts / token_counts.sum().clamp_min(1)
    metrics["perplexity"] = float(
        torch.exp(-(probabilities * (probabilities + 1e-10).log()).sum()).cpu()
    )
    metrics["active_ratio"] = float((token_counts > 0).float().mean().cpu())
    metrics["rmse"] = metrics.pop("quantized_mse") ** 0.5
    metrics["continuous_rmse"] = metrics.pop("continuous_mse") ** 0.5
    metrics["no_motion_rmse"] = metrics.pop("no_motion_mse") ** 0.5
    baseline = max(metrics["no_motion_rmse"], 1e-12)
    metrics["continuous_relative_rmse_reduction"] = (
        baseline - metrics["continuous_rmse"]
    ) / baseline
    metrics["quantized_relative_rmse_reduction"] = (
        baseline - metrics["rmse"]
    ) / baseline
    metrics["quantization_rmse_gap"] = metrics["rmse"] - metrics["continuous_rmse"]
    return metrics


def save_checkpoint(model, optimizer, args, dataset, step: int, metrics: dict[str, float]) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    config = {
        "feature_dim": dataset.frame_features.shape[-1],
        "num_frames": dataset.source_indices.shape[1],
        "spatial_tokens": dataset.frame_features.shape[1],
        "queries_per_transition": args.queries_per_transition,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "num_heads": args.num_heads,
        "quantizer": args.quantizer,
        "codebook_size": args.codebook_size,
        "fsq_levels": parse_ints(args.fsq_levels),
        "ema_decay": args.ema_decay,
        "commitment_weight": args.commitment_weight,
        "dead_code_steps": args.dead_code_steps,
        "action_aux_dim": 14 if args.action_target_cache is not None else 0,
        "action_aux_weight": args.action_aux_weight,
        "action_velocity_weight": args.action_velocity_weight,
        "action_gripper_weight": args.action_gripper_weight,
        "action_aux_target_mode": args.action_aux_target_mode,
        "action_aux_pooling": "mean",
    }
    # Preserve exact compatibility with checkpoints created before routing was configurable.
    if args.token_routing != "global_qformer":
        config["token_routing"] = args.token_routing
    payload = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": config,
        "input": {"path": str(dataset.path), "num_samples": len(dataset)},
        "metrics": metrics,
    }
    target = args.output_dir / f"step-{step:08d}.pt"
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def main() -> None:
    args = parse_args()
    device, rank, world_size = setup_distributed()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    if args.action_aux_weight > 0 and args.action_target_cache is None:
        raise ValueError("positive action_aux_weight requires --action-target-cache")
    dataset = SpatiotemporalFeatureDataset(args.input, args.action_target_cache)
    validation_episodes = parse_ints(args.validation_episodes)
    all_train_indices, validation_indices = split_complete_indices(
        dataset,
        validation_episodes,
        args.validation_episodes_per_task,
    )
    train_indices = cap_indices_per_task(
        dataset,
        all_train_indices,
        args.max_train_clips_per_task,
        args.seed,
    )
    mean, std = feature_statistics(
        dataset, train_indices, args.normalization_samples, args.seed
    )
    train_set = Subset(dataset, train_indices)
    validation_set = Subset(
        dataset, shard_indices_without_padding(validation_indices, rank, world_size)
    )
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if world_size > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = SpatiotemporalPhysicalLanguageTokenizer(
        feature_dim=dataset.frame_features.shape[-1],
        num_frames=dataset.source_indices.shape[1],
        spatial_tokens=dataset.frame_features.shape[1],
        queries_per_transition=args.queries_per_transition,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        quantizer=args.quantizer,
        codebook_size=args.codebook_size,
        fsq_levels=parse_ints(args.fsq_levels),
        ema_decay=args.ema_decay,
        commitment_weight=args.commitment_weight,
        dead_code_steps=args.dead_code_steps,
        token_routing=args.token_routing,
        action_aux_dim=14 if args.action_target_cache is not None else 0,
        action_aux_weight=args.action_aux_weight,
        action_velocity_weight=args.action_velocity_weight,
        action_gripper_weight=args.action_gripper_weight,
        action_aux_target_mode=args.action_aux_target_mode,
    ).to(device)
    model.set_normalization(mean.to(device), std.to(device))
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(
            evaluation_metric_schema_version=2,
            train_complete_clips=len(train_indices),
            train_complete_clips_before_task_cap=len(all_train_indices),
            validation_complete_clips=len(validation_indices),
            num_tasks=len(dataset.task_names),
            task_names=list(dataset.task_names),
            emitted_tokens=(dataset.source_indices.shape[1] - 1) * args.queries_per_transition,
            world_size=world_size,
        )
        (args.output_dir / "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
    if world_size > 1:
        dist.barrier()

    step = 0
    epoch = 0
    iterator = iter(train_loader)
    started = time.time()
    metrics: dict[str, float] = {}
    while step < args.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        model.train()
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = model(**batch)
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            metrics = evaluate(model, validation_loader, device, world_size)
            metrics.update(step=step, elapsed_seconds=time.time() - started)
            if rank == 0:
                with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(json.dumps(metrics, sort_keys=True), flush=True)
        if rank == 0 and (step % args.save_interval == 0 or step == args.max_steps):
            save_checkpoint(model, optimizer, args, dataset, step, metrics)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
