#!/usr/bin/env python3
"""Train a compact EMA-VQ/FSQ tokenizer on future visual-feature deltas."""

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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.tokenizers import PhysicalLanguageTokenizer


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


class FutureFeatureCacheDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as data:
            required = {"feature_deltas", "future_valid_mask", "episode_ids", "base_indices"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            self.features = np.asarray(data["feature_deltas"])
            self.valid_mask = np.asarray(data["future_valid_mask"])
            self.episode_ids = np.asarray(data["episode_ids"])
            self.base_indices = np.asarray(data["base_indices"])
            self.offsets = np.asarray(data["offsets"]) if "offsets" in data else None
        if self.features.ndim != 3:
            raise ValueError(f"feature_deltas must be [N,T,D], got {self.features.shape}")
        if self.valid_mask.shape != self.features.shape[:2]:
            raise ValueError("future_valid_mask shape must match feature_deltas[:2]")
        if len(self.episode_ids) != len(self.features) or len(self.base_indices) != len(self.features):
            raise ValueError("cache key arrays must match feature_deltas length")
        if not np.all(self.features[:, 0] == 0):
            raise ValueError("feature delta at offset 0 must be exactly zero")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(np.array(self.features[index], dtype=np.float32)),
            "valid_mask": torch.from_numpy(np.array(self.valid_mask[index], dtype=np.bool_)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantizer", choices=("ema_vq", "fsq"), default="ema_vq")
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--fsq-levels", default="4,4,8")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--dead-code-steps", type=int, default=200)
    parser.add_argument("--cosine-weight", type=float, default=0.25)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument("--current-frame-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-episodes", default="5,20,25,39,49")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device, local_rank, world_size


def split_indices(
    dataset: FutureFeatureCacheDataset, validation_episodes: tuple[int, ...]
) -> tuple[list[int], list[int]]:
    validation_set = set(validation_episodes)
    validation = [i for i, episode in enumerate(dataset.episode_ids) if int(episode) in validation_set]
    training = [i for i, episode in enumerate(dataset.episode_ids) if int(episode) not in validation_set]
    present = set(int(value) for value in np.unique(dataset.episode_ids))
    missing = validation_set.difference(present)
    if missing:
        raise ValueError(f"validation episodes are absent from cache: {sorted(missing)}")
    if not training or not validation:
        raise ValueError("training and validation splits must both be non-empty")
    return training, validation


def feature_statistics(
    dataset: FutureFeatureCacheDataset, indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = dataset.features[indices].astype(np.float32)
    valid = dataset.valid_mask[indices]
    flattened = features[valid]
    mean = torch.from_numpy(flattened.mean(axis=0))
    std = torch.from_numpy(flattened.std(axis=0)).clamp_min(1e-5)
    time_mean = []
    for time_index in range(features.shape[1]):
        selected = features[:, time_index][valid[:, time_index]]
        time_mean.append(selected.mean(axis=0) if len(selected) else np.zeros(features.shape[-1]))
    return mean, std, torch.from_numpy(np.stack(time_mean).astype(np.float32))


def _reduce_sums(values: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    if world_size > 1:
        for value in values.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    count = values.pop("count").clamp_min(1)
    return {key: float((value / count).cpu()) for key, value in values.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_kwargs: dict,
    time_mean: torch.Tensor,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    totals = {
        key: torch.zeros((), device=device)
        for key in (
            "loss",
            "reconstruction_loss",
            "cosine_loss",
            "temporal_loss",
            "rmse",
            "no_motion_rmse",
            "train_mean_rmse",
            "perplexity",
            "active_ratio",
            "shuffle_token_change",
            "count",
        )
    }
    time_mean = time_mean.to(device)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = raw_model.loss(**batch, **loss_kwargs)
        valid = batch["valid_mask"].unsqueeze(-1).expand_as(batch["features"])
        valid_count = valid.sum().clamp_min(1)
        squared = (output["reconstruction"] - batch["features"]).square()
        no_motion_squared = batch["features"].square()
        mean_squared = (time_mean.unsqueeze(0) - batch["features"]).square()
        permutation = torch.arange(len(batch["features"]) - 1, -1, -1, device=device)
        shuffled = batch["features"].clone()
        shuffled[:, 1:] = shuffled[permutation, 1:]
        shuffled_ids, _, _ = raw_model.encode(shuffled, batch["valid_mask"])
        sample_count = torch.tensor(float(len(batch["features"])), device=device)
        for key in ("loss", "reconstruction_loss", "cosine_loss", "temporal_loss", "perplexity", "active_ratio"):
            totals[key] += output[key].detach() * sample_count
        totals["rmse"] += torch.sqrt(squared[valid].sum() / valid_count) * sample_count
        totals["no_motion_rmse"] += torch.sqrt(no_motion_squared[valid].sum() / valid_count) * sample_count
        totals["train_mean_rmse"] += torch.sqrt(mean_squared[valid].sum() / valid_count) * sample_count
        totals["shuffle_token_change"] += (output["token_ids"] != shuffled_ids).float().mean() * sample_count
        totals["count"] += sample_count
    return _reduce_sums(totals, world_size)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dataset: FutureFeatureCacheDataset,
    step: int,
    metrics: dict[str, float],
) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": {
            "feature_dim": dataset.features.shape[-1],
            "horizon": dataset.features.shape[1],
            "num_tokens": args.num_tokens,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "quantizer": args.quantizer,
            "codebook_size": args.codebook_size,
            "fsq_levels": parse_ints(args.fsq_levels),
            "ema_decay": args.ema_decay,
            "commitment_weight": args.commitment_weight,
            "dead_code_steps": args.dead_code_steps,
        },
        "input": {
            "path": str(dataset.path),
            "size": dataset.path.stat().st_size,
            "mtime_ns": dataset.path.stat().st_mtime_ns,
            "num_samples": len(dataset),
        },
        "metrics": metrics,
    }
    target = args.output_dir / f"step-{step:08d}.pt"
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def main() -> None:
    args = parse_args()
    device, rank, world_size = setup_distributed(args)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    dataset = FutureFeatureCacheDataset(args.input)
    validation_episodes = parse_ints(args.validation_episodes)
    training_indices, validation_indices = split_indices(dataset, validation_episodes)
    mean, std, time_mean = feature_statistics(dataset, training_indices)
    train_set = Subset(dataset, training_indices)
    validation_set = Subset(dataset, validation_indices)
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if world_size > 1 else None
    validation_sampler = DistributedSampler(validation_set, shuffle=False) if world_size > 1 else None
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
        sampler=validation_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if not len(train_loader):
        raise ValueError("training split is smaller than one per-rank batch")

    model = PhysicalLanguageTokenizer(
        feature_dim=dataset.features.shape[-1],
        horizon=dataset.features.shape[1],
        num_tokens=args.num_tokens,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        quantizer=args.quantizer,
        codebook_size=args.codebook_size,
        fsq_levels=parse_ints(args.fsq_levels),
        ema_decay=args.ema_decay,
        commitment_weight=args.commitment_weight,
        dead_code_steps=args.dead_code_steps,
    ).to(device)
    model.set_normalization(mean.to(device), std.to(device))
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    loss_kwargs = {
        "cosine_weight": args.cosine_weight,
        "temporal_weight": args.temporal_weight,
        "current_frame_weight": args.current_frame_weight,
    }

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        manifest.update(
            feature_shape=list(dataset.features.shape),
            offsets=None if dataset.offsets is None else dataset.offsets.tolist(),
            train_samples=len(training_indices),
            validation_samples=len(validation_indices),
            validation_episodes=list(validation_episodes),
            world_size=world_size,
        )
        (args.output_dir / "training_config.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
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
        output = model(
            features=batch["features"],
            valid_mask=batch["valid_mask"],
            loss_kwargs=loss_kwargs,
        )
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            metrics = evaluate(model, validation_loader, device, loss_kwargs, time_mean, world_size)
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
