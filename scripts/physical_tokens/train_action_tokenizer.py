#!/usr/bin/env python3
"""Train an EMA-VQ or FSQ tokenizer from a versioned action-chunk NPZ cache."""

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

from starVLA.model.modules.tokenizers import ActionChunkTokenizer


class ActionCacheDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as data:
            if "actions" not in data:
                raise ValueError(f"{path} must contain an 'actions' array")
            self.actions = np.asarray(data["actions"])
            self.valid_mask = np.asarray(data["action_valid_mask"]) if "action_valid_mask" in data else None
            self.dim_mask = np.asarray(data["action_dim_mask"]) if "action_dim_mask" in data else None
            self.episode_ids = np.asarray(data["episode_ids"]) if "episode_ids" in data else None
            self.task_ids = np.asarray(data["task_ids"]) if "task_ids" in data else None
        if self.actions.ndim != 3:
            raise ValueError(f"actions must be [N, T, D], got {self.actions.shape}")
        if self.valid_mask is not None and self.valid_mask.shape != self.actions.shape[:2]:
            raise ValueError("action_valid_mask shape must match actions[:2]")
        if self.dim_mask is not None and self.dim_mask.shape not in {
            (self.actions.shape[-1],),
            (len(self.actions), self.actions.shape[-1]),
        }:
            raise ValueError("action_dim_mask must be [D] or [N, D]")

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {"actions": torch.from_numpy(np.array(self.actions[index], dtype=np.float32))}
        if self.valid_mask is not None:
            sample["action_valid_mask"] = torch.from_numpy(
                np.array(self.valid_mask[index], dtype=np.bool_)
            )
        if self.dim_mask is not None:
            value = self.dim_mask if self.dim_mask.ndim == 1 else self.dim_mask[index]
            sample["action_dim_mask"] = torch.from_numpy(np.array(value, dtype=np.bool_))
        return sample


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantizer", choices=("ema_vq", "fsq"), default="ema_vq")
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--fsq-levels", default="8,8,8")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--dead-code-steps", type=int, default=200)
    parser.add_argument("--gripper-dims", default="12,13")
    parser.add_argument("--angular-dims", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--gripper-weight", type=float, default=0.25)
    parser.add_argument("--wrap-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional fixed episode split with training_episode_ids/validation_episode_ids.",
    )
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
    dataset: ActionCacheDataset,
    validation_fraction: float,
    seed: int,
    split_manifest: Path | None = None,
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    if dataset.episode_ids is not None:
        episode_ids = np.asarray(dataset.episode_ids)
        if len(episode_ids) != len(dataset):
            raise ValueError("episode_ids length must match actions")
        if split_manifest is not None:
            split = json.loads(split_manifest.read_text(encoding="utf-8"))
            training_episodes = set(map(int, split["training_episode_ids"]))
            validation_episodes = set(map(int, split["validation_episode_ids"]))
            if training_episodes & validation_episodes:
                raise ValueError("training and validation episode sets overlap")
            unknown = set(np.unique(episode_ids).tolist()) - training_episodes - validation_episodes
            if unknown:
                raise ValueError(f"split manifest omits episode IDs: {sorted(unknown)}")
        else:
            unique = np.unique(episode_ids)
            rng.shuffle(unique)
            validation_count = max(1, round(len(unique) * validation_fraction))
            validation_episodes = set(unique[:validation_count].tolist())
            training_episodes = set(unique.tolist()) - validation_episodes
        validation = [i for i, value in enumerate(episode_ids) if value.item() in validation_episodes]
        training = [i for i, value in enumerate(episode_ids) if value.item() in training_episodes]
    else:
        indices = np.arange(len(dataset))
        rng.shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation = indices[:validation_count].tolist()
        training = indices[validation_count:].tolist()
    if not training or not validation:
        raise ValueError("both training and validation splits must be non-empty")
    return training, validation


def reduce_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    reduced = {}
    for key, value in metrics.items():
        tensor = value.detach().float()
        if world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size
        reduced[key] = float(tensor.cpu())
    return reduced


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_kwargs: dict,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    batches = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = model.module.loss(**batch, **loss_kwargs) if hasattr(model, "module") else model.loss(**batch, **loss_kwargs)
        for key in (
            "loss",
            "reconstruction_loss",
            "velocity_loss",
            "gripper_loss",
            "wrap_loss",
            "perplexity",
            "active_ratio",
        ):
            totals[key] = totals.get(key, output[key].new_zeros(())) + output[key]
        batches += 1
    if batches == 0:
        raise RuntimeError("validation loader produced no batches")
    return reduce_metrics({key: value / batches for key, value in totals.items()}, world_size)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dataset: ActionCacheDataset,
    step: int,
    metrics: dict[str, float],
) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": {
            "action_dim": dataset.actions.shape[-1],
            "horizon": dataset.actions.shape[1],
            "num_tokens": args.num_tokens,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
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

    dataset = ActionCacheDataset(args.input)
    training_indices, validation_indices = split_indices(
        dataset, args.validation_fraction, args.seed, args.split_manifest
    )
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
        raise ValueError("training split is smaller than one batch; reduce --batch-size")

    model = ActionChunkTokenizer(
        action_dim=dataset.actions.shape[-1],
        horizon=dataset.actions.shape[1],
        num_tokens=args.num_tokens,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        quantizer=args.quantizer,
        codebook_size=args.codebook_size,
        fsq_levels=parse_ints(args.fsq_levels),
        ema_decay=args.ema_decay,
        commitment_weight=args.commitment_weight,
        dead_code_steps=args.dead_code_steps,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    loss_kwargs = {
        "angular_dim_mask": torch.tensor(
            [i in parse_ints(args.angular_dims) for i in range(dataset.actions.shape[-1])],
            device=device,
        ),
        "gripper_dims": parse_ints(args.gripper_dims),
        "velocity_weight": args.velocity_weight,
        "gripper_weight": args.gripper_weight,
        "wrap_weight": args.wrap_weight,
    }

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = vars(args).copy()
        manifest.update(
            action_shape=list(dataset.actions.shape),
            train_samples=len(training_indices),
            validation_samples=len(validation_indices),
            world_size=world_size,
        )
        manifest = {key: str(value) if isinstance(value, Path) else value for key, value in manifest.items()}
        (args.output_dir / "training_config.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()

    step = 0
    epoch = 0
    iterator = iter(train_loader)
    started = time.time()
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
        output = model.module.loss(**batch, **loss_kwargs) if hasattr(model, "module") else model.loss(**batch, **loss_kwargs)
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            metrics = evaluate(model, validation_loader, device, loss_kwargs, world_size)
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
