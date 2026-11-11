#!/usr/bin/env python3
"""Train PhiZero-compatible tokenizers with an exact effective global batch."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from diffsynth.training.phizero import PhiZeroDiffusionTrainingModel, VideoPathDataset


class ParameterEMA:
    """FP32 EMA shadow for trainable tokenizer, conditioner, and LoRA weights."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("Cannot create parameter EMA without trainable parameters")

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, average in self.shadow.items():
            average.lerp_(parameters[name].detach().float(), 1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        parameters = dict(model.named_parameters())
        backup = {}
        try:
            with torch.no_grad():
                for name, average in self.shadow.items():
                    parameter = parameters[name]
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(average.to(dtype=parameter.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)


class ExactGlobalBatchSampler(Sampler[list[int]]):
    """Assign a fixed-size local batch to each rank for every global batch.

    Ranks with no real item receive a dummy for DDP synchronization. If a local
    batch has multiple items, the final partial global batch is filled by
    repeating items from the current shuffled epoch so every rank keeps a
    rectangular batch. This only affects the final, small remainder.
    """

    def __init__(
        self, size: int, global_batch_size: int, rank: int, world_size: int, seed: int
    ) -> None:
        self.size = int(size)
        self.global_batch_size = int(global_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.local_batch_size = math.ceil(self.global_batch_size / self.world_size)
        self.num_batches = math.ceil(self.size / self.global_batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        for batch_index in range(self.num_batches):
            offset = batch_index * self.global_batch_size
            local_indices = []
            for item_index in range(self.local_batch_size):
                global_index = offset + self.rank * self.local_batch_size + item_index
                in_batch = global_index < offset + self.global_batch_size
                if in_batch and global_index < self.size:
                    local_indices.append(indices[global_index])
                elif in_batch and self.local_batch_size > 1 and self.size:
                    local_indices.append(indices[global_index % self.size])
                else:
                    local_indices.append(-1)
            yield local_indices


class DummyAwareVideoDataset(Dataset):
    def __init__(self, videos: VideoPathDataset) -> None:
        self.videos = videos

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int):
        if index < 0:
            return torch.zeros(
                3, self.videos.num_frames, self.videos.height, self.videos.width
            ), False
        return self.videos[index], True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dit-path", nargs="+", required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--motion-height", type=int, default=256)
    parser.add_argument("--motion-width", type=int, default=448)
    parser.add_argument("--global-batch-size", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--queries-per-transition", type=int, default=None,
                        help="Adapt pretrained learned queries (32 to 8 gives 64 tokens for 33 frames).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lr-start", type=float, default=1e-6)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--motion-dropout", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsq-entropy-weight", type=float, default=0.05)
    parser.add_argument("--levels", default="8,5,5,5,5,5")
    parser.add_argument("--quantizer", choices=("fsq", "ema_vq"), default="fsq")
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--vq-ema-decay", type=float, default=0.99)
    parser.add_argument("--dead-code-threshold", type=float, default=1.0)
    parser.add_argument("--fresh-phi-zero-components", action="store_true")
    parser.add_argument("--model-ema-decay", type=float, default=None)
    args = parser.parse_args()
    args.quantizer_levels = tuple(int(value) for value in args.levels.split(","))
    if args.global_batch_size < 1 or args.epochs < 1 or args.gradient_accumulation_steps < 1:
        parser.error("global batch size, epochs, and gradient accumulation must be positive")
    if args.num_frames != 33:
        parser.error("The PhiZero Wan2.2 training path requires 33-frame clips")
    if args.quantizer == "ema_vq" and args.codebook_size < 2:
        parser.error("EMA-VQ codebook size must be at least 2")
    if args.model_ema_decay is not None and not 0.0 < args.model_ema_decay < 1.0:
        parser.error("Model EMA decay must be in (0, 1)")
    if args.global_batch_size % args.gradient_accumulation_steps:
        parser.error("global batch size must be divisible by gradient accumulation steps")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("--max-train-steps must be positive")
    if args.queries_per_transition is not None and args.queries_per_transition < 1:
        parser.error("--queries-per-transition must be positive")
    return args


def adapt_pretrained_queries(model: PhiZeroDiffusionTrainingModel, target_count: int) -> list[int]:
    """Downsample pretrained QFormer queries without changing the other loaded weights."""
    old = model.tokenizer.learned_queries
    source_count, hidden_dim = old.weight.shape
    if target_count > source_count:
        raise ValueError(f"Cannot expand {source_count} pretrained queries to {target_count}")
    # Select the center of each equal-sized source interval. For 32 -> 8 this
    # selects [2, 6, ..., 30], retaining the source ordering and parameters.
    indices = (list(range(source_count)) if target_count == source_count else
               [(2 * index + 1) * source_count // (2 * target_count)
                for index in range(target_count)])
    if len(set(indices)) != target_count:
        raise ValueError("Query selection produced duplicate indices")
    if target_count != source_count:
        replacement = torch.nn.Embedding(target_count, hidden_dim, device=old.weight.device,
                                         dtype=old.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(old.weight[indices])
        model.tokenizer.learned_queries = replacement
    tokenizer_config = replace(model.model_config.motion_tokenizer,
                               act_embedding_num=target_count)
    model.tokenizer.config = tokenizer_config
    token_count = tokenizer_config.physical_language_length(33)
    model.model_config = replace(model.model_config, motion_tokenizer=tokenizer_config,
                                 max_motion_tokens=token_count)
    model.conditioner.max_motion_tokens = token_count
    return indices


def build_lr_scheduler(optimizer, total_steps: int, args):
    warmup = max(int(args.warmup_steps), 0)
    start_ratio = args.lr_start / args.learning_rate
    min_ratio = args.min_learning_rate / args.learning_rate

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return start_ratio + (1.0 - start_ratio) * step / warmup
        if total_steps <= warmup:
            return 1.0
        progress = min(max((step - warmup) / (total_steps - warmup), 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(
    accelerator,
    model,
    output_dir: Path,
    step: int,
    model_ema: ParameterEMA | None = None,
    *,
    final: bool = False,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw_model = accelerator.unwrap_model(model)
        state = raw_model.export_state_dict()
        state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
        save_file(state, output_dir / f"step-{step}.safetensors")
        if model_ema is not None:
            with model_ema.average_parameters(raw_model):
                ema_state = raw_model.export_state_dict()
            ema_state = {
                key: value.detach().cpu().contiguous()
                for key, value in ema_state.items()
            }
            save_file(ema_state, output_dir / f"step-{step}-ema.safetensors")
            if final:
                save_file(ema_state, output_dir / "final-ema.safetensors")
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)
        ],
    )
    world_size = accelerator.num_processes
    if world_size < 1:
        raise RuntimeError(f"Expected at least one distributed process, found {world_size}")
    micro_global_batch_size = args.global_batch_size // args.gradient_accumulation_steps
    local_batch_size = math.ceil(micro_global_batch_size / world_size)
    if local_batch_size > 1 and micro_global_batch_size % world_size:
        raise ValueError(
            "When a rank handles multiple samples, global batch / accumulation "
            "must be divisible by the process count"
        )

    set_seed(args.seed, device_specific=True)
    output_dir = args.output_dir.resolve()
    if (output_dir / "training_args.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    raw_dataset = VideoPathDataset(args.csv, args.num_frames, args.height, args.width)
    if len(raw_dataset) == 0:
        raise ValueError(f"No video clips listed in {args.csv}")
    sampler = ExactGlobalBatchSampler(
        len(raw_dataset), micro_global_batch_size, accelerator.process_index,
        world_size, args.seed
    )
    dataset = DummyAwareVideoDataset(raw_dataset)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model = PhiZeroDiffusionTrainingModel.from_pretrained(
        args.init_checkpoint,
        dit_path=args.dit_path,
        vae_path=str(args.vae_path),
        device=accelerator.device,
        motion_height=args.motion_height,
        motion_width=args.motion_width,
        motion_dropout=args.motion_dropout,
        gradient_checkpointing=True,
        train_tokenizer=True,
        fsq_entropy_weight=args.fsq_entropy_weight,
        quantizer=args.quantizer,
        quantizer_levels=args.quantizer_levels,
        codebook_size=args.codebook_size if args.quantizer == "ema_vq" else None,
        ema_decay=args.vq_ema_decay if args.quantizer == "ema_vq" else None,
        dead_code_threshold=(
            args.dead_code_threshold if args.quantizer == "ema_vq" else None
        ),
        fresh_phi_zero_components=args.fresh_phi_zero_components,
    )
    query_indices = None
    if args.queries_per_transition is not None:
        query_indices = adapt_pretrained_queries(model, args.queries_per_transition)
    trainable_parameters = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    raw_model = accelerator.unwrap_model(model)
    model_ema = (
        ParameterEMA(raw_model, args.model_ema_decay)
        if args.model_ema_decay is not None
        else None
    )

    batches_per_epoch = sampler.num_batches
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    available_steps = optimizer_steps_per_epoch * args.epochs
    if args.max_train_steps is not None and available_steps < args.max_train_steps:
        raise ValueError(f"--epochs permits only {available_steps} optimizer steps; "
                         f"need {args.max_train_steps}")
    total_steps = args.max_train_steps or available_steps
    scheduler = build_lr_scheduler(optimizer, total_steps, args)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_config = accelerator.unwrap_model(model).model_config
        (output_dir / "model_config.json").write_text(
            json.dumps(model_config.to_dict(), indent=2) + "\n"
        )
        run_args = {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}
        run_args.update({
            "dataset_size": len(raw_dataset),
            "world_size": world_size,
            "micro_global_batch_size": micro_global_batch_size,
            "batches_per_epoch": batches_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "effective_global_batch_size": args.global_batch_size,
            "pretrained_query_indices": query_indices,
        })
        (output_dir / "training_args.json").write_text(json.dumps(run_args, indent=2) + "\n")
    accelerator.wait_for_everyone()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=total_steps,
        disable=not accelerator.is_main_process,
        desc=f"PhiZero {args.quantizer} training",
    )
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch_index, (video, valid) in enumerate(dataloader):
            accumulation_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            if local_batch_size > 1:
                accumulation_valid_count = micro_global_batch_size * min(
                    args.gradient_accumulation_steps,
                    batches_per_epoch - accumulation_start,
                )
            else:
                accumulation_valid_count = min(
                    args.global_batch_size,
                    len(raw_dataset) - accumulation_start * micro_global_batch_size,
                )
            sync_now = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == batches_per_epoch
            )
            sync_context = nullcontext() if sync_now else accelerator.no_sync(model)
            valid_sample_count = int(valid.sum().item())
            with sync_context:
                with accelerator.autocast():
                    raw_loss = model(
                        video.to(accelerator.device, non_blocking=True),
                        sample_valid_mask=valid.to(
                            accelerator.device, non_blocking=True
                        ),
                    )
                    if valid_sample_count:
                        scaled_loss = raw_loss * (
                            world_size * valid_sample_count / accumulation_valid_count
                        )
                    else:
                        scaled_loss = raw_loss * 0.0
                accelerator.backward(scaled_loss)

            local_metrics = torch.stack([
                raw_loss.detach().float() * float(valid_sample_count),
                torch.tensor(float(valid_sample_count), device=accelerator.device),
            ]).unsqueeze(0)
            gathered = accelerator.gather(local_metrics)
            logged_loss = (gathered[:, 0].sum() / gathered[:, 1].sum().clamp_min(1)).item()
            if sync_now:
                accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(accelerator.unwrap_model(model))
                scheduler.step()
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{logged_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(
                        accelerator, model, output_dir, global_step, model_ema
                    )
                if global_step >= total_steps:
                    break

        save_checkpoint(
            accelerator,
            model,
            output_dir,
            global_step,
            model_ema,
            final=(global_step >= total_steps),
        )
        if global_step >= total_steps:
            break

    progress.close()
    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
