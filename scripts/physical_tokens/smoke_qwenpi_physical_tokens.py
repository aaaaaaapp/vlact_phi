#!/usr/bin/env python3
"""One-batch engineering smoke test for QwenPIPhysicalTokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from accelerate import PartialState
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-conditioning", action="store_true")
    parser.add_argument("--token-cache", type=Path, default=None)
    parser.add_argument("--action-vocab-size", type=int, default=None)
    parser.add_argument("--max-action-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.disable_conditioning:
        config.framework.discrete_conditioning.enabled = False
    if args.token_cache is not None:
        config.datasets.vla_data.discrete_token_cache = str(args.token_cache.resolve())
    if args.action_vocab_size is not None:
        config.framework.discrete_conditioning.action_vocab_size = args.action_vocab_size
    if args.max_action_tokens is not None:
        config.framework.discrete_conditioning.max_action_tokens = args.max_action_tokens
    PartialState()
    data_config = config.datasets.vla_data
    dataset = make_LeRobotSingleDataset(
        Path(data_config.data_root_dir),
        "click_alarmclock",
        "robotwin_wrap32",
        data_cfg=OmegaConf.to_container(data_config, resolve=True),
    )
    sample = dataset[0]
    if "action_token_ids" not in sample:
        raise RuntimeError("token-conditioned dataset did not attach action_token_ids")
    model = build_framework(config)
    checkpoint = config.trainer.get("pretrained_checkpoint", None)
    reload_modules = config.trainer.get("reload_modules", None)
    if checkpoint:
        model = TrainerUtils.load_pretrained_backbones(
            model,
            checkpoint_path=str(checkpoint),
            reload_modules=str(reload_modules) if reload_modules else None,
        )
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).train()
    output = model([sample])
    loss = output["action_loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite action loss: {loss}")
    loss.backward()
    condition_gradients = [
        parameter.grad
        for parameter in model.discrete_conditioner.parameters()
        if parameter.requires_grad
    ]
    has_condition_gradient = any(gradient is not None for gradient in condition_gradients)
    has_finite_condition_gradient = any(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in condition_gradients
    )
    if model.discrete_conditioning_enabled and not has_finite_condition_gradient:
        raise RuntimeError("enabled discrete conditioner received no finite gradient")
    if not model.discrete_conditioning_enabled and has_condition_gradient:
        raise RuntimeError("disabled discrete conditioner unexpectedly received a gradient")
    print(
        {
            "action_loss": float(loss.detach().cpu()),
            "action_shape": tuple(sample["action"].shape),
            "action_token_shape": tuple(sample["action_token_ids"].shape),
            "conditioned_prefix_extra_tokens": (
                len(sample["action_token_ids"])
                if model.discrete_conditioning_enabled
                else 0
            ),
            "conditioning_enabled": bool(model.discrete_conditioning_enabled),
            "conditioner_has_gradient": has_condition_gradient,
            "device": str(device),
            "dtype": str(dtype),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
