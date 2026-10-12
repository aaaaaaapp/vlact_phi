#!/usr/bin/env python3
"""One-sample forward/backward and constrained-generation smoke for the token reasoner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    PartialState()
    data_config = config.datasets.vla_data
    dataset = make_LeRobotSingleDataset(
        Path(data_config.data_root_dir),
        "click_alarmclock",
        "robotwin_wrap32",
        data_cfg=OmegaConf.to_container(data_config, resolve=True),
    )
    sample = dataset[0]
    target = np.asarray(sample["action_token_ids"], dtype=np.int64)
    model = build_framework(config)
    model = TrainerUtils.load_pretrained_backbones(
        model,
        checkpoint_path=str(config.trainer.pretrained_checkpoint),
        reload_modules=str(config.trainer.reload_modules),
    )
    device = torch.device(args.device)
    model = model.to(device=device, dtype=torch.bfloat16).train()
    output = model([sample])
    loss = output["action_loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite token loss: {loss}")
    loss.backward()
    finite_gradients = [
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not finite_gradients or not all(finite_gradients):
        raise RuntimeError("reasoner did not receive finite gradients")

    model.eval()
    prediction = model.predict_token_ids([sample])[0]
    if prediction.shape != target.shape:
        raise RuntimeError(f"prediction shape {prediction.shape} != target shape {target.shape}")
    print(
        {
            "loss": float(loss.detach().cpu()),
            "target_token_ids": target.tolist(),
            "predicted_token_ids": prediction.tolist(),
            "token_accuracy": float(np.mean(prediction == target)),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
            "finite_gradient_tensors": len(finite_gradients),
            "device": str(device),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
