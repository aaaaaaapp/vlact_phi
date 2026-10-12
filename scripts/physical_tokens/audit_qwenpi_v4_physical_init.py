#!/usr/bin/env python3
"""Audit strict initialization of the physical-token Pi0.5 framework."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from accelerate import PartialState
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-initialized-extension", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PartialState()
    config = OmegaConf.load(args.config)
    config.framework.name = "QwenPIv4PhysicalTokens"
    config.framework.action_model.expert_model_path = str(
        REPO_ROOT / "playground/Pretrained_models/Qwen3-0.6B"
    )
    if not hasattr(config.framework, "discrete_conditioning"):
        config.framework.discrete_conditioning = {
            "enabled": True,
            "action_vocab_size": 128,
            "physics_vocab_size": 256,
            "max_action_tokens": 4,
            "max_physics_tokens": 256,
            "action_gate_init": 0.1,
            "physics_gate_init": 0.1,
            "action_condition_dropout": 0.0,
            "physics_condition_dropout": 0.2,
            "physics_condition_enabled": True,
            "physics_fusion_mode": "concat",
        }

    torch.manual_seed(int(config.get("seed", 42)))
    model = build_framework(config)
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", mmap=True, weights_only=True
    )
    incompatible = model.load_state_dict(checkpoint, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    allowed_missing_prefixes = ["pi_discrete_conditioner."]
    if hasattr(model, "pi_discrete_layer_gates"):
        allowed_missing_prefixes.append("pi_discrete_layer_gates.")
    if hasattr(model, "pi_action_layer_gates"):
        allowed_missing_prefixes.append("pi_action_layer_gates.")
    if hasattr(model, "pi_physics_layer_gates"):
        allowed_missing_prefixes.append("pi_physics_layer_gates.")
    forbidden_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected or forbidden_missing:
        raise RuntimeError(
            f"incompatible Pi checkpoint: unexpected={unexpected[:8]}, "
            f"forbidden_missing={forbidden_missing[:8]}"
        )

    action_init = None
    physics_init = None
    action_tokenizer = getattr(
        config.trainer, "action_tokenizer_init_checkpoint", None
    )
    if action_tokenizer:
        action_init = TrainerUtils.initialize_action_conditioner_from_tokenizer(
            model,
            action_tokenizer,
            config.trainer.action_tokenizer_init_sha256,
        )
    physics_tokenizer = getattr(
        config.trainer, "physics_tokenizer_init_checkpoint", None
    )
    if physics_tokenizer:
        physics_init = TrainerUtils.initialize_physics_adapter_from_tokenizer(
            model,
            physics_tokenizer,
            config.trainer.physics_tokenizer_init_sha256,
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.pi_discrete_conditioner.parameters():
        parameter.requires_grad_(True)
    if hasattr(model, "pi_discrete_layer_gates"):
        for parameter in model.pi_discrete_layer_gates.parameters():
            parameter.requires_grad_(True)
    if hasattr(model, "pi_action_layer_gates"):
        for parameter in model.pi_action_layer_gates.parameters():
            parameter.requires_grad_(True)
    if hasattr(model, "pi_physics_layer_gates"):
        for parameter in model.pi_physics_layer_gates.parameters():
            parameter.requires_grad_(True)

    initialized_extension = None
    if args.save_initialized_extension is not None:
        args.save_initialized_extension.parent.mkdir(parents=True, exist_ok=True)
        extension_state = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        torch.save(extension_state, args.save_initialized_extension)
        initialized_extension = {
            "checkpoint": str(args.save_initialized_extension.resolve()),
            "sha256": sha256_file(args.save_initialized_extension.resolve()),
            "tensor_count": len(extension_state),
        }

    fusion_mode = str(
        config.framework.discrete_conditioning.get(
            "physics_fusion_mode", "concat"
        )
    )
    grouped_modes = {
        "grouped_concat",
        "pi_multilayer_residual",
        "pi_separate_multilayer_residual",
    }
    discrete_memory_token_count = int(
        config.framework.discrete_conditioning.max_action_tokens
        + (
            config.framework.discrete_conditioning.max_physics_tokens
            // config.framework.discrete_conditioning.get("physics_group_size", 1)
            if fusion_mode in grouped_modes
            else config.framework.discrete_conditioning.max_physics_tokens
        )
    )

    report = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "checkpoint_tensor_count": len(checkpoint),
        "missing_new_conditioner_keys": missing,
        "unexpected_keys": unexpected,
        "action_tokenizer_initialization": action_init,
        "physics_tokenizer_initialization": physics_init,
        "physics_fusion_mode": fusion_mode,
        "physics_group_size": int(
            config.framework.discrete_conditioning.get("physics_group_size", 1)
        ),
        "discrete_input_token_count": int(
            config.framework.discrete_conditioning.max_action_tokens
            + config.framework.discrete_conditioning.max_physics_tokens
        ),
        "discrete_memory_token_count": discrete_memory_token_count,
        "appended_prefix_token_count": (
            0 if fusion_mode in {
                "pi_multilayer_residual", "pi_separate_multilayer_residual"
            }
            else discrete_memory_token_count
        ),
        "residual_layer_gate_count": (
            len(model.pi_discrete_layer_gates)
            if hasattr(model, "pi_discrete_layer_gates")
            else 0
        ),
        "action_residual_layer_gate_count": (
            len(model.pi_action_layer_gates)
            if hasattr(model, "pi_action_layer_gates")
            else 0
        ),
        "physics_residual_layer_gate_count": (
            len(model.pi_physics_layer_gates)
            if hasattr(model, "pi_physics_layer_gates")
            else 0
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_tensors": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
        "initialized_extension": initialized_extension,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
