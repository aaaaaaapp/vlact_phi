#!/usr/bin/env python3
"""Evaluate a QwenPI oracle-token checkpoint on fixed held-out episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import PartialState
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("e0", "e1", "e2", "b2", "b3", "b4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--statistics", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-timesteps", type=int, default=None)
    parser.add_argument("--token-cache", type=Path, default=None)
    parser.add_argument("--action-vocab-size", type=int, default=None)
    parser.add_argument("--max-action-tokens", type=int, default=None)
    parser.add_argument("--physics-vocab-size", type=int, default=None)
    parser.add_argument("--max-physics-tokens", type=int, default=None)
    parser.add_argument("--physics-group-size", type=int, default=None)
    parser.add_argument(
        "--physics-fusion-mode",
        choices=(
            "concat",
            "residual_cross_attention",
            "dit_multilayer_cross_attention",
        ),
        default=None,
    )
    parser.add_argument("--physics-fusion-dim", type=int, default=None)
    parser.add_argument("--physics-fusion-heads", type=int, default=None)
    parser.add_argument("--dit-injection-layers", default=None)
    parser.add_argument("--dit-attention-dropout", type=float, default=None)
    parser.add_argument(
        "--dit-physics-adapter-type",
        choices=("cross_attention", "temporal_film"),
        default=None,
    )
    parser.add_argument(
        "--dit-physics-memory-pooling",
        choices=("none", "transition_mean"),
        default=None,
    )
    parser.add_argument("--physics-tokens-per-transition", type=int, default=None)
    parser.add_argument(
        "--dit-action-tokens-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--physics-condition-dropout", type=float, default=None)
    parser.add_argument(
        "--joint-condition-source",
        choices=(
            "cached",
            "predicted",
            "soft",
            "predicted_action_cached_physics",
            "cached_action_predicted_physics",
        ),
        default=None,
    )
    parser.add_argument("--joint-soft-temperature", type=float, default=None)
    parser.add_argument(
        "--physics-condition-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--action-init-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained-checkpoint-sha256", default=None)
    parser.add_argument("--task-name", default=None)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_run_file(checkpoint: Path, filename: str) -> Path:
    candidates = (checkpoint.parent / filename, checkpoint.parent.parent / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer {filename} beside checkpoint {checkpoint}; pass it explicitly"
    )


def load_validation_episodes(config_path: Path, manifest_path: Path | None) -> tuple[list[int], Path | None]:
    if manifest_path is not None:
        resolved = manifest_path.resolve()
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
        return [int(value) for value in manifest["validation_episode_ids"]], resolved

    config = OmegaConf.load(config_path)
    episodes = config.datasets.vla_data.get("exclude_episode_ids", None)
    if not episodes:
        raise ValueError(
            "Config has no exclude_episode_ids; pass --split-manifest with validation_episode_ids"
        )
    return [int(value) for value in episodes], None


def build_validation_dataset(
    config, validation_episode_ids: list[int], task_name: str | None = None
):
    data_cfg = config.datasets.vla_data
    data_cfg.exclude_episode_ids = None
    data_cfg.include_episode_ids = validation_episode_ids
    mixture = DATASET_NAMED_MIXTURES[str(data_cfg.data_mix)]
    if task_name is not None:
        matches = [entry for entry in mixture if entry[0] == task_name]
        if len(matches) != 1:
            raise ValueError(
                f"task {task_name!r} did not resolve uniquely in mixture {data_cfg.data_mix}"
            )
        mixture = matches
    if len(mixture) != 1:
        raise ValueError(
            "Fixed oracle evaluator currently requires a single-dataset mixture, "
            f"got {len(mixture)} entries"
        )
    data_name, _, robot_type = mixture[0]
    dataset = make_LeRobotSingleDataset(
        Path(str(data_cfg.data_root_dir)),
        data_name,
        robot_type,
        delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
        data_cfg=data_cfg,
    )
    return dataset


def select_panel(dataset, num_samples: int, seed: int) -> list[dict[str, int]]:
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_samples > len(dataset):
        raise ValueError(f"num_samples={num_samples} exceeds validation size {len(dataset)}")
    rng = np.random.default_rng(seed)
    episode_to_indices: dict[int, list[int]] = {}
    for index, (episode_id, _) in enumerate(dataset.all_steps):
        episode_to_indices.setdefault(int(episode_id), []).append(index)
    episodes = sorted(episode_to_indices)
    quotas = {episode: num_samples // len(episodes) for episode in episodes}
    for episode in rng.permutation(episodes)[: num_samples % len(episodes)]:
        quotas[int(episode)] += 1
    selected = []
    for episode in episodes:
        candidates = episode_to_indices[episode]
        quota = quotas[episode]
        if quota > len(candidates):
            raise ValueError(
                f"Episode {episode} only has {len(candidates)} samples, requested {quota}"
            )
        selected.extend(rng.choice(candidates, size=quota, replace=False).tolist())
    indices = np.sort(np.asarray(selected, dtype=np.int64))
    return [
        {
            "dataset_index": int(index),
            "episode_id": int(dataset.all_steps[int(index)][0]),
            "base_index": int(dataset.all_steps[int(index)][1]),
        }
        for index in indices
    ]


def wrapped_difference(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.remainder(prediction - target + math.pi, 2.0 * math.pi) - math.pi


class MetricAccumulator:
    STATE_FIELDS = (
        "action_abs_sum",
        "action_squared_sum",
        "action_count",
        "joint_abs_sum",
        "joint_count",
        "velocity_abs_sum",
        "velocity_count",
        "endpoint_joint_abs_sum",
        "endpoint_joint_count",
        "gripper_abs_sum",
        "gripper_correct",
        "gripper_count",
    )

    def __init__(self) -> None:
        self.action_abs_sum = 0.0
        self.action_squared_sum = 0.0
        self.action_count = 0
        self.joint_abs_sum = 0.0
        self.joint_count = 0
        self.velocity_abs_sum = 0.0
        self.velocity_count = 0
        self.endpoint_joint_abs_sum = 0.0
        self.endpoint_joint_count = 0
        self.gripper_abs_sum = 0.0
        self.gripper_correct = 0
        self.gripper_count = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        time_mask: np.ndarray | None = None,
        dim_mask: np.ndarray | None = None,
    ) -> None:
        horizon = min(len(prediction), len(target))
        action_dim = min(prediction.shape[-1], target.shape[-1])
        prediction = np.asarray(prediction[:horizon, :action_dim], dtype=np.float64)
        target = np.asarray(target[:horizon, :action_dim], dtype=np.float64)

        valid_time = np.ones(horizon, dtype=bool)
        if time_mask is not None:
            valid_time &= np.asarray(time_mask[:horizon]).astype(bool)
        valid_dim = np.ones(action_dim, dtype=bool)
        if dim_mask is not None:
            candidate = np.asarray(dim_mask).astype(bool)
            candidate = candidate[-1] if candidate.ndim > 1 else candidate
            valid_dim &= candidate[:action_dim]
        valid = valid_time[:, None] & valid_dim[None, :]

        difference = prediction - target
        self.action_abs_sum += float(np.abs(difference)[valid].sum())
        self.action_squared_sum += float(np.square(difference)[valid].sum())
        self.action_count += int(valid.sum())

        # RoboTwin abs_qpos transforms expose joint radians and binary grippers
        # directly; despite its legacy name, predict_action's output is in this
        # same raw action space and must not be percentile-un-normalized.
        physical_prediction = prediction
        physical_target = target
        joint_dims = np.arange(min(12, action_dim))
        gripper_dims = np.arange(12, min(14, action_dim))

        joint_valid = valid[:, joint_dims]
        joint_error = wrapped_difference(
            physical_prediction[:, joint_dims], physical_target[:, joint_dims]
        )
        self.joint_abs_sum += float(np.abs(joint_error)[joint_valid].sum())
        self.joint_count += int(joint_valid.sum())

        if horizon > 1:
            velocity_valid = (
                valid_time[1:, None]
                & valid_time[:-1, None]
                & valid_dim[joint_dims][None, :]
            )
            predicted_velocity = wrapped_difference(
                physical_prediction[1:, joint_dims], physical_prediction[:-1, joint_dims]
            )
            target_velocity = wrapped_difference(
                physical_target[1:, joint_dims], physical_target[:-1, joint_dims]
            )
            velocity_error = predicted_velocity - target_velocity
            self.velocity_abs_sum += float(np.abs(velocity_error)[velocity_valid].sum())
            self.velocity_count += int(velocity_valid.sum())

        valid_indices = np.flatnonzero(valid_time)
        if len(valid_indices) and len(joint_dims):
            endpoint = int(valid_indices[-1])
            endpoint_valid = valid_dim[joint_dims]
            endpoint_error = joint_error[endpoint]
            self.endpoint_joint_abs_sum += float(np.abs(endpoint_error)[endpoint_valid].sum())
            self.endpoint_joint_count += int(endpoint_valid.sum())

        if len(gripper_dims):
            gripper_valid = valid[:, gripper_dims]
            predicted_gripper = physical_prediction[:, gripper_dims]
            target_gripper = physical_target[:, gripper_dims]
            self.gripper_abs_sum += float(
                np.abs(predicted_gripper - target_gripper)[gripper_valid].sum()
            )
            self.gripper_correct += int(
                ((predicted_gripper >= 0.5) == (target_gripper >= 0.5))[gripper_valid].sum()
            )
            self.gripper_count += int(gripper_valid.sum())

    def compute(self) -> dict[str, float | int]:
        def divide(numerator: float, denominator: int) -> float:
            return float(numerator / denominator) if denominator else float("nan")

        joint_mae = divide(self.joint_abs_sum, self.joint_count)
        return {
            "action_mae": divide(self.action_abs_sum, self.action_count),
            "action_rmse": math.sqrt(
                divide(self.action_squared_sum, self.action_count)
            ),
            "joint_wrapped_mae_rad": joint_mae,
            "joint_wrapped_mae_deg": math.degrees(joint_mae),
            "joint_velocity_mae_rad": divide(self.velocity_abs_sum, self.velocity_count),
            "endpoint_joint_mae_rad": divide(
                self.endpoint_joint_abs_sum, self.endpoint_joint_count
            ),
            "endpoint_joint_mae_deg": math.degrees(
                divide(self.endpoint_joint_abs_sum, self.endpoint_joint_count)
            ),
            "gripper_mae": divide(self.gripper_abs_sum, self.gripper_count),
            "gripper_accuracy": divide(self.gripper_correct, self.gripper_count),
            "valid_action_values": self.action_count,
        }

    def state_dict(self) -> dict[str, float | int]:
        return {field: getattr(self, field) for field in self.STATE_FIELDS}


def materialize_sample(dataset, panel_item: dict[str, int], seed: int) -> dict[str, Any]:
    transforms = getattr(dataset, "transforms", None)
    was_training = getattr(transforms, "training", None)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        if was_training is not None:
            transforms.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return dataset[panel_item["dataset_index"]]
    finally:
        if was_training is not None:
            transforms.train() if was_training else transforms.eval()
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def load_model(config, checkpoint: Path, mode: str, device: torch.device):
    PartialState()
    torch.manual_seed(int(config.get("seed", 42)))
    model = build_framework(config)
    is_joint_extension = str(config.framework.name) == "QwenPIv4JointTokens"
    is_pi_v4_extension = str(config.framework.name) in {
        "QwenPIv4PhysicalTokens",
        "QwenPIv4JointTokens",
    }
    if is_pi_v4_extension:
        extension_prefixes = ["pi_discrete_conditioner"]
        if (
            str(config.framework.discrete_conditioning.physics_fusion_mode)
            == "pi_multilayer_residual"
        ):
            extension_prefixes.append("pi_discrete_layer_gates")
        if (
            str(config.framework.discrete_conditioning.physics_fusion_mode)
            == "pi_separate_multilayer_residual"
        ):
            extension_prefixes.extend(
                ["pi_action_layer_gates", "pi_physics_layer_gates"]
            )
        if is_joint_extension:
            extension_prefixes.extend(
                [
                    "action_token_probe",
                    "action_condition_embedding",
                    "physics_token_probe",
                ]
            )
        model = TrainerUtils.load_extension_checkpoint(
            model,
            str(config.trainer.pretrained_checkpoint),
            extension_prefixes,
        )
    else:
        model = TrainerUtils.load_pretrained_backbones(
            model,
            str(config.trainer.pretrained_checkpoint),
            reload_modules="qwen_vl_interface",
        )
    action_init_checkpoint = getattr(
        config.trainer, "action_init_checkpoint", None
    )
    if action_init_checkpoint:
        model = TrainerUtils.load_action_condition_checkpoint(
            model, str(action_init_checkpoint)
        )
    head_state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    incompatible = model.load_state_dict(head_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if is_pi_v4_extension:
        required_prefixes = ["pi_discrete_conditioner."]
        if hasattr(model, "pi_discrete_layer_gates"):
            required_prefixes.append("pi_discrete_layer_gates.")
        if hasattr(model, "pi_action_layer_gates"):
            required_prefixes.append("pi_action_layer_gates.")
        if hasattr(model, "pi_physics_layer_gates"):
            required_prefixes.append("pi_physics_layer_gates.")
        if is_joint_extension:
            required_prefixes.extend(
                [
                    "action_token_probe.",
                    "action_condition_embedding.",
                    "physics_token_probe.",
                ]
            )
        required_adapter_keys = {
            key for key in model.state_dict()
            if any(key.startswith(prefix) for prefix in required_prefixes)
        }
        missing_required = sorted(required_adapter_keys.difference(head_state))
        unexpected.extend(sorted(set(head_state).difference(required_adapter_keys)))
    elif action_init_checkpoint:
        required_adapter_keys = {
            key
            for key in model.state_dict()
            if key.startswith("action_model.physics_adapter.")
        }
        missing_required = sorted(required_adapter_keys.difference(head_state))
        unexpected.extend(sorted(set(head_state).difference(required_adapter_keys)))
    else:
        missing_required = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("qwen_vl_interface.")
        ]
    if unexpected or missing_required:
        raise RuntimeError(
            f"Head checkpoint mismatch: unexpected={unexpected}, "
            f"missing_required={missing_required}"
        )
    model.set_discrete_conditioning_enabled(mode != "e0")
    if hasattr(model, "set_physics_condition_enabled"):
        model.set_physics_condition_enabled(
            bool(config.framework.discrete_conditioning.physics_condition_enabled)
        )
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.pretrained_checkpoint_sha256 is not None:
        digest = args.pretrained_checkpoint_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("--pretrained-checkpoint-sha256 must be 64 hexadecimal characters")
        args.pretrained_checkpoint_sha256 = digest
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    validation_episodes, manifest_path = load_validation_episodes(
        config_path, args.split_manifest
    )
    config = OmegaConf.load(config_path)
    config.framework.discrete_conditioning.enabled = args.mode != "e0"
    if args.token_cache is not None:
        config.datasets.vla_data.discrete_token_cache = str(args.token_cache.resolve())
    if args.action_vocab_size is not None:
        config.framework.discrete_conditioning.action_vocab_size = args.action_vocab_size
    if args.max_action_tokens is not None:
        config.framework.discrete_conditioning.max_action_tokens = args.max_action_tokens
    if args.physics_vocab_size is not None:
        config.framework.discrete_conditioning.physics_vocab_size = args.physics_vocab_size
    if args.max_physics_tokens is not None:
        config.framework.discrete_conditioning.max_physics_tokens = args.max_physics_tokens
    if args.physics_group_size is not None:
        if args.physics_group_size < 1:
            raise ValueError("physics group size must be positive")
        config.framework.discrete_conditioning.physics_group_size = args.physics_group_size
    if args.physics_fusion_mode is not None:
        config.framework.discrete_conditioning.physics_fusion_mode = args.physics_fusion_mode
    if args.physics_fusion_dim is not None:
        config.framework.discrete_conditioning.physics_fusion_dim = args.physics_fusion_dim
    if args.physics_fusion_heads is not None:
        config.framework.discrete_conditioning.physics_fusion_heads = args.physics_fusion_heads
    if args.dit_injection_layers is not None:
        config.framework.discrete_conditioning.dit_injection_layers = (
            args.dit_injection_layers
        )
    if args.dit_attention_dropout is not None:
        config.framework.discrete_conditioning.dit_attention_dropout = (
            args.dit_attention_dropout
        )
    if args.dit_action_tokens_only is not None:
        config.framework.discrete_conditioning.dit_action_tokens_only = (
            args.dit_action_tokens_only
        )
    if args.dit_physics_adapter_type is not None:
        config.framework.discrete_conditioning.dit_physics_adapter_type = (
            args.dit_physics_adapter_type
        )
    if args.dit_physics_memory_pooling is not None:
        config.framework.discrete_conditioning.dit_physics_memory_pooling = (
            args.dit_physics_memory_pooling
        )
    if args.physics_tokens_per_transition is not None:
        config.framework.discrete_conditioning.physics_tokens_per_transition = (
            args.physics_tokens_per_transition
        )
    if args.physics_condition_dropout is not None:
        if not 0.0 <= args.physics_condition_dropout < 1.0:
            raise ValueError("physics condition dropout must be in [0, 1)")
        config.framework.discrete_conditioning.physics_condition_dropout = (
            args.physics_condition_dropout
        )
    if args.physics_condition_enabled is not None:
        config.framework.discrete_conditioning.physics_condition_enabled = (
            args.physics_condition_enabled
        )
    if args.joint_condition_source is not None:
        if str(config.framework.name) != "QwenPIv4JointTokens":
            raise ValueError("--joint-condition-source requires QwenPIv4JointTokens")
        config.framework.joint_token_prediction.condition_source = (
            args.joint_condition_source
        )
    if args.joint_soft_temperature is not None:
        if str(config.framework.name) != "QwenPIv4JointTokens":
            raise ValueError("--joint-soft-temperature requires QwenPIv4JointTokens")
        if args.joint_soft_temperature <= 0.0:
            raise ValueError("--joint-soft-temperature must be positive")
        config.framework.joint_token_prediction.soft_temperature = (
            args.joint_soft_temperature
        )
    if args.num_inference_timesteps is not None:
        config.framework.action_model.num_inference_timesteps = args.num_inference_timesteps
    if args.pretrained_checkpoint is not None:
        config.trainer.pretrained_checkpoint = str(args.pretrained_checkpoint.resolve())
    if args.action_init_checkpoint is not None:
        config.trainer.action_init_checkpoint = str(
            args.action_init_checkpoint.resolve()
        )

    dataset = build_validation_dataset(config, validation_episodes, args.task_name)
    panel = select_panel(dataset, args.num_samples, args.seed)
    statistics_path = (
        args.statistics.resolve()
        if args.statistics is not None
        else infer_run_file(checkpoint_path, "dataset_statistics.json")
    )
    metrics = MetricAccumulator()
    episode_metrics: dict[int, MetricAccumulator] = {}
    episode_sample_counts: dict[int, int] = {}

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    model = load_model(config, checkpoint_path, args.mode, device)

    latencies = []
    for ordinal, panel_item in enumerate(panel):
        sample_seed = args.seed + ordinal
        sample = materialize_sample(dataset, panel_item, sample_seed)
        torch.manual_seed(sample_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sample_seed)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = model.predict_action([sample])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - started)
        metrics.update(
            output["normalized_actions"][0],
            sample["action"],
            sample.get("action_valid_mask"),
            sample.get("action_dim_valid_mask"),
        )
        episode_id = int(panel_item["episode_id"])
        episode_accumulator = episode_metrics.setdefault(
            episode_id, MetricAccumulator()
        )
        episode_accumulator.update(
            output["normalized_actions"][0],
            sample["action"],
            sample.get("action_valid_mask"),
            sample.get("action_dim_valid_mask"),
        )
        episode_sample_counts[episode_id] = episode_sample_counts.get(episode_id, 0) + 1

    checkpoint_stat = checkpoint_path.stat()
    base_checkpoint = Path(str(config.trainer.pretrained_checkpoint)).resolve()
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "task_name": args.task_name,
        "conditioning_enabled": args.mode != "e0",
        "discrete_conditioning": OmegaConf.to_container(
            config.framework.discrete_conditioning,
            resolve=True,
        ),
        "joint_token_prediction": (
            OmegaConf.to_container(
                config.framework.joint_token_prediction,
                resolve=True,
            )
            if str(config.framework.name) == "QwenPIv4JointTokens"
            else None
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "token_cache": str(
            Path(str(config.datasets.vla_data.discrete_token_cache)).resolve()
        ),
        "token_cache_sha256": sha256_file(
            Path(str(config.datasets.vla_data.discrete_token_cache)).resolve()
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_stat.st_size,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": (
            args.pretrained_checkpoint_sha256 or sha256_file(base_checkpoint)
        ),
        "base_checkpoint_size_bytes": base_checkpoint.stat().st_size,
        "action_init_checkpoint": (
            str(Path(str(config.trainer.action_init_checkpoint)).resolve())
            if getattr(config.trainer, "action_init_checkpoint", None)
            else None
        ),
        "action_init_checkpoint_sha256": (
            sha256_file(Path(str(config.trainer.action_init_checkpoint)).resolve())
            if getattr(config.trainer, "action_init_checkpoint", None)
            else None
        ),
        "action_init_checkpoint_size_bytes": (
            Path(str(config.trainer.action_init_checkpoint)).resolve().stat().st_size
            if getattr(config.trainer, "action_init_checkpoint", None)
            else None
        ),
        "statistics": str(statistics_path),
        "statistics_sha256": sha256_file(statistics_path),
        "split_manifest": str(manifest_path) if manifest_path else None,
        "validation_episode_ids": validation_episodes,
        "validation_dataset_size": len(dataset),
        "num_samples": len(panel),
        "panel_seed": args.seed,
        "sample_ids": panel,
        "num_inference_timesteps": int(
            config.framework.action_model.num_inference_timesteps
        ),
        "metrics_space": "raw_dataset_action_joint_radians_and_binary_grippers",
        "metrics": metrics.compute(),
        "episode_metrics": {
            str(episode_id): {
                "num_samples": episode_sample_counts[episode_id],
                "metrics": accumulator.compute(),
                "sufficient_statistics": accumulator.state_dict(),
            }
            for episode_id, accumulator in sorted(episode_metrics.items())
        },
        "latency_seconds": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "total": float(np.sum(latencies)),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
