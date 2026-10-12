#!/usr/bin/env python3
"""Evaluate action preservation and predicted-physics utility for D23."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_action_token_reasoner import (
    distribution_stats,
    load_reasoner,
    load_tokenizer,
    token_accuracy_metrics,
)
from evaluate_qwenpi_oracle import MetricAccumulator, materialize_sample, select_panel, sha256_file
from export_spatiotemporal_physics_token_ids import build_model as build_physics_tokenizer
from train_action_conditioned_physics_residual_probe import make_head
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


DECODER_INDICES = np.asarray(list(range(2, 32, 2)) + [31], dtype=np.int64)
PHYSICS_BLEND_SCALES = (0.125, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--physics-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--conditioned-decoder-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-episodes", default="7,8")
    parser.add_argument("--samples-per-task", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--soft-physics-temperatures",
        default="",
        help="Comma-separated positive temperatures for soft codebook expectation",
    )
    parser.add_argument(
        "--hard-confidence-thresholds",
        default="",
        help="Comma-separated mean-max-probability thresholds for gated hard physics IDs",
    )
    return parser.parse_args()


def decode_actions(ids: np.ndarray, tokenizer, device: torch.device) -> np.ndarray:
    return (
        tokenizer.decode_tokens(
            torch.as_tensor(ids, dtype=torch.long, device=device), gripper_dims=(12, 13)
        )
        .float()
        .cpu()
        .numpy()
    )


def decode_physics_residuals(
    ids: np.ndarray, tokenizer, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    if tokenizer.action_aux_head is None:
        raise ValueError("physics tokenizer has no action auxiliary decoder")
    outputs = []
    for start in range(0, len(ids), batch_size):
        batch = torch.as_tensor(ids[start : start + batch_size], dtype=torch.long, device=device)
        quantized = tokenizer.quantizer.decode(batch).reshape(
            len(batch), tokenizer.num_transitions, tokenizer.queries_per_transition, -1
        )
        outputs.append(tokenizer.predict_action_aux(quantized).float().cpu().numpy())
    return np.concatenate(outputs)


def decode_conditioned_residuals(
    ids: np.ndarray,
    base_actions: np.ndarray,
    tokenizer,
    head,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(ids), batch_size):
        batch = torch.as_tensor(ids[start : start + batch_size], dtype=torch.long, device=device)
        quantized = tokenizer.quantizer.decode(batch).reshape(
            len(batch), tokenizer.num_transitions, tokenizer.queries_per_transition, -1
        )
        physics_features = quantized.mean(dim=2)
        action_features = torch.as_tensor(
            base_actions[start : start + batch_size, DECODER_INDICES],
            dtype=physics_features.dtype,
            device=device,
        )
        outputs.append(head(torch.cat((physics_features, action_features), dim=-1)).cpu().numpy())
    return np.concatenate(outputs)


@torch.inference_mode()
def decode_conditioned_soft_residuals(
    logits: np.ndarray,
    base_actions: np.ndarray,
    tokenizer,
    head,
    device: torch.device,
    temperature: float,
    batch_size: int = 32,
) -> np.ndarray:
    """Decode expected EMA-codebook latents under the VLM physics distribution."""
    if temperature <= 0:
        raise ValueError("soft physics temperature must be positive")
    expected_shape = (tokenizer.num_tokens, tokenizer.codebook_size)
    if logits.ndim != 3 or logits.shape[1:] != expected_shape:
        raise ValueError(
            f"physics logits must have shape [B,{expected_shape[0]},{expected_shape[1]}]"
        )
    codebook = tokenizer.quantizer.codebook.float().to(device)
    outputs = []
    for start in range(0, len(logits), batch_size):
        batch_logits = torch.as_tensor(
            logits[start : start + batch_size], dtype=torch.float32, device=device
        )
        probabilities = torch.softmax(batch_logits / temperature, dim=-1)
        quantized = (probabilities @ codebook).reshape(
            len(batch_logits),
            tokenizer.num_transitions,
            tokenizer.queries_per_transition,
            -1,
        )
        physics_features = quantized.mean(dim=2)
        action_features = torch.as_tensor(
            base_actions[start : start + batch_size, DECODER_INDICES],
            dtype=physics_features.dtype,
            device=device,
        )
        outputs.append(head(torch.cat((physics_features, action_features), dim=-1)).cpu().numpy())
    return np.concatenate(outputs)


def probability_diagnostics(logits: np.ndarray) -> dict[str, float]:
    values = torch.as_tensor(logits, dtype=torch.float32)
    probabilities = values.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return {
        "mean_max_probability": float(probabilities.amax(dim=-1).mean()),
        "mean_normalized_entropy": float(entropy.mean() / np.log(logits.shape[-1])),
    }


def mean_token_confidence(logits: np.ndarray) -> np.ndarray:
    values = torch.as_tensor(logits, dtype=torch.float32)
    return values.softmax(dim=-1).amax(dim=-1).mean(dim=-1).numpy()


def confidence_gated_residuals(
    action_only: np.ndarray,
    physics_conditioned: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    scale: float,
) -> np.ndarray:
    if confidence.shape != (len(action_only),):
        raise ValueError("confidence must contain one scalar per sample")
    active = (confidence >= threshold).astype(action_only.dtype)[:, None, None]
    return action_only + active * scale * (physics_conditioned - action_only)


def add_residual(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    result = base[:, DECODER_INDICES].copy() + residual
    result[..., :12] = np.remainder(result[..., :12] + np.pi, 2.0 * np.pi) - np.pi
    result[..., 12:14] = np.clip(result[..., 12:14], 0.0, 1.0)
    return result


def blend_residuals(
    action_only: np.ndarray, physics_conditioned: np.ndarray, scale: float
) -> np.ndarray:
    """Interpolate from the action-only residual to the physics-conditioned residual."""
    if action_only.shape != physics_conditioned.shape:
        raise ValueError("action-only and physics-conditioned residual shapes must match")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("physics blend scale must be in [0, 1]")
    return action_only + scale * (physics_conditioned - action_only)


def scale_label(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def action_metrics(values: np.ndarray, samples: list[dict], sampled: bool = False) -> dict:
    accumulator = MetricAccumulator()
    for index, sample in enumerate(samples):
        target = np.asarray(sample["action"])
        valid = sample.get("action_valid_mask")
        dim_valid = sample.get("action_dim_valid_mask")
        if sampled:
            target = target[DECODER_INDICES]
            if valid is not None:
                valid = np.asarray(valid)[DECODER_INDICES]
            if dim_valid is not None and np.asarray(dim_valid).ndim == 2:
                dim_valid = np.asarray(dim_valid)[DECODER_INDICES]
        accumulator.update(values[index], target, valid, dim_valid)
    return accumulator.compute()


def mean_task_metric(tasks: dict, section: str, metric: str) -> float:
    return float(np.mean([task[section][metric] for task in tasks.values()]))


def episode_local_shuffled_rows(
    panel: list[dict[str, int]], rng: np.random.Generator
) -> np.ndarray:
    """Return a non-identity cyclic shuffle within each episode in a task panel."""
    episode_ids = np.asarray([item["episode_id"] for item in panel], dtype=np.int64)
    rows = np.arange(len(panel), dtype=np.int64)
    shuffled = rows.copy()
    for episode_id in np.unique(episode_ids):
        group = rows[episode_ids == episode_id]
        if len(group) > 1:
            shuffled[group] = np.roll(group, int(rng.integers(1, len(group))))
    return shuffled


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    validation_episodes = [int(value) for value in args.validation_episodes.split(",") if value]
    soft_temperatures = tuple(
        float(value) for value in args.soft_physics_temperatures.split(",") if value
    )
    if any(value <= 0 for value in soft_temperatures) or len(set(soft_temperatures)) != len(
        soft_temperatures
    ):
        raise ValueError("soft physics temperatures must be unique positive values")
    confidence_thresholds = tuple(
        float(value) for value in args.hard_confidence_thresholds.split(",") if value
    )
    if any(not 0.0 <= value <= 1.0 for value in confidence_thresholds) or len(
        set(confidence_thresholds)
    ) != len(confidence_thresholds):
        raise ValueError("hard confidence thresholds must be unique values in [0, 1]")
    device = torch.device(args.device)
    config = OmegaConf.load(args.config)
    config.framework.dual_token_probe.enable_action_teacher = False
    config.datasets.vla_data.discrete_token_cache = str(args.token_cache.resolve())
    baseline_config = OmegaConf.load(args.baseline_config)
    baseline_config.datasets.vla_data.discrete_token_cache = str(args.token_cache.resolve())
    dual_model = load_reasoner(config, args.checkpoint.resolve(), device)
    baseline_model = load_reasoner(baseline_config, args.baseline_checkpoint.resolve(), device)
    action_tokenizer = load_tokenizer(args.action_tokenizer_checkpoint.resolve(), device)
    physics_state = torch.load(
        args.physics_tokenizer_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    physics_tokenizer = build_physics_tokenizer(physics_state["model_config"])
    physics_tokenizer.load_state_dict(physics_state["model"], strict=True)
    physics_tokenizer.to(device).eval()
    conditioned_state = None
    conditioned_physics_head = None
    conditioned_action_head = None
    if args.conditioned_decoder_checkpoint is not None:
        conditioned_state = torch.load(
            args.conditioned_decoder_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        latent_dim = int(physics_state["model_config"]["latent_dim"])
        conditioned_physics_head = make_head(latent_dim + 14, 256)
        conditioned_action_head = make_head(14, 256)
        conditioned_physics_head.load_state_dict(conditioned_state["physics_head"], strict=True)
        conditioned_action_head.load_state_dict(conditioned_state["action_head"], strict=True)
        conditioned_physics_head.to(device).eval()
        conditioned_action_head.to(device).eval()

    with np.load(args.token_cache, allow_pickle=False) as cache:
        task_names = [str(value) for value in cache["task_names"].tolist()]
    task_results = {}
    all_action_predictions, all_action_targets = [], []
    all_physics_predictions, all_physics_targets = [], []
    rng = np.random.default_rng(args.seed)

    for task_id, task_name in enumerate(task_names):
        data_cfg = OmegaConf.create(OmegaConf.to_container(config.datasets.vla_data, resolve=True))
        data_cfg.include_episode_ids = validation_episodes
        data_cfg.exclude_episode_ids = None
        dataset = make_LeRobotSingleDataset(
            Path(str(data_cfg.data_root_dir)), task_name, "robotwin_wrap32", data_cfg=data_cfg
        )
        panel = select_panel(dataset, args.samples_per_task, args.seed + task_id)
        samples = [
            materialize_sample(dataset, item, args.seed + task_id * 10000 + ordinal)
            for ordinal, item in enumerate(panel)
        ]
        action_targets = np.stack([sample["action_token_ids"] for sample in samples])
        physics_targets = np.stack([sample["physics_token_ids"] for sample in samples])
        dual_action, predicted_physics, baseline_action, physics_logits = [], [], [], []
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            dual = dual_model.predict_token_ids(
                batch,
                return_physics_logits=bool(soft_temperatures or confidence_thresholds),
            )
            dual_action.append(dual["action_token_ids"])
            predicted_physics.append(dual["physics_token_ids"])
            if soft_temperatures or confidence_thresholds:
                physics_logits.append(dual["physics_logits"])
            baseline_action.append(baseline_model.predict_token_ids(batch))
        dual_action = np.concatenate(dual_action)
        predicted_physics = np.concatenate(predicted_physics)
        baseline_action = np.concatenate(baseline_action)
        physics_logits = np.concatenate(physics_logits) if physics_logits else None

        dual_base = decode_actions(dual_action, action_tokenizer, device)
        baseline_base = decode_actions(baseline_action, action_tokenizer, device)
        predicted_residual = decode_physics_residuals(
            predicted_physics, physics_tokenizer, device
        )
        oracle_residual = decode_physics_residuals(
            physics_targets, physics_tokenizer, device
        )
        permutation = episode_local_shuffled_rows(panel, rng)
        shuffled_residual = decode_physics_residuals(
            predicted_physics[permutation], physics_tokenizer, device
        )
        conditioned_metrics = {}
        if conditioned_physics_head is not None and conditioned_action_head is not None:
            conditioned_predicted = decode_conditioned_residuals(
                predicted_physics, dual_base, physics_tokenizer, conditioned_physics_head, device
            )
            conditioned_oracle = decode_conditioned_residuals(
                physics_targets, dual_base, physics_tokenizer, conditioned_physics_head, device
            )
            conditioned_shuffled = decode_conditioned_residuals(
                predicted_physics[permutation],
                dual_base,
                physics_tokenizer,
                conditioned_physics_head,
                device,
            )
            base_tensor = torch.as_tensor(
                dual_base[:, DECODER_INDICES], dtype=torch.float32, device=device
            )
            conditioned_action_only = conditioned_action_head(base_tensor).cpu().numpy()
            reference_predicted = decode_conditioned_residuals(
                predicted_physics,
                baseline_base,
                physics_tokenizer,
                conditioned_physics_head,
                device,
            )
            reference_oracle = decode_conditioned_residuals(
                physics_targets,
                baseline_base,
                physics_tokenizer,
                conditioned_physics_head,
                device,
            )
            reference_shuffled = decode_conditioned_residuals(
                predicted_physics[permutation],
                baseline_base,
                physics_tokenizer,
                conditioned_physics_head,
                device,
            )
            reference_base_tensor = torch.as_tensor(
                baseline_base[:, DECODER_INDICES], dtype=torch.float32, device=device
            )
            reference_action_only = conditioned_action_head(
                reference_base_tensor
            ).cpu().numpy()
            conditioned_metrics = {
                "conditioned_action_only": action_metrics(
                    add_residual(dual_base, conditioned_action_only), samples, sampled=True
                ),
                "conditioned_action_plus_predicted_physics": action_metrics(
                    add_residual(dual_base, conditioned_predicted), samples, sampled=True
                ),
                "conditioned_action_plus_oracle_physics": action_metrics(
                    add_residual(dual_base, conditioned_oracle), samples, sampled=True
                ),
                "conditioned_action_plus_shuffled_predicted_physics": action_metrics(
                    add_residual(dual_base, conditioned_shuffled), samples, sampled=True
                ),
                "d17_conditioned_action_only": action_metrics(
                    add_residual(baseline_base, reference_action_only),
                    samples,
                    sampled=True,
                ),
                "d17_conditioned_action_plus_predicted_physics": action_metrics(
                    add_residual(baseline_base, reference_predicted),
                    samples,
                    sampled=True,
                ),
                "d17_conditioned_action_plus_oracle_physics": action_metrics(
                    add_residual(baseline_base, reference_oracle),
                    samples,
                    sampled=True,
                ),
                "d17_conditioned_action_plus_shuffled_predicted_physics": action_metrics(
                    add_residual(baseline_base, reference_shuffled),
                    samples,
                    sampled=True,
                ),
            }
            for scale in PHYSICS_BLEND_SCALES:
                label = scale_label(scale)
                for variant, residual in (
                    ("predicted", conditioned_predicted),
                    ("oracle", conditioned_oracle),
                    ("shuffled_predicted", conditioned_shuffled),
                ):
                    blended = blend_residuals(conditioned_action_only, residual, scale)
                    conditioned_metrics[
                        f"conditioned_action_plus_{variant}_physics_scale_{label}"
                    ] = action_metrics(
                        add_residual(dual_base, blended), samples, sampled=True
                    )
            if physics_logits is not None:
                for temperature in soft_temperatures:
                    temperature_label = scale_label(temperature)
                    soft_residual = decode_conditioned_soft_residuals(
                        physics_logits,
                        dual_base,
                        physics_tokenizer,
                        conditioned_physics_head,
                        device,
                        temperature,
                    )
                    soft_shuffled_residual = decode_conditioned_soft_residuals(
                        physics_logits[permutation],
                        dual_base,
                        physics_tokenizer,
                        conditioned_physics_head,
                        device,
                        temperature,
                    )
                    for scale in PHYSICS_BLEND_SCALES:
                        label = scale_label(scale)
                        for variant, residual in (
                            ("soft", soft_residual),
                            ("shuffled_soft", soft_shuffled_residual),
                        ):
                            blended = blend_residuals(
                                conditioned_action_only, residual, scale
                            )
                            conditioned_metrics[
                                "conditioned_action_plus_"
                                f"{variant}_physics_t{temperature_label}_scale_{label}"
                            ] = action_metrics(
                                add_residual(dual_base, blended), samples, sampled=True
                            )
            if physics_logits is not None and confidence_thresholds:
                confidence = mean_token_confidence(physics_logits)
                shuffled_confidence = confidence[permutation]
                for threshold in confidence_thresholds:
                    threshold_label = scale_label(threshold)
                    conditioned_metrics[
                        f"hard_confidence_coverage_t{threshold_label}"
                    ] = float(np.mean(confidence >= threshold))
                    conditioned_metrics[
                        f"hard_shuffled_confidence_coverage_t{threshold_label}"
                    ] = float(np.mean(shuffled_confidence >= threshold))
                    for scale in PHYSICS_BLEND_SCALES:
                        label = scale_label(scale)
                        gated = confidence_gated_residuals(
                            conditioned_action_only,
                            conditioned_predicted,
                            confidence,
                            threshold,
                            scale,
                        )
                        shuffled_gated = confidence_gated_residuals(
                            conditioned_action_only,
                            conditioned_shuffled,
                            shuffled_confidence,
                            threshold,
                            scale,
                        )
                        conditioned_metrics[
                            "conditioned_action_plus_hard_confidence_physics_"
                            f"t{threshold_label}_scale_{label}"
                        ] = action_metrics(
                            add_residual(dual_base, gated), samples, sampled=True
                        )
                        conditioned_metrics[
                            "conditioned_action_plus_shuffled_hard_confidence_physics_"
                            f"t{threshold_label}_scale_{label}"
                        ] = action_metrics(
                            add_residual(dual_base, shuffled_gated), samples, sampled=True
                        )
        task_results[task_name] = {
            "num_samples": len(samples),
            "action_token_metrics": token_accuracy_metrics(
                dual_action, action_targets, dual_model.action_vocab_size
            ),
            "physics_token_metrics": token_accuracy_metrics(
                predicted_physics, physics_targets, dual_model.physics_vocab_size
            ),
            "action_prediction_agreement_with_d17": float(
                np.mean(dual_action == baseline_action)
            ),
            "dual_action_only": action_metrics(dual_base, samples),
            "d17_action_only": action_metrics(baseline_base, samples),
            "dual_action_sampled_only": action_metrics(
                dual_base[:, DECODER_INDICES], samples, sampled=True
            ),
            "dual_action_plus_predicted_physics": action_metrics(
                add_residual(dual_base, predicted_residual), samples, sampled=True
            ),
            "dual_action_plus_oracle_physics": action_metrics(
                add_residual(dual_base, oracle_residual), samples, sampled=True
            ),
            "dual_action_plus_shuffled_predicted_physics": action_metrics(
                add_residual(dual_base, shuffled_residual), samples, sampled=True
            ),
            **conditioned_metrics,
        }
        if physics_logits is not None:
            task_results[task_name]["physics_probability_diagnostics"] = (
                probability_diagnostics(physics_logits)
            )
        all_action_predictions.append(dual_action)
        all_action_targets.append(action_targets)
        all_physics_predictions.append(predicted_physics)
        all_physics_targets.append(physics_targets)

    action_only = mean_task_metric(task_results, "dual_action_sampled_only", "action_mae")
    predicted = mean_task_metric(
        task_results, "dual_action_plus_predicted_physics", "action_mae"
    )
    oracle = mean_task_metric(
        task_results, "dual_action_plus_oracle_physics", "action_mae"
    )
    shuffled = mean_task_metric(
        task_results, "dual_action_plus_shuffled_predicted_physics", "action_mae"
    )
    dual_mae = mean_task_metric(task_results, "dual_action_only", "action_mae")
    baseline_mae = mean_task_metric(task_results, "d17_action_only", "action_mae")
    all_action_predictions = np.concatenate(all_action_predictions)
    all_action_targets = np.concatenate(all_action_targets)
    all_physics_predictions = np.concatenate(all_physics_predictions)
    all_physics_targets = np.concatenate(all_physics_targets)
    report = {
        "schema_version": 1,
        "kind": "d23_multitask_joint_action_physics_probe",
        "model_config": {
            "action_vocab_size": int(dual_model.action_vocab_size),
            "action_token_count": int(dual_model.action_token_count),
            "physics_vocab_size": int(dual_model.physics_vocab_size),
            "physics_token_count": int(dual_model.physics_token_count),
        },
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
            "baseline_checkpoint_sha256": sha256_file(args.baseline_checkpoint.resolve()),
            "token_cache": str(args.token_cache.resolve()),
            "token_cache_sha256": sha256_file(args.token_cache.resolve()),
            "physics_tokenizer_checkpoint": str(args.physics_tokenizer_checkpoint.resolve()),
            "physics_tokenizer_checkpoint_sha256": sha256_file(
                args.physics_tokenizer_checkpoint.resolve()
            ),
        },
        "validation_episode_ids": validation_episodes,
        "physics_shuffle_mode": "task_and_episode_local_cyclic",
        "samples_per_task": args.samples_per_task,
        "macro": {
            "dual_action_mae": dual_mae,
            "d17_action_mae": baseline_mae,
            "action_mae_relative_change_vs_d17": (dual_mae - baseline_mae) / baseline_mae,
            "sampled_action_only_mae": action_only,
            "action_plus_predicted_physics_mae": predicted,
            "action_plus_oracle_physics_mae": oracle,
            "action_plus_shuffled_predicted_physics_mae": shuffled,
            "predicted_physics_improvement_vs_action_only": (action_only - predicted)
            / action_only,
            "oracle_physics_improvement_vs_action_only": (action_only - oracle)
            / action_only,
            "predicted_physics_improvement_vs_shuffled": (shuffled - predicted) / shuffled,
            "action_prediction_agreement_with_d17": float(
                np.mean(
                    [task["action_prediction_agreement_with_d17"] for task in task_results.values()]
                )
            ),
        },
        "global_action_token_metrics": token_accuracy_metrics(
            all_action_predictions, all_action_targets, dual_model.action_vocab_size
        ),
        "global_physics_token_metrics": token_accuracy_metrics(
            all_physics_predictions, all_physics_targets, dual_model.physics_vocab_size
        ),
        "global_physics_prediction_distribution": distribution_stats(
            all_physics_predictions, dual_model.physics_vocab_size
        ),
        "tasks": task_results,
    }
    if conditioned_physics_head is not None:
        conditioned_action_only = mean_task_metric(
            task_results, "conditioned_action_only", "action_mae"
        )
        conditioned_predicted = mean_task_metric(
            task_results, "conditioned_action_plus_predicted_physics", "action_mae"
        )
        conditioned_oracle = mean_task_metric(
            task_results, "conditioned_action_plus_oracle_physics", "action_mae"
        )
        conditioned_shuffled = mean_task_metric(
            task_results,
            "conditioned_action_plus_shuffled_predicted_physics",
            "action_mae",
        )
        report["inputs"]["conditioned_decoder_checkpoint"] = str(
            args.conditioned_decoder_checkpoint.resolve()
        )
        report["inputs"]["conditioned_decoder_checkpoint_sha256"] = sha256_file(
            args.conditioned_decoder_checkpoint.resolve()
        )
        report["conditioned_macro"] = {
            "action_only_mae": conditioned_action_only,
            "action_plus_predicted_physics_mae": conditioned_predicted,
            "action_plus_oracle_physics_mae": conditioned_oracle,
            "action_plus_shuffled_predicted_physics_mae": conditioned_shuffled,
            "predicted_improvement_vs_action_only": (
                conditioned_action_only - conditioned_predicted
            ) / conditioned_action_only,
            "predicted_improvement_vs_shuffled": (
                conditioned_shuffled - conditioned_predicted
            ) / conditioned_shuffled,
            "oracle_improvement_vs_action_only": (
                conditioned_action_only - conditioned_oracle
            ) / conditioned_action_only,
        }
        reference_action_only = mean_task_metric(
            task_results, "d17_conditioned_action_only", "action_mae"
        )
        reference_predicted = mean_task_metric(
            task_results, "d17_conditioned_action_plus_predicted_physics", "action_mae"
        )
        reference_oracle = mean_task_metric(
            task_results, "d17_conditioned_action_plus_oracle_physics", "action_mae"
        )
        reference_shuffled = mean_task_metric(
            task_results,
            "d17_conditioned_action_plus_shuffled_predicted_physics",
            "action_mae",
        )
        report["d17_reference_conditioned_macro"] = {
            "action_only_mae": reference_action_only,
            "action_plus_predicted_physics_mae": reference_predicted,
            "action_plus_oracle_physics_mae": reference_oracle,
            "action_plus_shuffled_predicted_physics_mae": reference_shuffled,
            "predicted_improvement_vs_action_only": (
                reference_action_only - reference_predicted
            ) / reference_action_only,
            "predicted_improvement_vs_shuffled": (
                reference_shuffled - reference_predicted
            ) / reference_shuffled,
            "oracle_improvement_vs_action_only": (
                reference_action_only - reference_oracle
            ) / reference_action_only,
        }
        scale_sweep = {}
        for scale in PHYSICS_BLEND_SCALES:
            label = scale_label(scale)
            predicted_scaled = mean_task_metric(
                task_results,
                f"conditioned_action_plus_predicted_physics_scale_{label}",
                "action_mae",
            )
            oracle_scaled = mean_task_metric(
                task_results,
                f"conditioned_action_plus_oracle_physics_scale_{label}",
                "action_mae",
            )
            shuffled_scaled = mean_task_metric(
                task_results,
                f"conditioned_action_plus_shuffled_predicted_physics_scale_{label}",
                "action_mae",
            )
            scale_sweep[f"{scale:g}"] = {
                "action_plus_predicted_physics_mae": predicted_scaled,
                "action_plus_oracle_physics_mae": oracle_scaled,
                "action_plus_shuffled_predicted_physics_mae": shuffled_scaled,
                "predicted_improvement_vs_action_only": (
                    conditioned_action_only - predicted_scaled
                ) / conditioned_action_only,
                "predicted_improvement_vs_shuffled": (
                    shuffled_scaled - predicted_scaled
                ) / shuffled_scaled,
                "oracle_improvement_vs_action_only": (
                    conditioned_action_only - oracle_scaled
                ) / conditioned_action_only,
            }
        report["conditioned_scale_sweep"] = {
            "definition": "action_only + scale * (physics_conditioned - action_only)",
            "action_only_mae": conditioned_action_only,
            "scales": scale_sweep,
        }
        if soft_temperatures:
            soft_sweeps = {}
            for temperature in soft_temperatures:
                temperature_label = scale_label(temperature)
                temperature_scales = {}
                for scale in PHYSICS_BLEND_SCALES:
                    label = scale_label(scale)
                    predicted_scaled = mean_task_metric(
                        task_results,
                        "conditioned_action_plus_"
                        f"soft_physics_t{temperature_label}_scale_{label}",
                        "action_mae",
                    )
                    shuffled_scaled = mean_task_metric(
                        task_results,
                        "conditioned_action_plus_"
                        f"shuffled_soft_physics_t{temperature_label}_scale_{label}",
                        "action_mae",
                    )
                    temperature_scales[f"{scale:g}"] = {
                        "action_plus_soft_physics_mae": predicted_scaled,
                        "action_plus_shuffled_soft_physics_mae": shuffled_scaled,
                        "predicted_improvement_vs_action_only": (
                            conditioned_action_only - predicted_scaled
                        )
                        / conditioned_action_only,
                        "predicted_improvement_vs_shuffled": (
                            shuffled_scaled - predicted_scaled
                        )
                        / shuffled_scaled,
                    }
                soft_sweeps[f"{temperature:g}"] = {"scales": temperature_scales}
            report["soft_conditioned_sweeps"] = {
                "definition": (
                    "softmax(physics_logits / temperature) @ EMA codebook, followed by "
                    "action_only + scale * (soft_physics - action_only)"
                ),
                "action_only_mae": conditioned_action_only,
                "temperatures": soft_sweeps,
            }
        if confidence_thresholds:
            confidence_sweeps = {}
            for threshold in confidence_thresholds:
                threshold_label = scale_label(threshold)
                threshold_scales = {}
                coverage = float(
                    np.mean(
                        [
                            task_results[name][
                                f"hard_confidence_coverage_t{threshold_label}"
                            ]
                            for name in task_results
                        ]
                    )
                )
                shuffled_coverage = float(
                    np.mean(
                        [
                            task_results[name][
                                f"hard_shuffled_confidence_coverage_t{threshold_label}"
                            ]
                            for name in task_results
                        ]
                    )
                )
                for scale in PHYSICS_BLEND_SCALES:
                    label = scale_label(scale)
                    predicted_scaled = mean_task_metric(
                        task_results,
                        "conditioned_action_plus_hard_confidence_physics_"
                        f"t{threshold_label}_scale_{label}",
                        "action_mae",
                    )
                    shuffled_scaled = mean_task_metric(
                        task_results,
                        "conditioned_action_plus_shuffled_hard_confidence_physics_"
                        f"t{threshold_label}_scale_{label}",
                        "action_mae",
                    )
                    threshold_scales[f"{scale:g}"] = {
                        "action_plus_gated_physics_mae": predicted_scaled,
                        "action_plus_shuffled_gated_physics_mae": shuffled_scaled,
                        "predicted_improvement_vs_action_only": (
                            conditioned_action_only - predicted_scaled
                        )
                        / conditioned_action_only,
                        "predicted_improvement_vs_shuffled": (
                            shuffled_scaled - predicted_scaled
                        )
                        / shuffled_scaled,
                    }
                confidence_sweeps[f"{threshold:g}"] = {
                    "active_fraction": coverage,
                    "shuffled_active_fraction": shuffled_coverage,
                    "scales": threshold_scales,
                }
            report["hard_confidence_gated_sweeps"] = {
                "confidence": "mean over physics-token positions of max softmax probability",
                "action_only_mae": conditioned_action_only,
                "thresholds": confidence_sweeps,
            }
        report["conditioned_gate"] = {
            "predicted_improves_action_only_by_1pct": report["conditioned_macro"][
                "predicted_improvement_vs_action_only"
            ]
            >= 0.01,
            "predicted_improves_shuffled_by_1pct": report["conditioned_macro"][
                "predicted_improvement_vs_shuffled"
            ]
            >= 0.01,
        }
        report["conditioned_gate"]["passed"] = all(report["conditioned_gate"].values())
    report["gate"] = {
        "action_mae_degradation_at_most_1pct": report["macro"][
            "action_mae_relative_change_vs_d17"
        ]
        <= 0.01,
        "physics_token_accuracy_above_chance": report["global_physics_token_metrics"][
            "accuracy"
        ]
        > 1.0 / dual_model.physics_vocab_size,
        "predicted_physics_improves_action_only_by_1pct": report["macro"][
            "predicted_physics_improvement_vs_action_only"
        ]
        >= 0.01,
        "predicted_physics_improves_shuffled_by_1pct": report["macro"][
            "predicted_physics_improvement_vs_shuffled"
        ]
        >= 0.01,
    }
    report["gate"]["passed"] = all(report["gate"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
