#!/usr/bin/env python3
"""Evaluate free-running EMA-VQ action-token prediction on held-out episodes."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_action_tokenizer import build_model as build_action_tokenizer
from evaluate_qwenpi_oracle import (
    MetricAccumulator,
    build_validation_dataset,
    load_validation_episodes,
    materialize_sample,
    select_panel,
    sha256_file,
)
from starVLA.model.framework import build_framework
from starVLA.model.framework.QwenActionTokenReasoner import format_action_token_solution
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-count", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--token-cache", type=Path, default=None)
    return parser.parse_args()


def load_reasoner(config, checkpoint: Path, device: torch.device):
    PartialState()
    torch.manual_seed(int(config.get("seed", 42)))
    model = build_framework(config)
    model = TrainerUtils.load_pretrained_backbones(
        model,
        str(config.trainer.pretrained_checkpoint),
        reload_modules="qwen_vl_interface",
    )
    action_init_checkpoint = config.trainer.get(
        "dual_action_probe_init_checkpoint", None
    )
    if action_init_checkpoint:
        if not hasattr(model, "initialize_from_action_probe"):
            raise RuntimeError(
                "dual_action_probe_init_checkpoint requires initialize_from_action_probe"
            )
        model.initialize_from_action_probe(
            action_init_checkpoint,
            config.trainer.get("dual_action_probe_init_sha256", None),
        )
    model = TrainerUtils.freeze_backbones(
        model, freeze_modules=config.trainer.get("freeze_modules", "")
    )
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = set(state)
    if actual != expected:
        raise RuntimeError(
            "reasoner checkpoint/trainable parameter mismatch: "
            f"missing={sorted(expected - actual)[:8]}, unexpected={sorted(actual - expected)[:8]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected reasoner keys: {incompatible.unexpected_keys}")
    model.to(device=device, dtype=torch.bfloat16).eval()
    return model


def load_tokenizer(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_action_tokenizer(state["model_config"])
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def distribution_stats(ids: np.ndarray, vocab_size: int) -> dict[str, float | int]:
    counts = np.bincount(ids.reshape(-1), minlength=vocab_size)
    probabilities = counts / max(1, counts.sum())
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * np.log(probabilities[nonzero])).sum()
    return {
        "active_codes": int(np.count_nonzero(counts)),
        "perplexity": float(math.exp(entropy)),
    }


@torch.inference_mode()
def teacher_forced_predictions(reasoner, samples: list[dict], batch_size: int) -> np.ndarray:
    predictions = []
    offset = reasoner.action_token_offset
    vocabulary_size = reasoner.action_vocab_size
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        inputs = reasoner.qwen_vl_interface.build_qwenvl_inputs(
            images=[sample["image"] for sample in batch],
            instructions=reasoner._reasoner_instructions(batch),
            solutions=[
                format_action_token_solution(
                    sample["action_token_ids"], vocab_size=vocabulary_size
                )
                for sample in batch
            ],
        )
        outputs = reasoner.qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        shifted_labels = inputs["labels"][:, 1:]
        action_logits = outputs.logits[:, :-1, offset : offset + vocabulary_size]
        batch_predictions = []
        for row in range(len(batch)):
            action_mask = (shifted_labels[row] >= offset) & (
                shifted_labels[row] < offset + vocabulary_size
            )
            row_predictions = action_logits[row, action_mask].argmax(dim=-1)
            if row_predictions.numel() != reasoner.action_token_count:
                raise RuntimeError(
                    "teacher-forced label sequence has "
                    f"{row_predictions.numel()} action tokens, expected "
                    f"{reasoner.action_token_count}"
                )
            batch_predictions.append(row_predictions.cpu().numpy())
        predictions.extend(batch_predictions)
    return np.stack(predictions)


def token_accuracy_metrics(
    predictions: np.ndarray, targets: np.ndarray, vocab_size: int
) -> dict:
    correct = predictions == targets
    return {
        "accuracy": float(correct.mean()),
        "exact_sequence_accuracy": float(correct.all(axis=1).mean()),
        "per_position_accuracy": correct.mean(axis=0).tolist(),
        "mean_correct_tokens_per_sequence": float(correct.sum(axis=1).mean()),
        "target_distribution": distribution_stats(targets, vocab_size),
        "prediction_distribution": distribution_stats(predictions, vocab_size),
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    tokenizer_path = args.action_tokenizer_checkpoint.resolve()
    manifest_path = args.split_manifest.resolve()
    config = OmegaConf.load(config_path)
    if args.token_count is not None:
        config.framework.action_token_probe.token_count = args.token_count
    if args.vocab_size is not None:
        config.framework.action_token_probe.vocab_size = args.vocab_size
    if args.token_cache is not None:
        config.datasets.vla_data.discrete_token_cache = str(args.token_cache.resolve())
    validation_episodes, _ = load_validation_episodes(config_path, manifest_path)
    dataset = build_validation_dataset(config, validation_episodes)
    panel = select_panel(dataset, args.num_samples, args.seed)
    samples = [
        materialize_sample(dataset, item, args.seed + ordinal)
        for ordinal, item in enumerate(panel)
    ]
    targets = np.stack(
        [np.asarray(sample["action_token_ids"], dtype=np.int64) for sample in samples]
    )

    device = torch.device(args.device)
    reasoner = load_reasoner(config, checkpoint_path, device)
    teacher_predictions = None
    if hasattr(reasoner, "action_token_offset"):
        teacher_predictions = teacher_forced_predictions(
            reasoner, samples, args.batch_size
        )
    predictions = []
    latencies = []
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions.append(reasoner.predict_token_ids(batch))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - started, len(batch)))
    predictions = np.concatenate(predictions, axis=0)

    tokenizer = load_tokenizer(tokenizer_path, device)
    predicted_actions = tokenizer.decode_tokens(
        torch.from_numpy(predictions).to(device), gripper_dims=(12, 13)
    ).float().cpu().numpy()
    oracle_actions = tokenizer.decode_tokens(
        torch.from_numpy(targets).to(device), gripper_dims=(12, 13)
    ).float().cpu().numpy()
    predicted_metrics = MetricAccumulator()
    oracle_metrics = MetricAccumulator()
    for index, sample in enumerate(samples):
        predicted_metrics.update(
            predicted_actions[index],
            sample["action"],
            sample.get("action_valid_mask"),
            sample.get("action_dim_valid_mask"),
        )
        oracle_metrics.update(
            oracle_actions[index],
            sample["action"],
            sample.get("action_valid_mask"),
            sample.get("action_dim_valid_mask"),
        )

    weighted_latency = sum(seconds for seconds, _ in latencies)
    return {
        "schema_version": 1,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "action_tokenizer_checkpoint": str(tokenizer_path),
        "action_tokenizer_sha256": sha256_file(tokenizer_path),
        "split_manifest": str(manifest_path),
        "validation_episode_ids": validation_episodes,
        "validation_dataset_size": len(dataset),
        "num_samples": len(samples),
        "sample_ids": panel,
        "token_metrics": token_accuracy_metrics(
            predictions, targets, reasoner.action_vocab_size
        ),
        "teacher_forced_token_metrics": (
            token_accuracy_metrics(teacher_predictions, targets, reasoner.action_vocab_size)
            if teacher_predictions is not None
            else None
        ),
        "decoded_action_metrics": {
            "predicted_tokens": predicted_metrics.compute(),
            "oracle_tokens": oracle_metrics.compute(),
        },
        "generation_latency_seconds": {
            "total": float(weighted_latency),
            "mean_per_sample": float(weighted_latency / len(samples)),
        },
    }


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
