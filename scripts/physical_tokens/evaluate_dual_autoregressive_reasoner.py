#!/usr/bin/env python3
"""Evaluate held-out action/physics token prediction on selected RoboTwin tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.physical_tokens.evaluate_qwenpi_oracle import (
    build_validation_dataset,
    load_validation_episodes,
    materialize_sample,
    select_panel,
)
from starVLA.model.framework.QwenDualAutoregressiveReasoner import (
    QwenDualAutoregressiveReasoner,
    format_dual_solution,
    keep_only_dual_target_labels,
    score_separate_heads,
)


def action_transition_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    """Score true code changes separately from easy previous-code repetitions."""
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if prediction.shape != target.shape or target.size < 2:
        raise ValueError("action prediction and target must have equal length >= 2")
    changed = target[1:] != target[:-1]
    return {
        "change_correct": int((prediction[1:][changed] == target[1:][changed]).sum()),
        "change_total": int(changed.sum()),
        "repeat_previous_correct": int((~changed).sum()),
        "repeat_previous_total": int(changed.size),
    }


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--free-samples", type=int, default=2)
    head_mode = parser.add_mutually_exclusive_group()
    head_mode.add_argument("--separate-heads", action="store_true")
    head_mode.add_argument("--shared-heads", action="store_true")
    tie_mode = parser.add_mutually_exclusive_group()
    tie_mode.add_argument("--strict-modality-tying", action="store_true")
    tie_mode.add_argument("--output-only-heads", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_task < 1 or args.free_samples < 0:
        parser.error("sample counts must be non-negative, with samples-per-task positive")
    if args.shared_heads and (args.strict_modality_tying or args.output_only_heads):
        parser.error("--shared-heads cannot be combined with a separate-head mode")

    PartialState()
    config = OmegaConf.load(args.config)
    if args.separate_heads:
        config.framework.dual_autoregressive_reasoner.separate_heads = True
    elif args.shared_heads:
        config.framework.dual_autoregressive_reasoner.separate_heads = False
        config.framework.dual_autoregressive_reasoner.strict_modality_tying = False
    if args.strict_modality_tying:
        config.framework.dual_autoregressive_reasoner.separate_heads = True
        config.framework.dual_autoregressive_reasoner.strict_modality_tying = True
    elif args.output_only_heads:
        config.framework.dual_autoregressive_reasoner.separate_heads = True
        config.framework.dual_autoregressive_reasoner.strict_modality_tying = False
    validation_episodes, _ = load_validation_episodes(args.config, args.split)
    samples = []
    sample_tasks = []
    for task in args.tasks:
        task_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        dataset = build_validation_dataset(task_config, validation_episodes, task_name=task)
        panel = select_panel(dataset, args.samples_per_task, seed=42)
        for index, item in enumerate(panel):
            samples.append(materialize_sample(dataset, item, 42 + index))
            sample_tasks.append(task)

    torch.manual_seed(int(config.get("seed", 42)))
    model = QwenDualAutoregressiveReasoner(config)
    if not config.trainer.get("expanded_vocab_warmstart", False):
        raise ValueError("evaluator requires expanded-vocabulary pretrained warm-start")
    model.initialize_expanded_vocab_pretrained(config.trainer.pretrained_checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError(f"checkpoint trainable keys mismatch: missing={list(expected-set(state))[:5]}, "
                         f"extra={list(set(state)-expected)[:5]}")
    model.load_state_dict(state, strict=False)
    model = model.to(device="cuda:0", dtype=torch.bfloat16).eval()

    action_correct = action_total = physics_correct = physics_total = 0
    action_change_correct = action_change_total = 0
    action_repeat_previous_correct = action_repeat_previous_total = 0
    ce_sum = 0.0
    action_ce_sum = physics_ce_sum = 0.0
    free_results = []
    per_task = {task: {
        "num_samples": 0, "action_ce_sum": 0.0, "physics_ce_sum": 0.0,
        "action_correct": 0, "action_total": 0,
        "action_change_correct": 0, "action_change_total": 0,
        "action_repeat_previous_correct": 0, "action_repeat_previous_total": 0,
        "physics_correct": 0, "physics_total": 0,
        "free_running": [],
    } for task in args.tasks}
    for index, sample in enumerate(samples):
        task = sample_tasks[index]
        task_stats = per_task[task]
        solution = format_dual_solution(
            sample["action_token_ids"], sample["physics_token_ids"],
            action_vocab_size=model.action_vocab_size, physics_vocab_size=model.physics_vocab_size,
            action_count=model.action_token_count, physics_count=model.physics_token_count,
        )
        inputs = model.qwen_vl_interface.build_qwenvl_inputs(
            images=[sample["image"]], instructions=model._reasoner_instructions([sample]),
            solutions=[solution],
        )
        inputs = keep_only_dual_target_labels(
            inputs, action_offset=model.action_token_offset, action_vocab_size=model.action_vocab_size,
            physics_offset=model.physics_token_offset, physics_vocab_size=model.physics_vocab_size,
            action_count=model.action_token_count, physics_count=model.physics_token_count,
        )
        with torch.inference_mode():
            if model.separate_heads:
                targets = inputs.pop("labels")
                if model.strict_modality_tying:
                    hidden = model._strict_backbone_forward(inputs)[0].last_hidden_state
                else:
                    hidden = model.qwen_vl_interface.model.model(
                        **inputs, use_cache=False, return_dict=True,
                    ).last_hidden_state
                scores = score_separate_heads(
                    hidden, targets, action_head=model.action_logits, physics_head=model.physics_logits,
                    action_offset=model.action_token_offset, action_vocab_size=525,
                    physics_offset=model.physics_token_offset, physics_vocab_size=2048,
                )
                action_logits, physics_logits = scores["action_logits"], scores["physics_logits"]
                action_targets, physics_targets = scores["action_targets"], scores["physics_targets"]
                action_ce, physics_ce = scores["action_ce"], scores["physics_ce"]
                ce = model.action_ce_weight * action_ce + model.physics_ce_weight * physics_ce
            else:
                output = model.qwen_vl_interface(
                    **inputs, output_attentions=False, output_hidden_states=False,
                    return_dict=True,
                )
                labels = inputs["labels"][:, 1:]
                shifted_logits = output.logits[:, :-1]
                action_mask = (labels >= model.action_token_offset) & (labels < model.action_token_offset + 525)
                physics_mask = (labels >= model.physics_token_offset) & (labels < model.physics_token_offset + 2048)
                action_logits = shifted_logits[action_mask][:, model.action_token_offset:model.action_token_offset + 525]
                physics_logits = shifted_logits[physics_mask][:, model.physics_token_offset:model.physics_token_offset + 2048]
                action_targets = labels[action_mask] - model.action_token_offset
                physics_targets = labels[physics_mask] - model.physics_token_offset
                action_ce = torch.nn.functional.cross_entropy(shifted_logits[action_mask].float(), labels[action_mask])
                physics_ce = torch.nn.functional.cross_entropy(shifted_logits[physics_mask].float(), labels[physics_mask])
                ce = output.loss
        ce_sum += float(ce)
        action_ce_sum += float(action_ce)
        physics_ce_sum += float(physics_ce)
        action_correct += int((action_logits.argmax(-1) == action_targets).sum())
        transition_counts = action_transition_counts(
            action_logits.argmax(-1).detach().cpu().numpy(),
            action_targets.detach().cpu().numpy(),
        )
        action_change_correct += transition_counts["change_correct"]
        action_change_total += transition_counts["change_total"]
        action_repeat_previous_correct += transition_counts["repeat_previous_correct"]
        action_repeat_previous_total += transition_counts["repeat_previous_total"]
        physics_correct += int((physics_logits.argmax(-1) == physics_targets).sum())
        action_total += int(action_targets.numel())
        physics_total += int(physics_targets.numel())
        task_stats["num_samples"] += 1
        task_stats["action_ce_sum"] += float(action_ce)
        task_stats["physics_ce_sum"] += float(physics_ce)
        task_stats["action_correct"] += int((action_logits.argmax(-1) == action_targets).sum())
        for metric, value in transition_counts.items():
            task_stats["action_" + metric] += value
        task_stats["physics_correct"] += int((physics_logits.argmax(-1) == physics_targets).sum())
        task_stats["action_total"] += int(action_targets.numel())
        task_stats["physics_total"] += int(physics_targets.numel())
        if len(task_stats["free_running"]) < args.free_samples:
            prediction = model.predict_token_ids([sample])
            action_target = np.asarray(sample["action_token_ids"])
            physics_target = np.asarray(sample["physics_token_ids"])
            free_result = {
                "task": task,
                "action_accuracy": float((prediction["action_token_ids"][0] == action_target).mean()),
                "physics_accuracy": float((prediction["physics_token_ids"][0] == physics_target).mean()),
                "action_exact": bool(np.array_equal(prediction["action_token_ids"][0], action_target)),
                "physics_exact": bool(np.array_equal(prediction["physics_token_ids"][0], physics_target)),
            }
            free_transition = action_transition_counts(prediction["action_token_ids"][0], action_target)
            free_result["action_change_accuracy"] = safe_ratio(
                free_transition["change_correct"], free_transition["change_total"]
            )
            free_result["action_change_total"] = free_transition["change_total"]
            free_result["action_repeat_previous_baseline_accuracy"] = safe_ratio(
                free_transition["repeat_previous_correct"], free_transition["repeat_previous_total"]
            )
            free_results.append(free_result)
            task_stats["free_running"].append(free_result)
    per_task_report = {}
    for task, stats in per_task.items():
        if not stats["num_samples"]:
            raise ValueError(f"validation panel is empty for task {task}")
        per_task_report[task] = {
            "num_samples": stats["num_samples"],
            "teacher_forced_action_ce": stats["action_ce_sum"] / stats["num_samples"],
            "teacher_forced_physics_ce": stats["physics_ce_sum"] / stats["num_samples"],
            "teacher_forced_action_accuracy": stats["action_correct"] / stats["action_total"],
            "teacher_forced_action_change_accuracy": safe_ratio(
                stats["action_change_correct"], stats["action_change_total"]
            ),
            "teacher_forced_action_change_total": stats["action_change_total"],
            "action_repeat_previous_baseline_accuracy": safe_ratio(
                stats["action_repeat_previous_correct"], stats["action_repeat_previous_total"]
            ),
            "teacher_forced_physics_accuracy": stats["physics_correct"] / stats["physics_total"],
            "free_running": stats["free_running"],
        }
    report = {
        "num_samples": len(samples), "tasks": args.tasks,
        "validation_episode_ids": validation_episodes,
        "teacher_forced_ce": ce_sum / len(samples),
        "teacher_forced_action_ce": action_ce_sum / len(samples),
        "teacher_forced_physics_ce": physics_ce_sum / len(samples),
        "teacher_forced_action_accuracy": action_correct / action_total,
        "teacher_forced_action_change_accuracy": safe_ratio(action_change_correct, action_change_total),
        "teacher_forced_action_change_total": action_change_total,
        "action_repeat_previous_baseline_accuracy": safe_ratio(
            action_repeat_previous_correct, action_repeat_previous_total
        ),
        "teacher_forced_physics_accuracy": physics_correct / physics_total,
        "free_running": free_results,
        "per_task": per_task_report,
        "output_heads": ("strict_tied" if model.strict_modality_tying else
                         "separate" if model.separate_heads else "shared"),
        "note": "Fixed selected panel per listed task; token-level diagnostic, not rollout evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
