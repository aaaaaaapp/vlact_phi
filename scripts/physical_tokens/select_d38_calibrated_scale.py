#!/usr/bin/env python3
"""Select a residual scale on calibration episodes and verify it on disjoint episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_report(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def validate_reports(calibration: dict, confirmation: dict) -> None:
    calibration_episodes = set(calibration["validation_episode_ids"])
    confirmation_episodes = set(confirmation["validation_episode_ids"])
    if calibration_episodes & confirmation_episodes:
        raise ValueError("calibration and confirmation episodes must be disjoint")
    for key in (
        "checkpoint_sha256",
        "baseline_checkpoint_sha256",
        "token_cache_sha256",
        "physics_tokenizer_checkpoint_sha256",
        "conditioned_decoder_checkpoint_sha256",
    ):
        if calibration["inputs"].get(key) != confirmation["inputs"].get(key):
            raise ValueError(f"calibration and confirmation differ in {key}")
    if calibration["model_config"] != confirmation["model_config"]:
        raise ValueError("calibration and confirmation model configs differ")
    if calibration.get("physics_shuffle_mode") != "task_and_episode_local_cyclic":
        raise ValueError("calibration report has the wrong physics shuffle mode")
    if confirmation.get("physics_shuffle_mode") != "task_and_episode_local_cyclic":
        raise ValueError("confirmation report has the wrong physics shuffle mode")


def select_scale(calibration: dict) -> tuple[str, dict]:
    scales = calibration["conditioned_scale_sweep"]["scales"]
    candidates = []
    for label, metrics in scales.items():
        scale = float(label)
        if scale <= 0.0:
            continue
        score = min(
            float(metrics["predicted_improvement_vs_action_only"]),
            float(metrics["predicted_improvement_vs_shuffled"]),
        )
        candidates.append((score, -scale, label, metrics))
    if not candidates:
        raise ValueError("calibration report contains no nonzero scale candidates")
    _, _, label, metrics = max(candidates)
    return label, metrics


def build_report(calibration: dict, confirmation: dict) -> dict:
    validate_reports(calibration, confirmation)
    selected_label, calibration_metrics = select_scale(calibration)
    confirmation_scales = confirmation["conditioned_scale_sweep"]["scales"]
    if selected_label not in confirmation_scales:
        raise ValueError(f"confirmation report is missing selected scale {selected_label}")
    confirmation_metrics = confirmation_scales[selected_label]
    calibration_candidate = (
        float(calibration_metrics["predicted_improvement_vs_action_only"]) > 0.0
        and float(calibration_metrics["predicted_improvement_vs_shuffled"]) > 0.0
    )
    chance = 1.0 / float(confirmation["model_config"]["physics_vocab_size"])
    checks = {
        "calibration_improves_both_controls": calibration_candidate,
        "confirmation_action_degradation_at_most_1pct": float(
            confirmation["macro"]["action_mae_relative_change_vs_d17"]
        )
        <= 0.01,
        "confirmation_physics_accuracy_above_chance": float(
            confirmation["global_physics_token_metrics"]["accuracy"]
        )
        > chance,
        "confirmation_improves_action_only_by_1pct": float(
            confirmation_metrics["predicted_improvement_vs_action_only"]
        )
        >= 0.01,
        "confirmation_improves_shuffled_by_1pct": float(
            confirmation_metrics["predicted_improvement_vs_shuffled"]
        )
        >= 0.01,
    }
    return {
        "schema_version": 1,
        "kind": "d38_disjoint_residual_scale_confirmation",
        "selection_rule": (
            "maximize min(predicted improvement vs action-only, predicted improvement vs "
            "shuffled) over the fixed nonzero scale grid; ties choose the smaller scale"
        ),
        "calibration_episode_ids": calibration["validation_episode_ids"],
        "confirmation_episode_ids": confirmation["validation_episode_ids"],
        "selected_scale": float(selected_label),
        "calibration_selected_scale_metrics": calibration_metrics,
        "confirmation_selected_scale_metrics": confirmation_metrics,
        "confirmation_action_mae_relative_change_vs_d17": confirmation["macro"][
            "action_mae_relative_change_vs_d17"
        ],
        "confirmation_physics_token_accuracy": confirmation[
            "global_physics_token_metrics"
        ]["accuracy"],
        "physics_chance_accuracy": chance,
        "checks": checks,
        "independent_scale_gate_passed": all(checks.values()),
        "dit_authorized": False,
        "dit_status": (
            "D38 is a calibrated-head diagnostic; a pass is evidence for a fixed-scale "
            "candidate, not automatic authorization to train DiT"
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(load_report(args.calibration), load_report(args.confirmation))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
