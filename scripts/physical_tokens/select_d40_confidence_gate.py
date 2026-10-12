#!/usr/bin/env python3
"""Select a hard-ID confidence gate on calibration and verify it on confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from select_d38_calibrated_scale import load_report, validate_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_candidate(calibration: dict) -> tuple[float, float, dict, float]:
    ranked = []
    thresholds = calibration["hard_confidence_gated_sweeps"]["thresholds"]
    for threshold_label, threshold_report in thresholds.items():
        for scale_label, metrics in threshold_report["scales"].items():
            score = min(
                float(metrics["predicted_improvement_vs_action_only"]),
                float(metrics["predicted_improvement_vs_shuffled"]),
            )
            ranked.append(
                (
                    score,
                    -float(scale_label),
                    float(threshold_label),
                    float(scale_label),
                    metrics,
                    float(threshold_report["active_fraction"]),
                )
            )
    if not ranked:
        raise ValueError("calibration report contains no confidence-gate candidates")
    _, _, threshold, scale, metrics, coverage = max(ranked)
    return threshold, scale, metrics, coverage


def lookup(report: dict, threshold: float, scale: float) -> tuple[dict, float]:
    threshold_report = report["hard_confidence_gated_sweeps"]["thresholds"][
        f"{threshold:g}"
    ]
    return threshold_report["scales"][f"{scale:g}"], float(
        threshold_report["active_fraction"]
    )


def build_report(calibration: dict, confirmation: dict) -> dict:
    validate_reports(calibration, confirmation)
    threshold, scale, calibration_metrics, calibration_coverage = select_candidate(calibration)
    confirmation_metrics, confirmation_coverage = lookup(confirmation, threshold, scale)
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
        "kind": "d40_disjoint_hard_confidence_gate_confirmation",
        "selection_rule": (
            "maximize min(predicted improvement vs action-only, predicted improvement vs "
            "shuffled) over fixed absolute confidence-threshold x scale grid; ties choose "
            "smaller scale, then larger threshold"
        ),
        "calibration_episode_ids": calibration["validation_episode_ids"],
        "confirmation_episode_ids": confirmation["validation_episode_ids"],
        "selected_confidence_threshold": threshold,
        "selected_scale": scale,
        "calibration_active_fraction": calibration_coverage,
        "confirmation_active_fraction": confirmation_coverage,
        "calibration_selected_metrics": calibration_metrics,
        "confirmation_selected_metrics": confirmation_metrics,
        "confirmation_action_mae_relative_change_vs_d17": confirmation["macro"][
            "action_mae_relative_change_vs_d17"
        ],
        "confirmation_physics_token_accuracy": confirmation[
            "global_physics_token_metrics"
        ]["accuracy"],
        "physics_chance_accuracy": chance,
        "checks": checks,
        "independent_confidence_gate_passed": all(checks.values()),
        "dit_authorized": False,
        "dit_status": (
            "D40 evaluates a frozen confidence gate; a pass requires an interface audit "
            "before any DiT training"
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
