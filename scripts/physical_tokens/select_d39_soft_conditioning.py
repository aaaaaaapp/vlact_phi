#!/usr/bin/env python3
"""Select soft-physics temperature/scale on calibration and verify on confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from select_d38_calibrated_scale import load_report, validate_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def candidates(report: dict):
    for temperature_label, temperature_report in report["soft_conditioned_sweeps"][
        "temperatures"
    ].items():
        for scale_label, metrics in temperature_report["scales"].items():
            yield float(temperature_label), float(scale_label), metrics


def select_candidate(calibration: dict) -> tuple[float, float, dict]:
    ranked = []
    for temperature, scale, metrics in candidates(calibration):
        score = min(
            float(metrics["predicted_improvement_vs_action_only"]),
            float(metrics["predicted_improvement_vs_shuffled"]),
        )
        ranked.append((score, -scale, -abs(math.log(temperature)), temperature, scale, metrics))
    if not ranked:
        raise ValueError("calibration report contains no soft-conditioning candidates")
    _, _, _, temperature, scale, metrics = max(ranked)
    return temperature, scale, metrics


def lookup(report: dict, temperature: float, scale: float) -> dict:
    return report["soft_conditioned_sweeps"]["temperatures"][f"{temperature:g}"][
        "scales"
    ][f"{scale:g}"]


def build_report(calibration: dict, confirmation: dict) -> dict:
    validate_reports(calibration, confirmation)
    temperature, scale, calibration_metrics = select_candidate(calibration)
    confirmation_metrics = lookup(confirmation, temperature, scale)
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
        "kind": "d39_disjoint_soft_conditioning_confirmation",
        "selection_rule": (
            "maximize min(predicted improvement vs action-only, predicted improvement vs "
            "shuffled) over fixed temperature x scale grid; ties choose smaller scale, then "
            "temperature closest to 1"
        ),
        "calibration_episode_ids": calibration["validation_episode_ids"],
        "confirmation_episode_ids": confirmation["validation_episode_ids"],
        "selected_temperature": temperature,
        "selected_scale": scale,
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
        "independent_soft_gate_passed": all(checks.values()),
        "dit_authorized": False,
        "dit_status": (
            "D39 evaluates a frozen soft-condition candidate; a pass requires a separate "
            "decision before any DiT training"
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
