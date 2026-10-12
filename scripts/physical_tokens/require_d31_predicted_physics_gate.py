#!/usr/bin/env python3
"""Apply the final predicted-physics gate to the D31 VLM probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_gate(report: dict) -> dict:
    conditioned = report.get("conditioned_macro")
    if conditioned is None:
        raise ValueError("D31 evaluation is missing conditioned metrics")
    shuffle_mode = report.get("physics_shuffle_mode")
    if shuffle_mode != "task_and_episode_local_cyclic":
        raise ValueError(
            "D31 evaluation must use task-and-episode-local cyclic physics shuffle"
        )
    vocab_size = int(report["model_config"]["physics_vocab_size"])
    accuracy = float(report["global_physics_token_metrics"]["accuracy"])
    action_change = float(report["macro"]["action_mae_relative_change_vs_d17"])
    versus_action = float(conditioned["predicted_improvement_vs_action_only"])
    versus_shuffle = float(conditioned["predicted_improvement_vs_shuffled"])
    checks = {
        "action_mae_degradation_at_most_1pct": action_change <= 0.01,
        "physics_token_accuracy_above_chance": accuracy > 1.0 / vocab_size,
        "conditioned_predicted_improves_action_only_by_1pct": versus_action >= 0.01,
        "conditioned_predicted_improves_shuffled_by_1pct": versus_shuffle >= 0.01,
    }
    return {
        "schema_version": 1,
        "gate": "d31_full_diversity_predicted_physics",
        "physics_vocab_size": vocab_size,
        "physics_token_accuracy": accuracy,
        "chance_accuracy": 1.0 / vocab_size,
        "physics_shuffle_mode": shuffle_mode,
        "action_mae_relative_change_vs_d17": action_change,
        "predicted_improvement_vs_action_only": versus_action,
        "predicted_improvement_vs_shuffled": versus_shuffle,
        "checks": checks,
        "passed": all(checks.values()),
        "dit_authorized": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_gate(json.loads(args.input.read_text()))
    report["input"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
