#!/usr/bin/env python3
"""Gate spatial routing on reconstruction non-regression and token diversity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.physical_tokens.require_physics_token_structure_gate import (
    evaluate_gate as evaluate_structure_gate,
)


def evaluate_joint_gate(comparison: dict[str, Any], structure: dict[str, Any]) -> dict[str, Any]:
    if comparison.get("comparison") != "paired_same_cache_d1_vs_d2_spatial_routing":
        raise ValueError("D2 comparison has an unexpected comparison name")
    if comparison.get("allowed_model_config_differences") != ["token_routing"]:
        raise ValueError("D2 comparison must vary only token_routing")
    structure_gate = evaluate_structure_gate(structure)
    macro_floor = -0.005
    task_floor = -0.01
    required_tasks = max(1, (3 * int(comparison["num_tasks"]) + 3) // 4)
    continuous_task_non_regressions = sum(
        entry["metrics"]["continuous_rmse"]["relative_improvement"] >= task_floor
        for entry in comparison["tasks"].values()
    )
    quantized_task_non_regressions = sum(
        entry["metrics"]["rmse"]["relative_improvement"] >= task_floor
        for entry in comparison["tasks"].values()
    )
    checks = {
        "continuous_macro_no_regression": (
            comparison["macro"]["continuous_rmse"]["relative_improvement"]
            >= macro_floor
        ),
        "quantized_macro_no_regression": (
            comparison["macro"]["rmse"]["relative_improvement"] >= macro_floor
        ),
        "continuous_task_non_regressions": (
            continuous_task_non_regressions >= required_tasks
        ),
        "quantized_task_non_regressions": (
            quantized_task_non_regressions >= required_tasks
        ),
        "candidate_continuous_no_motion_reduction": comparison["gate"]["checks"][
            "candidate_continuous_no_motion_reduction"
        ],
        "candidate_quantized_no_motion_reduction": comparison["gate"]["checks"][
            "candidate_quantized_no_motion_reduction"
        ],
        "within_transition_unique_ratio": structure_gate["checks"][
            "within_transition_unique_ratio"
        ],
        "within_transition_pairwise_token_change": structure_gate["checks"][
            "within_transition_pairwise_token_change"
        ],
    }
    return {
        "schema_version": 1,
        "kind": "d2_spatial_routing_joint_gate",
        "thresholds": {
            "macro_relative_improvement_floor": macro_floor,
            "task_relative_improvement_floor": task_floor,
            "required_task_non_regressions": required_tasks,
            "candidate_continuous_no_motion_reduction": 0.05,
            "candidate_quantized_no_motion_reduction": 0.03,
            **structure_gate["thresholds"],
        },
        "observed": {
            "continuous_macro_relative_improvement": comparison["macro"][
                "continuous_rmse"
            ]["relative_improvement"],
            "quantized_macro_relative_improvement": comparison["macro"]["rmse"][
                "relative_improvement"
            ],
            "continuous_task_non_regressions": continuous_task_non_regressions,
            "quantized_task_non_regressions": quantized_task_non_regressions,
            **structure_gate["observed"],
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_joint_gate(
        json.loads(args.comparison.read_text()),
        json.loads(args.structure.read_text()),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")
    if not result["passed"]:
        failed = [name for name, passed in result["checks"].items() if not passed]
        raise RuntimeError(f"D2 joint gate failed: {failed}")


if __name__ == "__main__":
    main()
