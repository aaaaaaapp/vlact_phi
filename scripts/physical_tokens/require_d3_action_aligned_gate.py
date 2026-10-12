#!/usr/bin/env python3
"""Gate action-aligned tokenizer training on visual non-regression and diversity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from require_physics_token_structure_gate import evaluate_gate as structure_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--histogram-probe", type=Path, default=None)
    parser.add_argument("--positional-probe", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    structure = structure_gate(json.loads(args.structure.read_text(encoding="utf-8")))
    expected_differences = [
        "action_aux_dim",
        "action_aux_weight",
        "action_gripper_weight",
        "action_velocity_weight",
    ]
    if comparison.get("comparison") != "paired_same_cache_d2_vs_d3_action_aux":
        raise ValueError("unexpected D3 comparison name")
    if comparison.get("allowed_model_config_differences") != expected_differences:
        raise ValueError("unexpected D3 model config differences")
    macro_floor = -0.005
    task_floor = -0.01
    required_tasks = max(1, (3 * int(comparison["num_tasks"]) + 3) // 4)
    continuous_tasks = sum(
        entry["metrics"]["continuous_rmse"]["relative_improvement"] >= task_floor
        for entry in comparison["tasks"].values()
    )
    quantized_tasks = sum(
        entry["metrics"]["rmse"]["relative_improvement"] >= task_floor
        for entry in comparison["tasks"].values()
    )
    checks = {
        "continuous_macro_no_regression": comparison["macro"]["continuous_rmse"]["relative_improvement"] >= macro_floor,
        "quantized_macro_no_regression": comparison["macro"]["rmse"]["relative_improvement"] >= macro_floor,
        "continuous_task_non_regressions": continuous_tasks >= required_tasks,
        "quantized_task_non_regressions": quantized_tasks >= required_tasks,
        "candidate_continuous_no_motion_reduction": comparison["gate"]["checks"]["candidate_continuous_no_motion_reduction"],
        "candidate_quantized_no_motion_reduction": comparison["gate"]["checks"]["candidate_quantized_no_motion_reduction"],
        "within_transition_unique_ratio": structure["checks"]["within_transition_unique_ratio"],
        "within_transition_pairwise_token_change": structure["checks"]["within_transition_pairwise_token_change"],
    }
    result = {
        "schema_version": 1,
        "kind": "d3_action_aligned_representation_gate",
        "thresholds": {
            "macro_relative_improvement_floor": macro_floor,
            "task_relative_improvement_floor": task_floor,
            "required_task_non_regressions": required_tasks,
            "action_relevance_baseline_improvement": 0.01,
            "action_relevance_shuffled_improvement": 0.01,
            **structure["thresholds"],
        },
        "observed": {
            "continuous_macro_relative_improvement": comparison["macro"]["continuous_rmse"]["relative_improvement"],
            "quantized_macro_relative_improvement": comparison["macro"]["rmse"]["relative_improvement"],
            "continuous_task_non_regressions": continuous_tasks,
            "quantized_task_non_regressions": quantized_tasks,
            **structure["observed"],
        },
        "checks": checks,
        "visual_structure_passed": all(checks.values()),
        "passed": False,
        "pending": ["action_relevance_baseline_improvement", "action_relevance_shuffled_improvement"],
    }
    probe_paths = (args.histogram_probe, args.positional_probe)
    if any(path is not None for path in probe_paths):
        if not all(path is not None for path in probe_paths):
            raise ValueError("both D3 action relevance probes must be provided together")
        probe_results = {}
        action_checks = {}
        for name, path in zip(("transition_histogram", "positional_count_sketch"), probe_paths):
            probe = json.loads(path.read_text(encoding="utf-8"))
            aligned = probe["results"]["plus_aligned_physics"]["validation"]
            baseline_gain = aligned["action_mae_relative_improvement_vs_baseline"]
            shuffled_gain = probe["aligned_minus_shuffled_action_mae_relative_improvement"]
            probe_results[name] = {
                "aligned_action_mae": aligned["action_mae"],
                "baseline_relative_improvement": baseline_gain,
                "shuffled_relative_improvement": shuffled_gain,
            }
            action_checks[f"{name}_baseline_improvement"] = baseline_gain >= 0.01
            action_checks[f"{name}_shuffled_improvement"] = shuffled_gain >= 0.01
        result["action_relevance"] = probe_results
        result["checks"].update(action_checks)
        result["pending"] = []
        result["passed"] = result["visual_structure_passed"] and all(action_checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["visual_structure_passed"]:
        raise RuntimeError("D3 visual/structure gate failed")
    if not result["pending"] and not result["passed"]:
        raise RuntimeError("D3 action relevance gate failed")


if __name__ == "__main__":
    main()
