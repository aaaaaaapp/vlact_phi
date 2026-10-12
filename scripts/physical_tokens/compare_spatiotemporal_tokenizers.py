#!/usr/bin/env python3
"""Compare two tokenizer checkpoints on the exact same validation cache."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = (
    "continuous_rmse",
    "rmse",
    "quantization_rmse_gap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allowed-model-config-differences",
        default="",
        help="Comma-separated model_config keys intentionally varied by an ablation.",
    )
    parser.add_argument("--comparison-name", default="paired_same_cache_d0_vs_d1")
    return parser.parse_args()


def relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1e-12)


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    allowed_model_config_differences: tuple[str, ...] = (),
    comparison_name: str = "paired_same_cache_d0_vs_d1",
) -> dict[str, Any]:
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("metric_schema_version") != 2:
            raise ValueError(f"{name} report must use metric schema v2")
    for key in ("input", "task_names", "validation_complete_clips"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"baseline and candidate differ in {key}")
    baseline_config = dict(baseline.get("model_config", {}))
    candidate_config = dict(candidate.get("model_config", {}))
    for key in allowed_model_config_differences:
        baseline_config.pop(key, None)
        candidate_config.pop(key, None)
    if baseline_config != candidate_config:
        raise ValueError("baseline and candidate tokenizer model_config differs")
    if set(baseline["task_metrics"]) != set(candidate["task_metrics"]):
        raise ValueError("baseline and candidate task sets differ")
    for scope, base_value, candidate_value in (
        (
            "aggregate",
            baseline["metrics"]["no_motion_rmse"],
            candidate["metrics"]["no_motion_rmse"],
        ),
        (
            "macro",
            baseline["macro_metrics"]["no_motion_rmse"],
            candidate["macro_metrics"]["no_motion_rmse"],
        ),
    ):
        if not math.isclose(
            float(base_value), float(candidate_value), rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(f"baseline and candidate differ in {scope} no-motion RMSE")

    macro = {}
    for key in LOWER_IS_BETTER:
        base_value = float(baseline["macro_metrics"][key])
        candidate_value = float(candidate["macro_metrics"][key])
        macro[key] = {
            "baseline": base_value,
            "candidate": candidate_value,
            "delta_candidate_minus_baseline": candidate_value - base_value,
            "relative_improvement": relative_improvement(base_value, candidate_value),
        }

    task_comparison = {}
    wins = {key: 0 for key in LOWER_IS_BETTER}
    for task_name in baseline["task_names"]:
        base_entry = baseline["task_metrics"][task_name]
        candidate_entry = candidate["task_metrics"][task_name]
        if base_entry["validation_complete_clips"] != candidate_entry["validation_complete_clips"]:
            raise ValueError(f"clip count differs for task {task_name}")
        if not math.isclose(
            float(base_entry["metrics"]["no_motion_rmse"]),
            float(candidate_entry["metrics"]["no_motion_rmse"]),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"no-motion RMSE differs for task {task_name}")
        metrics = {}
        for key in LOWER_IS_BETTER:
            base_value = float(base_entry["metrics"][key])
            candidate_value = float(candidate_entry["metrics"][key])
            improvement = relative_improvement(base_value, candidate_value)
            metrics[key] = {
                "baseline": base_value,
                "candidate": candidate_value,
                "relative_improvement": improvement,
            }
            wins[key] += int(candidate_value < base_value)
        task_comparison[task_name] = {
            "validation_complete_clips": base_entry["validation_complete_clips"],
            "metrics": metrics,
        }

    num_tasks = len(task_comparison)
    thresholds = {
        "continuous_macro_relative_improvement": 0.02,
        "quantized_macro_relative_improvement": 0.01,
        "continuous_task_wins": max(1, (3 * num_tasks + 3) // 4),
        "quantized_task_wins": max(1, (3 * num_tasks + 3) // 4),
        "candidate_continuous_no_motion_reduction": 0.05,
        "candidate_quantized_no_motion_reduction": 0.03,
    }
    checks = {
        "continuous_macro_relative_improvement": (
            macro["continuous_rmse"]["relative_improvement"]
            >= thresholds["continuous_macro_relative_improvement"]
        ),
        "quantized_macro_relative_improvement": (
            macro["rmse"]["relative_improvement"]
            >= thresholds["quantized_macro_relative_improvement"]
        ),
        "continuous_task_wins": (
            wins["continuous_rmse"] >= thresholds["continuous_task_wins"]
        ),
        "quantized_task_wins": wins["rmse"] >= thresholds["quantized_task_wins"],
        "candidate_continuous_no_motion_reduction": (
            float(candidate["macro_metrics"]["continuous_relative_rmse_reduction"])
            >= thresholds["candidate_continuous_no_motion_reduction"]
        ),
        "candidate_quantized_no_motion_reduction": (
            float(candidate["macro_metrics"]["quantized_relative_rmse_reduction"])
            >= thresholds["candidate_quantized_no_motion_reduction"]
        ),
    }
    return {
        "schema_version": 1,
        "comparison": comparison_name,
        "baseline_checkpoint_sha256": baseline.get("checkpoint_sha256"),
        "candidate_checkpoint_sha256": candidate.get("checkpoint_sha256"),
        "model_config": baseline.get("model_config"),
        "candidate_model_config": candidate.get("model_config"),
        "allowed_model_config_differences": list(allowed_model_config_differences),
        "num_tasks": num_tasks,
        "macro": macro,
        "task_wins": wins,
        "tasks": task_comparison,
        "gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    allowed_differences = tuple(
        value.strip()
        for value in args.allowed_model_config_differences.split(",")
        if value.strip()
    )
    report = compare_reports(
        baseline,
        candidate,
        allowed_differences,
        comparison_name=args.comparison_name,
    )
    report["baseline_report"] = str(args.baseline.resolve())
    report["candidate_report"] = str(args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
