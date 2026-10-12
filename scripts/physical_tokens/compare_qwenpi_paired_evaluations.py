#!/usr/bin/env python3
"""Compare enabled and hard-disabled evaluations from one adapter checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = (
    "action_mae",
    "action_rmse",
    "joint_wrapped_mae_rad",
    "joint_velocity_mae_rad",
    "endpoint_joint_mae_rad",
    "gripper_mae",
)
HIGHER_IS_BETTER = ("gripper_accuracy",)


def _metrics_from_statistics(state: dict[str, float | int]) -> dict[str, float]:
    def divide(numerator: str, denominator: str) -> float:
        count = float(state[denominator])
        return float(state[numerator]) / count if count else float("nan")

    joint_mae = divide("joint_abs_sum", "joint_count")
    return {
        "action_mae": divide("action_abs_sum", "action_count"),
        "action_rmse": math.sqrt(divide("action_squared_sum", "action_count")),
        "joint_wrapped_mae_rad": joint_mae,
        "joint_velocity_mae_rad": divide("velocity_abs_sum", "velocity_count"),
        "endpoint_joint_mae_rad": divide(
            "endpoint_joint_abs_sum", "endpoint_joint_count"
        ),
        "gripper_mae": divide("gripper_abs_sum", "gripper_count"),
        "gripper_accuracy": divide("gripper_correct", "gripper_count"),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_episode_bootstrap(
    control_episodes: dict[str, Any],
    enabled_episodes: dict[str, Any],
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if set(control_episodes) != set(enabled_episodes):
        raise ValueError("paired evaluations differ in episode_metrics keys")
    episode_ids = sorted(control_episodes, key=int)
    if not episode_ids:
        raise ValueError("episode_metrics must not be empty")
    states = {}
    episode_comparison = {}
    wins = {key: 0 for key in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER)}
    for episode_id in episode_ids:
        control_entry = control_episodes[episode_id]
        enabled_entry = enabled_episodes[episode_id]
        if control_entry["num_samples"] != enabled_entry["num_samples"]:
            raise ValueError(f"paired evaluations differ in episode {episode_id} sample count")
        control_state = control_entry["sufficient_statistics"]
        enabled_state = enabled_entry["sufficient_statistics"]
        if set(control_state) != set(enabled_state):
            raise ValueError(f"paired evaluations differ in episode {episode_id} statistics")
        states[episode_id] = (control_state, enabled_state)
        control_metrics = _metrics_from_statistics(control_state)
        enabled_metrics = _metrics_from_statistics(enabled_state)
        episode_comparison[episode_id] = {
            "num_samples": control_entry["num_samples"],
            "control": control_metrics,
            "enabled": enabled_metrics,
        }
        for key in LOWER_IS_BETTER:
            wins[key] += int(enabled_metrics[key] < control_metrics[key])
        for key in HIGHER_IS_BETTER:
            wins[key] += int(enabled_metrics[key] > control_metrics[key])

    statistic_keys = tuple(states[episode_ids[0]][0])
    distributions = {key: [] for key in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER)}
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled = rng.choices(episode_ids, k=len(episode_ids))
        merged_control = {key: 0.0 for key in statistic_keys}
        merged_enabled = {key: 0.0 for key in statistic_keys}
        for episode_id in sampled:
            control_state, enabled_state = states[episode_id]
            for key in statistic_keys:
                merged_control[key] += float(control_state[key])
                merged_enabled[key] += float(enabled_state[key])
        control_metrics = _metrics_from_statistics(merged_control)
        enabled_metrics = _metrics_from_statistics(merged_enabled)
        for key in LOWER_IS_BETTER:
            baseline = control_metrics[key]
            distributions[key].append(
                (baseline - enabled_metrics[key]) / max(abs(baseline), 1e-12)
            )
        for key in HIGHER_IS_BETTER:
            distributions[key].append(enabled_metrics[key] - control_metrics[key])

    intervals = {
        key: {
            "effect": (
                "relative_improvement" if key in LOWER_IS_BETTER else "enabled_minus_control"
            ),
            "median": _percentile(values, 0.5),
            "ci95_low": _percentile(values, 0.025),
            "ci95_high": _percentile(values, 0.975),
        }
        for key, values in distributions.items()
    }
    return {
        "method": "paired_episode_cluster_bootstrap",
        "seed": seed,
        "replicates": replicates,
        "num_episode_clusters": len(episode_ids),
        "episode_wins": wins,
        "episodes": episode_comparison,
        "intervals": intervals,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--enabled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.005)
    parser.add_argument("--allow-checkpoint-change", action="store_true")
    return parser.parse_args()


def _without_physics_toggle(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("physics_condition_enabled", None)
    return normalized


def compare_pair(
    control: dict[str, Any],
    enabled: dict[str, Any],
    *,
    relative_tolerance: float = 0.005,
    allow_checkpoint_change: bool = False,
) -> dict[str, Any]:
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    exact_keys = [
        "task_name",
        "base_checkpoint_sha256",
        "action_init_checkpoint_sha256",
        "token_cache_sha256",
        "validation_episode_ids",
        "num_samples",
        "panel_seed",
        "sample_ids",
        "num_inference_timesteps",
        "metrics_space",
    ]
    if not allow_checkpoint_change:
        exact_keys.append("checkpoint_sha256")
    for key in exact_keys:
        if control.get(key) != enabled.get(key):
            raise ValueError(f"paired evaluations differ in {key}")
    if allow_checkpoint_change:
        if control["discrete_conditioning"] != enabled["discrete_conditioning"]:
            raise ValueError("paired checkpoint evaluations differ in discrete conditioning")
        if control.get("joint_token_prediction") != enabled.get(
            "joint_token_prediction"
        ):
            raise ValueError("paired checkpoint evaluations differ in joint token config")
    else:
        if _without_physics_toggle(
            control["discrete_conditioning"]
        ) != _without_physics_toggle(enabled["discrete_conditioning"]):
            raise ValueError("paired evaluations differ outside physics_condition_enabled")
        if control["discrete_conditioning"].get("physics_condition_enabled") is not False:
            raise ValueError("control report is not hard-disabled")
        if enabled["discrete_conditioning"].get("physics_condition_enabled") is not True:
            raise ValueError("enabled report does not enable physics conditioning")

    comparisons = {}
    checks = {}
    for key in LOWER_IS_BETTER:
        base = float(control["metrics"][key])
        value = float(enabled["metrics"][key])
        relative_improvement = (base - value) / max(abs(base), 1e-12)
        comparisons[key] = {
            "control": base,
            "enabled": value,
            "delta_enabled_minus_control": value - base,
            "relative_improvement": relative_improvement,
        }
        if key in ("action_mae", "joint_wrapped_mae_rad", "endpoint_joint_mae_rad"):
            checks[f"{key}_not_worse"] = relative_improvement >= -relative_tolerance
    for key in HIGHER_IS_BETTER:
        base = float(control["metrics"][key])
        value = float(enabled["metrics"][key])
        comparisons[key] = {
            "control": base,
            "enabled": value,
            "delta_enabled_minus_control": value - base,
        }
        checks[f"{key}_not_worse"] = value + relative_tolerance >= base
    result = {
        "schema_version": 1,
        "comparison": (
            "paired_checkpoint_change"
            if allow_checkpoint_change
            else "same_checkpoint_physics_enabled_vs_hard_disabled"
        ),
        "control_checkpoint_sha256": control["checkpoint_sha256"],
        "enabled_checkpoint_sha256": enabled["checkpoint_sha256"],
        "checkpoint_sha256": control["checkpoint_sha256"],
        "relative_tolerance": relative_tolerance,
        "metrics": comparisons,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }
    control_episode_metrics = control.get("episode_metrics")
    enabled_episode_metrics = enabled.get("episode_metrics")
    if (control_episode_metrics is None) != (enabled_episode_metrics is None):
        raise ValueError("paired evaluations differ in episode_metrics availability")
    if control_episode_metrics is not None:
        result["uncertainty"] = paired_episode_bootstrap(
            control_episode_metrics, enabled_episode_metrics
        )
    return result


def main() -> None:
    args = parse_args()
    control = json.loads(args.control.read_text())
    enabled = json.loads(args.enabled.read_text())
    report = compare_pair(
        control,
        enabled,
        relative_tolerance=args.relative_tolerance,
        allow_checkpoint_change=args.allow_checkpoint_change,
    )
    report["control_report"] = str(args.control.resolve())
    report["enabled_report"] = str(args.enabled.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
