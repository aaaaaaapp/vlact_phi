import copy

import pytest

from scripts.physical_tokens.compare_qwenpi_paired_evaluations import (
    compare_pair,
    paired_episode_bootstrap,
)


def report(enabled: bool, action_mae: float = 1.0) -> dict:
    metrics = {
        "action_mae": action_mae,
        "action_rmse": action_mae,
        "joint_wrapped_mae_rad": action_mae,
        "joint_velocity_mae_rad": action_mae,
        "endpoint_joint_mae_rad": action_mae,
        "gripper_mae": action_mae,
        "gripper_accuracy": 0.8,
    }
    return {
        "checkpoint_sha256": "checkpoint",
        "base_checkpoint_sha256": "base",
        "action_init_checkpoint_sha256": "action",
        "token_cache_sha256": "tokens",
        "validation_episode_ids": [1, 2],
        "num_samples": 2,
        "panel_seed": 42,
        "sample_ids": [{"dataset_index": 1}, {"dataset_index": 2}],
        "num_inference_timesteps": 4,
        "metrics_space": "raw",
        "discrete_conditioning": {
            "physics_fusion_mode": "dit_multilayer_cross_attention",
            "physics_condition_enabled": enabled,
        },
        "joint_token_prediction": {"condition_source": "soft", "soft_temperature": 1.0},
        "metrics": metrics,
    }


def test_same_checkpoint_pair_reports_non_degradation() -> None:
    control = report(False, action_mae=1.0)
    enabled = report(True, action_mae=0.99)
    result = compare_pair(control, enabled)
    assert result["gate"]["passed"]
    assert result["metrics"]["action_mae"]["relative_improvement"] == pytest.approx(0.01)


def test_pair_rejects_different_checkpoint() -> None:
    control = report(False)
    enabled = copy.deepcopy(report(True))
    enabled["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        compare_pair(control, enabled)


def test_checkpoint_curve_allows_only_checkpoint_to_change() -> None:
    control = report(True, action_mae=1.0)
    enabled = copy.deepcopy(report(True, action_mae=0.98))
    enabled["checkpoint_sha256"] = "different"

    result = compare_pair(control, enabled, allow_checkpoint_change=True)

    assert result["comparison"] == "paired_checkpoint_change"
    assert result["metrics"]["action_mae"]["relative_improvement"] == pytest.approx(
        0.02
    )


def test_checkpoint_curve_rejects_conditioning_change() -> None:
    control = report(True)
    enabled = copy.deepcopy(report(True))
    enabled["checkpoint_sha256"] = "different"
    enabled["joint_token_prediction"]["soft_temperature"] = 0.5

    with pytest.raises(ValueError, match="joint token config"):
        compare_pair(control, enabled, allow_checkpoint_change=True)


def _episode_entry(error: float, correct: int) -> dict:
    return {
        "num_samples": 2,
        "sufficient_statistics": {
            "action_abs_sum": error * 10,
            "action_squared_sum": error * error * 10,
            "action_count": 10,
            "joint_abs_sum": error * 8,
            "joint_count": 8,
            "velocity_abs_sum": error * 6,
            "velocity_count": 6,
            "endpoint_joint_abs_sum": error * 4,
            "endpoint_joint_count": 4,
            "gripper_abs_sum": error * 2,
            "gripper_correct": correct,
            "gripper_count": 2,
        },
    }


def test_episode_cluster_bootstrap_reports_paired_improvement() -> None:
    control = {str(index): _episode_entry(2.0 + index, 1) for index in range(3)}
    enabled = {str(index): _episode_entry(1.0 + index, 2) for index in range(3)}

    result = paired_episode_bootstrap(control, enabled, replicates=200, seed=7)

    assert result["num_episode_clusters"] == 3
    assert result["episode_wins"]["action_mae"] == 3
    assert result["intervals"]["action_mae"]["ci95_low"] > 0
    assert result["intervals"]["gripper_accuracy"]["ci95_low"] > 0


def test_pair_rejects_one_sided_episode_statistics() -> None:
    control = report(False)
    enabled = report(True)
    control["episode_metrics"] = {"1": _episode_entry(1.0, 1)}

    with pytest.raises(ValueError, match="episode_metrics availability"):
        compare_pair(control, enabled)
