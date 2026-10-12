import copy

import pytest

from scripts.physical_tokens.compare_qwenpi_matched_checkpoints import compare


def report(value: float, task: str = "task_a") -> dict:
    metrics = {
        "action_mae": value,
        "action_rmse": value,
        "joint_wrapped_mae_rad": value,
        "joint_wrapped_rmse_rad": value,
        "joint_velocity_mae_rad": value,
        "endpoint_joint_mae_rad": value,
        "endpoint_joint_mae_deg": value,
        "gripper_mae": value,
        "gripper_accuracy": 1.0 - value,
    }
    sufficient = {
        "action_abs_sum": value,
        "action_squared_sum": value * value,
        "action_count": 1,
        "joint_abs_sum": value,
        "joint_sq_sum": value * value,
        "joint_count": 1,
        "velocity_abs_sum": value,
        "velocity_count": 1,
        "endpoint_joint_abs_sum": value,
        "endpoint_joint_count": 1,
        "gripper_abs_sum": value,
        "gripper_correct": 1.0 - value,
        "gripper_count": 1,
    }
    return {
        "task_name": task,
        "base_checkpoint_sha256": "base",
        "checkpoint_sha256": f"checkpoint-{value}",
        "validation_episode_ids": [5],
        "num_samples": 1,
        "panel_seed": 42,
        "sample_ids": [{"episode_id": 5, "base_index": 0}],
        "num_inference_timesteps": 4,
        "metrics_space": "normalized",
        "metrics": metrics,
        "episode_metrics": {
            "5": {
                "num_samples": 1,
                "metrics": metrics,
                "sufficient_statistics": sufficient,
            }
        },
    }


def test_matched_comparison_reports_relative_action_improvement():
    result = compare(report(0.2), report(0.1), "test")
    assert result["metrics"]["action_mae"]["relative_improvement"] == pytest.approx(0.5)


def test_matched_comparison_rejects_cross_task_pair():
    baseline = report(0.2)
    candidate = copy.deepcopy(report(0.1))
    candidate["task_name"] = "task_b"
    with pytest.raises(ValueError, match="task_name"):
        compare(baseline, candidate, "test")
