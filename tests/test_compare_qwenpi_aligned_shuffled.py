import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_qwenpi_aligned_shuffled import compare


def report(cache, value):
    statistics = {
        "action_abs_sum": value,
        "action_count": 1,
        "action_squared_sum": value * value,
        "joint_abs_sum": value,
        "joint_count": 1,
        "velocity_abs_sum": value,
        "velocity_count": 1,
        "endpoint_joint_abs_sum": value,
        "endpoint_joint_count": 1,
        "gripper_abs_sum": value,
        "gripper_count": 1,
        "gripper_correct": 1,
    }
    return {
        "checkpoint_sha256": "model",
        "base_checkpoint_sha256": "base",
        "action_init_checkpoint_sha256": "action",
        "token_cache_sha256": cache,
        "validation_episode_ids": [5],
        "num_samples": 1,
        "panel_seed": 42,
        "sample_ids": [1],
        "num_inference_timesteps": 4,
        "metrics_space": "x",
        "discrete_conditioning": {"physics_condition_enabled": True},
        "metrics": {
            "action_mae": value,
            "action_rmse": value,
            "joint_wrapped_mae_rad": value,
            "joint_velocity_mae_rad": value,
            "endpoint_joint_mae_rad": value,
            "gripper_mae": value,
            "gripper_accuracy": 1.0,
        },
        "episode_metrics": {"5": {"num_samples": 1, "sufficient_statistics": statistics}},
    }


def test_aligned_shuffled_comparison_allows_only_cache_difference():
    result = compare(report("shuffled", 2.0), report("aligned", 1.0))
    assert result["metrics"]["action_mae"]["relative_improvement"] == 0.5
