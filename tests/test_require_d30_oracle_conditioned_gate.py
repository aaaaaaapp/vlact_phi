from scripts.physical_tokens.require_d30_oracle_conditioned_gate import evaluate_gate


def _report(action: float, shuffled: float) -> dict:
    return {
        "kind": "action_conditioned_oracle_physics_residual_probe",
        "final": {
            "aligned_relative_improvement": {
                "action_only": {"joint_mae": action},
                "episode_shuffled": {"joint_mae": shuffled},
            }
        },
    }


def test_d30_oracle_gate_requires_both_controls() -> None:
    assert evaluate_gate(_report(0.02, 0.011))["passed"] is True
    result = evaluate_gate(_report(0.02, 0.009))
    assert result["passed"] is False
    assert result["checks"]["oracle_improves_episode_local_shuffle_by_1pct"] is False
