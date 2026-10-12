from scripts.physical_tokens.require_d31_predicted_physics_gate import evaluate_gate


def _report(*, action: float = 0.005, accuracy: float = 0.02, shuffled: float = 0.011) -> dict:
    return {
        "physics_shuffle_mode": "task_and_episode_local_cyclic",
        "model_config": {"physics_vocab_size": 128},
        "global_physics_token_metrics": {"accuracy": accuracy},
        "macro": {"action_mae_relative_change_vs_d17": action},
        "conditioned_macro": {
            "predicted_improvement_vs_action_only": 0.012,
            "predicted_improvement_vs_shuffled": shuffled,
        },
    }


def test_d31_gate_passes_only_all_four_requirements() -> None:
    assert evaluate_gate(_report())["passed"] is True
    assert evaluate_gate(_report(action=0.02))["passed"] is False
    assert evaluate_gate(_report(accuracy=0.007))["passed"] is False
    assert evaluate_gate(_report(shuffled=0.009))["passed"] is False


def test_d31_gate_rejects_wrong_shuffle_scope() -> None:
    report = _report()
    report["physics_shuffle_mode"] = "task_local"
    try:
        evaluate_gate(report)
    except ValueError as error:
        assert "task-and-episode-local" in str(error)
    else:
        raise AssertionError("D31 accepted the wrong shuffle scope")
