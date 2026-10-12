import copy

from scripts.physical_tokens.compare_spatiotemporal_tokenizers import compare_reports
from scripts.physical_tokens.require_d2_joint_gate import evaluate_joint_gate
from tests.test_compare_spatiotemporal_tokenizers import make_report


def make_inputs() -> tuple[dict, dict]:
    baseline = make_report(continuous=1.0, quantized=1.1)
    candidate = make_report(continuous=1.004, quantized=1.105)
    baseline["model_config"]["token_routing"] = "global_qformer"
    candidate["model_config"]["token_routing"] = "spatial_aligned"
    comparison = compare_reports(
        baseline,
        candidate,
        allowed_model_config_differences=("token_routing",),
        comparison_name="paired_same_cache_d1_vs_d2_spatial_routing",
    )
    structure = {
        "schema_version": 1,
        "kind": "spatiotemporal_physics_token_structure",
        "checkpoint": "/tmp/d2.pt",
        "metrics": {
            "within_transition_unique_ratio": 0.8,
            "within_transition_pairwise_token_change": 0.9,
        },
    }
    return comparison, structure


def test_d2_joint_gate_allows_small_reconstruction_tradeoff_with_diversity() -> None:
    comparison, structure = make_inputs()
    result = evaluate_joint_gate(comparison, structure)
    assert result["passed"] is True
    assert result["observed"]["continuous_task_non_regressions"] == 8


def test_d2_joint_gate_rejects_token_collapse() -> None:
    comparison, structure = make_inputs()
    collapsed = copy.deepcopy(structure)
    collapsed["metrics"]["within_transition_unique_ratio"] = 0.07
    collapsed["metrics"]["within_transition_pairwise_token_change"] = 0.01
    result = evaluate_joint_gate(comparison, collapsed)
    assert result["passed"] is False
    assert result["checks"]["within_transition_unique_ratio"] is False
