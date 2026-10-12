import pytest

from scripts.physical_tokens.require_physics_token_structure_gate import evaluate_gate


def make_report(unique_ratio: float, pairwise_change: float) -> dict:
    return {
        "schema_version": 1,
        "kind": "spatiotemporal_physics_token_structure",
        "checkpoint": "/tmp/tokenizer.pt",
        "metrics": {
            "within_transition_unique_ratio": unique_ratio,
            "within_transition_pairwise_token_change": pairwise_change,
        },
    }


def test_structure_gate_requires_both_diversity_checks() -> None:
    result = evaluate_gate(make_report(0.30, 0.25))
    assert result["passed"] is True
    assert all(result["checks"].values())

    result = evaluate_gate(make_report(0.30, 0.10))
    assert result["passed"] is False
    assert result["checks"] == {
        "within_transition_unique_ratio": True,
        "within_transition_pairwise_token_change": False,
    }


def test_structure_gate_rejects_wrong_schema() -> None:
    report = make_report(0.30, 0.25)
    report["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported schema"):
        evaluate_gate(report)
