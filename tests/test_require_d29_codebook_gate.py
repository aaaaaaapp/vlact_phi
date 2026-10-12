from scripts.physical_tokens.require_d29_codebook_gate import evaluate_gate


def _report(size: int, *, rmse: float, continuous: float, action_aux: float) -> dict:
    return {
        "metric_schema_version": 2,
        "input": "/cache.npz",
        "task_names": ["a", "b"],
        "validation_complete_clips": 10,
        "checkpoint": f"/k{size}.pt",
        "checkpoint_sha256": str(size),
        "model_config": {"codebook_size": size},
        "macro_metrics": {
            "rmse": rmse,
            "continuous_rmse": continuous,
            "action_aux_loss": action_aux,
            "active_ratio": 0.98,
            "perplexity": 220.0 if size == 256 else 120.0,
        },
    }


def test_d29_gate_selects_k256_only_for_material_clean_gain() -> None:
    result = evaluate_gate(
        _report(128, rmse=0.10, continuous=0.09, action_aux=0.01),
        _report(256, rmse=0.098, continuous=0.0898, action_aux=0.0099),
    )
    assert result["passed"] is True
    assert result["selected_codebook_size"] == 256


def test_d29_gate_keeps_k128_when_action_aux_regresses() -> None:
    result = evaluate_gate(
        _report(128, rmse=0.10, continuous=0.09, action_aux=0.01),
        _report(256, rmse=0.098, continuous=0.0898, action_aux=0.0101),
    )
    assert result["passed"] is False
    assert result["checks"]["action_aux_loss_no_regression"] is False
    assert result["selected_codebook_size"] == 128
