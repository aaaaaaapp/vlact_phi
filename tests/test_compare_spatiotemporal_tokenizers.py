import copy

import pytest

from scripts.physical_tokens.compare_spatiotemporal_tokenizers import compare_reports


def make_report(continuous: float, quantized: float) -> dict:
    task_names = [f"task_{index}" for index in range(8)]
    metrics = {
        "continuous_rmse": continuous,
        "rmse": quantized,
        "quantization_rmse_gap": quantized - continuous,
        "no_motion_rmse": 1.2,
        "continuous_relative_rmse_reduction": 0.08,
        "quantized_relative_rmse_reduction": 0.06,
    }
    return {
        "metric_schema_version": 2,
        "input": "/same/cache.npz",
        "task_names": task_names,
        "validation_complete_clips": 160,
        "checkpoint_sha256": f"checkpoint-{continuous}-{quantized}",
        "model_config": {"queries_per_transition": 16, "codebook_size": 128},
        "metrics": dict(metrics),
        "macro_metrics": metrics,
        "task_metrics": {
            name: {"validation_complete_clips": 20, "metrics": dict(metrics)}
            for name in task_names
        },
    }


def test_paired_comparison_passes_preregistered_gate() -> None:
    baseline = make_report(continuous=1.0, quantized=1.1)
    candidate = make_report(continuous=0.95, quantized=1.05)
    report = compare_reports(baseline, candidate)
    assert report["gate"]["passed"]
    assert report["task_wins"]["continuous_rmse"] == 8
    assert report["macro"]["continuous_rmse"]["relative_improvement"] == pytest.approx(0.05)


def test_paired_comparison_rejects_different_evaluation_cache() -> None:
    baseline = make_report(continuous=1.0, quantized=1.1)
    candidate = copy.deepcopy(baseline)
    candidate["input"] = "/different/cache.npz"
    with pytest.raises(ValueError, match="input"):
        compare_reports(baseline, candidate)


def test_paired_comparison_rejects_changed_no_motion_baseline() -> None:
    baseline = make_report(continuous=1.0, quantized=1.1)
    candidate = make_report(continuous=0.95, quantized=1.05)
    candidate["task_metrics"]["task_3"]["metrics"]["no_motion_rmse"] = 1.3

    with pytest.raises(ValueError, match="task_3"):
        compare_reports(baseline, candidate)


def test_paired_comparison_rejects_different_tokenizer_architecture() -> None:
    baseline = make_report(continuous=1.0, quantized=1.1)
    candidate = make_report(continuous=0.95, quantized=1.05)
    candidate["model_config"]["codebook_size"] = 256

    with pytest.raises(ValueError, match="model_config"):
        compare_reports(baseline, candidate)


def test_comparison_allows_preregistered_single_config_ablation() -> None:
    baseline = make_report(continuous=1.0, quantized=1.1)
    candidate = make_report(continuous=0.95, quantized=1.05)
    baseline["model_config"]["token_routing"] = "global_qformer"
    candidate["model_config"]["token_routing"] = "spatial_aligned"

    report = compare_reports(
        baseline, candidate, allowed_model_config_differences=("token_routing",)
    )

    assert report["allowed_model_config_differences"] == ["token_routing"]
    assert report["candidate_model_config"]["token_routing"] == "spatial_aligned"
