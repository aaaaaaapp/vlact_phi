#!/usr/bin/env python3
"""Select K128 or K256 after the D29 full-diversity tokenizer sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1e-12)


def evaluate_gate(k128: dict, k256: dict) -> dict:
    for name, report, size in (("K128", k128, 128), ("K256", k256, 256)):
        if report.get("metric_schema_version") != 2:
            raise ValueError(f"{name} report must use metric schema v2")
        if int(report["model_config"]["codebook_size"]) != size:
            raise ValueError(f"{name} report has the wrong codebook size")
    for key in ("input", "task_names", "validation_complete_clips"):
        if k128.get(key) != k256.get(key):
            raise ValueError(f"K128 and K256 reports differ in {key}")

    base = k128["macro_metrics"]
    candidate = k256["macro_metrics"]
    quantized_improvement = relative_improvement(float(base["rmse"]), float(candidate["rmse"]))
    continuous_improvement = relative_improvement(
        float(base["continuous_rmse"]), float(candidate["continuous_rmse"])
    )
    action_aux_improvement = relative_improvement(
        float(base["action_aux_loss"]), float(candidate["action_aux_loss"])
    )
    thresholds = {
        "quantized_rmse_relative_improvement": 0.01,
        "continuous_rmse_relative_improvement_floor": -0.005,
        "action_aux_loss_relative_improvement_floor": 0.0,
        "active_ratio_floor": 0.9,
        "perplexity_floor": 128.0,
    }
    checks = {
        "quantized_rmse_improves_by_1pct": quantized_improvement >= 0.01,
        "continuous_rmse_no_material_regression": continuous_improvement >= -0.005,
        "action_aux_loss_no_regression": action_aux_improvement >= 0.0,
        "healthy_active_ratio": float(candidate["active_ratio"]) >= 0.9,
        "healthy_perplexity": float(candidate["perplexity"]) >= 128.0,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "gate": "d29_full_diversity_codebook256",
        "thresholds": thresholds,
        "metrics": {
            "k128_macro": base,
            "k256_macro": candidate,
            "quantized_rmse_relative_improvement": quantized_improvement,
            "continuous_rmse_relative_improvement": continuous_improvement,
            "action_aux_loss_relative_improvement": action_aux_improvement,
        },
        "checks": checks,
        "passed": passed,
        "selected_codebook_size": 256 if passed else 128,
        "selected_checkpoint": k256["checkpoint"] if passed else k128["checkpoint"],
        "selected_checkpoint_sha256": (
            k256["checkpoint_sha256"] if passed else k128["checkpoint_sha256"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k128", type=Path, required=True)
    parser.add_argument("--k256", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_gate(json.loads(args.k128.read_text()), json.loads(args.k256.read_text()))
    report["k128_report"] = str(args.k128.resolve())
    report["k256_report"] = str(args.k256.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
