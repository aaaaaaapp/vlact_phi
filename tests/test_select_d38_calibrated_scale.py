import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from select_d38_calibrated_scale import build_report


def make_report(episodes, scales):
    return {
        "validation_episode_ids": episodes,
        "model_config": {"physics_vocab_size": 256},
        "physics_shuffle_mode": "task_and_episode_local_cyclic",
        "inputs": {
            "checkpoint_sha256": "a",
            "baseline_checkpoint_sha256": "b",
            "token_cache_sha256": "c",
            "physics_tokenizer_checkpoint_sha256": "d",
            "conditioned_decoder_checkpoint_sha256": "e",
        },
        "macro": {"action_mae_relative_change_vs_d17": 0.001},
        "global_physics_token_metrics": {"accuracy": 0.1},
        "conditioned_scale_sweep": {"scales": scales},
    }


def metrics(action_gain, shuffle_gain):
    return {
        "predicted_improvement_vs_action_only": action_gain,
        "predicted_improvement_vs_shuffled": shuffle_gain,
    }


def test_selects_maximin_scale_and_confirms_only_that_scale():
    calibration = make_report(
        [5, 20],
        {
            "0.125": metrics(0.02, 0.03),
            "0.25": metrics(0.04, 0.021),
            "0.5": metrics(0.03, 0.01),
        },
    )
    confirmation = make_report(
        [25, 39, 49],
        {
            "0.125": metrics(-0.5, -0.5),
            "0.25": metrics(0.011, 0.012),
            "0.5": metrics(0.5, 0.5),
        },
    )
    report = build_report(calibration, confirmation)
    assert report["selected_scale"] == 0.25
    assert report["independent_scale_gate_passed"] is True
    assert report["dit_authorized"] is False


def test_rejects_episode_overlap():
    calibration = make_report([5, 20], {"0.25": metrics(0.02, 0.02)})
    confirmation = make_report([20, 25], {"0.25": metrics(0.02, 0.02)})
    with pytest.raises(ValueError, match="disjoint"):
        build_report(calibration, confirmation)
