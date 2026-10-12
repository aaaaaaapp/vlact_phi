import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_multitask_dual_token_probe import confidence_gated_residuals


def test_confidence_gate_uses_physics_only_for_active_samples():
    action = np.zeros((2, 1, 1), dtype=np.float32)
    physics = np.asarray([[[4.0]], [[8.0]]], dtype=np.float32)
    confidence = np.asarray([0.2, 0.05], dtype=np.float32)
    result = confidence_gated_residuals(action, physics, confidence, 0.1, 0.25)
    np.testing.assert_allclose(result, np.asarray([[[1.0]], [[0.0]]]))
