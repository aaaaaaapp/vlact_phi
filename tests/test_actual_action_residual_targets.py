import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_actual_action_residual_targets import actual_residual_targets


def test_actual_residual_wraps_joints_but_not_grippers() -> None:
    decoded = np.zeros((1, 2, 14), dtype=np.float32)
    actions = decoded.copy()
    actions[0, 0, 0] = 2.0 * np.pi - 0.2
    actions[0, 1, 12:] = (1.0, -1.0)

    residual = actual_residual_targets(actions, decoded)

    assert residual[0, 0, 0] == pytest.approx(-0.2, abs=1e-6)
    np.testing.assert_allclose(residual[0, 1, 12:], (1.0, -1.0))


def test_actual_residual_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal"):
        actual_residual_targets(
            np.zeros((2, 3, 14), dtype=np.float32),
            np.zeros((2, 2, 14), dtype=np.float32),
        )
