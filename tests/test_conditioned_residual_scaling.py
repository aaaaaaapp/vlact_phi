import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_multitask_dual_token_probe import blend_residuals


def test_blend_residuals_has_exact_action_and_physics_endpoints() -> None:
    action = np.asarray([[[1.0, 2.0]]], dtype=np.float32)
    physics = np.asarray([[[3.0, -2.0]]], dtype=np.float32)

    np.testing.assert_array_equal(blend_residuals(action, physics, 0.0), action)
    np.testing.assert_array_equal(blend_residuals(action, physics, 1.0), physics)
    np.testing.assert_allclose(
        blend_residuals(action, physics, 0.25), np.asarray([[[1.5, 1.0]]])
    )


def test_blend_residuals_rejects_invalid_scale_and_shape() -> None:
    values = np.zeros((1, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="\[0, 1\]"):
        blend_residuals(values, values, 1.1)
    with pytest.raises(ValueError, match="shapes"):
        blend_residuals(values, values[:, :1], 0.5)
