from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/physical_tokens/export_robotwin_future_features.py"
SPEC = importlib.util.spec_from_file_location("export_robotwin_future_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_future_feature_targets_clamps_and_masks_episode_end() -> None:
    features = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    deltas, valid, indices = MODULE.build_future_feature_targets(
        features,
        np.asarray([0, 4, 5]),
        (0, 2, 5),
    )

    assert deltas.shape == (3, 3, 3)
    assert np.all(deltas[:, 0] == 0)
    assert indices.tolist() == [[0, 2, 5], [4, 5, 5], [5, 5, 5]]
    assert valid.tolist() == [[True, True, True], [True, False, False], [True, False, False]]
    assert np.array_equal(deltas[0, 1], features[2] - features[0])
    assert np.array_equal(deltas[1, 1], features[5] - features[4])


def test_parse_offsets_requires_current_frame_and_increasing_values() -> None:
    assert MODULE.parse_ints("0,8,16,31") == (0, 8, 16, 31)
    for value in ("8,16", "0,8,8", "0,16,8", "0,-1"):
        try:
            MODULE.parse_ints(value)
        except Exception:
            pass
        else:
            raise AssertionError(f"expected invalid offsets: {value}")
