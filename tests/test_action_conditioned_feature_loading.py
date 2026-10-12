import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_action_conditioned_physics_residual_probe import load_feature_index


def test_load_feature_index_skips_visual_payload(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    np.savez(
        path,
        task_ids=np.asarray([0, 0]),
        episode_ids=np.asarray([1, 2]),
        base_indices=np.asarray([3, 4]),
        future_valid_mask=np.ones((2, 17), dtype=bool),
        frame_features=np.zeros((2, 16, 8), dtype=np.float16),
    )

    loaded = load_feature_index(path)

    assert set(loaded) == {
        "task_ids",
        "episode_ids",
        "base_indices",
        "future_valid_mask",
    }


def test_load_feature_index_requires_split_mask(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    np.savez(
        path,
        task_ids=np.asarray([0]),
        episode_ids=np.asarray([1]),
        base_indices=np.asarray([2]),
    )

    with pytest.raises(ValueError, match="future_valid_mask"):
        load_feature_index(path)
