from pathlib import Path

import numpy as np
import pytest

from scripts.physical_tokens.export_predictable_residual_spatiotemporal_sidecar import (
    load_key_arrays,
)


def test_load_key_arrays_does_not_copy_large_feature_payload(tmp_path: Path) -> None:
    path = tmp_path / "template.npz"
    np.savez(
        path,
        task_ids=np.asarray([0]),
        task_names=np.asarray(["task"]),
        episode_ids=np.asarray([1]),
        base_indices=np.asarray([2]),
        offsets=np.arange(0, 33, 2),
        frame_features=np.zeros((4, 16, 8), dtype=np.float16),
        future_valid_mask=np.ones((1, 17), dtype=np.bool_),
    )
    arrays = load_key_arrays(path)
    assert set(arrays) == {
        "task_ids",
        "task_names",
        "episode_ids",
        "base_indices",
        "offsets",
    }


def test_load_key_arrays_requires_qualified_keys(tmp_path: Path) -> None:
    path = tmp_path / "template.npz"
    np.savez(path, offsets=np.arange(0, 33, 2))
    with pytest.raises(ValueError, match="missing arrays"):
        load_key_arrays(path)
