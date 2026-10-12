from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/physical_tokens/train_physical_language_tokenizer.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_split_and_statistics_are_episode_disjoint(tmp_path: Path) -> None:
    features = np.zeros((6, 4, 3), dtype=np.float16)
    features[:, 1:] = np.arange(6, dtype=np.float16)[:, None, None]
    valid = np.ones((6, 4), dtype=np.bool_)
    path = tmp_path / "features.npz"
    np.savez(
        path,
        feature_deltas=features,
        future_valid_mask=valid,
        episode_ids=np.asarray([0, 0, 1, 1, 2, 2]),
        base_indices=np.asarray([0, 1, 0, 1, 0, 1]),
        offsets=np.asarray([0, 1, 2, 3]),
    )
    dataset = MODULE["FutureFeatureCacheDataset"](path)
    training, validation = MODULE["split_indices"](dataset, (2,))
    mean, std, time_mean = MODULE["feature_statistics"](dataset, training)

    assert training == [0, 1, 2, 3]
    assert validation == [4, 5]
    assert mean.shape == std.shape == (3,)
    assert time_mean.shape == (4, 3)
    assert np.isclose(float(time_mean[0, 0]), 0.0)
