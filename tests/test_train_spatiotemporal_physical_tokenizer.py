from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.physical_tokens.train_spatiotemporal_physical_tokenizer import (
    SpatiotemporalFeatureDataset,
    cap_indices_per_task,
    evaluate,
    shard_indices_without_padding,
    split_complete_indices,
)
from scripts.physical_tokens.evaluate_spatiotemporal_physical_tokenizer import (
    group_indices_by_task,
)


def test_multi_task_split_uses_composite_episode_keys(tmp_path: Path) -> None:
    path = tmp_path / "multi_task_features.npz"
    np.savez(
        path,
        frame_features=np.zeros((8, 2, 3), dtype=np.float16),
        source_frame_indices=np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]]),
        future_valid_mask=np.ones((4, 2), dtype=np.bool_),
        task_ids=np.asarray([0, 0, 1, 1]),
        task_names=np.asarray(["task_a", "task_b"]),
        episode_ids=np.asarray([0, 1, 0, 1]),
        base_indices=np.zeros(4, dtype=np.int64),
        offsets=np.asarray([0, 1]),
    )
    dataset = SpatiotemporalFeatureDataset(path)
    training, validation = split_complete_indices(
        dataset,
        validation_episodes=(),
        validation_episodes_per_task=1,
    )

    assert training == [0, 2]
    assert validation == [1, 3]
    assert dataset.task_names == ("task_a", "task_b")
    assert group_indices_by_task(dataset, [3, 0, 2]) == {1: [3, 2], 0: [0]}
    assert cap_indices_per_task(dataset, [0, 1, 2, 3], 1, seed=42) in (
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
    )


def test_legacy_cache_defaults_to_single_task(tmp_path: Path) -> None:
    path = tmp_path / "legacy_features.npz"
    np.savez(
        path,
        frame_features=np.zeros((4, 2, 3), dtype=np.float16),
        source_frame_indices=np.asarray([[0, 1], [2, 3]]),
        future_valid_mask=np.ones((2, 2), dtype=np.bool_),
        episode_ids=np.asarray([0, 1]),
        base_indices=np.zeros(2, dtype=np.int64),
        offsets=np.asarray([0, 1]),
    )
    dataset = SpatiotemporalFeatureDataset(path)

    assert dataset.task_names == ("single_task",)
    np.testing.assert_array_equal(dataset.task_ids, [0, 0])


def test_external_feature_array_matches_archive_samples(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    array_path = tmp_path / "frame_features.npy"
    features = np.arange(24, dtype=np.float16).reshape(4, 2, 3)
    np.save(array_path, features)
    np.savez(
        path,
        frame_features=features,
        source_frame_indices=np.asarray([[0, 1], [2, 3]]),
        future_valid_mask=np.ones((2, 2), dtype=np.bool_),
        episode_ids=np.asarray([0, 1]),
        base_indices=np.zeros(2, dtype=np.int64),
        offsets=np.asarray([0, 1]),
    )
    archived = SpatiotemporalFeatureDataset(path)
    cached = SpatiotemporalFeatureDataset(path, feature_array_cache=array_path)

    assert isinstance(cached.frame_features, np.memmap)
    for index in range(len(cached)):
        torch.testing.assert_close(cached[index]["features"], archived[index]["features"])


def test_validation_shards_cover_indices_once_without_padding() -> None:
    indices = list(range(11))
    shards = [shard_indices_without_padding(indices, rank, 4) for rank in range(4)]

    assert shards == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7]]
    assert sorted(index for shard in shards for index in shard) == indices
    with pytest.raises(ValueError):
        shard_indices_without_padding(indices, rank=4, world_size=4)


def test_evaluate_metrics_are_batch_size_invariant() -> None:
    class Features(Dataset):
        def __init__(self) -> None:
            self.features = torch.randn(5, 3, 2, 4, generator=torch.Generator().manual_seed(7))

        def __len__(self) -> int:
            return len(self.features)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "features": self.features[index],
                "valid_mask": torch.ones(3, dtype=torch.bool),
            }

    from starVLA.model.modules.tokenizers import SpatiotemporalPhysicalLanguageTokenizer

    model = SpatiotemporalPhysicalLanguageTokenizer(
        feature_dim=4,
        num_frames=3,
        spatial_tokens=2,
        queries_per_transition=2,
        hidden_dim=8,
        latent_dim=4,
        num_heads=2,
        codebook_size=8,
    ).eval()
    dataset = Features()
    metrics_two = evaluate(model, DataLoader(dataset, batch_size=2), torch.device("cpu"), 1)
    metrics_four = evaluate(model, DataLoader(dataset, batch_size=4), torch.device("cpu"), 1)

    for key in (
        "rmse",
        "continuous_rmse",
        "no_motion_rmse",
        "future_feature_rmse",
        "final_feature_rmse",
        "frozen_future_feature_rmse",
        "effective_bits_per_token",
        "perplexity",
        "active_ratio",
        "temporal_reverse_token_change",
    ):
        assert metrics_two[key] == pytest.approx(metrics_four[key], abs=1e-7)
    assert metrics_two["effective_bits_per_token"] == pytest.approx(
        np.log2(metrics_two["perplexity"]), abs=1e-6
    )
    assert metrics_two["fixed_bits_per_token"] == pytest.approx(3.0)
