import numpy as np
import pytest
import torch
import argparse
from types import SimpleNamespace

from scripts.physical_tokens import export_robotwin_spatiotemporal_features as exporter
from scripts.physical_tokens.export_robotwin_spatiotemporal_features import (
    build_source_frame_indices,
    pool_image_embedding,
    resolve_data_names,
)


def test_pool_image_embedding_restores_and_pools_grid():
    embedding = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    pooled = pool_image_embedding(
        embedding,
        torch.tensor([1, 8, 8]),
        spatial_merge_size=2,
        output_grid_size=2,
    )
    expected = embedding.reshape(4, 4, 3).permute(2, 0, 1).unsqueeze(0)
    expected = torch.nn.functional.adaptive_avg_pool2d(expected, (2, 2))
    expected = expected.squeeze(0).permute(1, 2, 0).reshape(4, 3)
    assert torch.equal(pooled, expected)


def test_pool_image_embedding_rejects_inconsistent_grid():
    with pytest.raises(ValueError, match="expected 16"):
        pool_image_embedding(
            torch.zeros(15, 3),
            torch.tensor([1, 8, 8]),
            spatial_merge_size=2,
            output_grid_size=2,
        )


def test_build_source_frame_indices_clamps_and_offsets_global_rows():
    indices, valid = build_source_frame_indices(
        np.array([0, 3]),
        (0, 2, 4),
        episode_length=5,
        global_frame_start=10,
    )
    np.testing.assert_array_equal(indices, [[10, 12, 14], [13, 14, 14]])
    np.testing.assert_array_equal(valid, [[True, True, True], [True, False, False]])


def test_resolve_data_names_supports_multi_task_and_rejects_ambiguous_input():
    assert resolve_data_names(None, "task_a, task_b") == ("task_a", "task_b")
    assert resolve_data_names(None, None) == ("click_alarmclock",)
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_data_names("task_a", "task_b")
    with pytest.raises(ValueError, match="unique"):
        resolve_data_names(None, "task_a,task_a")


def test_export_cache_keeps_task_episode_keys_disjoint(tmp_path, monkeypatch):
    class FakeTransforms:
        def eval(self):
            return None

    class FakeDataset:
        trajectory_ids = [0, 1]
        trajectory_lengths = [2, 2]
        all_steps = [(0, 0), (0, 1), (1, 0), (1, 1)]
        modality_keys = {"video": ["video.cam_high"]}
        transforms = FakeTransforms()
        delta_indices = {"video.cam_high": np.asarray([0])}

        def get_trajectory_data(self, episode_id):
            return None

        def get_video(self, episode_id, camera, base_index):
            return np.zeros((2, 8, 8, 3), dtype=np.uint8)

    for name in ("task_a", "task_b"):
        meta = tmp_path / name / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text('{"fps": 50}\n')
        (meta / "modality.json").write_text("{}\n")
    base_vlm = tmp_path / "vlm"
    base_vlm.mkdir()
    (base_vlm / "config.json").write_text("{}\n")

    monkeypatch.setattr(exporter, "make_LeRobotSingleDataset", lambda *args, **kwargs: FakeDataset())
    monkeypatch.setattr(exporter, "load_visual_encoder", lambda args: (object(), object(), None))
    monkeypatch.setattr(
        exporter,
        "encode_spatial_frames",
        lambda *args, **kwargs: np.zeros((2, 4, 3), dtype=np.float16),
    )
    args = SimpleNamespace(
        data_root=tmp_path,
        data_name=None,
        data_names="task_a,task_b",
        robot_type="robotwin_wrap32",
        camera="video.cam_high",
        offsets=(0, 1),
        spatial_grid_size=2,
        base_vlm=base_vlm,
        pretrained_checkpoint=None,
        pretrained_checkpoint_sha256=None,
        batch_size=2,
        device="cpu",
        dtype="bfloat16",
        max_episodes=0,
        exclude_episode_ids=(),
        stride=1,
    )

    manifest, arrays = exporter.export_cache(args)

    assert manifest["data_names"] == ["task_a", "task_b"]
    assert manifest["num_episodes"] == 4
    np.testing.assert_array_equal(arrays["task_ids"], [0, 0, 0, 0, 1, 1, 1, 1])
    assert arrays["source_frame_indices"].max() == 7


def test_episode_exclusion_happens_before_max_episode_cap():
    assert exporter.select_episode_ids(
        [0, 1, 2, 3, 4, 5, 6, 7, 8], {5, 20, 25, 39, 49}, 8
    ) == [0, 1, 2, 3, 4, 6, 7, 8]


def test_exact_episode_inclusion_preserves_dataset_order():
    assert exporter.select_episode_ids(
        [0, 1, 2, 3, 4], set(), 0, {1, 3}
    ) == [1, 3]
    with pytest.raises(ValueError, match="mutually exclusive"):
        exporter.select_episode_ids([0, 1], {0}, 0, {1})


def test_parse_episode_ids_accepts_nonzero_start_and_rejects_invalid_values():
    assert exporter.parse_episode_ids("5,20,25,39,49") == (5, 20, 25, 39, 49)
    for value in ("", "-1,5", "5,5", "five"):
        with pytest.raises((ValueError, argparse.ArgumentTypeError)):
            exporter.parse_episode_ids(value)
