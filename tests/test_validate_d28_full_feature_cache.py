import json
from pathlib import Path

import numpy as np
import pytest

from scripts.physical_tokens.validate_d28_full_feature_cache import validate_cache


def _write_caches(tmp_path: Path, *, corrupt_episode: bool = False) -> tuple[Path, Path]:
    task_ids = np.repeat(np.arange(8), 50)
    episode_ids = np.tile(np.arange(50), 8)
    if corrupt_episode:
        episode_ids[-1] = 48
    base_indices = np.zeros(400, dtype=np.int64)
    offsets = np.arange(0, 33, 2)
    indices = np.zeros((400, 17), dtype=np.int64)
    names = np.asarray([f"task_{index}" for index in range(8)])
    cache = tmp_path / "features.npz"
    np.savez(
        cache,
        frame_features=np.zeros((1, 16, 4), dtype=np.float16),
        source_frame_indices=indices,
        future_valid_mask=np.ones_like(indices, dtype=np.bool_),
        task_ids=task_ids,
        task_names=names,
        episode_ids=episode_ids,
        base_indices=base_indices,
        offsets=offsets,
    )
    cache.with_suffix(".npz.manifest.json").write_text(
        json.dumps(
            {
                "kind": "robotwin_merged_multitask_visual_features",
                "exact_action_key_alignment": True,
                "num_samples": 400,
                "num_unique_frames": 1,
            }
        )
    )
    action = tmp_path / "actions.npz"
    np.savez(
        action,
        task_ids=task_ids,
        task_names=names,
        episode_ids=episode_ids,
        base_indices=base_indices,
    )
    return cache, action


def test_validate_d28_cache_reports_fixed_episode_split(tmp_path: Path) -> None:
    cache, action = _write_caches(tmp_path)
    report = validate_cache(cache, action)
    assert report["num_samples"] == 400
    assert report["train_complete_clips"] == 360
    assert report["validation_complete_clips"] == 40
    assert report["exact_action_key_alignment"] is True


def test_validate_d28_cache_rejects_missing_episode(tmp_path: Path) -> None:
    cache, action = _write_caches(tmp_path, corrupt_episode=True)
    with pytest.raises(ValueError, match="duplicate qualified keys|exact episodes 0..49"):
        validate_cache(cache, action)
