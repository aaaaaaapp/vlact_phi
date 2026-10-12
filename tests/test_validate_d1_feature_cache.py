import json
from pathlib import Path

import numpy as np
import pytest

from scripts.physical_tokens.validate_d1_feature_cache import validate_cache


def _write_cache(tmp_path: Path, *, second_task_episodes=(0, 1, 2)) -> Path:
    cache = tmp_path / "features.npz"
    task_ids = np.asarray([0, 0, 0, 1, 1, 1])
    episode_ids = np.asarray([0, 1, 2, *second_task_episodes])
    np.savez(
        cache,
        task_ids=task_ids,
        task_names=np.asarray(["task_a", "task_b"]),
        episode_ids=episode_ids,
        future_valid_mask=np.ones((6, 2), dtype=np.bool_),
    )
    manifest = {
        "schema_version": 2,
        "data_names": ["task_a", "task_b"],
        "excluded_episode_ids": [5],
        "num_unique_frames": 12,
        "num_samples": 6,
        "num_episodes": 6,
        "datasets": [
            {"task_id": 0, "data_name": "task_a", "episode_ids": [0, 1, 2]},
            {
                "task_id": 1,
                "data_name": "task_b",
                "episode_ids": list(second_task_episodes),
            },
        ],
    }
    cache.with_suffix(".npz.manifest.json").write_text(json.dumps(manifest))
    return cache


def test_validate_d1_cache_reports_exact_split_counts(tmp_path: Path) -> None:
    report = validate_cache(
        _write_cache(tmp_path),
        ("task_a", "task_b"),
        (5,),
        episodes_per_task=3,
        validation_episodes_per_task=1,
        max_train_clips_per_task=2,
    )

    assert report["num_episodes"] == 6
    assert report["train_complete_clips"] == 4
    assert report["validation_complete_clips"] == 2
    assert report["capped_train_clips"] == 4


def test_validate_d1_cache_rejects_policy_episode_leakage(tmp_path: Path) -> None:
    cache = _write_cache(tmp_path, second_task_episodes=(0, 1, 5))

    with pytest.raises(ValueError, match="policy-held-out"):
        validate_cache(
            cache,
            ("task_a", "task_b"),
            (5,),
            episodes_per_task=3,
            validation_episodes_per_task=1,
            max_train_clips_per_task=2,
        )
