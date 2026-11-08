from argparse import Namespace
import json
from pathlib import Path

import numpy as np

from scripts.physical_tokens.merge_robotwin_clean_randomized_dual_targets import (
    combine, combine_shards, join_task, write,
)
from scripts.physical_tokens import run_node002_robotwin_clean_randomized_parallel_20k as parallel
from starVLA.dataloader.lerobot_datasets import _shared_discrete_token_cache


def test_discrete_cache_is_shared_across_task_wrappers(tmp_path: Path) -> None:
    path = tmp_path / "tokens.npz"
    np.savez_compressed(path, action_token_ids=np.zeros((1, 32), dtype=np.int32),
                        task_ids=np.array([0]), task_names=np.array(["task_0"]),
                        episode_ids=np.array([0]), base_indices=np.array([0]))
    resolved = str(path.resolve())
    assert _shared_discrete_token_cache(resolved) is _shared_discrete_token_cache(resolved)


def test_incomplete_output_pair_is_recoverable(tmp_path: Path) -> None:
    path = tmp_path / "partial.npz"
    np.savez_compressed(path, value=np.array([0]))
    write(path, {"value": np.array([1])}, {"source": "test"})
    with np.load(path, allow_pickle=False) as result:
        assert result["value"].tolist() == [1]
    assert path.with_suffix(".npz.manifest.json").is_file()


def test_join_randomized_task_uses_exact_keys_and_complete_future(tmp_path: Path) -> None:
    action = tmp_path / "action.npz"
    physics = tmp_path / "physics.npz"
    feature = tmp_path / "feature.npz"
    output = tmp_path / "dual.npz"
    np.savez_compressed(action, action_token_ids=np.array([[2] * 32, [1] * 32]),
                        episode_ids=np.array([0, 0]), base_indices=np.array([4, 0]))
    np.savez_compressed(physics, physics_token_ids=np.array([[3] * 64, [4] * 64]),
                        episode_ids=np.array([0, 0]), base_indices=np.array([0, 4]),
                        task_names=np.array(["Randomized/adjust_bottle"]))
    np.savez_compressed(feature, future_valid_mask=np.array([[True, True], [True, False]]),
                        episode_ids=np.array([0, 0]), base_indices=np.array([0, 4]),
                        task_names=np.array(["Randomized/adjust_bottle"]))
    join_task(Namespace(task="adjust_bottle", action=action, physics=physics,
                        feature=feature, output=output))
    with np.load(output, allow_pickle=False) as result:
        assert result["action_token_ids"].shape == (1, 32)
        assert result["action_token_ids"][0, 0] == 1
        assert result["physics_token_ids"][0, 0] == 3
        assert result["base_indices"].tolist() == [0]


def test_combine_assigns_distinct_task_ids_to_randomized(tmp_path: Path) -> None:
    tasks = [f"task_{index}" for index in range(50)]
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("\n".join(tasks) + "\n")
    clean = tmp_path / "clean.npz"
    np.savez_compressed(clean, action_token_ids=np.zeros((50, 32), dtype=np.int32),
                        physics_token_ids=np.zeros((50, 64), dtype=np.int32),
                        task_ids=np.arange(50), task_names=np.asarray(tasks),
                        episode_ids=np.zeros(50, dtype=np.int64),
                        base_indices=np.zeros(50, dtype=np.int64))
    random_dir = tmp_path / "random"
    random_dir.mkdir()
    for task in tasks:
        np.savez_compressed(random_dir / f"{task}.npz",
                            action_token_ids=np.ones((1, 32), dtype=np.int32),
                            physics_token_ids=np.ones((1, 64), dtype=np.int32),
                            episode_ids=np.array([0]), base_indices=np.array([0]))
        (random_dir / f"{task}.npz.manifest.json").write_text(
            json.dumps({"task": f"Randomized/{task}", "samples": 1})
        )
    output = tmp_path / "combined.npz"
    combine(Namespace(clean=clean, random_dir=random_dir, task_list=task_list, output=output))
    with np.load(output, allow_pickle=False) as result:
        assert len(result["action_token_ids"]) == 100
        assert result["task_ids"].tolist() == list(range(100))
        assert result["task_names"][50] == "Randomized/task_0"


def test_episode_shards_balance_frames_and_cover_each_episode(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "Randomized"
    meta = root / "test_task/meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text("\n".join(
        json.dumps({"episode_index": index, "length": 100 + index})
        for index in range(500)
    ))
    monkeypatch.setattr(parallel, "DATA_ROOT", tmp_path)
    shards = parallel.episode_shards("test_task", 4)
    ids = [episode for episodes, _ in shards for episode in episodes]
    assert sorted(ids) == list(range(500))
    assert max(size for _, size in shards) - min(size for _, size in shards) < 500


def test_combine_shards_reorders_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for shard, episode in enumerate((7, 2)):
        path = shard_dir / f"task.shard{shard}.npz"
        np.savez_compressed(path, action_token_ids=np.full((1, 32), shard),
                            physics_token_ids=np.full((1, 64), shard),
                            episode_ids=np.array([episode]), base_indices=np.array([0]))
        path.with_suffix(".npz.manifest.json").write_text(json.dumps({
            "task": "Randomized/task", "samples": 1,
            "shard_index": shard, "shard_count": 2,
        }))
    output = tmp_path / "combined.npz"
    args = Namespace(task="task", shard_dir=shard_dir, shard_count=2, output=output)
    combine_shards(args)
    with np.load(output, allow_pickle=False) as result:
        assert result["episode_ids"].tolist() == [2, 7]
        assert result["action_token_ids"][:, 0].tolist() == [1, 0]
