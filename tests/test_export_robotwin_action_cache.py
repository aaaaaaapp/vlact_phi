from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/physical_tokens/export_robotwin_action_cache.py"
SPEC = importlib.util.spec_from_file_location("export_robotwin_action_cache", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_export_one_robotwin_episode() -> None:
    data_root = Path(
        "/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin"
    )
    if not data_root.exists():
        return
    args = SimpleNamespace(
        data_root=data_root,
        data_name="click_alarmclock",
        robot_type="robotwin_wrap32",
        max_episodes=1,
        stride=8,
    )
    manifest, arrays = MODULE.export_cache(args)
    assert arrays["actions"].shape == (10, 32, 14)
    assert arrays["action_valid_mask"].shape == (10, 32)
    assert arrays["episode_ids"].tolist() == [0] * 10
    assert arrays["base_indices"].tolist() == list(range(0, 76, 8))
    assert np.all(arrays["actions"][..., :12] >= -np.pi)
    assert np.all(arrays["actions"][..., :12] < np.pi)
    assert set(np.unique(arrays["actions"][..., 12:]).tolist()) <= {0.0, 1.0}
    assert arrays["action_valid_mask"][-1].sum() == 4
    assert manifest["gripper_dims"] == [12, 13]
    assert manifest["angular_dims"] == list(range(12))
