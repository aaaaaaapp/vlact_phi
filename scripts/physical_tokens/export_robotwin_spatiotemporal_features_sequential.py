#!/usr/bin/env python3
"""Use a sequential full-video decode for the all-frames feature exporter."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.video import get_all_frames
from export_robotwin_spatiotemporal_features import main


_timestamp_get_video = LeRobotSingleDataset.get_video


def _get_video_sequential(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
    """Decode all frames once when the caller requests a full v2 episode."""
    if self._lerobot_version != "v2.0" or self._video_uses_image_backend(key):
        return _timestamp_get_video(self, trajectory_id, key, base_index)

    trajectory_index = self.get_trajectory_index(trajectory_id)
    length = int(self.trajectory_lengths[trajectory_index])
    step_indices = np.asarray(self.delta_indices[key], dtype=np.int64) + int(base_index)
    step_indices = np.clip(step_indices, 0, length - 1)

    # Sequential frame indices and timestamps coincide in the 15-FPS v2
    # RobotWin recordings. Fall back to the generic reader if an episode is
    # not frame-aligned, preserving its timestamp-seeking behavior.
    if self.curr_traj_data is None or "timestamp" not in self.curr_traj_data.columns:
        return _timestamp_get_video(self, trajectory_id, key, base_index)
    timestamps = self.curr_traj_data["timestamp"].to_numpy(dtype=np.float64)
    fps = float(self._lerobot_info_meta.get("fps", 0))
    if fps <= 0 or not np.allclose(
        timestamps[step_indices] * fps, step_indices, rtol=0.0, atol=1e-3
    ):
        return _timestamp_get_video(self, trajectory_id, key, base_index)

    video_key = key.replace("video.", "", 1)
    frames = get_all_frames(
        self.get_video_path(trajectory_id, video_key).as_posix(),
        video_backend="pyav",
    )
    if len(frames) != length:
        raise ValueError(
            f"{self.dataset_name} episode {trajectory_id}: video has {len(frames)} frames, "
            f"metadata has {length} steps"
        )
    return frames[step_indices]


LeRobotSingleDataset.get_video = _get_video_sequential


if __name__ == "__main__":
    main()
