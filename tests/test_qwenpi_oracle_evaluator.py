import math

import numpy as np

from scripts.physical_tokens.evaluate_qwenpi_oracle import (
    MetricAccumulator,
    select_panel,
    wrapped_difference,
)


def test_wrapped_difference_uses_shortest_angular_distance():
    prediction = np.asarray([[math.pi - 0.1]])
    target = np.asarray([[-math.pi + 0.1]])
    assert np.allclose(wrapped_difference(prediction, target), -0.2)


def test_metrics_respect_time_mask_and_report_physical_errors():
    target = np.zeros((3, 14), dtype=np.float64)
    prediction = target.copy()
    prediction[0, 0] = 0.5
    prediction[1, 0] = 1.0
    prediction[2, 0] = 1.0
    prediction[0, 12] = 0.6
    accumulator = MetricAccumulator()
    accumulator.update(prediction, target, time_mask=np.asarray([1, 0, 0]))
    result = accumulator.compute()
    assert np.isclose(result["action_mae"], 1.1 / 14.0)
    assert np.isclose(result["joint_wrapped_mae_rad"], 0.5 / 12.0)
    assert np.isclose(result["endpoint_joint_mae_rad"], 0.5 / 12.0)
    assert np.isclose(result["gripper_accuracy"], 0.5)


def test_panel_selection_is_fixed_and_records_source_keys():
    class Dataset:
        all_steps = [(episode, episode * 10) for episode in range(10)]

        def __len__(self):
            return len(self.all_steps)

    first = select_panel(Dataset(), 4, 42)
    second = select_panel(Dataset(), 4, 42)
    assert first == second
    assert len({item["dataset_index"] for item in first}) == 4
    assert all(item["base_index"] == item["episode_id"] * 10 for item in first)
