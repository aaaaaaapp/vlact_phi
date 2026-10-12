from collections import deque

import numpy as np

from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient


class _FakeClient:
    def __init__(self, actions):
        self.actions = actions
        self.request = None

    def predict_action(self, request):
        self.request = request
        return {"data": {"actions": self.actions}}


def _server_preprocessed_client(actions):
    model = ModelClient.__new__(ModelClient)
    model.client = _FakeClient(actions)
    model.policy_setup = "robotwin"
    model.unnorm_key = "new_embodiment"
    model.action_mode = "abs"
    model.action_postprocess = "none"
    model.robot_tag = None
    model.use_ddim = True
    model.num_ddim_steps = 10
    model.image_size = [4, 4]
    model.image_size_buckets = []
    model.action_chunk_size = 2
    model.action_ensemble = False
    model.image_history = deque(maxlen=0)
    model.num_image_history = 0
    model.initial_state = None
    model.prev_action = None
    model.task_description = None
    model.raw_actions = None
    model._send_state_to_server = True
    model._server_preprocesses_robotwin = True
    return model


def test_robotwin_compat_server_owns_normalization_and_returns_raw_actions():
    model_order_actions = np.arange(28, dtype=np.float32).reshape(1, 2, 14)
    model = _server_preprocessed_client(model_order_actions)
    runtime_state = np.arange(14, dtype=np.float32) + 100.0
    example = {
        "lang": "pick the bottles",
        "image": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)],
        "state": runtime_state.copy(),
    }

    action = model.step(example, step=0)

    request = model.client.request
    assert request["unnorm_key"] == "new_embodiment"
    np.testing.assert_array_equal(
        request["examples"][0]["state"],
        runtime_state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]],
    )
    np.testing.assert_array_equal(
        action,
        model_order_actions[0, 0, [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]],
    )
