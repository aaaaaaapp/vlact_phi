import torch
from torch import nn

from starVLA.model.framework.QwenPIv4JointTokens import QwenPIv4JointTokens
from starVLA.model.modules.conditioning import DiscreteTokenConditioner


def build_lightweight_framework(condition_source: str):
    model = QwenPIv4JointTokens.__new__(QwenPIv4JointTokens)
    nn.Module.__init__(model)
    model.joint_condition_source = condition_source
    model.physics_condition_enabled = True
    return model


def test_joint_condition_sources_select_each_discrete_stream_independently():
    examples = [
        {"action_token_ids": [1, 2], "physics_token_ids": [3, 4, 5]},
        {"action_token_ids": [6, 7], "physics_token_ids": [8, 9, 10]},
    ]
    action_logits = torch.zeros(2, 2, 12)
    physics_logits = torch.zeros(2, 3, 16)
    predicted_action = torch.tensor([[11, 10], [9, 8]])
    predicted_physics = torch.tensor([[15, 14, 13], [12, 11, 10]])
    action_logits.scatter_(-1, predicted_action.unsqueeze(-1), 1.0)
    physics_logits.scatter_(-1, predicted_physics.unsqueeze(-1), 1.0)
    cached_action = torch.tensor([[1, 2], [6, 7]])
    cached_physics = torch.tensor([[3, 4, 5], [8, 9, 10]])

    expected = {
        "cached": (cached_action, cached_physics),
        "predicted": (predicted_action, predicted_physics),
        "predicted_action_cached_physics": (predicted_action, cached_physics),
        "cached_action_predicted_physics": (cached_action, predicted_physics),
    }
    for source, (expected_action, expected_physics) in expected.items():
        model = build_lightweight_framework(source)
        action_ids, physics_ids = model._conditioning_token_ids(
            examples, action_logits, physics_logits
        )
        assert torch.equal(action_ids, expected_action)
        assert torch.equal(physics_ids, expected_physics)


def test_joint_condition_source_omits_physics_when_disabled():
    model = build_lightweight_framework("cached_action_predicted_physics")
    model.physics_condition_enabled = False
    action_ids, physics_ids = model._conditioning_token_ids(
        [{"action_token_ids": [1, 2]}],
        torch.zeros(1, 2, 4),
        torch.zeros(1, 3, 4),
    )
    assert torch.equal(action_ids, torch.tensor([[1, 2]]))
    assert physics_ids is None


def test_soft_one_hot_conditioning_matches_hard_token_embeddings():
    conditioner = DiscreteTokenConditioner(
        hidden_dim=8,
        action_vocab_size=4,
        physics_vocab_size=5,
        max_action_tokens=2,
        max_physics_tokens=4,
        physics_fusion_mode="pi_separate_multilayer_residual",
        physics_group_size=2,
    )
    conditioner.eval()
    action_ids = torch.tensor([[1, 3]])
    physics_ids = torch.tensor([[0, 2, 4, 1]])
    hard_action, hard_physics = conditioner.build_token_groups(
        action_ids, physics_ids
    )
    soft_action, soft_physics = conditioner.build_soft_token_groups(
        torch.nn.functional.one_hot(action_ids, 4).float(),
        torch.nn.functional.one_hot(physics_ids, 5).float(),
    )
    torch.testing.assert_close(soft_action, hard_action)
    torch.testing.assert_close(soft_physics, hard_physics)


def test_soft_conditioning_preserves_gradients_to_both_predictors():
    conditioner = DiscreteTokenConditioner(
        hidden_dim=8,
        action_vocab_size=4,
        physics_vocab_size=5,
        max_action_tokens=2,
        max_physics_tokens=4,
        physics_fusion_mode="pi_separate_multilayer_residual",
        physics_group_size=2,
    )
    action_logits = torch.randn(1, 2, 4, requires_grad=True)
    physics_logits = torch.randn(1, 4, 5, requires_grad=True)
    action_tokens, physics_tokens = conditioner.build_soft_token_groups(
        action_logits.softmax(dim=-1), physics_logits.softmax(dim=-1)
    )

    (action_tokens.square().sum() + physics_tokens.square().sum()).backward()

    assert action_logits.grad is not None
    assert physics_logits.grad is not None
    assert action_logits.grad.abs().sum() > 0
    assert physics_logits.grad.abs().sum() > 0
