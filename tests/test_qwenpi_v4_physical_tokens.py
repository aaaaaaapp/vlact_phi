import torch
from torch import nn

from starVLA.model.framework.QwenPI_v4 import Qwen3ExpertLayer
from starVLA.model.framework.QwenPIv4PhysicalTokens import QwenPIv4PhysicalTokens
from starVLA.model.modules.conditioning import DiscreteTokenConditioner


def build_lightweight_framework():
    model = QwenPIv4PhysicalTokens.__new__(QwenPIv4PhysicalTokens)
    nn.Module.__init__(model)
    model.pi_discrete_conditioner = DiscreteTokenConditioner(
        hidden_dim=16,
        action_vocab_size=8,
        physics_vocab_size=12,
        max_action_tokens=4,
        max_physics_tokens=6,
        action_condition_dropout=0.0,
        physics_condition_dropout=0.0,
        physics_fusion_mode="concat",
    )
    model.discrete_conditioning_enabled = True
    model.physics_condition_enabled = True
    model.eval()
    return model


def examples():
    return [
        {
            "action_token_ids": [0, 1, 2, 3],
            "physics_token_ids": [0, 1, 2, 3, 4, 5],
        },
        {
            "action_token_ids": [4, 5, 6, 7],
            "physics_token_ids": [6, 7, 8, 9, 10, 11],
        },
    ]


def test_pi_prefix_conditioning_routes_action_and_physics_tokens():
    model = build_lightweight_framework()
    prefix = torch.randn(2, 5, 16)
    conditioned = model._condition_prefix_hidden(prefix, examples())
    assert conditioned.shape == (2, 15, 16)
    torch.testing.assert_close(conditioned[:, :5], prefix)

    model.set_physics_condition_enabled(False)
    action_only = model._condition_prefix_hidden(prefix, examples())
    assert action_only.shape == (2, 9, 16)
    torch.testing.assert_close(action_only[:, :5], prefix)


def test_pi_discrete_hard_disable_is_exact_identity():
    model = build_lightweight_framework()
    prefix = torch.randn(2, 5, 16)
    model.set_discrete_conditioning_enabled(False)
    output = model._condition_prefix_hidden(prefix, examples())
    assert output is prefix


def test_pi_grouped_physics_keeps_256_ids_as_16_memory_tokens():
    model = build_lightweight_framework()
    model.pi_discrete_conditioner = DiscreteTokenConditioner(
        hidden_dim=16,
        action_vocab_size=8,
        physics_vocab_size=12,
        max_action_tokens=4,
        max_physics_tokens=256,
        physics_fusion_mode="grouped_concat",
        physics_group_size=16,
    )
    grouped_examples = [
        {
            "action_token_ids": [0, 1, 2, 3],
            "physics_token_ids": [index % 12 for index in range(256)],
        }
        for _ in range(2)
    ]
    prefix = torch.randn(2, 5, 16)
    output = model._condition_prefix_hidden(prefix, grouped_examples)
    assert output.shape == (2, 5 + 4 + 16, 16)


def test_pi_multilayer_residual_splits_discrete_memory_from_base_prefix():
    model = build_lightweight_framework()
    model.pi_discrete_conditioner = DiscreteTokenConditioner(
        hidden_dim=16,
        action_vocab_size=8,
        physics_vocab_size=12,
        max_action_tokens=4,
        max_physics_tokens=256,
        physics_fusion_mode="pi_multilayer_residual",
        physics_group_size=16,
    )
    model.pi_residual_fusion_enabled = True
    model.pi_discrete_layer_gates = nn.ParameterList(
        [nn.Parameter(torch.zeros(())) for _ in range(2)]
    )
    model._pi_residual_token_count = 0
    grouped_examples = [
        {
            "action_token_ids": [0, 1, 2, 3],
            "physics_token_ids": [index % 12 for index in range(256)],
        }
        for _ in range(2)
    ]
    prefix = torch.randn(2, 5, 16)
    conditioned = model._condition_prefix_hidden(prefix, grouped_examples)
    base, residual = model._split_expert_prefix(conditioned)
    torch.testing.assert_close(base, prefix)
    assert residual.shape == (2, 20, 16)


def test_pi_separate_residual_splits_action_and_physics_memories():
    model = build_lightweight_framework()
    model.pi_discrete_conditioner = DiscreteTokenConditioner(
        hidden_dim=16,
        action_vocab_size=8,
        physics_vocab_size=12,
        max_action_tokens=4,
        max_physics_tokens=256,
        physics_fusion_mode="pi_separate_multilayer_residual",
        physics_group_size=16,
    )
    model.pi_residual_fusion_enabled = False
    model.pi_separate_residual_fusion_enabled = True
    model.pi_action_layer_gates = nn.ParameterList(
        [nn.Parameter(torch.zeros(())) for _ in range(2)]
    )
    model.pi_physics_layer_gates = nn.ParameterList(
        [nn.Parameter(torch.zeros(())) for _ in range(2)]
    )
    grouped_examples = [
        {
            "action_token_ids": [0, 1, 2, 3],
            "physics_token_ids": [index % 12 for index in range(256)],
        }
        for _ in range(2)
    ]
    prefix = torch.randn(2, 5, 16)
    conditioned = model._condition_prefix_hidden(prefix, grouped_examples)
    base, action, physics = model._split_expert_memories(conditioned)
    torch.testing.assert_close(base, prefix)
    assert action.shape == (2, 4, 16)
    assert physics.shape == (2, 16, 16)
    assert action.data_ptr() != physics.data_ptr()

    model.set_discrete_conditioning_enabled(False)
    unconditioned = model._condition_prefix_hidden(prefix, grouped_examples)
    base, action, physics = model._split_expert_memories(unconditioned)
    assert base is prefix
    assert action is None
    assert physics is None


class _IdentityAda(nn.Module):
    def forward(self, x, _condition):
        return x


class _ZeroSelfAttention(nn.Module):
    def forward(self, *, hidden_states, **_kwargs):
        return torch.zeros_like(hidden_states)


class _MeanCrossAttention(nn.Module):
    def forward(self, query, key_value):
        return key_value.mean(dim=1, keepdim=True).expand_as(query)


class _FirstCrossAttention(nn.Module):
    def forward(self, query, key_value):
        return key_value[:, :1].expand_as(query)


class _ZeroMlp(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_pi_layer_residual_gate_is_zero_impact_then_memory_sensitive():
    layer = Qwen3ExpertLayer.__new__(Qwen3ExpertLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = _IdentityAda()
    layer.self_attn = _ZeroSelfAttention()
    layer.cross_attn_norm = _IdentityAda()
    layer.cross_attn = _MeanCrossAttention()
    layer.post_attention_layernorm = _IdentityAda()
    layer.mlp = _ZeroMlp()

    x = torch.randn(2, 3, 4)
    prefix = torch.randn(2, 5, 4)
    residual_a = torch.randn(2, 7, 4)
    residual_b = residual_a + 2.0
    time = torch.zeros(2, 4)

    baseline = layer(x, prefix, time)
    zero_gated = layer(
        x,
        prefix,
        time,
        residual_prefix_hidden=residual_a,
        residual_gate=torch.tensor(0.0),
    )
    assert torch.equal(zero_gated, baseline)

    enabled_a = layer(
        x,
        prefix,
        time,
        residual_prefix_hidden=residual_a,
        residual_gate=torch.tensor(0.5),
    )
    enabled_b = layer(
        x,
        prefix,
        time,
        residual_prefix_hidden=residual_b,
        residual_gate=torch.tensor(0.5),
    )
    assert not torch.equal(enabled_a, enabled_b)


def test_pi_layer_separate_residuals_remove_action_physics_softmax_competition():
    layer = Qwen3ExpertLayer.__new__(Qwen3ExpertLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = _IdentityAda()
    layer.self_attn = _ZeroSelfAttention()
    layer.cross_attn_norm = _IdentityAda()
    layer.cross_attn = _FirstCrossAttention()
    layer.post_attention_layernorm = _IdentityAda()
    layer.mlp = _ZeroMlp()

    x = torch.randn(2, 3, 4)
    prefix = torch.randn(2, 5, 4)
    action = torch.randn(2, 4, 4)
    physics_a = torch.randn(2, 16, 4)
    physics_b = physics_a.roll(1, dims=1) + 0.25
    time = torch.zeros(2, 4)

    baseline = layer(x, prefix, time)
    both_zero = layer(
        x,
        prefix,
        time,
        residual_prefix_hidden=action,
        residual_gate=torch.tensor(0.0),
        physics_residual_prefix_hidden=physics_a,
        physics_residual_gate=torch.tensor(0.0),
    )
    assert torch.equal(both_zero, baseline)

    action_only = layer(
        x,
        prefix,
        time,
        residual_prefix_hidden=action,
        residual_gate=torch.tensor(0.5),
    )
    physics_zero_a = layer(
        x,
        prefix,
        time,
        residual_prefix_hidden=action,
        residual_gate=torch.tensor(0.5),
        physics_residual_prefix_hidden=physics_a,
        physics_residual_gate=torch.tensor(0.0),
    )
    physics_zero_b = layer(
        x,
        prefix,
        time,
        residual_prefix_hidden=action,
        residual_gate=torch.tensor(0.5),
        physics_residual_prefix_hidden=physics_b,
        physics_residual_gate=torch.tensor(0.0),
    )
    assert torch.equal(action_only, physics_zero_a)
    assert torch.equal(action_only, physics_zero_b)

    physics_enabled_a = layer(
        x,
        prefix,
        time,
        residual_prefix_hidden=action,
        residual_gate=torch.tensor(0.5),
        physics_residual_prefix_hidden=physics_a,
        physics_residual_gate=torch.tensor(0.5),
    )
    physics_enabled_b = layer(
        x,
        prefix,
        time,
        residual_prefix_hidden=action,
        residual_gate=torch.tensor(0.5),
        physics_residual_prefix_hidden=physics_b,
        physics_residual_gate=torch.tensor(0.5),
    )
    assert not torch.equal(physics_enabled_a, physics_enabled_b)
