import pytest
import torch
from types import SimpleNamespace

from starVLA.model.framework.QwenDualAutoregressiveReasoner import (
    aligned_dual_token_hidden,
    extract_dual_generated_ids,
    format_dual_solution,
    keep_only_dual_target_labels,
    uniformize_physics_embedding_start,
    score_separate_heads,
    QwenDualAutoregressiveReasoner,
    attach_text_lora,
)
from starVLA.model.framework.QwenPI_v4 import ConcatCrossAttentionFusion
from starVLA.model.framework.QwenPIv4DualAR32x64 import TokenStateMemory


def test_token_hidden_uses_causal_predictor_position_with_padding():
    hidden = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8, 1)
    labels = torch.tensor([
        [-100, 10, 11, 20, 21, -100, -100, -100],
        [-100, -100, 10, 11, 20, 21, -100, -100],
    ])
    action, physics = aligned_dual_token_hidden(
        hidden, labels, action_offset=10, action_vocab_size=2,
        physics_offset=20, physics_vocab_size=2,
        action_count=2, physics_count=2,
    )
    assert action[..., 0].tolist() == [[0.0, 1.0], [9.0, 10.0]]
    assert physics[..., 0].tolist() == [[2.0, 3.0], [11.0, 12.0]]


def test_token_state_memory_and_concat_fusion_trainable_path():
    table = torch.nn.Embedding(7, 8)
    memory = TokenStateMemory(vlm_dim=8, expert_dim=4, token_count=3)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    projected = memory(ids, hidden, table)
    assert projected.shape == (2, 3, 4)
    fusion = ConcatCrossAttentionFusion(hidden_dim=4, bottleneck_dim=6)
    prefix = torch.randn(2, 5, 4)
    action = projected.mean(dim=1, keepdim=True).expand(-1, 5, -1)
    physics = torch.randn_like(action)
    torch.testing.assert_close(fusion(prefix, action, physics), torch.zeros_like(prefix))
    # The zero-initialized output projection learns first; afterwards the
    # token-state bridge receives gradients from the continuous-action loss.
    with torch.no_grad():
        fusion.output_proj.weight.fill_(0.01)
    fusion(prefix, action, physics).square().mean().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert table.weight.grad is not None and table.weight.grad.abs().sum() > 0


def test_dual_solution_has_distinct_ordered_vocabularies():
    result = format_dual_solution(
        [0] * 7 + [524], [0] * 63 + [2047],
        action_vocab_size=525, physics_vocab_size=2048,
    )
    assert result.startswith("<robot_action_0>" * 7 + "<robot_action_524>")
    assert result.endswith("<robot_physics_0>" * 63 + "<robot_physics_2047>")


def test_dual_solution_rejects_shape_and_range():
    with pytest.raises(ValueError, match="expected action"):
        format_dual_solution([0] * 7, [0] * 64,
                             action_vocab_size=525, physics_vocab_size=2048)
    with pytest.raises(ValueError, match="outside"):
        format_dual_solution([0] * 8, [2048] * 64,
                             action_vocab_size=525, physics_vocab_size=2048)


def test_extract_dual_suffix_and_validate_vocab():
    ids = torch.tensor([[8, 9] + [151669] * 7 + [152193] + [153717] * 63 + [155764]])
    action, physics = extract_dual_generated_ids(
        ids, prompt_width=2, action_offset=151669, action_vocab_size=525,
        physics_offset=153717, physics_vocab_size=2048,
    )
    assert action.shape == (1, 8) and action[0, -1].item() == 524
    assert physics.shape == (1, 64) and physics[0, -1].item() == 2047
    ids[0, -1] = 151669
    with pytest.raises(ValueError, match="invalid physics"):
        extract_dual_generated_ids(
            ids, prompt_width=2, action_offset=151669, action_vocab_size=525,
            physics_offset=153717, physics_vocab_size=2048,
        )


def test_dual_loss_masks_prompt_and_assistant_terminator():
    targets = torch.tensor([[11] + [151669] * 8 + [153717] * 64 + [12]])
    result = keep_only_dual_target_labels(
        {"labels": targets}, action_offset=151669, action_vocab_size=525,
        physics_offset=153717, physics_vocab_size=2048,
    )["labels"]
    assert result.shape == targets.shape
    assert result[0, 0] == -100 and result[0, -1] == -100
    assert (result != -100).sum().item() == 72


def test_32_action_64_physics_target_and_generation_contract():
    solution = format_dual_solution(
        [0] * 31 + [524], [0] * 63 + [2047],
        action_vocab_size=525, physics_vocab_size=2048,
        action_count=32, physics_count=64,
    )
    assert solution.count("<robot_action_") == 32
    assert solution.count("<robot_physics_") == 64
    labels = torch.tensor([[11] + [151669] * 32 + [153717] * 64 + [12]])
    masked = keep_only_dual_target_labels(
        {"labels": labels}, action_offset=151669, action_vocab_size=525,
        physics_offset=153717, physics_vocab_size=2048,
        action_count=32, physics_count=64,
    )["labels"]
    assert (masked != -100).sum().item() == 96
    generated = torch.tensor([[8, 9] + [151669] * 31 + [152193] + [153717] * 63 + [155764]])
    action, physics = extract_dual_generated_ids(
        generated, prompt_width=2, action_offset=151669, action_vocab_size=525,
        physics_offset=153717, physics_vocab_size=2048,
        action_count=32, physics_count=64,
    )
    assert action.shape == (1, 32) and action[0, -1].item() == 524
    assert physics.shape == (1, 64) and physics[0, -1].item() == 2047
    with pytest.raises(ValueError, match="32 action and 64 physics"):
        keep_only_dual_target_labels(
            {"labels": labels[:, :-2]}, action_offset=151669, action_vocab_size=525,
            physics_offset=153717, physics_vocab_size=2048,
            action_count=32, physics_count=64,
        )


def test_uniform_physics_init_only_replaces_inherited_padding_rows():
    old = torch.ones((15, 16), dtype=torch.bfloat16)
    old[10:13] = 8
    old[13:] = torch.tensor([[0.01] * 16, [-0.01] * 16], dtype=torch.bfloat16)
    new = uniformize_physics_embedding_start(
        old, physics_offset=10, old_matrix_rows=13, seed=42,
    )
    assert torch.equal(new[:10], old[:10])
    assert torch.equal(new[13:], old[13:])
    assert not torch.equal(new[10:13], old[10:13])
    assert new[10:13].float().abs().max() < 0.1
    assert torch.equal(new, uniformize_physics_embedding_start(
        old, physics_offset=10, old_matrix_rows=13, seed=42,
    ))


def test_separate_heads_use_only_their_own_vocab_and_receive_gradients():
    action_head = torch.nn.Linear(3, 2, bias=False)
    physics_head = torch.nn.Linear(3, 4, bias=False)
    hidden = torch.randn(1, 4, 3, requires_grad=True)
    labels = torch.tensor([[99, 10, 23, -100]])
    scores = score_separate_heads(
        hidden, labels, action_head=action_head, physics_head=physics_head,
        action_offset=10, action_vocab_size=2, physics_offset=20, physics_vocab_size=4,
    )
    assert scores["action_logits"].shape == (1, 2)
    assert scores["physics_logits"].shape == (1, 4)
    assert scores["action_targets"].tolist() == [0]
    assert scores["physics_targets"].tolist() == [3]
    (scores["action_ce"] + scores["physics_ce"]).backward()
    assert action_head.weight.grad is not None
    assert physics_head.weight.grad is not None
    assert hidden.grad is not None


def test_separate_physics_head_scores_full_phizero_fsq_vocab():
    action_head = torch.nn.Linear(3, 525, bias=False)
    physics_head = torch.nn.Linear(3, 25000, bias=False)
    hidden = torch.randn(1, 3, 3)
    labels = torch.tensor([[99, 10, 153717 + 24999]])
    scores = score_separate_heads(
        hidden, labels, action_head=action_head, physics_head=physics_head,
        action_offset=10, action_vocab_size=525,
        physics_offset=153717, physics_vocab_size=25000,
    )
    assert scores["physics_logits"].shape == (1, 25000)
    assert scores["physics_targets"].tolist() == [24999]


def test_separate_head_generation_feeds_ids_back_autoregressively():
    class FakeBackbone:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                last_hidden_state=torch.ones((1, 1, 3)),
                past_key_values=object(),
            )

    backbone = FakeBackbone()
    reasoner = SimpleNamespace(
        qwen_vl_interface=SimpleNamespace(model=SimpleNamespace(model=backbone)),
        action_head=torch.nn.Linear(3, 2, bias=False),
        physics_head=torch.nn.Linear(3, 4, bias=False),
        action_token_count=2, physics_token_count=3,
        action_token_offset=10, physics_token_offset=20,
        strict_modality_tying=False,
    )
    reasoner.action_logits = reasoner.action_head
    reasoner.physics_logits = reasoner.physics_head
    with torch.no_grad():
        reasoner.action_head.weight.zero_()
        reasoner.physics_head.weight.zero_()
    result = QwenDualAutoregressiveReasoner._predict_with_separate_heads(
        reasoner, {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)},
    )
    assert result["action_token_ids"].tolist() == [[0, 0]]
    assert result["physics_token_ids"].tolist() == [[0, 0, 0]]
    assert [call["input_ids"].tolist() for call in backbone.calls] == [
        [[1, 2]], [[10]], [[10]], [[20]], [[20]],
    ]
    assert [call["cache_position"].tolist() for call in backbone.calls] == [
        [0, 1], [2], [3], [4], [5],
    ]


def test_strict_modality_tables_are_input_output_tied_and_text_frozen():
    text = torch.nn.Embedding(40, 3)
    text.weight.requires_grad_(False)
    action = torch.nn.Embedding(2, 3)
    physics = torch.nn.Embedding(4, 3)
    reasoner = SimpleNamespace(
        qwen_vl_interface=SimpleNamespace(model=SimpleNamespace(get_input_embeddings=lambda: text)),
        action_embeddings=action, physics_embeddings=physics,
        action_token_offset=10, action_vocab_size=2,
        physics_token_offset=20, physics_vocab_size=4,
        strict_modality_tying=True,
    )
    ids = torch.tensor([[1, 10, 21]])
    routed = QwenDualAutoregressiveReasoner._routed_input_embeddings(reasoner, ids)
    assert torch.equal(routed[0, 0], text.weight[1])
    assert torch.equal(routed[0, 1], action.weight[0])
    assert torch.equal(routed[0, 2], physics.weight[1])
    action_logits = torch.nn.functional.linear(routed[0, 1], action.weight)
    physics_logits = torch.nn.functional.linear(routed[0, 2], physics.weight)
    (routed.sum() + action_logits.sum() + physics_logits.sum()).backward()
    assert text.weight.grad is None
    assert action.weight.grad is not None and action.weight.grad.abs().sum() > 0
    assert physics.weight.grad is not None and physics.weight.grad.abs().sum() > 0


def test_strict_backbone_retains_original_ids_for_multimodal_rope():
    class FakeBackbone:
        def __init__(self):
            self.calls = []
            self.rope_ids = None

        def get_rope_index(self, ids, image_grid, video_grid, attention_mask):
            self.rope_ids = ids.clone()
            return torch.zeros((3, 1, ids.shape[1]), dtype=torch.long), torch.tensor([[5]])

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"], past_key_values=object())

    backbone = FakeBackbone()
    text = torch.nn.Embedding(40, 3)
    reasoner = SimpleNamespace(
        qwen_vl_interface=SimpleNamespace(model=SimpleNamespace(
            model=backbone, get_input_embeddings=lambda: text,
        )),
        action_embeddings=torch.nn.Embedding(2, 3),
        physics_embeddings=torch.nn.Embedding(4, 3),
        action_token_offset=10, action_vocab_size=2,
        physics_token_offset=20, physics_vocab_size=4,
    )
    reasoner._routed_input_embeddings = lambda ids: QwenDualAutoregressiveReasoner._routed_input_embeddings(reasoner, ids)
    output, delta = QwenDualAutoregressiveReasoner._strict_backbone_forward(
        reasoner,
        {"input_ids": torch.tensor([[1, 10]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)},
        cache_position=torch.arange(2), use_cache=True,
    )
    assert backbone.rope_ids.tolist() == [[1, 10]]
    assert "input_ids" not in backbone.calls[0]
    assert torch.equal(output.last_hidden_state[0, 1], reasoner.action_embeddings.weight[0])
    QwenDualAutoregressiveReasoner._strict_backbone_forward(
        reasoner,
        {"input_ids": torch.tensor([[20]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)},
        past_key_values=output.past_key_values, cache_position=torch.tensor([2]),
        rope_deltas=delta, use_cache=True,
    )
    assert backbone.calls[1]["position_ids"].shape == (3, 1, 1)
    assert backbone.calls[1]["position_ids"][0, 0, 0].item() == 7


def test_native_lora_hook_preserves_base_weights_and_trains_only_delta():
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_proj = torch.nn.Linear(3, 4, bias=False)
    language_model = torch.nn.Module()
    language_model.layers = torch.nn.ModuleList([layer])
    for parameter in language_model.parameters():
        parameter.requires_grad_(False)
    old_keys = set(language_model.state_dict())
    x = torch.randn(2, 3)
    baseline = layer.self_attn.q_proj(x).detach().clone()
    adapters, handles = attach_text_lora(
        language_model, rank=2, alpha=4, dropout=0,
        target_names=("self_attn.q_proj",),
    )
    try:
        assert set(language_model.state_dict()) == old_keys
        assert len(adapters) == 1 and len(handles) == 1
        output = layer.self_attn.q_proj(x)
        assert torch.allclose(output, baseline)
        output.sum().backward()
        assert layer.self_attn.q_proj.weight.grad is None
        assert adapters[0].up.weight.grad is not None
        assert adapters[0].up.weight.grad.abs().sum() > 0
    finally:
        for handle in handles:
            handle.remove()
