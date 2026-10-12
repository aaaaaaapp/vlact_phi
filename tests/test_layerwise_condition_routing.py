import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    layerwise_encoder_hidden,
)
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


def test_interleaved_layers_route_only_even_layers_to_vlm_prefix():
    prefixes = [torch.randn(1, 3, 5) for _ in range(4)]
    assert layerwise_encoder_hidden(prefixes, 0, True) is prefixes[0]
    assert layerwise_encoder_hidden(prefixes, 1, True) is None
    assert layerwise_encoder_hidden(prefixes, 2, True) is prefixes[2]
    assert layerwise_encoder_hidden(prefixes, 3, True) is None


def test_non_interleaved_layers_all_receive_vlm_prefix():
    prefixes = [torch.randn(1, 3, 5) for _ in range(3)]
    assert all(
        layerwise_encoder_hidden(prefixes, index, False) is prefixes[index]
        for index in range(3)
    )


def make_small_physics_head(
    *,
    freeze_base: bool = False,
    usefulness_weight: float = 0.0,
    action_tokens_only: bool = False,
    injection_layers: str = "all",
) -> LayerwiseFlowmatchingActionHead:
    config = OmegaConf.create(
        {
            "framework": {
                "discrete_conditioning": {
                    "physics_fusion_mode": "dit_multilayer_cross_attention",
                    "physics_vocab_size": 32,
                    "max_physics_tokens": 256,
                    "physics_fusion_heads": 4,
                    "physics_condition_dropout": 0.0,
                    "physics_condition_enabled": True,
                    "dit_injection_layers": injection_layers,
                    "dit_freeze_base_action_model": freeze_base,
                    "physics_usefulness_weight": usefulness_weight,
                    "physics_usefulness_margin": 0.01,
                    "dit_action_tokens_only": action_tokens_only,
                },
                "action_model": {
                    "action_dim": 3,
                    "action_horizon": 4,
                    "state_dim": 0,
                    "num_target_vision_tokens": 2,
                    "add_pos_embed": False,
                    "max_seq_len": 32,
                    "num_inference_timesteps": 2,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 100,
                    "diffusion_model_cfg": {
                        "num_layers": 2,
                        "input_embedding_dim": 16,
                        "cross_attention_dim": 12,
                        "attention_head_dim": 4,
                        "num_attention_heads": 4,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "interleave_self_attention": False,
                        "norm_type": "ada_norm",
                        "output_dim": 16,
                        "positional_embeddings": None,
                    },
                },
            }
        }
    )
    return LayerwiseFlowmatchingActionHead(config).eval()


def test_layerwise_action_head_dit_physics_branch_has_exact_control() -> None:
    torch.manual_seed(11)
    head = make_small_physics_head()
    hidden = torch.randn(2, 6, 16)
    prefixes = [torch.randn(2, 5, 12) for _ in range(2)]
    timestep = head.model.timestep_encoder(torch.tensor([3, 7]))
    physics_ids = torch.randint(0, 32, (2, 256))

    enabled_at_init = head._run_transformer_blocks(
        hidden, prefixes, timestep, physics_ids
    )
    head.physics_condition_enabled = False
    hard_disabled = head._run_transformer_blocks(
        hidden, prefixes, timestep, physics_ids
    )
    assert torch.equal(enabled_at_init, hard_disabled)

    head.physics_condition_enabled = True
    assert head.physics_adapter is not None
    torch.nn.init.normal_(
        head.physics_adapter.adapters["0"].output_proj.weight, std=0.1
    )
    trained_branch = head._run_transformer_blocks(
        hidden, prefixes, timestep, physics_ids
    )
    assert not torch.equal(trained_branch, hard_disabled)


def test_adapter_only_mode_freezes_original_action_head() -> None:
    head = make_small_physics_head(freeze_base=True)
    assert head.physics_adapter is not None
    trainable = {name for name, parameter in head.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith("physics_adapter.") for name in trainable)
    assert any(name.startswith("model.") for name, _ in head.named_parameters())


def test_action_only_injection_preserves_prefix_hidden_states() -> None:
    head = make_small_physics_head(
        action_tokens_only=True, injection_layers="1"
    )
    assert head.physics_adapter is not None
    torch.nn.init.normal_(
        head.physics_adapter.adapters["1"].output_proj.weight, std=0.1
    )
    hidden = torch.randn(2, 6, 16)
    prefixes = [torch.randn(2, 5, 12) for _ in range(2)]
    timestep = head.model.timestep_encoder(torch.tensor([3, 7]))
    output = head._run_transformer_blocks(
        hidden,
        prefixes,
        timestep,
        torch.randint(0, 32, (2, 256)),
        physics_query_start=2,
    )
    base = head._run_transformer_blocks(hidden, prefixes, timestep, None)
    assert torch.equal(output[:, :2], base[:, :2])
    assert not torch.equal(output[:, 2:], base[:, 2:])


def test_optimizer_groups_contain_only_physics_adapter_for_paired_run() -> None:
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qwen_vl_interface = nn.Linear(2, 2)
            self.discrete_conditioner = nn.Linear(2, 2)
            self.action_model = make_small_physics_head(freeze_base=True)

    model = Wrapper()
    config = OmegaConf.create(
        {
            "trainer": {
                "freeze_modules": "qwen_vl_interface,discrete_conditioner",
                "learning_rate": {"base": 1e-5, "action_model": 1e-4},
            }
        }
    )
    groups = build_param_lr_groups(model, config)
    optimizer_ids = {id(parameter) for group in groups for parameter in group["params"]}
    adapter_ids = {id(parameter) for parameter in model.action_model.physics_adapter.parameters()}
    assert optimizer_ids == adapter_ids


def test_adapter_only_action_head_can_backward_when_physics_is_dropped(
    monkeypatch,
) -> None:
    head = make_small_physics_head(freeze_base=True).train()
    assert head.physics_adapter is not None
    head.physics_adapter.condition_dropout = 0.5
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(()))
    hidden = torch.randn(2, 6, 16)
    prefixes = [torch.randn(2, 5, 12) for _ in range(2)]
    timestep = head.model.timestep_encoder(torch.tensor([3, 7]))

    output = head._run_transformer_blocks(
        hidden, prefixes, timestep, torch.randint(0, 32, (2, 256))
    )
    output.square().mean().backward()

    assert output.requires_grad
    assert all(
        parameter.grad is not None for parameter in head.physics_adapter.parameters()
    )


def test_per_sample_physics_usefulness_loss_is_finite_and_adapter_only() -> None:
    torch.manual_seed(29)
    head = make_small_physics_head(
        freeze_base=True, usefulness_weight=0.5
    ).train()
    prefixes = [torch.randn(4, 5, 12) for _ in range(2)]
    actions = torch.randn(4, 4, 3)
    physics_ids = torch.randint(0, 32, (4, 256))

    total_loss, metrics = head(
        prefixes,
        actions,
        physics_token_ids=physics_ids,
        physics_shuffle_group_size=2,
    )
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert metrics["physics_usefulness_loss"] == pytest.approx(0.01, abs=1e-5)
    adapter_gradients = [
        parameter.grad for parameter in head.physics_adapter.parameters()
    ]
    assert any(
        gradient is not None and torch.count_nonzero(gradient)
        for gradient in adapter_gradients
    )
    assert all(
        parameter.grad is None
        for name, parameter in head.named_parameters()
        if not name.startswith("physics_adapter.")
    )
