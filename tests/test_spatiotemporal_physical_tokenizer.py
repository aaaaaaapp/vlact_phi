import pytest
import torch

from starVLA.model.modules.tokenizers import SpatiotemporalPhysicalLanguageTokenizer
from scripts.physical_tokens.export_spatiotemporal_physics_token_ids import build_model


def make_model(quantizer="ema_vq"):
    return SpatiotemporalPhysicalLanguageTokenizer(
        feature_dim=12,
        num_frames=5,
        spatial_tokens=4,
        queries_per_transition=3,
        hidden_dim=24,
        latent_dim=8,
        num_heads=4,
        quantizer=quantizer,
        codebook_size=16,
        fsq_levels=(4, 4),
    )


@pytest.mark.parametrize("quantizer", ["ema_vq", "fsq"])
def test_spatiotemporal_tokenizer_shapes_and_decode(quantizer):
    model = make_model(quantizer)
    features = torch.randn(2, 5, 4, 12)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    output = model(features, valid)
    assert output["token_ids"].shape == (2, 12)
    assert output["reconstruction"].shape == (2, 4, 4, 12)
    assert model.encode_continuous(features, valid).shape == (2, 4, 3, 8)
    assert model.reconstruct_continuous(features, valid).shape == (2, 4, 4, 12)
    assert torch.isfinite(output["loss"])
    decoded = model.decode_tokens(output["token_ids"])
    assert decoded.shape == (2, 4, 4, 12)


def test_default_contract_emits_256_tokens():
    model = SpatiotemporalPhysicalLanguageTokenizer(feature_dim=32)
    assert model.num_transitions == 16
    assert model.queries_per_transition == 16
    assert model.num_tokens == 256


def test_action_auxiliary_loss_reaches_quantized_spatial_tokens():
    model = SpatiotemporalPhysicalLanguageTokenizer(
        feature_dim=12,
        num_frames=3,
        spatial_tokens=4,
        queries_per_transition=4,
        hidden_dim=24,
        latent_dim=8,
        num_heads=4,
        codebook_size=16,
        token_routing="spatial_aligned",
        action_aux_dim=14,
        action_aux_weight=0.25,
    )
    features = torch.randn(2, 3, 4, 12)
    actions = torch.randn(2, 3, 14)
    actions[..., 12:14] = torch.randint(0, 2, (2, 3, 2)).float()
    output = model(features, torch.ones(2, 3, dtype=torch.bool), actions)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["action_aux_loss"])
    assert output["action_prediction"].shape == (2, 2, 14)
    output["loss"].backward()
    assert model.to_latent.weight.grad is not None
    assert torch.isfinite(model.to_latent.weight.grad).all()


def test_flatten_action_auxiliary_preserves_spatial_slots_and_rebuilds():
    model = SpatiotemporalPhysicalLanguageTokenizer(
        feature_dim=12,
        num_frames=3,
        spatial_tokens=4,
        queries_per_transition=4,
        hidden_dim=24,
        latent_dim=8,
        num_heads=4,
        codebook_size=16,
        token_routing="spatial_aligned",
        action_aux_dim=14,
        action_aux_pooling="flatten",
    )
    quantized = torch.randn(2, 2, 4, 8)
    assert model.action_aux_features(quantized).shape == (2, 2, 32)
    assert model.predict_action_aux(quantized).shape == (2, 2, 14)
    config = {
        "feature_dim": 12,
        "num_frames": 3,
        "spatial_tokens": 4,
        "queries_per_transition": 4,
        "hidden_dim": 24,
        "latent_dim": 8,
        "num_heads": 4,
        "quantizer": "ema_vq",
        "codebook_size": 16,
        "fsq_levels": (4, 4),
        "ema_decay": 0.99,
        "commitment_weight": 0.25,
        "dead_code_steps": 200,
        "token_routing": "spatial_aligned",
        "action_aux_dim": 14,
        "action_aux_pooling": "flatten",
    }
    rebuilt = build_model(config)
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt.action_aux_pooling == "flatten"


def test_spatiotemporal_tokenizer_validates_shapes():
    model = make_model()
    with pytest.raises(ValueError, match="expected features"):
        model(torch.randn(2, 4, 4, 12))


def test_spatial_aligned_routing_rejects_non_nested_query_grid():
    with pytest.raises(ValueError, match="square spatial/query grids"):
        SpatiotemporalPhysicalLanguageTokenizer(
            feature_dim=12,
            num_frames=3,
            spatial_tokens=4,
            queries_per_transition=3,
            hidden_dim=24,
            latent_dim=8,
            num_heads=4,
            token_routing="spatial_aligned",
        )


@pytest.mark.parametrize("queries_per_transition", [1, 4, 16])
def test_spatial_aligned_routing_supports_nested_query_grids(queries_per_transition):
    model = SpatiotemporalPhysicalLanguageTokenizer(
        feature_dim=12,
        num_frames=3,
        spatial_tokens=16,
        queries_per_transition=queries_per_transition,
        hidden_dim=24,
        latent_dim=8,
        num_heads=4,
        token_routing="spatial_aligned",
    )
    encoder_mask = model.spatial_alignment_mask
    decoder_mask = model.spatial_decode_alignment_mask
    assert encoder_mask.shape == (queries_per_transition, 16)
    assert decoder_mask.shape == (16, queries_per_transition)
    assert torch.equal(decoder_mask, encoder_mask.transpose(0, 1))
    assert torch.isfinite(encoder_mask).sum(dim=0).eq(1).all()


def test_spatial_aligned_routing_preserves_slot_locality():
    torch.manual_seed(23)
    model = SpatiotemporalPhysicalLanguageTokenizer(
        feature_dim=12,
        num_frames=3,
        spatial_tokens=4,
        queries_per_transition=4,
        hidden_dim=24,
        latent_dim=8,
        num_heads=4,
        codebook_size=16,
        token_routing="spatial_aligned",
    ).eval()
    features = torch.randn(1, 3, 4, 12)
    changed = features.clone()
    changed[:, 1, 2] += 3.0
    original_latents = model.encode_continuous(features)
    changed_latents = model.encode_continuous(changed)
    assert not torch.equal(original_latents[:, 0, 2], changed_latents[:, 0, 2])
    assert torch.equal(original_latents[:, 0, :2], changed_latents[:, 0, :2])
    assert torch.equal(original_latents[:, 0, 3:], changed_latents[:, 0, 3:])

    latents = torch.randn(1, 2, 4, 8)
    changed_latents = latents.clone()
    changed_latents[:, 0, 1] += 2.0
    original_reconstruction = model._decode_latents(latents)
    changed_reconstruction = model._decode_latents(changed_latents)
    assert not torch.equal(
        original_reconstruction[:, 0, 1], changed_reconstruction[:, 0, 1]
    )
    assert torch.equal(
        original_reconstruction[:, 0, 0], changed_reconstruction[:, 0, 0]
    )
    assert torch.equal(
        original_reconstruction[:, 0, 2:], changed_reconstruction[:, 0, 2:]
    )


def test_export_rebuild_preserves_checkpoint_token_routing():
    source = SpatiotemporalPhysicalLanguageTokenizer(
        feature_dim=12,
        num_frames=3,
        spatial_tokens=4,
        queries_per_transition=4,
        hidden_dim=24,
        latent_dim=8,
        num_heads=4,
        codebook_size=16,
        token_routing="spatial_aligned",
    )
    config = {
        "feature_dim": source.feature_dim,
        "num_frames": source.num_frames,
        "spatial_tokens": source.spatial_tokens,
        "queries_per_transition": source.queries_per_transition,
        "hidden_dim": 24,
        "latent_dim": 8,
        "num_heads": 4,
        "quantizer": "ema_vq",
        "codebook_size": 16,
        "fsq_levels": (4, 4),
        "ema_decay": 0.99,
        "commitment_weight": 0.25,
        "dead_code_steps": 200,
        "token_routing": "spatial_aligned",
    }

    rebuilt = build_model(config)

    assert rebuilt.token_routing == "spatial_aligned"
    assert rebuilt.spatial_alignment_mask.shape == (4, 4)
