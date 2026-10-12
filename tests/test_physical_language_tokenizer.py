import torch

from starVLA.model.modules.tokenizers import PhysicalLanguageTokenizer


def test_physical_tokenizer_supports_more_tokens_than_time_anchors() -> None:
    torch.manual_seed(23)
    model = PhysicalLanguageTokenizer(
        feature_dim=32,
        horizon=4,
        num_tokens=8,
        hidden_dim=64,
        latent_dim=16,
        num_layers=1,
        num_heads=4,
        codebook_size=32,
        dead_code_steps=0,
    )
    features = torch.randn(3, 4, 32)
    features[:, 0] = 0
    valid = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool
    )
    model.set_normalization(features.mean((0, 1)), features.std((0, 1)).clamp_min(0.1))

    output = model.loss(features, valid_mask=valid)
    output["loss"].backward()

    assert output["token_ids"].shape == (3, 8)
    assert output["reconstruction"].shape == features.shape
    assert torch.isfinite(output["loss"])
    assert any(parameter.grad is not None for parameter in model.temporal_encoder.parameters())


def test_physical_tokenizer_fsq_checkpoint_round_trip() -> None:
    torch.manual_seed(29)
    config = dict(
        feature_dim=16,
        horizon=4,
        num_tokens=2,
        hidden_dim=32,
        latent_dim=8,
        num_layers=1,
        num_heads=4,
        quantizer="fsq",
        fsq_levels=(4, 4, 4),
    )
    first = PhysicalLanguageTokenizer(**config).eval()
    first.set_normalization(torch.randn(16), torch.rand(16) + 0.1)
    features = torch.randn(2, 4, 16)
    valid = torch.ones(2, 4, dtype=torch.bool)
    first_output = first(features, valid)

    second = PhysicalLanguageTokenizer(**config).eval()
    second.load_state_dict(first.state_dict())
    second_output = second(features, valid)

    assert torch.equal(first_output["token_ids"], second_output["token_ids"])
    assert torch.allclose(first_output["reconstruction"], second_output["reconstruction"])
    assert torch.allclose(
        second.decode_tokens(second_output["token_ids"]),
        second_output["reconstruction"],
        atol=1e-6,
    )
