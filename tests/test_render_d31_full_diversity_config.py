import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.physical_tokens.render_d31_full_diversity_config import render_config


def test_render_d31_config_uses_episode_exclusion_and_selected_vocab(tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    template.write_text(
        "run_id: old\nframework:\n  dual_token_probe:\n    physics_vocab_size: 128\n"
        "datasets:\n  vla_data:\n    discrete_token_cache: old.npz\n"
        "    include_episode_ids: [0, 1]\ntrainer:\n  max_train_steps: 1\n"
        "  num_warmup_steps: 1\n  save_interval: 1\n  eval_interval: 2\n"
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected_codebook_size": 256}))
    output = tmp_path / "rendered.yaml"
    _, run_id = render_config(template, selection, tmp_path / "dual.npz", output, 2000)
    config = OmegaConf.load(output)
    assert config.framework.dual_token_probe.physics_vocab_size == 256
    assert list(config.datasets.vla_data.exclude_episode_ids) == [5, 20, 25, 39, 49]
    assert "include_episode_ids" not in config.datasets.vla_data
    assert config.trainer.max_train_steps == 2000
    assert config.run_id == run_id
