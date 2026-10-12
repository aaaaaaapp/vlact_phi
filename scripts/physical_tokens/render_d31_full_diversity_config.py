#!/usr/bin/env python3
"""Render the selected-codebook D31 VLM probe configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def render_config(
    template: Path,
    selection: Path,
    dual_cache: Path,
    output: Path,
    steps: int,
) -> tuple[int, str]:
    selected = json.loads(selection.read_text())
    codebook_size = int(selected["selected_codebook_size"])
    if codebook_size not in (128, 256):
        raise ValueError(f"unsupported selected codebook size: {codebook_size}")
    config = OmegaConf.load(template)
    run_id = f"qwen_dual_token_probe_d31_full_diversity_k{codebook_size}_steps{steps}"
    config.run_id = run_id
    config.framework.dual_token_probe.physics_vocab_size = codebook_size
    config.datasets.vla_data.discrete_token_cache = str(dual_cache.resolve())
    if "include_episode_ids" in config.datasets.vla_data:
        del config.datasets.vla_data.include_episode_ids
    config.datasets.vla_data.exclude_episode_ids = [5, 20, 25, 39, 49]
    config.trainer.max_train_steps = steps
    config.trainer.num_warmup_steps = max(1, steps // 10)
    config.trainer.save_interval = steps
    config.trainer.eval_interval = steps + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output)
    return codebook_size, run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dual-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    codebook_size, run_id = render_config(
        args.template, args.selection, args.dual_cache, args.output, args.steps
    )
    print(json.dumps({"codebook_size": codebook_size, "run_id": run_id}, sort_keys=True))


if __name__ == "__main__":
    main()
