"""QwenPI with oracle or predicted discrete action/physics conditions."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from starVLA.model.framework.QwenPI import Qwen_PI
from starVLA.model.modules.conditioning import DiscreteTokenConditioner
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _config_value(config, name: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    return default if value is None else value


@FRAMEWORK_REGISTRY.register("QwenPIPhysicalTokens")
class QwenPIPhysicalTokens(Qwen_PI):
    """Lightweight QwenPI whose DiT prefix can include discrete token IDs."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        condition_config = getattr(config.framework, "discrete_conditioning", None)
        if condition_config is None:
            raise ValueError("framework.discrete_conditioning is required")
        self.discrete_conditioner = DiscreteTokenConditioner(
            hidden_dim=self.qwen_vl_interface.model.config.hidden_size,
            action_vocab_size=int(_config_value(condition_config, "action_vocab_size")),
            physics_vocab_size=int(_config_value(condition_config, "physics_vocab_size", 0)),
            max_action_tokens=int(_config_value(condition_config, "max_action_tokens", 32)),
            max_physics_tokens=int(_config_value(condition_config, "max_physics_tokens", 32)),
            gate_init=float(_config_value(condition_config, "gate_init", 0.1)),
            action_gate_init=_config_value(condition_config, "action_gate_init"),
            physics_gate_init=_config_value(condition_config, "physics_gate_init"),
            action_condition_dropout=float(
                _config_value(condition_config, "action_condition_dropout", 0.0)
            ),
            physics_condition_dropout=float(
                _config_value(condition_config, "physics_condition_dropout", 0.0)
            ),
            physics_fusion_mode=str(
                _config_value(condition_config, "physics_fusion_mode", "concat")
            ),
            physics_fusion_dim=int(
                _config_value(condition_config, "physics_fusion_dim", 256)
            ),
            physics_fusion_heads=int(
                _config_value(condition_config, "physics_fusion_heads", 8)
            ),
            physics_condition_enabled=bool(
                _config_value(condition_config, "physics_condition_enabled", True)
            ),
        )
        self.discrete_conditioning_enabled = bool(
            _config_value(condition_config, "enabled", True)
        )
        self.physics_fusion_mode = str(
            _config_value(condition_config, "physics_fusion_mode", "concat")
        )

    def set_discrete_conditioning_enabled(self, enabled: bool) -> None:
        self.discrete_conditioning_enabled = bool(enabled)

    @staticmethod
    def _batch_token_ids(
        examples: List[dict],
        key: str,
        device: torch.device,
    ) -> torch.Tensor | None:
        present = [key in example and example[key] is not None for example in examples]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"{key} must be present for every sample or none")
        values = np.stack([np.asarray(example[key], dtype=np.int64) for example in examples])
        if values.ndim != 2:
            raise ValueError(f"{key} must contain one-dimensional token sequences")
        return torch.from_numpy(values).to(device=device)

    def _condition_vl_embeddings(
        self,
        vl_embs_list: list[torch.Tensor],
        examples: List[dict],
    ) -> list[torch.Tensor]:
        if not self.discrete_conditioning_enabled:
            return list(vl_embs_list)
        device = vl_embs_list[0].device
        action_ids = self._batch_token_ids(examples, "action_token_ids", device)
        physics_ids = self._batch_token_ids(examples, "physics_token_ids", device)
        if action_ids is None and physics_ids is None:
            raise ValueError(
                "discrete conditioning is enabled but examples contain no action_token_ids "
                "or physics_token_ids"
            )
        prefix_physics_ids = (
            None
            if self.physics_fusion_mode == "dit_multilayer_cross_attention"
            else physics_ids
        )
        return self.discrete_conditioner(
            vl_embs_list,
            action_token_ids=action_ids,
            physics_token_ids=prefix_physics_ids,
            enabled=True,
        )

    def _action_model_condition_kwargs(
        self,
        examples: List[dict],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if (
            not self.discrete_conditioning_enabled
            or self.physics_fusion_mode != "dit_multilayer_cross_attention"
        ):
            return {}
        physics_ids = self._batch_token_ids(examples, "physics_token_ids", device)
        if physics_ids is None:
            raise ValueError(
                "dit_multilayer_cross_attention requires physics_token_ids for every sample"
            )
        return {"physics_token_ids": physics_ids}
