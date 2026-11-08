"""Pi flow policy conditioned on 32+64 dual-AR IDs and predictor states."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.QwenDualAutoregressiveReasoner import (
    QwenDualAutoregressiveReasoner,
    aligned_dual_token_hidden,
    attach_text_lora,
    format_dual_solution,
    keep_only_dual_target_labels,
    score_separate_heads,
)
from starVLA.model.framework.QwenPI_v4 import ConcatCrossAttentionFusion, Qwen_PI_v4
from starVLA.model.tools import FRAMEWORK_REGISTRY


class TokenStateMemory(nn.Module):
    """Combine a trained code embedding with the hidden state that predicted it."""

    def __init__(self, vlm_dim: int, expert_dim: int, token_count: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(2 * vlm_dim)
        self.proj = nn.Sequential(
            nn.Linear(2 * vlm_dim, expert_dim), nn.SiLU(),
            nn.Linear(expert_dim, expert_dim),
        )
        self.position = nn.Embedding(token_count, expert_dim)
        self.output_norm = nn.LayerNorm(expert_dim)
        nn.init.normal_(self.position.weight, std=0.02)

    def forward(self, ids: torch.Tensor, hidden: torch.Tensor,
                codebook: nn.Embedding) -> torch.Tensor:
        if ids.ndim != 2 or hidden.ndim != 3 or ids.shape != hidden.shape[:2]:
            raise ValueError("each token ID needs one aligned final-layer hidden state")
        if hidden.shape[-1] != codebook.embedding_dim:
            raise ValueError("token hidden dimension differs from code embedding")
        dtype = self.proj[0].weight.dtype
        values = torch.cat((codebook(ids).to(dtype), hidden.to(dtype)), dim=-1)
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.output_norm(self.proj(self.input_norm(values)) + self.position(positions))


@FRAMEWORK_REGISTRY.register("QwenPIv4DualAR32x64")
class QwenPIv4DualAR32x64(Qwen_PI_v4):
    """Warm-start the mature Pi and dual AR, then train their memory bridge."""

    _routed_input_embeddings = QwenDualAutoregressiveReasoner._routed_input_embeddings
    _strict_backbone_forward = QwenDualAutoregressiveReasoner._strict_backbone_forward
    action_logits = QwenDualAutoregressiveReasoner.action_logits
    physics_logits = QwenDualAutoregressiveReasoner.physics_logits

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        ar = config.framework.dual_autoregressive_reasoner
        policy = config.framework.dual_ar_policy
        action = config.framework.action_token_reasoner
        self.action_token_count = int(action.token_count)
        self.action_vocab_size = int(action.vocab_size)
        self.action_token_offset = int(action.token_offset)
        self.physics_token_count = int(ar.physics_token_count)
        self.physics_vocab_size = int(ar.physics_vocab_size)
        if self.action_token_count not in (8, 32) or self.physics_token_count != 64:
            raise ValueError("policy bridge requires 8 or 32 action and 64 physics tokens")
        if self.action_vocab_size != 525 or self.physics_vocab_size not in (2048, 25000):
            raise ValueError("dual AR checkpoint requires action K525 and physics K2048 or K25000")
        if not bool(ar.strict_modality_tying) or not bool(ar.separate_heads):
            raise ValueError("source dual AR must use strictly tied separate tables")
        self.strict_modality_tying = True
        self.prompt_template = str(ar.prompt_template)
        self.condition_source = str(policy.get("condition_source", "predicted"))
        if self.condition_source not in {"predicted", "target"}:
            raise ValueError("condition_source must be predicted or target")
        self.action_token_loss_weight = float(policy.get("action_loss_weight", 1e-3))
        self.physics_token_loss_weight = float(policy.get("physics_loss_weight", 1e-4))

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        for code in (0, self.action_vocab_size - 1):
            if int(tokenizer.convert_tokens_to_ids(f"<robot_action_{code}>")) != self.action_token_offset + code:
                raise ValueError("action token vocabulary differs from the dual AR")
        symbols = [f"<robot_physics_{index}>" for index in range(self.physics_vocab_size)]
        existing = [tokenizer.convert_tokens_to_ids(symbol) for symbol in symbols]
        if all(value == tokenizer.unk_token_id for value in existing):
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": symbols}, replace_additional_special_tokens=False
            )
            if added != self.physics_vocab_size:
                raise ValueError("failed to add complete physics vocabulary")
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        elif any(value == tokenizer.unk_token_id for value in existing):
            raise ValueError("physics vocabulary is partially installed")
        self.physics_token_offset = int(tokenizer.convert_tokens_to_ids(symbols[0]))
        if any(int(tokenizer.convert_tokens_to_ids(symbol)) != self.physics_token_offset + index
               for index, symbol in enumerate(symbols)):
            raise ValueError("physics token IDs must be contiguous")

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        # Joint training must update the continuous-action policy, not only
        # its new conditioning bridge. Keep the large Qwen-VL base frozen;
        # the Pi projection and complete DiT/action expert are trainable.
        for module in (
            self.prefix_proj, self.action_in_proj, self.action_out_proj,
            self.time_mlp, self.action_pos_embed, self.expert_layers,
            self.expert_final_norm,
        ):
            module.requires_grad_(True)
        vlm_dim = int(self.qwen_vl_interface.model.config.text_config.hidden_size)
        self.action_embeddings = nn.Embedding(self.action_vocab_size, vlm_dim)
        self.physics_embeddings = nn.Embedding(self.physics_vocab_size, vlm_dim)
        lora = ar.lora
        self.lora_adapters, self._lora_hook_handles = attach_text_lora(
            self.qwen_vl_interface.model.model.language_model,
            rank=int(lora.rank), alpha=float(lora.alpha), dropout=float(lora.dropout),
            target_names=tuple(lora.target_modules),
        )
        self.action_token_memory = TokenStateMemory(vlm_dim, self.expert_hidden_dim, self.action_token_count)
        self.physics_token_memory = TokenStateMemory(vlm_dim, self.expert_hidden_dim, self.physics_token_count)
        bottleneck = int(policy.get("fusion_bottleneck", 256))
        self.pi_concat_fusers = nn.ModuleList([
            ConcatCrossAttentionFusion(self.expert_hidden_dim, bottleneck)
            for _ in self.expert_layers
        ])
        self._token_losses = None
        self._in_joint_forward = False
        self._action_memory_length = 0
        self._physics_memory_length = 0

    def initialize_expanded_vocab_pretrained(self, checkpoint_path: str | Path) -> dict:
        """Load the mature Pi, allowing only new AR and bridge parameters."""
        path = Path(checkpoint_path).resolve()
        source = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        target = self.state_dict()
        prefixes = (
            "action_embeddings.", "physics_embeddings.", "lora_adapters.",
            "action_token_memory.", "physics_token_memory.", "pi_concat_fusers.",
        )
        base_keys = {key for key in target if not key.startswith(prefixes)}
        if set(source) != base_keys:
            raise ValueError(
                f"mature Pi keys differ: missing={list(base_keys-set(source))[:5]}, "
                f"extra={list(set(source)-base_keys)[:5]}"
            )
        expanded = {
            "qwen_vl_interface.model.model.language_model.embed_tokens.weight",
            "qwen_vl_interface.model.lm_head.weight",
        }
        adapted = {}
        for key, value in source.items():
            expected = target[key]
            if value.shape == expected.shape:
                adapted[key] = value
            elif (key in expanded and value.ndim == 2 and expected.ndim == 2
                  and value.shape[1] == expected.shape[1]
                  and value.shape[0] >= self.physics_token_offset
                  and expected.shape[0] == self.physics_token_offset + self.physics_vocab_size):
                full = expected.detach().clone()
                full[:self.physics_token_offset].copy_(value[:self.physics_token_offset])
                adapted[key] = full
            else:
                raise ValueError(f"mature Pi tensor has incompatible shape: {key}")
        result = self.load_state_dict(adapted, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != set(target) - base_keys:
            raise ValueError("mature Pi checkpoint did not load with exact extension boundary")
        return {"checkpoint": str(path), "loaded_tensors": len(source)}

    def initialize_trainable_checkpoint(self, checkpoint_path: str | Path) -> dict:
        """Load only the dual-AR LoRA and tied code tables."""
        path = Path(checkpoint_path).resolve()
        source = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        expected = {"action_embeddings.weight", "physics_embeddings.weight"}
        expected.update(key for key in self.state_dict() if key.startswith("lora_adapters."))
        if set(source) != expected:
            raise ValueError(
                f"dual AR keys differ: missing={list(expected-set(source))[:5]}, "
                f"extra={list(set(source)-expected)[:5]}"
            )
        self.load_state_dict(source, strict=False)
        return {"checkpoint": str(path), "loaded_tensors": len(source)}

    def _reasoner_instructions(self, examples: List[dict]) -> list[str]:
        return [self.prompt_template.format(instruction=example["lang"]) for example in examples]

    def _teacher_forced_token_features(self, images, examples):
        solutions = [format_dual_solution(
            example["action_token_ids"], example["physics_token_ids"],
            action_vocab_size=self.action_vocab_size,
            physics_vocab_size=self.physics_vocab_size,
            action_count=self.action_token_count, physics_count=self.physics_token_count,
        ) for example in examples]
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=self._reasoner_instructions(examples),
            solutions=solutions,
        )
        inputs = keep_only_dual_target_labels(
            inputs, action_offset=self.action_token_offset,
            action_vocab_size=self.action_vocab_size,
            physics_offset=self.physics_token_offset,
            physics_vocab_size=self.physics_vocab_size,
            action_count=self.action_token_count, physics_count=self.physics_token_count,
        )
        labels = inputs.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=labels.is_cuda):
            hidden = self._strict_backbone_forward(inputs)[0].last_hidden_state
            scores = score_separate_heads(
                hidden, labels, action_head=self.action_logits,
                physics_head=self.physics_logits,
                action_offset=self.action_token_offset,
                action_vocab_size=self.action_vocab_size,
                physics_offset=self.physics_token_offset,
                physics_vocab_size=self.physics_vocab_size,
            )
        action_hidden, physics_hidden = aligned_dual_token_hidden(
            hidden, labels, action_offset=self.action_token_offset,
            action_vocab_size=self.action_vocab_size,
            physics_offset=self.physics_token_offset,
            physics_vocab_size=self.physics_vocab_size,
            action_count=self.action_token_count, physics_count=self.physics_token_count,
        )
        self._token_losses = (scores["action_ce"], scores["physics_ce"])
        if self.condition_source == "target":
            action_ids = torch.as_tensor(np.stack([
                example["action_token_ids"] for example in examples
            ]), device=hidden.device, dtype=torch.long)
            physics_ids = torch.as_tensor(np.stack([
                example["physics_token_ids"] for example in examples
            ]), device=hidden.device, dtype=torch.long)
        else:
            batch = hidden.shape[0]
            action_ids = scores["action_logits"].reshape(
                batch, self.action_token_count, self.action_vocab_size
            ).argmax(dim=-1)
            physics_ids = scores["physics_logits"].reshape(
                batch, self.physics_token_count, self.physics_vocab_size
            ).argmax(dim=-1)
        return action_ids, physics_ids, action_hidden, physics_hidden

    @torch.no_grad()
    def _free_running_token_features(self, images, examples):
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=self._reasoner_instructions(examples)
        )
        prompt_length = int(inputs["input_ids"].shape[1])
        attention_mask = inputs["attention_mask"]
        position = torch.arange(prompt_length, device=attention_mask.device)
        outputs, rope_deltas = self._strict_backbone_forward(
            inputs, cache_position=position, use_cache=True
        )
        action_ids, physics_ids, action_hidden, physics_hidden = [], [], [], []
        total_steps = self.action_token_count + self.physics_token_count
        for step in range(total_steps):
            predictor = outputs.last_hidden_state[:, -1, :]
            if step < self.action_token_count:
                code = self.action_logits(predictor).argmax(dim=-1)
                action_ids.append(code)
                action_hidden.append(predictor)
                next_id = code + self.action_token_offset
            else:
                code = self.physics_logits(predictor).argmax(dim=-1)
                physics_ids.append(code)
                physics_hidden.append(predictor)
                next_id = code + self.physics_token_offset
            if step == total_steps - 1:
                break
            attention_mask = torch.cat((
                attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))
            ), dim=1)
            next_position = torch.tensor([prompt_length + step], device=next_id.device)
            outputs, _ = self._strict_backbone_forward(
                {"input_ids": next_id[:, None], "attention_mask": attention_mask},
                past_key_values=outputs.past_key_values,
                cache_position=next_position, rope_deltas=rope_deltas, use_cache=True,
            )
        return (
            torch.stack(action_ids, dim=1), torch.stack(physics_ids, dim=1),
            torch.stack(action_hidden, dim=1), torch.stack(physics_hidden, dim=1),
        )

    def _encode_qwen_hidden(self, batch_images, instructions, examples=None):
        if not examples:
            raise ValueError("dual-AR policy requires examples")
        # The mature Pi prefix sees only the observation.  The AR pass sees
        # teacher-forced targets during training, but its target positions are
        # never included in the mature prefix memory.
        last_hidden, prefix, inputs = Qwen_PI_v4._encode_qwen_hidden(
            self, batch_images, instructions, examples=None
        )
        if self._in_joint_forward:
            action_ids, physics_ids, action_hidden, physics_hidden = (
                self._teacher_forced_token_features(batch_images, examples)
            )
        else:
            self._token_losses = None
            action_ids, physics_ids, action_hidden, physics_hidden = (
                self._free_running_token_features(batch_images, examples)
            )
        action_memory = self.action_token_memory(
            action_ids, action_hidden, self.action_embeddings
        ).to(prefix.dtype)
        physics_memory = self.physics_token_memory(
            physics_ids, physics_hidden, self.physics_embeddings
        ).to(prefix.dtype)
        self._action_memory_length = int(action_memory.shape[1])
        self._physics_memory_length = int(physics_memory.shape[1])
        return last_hidden, torch.cat((prefix, action_memory, physics_memory), dim=1), inputs

    def _split_expert_memories(self, prefix_hidden):
        action_count = self._action_memory_length
        physics_count = self._physics_memory_length
        if (action_count, physics_count) != (self.action_token_count, self.physics_token_count):
            raise ValueError("dual-AR memory is missing or has wrong token counts")
        base_end = prefix_hidden.shape[1] - action_count - physics_count
        if base_end <= 0:
            raise ValueError("dual-AR memory requires a nonempty Pi prefix")
        action_end = base_end + action_count
        return (
            prefix_hidden[:, :base_end],
            prefix_hidden[:, base_end:action_end],
            prefix_hidden[:, action_end:],
        )

    def _expert_concat_fuser(self, layer_index: int) -> nn.Module:
        return self.pi_concat_fusers[layer_index]

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        self._in_joint_forward = True
        try:
            result = super().forward(examples=examples, **kwargs)
        finally:
            self._in_joint_forward = False
        if self._token_losses is None:
            raise RuntimeError("joint forward did not compute token losses")
        action_ce, physics_ce = self._token_losses
        flow_loss = result["action_loss"]
        result.update({
            "action_loss": flow_loss + self.action_token_loss_weight * action_ce
            + self.physics_token_loss_weight * physics_ce,
            "action_dit_loss": flow_loss.detach(),
            "action_token_ce_loss": action_ce,
            "physics_token_ce_loss": physics_ce,
        })
        return result
