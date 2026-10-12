# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Jinhui YE/ HKUST] in [2026].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].


from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_loss import (
    flow_matching_loss_with_endpoint_wrap,
    masked_mean_per_sample,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.conditioning import MultiLayerPhysicsTokenAdapter

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024, num_embodiments=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}  # default for qwen2.5-vl


def layerwise_encoder_hidden(
    vl_embs_list: list[torch.Tensor],
    layer_idx: int,
    interleave_self_attention: bool,
) -> torch.Tensor | None:
    if interleave_self_attention and layer_idx % 2 == 1:
        return None
    return vl_embs_list[layer_idx]


class LayerwiseFlowmatchingActionHead(nn.Module):
    """
    Layer-wise cross-attention DiT action head.

    NOTE on configuration boundary:
        This module is intentionally decoupled from any specific VLM backbone.
        It ONLY reads from ``global_config.framework.action_model`` (and its
        ``diffusion_model_cfg`` sub-tree).  The framework (e.g. ``Qwen_PI`` /
        ``Qwen_PI_v3``) is responsible for populating ``diffusion_model_cfg``
        with the correct DiT shape **before** calling ``get_action_model``:

            diffusion_model_cfg.num_layers           = <DiT depth>
            diffusion_model_cfg.input_embedding_dim  = <DiT internal hidden>
            diffusion_model_cfg.cross_attention_dim  = <DiT internal hidden>
            diffusion_model_cfg.num_attention_heads  = input_embedding_dim // attention_head_dim

        The head no longer reads ``framework.qwenvl.*`` directly.  See
        ``starVLA/model/framework/VLM4A/diffusion_model_cfg.md`` for details.
    """

    def __init__(
        self,
        global_config,
        **kwargs,
    ):
        super().__init__()
        self.full_config = global_config
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        # ------------------------------------------------------------------
        # Pure consumer: trust diffusion_model_cfg.  Apply DiTConfig only as
        # a fallback for keys the framework forgot to set.  This keeps the
        # head free of any VLM-specific knowledge.
        # ------------------------------------------------------------------
        for k, v in DiTConfig.items():
            if diffusion_model_cfg.get(k, None) is None:
                diffusion_model_cfg[k] = v

        # Build a plain dict view and drop framework-side hints that are NOT
        # DiT constructor kwargs (e.g. `action_dit_hidden_dim`).  This keeps the
        # head agnostic of any framework convention while still being safe
        # against accidentally-leaked hint keys.
        _DIT_NON_KWARGS = {"action_dit_hidden_dim"}
        diffusion_model_cfg_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in _DIT_NON_KWARGS}

        self.input_embedding_dim = diffusion_model_cfg_kwargs["input_embedding_dim"]
        self.model = DiT(**diffusion_model_cfg_kwargs)  # TODO: ideally copy LLM init from VLM
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        # `action_horizon` is the canonical chunk length, provided directly by
        # the YAML / CLI overrides.
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = (
            MLP(
                input_dim=action_config.state_dim,
                output_dim=self.input_embedding_dim,
            )
            if action_config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config
        condition_config = getattr(global_config.framework, "discrete_conditioning", None)
        physics_fusion_mode = (
            condition_config.get("physics_fusion_mode", "concat")
            if condition_config is not None
            else "concat"
        )
        if physics_fusion_mode == "dit_multilayer_cross_attention":
            self.physics_adapter = MultiLayerPhysicsTokenAdapter(
                hidden_dim=self.input_embedding_dim,
                physics_vocab_size=int(condition_config.physics_vocab_size),
                num_layers=len(self.model.transformer_blocks),
                injection_layers=condition_config.get("dit_injection_layers", "all"),
                max_physics_tokens=int(condition_config.get("max_physics_tokens", 256)),
                num_heads=int(condition_config.get("physics_fusion_heads", 8)),
                attention_dropout=float(
                    condition_config.get("dit_attention_dropout", 0.0)
                ),
                condition_dropout=float(
                    condition_config.get("physics_condition_dropout", 0.2)
                ),
                memory_pooling=str(
                    condition_config.get("dit_physics_memory_pooling", "none")
                ),
                tokens_per_transition=int(
                    condition_config.get("physics_tokens_per_transition", 16)
                ),
                adapter_type=str(
                    condition_config.get("dit_physics_adapter_type", "cross_attention")
                ),
            )
            self.physics_condition_enabled = bool(
                condition_config.get("physics_condition_enabled", True)
            )
            self.physics_usefulness_weight = float(
                condition_config.get("physics_usefulness_weight", 0.0)
            )
            self.physics_usefulness_margin = float(
                condition_config.get("physics_usefulness_margin", 0.01)
            )
            self.dit_action_tokens_only = bool(
                condition_config.get("dit_action_tokens_only", False)
            )
            if self.physics_adapter.adapter_type == "temporal_film" and (
                not self.dit_action_tokens_only
                or self.physics_adapter.memory_pooling != "transition_mean"
            ):
                raise ValueError(
                    "temporal_film requires dit_action_tokens_only=true and "
                    "dit_physics_memory_pooling=transition_mean"
                )
            if self.physics_usefulness_weight < 0 or self.physics_usefulness_margin < 0:
                raise ValueError("physics usefulness weight and margin must be non-negative")
            if bool(condition_config.get("dit_freeze_base_action_model", False)):
                for parameter in self.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.physics_adapter.parameters():
                    parameter.requires_grad_(True)
        else:
            self.physics_adapter = None
            self.physics_condition_enabled = False
            self.physics_usefulness_weight = 0.0
            self.physics_usefulness_margin = 0.0
            self.dit_action_tokens_only = False

    def _run_transformer_blocks(
        self,
        model_output: torch.Tensor,
        vl_embs_list: list[torch.Tensor],
        temb: torch.Tensor,
        physics_token_ids: torch.Tensor | None = None,
        physics_condition_drop: bool | None = None,
        physics_query_start: int | None = None,
    ) -> torch.Tensor:
        physics_memory = (
            self.physics_adapter.prepare_memory(
                physics_token_ids,
                enabled=self.physics_condition_enabled,
                drop_condition=physics_condition_drop,
            )
            if self.physics_adapter is not None
            else None
        )
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=layerwise_encoder_hidden(
                    vl_embs_list,
                    layer_idx,
                    self.model.config.interleave_self_attention,
                ),
                temb=temb,
            )
            if self.physics_adapter is not None:
                if self.dit_action_tokens_only and physics_query_start is not None:
                    prefix = model_output[:, :physics_query_start]
                    action_suffix = self.physics_adapter.apply_layer(
                        layer_idx,
                        model_output[:, physics_query_start:],
                        physics_memory,
                        enabled=self.physics_condition_enabled,
                    )
                    model_output = torch.cat((prefix, action_suffix), dim=1)
                else:
                    model_output = self.physics_adapter.apply_layer(
                        layer_idx,
                        model_output,
                        physics_memory,
                        enabled=self.physics_condition_enabled,
                    )
        if (
            self.physics_adapter is not None
            and self.physics_condition_enabled
            and self.training
            and physics_token_ids is not None
            and physics_memory is None
        ):
            model_output = self.physics_adapter.attach_zero_gradient(model_output)
        return model_output

    @staticmethod
    def _cyclically_shuffle_conditions(
        token_ids: torch.Tensor, group_size: int
    ) -> torch.Tensor:
        if group_size < 2 or token_ids.shape[0] % group_size:
            raise ValueError(
                "physics usefulness requires group_size >= 2 and a divisible batch"
            )
        grouped = token_ids.reshape(-1, group_size, token_ids.shape[-1])
        return torch.roll(grouped, shifts=1, dims=1).reshape_as(token_ids)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        action_valid_mask: torch.Tensor | None = None,
        action_dim_mask: torch.Tensor | None = None,
        angular_dim_mask: torch.Tensor | None = None,
        action_loss_weights: torch.Tensor | None = None,
        physics_token_ids: torch.Tensor | None = None,
        physics_shuffle_group_size: int | None = None,
    ):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, D_action)
        """
        device = actions.device
        num_layers = len(vl_embs_list)
        B, L, D = vl_embs_list[0].shape
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        # Encode timesteps
        temb = self.model.timestep_encoder(t_discretized)

        # Layerwise cross-attention with vl_embs
        model_output = sa_embs
        physics_query_start = (
            model_output.shape[1] - actions.shape[1]
            if self.dit_action_tokens_only
            else None
        )
        use_physics_ranking = (
            self.physics_usefulness_weight > 0
            and physics_token_ids is not None
            and self.physics_condition_enabled
        )
        shared_condition_drop = None
        if use_physics_ranking and self.training and self.physics_adapter is not None:
            shared_condition_drop = bool(
                torch.rand((), device=device) < self.physics_adapter.condition_dropout
            )
        model_output = self._run_transformer_blocks(
            model_output,
            vl_embs_list,
            temb,
            physics_token_ids,
            physics_condition_drop=shared_condition_drop,
            physics_query_start=physics_query_start,
        )

        # TODO miss self att and _process_output, but work well
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        loss, _, _ = flow_matching_loss_with_endpoint_wrap(
            velocity_pred=pred_actions,
            velocity_target=velocity,
            noisy_actions=noisy_trajectory,
            target_actions=actions,
            t=t[:, 0, 0],
            config=self.full_config,
            action_valid_mask=action_valid_mask,
            action_dim_mask=action_dim_mask,
            angular_dim_mask=angular_dim_mask,
            sample_weights=action_loss_weights,
        )
        if not use_physics_ranking:
            return loss

        group_size = int(physics_shuffle_group_size or physics_token_ids.shape[0])
        shuffled_ids = self._cyclically_shuffle_conditions(
            physics_token_ids, group_size
        )
        shuffled_output = self._run_transformer_blocks(
            sa_embs,
            vl_embs_list,
            temb,
            shuffled_ids,
            physics_condition_drop=shared_condition_drop,
            physics_query_start=physics_query_start,
        )
        shuffled_pred = self.action_decoder(shuffled_output)[:, -actions.shape[1] :]
        shuffled_loss, _, _ = flow_matching_loss_with_endpoint_wrap(
            velocity_pred=shuffled_pred,
            velocity_target=velocity,
            noisy_actions=noisy_trajectory,
            target_actions=actions,
            t=t[:, 0, 0],
            config=self.full_config,
            action_valid_mask=action_valid_mask,
            action_dim_mask=action_dim_mask,
            angular_dim_mask=angular_dim_mask,
            sample_weights=action_loss_weights,
        )
        aligned_per_sample = masked_mean_per_sample(
            (pred_actions - velocity) ** 2,
            action_valid_mask=action_valid_mask,
            action_dim_mask=action_dim_mask,
        )
        shuffled_per_sample = masked_mean_per_sample(
            (shuffled_pred - velocity) ** 2,
            action_valid_mask=action_valid_mask,
            action_dim_mask=action_dim_mask,
        )
        ranking_per_sample = torch.relu(
            self.physics_usefulness_margin
            + aligned_per_sample
            - shuffled_per_sample
        )
        if action_loss_weights is None:
            ranking_loss = ranking_per_sample.mean()
        else:
            ranking_weights = action_loss_weights.to(
                device=ranking_per_sample.device,
                dtype=ranking_per_sample.dtype,
            ).reshape(-1)
            ranking_loss = (
                ranking_per_sample * ranking_weights
            ).sum() / ranking_weights.sum().clamp(min=1e-8)
        total_loss = loss + self.physics_usefulness_weight * ranking_loss
        return total_loss, {
            "physics_aligned_loss": loss.detach(),
            "physics_shuffled_loss": shuffled_loss.detach(),
            "physics_usefulness_loss": ranking_loss.detach(),
        }

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        physics_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # Encode timestep
            temb = self.model.timestep_encoder(timesteps_tensor)

            # Layerwise cross-attention with vl_embs_list
            model_output = sa_embs
            physics_query_start = (
                model_output.shape[1] - self.action_horizon
                if self.dit_action_tokens_only
                else None
            )
            model_output = self._run_transformer_blocks(
                model_output,
                vl_embs_list,
                temb,
                physics_token_ids,
                physics_query_start=physics_query_start,
            )
            # TODO miss self att and _process_output
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return LayerwiseFlowmatchingActionHead(global_config=config)
