# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Layer-wise cross-DiT flow-matching action head with a decoupled latent-action
branch ("world-model" variant, ``QwenWM_LA``).

Latent and action are **independent** flow-matching streams that share the same
DiT backbone and VLM layer-wise hidden states, but run as separate forward
passes (no joint ``[latent, state, action]`` sequence):

  - action stream: ``[state, action]`` with a pi0-style block-causal self-attn
    mask so the state token cannot attend to action tokens;
  - latent stream: latent tokens only (no state / action), free self-attn.

Independent (non-shared with the action stream) submodules:
  - ``latent_action_encoder``: embeds noisy latent tokens (own sinusoidal time).
  - ``latent_action_decoder``: maps DiT hidden states back to latent velocity.
  - ``latent_position_embedding``: positional embedding for latent tokens.
  - ``latent_norm_out``: final LayerNorm for the latent stream (action keeps
    ``self.model.norm_out``).

Latent-action targets are Sharla post-quantize / pre-``output_proj`` embeddings;
the framework computes (and optionally normalises) them and passes
``latent_targets``.
"""

import torch

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    ActionEncoder,
    MLP,
)
from starVLA.model.modules.action_model.LayerwiseMetaqueryFM_ActionHeader import (
    LayerwiseMetaqueryFlowmatchingActionHead,
)


def build_state_action_block_mask(action_horizon: int) -> torch.Tensor:
    """pi0-style block-causal mask for ``[state | actions]``.

    Returns a bool tensor of shape ``[1 + H, 1 + H]`` where ``True`` means the
    query may attend to the key.  State sits in block 1 and actions in block 2,
    so state cannot see actions while actions can see state and all actions.
    """
    horizon = int(action_horizon)
    att_masks = torch.zeros(1 + horizon, dtype=torch.bool)
    att_masks[0] = True  # state opens block 1
    if horizon > 0:
        att_masks[1] = True  # first action opens block 2
    cumsum = torch.cumsum(att_masks.long(), dim=0)
    return cumsum[None, :] <= cumsum[:, None]


class LayerwiseMetaqueryWMFlowmatchingActionHead(LayerwiseMetaqueryFlowmatchingActionHead):
    """Metaquery-conditioned head with decoupled latent-action + action flows."""

    def __init__(self, global_config, **kwargs):
        super().__init__(global_config, **kwargs)
        action_config = global_config.framework.action_model
        latent_action_dim = action_config.get("latent_action_dim", None)
        if latent_action_dim is None:
            raise ValueError(
                "LayerwiseMetaqueryWMFlowmatchingActionHead requires "
                "framework.action_model.latent_action_dim (Sharla codebook_dim)."
            )
        self.latent_action_dim = int(latent_action_dim)

        # Independent encoder (bundles its own sinusoidal time embedding) and decoder.
        self.latent_action_encoder = ActionEncoder(
            action_dim=self.latent_action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.latent_action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.latent_action_dim,
        )
        if self.config.add_pos_embed:
            self.latent_position_embedding = torch.nn.Embedding(
                action_config.max_seq_len, self.input_embedding_dim
            )
            torch.nn.init.normal_(self.latent_position_embedding.weight, mean=0.0, std=0.02)

        # Final norms are branch-specific: action keeps ``self.model.norm_out``
        # (from the metaquery parent); latent gets its own LayerNorm so the two
        # streams do not share affine parameters.
        self.latent_norm_out = torch.nn.LayerNorm(
            self.model.inner_dim, elementwise_affine=True, eps=1e-6
        )

        # Fixed [state | action] block-causal mask; expanded to batch at call time.
        self.register_buffer(
            "state_action_block_mask",
            build_state_action_block_mask(self.action_horizon),
            persistent=False,
        )

    def _embed_latent(self, noisy_latent, t_discretized, device):
        latent_features = self.latent_action_encoder(noisy_latent, t_discretized)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(latent_features.shape[1], dtype=torch.long, device=device)
            latent_features = latent_features + self.latent_position_embedding(pos_ids).unsqueeze(0)
        return latent_features

    def _expand_state_action_mask(self, batch_size: int, device) -> torch.Tensor:
        """Broadcast the cached ``[1+H, 1+H]`` mask to ``(B, 1+H, 1+H)``."""
        return self.state_action_block_mask.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)

    def _forward_action(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> dict:
        """Independent flow-matching over ``[state, action]`` with block-causal mask."""
        device = actions.device
        vl_embs_list, encoder_attention_mask, query_repeat = self._prepare_cross_attention_inputs(
            actions.shape[0], vl_embs_list, encoder_attention_mask
        )

        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        state_features = self.state_encoder(state) if state is not None else None
        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        if state_features is not None:
            sa_embs = torch.cat((state_features, action_features), dim=1)
            self_attention_mask = self._expand_state_action_mask(actions.shape[0], device)
        else:
            sa_embs = action_features
            self_attention_mask = None

        temb = self.model.timestep_encoder(t_discretized)
        hidden_states = self._apply_layerwise_cross_attention(
            sa_embs,
            vl_embs_list,
            temb,
            encoder_attention_mask=encoder_attention_mask,
            query_repeat=query_repeat,
            self_attention_mask=self_attention_mask,
        )
        pred_velocity = self._process_output(hidden_states, actions.shape[1])
        return {"action_loss": ((pred_velocity - velocity) ** 2).mean()}

    def _forward_latent(
        self,
        vl_embs_list: list,
        latent_targets: torch.Tensor,
        encoder_attention_mask=None,
    ) -> dict:
        """Independent flow-matching over latent tokens only (no state / action)."""
        device = latent_targets.device
        vl_embs_list, encoder_attention_mask, query_repeat = self._prepare_cross_attention_inputs(
            latent_targets.shape[0], vl_embs_list, encoder_attention_mask
        )

        noise = torch.randn(latent_targets.shape, device=device, dtype=latent_targets.dtype)
        t = self.sample_time(latent_targets.shape[0], device=device, dtype=latent_targets.dtype)
        t = t[:, None, None]
        noisy_latent = (1 - t) * noise + t * latent_targets
        velocity = latent_targets - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        latent_features = self._embed_latent(noisy_latent, t_discretized, device)
        temb = self.model.timestep_encoder(t_discretized)
        hidden_states = self._apply_layerwise_cross_attention(
            latent_features,
            vl_embs_list,
            temb,
            encoder_attention_mask=encoder_attention_mask,
            query_repeat=query_repeat,
            self_attention_mask=None,
        )
        hidden_states = self.latent_norm_out(hidden_states)
        pred_velocity = self.latent_action_decoder(hidden_states)
        return {"latent_loss": ((pred_velocity - velocity) ** 2).mean()}

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor = None,
        state: torch.Tensor = None,
        latent_targets: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> dict:
        """Decoupled flow-matching: action and/or latent streams run independently.

        Any of ``actions`` / ``latent_targets`` may be ``None``:
          - both given  -> two separate DiT forwards, both losses returned;
          - latent only -> ``latent_loss`` only;
          - action only -> ``action_loss`` only.

        Returns a dict with ``action_loss`` and/or ``latent_loss``.
        """
        if actions is None and latent_targets is None:
            raise ValueError("At least one of `actions` or `latent_targets` must be provided.")

        outputs: dict = {}
        if actions is not None:
            outputs.update(
                self._forward_action(
                    vl_embs_list,
                    actions,
                    state=state,
                    encoder_attention_mask=encoder_attention_mask,
                )
            )
        if latent_targets is not None:
            outputs.update(
                self._forward_latent(
                    vl_embs_list,
                    latent_targets,
                    encoder_attention_mask=encoder_attention_mask,
                )
            )
        return outputs

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """Denoise ``[state, action]`` from noise; latent is not involved at inference."""
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        dtype = vl_embs_list[0].dtype

        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )
        state_features = self.state_encoder(state) if state is not None else None
        self_attention_mask = (
            self._expand_state_action_mask(batch_size, device) if state_features is not None else None
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            if state_features is not None:
                sa_embs = torch.cat((state_features, action_features), dim=1)
            else:
                sa_embs = action_features

            temb = self.model.timestep_encoder(timesteps_tensor)
            hidden_states = self._apply_layerwise_cross_attention(
                sa_embs,
                vl_embs_list,
                temb,
                encoder_attention_mask=encoder_attention_mask,
                self_attention_mask=self_attention_mask,
            )
            pred_velocity = self._process_output(hidden_states, self.action_horizon)
            actions = actions + dt * pred_velocity

        return actions


def get_action_model(config=None):
    """Factory mirroring LayerwiseMetaqueryFM_ActionHeader.get_action_model."""
    return LayerwiseMetaqueryWMFlowmatchingActionHead(global_config=config)
