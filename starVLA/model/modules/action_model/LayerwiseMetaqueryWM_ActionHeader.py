# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Layer-wise cross-DiT flow-matching action head with an in-head latent-action
branch ("world-model" variant, ``QwenWM_LA``).

This head subclasses :class:`LayerwiseMetaqueryFlowmatchingActionHead` and adds a
second flow-matching stream for *latent actions*.  The latent-action tokens are
prepended before ``[state, action]`` so a single DiT pass jointly denoises the
sequence ``[latent, state, action]`` with **one shared diffusion timestep**.

Independent (non-shared with the action stream) submodules are added:
  - ``latent_action_encoder``: embeds noisy latent tokens (its own internal
    sinusoidal time embedding -> the "independent time embedding").
  - ``latent_action_decoder``: maps DiT hidden states back to latent velocity.
  - ``latent_position_embedding``: positional embedding for latent tokens.

The latent-action targets are the UniT pre-quantization continuous embeddings
(``before_quant``); the framework computes them and passes ``latent_targets``.
"""

import torch

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    ActionEncoder,
    MLP,
)
from starVLA.model.modules.action_model.LayerwiseMetaqueryFM_ActionHeader import (
    LayerwiseMetaqueryFlowmatchingActionHead,
)


class LayerwiseMetaqueryWMFlowmatchingActionHead(LayerwiseMetaqueryFlowmatchingActionHead):
    """Metaquery-conditioned head with a joint latent-action + action flow."""

    def __init__(self, global_config, **kwargs):
        super().__init__(global_config, **kwargs)
        action_config = global_config.framework.action_model
        latent_action_dim = action_config.get("latent_action_dim", None)
        if latent_action_dim is None:
            raise ValueError(
                "LayerwiseMetaqueryWMFlowmatchingActionHead requires "
                "framework.action_model.latent_action_dim (UniT VQ e_dim)."
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

    def _embed_latent(self, noisy_latent, t_discretized, device):
        latent_features = self.latent_action_encoder(noisy_latent, t_discretized)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(latent_features.shape[1], dtype=torch.long, device=device)
            latent_features = latent_features + self.latent_position_embedding(pos_ids).unsqueeze(0)
        return latent_features

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor = None,
        state: torch.Tensor = None,
        latent_targets: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> dict:
        """Joint flow-matching over ``[latent, state, action]`` with a shared timestep.

        Any of ``actions`` / ``latent_targets`` may be ``None``:
          - both given  -> joint sequence ``[latent, state, action]``;
          - latent only -> sequence ``[latent]``;
          - action only -> sequence ``[state, action]`` (parent behaviour).

        Returns a dict with ``action_loss`` and/or ``latent_loss``.
        """
        if actions is None and latent_targets is None:
            raise ValueError("At least one of `actions` or `latent_targets` must be provided.")

        reference = actions if actions is not None else latent_targets
        device = reference.device
        batch_size = reference.shape[0]

        # One shared diffusion timestep for both streams.
        t = self.sample_time(batch_size, device=device, dtype=reference.dtype)
        t_broadcast = t[:, None, None]
        t_discretized = (t * self.num_timestep_buckets).long()

        parts = []
        num_latent_tokens = 0
        latent_velocity = None
        if latent_targets is not None:
            latent_noise = torch.randn(latent_targets.shape, device=device, dtype=latent_targets.dtype)
            noisy_latent = (1 - t_broadcast) * latent_noise + t_broadcast * latent_targets
            latent_velocity = latent_targets - latent_noise
            latent_features = self._embed_latent(noisy_latent, t_discretized, device)
            num_latent_tokens = latent_features.shape[1]
            parts.append(latent_features)

        state_features = self.state_encoder(state) if state is not None else None
        if state_features is not None:
            parts.append(state_features)

        action_length = 0
        action_velocity = None
        if actions is not None:
            action_noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
            noisy_actions = (1 - t_broadcast) * action_noise + t_broadcast * actions
            action_velocity = actions - action_noise
            action_features = self.action_encoder(noisy_actions, t_discretized)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            action_length = action_features.shape[1]
            parts.append(action_features)

        sa_embs = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        temb = self.model.timestep_encoder(t_discretized)
        hidden_states = self._apply_layerwise_cross_attention(
            sa_embs,
            vl_embs_list,
            temb,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = self.model.norm_out(hidden_states)

        outputs: dict = {}
        if latent_targets is not None:
            latent_hidden = hidden_states[:, :num_latent_tokens]
            pred_latent_velocity = self.latent_action_decoder(latent_hidden)
            outputs["latent_loss"] = ((pred_latent_velocity - latent_velocity) ** 2).mean()
        if actions is not None:
            action_hidden = hidden_states[:, -action_length:]
            pred_action_velocity = self.action_decoder(action_hidden)
            outputs["action_loss"] = ((pred_action_velocity - action_velocity) ** 2).mean()
        return outputs

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        num_latent_tokens: int = 0,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """Jointly denoise ``[latent, state, action]`` from noise; return only actions."""
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        dtype = vl_embs_list[0].dtype

        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )
        latents = None
        if num_latent_tokens > 0:
            latents = torch.randn(
                size=(batch_size, int(num_latent_tokens), self.latent_action_dim),
                dtype=dtype,
                device=device,
            )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self.state_encoder(state) if state is not None else None

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            parts = []
            if latents is not None:
                parts.append(self._embed_latent(latents, timesteps_tensor, device))
            if state_features is not None:
                parts.append(state_features)
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            parts.append(action_features)

            sa_embs = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
            temb = self.model.timestep_encoder(timesteps_tensor)
            hidden_states = self._apply_layerwise_cross_attention(
                sa_embs,
                vl_embs_list,
                temb,
                encoder_attention_mask=encoder_attention_mask,
            )
            hidden_states = self.model.norm_out(hidden_states)

            pred_action_velocity = self.action_decoder(hidden_states[:, -self.action_horizon :])
            actions = actions + dt * pred_action_velocity
            if latents is not None:
                pred_latent_velocity = self.latent_action_decoder(hidden_states[:, : int(num_latent_tokens)])
                latents = latents + dt * pred_latent_velocity

        return actions


def get_action_model(config=None):
    """Factory mirroring LayerwiseMetaqueryFM_ActionHeader.get_action_model."""
    return LayerwiseMetaqueryWMFlowmatchingActionHead(global_config=config)
