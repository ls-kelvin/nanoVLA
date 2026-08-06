# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Layer-wise cross-DiT flow-matching action head for the *metaquery* conditioning
variant (ported from Mantis_flow_depth_use_metaquery/models/action_model).

This head is identical in construction to ``LayerwiseFlowmatchingActionHead``
(it subclasses it and reuses ``__init__`` / encoders / DiT / decoder), but its
``forward`` / ``predict_action`` use the **interleave-aware** cross-attention
loop required when ``diffusion_model_cfg.cross_attention_dim != input_embedding_dim``
together with ``interleave_self_attention=true``:

  - even DiT blocks (built with ``cross_attention_dim = VLM hidden``) cross-attend
    to the per-layer metaquery embeddings;
  - odd DiT blocks (built with ``cross_attention_dim = None``) self-attend only
    (``encoder_hidden_states=None``).

The plain ``LayerwiseFlowmatchingActionHead.forward`` passes ``encoder_hidden_states``
to *every* block, which would crash the odd self-attn blocks when the encoder dim
differs from the DiT inner dim.  That is why this variant exists.

Other Mantis-faithful details kept here:
  - ``sample_time`` uses ``noise_s * (1 - beta_sample)`` (Mantis formula).
  - output processing applies ``self.model.norm_out`` before ``action_decoder``.
  - no register/future tokens (``num_target_vision_tokens=0`` -> ``future_tokens=None``).
  - plain-mean flow-matching loss (single-embodiment: == Mantis masked loss).
"""

import torch

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
)


class LayerwiseMetaqueryFlowmatchingActionHead(LayerwiseFlowmatchingActionHead):
    """Metaquery-conditioned, interleave-aware flow-matching action head."""

    def __init__(self, global_config, **kwargs):
        super().__init__(global_config, **kwargs)
        # Mantis' DiT builds norm_out with a learnable affine
        # (``elementwise_affine=True``), but the shared starVLA DiT builds it
        # with ``elementwise_affine=False``.  Swap it here (subclass-only, no
        # change to the parent head or the DiT class) so ``_process_output``
        # matches Mantis exactly.  Fresh LayerNorm init is identity-affine, so
        # this only adds learnable scale/shift on top of the same normalization.
        self.model.norm_out = torch.nn.LayerNorm(
            self.model.inner_dim, elementwise_affine=True, eps=1e-6
        )
        self.model.proj_out_1 = None
        self.model.proj_out_2 = None

        # Cross-attention rows are independent, so ``repeated_diffusion_steps``
        # copies of one VL example can share a single set of projected K/V as
        # long as every other op in the block is position-wise.  ``ada_norm``
        # (per-batch-element modulation) and sinusoidal ``pos_embed`` (indexed by
        # sequence position) both break that, so fall back to materialising the
        # repeats for those DiT variants.
        self._cross_attention_foldable = all(
            block.norm_type != "ada_norm" and block.pos_embed is None
            for block in self.model.transformer_blocks
        )

    def sample_time(self, batch_size, device, dtype):
        # Mantis formula: noise_s * (1 - beta_sample)
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return self.config.noise_s * (1 - sample)

    @staticmethod
    def _fold_query_repeats(hidden_states: torch.Tensor, repeat: int) -> torch.Tensor:
        """``(repeat*B, T, D)`` in ``Tensor.repeat(repeat, 1, 1)`` order -> ``(B, repeat*T, D)``."""
        total_batch, seq_len, dim = hidden_states.shape
        batch = total_batch // repeat
        return (
            hidden_states.reshape(repeat, batch, seq_len, dim)
            .transpose(0, 1)
            .reshape(batch, repeat * seq_len, dim)
        )

    @staticmethod
    def _unfold_query_repeats(hidden_states: torch.Tensor, repeat: int) -> torch.Tensor:
        """Inverse of :meth:`_fold_query_repeats`."""
        batch, folded_len, dim = hidden_states.shape
        seq_len = folded_len // repeat
        return (
            hidden_states.reshape(batch, repeat, seq_len, dim)
            .transpose(0, 1)
            .reshape(repeat * batch, seq_len, dim)
        )

    def _resolve_query_repeat(self, query_batch: int, vl_batch: int) -> int:
        if query_batch == vl_batch:
            return 1
        if vl_batch <= 0 or query_batch % vl_batch != 0:
            raise ValueError(
                f"Query batch {query_batch} must be a multiple of the VL batch {vl_batch}."
            )
        return query_batch // vl_batch

    def _materialize_query_repeats(self, vl_embs_list, encoder_attention_mask, query_repeat):
        """Expand VL states to the query batch for DiT variants that cannot fold."""
        vl_embs_list = [hidden.repeat(query_repeat, 1, 1) for hidden in vl_embs_list]
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.repeat(query_repeat, 1)
        return vl_embs_list, encoder_attention_mask, 1

    def _prepare_cross_attention_inputs(self, query_batch, vl_embs_list, encoder_attention_mask):
        query_repeat = self._resolve_query_repeat(query_batch, vl_embs_list[0].shape[0])
        if query_repeat > 1 and not self._cross_attention_foldable:
            return self._materialize_query_repeats(vl_embs_list, encoder_attention_mask, query_repeat)
        return vl_embs_list, encoder_attention_mask, query_repeat

    def _apply_layerwise_cross_attention(
        self,
        sa_embs,
        vl_embs_list,
        temb,
        encoder_attention_mask=None,
        query_repeat=1,
        self_attention_mask=None,
    ):
        """Interleave self/cross attention across DiT blocks (Mantis semantics).

        ``query_repeat > 1`` means ``sa_embs`` carries that many diffusion
        samples per VL example while ``vl_embs_list`` is still at the
        un-repeated batch.  The repeats are folded into the query sequence for
        cross-attention blocks so K/V are projected once instead of once per
        repeat; self-attention blocks keep the unfolded layout so they never
        attend across repeats.

        ``self_attention_mask`` is forwarded only to the interleaved self-attn
        blocks (odd layers when ``interleave_self_attention`` is on). Default
        ``None`` preserves the previous unmasked self-attention behaviour.
        """
        hidden_states = sa_embs
        interleave = self.model.config.interleave_self_attention
        for layer_idx, block in enumerate(self.model.transformer_blocks):
            if layer_idx % 2 == 1 and interleave:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=self_attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            elif query_repeat > 1:
                folded = block(
                    hidden_states=self._fold_query_repeats(hidden_states, query_repeat),
                    encoder_hidden_states=vl_embs_list[layer_idx],
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
                hidden_states = self._unfold_query_repeats(folded, query_repeat)
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=vl_embs_list[layer_idx],
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
        return hidden_states

    def _process_output(self, hidden_states, actions_length):
        action_features = self.model.norm_out(hidden_states)
        pred = self.action_decoder(action_features)
        return pred[:, -actions_length:]

    def _embed_actions(self, noisy_or_clean_actions, t_discretized, state_features, device):
        action_features = self.action_encoder(noisy_or_clean_actions, t_discretized)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        if state_features is not None:
            return torch.cat((state_features, action_features), dim=1)
        return action_features

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ):
        """
        vl_embs_list: list of (Bvl, seq_length, vl_hidden) per DiT layer.
        actions:      (B, action_horizon, action_dim).
        state:        (B, 1, state_dim) or None.
        encoder_attention_mask: optional (Bvl, seq_length) bool/int mask for padded VL tokens.

        ``B`` may be an integer multiple of ``Bvl`` when the caller stacks
        several diffusion samples per VL example with ``Tensor.repeat``; the VL
        states are then shared across the repeats instead of being copied.
        """
        device = actions.device
        vl_embs_list, encoder_attention_mask, query_repeat = self._prepare_cross_attention_inputs(
            actions.shape[0], vl_embs_list, encoder_attention_mask
        )

        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)
        t = t[:, None, None]  # (B,1,1)

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        state_features = self.state_encoder(state) if state is not None else None
        sa_embs = self._embed_actions(noisy_trajectory, t_discretized, state_features, device)

        temb = self.model.timestep_encoder(t_discretized)
        hidden_states = self._apply_layerwise_cross_attention(
            sa_embs,
            vl_embs_list,
            temb,
            encoder_attention_mask=encoder_attention_mask,
            query_repeat=query_repeat,
        )
        pred_velocity = self._process_output(hidden_states, actions.shape[1])

        # Plain mean: == Mantis masked loss when action_mask is all-ones (single embodiment).
        loss = ((pred_velocity - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
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

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            sa_embs = self._embed_actions(actions, timesteps_tensor, state_features, device)
            temb = self.model.timestep_encoder(timesteps_tensor)
            hidden_states = self._apply_layerwise_cross_attention(
                sa_embs,
                vl_embs_list,
                temb,
                encoder_attention_mask=encoder_attention_mask,
            )
            pred_velocity = self._process_output(hidden_states, self.action_horizon)

            actions = actions + dt * pred_velocity
        return actions


def get_action_model(config=None):
    """Factory mirroring LayerwiseFM_ActionHeader.get_action_model."""
    return LayerwiseMetaqueryFlowmatchingActionHead(global_config=config)
