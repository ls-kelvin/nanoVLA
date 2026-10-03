# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Joint latent-action flow matching on top of ``DualStreamFlowMatchingWM`` (WMv21).

Unlike WMv2 (``DualStreamFlowMatchingWM``), where latent and action are two
**independent** flow-matching streams with separate forward passes, here they
share ONE suffix and ONE denoising trajectory:

  joint suffix: [state(1), latent(N, noisy), action(chunk, noisy)]

Block-causal semantics (see ``joint_attention.make_att_2d_masks``): the state
token opens block 1 and the first latent token opens block 2, so

  - state attends to itself only (it never sees latent/action);
  - latent and action tokens share block 2: they attend to each other
    bidirectionally AND to the state block.

Both streams are noised with the SAME flow-matching timestep (independent
noise tensors), so training runs a single expert pass and inference runs a
single Euler loop that denoises latent and action together (shared number of
denoising steps).

The latent-only path (dataloaders whose robot types are excluded from action
training) still uses WMv2's independent latent suffix
(``flow_matching_loss_latent``), mirroring WMv3's joint/latent-only split.
"""

from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import Tensor

from .flow_matching_wm import DualStreamFlowMatchingWM
from .joint_attention import create_sinusoidal_pos_embedding


class DualStreamFlowMatchingJointLA(DualStreamFlowMatchingWM):
    """``DualStreamFlowMatchingWM`` with a joint state|LA|action denoising suffix."""

    # -- joint suffix: [state, latent_time, action_time] -------------------
    def embed_suffix_joint(self, state, noisy_latent, noisy_actions, timestep):
        """Three-segment suffix; state is block 1, latent+action share block 2."""
        if state.ndim == 3:
            state = state.squeeze(1)
        bsize = state.shape[0]
        device = state.device
        # Expert path is fp32.
        state = state.float()
        noisy_latent = noisy_latent.float()
        noisy_actions = noisy_actions.float()
        timestep = timestep.float()

        state_emb = self.state_proj(state)

        # One shared timestep embedding conditions both denoising streams.
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.proj_width, min_period=4e-3, max_period=4.0, device=device
        ).to(dtype=torch.float32)

        latent_emb = self.latent_in_proj(noisy_latent)
        latent_time_rep = einops.repeat(time_emb, "b d -> b n d", n=latent_emb.shape[1])
        latent_time_emb = torch.cat([latent_emb, latent_time_rep], dim=-1)
        latent_time_emb = F.silu(self.latent_time_mlp_in(latent_time_emb))
        latent_time_emb = self.latent_time_mlp_out(latent_time_emb)
        latent_len = latent_time_emb.shape[1]

        action_emb = self.action_in_proj(noisy_actions)
        action_time_rep = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
        action_time_emb = torch.cat([action_emb, action_time_rep], dim=-1)
        action_time_emb = F.silu(self.action_time_mlp_in(action_time_emb))
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_len = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], latent_time_emb, action_time_emb], dim=1)
        suffix_len = 1 + latent_len + action_len
        pad_masks = torch.ones((bsize, suffix_len), device=device, dtype=torch.bool)
        # Blocks: state | latent+action. state opens block 1; the first latent
        # token opens block 2 which covers latent AND action, so they attend to
        # each other (and to state) while state attends to itself only.
        att_masks = torch.zeros((bsize, suffix_len), device=device, dtype=torch.bool)
        att_masks[:, 0] = True
        att_masks[:, 1] = True
        return time_emb, embs, pad_masks, att_masks

    # -- shared expert step (joint stream) --------------------------------
    def run_expert_joint(
        self,
        prefix: dict,
        prefix_kvs,
        state: Tensor,
        x_t_latent: Tensor,
        x_t_action: Tensor,
        timestep: Tensor,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Joint expert forward; returns full suffix hidden states."""
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_joint(
            state, x_t_latent, x_t_action, timestep
        )
        suffix_position_ids = self._build_suffix_position_ids(
            prefix["position_ids"], prefix["prompt_pad_masks"], suffix_pad_masks
        )
        return self.qwenvl_with_expert.run_expert(
            suffix_embs=suffix_embs,
            prefix_kvs=prefix_kvs,
            suffix_position_ids=suffix_position_ids,
            prefix_pad_masks=prefix["prompt_pad_masks"],
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            ada_cond=time_embs if self.adanorm_time else None,
            attention_implementation=attention_implementation,
        )

    @staticmethod
    def _split_joint_outputs(suffix_out: Tensor, latent_len: int, action_len: int):
        """Slice ``[state | latent | action]`` hidden states."""
        latent_out = suffix_out[:, 1 : 1 + latent_len]
        action_out = suffix_out[:, 1 + latent_len : 1 + latent_len + action_len]
        return latent_out, action_out

    # -- training: joint ---------------------------------------------------
    def flow_matching_loss_joint(
        self,
        prefix: dict,
        prefix_kvs,
        state: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        latent_targets: Tensor,
        noise: Optional[Tensor] = None,
        latent_noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        num_repeats: Optional[int] = None,
        attention_implementation: Optional[str] = None,
    ) -> tuple[Tensor, Tensor]:
        """One joint expert pass over ``[state | LA | action]`` -> ``(action_loss, latent_loss)``.

        Latent and action share the same flow-matching timestep (independent
        noise). Unlike the foresight variants, latent tokens are denoised too,
        so under ``repeated_diffusion_steps > 1`` the latent targets are tiled
        along with everything else and every repeated row contributes to both
        losses.
        """
        actions = actions.float()
        action_mask = action_mask.float()
        latent_targets = latent_targets.float()
        device = actions.device
        repeats = int(self.repeated_diffusion_steps if num_repeats is None else num_repeats)
        if repeats < 1:
            raise ValueError(f"num_repeats must be >= 1, got {repeats}.")

        if actions.shape[0] != prefix["prompt_pad_masks"].shape[0]:
            raise ValueError(
                f"actions batch {actions.shape[0]} must match prefix batch "
                f"{prefix['prompt_pad_masks'].shape[0]} before repeats."
            )
        if latent_targets.shape[0] != actions.shape[0]:
            raise ValueError(
                f"latent_targets batch {latent_targets.shape[0]} must match actions batch "
                f"{actions.shape[0]} before repeats."
            )
        if latent_targets.shape[-1] != self.latent_action_dim:
            raise ValueError(
                f"latent_targets last dim {latent_targets.shape[-1]} must equal "
                f"latent_action_dim={self.latent_action_dim}."
            )

        # See DualStreamFlowMatching.flow_matching_loss: force real fp32 compute
        # for the expert's autocast-eligible ops.
        with torch.autocast("cuda", dtype=torch.float32):
            if repeats > 1:
                actions = actions.repeat(repeats, 1, 1)
                action_mask = action_mask.repeat(repeats, 1, 1)
                latent_targets = latent_targets.repeat(repeats, 1, 1)
                if state is not None:
                    state = state.repeat(repeats, 1)

            if state is None:
                state = torch.zeros(
                    actions.shape[0], self.max_state_dim, device=device, dtype=torch.float32
                )

            if noise is None:
                noise = torch.randn(actions.shape, device=device, dtype=torch.float32)
            else:
                noise = noise.float()
            if latent_noise is None:
                latent_noise = torch.randn(latent_targets.shape, device=device, dtype=torch.float32)
            else:
                latent_noise = latent_noise.float()
            if time is None:
                time = self.sample_time(actions.size(0), device)
            else:
                time = time.float()

            time_expanded = time[:, None, None]
            x_t_action = time_expanded * noise + (1 - time_expanded) * actions
            u_t_action = noise - actions
            x_t_latent = time_expanded * latent_noise + (1 - time_expanded) * latent_targets
            u_t_latent = latent_noise - latent_targets

            expert_prefix, expert_kvs = self._expand_prefix_for_repeats(prefix, prefix_kvs, repeats)
            suffix_out = self.run_expert_joint(
                expert_prefix,
                expert_kvs,
                state,
                x_t_latent,
                x_t_action,
                time,
                attention_implementation=attention_implementation,
            )
            latent_out, action_out = self._split_joint_outputs(
                suffix_out, x_t_latent.shape[1], x_t_action.shape[1]
            )

            v_t_action = self.action_out_proj(action_out).float()
            if self.loss_type == "L1_fm":
                losses = F.l1_loss(u_t_action, v_t_action, reduction="none")
            else:
                losses = F.mse_loss(u_t_action, v_t_action, reduction="none")
            action_loss = (losses * action_mask).sum() / action_mask.sum().clamp_min(1.0)

            v_t_latent = self.latent_out_proj(latent_out).float()
            latent_loss = F.mse_loss(u_t_latent, v_t_latent)

        return action_loss, latent_loss

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        inputs: dict,
        state: Tensor,
        noise: Optional[Tensor] = None,
        latent_noise: Optional[Tensor] = None,
        num_latent_tokens: Optional[int] = None,
        attention_implementation: Optional[str] = None,
        extra_embs: Optional[Tensor] = None,
        return_latent: bool = False,
    ):
        """Euler sampling that jointly denoises latent and action (shared steps).

        ``num_latent_tokens`` is required when ``latent_noise`` is not given:
        it sets the latent token count (framework passes its latent query
        count). Returns the action trajectory by default; with
        ``return_latent=True`` returns ``(actions, latents)``.
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            noise = torch.randn(
                (bsize, self.n_action_steps, self.max_action_dim), device=device, dtype=torch.float32
            )
        else:
            noise = noise.float()
        if latent_noise is None:
            if num_latent_tokens is None:
                raise ValueError(
                    "num_latent_tokens is required for joint sampling when latent_noise is not given."
                )
            latent_noise = torch.randn(
                (bsize, int(num_latent_tokens), self.latent_action_dim),
                device=device,
                dtype=torch.float32,
            )
        else:
            latent_noise = latent_noise.float()

        prefix, _, prefix_kvs = self.encode_prefix_context(
            inputs, num_latent_tokens=0, extra_embs=extra_embs
        )

        # See the matching comment in ``forward``: force real fp32 compute for
        # the expert's autocast-eligible ops instead of silently inheriting
        # whatever ambient autocast dtype the caller established.
        with torch.autocast("cuda", dtype=torch.float32):
            dt = -1.0 / self.num_steps
            x_t_action = noise
            x_t_latent = latent_noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                suffix_out = self.run_expert_joint(
                    prefix,
                    prefix_kvs,
                    state,
                    x_t_latent,
                    x_t_action,
                    expanded_time,
                    attention_implementation=attention_implementation,
                )
                latent_out, action_out = self._split_joint_outputs(
                    suffix_out, x_t_latent.shape[1], x_t_action.shape[1]
                )
                v_t_action = self.action_out_proj(action_out).float()
                v_t_latent = self.latent_out_proj(latent_out).float()
                x_t_action = x_t_action + dt * v_t_action
                x_t_latent = x_t_latent + dt * v_t_latent
                time = time + dt
        if return_latent:
            return x_t_action, x_t_latent
        return x_t_action
