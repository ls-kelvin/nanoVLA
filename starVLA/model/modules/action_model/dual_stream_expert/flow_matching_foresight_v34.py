# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv34_LA foresight: V33 plus codebook latent tokens for state/action.

Logical suffix order is ``[query(N) | latent(N) | state(1) | action(chunk)]``:

  query pass:  [query(N)]                     bidirectional, no AdaRMSNorm FiLM;
                                              only produces the latent readout
                                              loss -- nothing downstream attends
                                              to query, so no stop-gradient is
                                              needed (there is no attention edge).
  action pass: [latent(N), state(1), action(chunk)]  pi0 block-causal, time-FiLM,
                                              attends to the VLM prefix only.
  latent-only: [query(N)]                     fully bidirectional, no FiLM.

Latent tokens are ground-truth codebook latents mapped to the expert hidden
size: ``latent_in_proj(soft_weights @ F.normalize(sharla_codebook))`` for
soft_kl targets, or ``latent_in_proj(continuous_targets)`` for embedding
targets. Codebook rows are L2-normalized, matching the tokenizer's SoftVQ
forward (``weights @ F.normalize(codebook)``) and therefore the continuous
embedding targets' space. The codebook is injected by the framework as a
frozen, non-persistent buffer (``set_latent_codebook``).

With ``action_model.latent_drop_prob > 0``, each (repeated) sample
independently drops its latent tokens: those rows run a ``[state | action]``
suffix that cannot see latent at all (CFG-style dropout, exact per-sample
semantics via a batch-split two-pass).

Inference is two-stage: the query pass predicts the latent (softmax over
codebook logits for soft_kl, direct embedding otherwise), which is mapped
through the same projection and held fixed across the Euler denoising loop.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .flow_matching_foresight_v33 import DualStreamFlowMatchingForesightV33


class DualStreamFlowMatchingForesightV34(DualStreamFlowMatchingForesightV33):
    """V33 with GT-codebook latent tokens conditioning state/action."""

    def __init__(self, global_config):
        super().__init__(global_config)
        action_cfg = global_config.framework.action_model
        # Probability of dropping the latent tokens for a (repeated) sample, so
        # state/action are trained to also denoise without latent conditioning.
        self.latent_drop_prob = float(action_cfg.get("latent_drop_prob", 0.0))
        if not 0.0 <= self.latent_drop_prob < 1.0:
            raise ValueError(
                f"framework.action_model.latent_drop_prob must be in [0, 1), got {self.latent_drop_prob}."
            )
        # Injected by the framework via set_latent_codebook; non-persistent so
        # checkpoints never store the frozen Sharla codebook.
        self.register_buffer("latent_codebook", None, persistent=False)

    def set_latent_codebook(self, codebook: Tensor) -> None:
        """Register the row-normalized Sharla codebook ``[C, latent_dim]``.

        Row L2-normalization matches the tokenizer's SoftVQ forward, whose
        quantized embeddings are ``weights @ F.normalize(codebook.weight)``.
        """
        if codebook.ndim != 2:
            raise ValueError(
                f"latent_codebook must be [codebook_size, latent_dim], got {tuple(codebook.shape)}."
            )
        expected_size = self.codebook_size if self.codebook_size is not None else codebook.shape[0]
        if codebook.shape[0] != expected_size or codebook.shape[1] != self.latent_action_dim:
            raise ValueError(
                f"latent_codebook shape {tuple(codebook.shape)} does not match "
                f"[codebook_size={expected_size}, latent_dim={self.latent_action_dim}]."
            )
        self.latent_codebook = F.normalize(codebook.detach().float(), dim=-1).to(
            device=self.latent_in_proj.weight.device
        )

    def _require_codebook(self) -> Tensor:
        if self.latent_codebook is None:
            raise RuntimeError(
                "latent_codebook is not set; the framework must call set_latent_codebook() "
                "with the Sharla quantizer codebook before running QwenWMv34_LA."
            )
        return self.latent_codebook

    def _embed_codebook_latents(self, latent_targets: Tensor) -> Tensor:
        """Map codebook latents to hidden-size token embeddings (inputs, detached)."""
        latent_targets = latent_targets.detach().float()
        if self.foresight_latent_loss_type == "soft_kl":
            codebook = self._require_codebook()
            latent_cont = latent_targets @ codebook
        else:
            latent_cont = latent_targets
        return self.latent_in_proj(latent_cont)

    # -- action pass: [latent(N), state(1), action_time(chunk)] -------------
    def embed_suffix_latent_state_action(
        self, latent_emb: Tensor, state: Tensor, noisy_actions: Tensor, timestep: Tensor
    ):
        """Latent + state + action suffix. Three pi0 blocks (not flash-compatible)."""
        if state.ndim == 3:
            state = state.squeeze(1)
        bsize = state.shape[0]
        device = state.device
        state = state.float()
        noisy_actions = noisy_actions.float()
        timestep = timestep.float()

        state_emb = self.state_proj(state)
        time_emb, action_time_emb, _, _ = self.embed_suffix_action(noisy_actions, timestep)
        action_len = action_time_emb.shape[1]

        latent_emb = latent_emb.float()
        num_lt = latent_emb.shape[1]
        embs = torch.cat([latent_emb, state_emb[:, None], action_time_emb], dim=1)
        suffix_len = num_lt + 1 + action_len
        pad_masks = torch.ones((bsize, suffix_len), device=device, dtype=torch.bool)
        # Blocks: latent | state | action (first token of each opens a block).
        att_masks = torch.zeros((bsize, suffix_len), device=device, dtype=torch.bool)
        att_masks[:, 0] = True
        att_masks[:, num_lt] = True
        att_masks[:, num_lt + 1] = True
        return time_emb, embs, pad_masks, att_masks

    def run_expert_latent_state_action(
        self,
        prefix: dict,
        prefix_kvs,
        latent_emb: Tensor,
        state: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Action pass over ``[latent | state | action]`` attending to the prefix only.

        Query tokens are absent from this pass, so position ids continue right
        after the prompt (no offset). Attention is block-causal over three
        blocks, so flash is rejected.
        """
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = (
            self.embed_suffix_latent_state_action(latent_emb, state, x_t, timestep)
        )
        suffix_position_ids = self._build_suffix_position_ids(
            prefix["position_ids"], prefix["prompt_pad_masks"], suffix_pad_masks
        )
        suffix_out = self.qwenvl_with_expert.run_expert(
            suffix_embs=suffix_embs,
            prefix_kvs=prefix_kvs,
            suffix_position_ids=suffix_position_ids,
            prefix_pad_masks=prefix["prompt_pad_masks"],
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            ada_cond=time_embs if self.adanorm_time else None,
            attention_implementation=self._resolve_block_causal_attention(attention_implementation),
        )
        return suffix_out[:, -x_t.shape[1] :]

    # -- latent drop (CFG-style) --------------------------------------------
    def _latent_drop_mask(self, bsize: int, device) -> Optional[Tensor]:
        """Per-row drop decisions; None disables the split path entirely."""
        if self.latent_drop_prob <= 0.0 or not self.training:
            return None
        return torch.rand(bsize, device=device) < self.latent_drop_prob

    @staticmethod
    def _select_prefix_rows(prefix: dict, prefix_kvs, rows: Tensor):
        sub_prefix = {
            **prefix,
            "position_ids": prefix["position_ids"][:, rows],
            "prompt_pad_masks": prefix["prompt_pad_masks"][rows],
        }
        sub_kvs = [(key[rows], value[rows]) for key, value in prefix_kvs]
        return sub_prefix, sub_kvs

    def _run_action_pass_with_latent_drop(
        self,
        prefix: dict,
        prefix_kvs,
        latent_emb: Tensor,
        state: Tensor,
        x_t: Tensor,
        time: Tensor,
        drop_mask: Optional[Tensor],
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Action pass where dropped rows run ``[state | action]`` (no latent).

        Dropped rows use a separate expert forward whose suffix contains no
        latent tokens at all, so state/action structurally cannot attend to
        them; kept rows run the usual ``[latent | state | action]`` pass.
        """
        if drop_mask is None or not bool(drop_mask.any()):
            return self.run_expert_latent_state_action(
                prefix, prefix_kvs, latent_emb, state, x_t, time,
                attention_implementation=attention_implementation,
            )
        if bool(drop_mask.all()):
            return self.run_expert_state_action(
                prefix, prefix_kvs, prefix["prompt_pad_masks"], state, x_t, time,
                attention_implementation=attention_implementation,
                position_offset=0,
            )
        keep = ~drop_mask
        action_out = torch.empty(
            x_t.shape[0], x_t.shape[1], self.proj_width, device=x_t.device, dtype=torch.float32
        )
        keep_prefix, keep_kvs = self._select_prefix_rows(prefix, prefix_kvs, keep)
        action_out[keep] = self.run_expert_latent_state_action(
            keep_prefix, keep_kvs, latent_emb[keep], state[keep], x_t[keep], time[keep],
            attention_implementation=attention_implementation,
        )
        drop_prefix, drop_kvs = self._select_prefix_rows(prefix, prefix_kvs, drop_mask)
        action_out[drop_mask] = self.run_expert_state_action(
            drop_prefix, drop_kvs, drop_prefix["prompt_pad_masks"],
            state[drop_mask], x_t[drop_mask], time[drop_mask],
            attention_implementation=attention_implementation,
            position_offset=0,
        )
        return action_out

    # -- training: joint ----------------------------------------------------
    def flow_matching_loss_joint_foresight(
        self,
        prefix: dict,
        prefix_kvs,
        state: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        latent_targets: Tensor,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        num_repeats: Optional[int] = None,
        attention_implementation: Optional[str] = None,
        latent_valid_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Query pass (latent loss) + GT-latent-conditioned action pass.

        The two passes share only the VLM prefix K/V; there is no attention
        edge from state/action into query, and latent token inputs are
        detached ground-truth targets (teacher forcing).
        """
        actions = actions.float()
        action_mask = action_mask.float()
        latent_targets = latent_targets.float()
        self._check_latent_token_count(latent_targets)
        device = actions.device
        bsize = actions.shape[0]
        repeats = int(self.repeated_diffusion_steps if num_repeats is None else num_repeats)
        if repeats < 1:
            raise ValueError(f"num_repeats must be >= 1, got {repeats}.")

        if actions.shape[0] != prefix["prompt_pad_masks"].shape[0]:
            raise ValueError(
                f"actions batch {actions.shape[0]} must match prefix batch "
                f"{prefix['prompt_pad_masks'].shape[0]} before repeats."
            )
        if latent_targets.shape[0] != bsize:
            raise ValueError(
                f"latent_targets batch {latent_targets.shape[0]} must match actions batch {bsize}."
            )

        with torch.autocast("cuda", dtype=torch.float32):
            query_out = self.run_expert_learnable_only(
                prefix, prefix_kvs, attention_implementation=attention_implementation
            )
            latent_loss = self._latent_loss_from_learnable(
                query_out, latent_targets, valid_mask=latent_valid_mask
            )

            latent_emb = self._embed_codebook_latents(latent_targets)

            if state is None:
                state = torch.zeros(
                    bsize, self.max_state_dim, device=device, dtype=torch.float32
                )
            elif state.ndim == 3:
                state = state.squeeze(1)
            if state.shape[0] != bsize:
                raise ValueError(
                    f"state batch {state.shape[0]} must match actions batch {bsize} before repeats."
                )

            if repeats > 1:
                actions = actions.repeat(repeats, 1, 1)
                action_mask = action_mask.repeat(repeats, 1, 1)
                state = state.repeat(repeats, 1)
                latent_emb = latent_emb.repeat(repeats, 1, 1)

            if noise is None:
                noise = torch.randn(actions.shape, device=device, dtype=torch.float32)
            else:
                noise = noise.float()
            if time is None:
                time = self.sample_time(actions.size(0), device)
            else:
                time = time.float()

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            expert_prefix, expert_kvs = self._expand_prefix_for_repeats(prefix, prefix_kvs, repeats)
            drop_mask = self._latent_drop_mask(actions.shape[0], device)
            action_out = self._run_action_pass_with_latent_drop(
                expert_prefix,
                expert_kvs,
                latent_emb,
                state,
                x_t,
                time,
                drop_mask,
                attention_implementation=attention_implementation,
            )

            v_t = self.action_out_proj(action_out).float()
            if self.loss_type == "L1_fm":
                losses = F.l1_loss(u_t, v_t, reduction="none")
            else:
                losses = F.mse_loss(u_t, v_t, reduction="none")
            action_loss = (losses * action_mask).sum() / action_mask.sum().clamp_min(1.0)

        return action_loss, latent_loss

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def _predict_latent_emb(self, prefix: dict, prefix_kvs) -> Tensor:
        """Query pass -> predicted latent -> hidden-size latent token embeddings."""
        suffix_out = self.run_expert_learnable_only(prefix, prefix_kvs)
        if self.foresight_latent_loss_type == "embedding":
            latent_cont = self.learnable_to_latent_proj(suffix_out).float()
        else:
            codebook = self._require_codebook()
            logits = self.learnable_to_logits_proj(suffix_out).float()
            latent_cont = F.softmax(logits, dim=-1) @ codebook
        return self.latent_in_proj(latent_cont)

    @torch.no_grad()
    def sample_actions(
        self,
        inputs: dict,
        state: Tensor,
        noise: Optional[Tensor] = None,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Two-stage Euler sampling: predict latents once, then denoise actions."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            noise = torch.randn(
                (bsize, self.n_action_steps, self.max_action_dim), device=device, dtype=torch.float32
            )
        else:
            noise = noise.float()

        prefix, _, prefix_kvs = self.encode_prefix_context(inputs, num_latent_tokens=0)

        with torch.autocast("cuda", dtype=torch.float32):
            latent_emb = self._predict_latent_emb(prefix, prefix_kvs)

            dt = -1.0 / self.num_steps
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                action_out = self.run_expert_latent_state_action(
                    prefix,
                    prefix_kvs,
                    latent_emb,
                    state,
                    x_t,
                    expanded_time,
                    attention_implementation=attention_implementation,
                )
                v_t = self.action_out_proj(action_out).float()
                x_t = x_t + dt * v_t
                time = time + dt
        return x_t
