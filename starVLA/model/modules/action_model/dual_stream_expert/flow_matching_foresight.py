# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Joint foresight tokens on top of ``DualStreamFlowMatchingWM``.

Replaces the independent latent flow-matching branch with InternVLA-style
learnable tokens that share one expert forward with state + noisy actions:

  joint suffix: [state(1), learnable(N), action_time(chunk)]  (pi0 block-causal)
  latent-only:  [learnable(N)]                                (fully bidirectional)

Latent readout is switchable via ``framework.latent_action.loss_type``:

  - ``embedding``: project learnable hiddens to Sharla embeddings, MSE
  - ``soft_kl``:   project to codebook logits, KL(teacher soft weights || student)

Latent loss in joint mode uses only the first ``B`` rows under
``repeated_diffusion_steps``; latent-only mode never repeats.
"""

from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .flow_matching_wm import DualStreamFlowMatchingWM
from .joint_attention import create_sinusoidal_pos_embedding


class DualStreamFlowMatchingForesight(DualStreamFlowMatchingWM):
    """``DualStreamFlowMatchingWM`` with joint learnable-token foresight readout."""

    def __init__(self, global_config):
        super().__init__(global_config)
        action_cfg = global_config.framework.action_model
        latent_cfg = global_config.framework.latent_action

        num_learnable_tokens = action_cfg.get("num_learnable_tokens", None)
        if num_learnable_tokens is None:
            raise ValueError(
                "DualStreamFlowMatchingForesight requires "
                "framework.action_model.num_learnable_tokens."
            )
        self.num_learnable_tokens = int(num_learnable_tokens)
        if self.num_learnable_tokens < 1:
            raise ValueError(
                f"num_learnable_tokens must be >= 1, got {self.num_learnable_tokens}."
            )

        loss_type = str(latent_cfg.get("loss_type", "embedding")).lower()
        if loss_type not in {"embedding", "soft_kl"}:
            raise ValueError(
                f"framework.latent_action.loss_type must be 'embedding' or 'soft_kl', "
                f"got {loss_type!r}."
            )
        self.foresight_latent_loss_type = loss_type

        self.learnable_tokens = nn.Parameter(
            torch.zeros(self.num_learnable_tokens, self.proj_width)
        )
        nn.init.trunc_normal_(self.learnable_tokens, std=0.02)
        self.learnable_tokens_in_proj = nn.Linear(self.proj_width, self.proj_width)
        # Fixed AdaRMSNorm condition for the latent-only foresight path (no time).
        self.foresight_cond = nn.Parameter(torch.zeros(self.proj_width))

        self.learnable_to_latent_proj = None
        self.learnable_to_logits_proj = None
        self.codebook_size = None
        self.kl_eps = 1e-8

        if self.foresight_latent_loss_type == "embedding":
            self.learnable_to_latent_proj = nn.Linear(self.proj_width, self.latent_action_dim)
        else:
            codebook_size = action_cfg.get("codebook_size", None)
            if codebook_size is None:
                codebook_size = latent_cfg.get("codebook_size", None)
            if codebook_size is None:
                raise ValueError(
                    "soft_kl foresight requires framework.latent_action.codebook_size "
                    "(or action_model.codebook_size)."
                )
            self.codebook_size = int(codebook_size)
            if self.codebook_size < 2:
                raise ValueError(f"codebook_size must be >= 2, got {self.codebook_size}.")
            backend = str(latent_cfg.get("backend", "sharla")).lower()
            backend_cfg = latent_cfg.get(backend, {}) or {}
            self.kl_eps = float(backend_cfg.get("kl_eps", 1e-8))
            self.learnable_to_logits_proj = nn.Linear(self.proj_width, self.codebook_size)

    # -- learnable token embedding ----------------------------------------
    def _embed_learnable_tokens(self, bsize: int, device: torch.device) -> Tensor:
        lt_emb = self.learnable_tokens_in_proj(self.learnable_tokens.float())
        return lt_emb[None].expand(bsize, -1, -1).to(device=device)

    def _check_latent_token_count(self, latent_targets: Tensor) -> None:
        n = int(latent_targets.shape[1])
        if n != self.num_learnable_tokens:
            raise ValueError(
                f"latent_targets length {n} must equal num_learnable_tokens "
                f"{self.num_learnable_tokens} (no silent truncate/pad)."
            )
        if self.foresight_latent_loss_type == "embedding":
            if latent_targets.shape[-1] != self.latent_action_dim:
                raise ValueError(
                    f"embedding targets last dim {latent_targets.shape[-1]} must equal "
                    f"latent_action_dim={self.latent_action_dim}."
                )
        else:
            if latent_targets.shape[-1] != self.codebook_size:
                raise ValueError(
                    f"soft_kl targets last dim {latent_targets.shape[-1]} must equal "
                    f"codebook_size={self.codebook_size}."
                )

    def _latent_loss_from_learnable(
        self,
        learnable_out: Tensor,
        latent_targets: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Read out latent prediction from learnable hiddens and apply configured loss."""
        if self.foresight_latent_loss_type == "embedding":
            if self.learnable_to_latent_proj is None:
                raise RuntimeError("learnable_to_latent_proj is not initialized for embedding mode.")
            latent_pred = self.learnable_to_latent_proj(learnable_out).float()
            return F.mse_loss(latent_pred, latent_targets.float())

        if self.learnable_to_logits_proj is None:
            raise RuntimeError("learnable_to_logits_proj is not initialized for soft_kl mode.")
        logits = self.learnable_to_logits_proj(learnable_out).float()
        targets = latent_targets.detach().float()
        if logits.shape != targets.shape:
            raise ValueError(
                f"Latent logits shape {tuple(logits.shape)} does not match target {tuple(targets.shape)}."
            )

        eps = float(self.kl_eps)
        target_probs = targets.clamp_min(eps)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        log_probs = F.log_softmax(logits, dim=-1)
        # KL(teacher || student), mean over tokens then over (valid) samples.
        kl_per_sample = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=-1).mean(dim=1)
        if valid_mask is None:
            return kl_per_sample.mean()
        valid_mask = valid_mask.to(device=kl_per_sample.device, dtype=kl_per_sample.dtype).detach()
        denom = valid_mask.sum().clamp_min(1.0)
        return (kl_per_sample * valid_mask).sum() / denom

    # -- joint suffix: [state, learnable, action_time] --------------------
    def embed_suffix_foresight(self, state, noisy_actions, timestep):
        """Three-block pi0 suffix used by the joint action+latent path."""
        if state.ndim == 3:
            state = state.squeeze(1)
        bsize = state.shape[0]
        device = state.device
        state = state.float()
        noisy_actions = noisy_actions.float()
        timestep = timestep.float()

        state_emb = self.state_proj(state)
        lt_emb = self._embed_learnable_tokens(bsize, device)

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.proj_width, min_period=4e-3, max_period=4.0, device=device
        ).to(dtype=torch.float32)

        action_emb = self.action_in_proj(noisy_actions)
        time_emb_rep = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
        action_time_emb = torch.cat([action_emb, time_emb_rep], dim=-1)
        action_time_emb = F.silu(self.action_time_mlp_in(action_time_emb))
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        num_lt = self.num_learnable_tokens

        embs = torch.cat([state_emb[:, None], lt_emb, action_time_emb], dim=1)
        suffix_len = 1 + num_lt + action_time_dim
        pad_masks = torch.ones((bsize, suffix_len), device=device, dtype=torch.bool)
        # Blocks: state | learnable | action (first token of each opens a block).
        att_masks = torch.zeros((bsize, suffix_len), device=device, dtype=torch.bool)
        att_masks[:, 0] = True
        att_masks[:, 1] = True
        att_masks[:, 1 + num_lt] = True
        return time_emb, embs, pad_masks, att_masks

    # -- latent-only suffix: [learnable(N)] -------------------------------
    def embed_suffix_learnable_only(self, bsize: int, device: torch.device):
        """Pure learnable-token suffix (no state / action); independent pad masks."""
        lt_emb = self._embed_learnable_tokens(bsize, device)
        num_lt = self.num_learnable_tokens
        pad_masks = torch.ones((bsize, num_lt), device=device, dtype=torch.bool)
        # Single free-attention block (same semantics as embed_suffix_latent).
        att_masks = torch.zeros((bsize, num_lt), device=device, dtype=torch.bool)
        att_masks[:, 0] = True
        return lt_emb, pad_masks, att_masks

    # -- expert runners ---------------------------------------------------
    def run_expert_foresight(
        self,
        prefix: dict,
        prefix_kvs,
        state: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Joint expert forward; returns full suffix hidden states."""
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_foresight(
            state, x_t, timestep
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

    def run_expert_learnable_only(
        self,
        prefix: dict,
        prefix_kvs,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Latent-only expert forward over pure learnable tokens."""
        bsize = prefix["prompt_pad_masks"].shape[0]
        device = prefix["prompt_pad_masks"].device
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_learnable_only(
            bsize, device
        )
        suffix_position_ids = self._build_suffix_position_ids(
            prefix["position_ids"], prefix["prompt_pad_masks"], suffix_pad_masks
        )
        ada_cond = None
        if self.adanorm_time:
            ada_cond = self.foresight_cond.float()[None].expand(bsize, -1)
        return self.qwenvl_with_expert.run_expert(
            suffix_embs=suffix_embs,
            prefix_kvs=prefix_kvs,
            suffix_position_ids=suffix_position_ids,
            prefix_pad_masks=prefix["prompt_pad_masks"],
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            ada_cond=ada_cond,
            attention_implementation=attention_implementation or self.latent_expert_attention,
        )

    @staticmethod
    def _split_foresight_outputs(suffix_out: Tensor, num_lt: int, action_len: int):
        """Slice ``[state | learnable | action]`` hidden states."""
        learnable_out = suffix_out[:, 1 : 1 + num_lt]
        action_out = suffix_out[:, 1 + num_lt : 1 + num_lt + action_len]
        return learnable_out, action_out

    # -- training: joint --------------------------------------------------
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
        """One joint expert pass -> ``(action_loss, latent_loss)``.

        Latent loss uses only the first ``B`` rows when ``repeats > 1``.
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
            if repeats > 1:
                actions = actions.repeat(repeats, 1, 1)
                action_mask = action_mask.repeat(repeats, 1, 1)
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
            if time is None:
                time = self.sample_time(actions.size(0), device)
            else:
                time = time.float()

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            expert_prefix, expert_kvs = self._expand_prefix_for_repeats(prefix, prefix_kvs, repeats)
            suffix_out = self.run_expert_foresight(
                expert_prefix,
                expert_kvs,
                state,
                x_t,
                time,
                attention_implementation=attention_implementation,
            )
            learnable_out, action_out = self._split_foresight_outputs(
                suffix_out, self.num_learnable_tokens, x_t.shape[1]
            )

            v_t = self.action_out_proj(action_out).float()
            if self.loss_type == "L1_fm":
                losses = F.l1_loss(u_t, v_t, reduction="none")
            else:
                losses = F.mse_loss(u_t, v_t, reduction="none")
            action_loss = (losses * action_mask).sum() / action_mask.sum().clamp_min(1.0)

            # Causal: learnable tokens never see the action block, so repeats are
            # identical; only the original B rows contribute to latent loss.
            latent_loss = self._latent_loss_from_learnable(
                learnable_out[:bsize], latent_targets, valid_mask=latent_valid_mask
            )

        return action_loss, latent_loss

    # -- training: latent-only --------------------------------------------
    def flow_matching_loss_latent_only(
        self,
        prefix: dict,
        prefix_kvs,
        latent_targets: Tensor,
        attention_implementation: Optional[str] = None,
        latent_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Deterministic learnable-token readout (no state, no repeats)."""
        latent_targets = latent_targets.float()
        self._check_latent_token_count(latent_targets)

        if latent_targets.shape[0] != prefix["prompt_pad_masks"].shape[0]:
            raise ValueError(
                f"latent_targets batch {latent_targets.shape[0]} must match prefix batch "
                f"{prefix['prompt_pad_masks'].shape[0]}."
            )

        with torch.autocast("cuda", dtype=torch.float32):
            suffix_out = self.run_expert_learnable_only(
                prefix, prefix_kvs, attention_implementation=attention_implementation
            )
            latent_loss = self._latent_loss_from_learnable(
                suffix_out, latent_targets, valid_mask=latent_valid_mask
            )

        return latent_loss

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        inputs: dict,
        state: Tensor,
        noise: Optional[Tensor] = None,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Euler sampling with foresight tokens present (matches joint training)."""
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
            dt = -1.0 / self.num_steps
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                suffix_out = self.run_expert_foresight(
                    prefix,
                    prefix_kvs,
                    state,
                    x_t,
                    expanded_time,
                    attention_implementation=attention_implementation,
                )
                _, action_out = self._split_foresight_outputs(
                    suffix_out, self.num_learnable_tokens, x_t.shape[1]
                )
                v_t = self.action_out_proj(action_out).float()
                x_t = x_t + dt * v_t
                time = time + dt
        return x_t

    @torch.no_grad()
    def sample_latent(self, prefix: dict, prefix_kvs) -> Tensor:
        """Single-pass foresight readout (embedding or codebook logits)."""
        with torch.autocast("cuda", dtype=torch.float32):
            suffix_out = self.run_expert_learnable_only(prefix, prefix_kvs)
            if self.foresight_latent_loss_type == "embedding":
                return self.learnable_to_latent_proj(suffix_out).float()
            return self.learnable_to_logits_proj(suffix_out).float()
