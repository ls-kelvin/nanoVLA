# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv4_LA foresight: WMv32 suffix plus world-model queries feeding WAN.

Joint suffix:  ``[state | LA | WM | action]`` (four pi0 blocks)
Latent-only:   ``[LA | WM]`` (two pi0 blocks)

World-model query hidden states condition a Wan2.1 DiT. Joint WAN loss uses
every ``repeated_diffusion_steps`` row, matching WMv32 latent supervision.
"""

from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .flow_matching_foresight_v32 import DualStreamFlowMatchingForesightV32
from .joint_attention import create_sinusoidal_pos_embedding


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
        return default if value is None else value
    return getattr(cfg, key, default)


def _optional_path(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "null", "None"}:
        return None
    return text


class DualStreamFlowMatchingForesightV4(DualStreamFlowMatchingForesightV32):
    """WMv32 foresight with world-model queries and an optional WAN video branch."""

    def __init__(self, global_config):
        super().__init__(global_config)
        self.qwenvl_with_expert.reuse_prefix_kv = bool(
            _cfg_get(global_config.framework.qwenvl, "reuse_prefix_kv", False)
        )
        wan_cfg = _cfg_get(global_config.framework, "wan", {}) or {}
        num_wm_queries = _cfg_get(wan_cfg, "num_wm_queries", None)
        if num_wm_queries is None:
            raise ValueError(
                "DualStreamFlowMatchingForesightV4 requires framework.wan.num_wm_queries."
            )
        self.num_wm_queries = int(num_wm_queries)
        if self.num_wm_queries < 1:
            raise ValueError(f"num_wm_queries must be >= 1, got {self.num_wm_queries}.")

        self.wm_query_embs = nn.Parameter(
            torch.zeros(self.num_wm_queries, self.proj_width)
        )
        nn.init.trunc_normal_(self.wm_query_embs, std=0.02)
        self.wm_query_in_proj = nn.Linear(self.proj_width, self.proj_width)

        self.wan_video = None
        wan_enabled = bool(_cfg_get(wan_cfg, "enabled", True))
        wan_model_path = _optional_path(_cfg_get(wan_cfg, "wan_model_path", None))
        # Training keeps enabled + wan_model_path. Action inference passes
        # load_wan=False (from_pretrained default), which disables the branch
        # before this constructor runs.
        if wan_enabled and wan_model_path is not None:
            from starVLA.model.modules.wan import WanVideoBranch

            self.wan_video = WanVideoBranch(
                wan_model_path=wan_model_path,
                query_width=self.proj_width,
                num_inference_steps=int(_cfg_get(wan_cfg, "num_inference_steps", 10)),
                flow_shift=float(_cfg_get(wan_cfg, "flow_shift", 5.0)),
                dtype=torch.bfloat16,
                freeze_dit=bool(_cfg_get(wan_cfg, "freeze_dit", False)),
                gradient_checkpointing=bool(_cfg_get(wan_cfg, "gradient_checkpointing", True)),
                vae_cache=_cfg_get(wan_cfg, "vae_cache", None),
            )

    def _embed_wm_queries(self, bsize: int, device: torch.device) -> Tensor:
        wm_emb = self.wm_query_in_proj(self.wm_query_embs.float())
        return wm_emb[None].expand(bsize, -1, -1).to(device=device)

    # -- joint suffix: [state, LA, WM, action] ----------------------------
    def embed_suffix_foresight(self, state, noisy_actions, timestep):
        """Four-block pi0 suffix: state | latent-action query | world-model query | action."""
        if state.ndim == 3:
            state = state.squeeze(1)
        bsize = state.shape[0]
        device = state.device
        state = state.float()
        noisy_actions = noisy_actions.float()
        timestep = timestep.float()

        state_emb = self.state_proj(state)
        lt_emb = self._embed_learnable_tokens(bsize, device)
        wm_emb = self._embed_wm_queries(bsize, device)

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
        num_wm = self.num_wm_queries

        embs = torch.cat([state_emb[:, None], lt_emb, wm_emb, action_time_emb], dim=1)
        suffix_len = 1 + num_lt + num_wm + action_time_dim
        pad_masks = torch.ones((bsize, suffix_len), device=device, dtype=torch.bool)
        att_masks = torch.zeros((bsize, suffix_len), device=device, dtype=torch.bool)
        att_masks[:, 0] = True
        att_masks[:, 1] = True
        att_masks[:, 1 + num_lt] = True
        att_masks[:, 1 + num_lt + num_wm] = True
        return time_emb, embs, pad_masks, att_masks

    # -- latent-only suffix: [LA, WM] -------------------------------------
    def embed_suffix_learnable_only(self, bsize: int, device: torch.device):
        """Learnable-token suffix plus world-model queries; two causal blocks."""
        lt_emb = self._embed_learnable_tokens(bsize, device)
        wm_emb = self._embed_wm_queries(bsize, device)
        embs = torch.cat([lt_emb, wm_emb], dim=1)
        num_lt = self.num_learnable_tokens
        num_wm = self.num_wm_queries
        pad_masks = torch.ones((bsize, num_lt + num_wm), device=device, dtype=torch.bool)
        att_masks = torch.zeros((bsize, num_lt + num_wm), device=device, dtype=torch.bool)
        att_masks[:, 0] = True
        att_masks[:, num_lt] = True
        return embs, pad_masks, att_masks

    def _split_foresight_outputs(self, suffix_out: Tensor, num_lt: int, action_len: int):
        """Slice ``[state | LA | WM | action]`` hidden states."""
        num_wm = self.num_wm_queries
        learnable_out = suffix_out[:, 1 : 1 + num_lt]
        wm_out = suffix_out[:, 1 + num_lt : 1 + num_lt + num_wm]
        action_out = suffix_out[:, 1 + num_lt + num_wm : 1 + num_lt + num_wm + action_len]
        return learnable_out, wm_out, action_out

    def _split_learnable_only_outputs(self, suffix_out: Tensor):
        """Slice ``[LA | WM]`` hidden states."""
        num_lt = self.num_learnable_tokens
        learnable_out = suffix_out[:, :num_lt]
        wm_out = suffix_out[:, num_lt : num_lt + self.num_wm_queries]
        return learnable_out, wm_out

    def _wan_video_loss(
        self,
        wm_out: Tensor,
        wm_frames: Optional[Tensor],
        frame_repeats: int = 1,
        wm_latents: Optional[Tensor] = None,
        wm_cache_keys=None,
    ) -> Optional[Tensor]:
        if self.wan_video is None:
            return None
        source = wm_frames if wm_latents is None else wm_latents
        if source is None:
            raise ValueError("The WAN video branch requires wm_frames at train time.")
        expected_batch = source.shape[0] * int(frame_repeats)
        if expected_batch != wm_out.shape[0]:
            raise ValueError(
                f"WAN source batch {source.shape[0]} * frame_repeats {frame_repeats} "
                f"must match WM hidden batch {wm_out.shape[0]}."
            )
        return self.wan_video(wm_out, wm_frames, frame_repeats=frame_repeats,
                              cached_latents=wm_latents, cache_keys=wm_cache_keys)

    # -- training: joint ----------------------------------------------------
    def flow_matching_loss_joint_foresight(
        self,
        prefix: dict,
        prefix_kvs,
        state: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        latent_targets: Tensor,
        wm_frames: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        num_repeats: Optional[int] = None,
        attention_implementation: Optional[str] = None,
        latent_valid_mask: Optional[Tensor] = None,
        wm_latents: Optional[Tensor] = None,
        wm_cache_keys=None,
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """One joint expert pass -> ``(action_loss, latent_loss, video_loss)``."""
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
        if wm_frames is not None and wm_frames.shape[0] != bsize:
            raise ValueError(
                f"wm_frames batch {wm_frames.shape[0]} must match actions batch {bsize} before repeats."
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
            learnable_out, wm_out, action_out = self._split_foresight_outputs(
                suffix_out, self.num_learnable_tokens, x_t.shape[1]
            )

            v_t = self.action_out_proj(action_out).float()
            if self.loss_type == "L1_fm":
                losses = F.l1_loss(u_t, v_t, reduction="none")
            else:
                losses = F.mse_loss(u_t, v_t, reduction="none")
            action_loss = (losses * action_mask).sum() / action_mask.sum().clamp_min(1.0)

            if repeats > 1:
                latent_targets = latent_targets.repeat(repeats, 1, 1)
                if latent_valid_mask is not None:
                    latent_valid_mask = latent_valid_mask.repeat(repeats)
            latent_loss = self._latent_loss_from_learnable(
                learnable_out, latent_targets, valid_mask=latent_valid_mask
            )

        # The frozen deterministic VAE only needs to encode each source video
        # once; its latents are then tiled in the same order as the expert rows.
        video_loss = self._wan_video_loss(wm_out, wm_frames, frame_repeats=repeats,
                                        wm_latents=wm_latents, wm_cache_keys=wm_cache_keys)
        return action_loss, latent_loss, video_loss

    # -- training: latent-only ----------------------------------------------
    def flow_matching_loss_latent_only(
        self,
        prefix: dict,
        prefix_kvs,
        latent_targets: Tensor,
        wm_frames: Optional[Tensor] = None,
        num_repeats: Optional[int] = None,
        attention_implementation: Optional[str] = None,
        latent_valid_mask: Optional[Tensor] = None,
        wm_latents: Optional[Tensor] = None,
        wm_cache_keys=None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Learnable + WM readout; latent-only never repeats."""
        latent_targets = latent_targets.float()
        self._check_latent_token_count(latent_targets)

        if latent_targets.shape[0] != prefix["prompt_pad_masks"].shape[0]:
            raise ValueError(
                f"latent_targets batch {latent_targets.shape[0]} must match prefix batch "
                f"{prefix['prompt_pad_masks'].shape[0]}."
            )
        repeats = 1

        with torch.autocast("cuda", dtype=torch.float32):
            if repeats > 1:
                latent_targets = latent_targets.repeat(repeats, 1, 1)
                if latent_valid_mask is not None:
                    latent_valid_mask = latent_valid_mask.repeat(repeats)
            expert_prefix, expert_kvs = self._expand_prefix_for_repeats(prefix, prefix_kvs, repeats)
            suffix_out = self.run_expert_learnable_only(
                expert_prefix, expert_kvs, attention_implementation=attention_implementation
            )
            learnable_out, wm_out = self._split_learnable_only_outputs(suffix_out)
            latent_loss = self._latent_loss_from_learnable(
                learnable_out, latent_targets, valid_mask=latent_valid_mask
            )

        video_loss = self._wan_video_loss(wm_out, wm_frames, wm_latents=wm_latents, wm_cache_keys=wm_cache_keys)
        return latent_loss, video_loss

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        inputs: dict,
        state: Tensor,
        noise: Optional[Tensor] = None,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Euler sampling with LA + WM queries present (matches joint training)."""
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
                _, _, action_out = self._split_foresight_outputs(
                    suffix_out, self.num_learnable_tokens, x_t.shape[1]
                )
                v_t = self.action_out_proj(action_out).float()
                x_t = x_t + dt * v_t
                time = time + dt
        return x_t

    @torch.no_grad()
    def sample_latent(self, prefix: dict, prefix_kvs) -> Tensor:
        """Read out only the LA query tokens; WM queries stay in the suffix."""
        bsize = prefix["prompt_pad_masks"].shape[0]
        device = prefix["prompt_pad_masks"].device
        with torch.autocast("cuda", dtype=torch.float32):
            timestep = torch.zeros(bsize, dtype=torch.float32, device=device)
            suffix_out = self.run_expert_learnable_only(prefix, prefix_kvs, timestep=timestep)
            learnable_out, _ = self._split_learnable_only_outputs(suffix_out)
            if self.foresight_latent_loss_type == "embedding":
                return self.learnable_to_latent_proj(learnable_out).float()
            return self.learnable_to_logits_proj(learnable_out).float()
