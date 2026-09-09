# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""WAN video branch: world-model queries drive a Wan2.1 DiT.

Ported from harobotsDL ``robotsdl/models/policy/wla/wan_video.py`` with two
WMv4-specific changes:

- ``query_width`` is the action-expert hidden size (768), not the VLM width.
- ``forward`` infers the number of future pixel frames from ``frames.shape[1]``
  instead of freezing ``num_future_frames`` at construction. WMv4 packing uses
  8 uniformly sampled future pixels (2 future latents) on both aloha and arx.

Query hidden states replace WAN's UMT5 prompt embedding: ``query_norm`` +
``query_proj`` map them to the DiT width and every block uses that tensor as
cross-attention K/V. The text encoder is never loaded.

The video stack is ``[current frame, *future frames]``. The WAN VAE compresses
4 pixel frames into 1 latent frame, so ``1 + F`` pixels become ``1 + F/4``
latents. The first latent is pinned to the clean current observation (TI2V
teacher forcing); the DiT only denoises the future.

The VAE is frozen and kept outside the module tree (held in a list) so its
weights stay out of checkpoints and DDP bookkeeping.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .scheduler import FlowMatchScheduler
from .latent_cache import WanLatentCache


class WanVideoBranch(nn.Module):
    """Wan2.1-T2V DiT conditioned on world-model query hidden states.

    ``wan_model_path`` must be a diffusers-format Wan2.1 checkpoint directory
    (``transformer/`` + ``vae/`` subfolders), e.g. a local clone of
    ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers``.
    """

    VAE_TEMPORAL_STRIDE = 4

    def __init__(
        self,
        wan_model_path,
        query_width,
        num_train_timesteps=1000,
        num_inference_steps=10,
        flow_shift=5.0,
        dtype=torch.bfloat16,
        freeze_dit=False,
        gradient_checkpointing=True,
        vae_cache=None,
    ):
        super().__init__()
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        if not wan_model_path:
            raise ValueError(
                "The WAN video branch requires wan_model_path pointing at a local "
                "diffusers-format Wan2.1 checkpoint (transformer/ + vae/ subfolders)."
            )

        self.transformer = WanTransformer3DModel.from_pretrained(
            wan_model_path, subfolder="transformer", torch_dtype=dtype
        )
        if freeze_dit:
            self.transformer.requires_grad_(False)
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        vae = AutoencoderKLWan.from_pretrained(
            wan_model_path, subfolder="vae", torch_dtype=dtype
        )
        vae.eval()
        vae.requires_grad_(False)
        self._vae = [vae]
        self.latent_cache = WanLatentCache.from_config(wan_model_path, vae_cache, dtype)

        dit_config = self.transformer.config
        dit_width = dit_config.num_attention_heads * dit_config.attention_head_dim
        self.query_norm = nn.LayerNorm(query_width)
        self.query_proj = nn.Linear(query_width, dit_width)

        self.num_inference_steps = num_inference_steps
        self.flow_shift = flow_shift

        z_dim = len(vae.config.latents_mean)
        self.register_buffer(
            "latents_mean",
            torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1),
            persistent=False,
        )

        self.scheduler = self._build_scheduler(num_train_timesteps)
        self._schedule_cache = {}

    def _schedule_on(self, device):
        """fp32 ``(sigmas, timesteps)`` of the training schedule, per device."""
        cached = self._schedule_cache.get(device)
        if cached is None:
            cached = tuple(
                table.to(device=device, dtype=torch.float32)
                for table in (self.scheduler.sigmas, self.scheduler.timesteps)
            )
            self._schedule_cache[device] = cached
        return cached

    def _build_scheduler(self, num_steps):
        return FlowMatchScheduler(
            num_inference_steps=num_steps,
            shift=self.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    @staticmethod
    def _future_pixel_frames(num_pixel_frames: int) -> int:
        if num_pixel_frames < 1 + WanVideoBranch.VAE_TEMPORAL_STRIDE:
            raise ValueError(
                f"WAN video stack must have current + at least "
                f"{WanVideoBranch.VAE_TEMPORAL_STRIDE} future frames, got {num_pixel_frames}."
            )
        future = num_pixel_frames - 1
        if future % WanVideoBranch.VAE_TEMPORAL_STRIDE != 0:
            raise ValueError(
                f"WAN future pixel frames must be a multiple of "
                f"{WanVideoBranch.VAE_TEMPORAL_STRIDE}, got {future} "
                f"(pixel stack T={num_pixel_frames})."
            )
        return future

    # -- VAE --------------------------------------------------------------
    def _vae_on(self, device):
        vae = self._vae[0]
        if next(vae.parameters()).device != device:
            vae.to(device)
        return vae

    @torch.no_grad()
    def encode_frames(self, frames, cache_keys=None):
        """frames: ``(B, T, 3, H, W)`` in [-1, 1]. Returns normalized latents
        ``(B, z_dim, 1 + (T - 1) // 4, H // 8, W // 8)`` in float32."""
        vae = self._vae_on(frames.device)
        video = frames.permute(0, 2, 1, 3, 4).to(dtype=vae.dtype)
        latent = vae.encode(video).latent_dist.mode()
        if self.latent_cache is not None and cache_keys is not None:
            self.latent_cache.write_batch(cache_keys, latent)
        return self.normalize_latent(latent)

    @torch.no_grad()
    def normalize_latent(self, latent):
        """Normalize online or cached raw posterior modes in the same dtype."""
        latent = latent.float()
        mean = self.latents_mean.to(latent.device)
        std = self.latents_std.to(latent.device)
        return (latent - mean) / std

    @torch.no_grad()
    def decode_latent(self, latent):
        """Inverse of :meth:`encode_frames`: returns ``(B, T, 3, H, W)`` in [0, 1]."""
        vae = self._vae_on(latent.device)
        mean = self.latents_mean.to(latent.device)
        std = self.latents_std.to(latent.device)
        latent = latent.float() * std + mean
        video = vae.decode(latent.to(vae.dtype)).sample.float()
        return (video / 2 + 0.5).clamp(0, 1).permute(0, 2, 1, 3, 4)

    # -- conditioning -----------------------------------------------------
    def query_context(self, query_hidden):
        """Project last-layer query states to the DiT width."""
        query_hidden = query_hidden.to(self.query_proj.weight.dtype)
        return self.query_proj(self.query_norm(query_hidden))

    def dit_forward(self, latent, context, timestep):
        """WAN DiT velocity prediction with `context` as cross-attention K/V.

        Mirrors ``WanTransformer3DModel.forward`` except that the projected
        queries go straight to the blocks instead of through ``text_embedder``.
        """
        dit = self.transformer
        p_t, p_h, p_w = dit.config.patch_size
        bsize, _, frames, height, width = latent.shape

        rotary_emb = dit.rope(latent)
        hidden_states = dit.patch_embedding(latent).flatten(2).transpose(1, 2)

        embedder = dit.condition_embedder
        time_freq = embedder.timesteps_proj(timestep)
        time_dtype = next(embedder.time_embedder.parameters()).dtype
        temb = embedder.time_embedder(time_freq.to(time_dtype)).type_as(context)
        timestep_proj = embedder.time_proj(embedder.act_fn(temb)).unflatten(1, (6, -1))

        if context.ndim == 4 and context.shape[1] != len(dit.blocks):
            raise ValueError(
                f"Per-layer WAN context has {context.shape[1]} layers, "
                f"but the DiT has {len(dit.blocks)} blocks."
            )
        use_checkpoint = dit.gradient_checkpointing and torch.is_grad_enabled()
        for block_idx, block in enumerate(dit.blocks):
            block_context = context[:, block_idx] if context.ndim == 4 else context
            if use_checkpoint:
                hidden_states = dit._gradient_checkpointing_func(
                    block, hidden_states, block_context, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, block_context, timestep_proj, rotary_emb)

        shift, scale = (dit.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (
            dit.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = dit.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            bsize, frames // p_t, height // p_h, width // p_w, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    # -- training ---------------------------------------------------------
    def forward(self, query_hidden, frames, sample_weight=None, frame_repeats=1,
                cached_latents=None, cache_keys=None):
        """Flow-matching velocity loss over the future latent frames.

        query_hidden: ``(B, num_queries, query_width)``, or
            ``(B, num_dit_blocks, num_queries, query_width)`` when feeding one
            hidden state per DiT block;
        frames: ``(B_frames, 1 + F, 3, H, W)`` in [-1, 1], with ``F % 4 == 0``;
        sample_weight: optional ``(B,)`` in [0, 1].
        frame_repeats: reuse each deterministic VAE result for repeated query
            batches; query order must match ``frames.repeat(frame_repeats, ...)``.
        """
        if cached_latents is None:
            if frames is None:
                raise ValueError("WAN requires frames or cached VAE posterior modes")
            self._future_pixel_frames(int(frames.shape[1]))
        elif cached_latents.ndim != 5 or cached_latents.shape[2] < 2:
            raise ValueError("Cached WAN modes must be (B, C, T_latent>=2, H, W)")
        frame_repeats = int(frame_repeats)
        if frame_repeats < 1:
            raise ValueError(f"frame_repeats must be >= 1, got {frame_repeats}.")

        context = self.query_context(query_hidden)
        clean_latent = (self.encode_frames(frames, cache_keys=cache_keys) if cached_latents is None
                        else self.normalize_latent(cached_latents.to(context.device, non_blocking=True)))
        frame_batch_size = clean_latent.shape[0]
        if frame_repeats > 1:
            clean_latent = clean_latent.repeat(frame_repeats, 1, 1, 1, 1)
        # Wan's causal encoder emits the current-frame latent first, so the
        # full-video encode already contains the clean TI2V condition.
        cond_latent = clean_latent[:, :, :1]

        bsize = clean_latent.shape[0]
        if context.shape[0] != bsize:
            raise ValueError(
                f"query batch {context.shape[0]} must equal encoded frame batch "
                f"{frame_batch_size} * frame_repeats {frame_repeats} = {bsize}."
            )
        if sample_weight is not None:
            sample_weight = sample_weight.to(clean_latent.device).float().reshape(-1)
            if sample_weight.numel() != bsize:
                raise ValueError(
                    f"sample_weight must have one entry per repeated sample, got "
                    f"{sample_weight.numel()} for a batch of {bsize}."
                )
        device = clean_latent.device
        train_sigmas, train_timesteps = self._schedule_on(device)
        timestep_id = torch.randint(0, train_timesteps.numel(), (bsize,), device=device)
        sigma = train_sigmas[timestep_id].to(clean_latent).view(bsize, 1, 1, 1, 1)
        timestep = train_timesteps[timestep_id]

        noise = torch.randn_like(clean_latent)
        noisy_latent = clean_latent * (1 - sigma) + noise * sigma
        noisy_latent[:, :, :1] = cond_latent
        target = noise - clean_latent

        pred = self.dit_forward(noisy_latent.to(context.dtype), context, timestep)
        if sample_weight is None:
            return F.mse_loss(pred[:, :, 1:].float(), target[:, :, 1:].float())

        per_sample = F.mse_loss(
            pred[:, :, 1:].float(), target[:, :, 1:].float(), reduction="none"
        ).flatten(1).mean(1)
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def generate_video(self, query_hidden, first_frame, num_future_frames, num_inference_steps=None):
        """Denoise future latents from the clean current frame.

        first_frame: ``(B, 1, 3, H, W)`` in [-1, 1].
        num_future_frames: pixel future count; must be a multiple of 4.
        Returns the decoded stack ``(B, 1 + num_future_frames, 3, H, W)`` in [0, 1].
        """
        future = int(num_future_frames)
        if future % self.VAE_TEMPORAL_STRIDE != 0:
            raise ValueError(
                f"num_future_frames must be a multiple of {self.VAE_TEMPORAL_STRIDE}, got {future}."
            )
        context = self.query_context(query_hidden)
        cond_latent = self.encode_frames(first_frame)

        scheduler = self._build_scheduler(num_inference_steps or self.num_inference_steps)
        bsize, z_dim, _, height, width = cond_latent.shape
        num_latent_frames = 1 + future // self.VAE_TEMPORAL_STRIDE
        latent = torch.randn(
            (bsize, z_dim, num_latent_frames, height, width),
            device=cond_latent.device,
            dtype=cond_latent.dtype,
        )
        latent[:, :, :1] = cond_latent

        for timestep in scheduler.timesteps:
            velocity = self.dit_forward(
                latent.to(context.dtype),
                context,
                timestep.to(latent.device).expand(bsize),
            ).float()
            velocity[:, :, :1] = 0
            latent = scheduler.step(velocity, timestep, latent)
            latent[:, :, :1] = cond_latent

        return self.decode_latent(latent)
