"""
MetaAdvantageHead: context encoder from support set stats + advantage correction (o_t, A_t, c_tau) -> delta.
Paper: support set encodes task context c_tau; query set uses A_tilde = A_t + delta_phi(o_t, A_t, c_tau).
"""
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


def _encode_task_id(year: int, day: int) -> int:
    """Encode (year, day) as single int for buffer storage."""
    return year * 1000 + day


def decode_task_id(encoded: int) -> tuple:
    """Decode buffer value to (year, day)."""
    return encoded // 1000, encoded % 1000


# Context input: state_mean (obs_dim), state_std (obs_dim), adv_mean, adv_std, reward_mean, reward_std, return_mean, return_std
# We use first state_dim obs for context to keep size manageable (e.g. 30)
STATE_DIM_FOR_CONTEXT = 30
CONTEXT_STAT_DIM = STATE_DIM_FOR_CONTEXT * 2 + 2 + 2 + 2 + 2  # state mean+std, adv, reward, return (each mean+std)


class MetaAdvantageHead(nn.Module):
    """
    Task context encoder (from support set statistics) + advantage correction head.
    - Context encoder: support stats -> c_tau (context vector).
    - Correction head: (obs, A_t, c_tau) -> scalar delta; A_tilde = A_t + delta, then normalize & clip.
    """

    def __init__(
        self,
        obs_dim: int,
        context_dim: int = 64,
        state_dim_for_context: int = STATE_DIM_FOR_CONTEXT,
        use_batch_norm: bool = True,
        use_output_clip: bool = True,
        use_obs_in_correction: bool = True,
        use_advantage_in_correction: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_dim = context_dim
        self.state_dim_for_context = min(state_dim_for_context, obs_dim)
        self.use_batch_norm = use_batch_norm
        self.use_output_clip = use_output_clip
        self.use_obs_in_correction = use_obs_in_correction
        self.use_advantage_in_correction = use_advantage_in_correction

        # Context encoder input: state mean (state_dim), state std (state_dim), adv mean, adv std, reward mean, reward std, return mean, return std
        context_input_dim = self.state_dim_for_context * 2 + 6
        layers = []
        in_dim = context_input_dim
        for out_dim in [128, context_dim]:
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batch_norm:
                # LayerNorm works with batch_size=1 (BatchNorm fails when support has 1 sample)
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.SiLU())
            in_dim = out_dim
        self.context_encoder = nn.Sequential(*layers)

        # Advantage correction head: (obs_reduced or full, A_t, c_tau) -> delta
        # Reduce obs for correction to avoid huge input (e.g. first 64 dims or project)
        self.obs_proj_dim = min(64, obs_dim)
        self.obs_proj = nn.Linear(obs_dim, self.obs_proj_dim) if obs_dim > self.obs_proj_dim else None
        correction_input_dim = 0
        if use_obs_in_correction:
            correction_input_dim += self.obs_proj_dim if self.obs_proj is not None else obs_dim
        if use_advantage_in_correction:
            correction_input_dim += 1
        correction_input_dim += context_dim

        self.correction_mlp = nn.Sequential(
            nn.Linear(correction_input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def compute_context_from_stats(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        adv_mean: torch.Tensor,
        adv_std: torch.Tensor,
        reward_mean: torch.Tensor,
        reward_std: torch.Tensor,
        return_mean: torch.Tensor,
        return_std: torch.Tensor,
    ) -> torch.Tensor:
        """Encode support set statistics into context vector c_tau. All inputs (B, dim) or (dim,)."""
        if state_mean.dim() == 1:
            state_mean = state_mean.unsqueeze(0)
            state_std = state_std.unsqueeze(0)
            adv_mean = adv_mean.unsqueeze(0)
            adv_std = adv_std.unsqueeze(0)
            reward_mean = reward_mean.unsqueeze(0)
            reward_std = reward_std.unsqueeze(0)
            return_mean = return_mean.unsqueeze(0)
            return_std = return_std.unsqueeze(0)
        state_mean = state_mean[:, : self.state_dim_for_context]
        state_std = state_std[:, : self.state_dim_for_context]
        scalars = torch.stack(
            [adv_mean.squeeze(), adv_std.squeeze(), reward_mean.squeeze(), reward_std.squeeze(), return_mean.squeeze(), return_std.squeeze()],
            dim=-1,
        )
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)
        ctx_in = torch.cat([state_mean, state_std, scalars], dim=-1)
        return self.context_encoder(ctx_in)

    def forward(
        self,
        obs: torch.Tensor,
        advantages: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantage correction delta for query steps.
        obs: (B, obs_dim), advantages: (B,), context: (B, context_dim)
        Returns: delta (B,) such that A_tilde = A + delta.
        """
        parts = []
        if self.use_obs_in_correction:
            if self.obs_proj is not None:
                parts.append(self.obs_proj(obs))
            else:
                parts.append(obs)
        if self.use_advantage_in_correction:
            parts.append(advantages.unsqueeze(-1))
        parts.append(context)
        x = torch.cat(parts, dim=-1)
        delta = self.correction_mlp(x).squeeze(-1)
        return delta

    def normalize_and_clip(
        self,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        clip_range: float = 2.0,
    ) -> torch.Tensor:
        """Normalize advantages (over valid mask) and clip to [-clip_range, clip_range]."""
        if mask is not None:
            valid = advantages[mask]
            if valid.numel() == 0:
                return advantages
            mean, std = valid.mean(), valid.std()
            if std < 1e-8:
                std = torch.ones_like(std, device=advantages.device)
            out = (advantages - mean) / std
        else:
            mean, std = advantages.mean(), advantages.std()
            if std < 1e-8:
                std = torch.ones_like(std, device=advantages.device)
            out = (advantages - mean) / std
        if self.use_output_clip:
            out = torch.clamp(out, -clip_range, clip_range)
        return out
