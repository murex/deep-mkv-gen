"""Time-feature layers used by the Z network."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for scalar time inputs."""

    def __init__(self, embed_dim: int, max_period: float = 10_000.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even")
        half = embed_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.float()
        args = t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
