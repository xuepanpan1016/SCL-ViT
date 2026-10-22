"""Paper-specific spatial and local feature modules operating on image tokens."""
import math

import torch
from torch import nn


class SupplementaryPositionalEmbedding(nn.Module):
    """Learned positional cue for a fixed image-token grid."""

    def __init__(self, grid_size: int, dim: int):
        super().__init__()
        self.grid_size = grid_size
        self.embedding = nn.Parameter(torch.zeros(1, grid_size * grid_size, dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, image_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image_tokens.ndim != 3 or image_tokens.shape[1] != self.grid_size**2:
            raise ValueError(f"Expected [B,{self.grid_size**2},D] image tokens, got {tuple(image_tokens.shape)}")
        pos = self.embedding.to(dtype=image_tokens.dtype)
        return image_tokens + pos, pos


class ContextAwareEnhancer(nn.Module):
    """Channel/spatial modulation with positional conditioning and local gated residual."""

    def __init__(self, dim: int, grid_size: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.grid_size = grid_size
        self.channel_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, dim), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(dim, 1, 1), nn.GroupNorm(1, 1), nn.SiLU(), nn.Sigmoid())
        self.position = nn.Linear(dim, dim)
        self.local_gate = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.project = nn.Conv2d(dim, dim, 1)

    def forward(self, tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        side = math.isqrt(n)
        if side * side != n or side != self.grid_size:
            raise ValueError("CAE requires a square token grid matching grid_size")
        x = tokens.transpose(1, 2).reshape(b, c, side, side)
        p = self.position(pos).transpose(1, 2).reshape(1, c, side, side)
        channel = self.channel_mlp(x.mean(dim=(2, 3))).view(b, c, 1, 1)
        spatial = self.spatial(x + p)
        modulated = self.project((x + p) * channel * spatial)
        gate = torch.sigmoid(self.local_gate(x))
        out = x + gate * modulated
        return out.flatten(2).transpose(1, 2)


class LocalFeatureRefinement(nn.Module):
    """Two depthwise convolutions separated by GELU, with residual and token LayerNorm."""

    def __init__(self, dim: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.dw1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.GELU()
        self.dw2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        side = math.isqrt(n)
        if side * side != n or side != self.grid_size:
            raise ValueError("LFR requires a square token grid matching grid_size")
        x = tokens.transpose(1, 2).reshape(b, c, side, side)
        y = self.dw2(self.act(self.dw1(x)))
        return self.norm(tokens + y.flatten(2).transpose(1, 2))
