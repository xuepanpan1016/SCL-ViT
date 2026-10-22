"""Compact U-shaped diffusion Transformer for latent text-conditioned generation."""
import math

import torch
from torch import nn

from .modules import ContextAwareEnhancer, LocalFeatureRefinement, SupplementaryPositionalEmbedding


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freq = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1))
    args = t.float()[:, None] * freq[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        q = self.n1(x)
        x = x + self.attn(q, q, q, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class SCLUViT(nn.Module):
    """U-shaped token Transformer with SPE/CAE/LFR image-token refinement."""
    def __init__(self, latent_channels=4, latent_size=32, patch_size=2, hidden_dim=512,
                 depth=8, heads=8, text_dim=768, text_length=77, spe=True,
                 cae=True, lfr=True, cae_reduction=4, dropout=0.0):
        super().__init__()
        if latent_size % patch_size:
            raise ValueError("latent_size must be divisible by patch_size")
        self.latent_channels, self.latent_size, self.patch_size = latent_channels, latent_size, patch_size
        self.grid = latent_size // patch_size
        self.patch_in = nn.Conv2d(latent_channels, hidden_dim, patch_size, stride=patch_size)
        self.patch_out = nn.ConvTranspose2d(hidden_dim, latent_channels, patch_size, stride=patch_size)
        self.null_text = nn.Parameter(torch.zeros(1, text_length, text_dim))
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.time_token = nn.Linear(hidden_dim, hidden_dim)
        n = self.grid * self.grid
        self.pos = nn.Parameter(torch.zeros(1, 1 + text_length + n, hidden_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.spe = SupplementaryPositionalEmbedding(self.grid, hidden_dim) if spe else None
        self.cae = ContextAwareEnhancer(hidden_dim, self.grid, cae_reduction) if cae else None
        self.lfr = LocalFeatureRefinement(hidden_dim, self.grid) if lfr else None
        if depth < 2:
            raise ValueError("U-shaped transformer requires depth >= 2")
        self.encoder = nn.ModuleList([Block(hidden_dim, heads, dropout=dropout) for _ in range(depth // 2)])
        self.decoder = nn.ModuleList([Block(hidden_dim, heads, dropout=dropout) for _ in range(depth - depth // 2)])
        self.skip_proj = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim) for _ in self.decoder])
        self.norm, self.output_norm = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)

    def forward(self, noisy_latents, timesteps, text_embeddings, conditional_mask=None):
        b = noisy_latents.shape[0]
        if text_embeddings is None:
            text_embeddings = self.null_text.expand(b, -1, -1)
        if conditional_mask is not None:
            null = self.null_text.expand(b, -1, -1)
            text_embeddings = torch.where(conditional_mask[:, None, None], text_embeddings, null)
        image = self.patch_in(noisy_latents).flatten(2).transpose(1, 2)
        pos_img = None
        if self.spe is not None:
            image, pos_img = self.spe(image)
        text = self.text_proj(text_embeddings)
        t = self.time_proj(timestep_embedding(timesteps, self.pos.shape[-1]))
        seq = torch.cat([self.time_token(t)[:, None], text, image], dim=1)
        seq = seq + self.pos
        skips = []
        for block in self.encoder:
            seq = block(seq)
            skips.append(seq)
        if self.cae is not None or self.lfr is not None:
            itok = seq[:, 1 + text.shape[1]:]
            if self.cae is not None:
                itok = self.cae(itok, pos_img if pos_img is not None else torch.zeros_like(itok[:1]))
            if self.lfr is not None:
                itok = self.lfr(itok)
            seq = torch.cat([seq[:, :1 + text.shape[1]], itok], dim=1)
        for i, block in enumerate(self.decoder):
            seq = self.skip_proj[i](torch.cat([seq, skips[-1 - min(i, len(skips)-1)]], dim=-1))
            seq = block(seq)
        image = self.output_norm(seq[:, 1 + text.shape[1]:])
        image = image.transpose(1, 2).reshape(b, -1, self.grid, self.grid)
        return self.patch_out(image)
