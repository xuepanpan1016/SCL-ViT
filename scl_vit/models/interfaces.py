"""Interfaces to bind project-specific pretrained components."""
from typing import Protocol
import torch


class LatentAutoencoder(Protocol):
    def encode(self, images: torch.Tensor) -> torch.Tensor: ...
    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...


class TextConditioner(Protocol):
    def encode(self, captions: list[str]) -> torch.Tensor: ...


class DiffusionBackbone(Protocol):
    def forward(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor: ...
