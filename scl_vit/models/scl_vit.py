"""SCL-ViT wrapper for an externally supplied U-ViT backbone.

The base backbone owns patchification, timestep/text tokens, and the diffusion
prediction head. The adapter applies SPE/CAE/LFR to the image-token subsequence
at its input. Adapt this hook to the precise layer placement in the final model.
"""
from torch import nn

from .modules import ContextAwareEnhancer, LocalFeatureRefinement, SupplementaryPositionalEmbedding


class SCLViT(nn.Module):
    def __init__(self, backbone: nn.Module, grid_size: int, dim: int, use_spe=True, use_cae=True, use_lfr=True):
        super().__init__()
        self.backbone = backbone
        self.spe = SupplementaryPositionalEmbedding(grid_size, dim) if use_spe else None
        self.cae = ContextAwareEnhancer(dim, grid_size) if use_cae else None
        self.lfr = LocalFeatureRefinement(dim, grid_size) if use_lfr else None

    def refine_image_tokens(self, image_tokens):
        pos = None
        if self.spe is not None:
            image_tokens, pos = self.spe(image_tokens)
        if self.cae is not None:
            if pos is None:
                pos = image_tokens.new_zeros((1, image_tokens.shape[1], image_tokens.shape[2]))
            image_tokens = self.cae(image_tokens, pos)
        if self.lfr is not None:
            image_tokens = self.lfr(image_tokens)
        return image_tokens

    def forward(self, *args, **kwargs):
        if not hasattr(self.backbone, "forward_scl"):
            raise NotImplementedError("Provide a U-ViT backbone adapter exposing forward_scl(..., refine_image_tokens=callable)")
        return self.backbone.forward_scl(*args, refine_image_tokens=self.refine_image_tokens, **kwargs)
