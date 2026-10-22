"""Generate a single image with DDIM sampling and classifier-free guidance."""
import argparse
from pathlib import Path

import torch, yaml
from PIL import Image

from .diffusion import GaussianDiffusion
from .models.pretrained import DiffusersAutoencoder, HFTextEncoder
from .models.uvit import SCLUViT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True); p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=50); p.add_argument("--cfg", type=float, default=None)
    a = p.parse_args(); cfg = yaml.safe_load(open(a.config, encoding="utf-8")); device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SCLUViT(**cfg["model"]).to(device).eval()
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False); model.load_state_dict(ckpt["model"])
    text = HFTextEncoder(cfg["training"].get("text_encoder") or "openai/clip-vit-large-patch14", device)
    vae = DiffusersAutoencoder(cfg["training"].get("autoencoder") or "stable-diffusion-v1-5/stable-diffusion-v1-5", device)
    condition = text.encode([a.prompt]); dc = cfg["diffusion"]
    diffusion = GaussianDiffusion(**{k: dc[k] for k in ("timesteps", "beta_start", "beta_end") if k in dc})
    z = diffusion.ddim_sample(model, (1, model.latent_channels, model.latent_size, model.latent_size), condition,
                              steps=a.steps, guidance_scale=a.cfg if a.cfg is not None else dc.get("cfg_scale", 7.5), device=device)
    img = (vae.decode(z)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Image.fromarray(img).save(a.output)
    print(f"Saved {a.output}")


if __name__ == "__main__": main()
