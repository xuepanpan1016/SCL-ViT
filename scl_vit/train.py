"""Train SCL-ViT with frozen pretrained VAE/text encoder."""
import argparse, random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import CaptionImageDataset
from .diffusion import GaussianDiffusion
from .models.pretrained import DiffusersAutoencoder, HFTextEncoder
from .models.uvit import SCLUViT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = cfg.get("seed", 42); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg, model_cfg, train_cfg, diff_cfg = cfg["data"], cfg["model"], cfg["training"], cfg["diffusion"]
    if device == "cpu":
        raise SystemExit("Training is configured for GPU; use a CUDA device.")
    dataset = CaptionImageDataset(data_cfg["train_manifest"], data_cfg.get("image_size", 256))
    loader = DataLoader(dataset, batch_size=data_cfg.get("batch_size", 24), shuffle=True, num_workers=data_cfg.get("num_workers", 4), pin_memory=True, drop_last=True)
    autoencoder = DiffusersAutoencoder(train_cfg.get("autoencoder") or "stable-diffusion-v1-5/stable-diffusion-v1-5", device)
    text_encoder = HFTextEncoder(train_cfg.get("text_encoder") or "openai/clip-vit-large-patch14", device)
    model = SCLUViT(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 2e-4), weight_decay=train_cfg.get("weight_decay", .3))
    diffusion = GaussianDiffusion(**{k: diff_cfg[k] for k in ("timesteps", "beta_start", "beta_end") if k in diff_cfg})
    out = Path(train_cfg.get("checkpoint_dir", "runs/scl_vit")); out.mkdir(parents=True, exist_ok=True)
    iteration = 0; model.train()
    while iteration < train_cfg.get("iterations", 100000):
        for images, captions in loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                latents = autoencoder.encode(images)
                cond = text_encoder.encode(list(captions))
            keep = torch.rand(len(captions), device=device) >= diff_cfg.get("cfg_dropout", .1)
            loss = diffusion.loss(lambda x, t, c: model(x, t, c, conditional_mask=keep), latents, cond)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            iteration += 1
            if iteration % 1000 == 0:
                print(f"iteration={iteration} loss={loss.item():.6f}")
                torch.save({"iteration": iteration, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg}, out / "latest.pt")
            if iteration >= train_cfg.get("iterations", 100000): break
    torch.save({"iteration": iteration, "model": model.state_dict(), "config": cfg}, out / "final.pt")


if __name__ == "__main__": main()
