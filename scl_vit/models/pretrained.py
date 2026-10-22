"""Hugging Face CLIP text encoder and Stable Diffusion VAE adapters."""
import torch


class HFTextEncoder:
    def __init__(self, model_id="openai/clip-vit-large-patch14", device="cuda"):
        from transformers import CLIPTextModel, CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.model = CLIPTextModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, captions):
        tokens = self.tokenizer(captions, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
        return self.model(**tokens).last_hidden_state


class DiffusersAutoencoder:
    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device="cuda", dtype=torch.float32):
        from diffusers import AutoencoderKL
        self.model = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device).eval()
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images):
        return self.model.encode(images).latent_dist.sample() * self.model.config.scaling_factor

    @torch.no_grad()
    def decode(self, latents):
        images = self.model.decode(latents / self.model.config.scaling_factor).sample
        return (images.clamp(-1, 1) + 1) / 2
