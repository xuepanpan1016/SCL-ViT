import torch


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, clean, t, noise=None):
        noise = torch.randn_like(clean) if noise is None else noise
        alpha_bar = self.alpha_bars.to(clean.device)[t].view(-1, *([1] * (clean.ndim - 1)))
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise, noise

    def loss(self, model, clean_latents, text_embeddings):
        b = clean_latents.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=clean_latents.device)
        noisy, noise = self.q_sample(clean_latents, t)
        prediction = model(noisy, t, text_embeddings)
        return torch.nn.functional.mse_loss(prediction, noise)

    @torch.no_grad()
    def ddim_sample(self, model, shape, text_embeddings, steps=50, guidance_scale=7.5, device="cuda"):
        x = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        null_text = model.null_text.expand(shape[0], -1, -1)
        for i, ti in enumerate(times):
            t = torch.full((shape[0],), int(ti), device=device, dtype=torch.long)
            eps_c = model(x, t, text_embeddings)
            eps_u = model(x, t, null_text)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            ab = self.alpha_bars.to(device)[ti]
            prev_ab = self.alpha_bars.to(device)[times[i + 1]] if i + 1 < len(times) else torch.tensor(1., device=device)
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x = prev_ab.sqrt() * x0 + (1 - prev_ab).sqrt() * eps
        return x
