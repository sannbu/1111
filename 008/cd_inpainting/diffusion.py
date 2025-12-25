from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def make_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Extract values from a tensor for a batch of indices t and reshape for broadcasting."""
    b = t.shape[0]
    out = a.gather(-1, t.to(a.device))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Diffusion:
    """DDPM utilities for conditional diffusion."""

    def __init__(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu"):
        self.timesteps = timesteps
        self.device = device
        betas = make_beta_schedule(timesteps, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_hat = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_hat = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

    def p_mean_variance(self, model, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eps_theta = model(x_t, cond, t)
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha_hat = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        alpha_t = extract(self.alphas, t, x_t.shape)
        alpha_hat_t = extract(self.alphas_cumprod, t, x_t.shape)

        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - beta_t / sqrt_one_minus_alpha_hat * eps_theta)
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean, var

    def p_sample(self, model, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, var = self.p_mean_variance(model, x_t, cond, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        sample = mean + nonzero_mask * torch.sqrt(var) * noise
        return sample, mean

    def p_sample_inpaint(
        self,
        model,
        x_t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        background: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample, mean = self.p_sample(model, x_t, cond, t)
        # enforce background outside mask using noise level for current t
        bg_noisy = self.q_sample(background, t)
        sample = mask * sample + (1.0 - mask) * bg_noisy
        return sample, mean

    def p_losses(self, model, x0: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        predicted_noise = model(x_noisy, cond, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample_inpaint(
        self,
        model,
        cond: torch.Tensor,
        mask: torch.Tensor,
        background: torch.Tensor,
        img_size: int,
        device: str,
        timesteps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        model.eval()
        ts = timesteps if timesteps is not None else self.timesteps
        if seed is not None:
            torch.manual_seed(seed)
        b = background.shape[0]
        t_start = torch.full((b,), ts - 1, device=device, dtype=torch.long)
        x_t = self.q_sample(background, t_start)
        for i in reversed(range(ts)):
            t_step = torch.full((b,), i, device=device, dtype=torch.long)
            x_t, _ = self.p_sample_inpaint(model, x_t, cond, mask, background, t_step)
        return x_t
