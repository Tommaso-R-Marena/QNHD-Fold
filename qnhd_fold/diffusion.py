"""Dual-score diffusion with torch implementation and NumPy fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = object
    TORCH_AVAILABLE = False

NoiseSchedule = Literal["linear", "cosine", "quadratic"]
Sampler = Literal["ddpm", "ddim"]


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    noise_schedule: NoiseSchedule = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02


class DualScoreDiffusion(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, config: Optional[DiffusionConfig] = None):
        if TORCH_AVAILABLE:
            super().__init__()
        self.config = config or DiffusionConfig()
        self.betas = self._build_beta_schedule(self.config.noise_schedule)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas) if isinstance(self.betas, np.ndarray) else torch.cumprod(self.alphas, 0)

    def _build_beta_schedule(self, name: NoiseSchedule):
        lib = torch if TORCH_AVAILABLE else np
        t = lib.linspace(0, 1, self.config.num_timesteps, dtype=lib.float32)
        if name == "linear":
            betas = self.config.beta_start + t * (self.config.beta_end - self.config.beta_start)
        elif name == "quadratic":
            betas = self.config.beta_start + (t**2) * (self.config.beta_end - self.config.beta_start)
        elif name == "cosine":
            s = 0.008
            f = lib.cos(((t + s) / (1 + s)) * lib.pi / 2) ** 2
            alpha_bar = f / f[0]
            alpha_bar_next = lib.concatenate([alpha_bar[1:], alpha_bar[-1:]]) if lib is np else torch.cat([alpha_bar[1:], alpha_bar[-1:]])
            betas = 1 - (alpha_bar_next / (alpha_bar + 1e-8))
            betas = lib.clip(betas, self.config.beta_start, self.config.beta_end)
        else:
            raise ValueError(f"Unknown noise schedule: {name}")
        return betas.astype(np.float32) if lib is np else betas.to(dtype=torch.float32)

    def lambda_schedule(self, t: int) -> float:
        return float(0.3 * (1 - t / max(1, self.config.num_timesteps)))

    def fuse_scores(self, neural_score, quantum_score, t: int):
        lam = self.lambda_schedule(t)
        return (1 - lam) * neural_score + lam * quantum_score

    def forward_diffuse(self, x0, t: int, noise=None):
        a = self.alpha_bars[t]
        if TORCH_AVAILABLE and isinstance(x0, torch.Tensor):
            noise = noise if noise is not None else torch.randn_like(x0)
            return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

        noise = noise if noise is not None else np.random.normal(size=x0.shape).astype(np.float32)
        a_val = float(a)
        # Use np.asarray to ensure result is a standard numpy array and avoid __array_wrap__ warnings in NumPy 2.0
        return np.asarray(np.sqrt(a_val) * x0 + np.sqrt(1 - a_val) * noise, dtype=np.float32)

    def reverse_step(self, xt, t: int, neural_score_fn: Callable, quantum_score_fn: Callable, sampler: Sampler = "ddpm", eta: float = 0.0, t_prev: Optional[int] = None):
        neural_score = neural_score_fn(xt, t)
        quantum_score = quantum_score_fn(xt, t)
        score = self.fuse_scores(neural_score, quantum_score, t)

        is_torch = TORCH_AVAILABLE and isinstance(xt, torch.Tensor)
        if is_torch:
            score = torch.as_tensor(score, device=xt.device, dtype=xt.dtype)

        beta_t, alpha_t, alpha_bar_t = self.betas[t], self.alphas[t], self.alpha_bars[t]
        if t_prev is None:
            t_prev = t - 1
        alpha_bar_prev = self.alpha_bars[t_prev] if t_prev >= 0 else (torch.tensor(1.0, device=xt.device, dtype=xt.dtype) if is_torch else 1.0)

        if is_torch:
            beta_t = torch.as_tensor(beta_t, device=xt.device, dtype=xt.dtype)
            alpha_t = torch.as_tensor(alpha_t, device=xt.device, dtype=xt.dtype)
            alpha_bar_t = torch.as_tensor(alpha_bar_t, device=xt.device, dtype=xt.dtype)
            alpha_bar_prev = torch.as_tensor(alpha_bar_prev, device=xt.device, dtype=xt.dtype)
        else:
            beta_t, alpha_t, alpha_bar_t = float(beta_t), float(alpha_t), float(alpha_bar_t)
            alpha_bar_prev = float(alpha_bar_prev)

        sqrt = torch.sqrt if is_torch else np.sqrt
        randn = (lambda x: torch.randn_like(x)) if is_torch else (lambda x: np.random.normal(size=x.shape).astype(np.float32))

        eps_pred = -score * sqrt(1 - alpha_bar_t)
        x0_pred = (xt - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

        if sampler == "ddpm":
            mean = (xt - (beta_t / sqrt(1 - alpha_bar_t)) * eps_pred) / sqrt(alpha_t)
            if t == 0:
                return mean
            return mean + sqrt(beta_t) * randn(xt)
        elif sampler == "ddim":
            sigma_t = eta * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t / alpha_bar_prev)
            dir_xt = sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred
            xt_prev = sqrt(alpha_bar_prev) * x0_pred + dir_xt
            if t > 0 and eta > 0:
                xt_prev += sigma_t * randn(xt)
            return xt_prev

        raise ValueError(f"Unknown sampler: {sampler}")

    def sample(self, shape: tuple[int, ...], neural_score_fn: Callable, quantum_score_fn: Callable, num_steps: Optional[int] = None, sampler: Sampler = "ddpm", eta: float = 0.0):
        total_steps = self.config.num_timesteps
        sample_steps = num_steps or total_steps

        # Linear spacing for timesteps
        timesteps = np.linspace(total_steps - 1, 0, sample_steps).astype(int)

        xt = torch.randn(shape) if TORCH_AVAILABLE else np.random.normal(size=shape).astype(np.float32)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            xt = self.reverse_step(xt, t, neural_score_fn, quantum_score_fn, sampler=sampler, eta=eta, t_prev=t_prev)
        return xt
