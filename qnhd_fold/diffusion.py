"""Dual-score diffusion process for QNHD-Fold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

NoiseSchedule = Literal["linear", "cosine", "quadratic"]
Sampler = Literal["ddpm", "ddim"]


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    noise_schedule: NoiseSchedule = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02


class DualScoreDiffusion:
    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.betas = self._build_beta_schedule(self.config.noise_schedule)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def _build_beta_schedule(self, name: NoiseSchedule) -> np.ndarray:
        t = np.linspace(0, 1, self.config.num_timesteps, dtype=np.float32)
        if name == "linear":
            betas = self.config.beta_start + t * (self.config.beta_end - self.config.beta_start)
        elif name == "quadratic":
            betas = self.config.beta_start + (t**2) * (self.config.beta_end - self.config.beta_start)
        elif name == "cosine":
            s = 0.008
            f = np.cos(((t + s) / (1 + s)) * np.pi / 2) ** 2
            alpha_bar = f / f[0]
            alpha_bar_next = np.concatenate([alpha_bar[1:], alpha_bar[-1:]])
            betas = 1 - (alpha_bar_next / (alpha_bar + 1e-8))
            betas = np.clip(betas, self.config.beta_start, self.config.beta_end)
        else:
            raise ValueError(f"Unknown noise schedule: {name}")
        return betas.astype(np.float32)

    def lambda_schedule(self, t: int) -> float:
        return float(0.3 * (1 - t / max(1, self.config.num_timesteps)))

    def fuse_scores(self, neural_score: np.ndarray, quantum_score: np.ndarray, t: int) -> np.ndarray:
        lam = self.lambda_schedule(t)
        return (1 - lam) * neural_score + lam * quantum_score

    def forward_diffuse(self, x0: np.ndarray, t: int, noise: Optional[np.ndarray] = None) -> np.ndarray:
        noise = noise if noise is not None else np.random.normal(size=x0.shape).astype(np.float32)
        alpha_bar = self.alpha_bars[t]
        return np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise

    def reverse_step(
        self,
        xt: np.ndarray,
        t: int,
        neural_score_fn: Callable[[np.ndarray, int], np.ndarray],
        quantum_score_fn: Callable[[np.ndarray, int], np.ndarray],
        sampler: Sampler = "ddpm",
        eta: float = 0.0,
    ) -> np.ndarray:
        neural_score = neural_score_fn(xt, t)
        quantum_score = quantum_score_fn(xt, t)
        score = self.fuse_scores(neural_score, quantum_score, t)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        eps = -score * np.sqrt(1 - alpha_bar_t)

        mean = (xt - (beta_t / np.sqrt(1 - alpha_bar_t)) * eps) / np.sqrt(alpha_t)

        if t == 0:
            return mean.astype(np.float32)

        if sampler == "ddpm":
            noise = np.random.normal(size=xt.shape).astype(np.float32)
            sigma = np.sqrt(beta_t)
            return (mean + sigma * noise).astype(np.float32)

        if sampler == "ddim":
            alpha_bar_prev = self.alpha_bars[max(0, t - 1)]
            sigma = (
                eta
                * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
                * np.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            )
            noise = np.random.normal(size=xt.shape).astype(np.float32)
            pred_x0 = (xt - np.sqrt(1 - alpha_bar_t) * eps) / np.sqrt(alpha_bar_t)
            dir_xt = np.sqrt(1 - alpha_bar_prev - sigma**2) * eps
            return (np.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise).astype(np.float32)

        raise ValueError(f"Unknown sampler: {sampler}")

    def sample(
        self,
        shape: tuple[int, ...],
        neural_score_fn: Callable[[np.ndarray, int], np.ndarray],
        quantum_score_fn: Callable[[np.ndarray, int], np.ndarray],
        num_steps: Optional[int] = None,
        sampler: Sampler = "ddpm",
    ) -> np.ndarray:
        steps = num_steps or self.config.num_timesteps
        xt = np.random.normal(size=shape).astype(np.float32)
        for t in reversed(range(steps)):
            xt = self.reverse_step(xt, t, neural_score_fn, quantum_score_fn, sampler=sampler)
        return xt
