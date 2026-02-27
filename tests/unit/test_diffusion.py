import numpy as np

from qnhd_fold.diffusion import DiffusionConfig, DualScoreDiffusion


def test_lambda_schedule_decreases():
    d = DualScoreDiffusion(DiffusionConfig(num_timesteps=10))
    assert d.lambda_schedule(0) > d.lambda_schedule(9)


def test_forward_reverse_shapes():
    d = DualScoreDiffusion(DiffusionConfig(num_timesteps=8, noise_schedule="linear"))
    x0 = np.zeros((5, 5, 3), dtype=np.float32)
    xt = d.forward_diffuse(x0, 3)

    def score(x, t):
        return np.zeros_like(x)

    xprev = d.reverse_step(xt, 3, score, score)
    assert xprev.shape == x0.shape
