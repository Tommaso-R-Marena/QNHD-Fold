import numpy as np
import pytest
from qnhd_fold.diffusion import DiffusionConfig, DualScoreDiffusion

def test_ddim_sampler():
    d = DualScoreDiffusion(DiffusionConfig(num_timesteps=8, noise_schedule="linear"))
    x0 = np.zeros((5, 5, 3), dtype=np.float32)
    xt = d.forward_diffuse(x0, 3)

    def score(x, t):
        return np.zeros_like(x)

    # Test DDIM with eta=0
    xprev_ddim = d.reverse_step(xt, 3, score, score, sampler="ddim", eta=0.0)
    assert xprev_ddim.shape == x0.shape

    # Test DDIM with eta=1.0
    xprev_ddim_noise = d.reverse_step(xt, 3, score, score, sampler="ddim", eta=1.0)
    assert xprev_ddim_noise.shape == x0.shape

if __name__ == "__main__":
    pytest.main([__file__])
