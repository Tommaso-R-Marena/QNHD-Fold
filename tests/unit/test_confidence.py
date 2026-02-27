import numpy as np

from qnhd_fold.confidence import ConfidenceHead


def test_confidence_ranges():
    c = ConfidenceHead(ensemble_size=3)
    coords = np.random.default_rng(0).normal(size=(10, 3)).astype("float32")
    pair = np.random.default_rng(1).normal(size=(10, 10, 32)).astype("float32")
    out = c.predict(coords, pair)
    assert out.plddt.min() >= 0
    assert out.plddt.max() <= 100
    assert out.pae.shape == (10, 10)
