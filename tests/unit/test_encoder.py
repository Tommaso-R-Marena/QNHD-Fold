from qnhd_fold.encoder import PairformerEncoder


def test_encode_shape():
    enc = PairformerEncoder()
    pair = enc.encode("ACDEFG")
    assert pair.shape == (6, 6, enc.config.pair_dim)
