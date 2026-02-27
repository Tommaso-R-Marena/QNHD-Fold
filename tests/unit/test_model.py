from qnhd_fold.model import QNHDFold


def test_predict_structure_smoke(tmp_path):
    model = QNHDFold()
    pred = model.predict_structure("ACDEFGHIK", num_diffusion_steps=5)
    assert pred.coordinates.shape == (9, 3)
    out = tmp_path / "pred.pdb"
    pred.save(str(out))
    assert out.exists()
