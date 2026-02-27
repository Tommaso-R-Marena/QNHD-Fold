import numpy as np

from qnhd_fold.quantum_circuits import QuantumProteinCircuit


def test_energy_and_gradients():
    q = QuantumProteinCircuit(n_qubits=4)
    coords = np.random.default_rng(0).normal(size=(8, 3)).astype("float32")
    e = q.compute_energy(coords)
    g = q.quantum_energy_gradient(coords)
    assert isinstance(e, float)
    assert g.shape == coords.shape
