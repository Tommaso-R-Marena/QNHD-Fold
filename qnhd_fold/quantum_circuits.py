"""Quantum module with VQE-inspired energy and classical fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import pennylane as qml
except Exception:  # pragma: no cover - optional dependency
    qml = None


@dataclass
class HamiltonianWeights:
    backbone: float = 1.0
    sidechain: float = 0.6
    contact: float = 0.8
    torsion: float = 0.4


class QuantumProteinCircuit:
    """Protein-aware quantum energy model.

    Uses PennyLane when available; otherwise runs a deterministic classical
    approximation for reproducible tests and CPU-only environments.
    """

    def __init__(self, n_qubits: int = 10, weights: Optional[HamiltonianWeights] = None):
        self.n_qubits = n_qubits
        self.weights = weights or HamiltonianWeights()
        self.backend = "pennylane" if qml is not None else "classical"
        self.dev = qml.device("default.qubit", wires=n_qubits) if qml is not None else None

    def h_backbone(self, coordinates: np.ndarray) -> float:
        diffs = np.diff(coordinates, axis=0)
        bond = np.linalg.norm(diffs, axis=-1)
        return float(np.mean((bond - 3.8) ** 2)) if len(bond) else 0.0

    def h_sidechain(self, coordinates: np.ndarray) -> float:
        centered = coordinates - coordinates.mean(axis=0, keepdims=True)
        return float(np.mean(np.linalg.norm(centered, axis=-1) ** 2))

    def h_contact(self, coordinates: np.ndarray) -> float:
        d = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
        d = np.clip(d + np.eye(len(coordinates)), 1e-2, None)
        lj = (1.0 / (d**12)) - (2.0 / (d**6))
        return float(np.mean(lj))

    def h_torsion(self, coordinates: np.ndarray) -> float:
        if len(coordinates) < 4:
            return 0.0
        v1 = coordinates[1:-2] - coordinates[:-3]
        v2 = coordinates[2:-1] - coordinates[1:-2]
        v3 = coordinates[3:] - coordinates[2:-1]
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        n1 /= np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-6
        n2 /= np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-6
        cosang = np.sum(n1 * n2, axis=-1)
        return float(np.mean(1 - np.clip(cosang, -1.0, 1.0)))

    def compute_energy(self, coordinates: np.ndarray) -> float:
        return (
            self.weights.backbone * self.h_backbone(coordinates)
            + self.weights.sidechain * self.h_sidechain(coordinates)
            + self.weights.contact * self.h_contact(coordinates)
            + self.weights.torsion * self.h_torsion(coordinates)
        )

    def quantum_energy_gradient(self, coordinates: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        grad = np.zeros_like(coordinates, dtype=np.float32)
        base = self.compute_energy(coordinates)
        for i in range(coordinates.shape[0]):
            for j in range(3):
                perturbed = coordinates.copy()
                perturbed[i, j] += eps
                grad[i, j] = (self.compute_energy(perturbed) - base) / eps
        return grad

    def vqe_energy(self, params: np.ndarray, coordinates: np.ndarray) -> float:
        if qml is None:
            return float(np.mean(np.cos(params)) + 0.1 * self.compute_energy(coordinates))

        @qml.qnode(self.dev)
        def circuit(theta: np.ndarray) -> float:
            for i in range(min(len(theta), self.n_qubits)):
                qml.RY(theta[i], wires=i)
                qml.RZ(theta[i] * 0.5, wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        return float(circuit(params[: self.n_qubits])) + 0.1 * self.compute_energy(coordinates)
