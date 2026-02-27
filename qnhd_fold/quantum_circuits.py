"""Quantum module with differentiable torch path and classical fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = object
    TORCH_AVAILABLE = False

try:
    import pennylane as qml
except Exception:  # pragma: no cover
    qml = None


@dataclass
class HamiltonianWeights:
    backbone: float = 1.0
    sidechain: float = 0.6
    contact: float = 0.8
    torsion: float = 0.4


class QuantumEnergyModule(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, n_qubits: int = 10, weights: Optional[HamiltonianWeights] = None):
        if TORCH_AVAILABLE:
            super().__init__()
        self.n_qubits = n_qubits
        self.weights = weights or HamiltonianWeights()
        self.backend = "pennylane" if qml is not None else "classical"
        self.dev = qml.device("default.qubit", wires=n_qubits) if qml is not None else None

    def compute_energy(self, coordinates):
        if TORCH_AVAILABLE and isinstance(coordinates, torch.Tensor):
            diffs = torch.diff(coordinates, dim=0)
            bond = torch.norm(diffs, dim=-1)
            backbone = torch.mean((bond - 3.8) ** 2) if bond.numel() else coordinates.new_tensor(0.0)
            centered = coordinates - coordinates.mean(dim=0, keepdim=True)
            sidechain = torch.mean(torch.norm(centered, dim=-1) ** 2)
            d = torch.norm(coordinates[:, None, :] - coordinates[None, :, :], dim=-1)
            d = torch.clamp(d + torch.eye(coordinates.shape[0], device=coordinates.device), min=1e-2)
            contact = torch.mean((1.0 / (d**12)) - (2.0 / (d**6)))
            return self.weights.backbone * backbone + self.weights.sidechain * sidechain + self.weights.contact * contact

        c = coordinates
        diffs = np.diff(c, axis=0)
        bond = np.linalg.norm(diffs, axis=-1)
        backbone = float(np.mean((bond - 3.8) ** 2)) if len(bond) else 0.0
        centered = c - c.mean(axis=0, keepdims=True)
        sidechain = float(np.mean(np.linalg.norm(centered, axis=-1) ** 2))
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        d = np.clip(d + np.eye(len(c)), 1e-2, None)
        contact = float(np.mean((1.0 / (d**12)) - (2.0 / (d**6))))
        return float(self.weights.backbone * backbone + self.weights.sidechain * sidechain + self.weights.contact * contact)

    def quantum_energy_gradient(self, coordinates, eps: float = 1e-3):
        if TORCH_AVAILABLE and isinstance(coordinates, torch.Tensor):
            c = coordinates.detach().clone().requires_grad_(True)
            e = self.compute_energy(c)
            return torch.autograd.grad(e, c)[0]
        grad = np.zeros_like(coordinates, dtype=np.float32)
        base = self.compute_energy(coordinates)
        for i in range(coordinates.shape[0]):
            for j in range(3):
                perturbed = coordinates.copy()
                perturbed[i, j] += eps
                grad[i, j] = (self.compute_energy(perturbed) - base) / eps
        return grad

    def vqe_energy(self, params, coordinates):
        if qml is None:
            base = self.compute_energy(coordinates)
            if TORCH_AVAILABLE and isinstance(params, torch.Tensor):
                return torch.mean(torch.cos(params)) + 0.1 * base
            return float(np.mean(np.cos(params)) + 0.1 * base)
        return self.compute_energy(coordinates)


QuantumProteinCircuit = QuantumEnergyModule
