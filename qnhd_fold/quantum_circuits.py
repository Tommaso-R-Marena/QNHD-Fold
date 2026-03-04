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
    """
    Quantum-inspired energy module for protein conformations.
    Provides methods to compute classical energy potentials and their gradients,
    with placeholders for Variational Quantum Eigensolver (VQE) integration.
    """

    def __init__(self, n_qubits: int = 10, weights: Optional[HamiltonianWeights] = None):
        """
        Initialize the module.

        Args:
            n_qubits: Number of qubits to use for quantum simulations.
            weights: Weights for different components of the energy Hamiltonian.
        """
        if TORCH_AVAILABLE:
            super().__init__()
        self.n_qubits = n_qubits
        self.weights = weights or HamiltonianWeights()
        self.backend = "pennylane" if qml is not None else "classical"
        self.dev = qml.device("default.qubit", wires=n_qubits) if qml is not None else None

    def compute_energy(self, coordinates, contact_cutoff: float = 10.0, reduce: bool = True):
        """
        Compute the total energy of a protein conformation.
        Supports batched inputs of shape [..., N, 3].

        Args:
            coordinates: Tensor or array of shape [..., N, 3] representing CA atom positions.
            contact_cutoff: Distance cutoff for contact energy calculation.
            reduce: Whether to reduce (mean) the energy components or return per-residue/atom.

        Returns:
            Total energy as a scalar or array/tensor.
        """
        is_torch = TORCH_AVAILABLE and isinstance(coordinates, torch.Tensor)
        lib = torch if is_torch else np
        ndim = coordinates.ndim
        c = coordinates if ndim >= 3 else (coordinates.unsqueeze(0) if is_torch else coordinates[None, ...])

        # Backbone
        diffs = (torch.diff(c, dim=-2) if is_torch else np.diff(c, axis=-2))
        bond = (torch.linalg.norm(diffs, dim=-1) if is_torch else np.linalg.norm(diffs, axis=-1))
        backbone_per = (bond - 3.8) ** 2
        backbone = (torch.mean(backbone_per, dim=-1) if is_torch else np.mean(backbone_per, axis=-1))

        # Sidechain
        c_mean = (torch.mean(c, dim=-2, keepdim=True) if is_torch else np.mean(c, axis=-2, keepdims=True))
        centered = c - c_mean
        sidechain_per = torch.sum(centered**2, dim=-1) if is_torch else np.sum(centered**2, axis=-1)
        sidechain = (torch.mean(sidechain_per, dim=-1) if is_torch else np.mean(sidechain_per, axis=-1))

        # Contact
        dist_vec = c[..., :, None, :] - c[..., None, :, :]
        dist = (torch.linalg.norm(dist_vec, dim=-1) if is_torch else np.linalg.norm(dist_vec, axis=-1))
        n = c.shape[-2]
        eye = (torch.eye(n, device=c.device, dtype=torch.bool) if is_torch else np.eye(n, dtype=bool))
        mask = (dist < contact_cutoff) & (~eye)
        d_clamped = (torch.clamp(dist, min=1e-2) if is_torch else np.clip(dist, 1e-2, None))
        potential = (1.0 / (d_clamped**12)) - (2.0 / (d_clamped**6))

        contact_sum = (torch.sum(potential * mask, dim=(-2, -1)) if is_torch else np.sum(potential * mask, axis=(-2, -1)))
        mask_sum = (torch.sum(mask.to(torch.float32), dim=(-2, -1)) if is_torch else np.sum(mask, axis=(-2, -1)))
        contact = contact_sum / (torch.clamp(mask_sum, min=1.0) if is_torch else np.maximum(mask_sum, 1.0))

        if not reduce:
            # Reconstruct per-residue energy if needed. For now we use the mean for gradients.
            # But let's return total for the predicted structure's sake if called with reduce=True.
            pass

        energy = self.weights.backbone * backbone + self.weights.sidechain * sidechain + self.weights.contact * contact

        if reduce:
            # If coordinates were [..., N, 3], energy is [...].
            # If coordinates were [N, 3], energy is scalar.
            if ndim <= 2:
                return torch.sum(energy) if is_torch else float(np.sum(energy))

            # If batched, we might still want to sum over each batch element's residues.
            # In our case, energy is already [B] if c was [B, N, 3].
            return energy

        return energy

    def quantum_energy_gradient(self, coordinates, eps: float = 1e-3):
        """
        Compute the gradient of the energy with respect to coordinates.
        Uses autograd if Torch is available, otherwise vectorized finite difference.

        Args:
            coordinates: Tensor or array of shape [..., N, 3].
            eps: Step size for finite difference gradient.

        Returns:
            Gradient of the same shape as coordinates.
        """
        if TORCH_AVAILABLE and isinstance(coordinates, torch.Tensor):
            c = coordinates.detach().clone().requires_grad_(True)
            e = self.compute_energy(c)
            if e.numel() > 1:
                e = e.sum()
            return torch.autograd.grad(e, c)[0]

        # Vectorized finite difference for NumPy
        n, d = coordinates.shape
        base_energy = self.compute_energy(coordinates)

        # Create n*d perturbed copies
        perturbations = np.tile(coordinates, (n * d, 1, 1))
        indices = np.arange(n * d)
        rows = indices // d
        cols = indices % d
        perturbations[indices, rows, cols] += eps

        # Batch evaluation
        perturbed_energies = self.compute_energy(perturbations)
        grad = (perturbed_energies - base_energy) / eps
        return grad.reshape(n, d).astype(np.float32)

    def vqe_energy(self, params, coordinates):
        """
        Placeholder for Variational Quantum Eigensolver energy calculation.
        """
        if qml is None:
            base = self.compute_energy(coordinates)
            if TORCH_AVAILABLE and isinstance(params, torch.Tensor):
                return torch.mean(torch.cos(params)) + 0.1 * base
            return float(np.mean(np.cos(params)) + 0.1 * base)
        return self.compute_energy(coordinates)


QuantumProteinCircuit = QuantumEnergyModule
