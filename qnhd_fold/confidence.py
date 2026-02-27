"""Confidence heads for pLDDT, PAE, and epistemic uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False


@dataclass
class ConfidenceOutputs:
    plddt: np.ndarray
    pae: np.ndarray
    epistemic_uncertainty: np.ndarray
    ptm: float


class ConfidenceHead:
    def __init__(self, ensemble_size: int = 5):
        self.ensemble_size = ensemble_size

    def _to_numpy(self, x):
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def predict(self, coordinates, pair_repr) -> ConfidenceOutputs:
        coordinates = self._to_numpy(coordinates)
        pair_repr = self._to_numpy(pair_repr)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Coordinates must have shape [N,3].")
        distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
        local_compactness = np.exp(-distances.mean(axis=1) / 10.0)
        pair_signal = np.tanh(np.abs(pair_repr).mean(axis=(1, 2)))

        raw = 0.6 * local_compactness + 0.4 * pair_signal
        plddt = np.clip(100 * raw, 0, 100).astype(np.float32)
        pae = np.clip(distances / 2.0, 0, 31.0).astype(np.float32)

        ensemble = []
        for i in range(self.ensemble_size):
            scale = 1.0 + 0.02 * (i - self.ensemble_size // 2)
            ensemble.append(np.clip(plddt * scale, 0, 100))
        epistemic = np.stack(ensemble, axis=0).std(axis=0).astype(np.float32)
        ptm = float(np.clip(1.0 - pae.mean() / 31.0, 0.0, 1.0))
        return ConfidenceOutputs(plddt=plddt, pae=pae, epistemic_uncertainty=epistemic, ptm=ptm)

    def to_dict(self, outputs: ConfidenceOutputs) -> Dict[str, np.ndarray | float]:
        return {
            "plddt": outputs.plddt,
            "pae": outputs.pae,
            "epistemic_uncertainty": outputs.epistemic_uncertainty,
            "ptm": outputs.ptm,
        }
