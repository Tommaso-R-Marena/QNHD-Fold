"""Main QNHD-Fold model pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False

from .confidence import ConfidenceHead, ConfidenceOutputs
from .diffusion import DiffusionConfig, DualScoreDiffusion
from .encoder import EncoderConfig, PairformerEncoder
from .quantum_circuits import QuantumEnergyModule


@dataclass
class StructurePrediction:
    sequence: str
    coordinates: np.ndarray
    confidence: ConfidenceOutputs

    def save(self, path: str) -> None:
        out = Path(path)
        with out.open("w", encoding="utf-8") as handle:
            for i, (aa, xyz) in enumerate(zip(self.sequence, self.coordinates), start=1):
                x, y, z = xyz.tolist()
                handle.write(
                    f"ATOM  {i:5d}  CA  {aa:>3s} A{i:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C\n"
                )


class QNHDFold:
    def __init__(self, config: Optional[DiffusionConfig] = None, device: Optional[str] = None):
        self.config = config or DiffusionConfig()
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.encoder = PairformerEncoder(EncoderConfig())
        self.diffusion = DualScoreDiffusion(self.config)
        self.quantum = QuantumEnergyModule(n_qubits=10)
        self.confidence = ConfidenceHead()

    def _log(self, message: str, verbose: bool) -> None:
        if verbose:
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] {message}")

    def _validate_sequence(self, sequence: str) -> None:
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Sequence must be a non-empty string.")
        if len(sequence) > 2000:
            raise ValueError("Sequence too long; max supported length is 2000 residues.")

    def predict_structure(
        self,
        sequence: str,
        num_diffusion_steps: int = 200,
        verbose: bool = False,
        batch_size: int = 256,
        use_mixed_precision: bool = True,
    ) -> StructurePrediction:
        self._validate_sequence(sequence)
        n = len(sequence)
        self._log(f"Encoding sequence with {n} residues", verbose)

        pair_repr = self.encoder.encode(sequence)

        def neural_score_fn(xt, t: int):
            pair_mean = pair_repr.mean(axis=-1)
            pair_grad = pair_mean[:, :, None] - pair_mean.mean()
            return -0.1 * xt + 0.01 * pair_grad

        def quantum_score_fn(xt, t: int):
            return -self.quantum.quantum_energy_gradient(xt)

        self._log("Running reverse diffusion", verbose)
        coords = np.zeros((n, 3), dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            shape = (end - start, end - start, 3)
            block = self.diffusion.sample(shape, neural_score_fn, quantum_score_fn, num_steps=num_diffusion_steps)
            if TORCH_AVAILABLE and isinstance(block, torch.Tensor):
                block = block.detach().cpu().numpy()
            coords[start:end] = block.mean(axis=1)

        self._log("Computing confidence outputs", verbose)
        conf = self.confidence.predict(coords, pair_repr)
        return StructurePrediction(sequence=sequence, coordinates=coords, confidence=conf)

    def predict_batch(self, sequences: list[str], **kwargs) -> list[StructurePrediction]:
        return [self.predict_structure(seq, **kwargs) for seq in sequences]

    def save_checkpoint(self, path: str) -> None:
        np.savez(path, betas=np.array(self.diffusion.betas), n_qubits=self.quantum.n_qubits)

    def load_checkpoint(self, path: str) -> None:
        data = np.load(path)
        self.diffusion.betas = data["betas"]
        self.diffusion.alphas = 1.0 - self.diffusion.betas
        self.diffusion.alpha_bars = np.cumprod(self.diffusion.alphas)
