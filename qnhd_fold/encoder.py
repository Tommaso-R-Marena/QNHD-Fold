"""Pairformer-style encoder with torch implementation and NumPy fallback."""

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

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {a: i for i, a in enumerate(AA)}


@dataclass
class EncoderConfig:
    pair_dim: int = 32
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    msa_mode: bool = True


if TORCH_AVAILABLE:

    class PairformerEncoder(nn.Module):
        def __init__(self, config: Optional[EncoderConfig] = None):
            super().__init__()
            self.config = config or EncoderConfig()
            self.embedding = nn.Embedding(len(AA), self.config.pair_dim)
            self.layers = nn.ModuleList(
                [nn.TransformerEncoderLayer(self.config.pair_dim, self.config.num_heads, batch_first=True) for _ in range(self.config.num_layers)]
            )
            self.norm = nn.LayerNorm(self.config.pair_dim)

        def encode(self, sequence: str, msa: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
            if not sequence:
                raise ValueError("Sequence must be non-empty.")
            d = device or next(self.parameters()).device
            ids = torch.tensor([AA_TO_ID.get(aa, 0) for aa in sequence], device=d)
            res = self.embedding(ids)
            pair = res[:, None, :] + res[None, :, :]
            for layer in self.layers:
                pair = layer(pair)
            return self.norm(pair)

        forward = encode

else:

    class PairformerEncoder:
        def __init__(self, config: Optional[EncoderConfig] = None, seed: int = 7):
            self.config = config or EncoderConfig()
            rng = np.random.default_rng(seed)
            self.proj = rng.normal(0.0, 0.05, size=(20, self.config.pair_dim)).astype(np.float32)

        def _sequence_features(self, sequence: str) -> np.ndarray:
            ids = np.array([AA_TO_ID.get(aa, 0) for aa in sequence], dtype=np.int64)
            onehot = np.eye(20, dtype=np.float32)[ids]
            return onehot @ self.proj

        def _msa_bias(self, msa: Optional[np.ndarray], n: int) -> np.ndarray:
            if msa is None or not self.config.msa_mode:
                return np.zeros((n, n, 1), dtype=np.float32)
            if msa.ndim != 3 or msa.shape[1] != n:
                raise ValueError("MSA must have shape [num_seqs, num_residues, channels].")
            cov = np.einsum("mif,mjf->ij", msa, msa) / max(1, msa.shape[0])
            cov = cov / (np.std(cov) + 1e-6)
            return cov[..., None].astype(np.float32)

        def encode(self, sequence: str, msa: Optional[np.ndarray] = None):
            if not sequence:
                raise ValueError("Sequence must be non-empty.")
            n = len(sequence)
            res = self._sequence_features(sequence)
            pair = res[:, None, :] + res[None, :, :]
            pair = pair + self._msa_bias(msa, n)
            for _ in range(self.config.num_layers):
                tri_out = np.einsum("ikd,kjd->ijd", pair, pair) / np.sqrt(self.config.pair_dim)
                tri_in = np.einsum("kid,kjd->ijd", pair, pair) / np.sqrt(self.config.pair_dim)
                pair = 0.5 * pair + 0.25 * (tri_out + tri_in)
                row_context = pair.mean(axis=1, keepdims=True)
                col_context = pair.mean(axis=0, keepdims=True)
                pair = pair + 0.5 * row_context + 0.5 * col_context
                mean = pair.mean(axis=-1, keepdims=True)
                std = pair.std(axis=-1, keepdims=True) + 1e-6
                pair = (pair - mean) / std
            return pair.astype(np.float32)
