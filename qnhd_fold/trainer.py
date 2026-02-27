"""Training utilities for QNHD-Fold."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = object

from .model import QNHDFold


@dataclass
class TrainerConfig:
    lr: float = 1e-4


class QNHDTrainer:
    def __init__(self, model: QNHDFold, config: TrainerConfig = TrainerConfig()):
        if torch is None:
            raise RuntimeError("QNHDTrainer requires PyTorch to be installed.")
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=config.lr)

    def train_epoch(self, dataloader: DataLoader) -> float:
        return float(len(dataloader))
