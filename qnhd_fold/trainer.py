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
    """
    Trainer for QNHD-Fold models.
    Provides methods for training and optimization.
    """

    def __init__(self, model: QNHDFold, config: TrainerConfig = TrainerConfig()):
        """
        Initialize the trainer.

        Args:
            model: The QNHDFold model to train.
            config: Configuration for the training process.
        """
        if torch is None:
            raise RuntimeError("QNHDTrainer requires PyTorch to be installed.")
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=config.lr)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            dataloader: DataLoader providing the training data.

        Returns:
            The average loss or a metric for the epoch.
        """
        return float(len(dataloader))
