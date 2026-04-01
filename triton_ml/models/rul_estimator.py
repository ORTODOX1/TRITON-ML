"""
Remaining Useful Life (RUL) estimator using PyTorch DNN.

Predicts hours until next required maintenance action, with
epistemic uncertainty quantified via Monte-Carlo dropout.
Calibrated against hull machinery run-to-failure datasets
(NASA C-MAPSS adapted to marine diesel duty cycles).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional

import torch
import torch.nn as nn

from triton_ml.config import Settings


class _RULNetwork(nn.Module):
    """Feedforward DNN with dropout for MC uncertainty estimation."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RULEstimator:
    """Remaining useful life predictor with uncertainty bounds."""

    def __init__(self, input_dim: int = 12, settings: Optional[Settings] = None) -> None:
        self._cfg = settings or Settings()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _RULNetwork(
            input_dim,
            self._cfg.model.rul_hidden_dim,
            self._cfg.model.rul_dropout,
        ).to(self._device)

    def train(self, X: NDArray[np.float64], y: NDArray[np.float64],
              epochs: int = 100, lr: float = 1e-3) -> None:
        """Train the RUL network on feature/target pairs."""
        self._model.train()
        optimiser = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimiser.zero_grad()
                loss_fn(self._model(xb), yb).backward()
                optimiser.step()

    def predict(self, X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (mean_rul_hours, std_rul_hours) via MC-dropout sampling."""
        self._model.train()  # keep dropout active for MC sampling
        xt = torch.tensor(X, dtype=torch.float32).to(self._device)
        preds = []
        with torch.no_grad():
            for _ in range(self._cfg.model.rul_mc_samples):
                preds.append(self._model(xt).cpu().numpy())
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def get_model(self) -> nn.Module:
        """Return the underlying PyTorch model for export."""
        return self._model
