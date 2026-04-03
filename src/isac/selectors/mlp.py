"""Neural selector baselines implemented with PyTorch."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(slots=True)
class MLPClassifierSelector:
    """Small MLP that predicts the best portfolio member directly."""

    n_configs: int
    hidden_dim: int = 16
    epochs: int = 300
    learning_rate: float = 0.03
    l2_penalty: float = 1e-4
    seed: int | None = None
    name: str = "MLP Selector"
    feature_mean_: torch.Tensor = field(init=False)
    feature_scale_: torch.Tensor = field(init=False)
    network_: torch.nn.Module = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
    ) -> MLPClassifierSelector:
        del runtimes
        if self.seed is not None:
            torch.manual_seed(self.seed)

        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        labels = torch.as_tensor(best_configs, dtype=torch.long)
        self.feature_mean_ = feature_tensor.mean(dim=0)
        feature_scale = feature_tensor.std(dim=0, unbiased=False)
        self.feature_scale_ = torch.where(
            feature_scale < 1e-6,
            torch.ones_like(feature_scale),
            feature_scale,
        )
        normalized = self._normalize_tensor(feature_tensor)

        input_dim = normalized.shape[1]
        hidden_dim = max(self.hidden_dim, self.n_configs)
        self.network_ = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.n_configs),
        )
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_penalty,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        self.network_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.network_(normalized)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        return self

    def _normalize_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean_) / self.feature_scale_

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        normalized = self._normalize_tensor(feature_tensor)
        self.network_.eval()
        with torch.no_grad():
            logits = self.network_(normalized)
        return logits.cpu().numpy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(features)
        return logits.argmax(axis=1).astype(np.int64)
