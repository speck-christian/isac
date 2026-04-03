"""Neural selector baselines implemented with PyTorch."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.selectors.portfolio_learning import (
    assign_to_portfolio,
    derive_kmeans_portfolio,
    portfolio_regret_targets,
)


@dataclass(slots=True)
class MLPClassifierSelector:
    """Cost-sensitive MLP with validation-based regret selection."""

    n_configs: int
    max_portfolio_size: int = 12
    hidden_dim: int = 16
    epochs: int = 320
    learning_rates: tuple[float, ...] = (0.005, 0.01, 0.02)
    l2_penalty: float = 3e-4
    dropout: float = 0.10
    validation_fraction: float = 0.25
    patience: int = 35
    regret_weight: float = 0.75
    classification_weight: float = 0.25
    seed: int | None = None
    name: str = "MLP Selector"
    feature_mean_: torch.Tensor = field(init=False)
    feature_scale_: torch.Tensor = field(init=False)
    network_: torch.nn.Module = field(init=False)
    portfolio_values_: np.ndarray = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
        ideal_params: np.ndarray | None = None,
    ) -> MLPClassifierSelector:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if ideal_params is not None:
            self.portfolio_values_ = derive_kmeans_portfolio(
                ideal_params,
                max_portfolio_size=self.max_portfolio_size,
                seed=self.seed,
            )
            runtimes = portfolio_regret_targets(ideal_params, self.portfolio_values_)
            best_configs = assign_to_portfolio(ideal_params, self.portfolio_values_)
        else:
            self.portfolio_values_ = np.eye(self.n_configs, dtype=np.float64)

        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        runtime_tensor = torch.as_tensor(runtimes, dtype=torch.float32)
        labels = torch.as_tensor(best_configs, dtype=torch.long)

        self.feature_mean_ = feature_tensor.mean(dim=0)
        feature_scale = feature_tensor.std(dim=0, unbiased=False)
        self.feature_scale_ = torch.where(
            feature_scale < 1e-6,
            torch.ones_like(feature_scale),
            feature_scale,
        )
        normalized = self._normalize_tensor(feature_tensor)

        split = self._make_train_validation_split(normalized, runtime_tensor, labels)
        train_inputs, train_runtimes, train_labels, val_inputs, val_runtimes, val_labels = split

        input_dim = normalized.shape[1]
        best_state: dict[str, torch.Tensor] | None = None
        best_regret = float("inf")

        for learning_rate in self.learning_rates:
            network = self._build_network(input_dim)
            optimizer = torch.optim.Adam(
                network.parameters(),
                lr=learning_rate,
                weight_decay=self.l2_penalty,
            )
            cross_entropy = torch.nn.CrossEntropyLoss()
            patience_left = self.patience
            candidate_state = {
                key: value.detach().clone() for key, value in network.state_dict().items()
            }
            candidate_regret = float("inf")

            for _ in range(self.epochs):
                network.train()
                optimizer.zero_grad()
                logits = network(train_inputs)
                probabilities = torch.softmax(logits, dim=1)
                normalized_regrets = self._normalized_regrets(train_runtimes)
                expected_regret = (probabilities * normalized_regrets).sum(dim=1).mean()
                classification_loss = cross_entropy(logits, train_labels)
                loss = (
                    self.regret_weight * expected_regret
                    + self.classification_weight * classification_loss
                )
                loss.backward()
                optimizer.step()

                network.eval()
                with torch.no_grad():
                    validation_logits = network(val_inputs)
                    validation_predictions = validation_logits.argmax(dim=1)
                    validation_regret = self._average_regret(
                        predictions=validation_predictions,
                        runtimes=val_runtimes,
                    )

                if validation_regret + 1e-6 < candidate_regret:
                    candidate_regret = validation_regret
                    candidate_state = {
                        key: value.detach().clone()
                        for key, value in network.state_dict().items()
                    }
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

            if candidate_regret < best_regret:
                best_regret = candidate_regret
                best_state = candidate_state

        self.network_ = self._build_network(input_dim)
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.network_.eval()
        return self

    def _build_network(self, input_dim: int) -> torch.nn.Sequential:
        output_dim = len(self.portfolio_values_)
        hidden_dim = max(self.hidden_dim, output_dim)
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def _make_train_validation_split(
        self,
        features: torch.Tensor,
        runtimes: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_instances = features.shape[0]
        if n_instances < 5:
            return features, runtimes, labels, features, runtimes, labels

        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        permutation = torch.randperm(n_instances, generator=generator)
        val_count = max(1, int(round(self.validation_fraction * n_instances)))
        if val_count >= n_instances:
            val_count = n_instances - 1
        val_indices = permutation[:val_count]
        train_indices = permutation[val_count:]
        return (
            features[train_indices],
            runtimes[train_indices],
            labels[train_indices],
            features[val_indices],
            runtimes[val_indices],
            labels[val_indices],
        )

    def _normalized_regrets(self, runtimes: torch.Tensor) -> torch.Tensor:
        oracle = runtimes.min(dim=1, keepdim=True).values
        regret = runtimes - oracle
        scale = regret.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        return regret / scale

    def _average_regret(self, predictions: torch.Tensor, runtimes: torch.Tensor) -> float:
        chosen = runtimes[torch.arange(runtimes.shape[0]), predictions]
        oracle = runtimes.min(dim=1).values
        return float((chosen - oracle).mean().item())

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
