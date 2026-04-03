"""Neural selector baselines implemented with PyTorch."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.core import AlgorithmicEpisode, DynamicPortfolioEpisode
from isac.selectors.portfolio_learning import (
    derive_kmeans_portfolio,
    portfolio_regret_targets,
)


@dataclass(slots=True)
class MLPClassifierSelector:
    """MLP loss predictor that routes over a learned portfolio."""

    n_configs: int
    max_portfolio_size: int = 12
    hidden_dim: int = 24
    epochs: int = 320
    learning_rates: tuple[float, ...] = (0.003, 0.007, 0.015)
    l2_penalty: float = 3e-4
    dropout: float = 0.10
    validation_fraction: float = 0.25
    patience: int = 35
    runtime_weight: float = 0.80
    regret_weight: float = 0.20
    history_blend: float = 0.72
    trend_blend: float = 0.45
    switching_penalty_weight: float = 0.65
    softmin_temperature: float = 2.2
    seed: int | None = None
    name: str = "MLP Selector"
    feature_mean_: torch.Tensor = field(init=False)
    feature_scale_: torch.Tensor = field(init=False)
    network_: torch.nn.Module = field(init=False)
    portfolio_values_: np.ndarray = field(init=False)
    temporal_mode_: bool = field(init=False, default=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
        ideal_params: np.ndarray | None = None,
    ) -> MLPClassifierSelector:
        del best_configs
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if ideal_params is not None:
            self.portfolio_values_ = derive_kmeans_portfolio(
                ideal_params,
                max_portfolio_size=self.max_portfolio_size,
                seed=self.seed,
            )
            runtimes = portfolio_regret_targets(ideal_params, self.portfolio_values_)
        else:
            self.portfolio_values_ = np.eye(self.n_configs, dtype=np.float64)

        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        runtime_tensor = torch.as_tensor(runtimes, dtype=torch.float32)

        self.feature_mean_ = feature_tensor.mean(dim=0)
        feature_scale = feature_tensor.std(dim=0, unbiased=False)
        self.feature_scale_ = torch.where(
            feature_scale < 1e-6,
            torch.ones_like(feature_scale),
            feature_scale,
        )
        normalized = self._normalize_tensor(feature_tensor)

        train_inputs, train_runtimes, val_inputs, val_runtimes = self._make_train_validation_split(
            normalized,
            runtime_tensor,
        )

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
            mse_loss = torch.nn.MSELoss()
            patience_left = self.patience
            candidate_state = {
                key: value.detach().clone() for key, value in network.state_dict().items()
            }
            candidate_regret = float("inf")

            for _ in range(self.epochs):
                network.train()
                optimizer.zero_grad()
                runtime_predictions = network(train_inputs)
                normalized_regrets = self._normalized_regrets(train_runtimes)
                routing_weights = torch.softmax(
                    -self.softmin_temperature * runtime_predictions,
                    dim=1,
                )
                expected_regret = (routing_weights * normalized_regrets).sum(dim=1).mean()
                loss = (
                    self.runtime_weight * mse_loss(runtime_predictions, train_runtimes)
                    + self.regret_weight * expected_regret
                )
                loss.backward()
                optimizer.step()

                network.eval()
                with torch.no_grad():
                    validation_predictions = network(val_inputs).argmin(dim=1)
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
        self.temporal_mode_ = False
        return self

    def fit_dynamic(
        self,
        episodes: list[DynamicPortfolioEpisode],
    ) -> MLPClassifierSelector:
        temporal_features: list[np.ndarray] = []
        runtimes: list[np.ndarray] = []
        best_configs: list[int] = []
        ideal_params: list[np.ndarray] = []
        for episode in episodes:
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            temporal_features.extend(self._temporalize_episode_features(episode_features))
            runtimes.extend([state.runtimes for state in episode.states])
            best_configs.extend([state.best_config for state in episode.states])
            ideal_params.extend([state.ideal_params for state in episode.states])

        self.fit(
            np.stack(temporal_features, axis=0),
            np.stack(runtimes, axis=0),
            np.asarray(best_configs, dtype=np.int64),
            ideal_params=np.stack(ideal_params, axis=0),
        )
        self.temporal_mode_ = True
        return self

    def fit_algorithmic_with_portfolio(
        self,
        episodes: list[AlgorithmicEpisode],
        *,
        portfolio_values: np.ndarray,
        runtime_sequences: list[np.ndarray],
    ) -> MLPClassifierSelector:
        temporal_features: list[np.ndarray] = []
        runtime_rows: list[np.ndarray] = []
        for episode, runtime_sequence in zip(episodes, runtime_sequences, strict=True):
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            temporal_features.extend(self._temporalize_episode_features(episode_features))
            runtime_rows.extend(runtime_sequence)

        runtime_array = np.stack(runtime_rows, axis=0)
        self.portfolio_values_ = np.asarray(portfolio_values, dtype=np.float64)
        self.fit(
            np.stack(temporal_features, axis=0),
            runtime_array,
            runtime_array.argmin(axis=1).astype(np.int64),
            ideal_params=None,
        )
        self.portfolio_values_ = np.asarray(portfolio_values, dtype=np.float64)
        self.temporal_mode_ = True
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
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def _make_train_validation_split(
        self,
        features: torch.Tensor,
        runtimes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_instances = features.shape[0]
        if n_instances < 5:
            return features, runtimes, features, runtimes

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
            features[val_indices],
            runtimes[val_indices],
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

    def predict_runtimes(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        normalized = self._normalize_tensor(feature_tensor)
        self.network_.eval()
        with torch.no_grad():
            predictions = self.network_(normalized)
        return predictions.cpu().numpy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        predictions = self.predict_runtimes(features)
        return predictions.argmin(axis=1).astype(np.int64)

    def predict_episode(
        self,
        episode_features: np.ndarray,
        *,
        switching_cost: float = 0.0,
    ) -> np.ndarray:
        contextual_features = (
            self._temporalize_episode_features(episode_features)
            if self.temporal_mode_
            else np.asarray(episode_features, dtype=np.float64)
        )
        expected_runtimes = self.predict_runtimes(contextual_features)
        actions: list[int] = []
        previous_action: int | None = None
        for timestep in range(expected_runtimes.shape[0]):
            penalized = expected_runtimes[timestep].copy()
            if previous_action is not None:
                switch_penalty = self.switching_penalty_weight * switching_cost
                penalized += switch_penalty
                penalized[previous_action] -= switch_penalty
            action = int(np.argmin(penalized))
            actions.append(action)
            previous_action = action
        return np.asarray(actions, dtype=np.int64)

    def _temporalize_episode_features(self, episode_features: np.ndarray) -> np.ndarray:
        history = np.zeros(episode_features.shape[1], dtype=np.float64)
        previous = np.zeros(episode_features.shape[1], dtype=np.float64)
        contextual_rows: list[np.ndarray] = []
        for timestep, current in enumerate(episode_features):
            current = np.asarray(current, dtype=np.float64)
            if timestep == 0:
                delta = np.zeros_like(current)
            else:
                delta = current - previous
            history = self.history_blend * history + (1.0 - self.history_blend) * current
            trend = self.trend_blend * delta + (1.0 - self.trend_blend) * (current - history)
            contextual_rows.append(np.concatenate([current, history, delta, trend], axis=0))
            previous = current
        return np.stack(contextual_rows, axis=0)
