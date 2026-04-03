"""Regression-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.core import AlgorithmicEpisode, DynamicPortfolioEpisode
from isac.selectors.portfolio_learning import derive_kmeans_portfolio, portfolio_regret_targets


@dataclass(slots=True)
class LinearRuntimeRegressorSelector:
    """Predict portfolio-conditioned losses and choose the smallest estimate."""

    n_configs: int
    max_portfolio_size: int = 12
    l2_penalty: float = 1e-3
    history_blend: float = 0.72
    trend_blend: float = 0.45
    switching_penalty_weight: float = 0.60
    seed: int | None = None
    name: str = "Regressor"
    coefficients_: torch.Tensor = field(init=False)
    feature_mean_: torch.Tensor = field(init=False)
    feature_scale_: torch.Tensor = field(init=False)
    portfolio_values_: np.ndarray = field(init=False)
    temporal_mode_: bool = field(init=False, default=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
        ideal_params: np.ndarray | None = None,
    ) -> LinearRuntimeRegressorSelector:
        del best_configs
        if ideal_params is not None:
            self.portfolio_values_ = derive_kmeans_portfolio(
                ideal_params,
                max_portfolio_size=self.max_portfolio_size,
                seed=self.seed,
            )
            runtimes = portfolio_regret_targets(ideal_params, self.portfolio_values_)
        else:
            self.portfolio_values_ = np.eye(self.n_configs, dtype=np.float64)

        self._fit_runtime_model(
            np.asarray(features, dtype=np.float64),
            np.asarray(runtimes, dtype=np.float64),
        )
        self.temporal_mode_ = False
        return self

    def fit_dynamic(
        self,
        episodes: list[DynamicPortfolioEpisode],
    ) -> LinearRuntimeRegressorSelector:
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
    ) -> LinearRuntimeRegressorSelector:
        temporal_features: list[np.ndarray] = []
        runtime_rows: list[np.ndarray] = []
        for episode, runtime_sequence in zip(episodes, runtime_sequences, strict=True):
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            temporal_features.extend(self._temporalize_episode_features(episode_features))
            runtime_rows.extend(runtime_sequence)

        self.portfolio_values_ = np.asarray(portfolio_values, dtype=np.float64)
        self._fit_runtime_model(
            np.stack(temporal_features, axis=0),
            np.stack(runtime_rows, axis=0),
        )
        self.temporal_mode_ = True
        return self

    def _fit_runtime_model(self, features: np.ndarray, runtimes: np.ndarray) -> None:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        runtime_tensor = torch.as_tensor(runtimes, dtype=torch.float32)
        self.feature_mean_ = feature_tensor.mean(dim=0)
        feature_scale = feature_tensor.std(dim=0, unbiased=False)
        self.feature_scale_ = torch.where(
            feature_scale < 1e-6,
            torch.ones_like(feature_scale),
            feature_scale,
        )
        design = self._design_matrix(feature_tensor)
        gram = design.T @ design
        regularizer = self.l2_penalty * torch.eye(gram.shape[0], dtype=design.dtype)
        regularizer[-1, -1] = 0.0
        rhs = design.T @ runtime_tensor
        self.coefficients_ = torch.linalg.solve(gram + regularizer, rhs)

    def _normalize_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean_) / self.feature_scale_

    def _design_matrix(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_tensor(features)
        squared = normalized * normalized
        bias = torch.ones((normalized.shape[0], 1), dtype=normalized.dtype)
        return torch.cat([normalized, squared, bias], dim=1)

    def _predict_expected_runtimes(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        design = self._design_matrix(feature_tensor)
        predictions = design @ self.coefficients_
        return predictions.cpu().numpy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        expected_runtimes = self._predict_expected_runtimes(features)
        return expected_runtimes.argmin(axis=1).astype(np.int64)

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
        expected_runtimes = self._predict_expected_runtimes(contextual_features)
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
