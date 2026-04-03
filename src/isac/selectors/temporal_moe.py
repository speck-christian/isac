"""Temporal mixture-of-experts selector for dynamic ISAC scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.core import DynamicPortfolioEpisode
from isac.selectors.portfolio_learning import (
    assign_to_portfolio,
    derive_kmeans_portfolio,
    portfolio_regret_targets,
)


@dataclass(slots=True)
class TemporalMixtureOfExpertsSelector:
    """History-aware soft-routing selector with local runtime experts."""

    n_configs: int
    max_portfolio_size: int = 12
    n_experts: int = 4
    hidden_dim: int = 24
    epochs: int = 260
    learning_rate: float = 0.01
    history_blend: float = 0.72
    trend_blend: float = 0.45
    switching_penalty_weight: float = 0.75
    seed: int | None = None
    name: str = "Temporal MoE"
    feature_mean_: torch.Tensor = field(init=False)
    feature_scale_: torch.Tensor = field(init=False)
    encoder_: torch.nn.Module = field(init=False)
    gating_head_: torch.nn.Module = field(init=False)
    expert_heads_: torch.nn.ModuleList = field(init=False)
    portfolio_values_: np.ndarray = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
        ideal_params: np.ndarray | None = None,
    ) -> TemporalMixtureOfExpertsSelector:
        """Fallback fit for non-sequential data."""

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

        self._fit_model(
            features=np.asarray(features, dtype=np.float64),
            runtimes=np.asarray(runtimes, dtype=np.float64),
            labels=np.asarray(best_configs, dtype=np.int64),
        )
        return self

    def fit_dynamic(
        self,
        episodes: list[DynamicPortfolioEpisode],
    ) -> TemporalMixtureOfExpertsSelector:
        """Fit the selector on full episodes with temporal context."""

        temporal_features: list[np.ndarray] = []
        runtimes: list[np.ndarray] = []
        ideal_params: list[np.ndarray] = []
        for episode in episodes:
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            contextual_features = self._temporalize_episode_features(episode_features)
            temporal_features.extend(contextual_features)
            runtimes.extend([state.runtimes for state in episode.states])
            ideal_params.extend([state.ideal_params for state in episode.states])

        ideal_param_array = np.stack(ideal_params, axis=0)
        self.portfolio_values_ = derive_kmeans_portfolio(
            ideal_param_array,
            max_portfolio_size=self.max_portfolio_size,
            seed=self.seed,
        )
        surrogate_runtimes = portfolio_regret_targets(ideal_param_array, self.portfolio_values_)
        labels = assign_to_portfolio(ideal_param_array, self.portfolio_values_)
        self._fit_model(
            features=np.stack(temporal_features, axis=0),
            runtimes=surrogate_runtimes,
            labels=labels,
        )
        return self

    def _fit_model(
        self,
        *,
        features: np.ndarray,
        runtimes: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        runtime_tensor = torch.as_tensor(runtimes, dtype=torch.float32)
        label_tensor = torch.as_tensor(labels, dtype=torch.long)

        self.feature_mean_ = feature_tensor.mean(dim=0)
        feature_scale = feature_tensor.std(dim=0, unbiased=False)
        self.feature_scale_ = torch.where(
            feature_scale < 1e-6,
            torch.ones_like(feature_scale),
            feature_scale,
        )
        normalized = self._normalize_tensor(feature_tensor)

        input_dim = normalized.shape[1]
        portfolio_size = len(self.portfolio_values_)
        expert_count = min(self.n_experts, portfolio_size)

        self.encoder_ = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.gating_head_ = torch.nn.Linear(self.hidden_dim, expert_count)
        self.expert_heads_ = torch.nn.ModuleList(
            [torch.nn.Linear(self.hidden_dim, portfolio_size) for _ in range(expert_count)]
        )

        optimizer = torch.optim.Adam(
            list(self.encoder_.parameters())
            + list(self.gating_head_.parameters())
            + list(self.expert_heads_.parameters()),
            lr=self.learning_rate,
        )
        mse_loss = torch.nn.MSELoss()
        cross_entropy = torch.nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            encoded = self.encoder_(normalized)
            gating_logits = self.gating_head_(encoded)
            gating_weights = torch.softmax(gating_logits, dim=1)
            expert_predictions = torch.stack(
                [expert_head(encoded) for expert_head in self.expert_heads_],
                dim=1,
            )
            mixed_predictions = (gating_weights[:, :, None] * expert_predictions).sum(dim=1)
            loss = (
                0.75 * mse_loss(mixed_predictions, runtime_tensor)
                + 0.25 * cross_entropy(mixed_predictions, label_tensor)
            )
            loss.backward()
            optimizer.step()

    def _normalize_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean_) / self.feature_scale_

    def _predict_expected_runtimes(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        normalized = self._normalize_tensor(feature_tensor)
        with torch.no_grad():
            encoded = self.encoder_(normalized)
            gating_weights = torch.softmax(self.gating_head_(encoded), dim=1)
            expert_predictions = torch.stack(
                [expert_head(encoded) for expert_head in self.expert_heads_],
                dim=1,
            )
            mixed_predictions = (gating_weights[:, :, None] * expert_predictions).sum(dim=1)
        return mixed_predictions.cpu().numpy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._predict_expected_runtimes(features).argmin(axis=1).astype(np.int64)

    def predict_episode(
        self,
        episode_features: np.ndarray,
        *,
        switching_cost: float = 0.0,
    ) -> np.ndarray:
        contextual_features = self._temporalize_episode_features(episode_features)
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
        return np.array(actions, dtype=np.int64)

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
