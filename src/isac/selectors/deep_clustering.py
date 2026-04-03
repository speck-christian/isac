"""DGCAC-inspired selector with a learned nonlinear embedding stage."""

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
class DeepClusterEmbeddingSelector:
    """Approximate DGCAC with denoising, cost-aware embedding, and soft routing."""

    n_configs: int
    max_portfolio_size: int = 12
    embedding_dim: int = 4
    hidden_dim: int = 16
    n_clusters: int | None = None
    encoder_epochs: int = 300
    learning_rate: float = 0.02
    denoising_noise: float = 0.08
    assignment_temperature: float = 1.5
    history_blend: float = 0.72
    trend_blend: float = 0.45
    switching_penalty_weight: float = 0.75
    reconstruction_weight: float = 0.6
    runtime_weight: float = 0.3
    classification_weight: float = 0.1
    seed: int | None = None
    name: str = "DGCAC-inspired"
    feature_mean_: torch.Tensor = field(init=False)
    feature_scale_: torch.Tensor = field(init=False)
    encoder_: torch.nn.Module = field(init=False)
    decoder_: torch.nn.Module = field(init=False)
    runtime_head_: torch.nn.Module = field(init=False)
    classifier_head_: torch.nn.Module = field(init=False)
    cluster_centers_: torch.Tensor = field(init=False)
    cluster_runtime_means_: torch.Tensor = field(init=False)
    fallback_runtime_: torch.Tensor = field(init=False)
    portfolio_values_: np.ndarray = field(init=False)
    temporal_mode_: bool = field(init=False, default=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
        ideal_params: np.ndarray | None = None,
    ) -> DeepClusterEmbeddingSelector:
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
        label_tensor = torch.as_tensor(best_configs, dtype=torch.long)
        self.feature_mean_ = feature_tensor.mean(dim=0)
        feature_scale = feature_tensor.std(dim=0, unbiased=False)
        self.feature_scale_ = torch.where(
            feature_scale < 1e-6,
            torch.ones_like(feature_scale),
            feature_scale,
        )
        normalized = self._normalize_tensor(feature_tensor)

        input_dim = normalized.shape[1]
        embedding_dim = min(self.embedding_dim, input_dim)
        hidden_dim = max(self.hidden_dim, embedding_dim + 2)

        self.encoder_ = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder_ = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )
        self.runtime_head_ = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(self.portfolio_values_)),
        )
        self.classifier_head_ = torch.nn.Linear(embedding_dim, len(self.portfolio_values_))

        optimizer = torch.optim.Adam(
            list(self.encoder_.parameters())
            + list(self.decoder_.parameters())
            + list(self.runtime_head_.parameters())
            + list(self.classifier_head_.parameters()),
            lr=self.learning_rate,
        )
        mse_loss = torch.nn.MSELoss()
        cross_entropy = torch.nn.CrossEntropyLoss()

        self.encoder_.train()
        self.decoder_.train()
        self.runtime_head_.train()
        self.classifier_head_.train()
        for _ in range(self.encoder_epochs):
            optimizer.zero_grad()
            corrupted = normalized + torch.randn_like(normalized) * self.denoising_noise
            embedded = self.encoder_(corrupted)
            reconstructed = self.decoder_(embedded)
            runtime_predictions = self.runtime_head_(embedded)
            class_logits = self.classifier_head_(embedded)

            loss = (
                self.reconstruction_weight * mse_loss(reconstructed, normalized)
                + self.runtime_weight * mse_loss(runtime_predictions, runtime_tensor)
                + self.classification_weight * cross_entropy(class_logits, label_tensor)
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            embedded_features = self._transform_tensor(feature_tensor)
        self._fit_soft_clusters(embedded_features, runtime_tensor)
        self.fallback_runtime_ = runtime_tensor.mean(dim=0)
        self.temporal_mode_ = False
        return self

    def fit_dynamic(
        self,
        episodes: list[DynamicPortfolioEpisode],
    ) -> DeepClusterEmbeddingSelector:
        """Fit the selector on dynamic episodes using short-horizon temporal context."""

        temporal_features: list[np.ndarray] = []
        runtimes: list[np.ndarray] = []
        best_configs: list[int] = []
        ideal_params: list[np.ndarray] = []
        for episode in episodes:
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            contextual_features = self._temporalize_episode_features(episode_features)
            temporal_features.extend(contextual_features)
            runtimes.extend([state.runtimes for state in episode.states])
            best_configs.extend([state.best_config for state in episode.states])
            ideal_params.extend([state.ideal_params for state in episode.states])

        self.fit(
            np.stack(temporal_features, axis=0),
            np.stack(runtimes, axis=0),
            np.array(best_configs, dtype=np.int64),
            ideal_params=np.stack(ideal_params, axis=0),
        )
        self.temporal_mode_ = True
        return self

    def _normalize_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean_) / self.feature_scale_

    def _transform_tensor(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_tensor(features)
        self.encoder_.eval()
        return self.encoder_(normalized)

    def _fit_soft_clusters(
        self,
        embedded_features: torch.Tensor,
        runtimes: torch.Tensor,
    ) -> None:
        cluster_count = self.n_clusters or len(self.portfolio_values_)
        initial_indices = torch.randperm(embedded_features.shape[0])[:cluster_count]
        centers = embedded_features[initial_indices].clone()

        for _ in range(30):
            weights = self._soft_assignments(embedded_features, centers)
            denominator = weights.sum(dim=0, keepdim=True).T.clamp_min(1e-8)
            new_centers = (weights.T @ embedded_features) / denominator
            if torch.allclose(new_centers, centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        weights = self._soft_assignments(embedded_features, centers)
        runtime_means = []
        for cluster_index in range(cluster_count):
            cluster_weights = weights[:, cluster_index]
            total_weight = cluster_weights.sum().clamp_min(1e-8)
            weighted_runtime = (cluster_weights[:, None] * runtimes).sum(dim=0) / total_weight
            runtime_means.append(weighted_runtime)

        self.cluster_centers_ = centers
        self.cluster_runtime_means_ = torch.stack(runtime_means, dim=0)

    def _soft_assignments(
        self,
        embedded_features: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        squared_distances = torch.cdist(embedded_features, centers, p=2) ** 2
        scaled_logits = -self.assignment_temperature * squared_distances
        return torch.softmax(scaled_logits, dim=1)

    def transform(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            embedded = self._transform_tensor(feature_tensor)
        return embedded.cpu().numpy()

    def _predict_expected_runtimes(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            embedded = self._transform_tensor(feature_tensor)
            assignments = self._soft_assignments(embedded, self.cluster_centers_)
            cluster_expected_runtimes = assignments @ self.cluster_runtime_means_
            head_expected_runtimes = self.runtime_head_(embedded)
            expected_runtimes = (
                0.55 * cluster_expected_runtimes
                + 0.35 * head_expected_runtimes
                + 0.10 * self.fallback_runtime_[None, :]
            )
        return expected_runtimes.cpu().numpy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        expected_runtimes = self._predict_expected_runtimes(features)
        return expected_runtimes.argmin(axis=1).astype(np.int64)

    def predict_episode(
        self,
        episode_features: np.ndarray,
        *,
        switching_cost: float = 0.0,
    ) -> np.ndarray:
        """Predict a sequence of actions using temporal context and switch awareness."""

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
        return np.array(actions, dtype=np.int64)

    def _temporalize_episode_features(self, episode_features: np.ndarray) -> np.ndarray:
        """Build history-aware features from one normalized episode trace."""

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
