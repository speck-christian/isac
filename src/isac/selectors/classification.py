"""Classification-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.core import AlgorithmicEpisode
from isac.selectors.portfolio_learning import assign_to_portfolio, derive_kmeans_portfolio


@dataclass(slots=True)
class NearestCentroidClassifierSelector:
    """Privileged comparator that predicts the oracle best config by centroid."""

    n_configs: int
    max_portfolio_size: int = 12
    history_blend: float = 0.72
    trend_blend: float = 0.45
    seed: int | None = None
    name: str = "Privileged Classifier"
    centroids_: torch.Tensor = field(init=False)
    fallback_config_: int = field(init=False, default=0)
    portfolio_values_: np.ndarray = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
        ideal_params: np.ndarray | None = None,
    ) -> NearestCentroidClassifierSelector:
        del runtimes
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        if ideal_params is not None:
            self.portfolio_values_ = derive_kmeans_portfolio(
                ideal_params,
                max_portfolio_size=self.max_portfolio_size,
                seed=self.seed,
            )
            label_values = assign_to_portfolio(ideal_params, self.portfolio_values_)
        else:
            label_values = best_configs
            self.portfolio_values_ = np.eye(self.n_configs, dtype=np.float64)

        labels = torch.as_tensor(label_values, dtype=torch.long)
        self.fallback_config_ = int(
            np.bincount(label_values, minlength=len(self.portfolio_values_)).argmax()
        )

        centroids: list[torch.Tensor] = []
        for config_index in range(len(self.portfolio_values_)):
            mask = labels == config_index
            if bool(mask.any()):
                centroids.append(feature_tensor[mask].mean(dim=0))
            else:
                centroids.append(feature_tensor.mean(dim=0))
        self.centroids_ = torch.stack(centroids, dim=0)
        return self

    def fit_algorithmic_with_portfolio(
        self,
        episodes: list[AlgorithmicEpisode],
        *,
        portfolio_values: np.ndarray,
        runtime_sequences: list[np.ndarray],
    ) -> NearestCentroidClassifierSelector:
        temporal_features: list[np.ndarray] = []
        runtime_rows: list[np.ndarray] = []
        for episode, runtime_sequence in zip(episodes, runtime_sequences, strict=True):
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            temporal_features.extend(self._temporalize_episode_features(episode_features))
            runtime_rows.extend(runtime_sequence)

        runtime_array = np.stack(runtime_rows, axis=0).astype(np.float64)
        best_configs = runtime_array.argmin(axis=1).astype(np.int64)
        self.portfolio_values_ = np.asarray(portfolio_values, dtype=np.float64)
        self.fit(
            np.stack(temporal_features, axis=0),
            runtime_array,
            best_configs,
            ideal_params=None,
        )
        self.portfolio_values_ = np.asarray(portfolio_values, dtype=np.float64)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        distances = torch.cdist(feature_tensor, self.centroids_, p=2) ** 2
        return distances.argmin(dim=1).cpu().numpy().astype(np.int64)

    def predict_episode(
        self,
        episode_features: np.ndarray,
        *,
        switching_cost: float = 0.0,
    ) -> np.ndarray:
        del switching_cost
        temporal_features = self._temporalize_episode_features(np.asarray(episode_features))
        return self.predict(temporal_features)

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
