"""Clustering-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.core import AlgorithmicEpisode
from isac.selectors.portfolio_learning import assign_to_portfolio


@dataclass(slots=True)
class KMeansClusterSelector:
    """Cluster instances and assign the empirically best config to each cluster."""

    n_configs: int
    max_portfolio_size: int = 12
    n_clusters: int | None = None
    max_iter: int = 25
    history_blend: float = 0.72
    trend_blend: float = 0.45
    seed: int | None = None
    name: str = "Cluster ISAC"
    centers_: torch.Tensor = field(init=False)
    cluster_best_configs_: torch.Tensor = field(init=False)
    fallback_config_: int = field(init=False, default=0)
    portfolio_values_: np.ndarray = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
        ideal_params: np.ndarray | None = None,
    ) -> KMeansClusterSelector:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        runtime_tensor = torch.as_tensor(runtimes, dtype=torch.float32)
        cluster_count = self.n_clusters or (
            min(self.max_portfolio_size, feature_tensor.shape[0])
            if ideal_params is not None
            else self.n_configs
        )

        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        initial_indices = torch.randperm(
            feature_tensor.shape[0],
            generator=generator,
        )[:cluster_count]
        centers = feature_tensor[initial_indices].clone()

        assignments = torch.zeros(feature_tensor.shape[0], dtype=torch.long)
        for _ in range(self.max_iter):
            distances = torch.cdist(feature_tensor, centers, p=2) ** 2
            new_assignments = distances.argmin(dim=1)
            if torch.equal(assignments, new_assignments):
                break
            assignments = new_assignments
            for cluster_index in range(cluster_count):
                mask = assignments == cluster_index
                if bool(mask.any()):
                    centers[cluster_index] = feature_tensor[mask].mean(dim=0)

        self.centers_ = centers
        if ideal_params is not None:
            portfolio_values = []
            for cluster_index in range(cluster_count):
                mask = assignments == cluster_index
                if bool(mask.any()):
                    portfolio_values.append(np.asarray(ideal_params[mask.cpu().numpy()].mean(axis=0)))
                else:
                    portfolio_values.append(np.asarray(ideal_params.mean(axis=0)))
            self.portfolio_values_ = np.stack(portfolio_values, axis=0).astype(np.float64)
            cluster_targets = assign_to_portfolio(ideal_params, self.portfolio_values_)
            self.fallback_config_ = int(
                np.bincount(cluster_targets, minlength=len(self.portfolio_values_)).argmax()
            )
            self.cluster_best_configs_ = torch.arange(cluster_count, dtype=torch.long)
            return self

        self.portfolio_values_ = np.eye(self.n_configs, dtype=np.float64)
        self.fallback_config_ = int(np.bincount(best_configs, minlength=self.n_configs).argmax())

        cluster_best_configs: list[int] = []
        for cluster_index in range(cluster_count):
            mask = assignments == cluster_index
            if bool(mask.any()):
                mean_runtimes = runtime_tensor[mask].mean(dim=0)
                cluster_best_configs.append(int(mean_runtimes.argmin().item()))
            else:
                cluster_best_configs.append(self.fallback_config_)
        self.cluster_best_configs_ = torch.tensor(cluster_best_configs, dtype=torch.long)
        return self

    def fit_algorithmic_with_portfolio(
        self,
        episodes: list[AlgorithmicEpisode],
        *,
        portfolio_values: np.ndarray,
        runtime_sequences: list[np.ndarray],
    ) -> KMeansClusterSelector:
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
        distances = torch.cdist(feature_tensor, self.centers_, p=2) ** 2
        nearest_clusters = distances.argmin(dim=1)
        return self.cluster_best_configs_[nearest_clusters].cpu().numpy().astype(np.int64)

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
