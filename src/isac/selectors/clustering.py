"""Clustering-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class KMeansClusterSelector:
    """Cluster instances and assign the empirically best config to each cluster."""

    n_configs: int
    n_clusters: int | None = None
    max_iter: int = 25
    seed: int | None = None
    name: str = "Cluster ISAC"
    centers_: np.ndarray = field(init=False)
    cluster_best_configs_: np.ndarray = field(init=False)
    fallback_config_: int = field(init=False, default=0)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
    ) -> KMeansClusterSelector:
        cluster_count = self.n_clusters or self.n_configs
        rng = np.random.default_rng(self.seed)
        initial_indices = rng.choice(len(features), size=cluster_count, replace=False)
        centers = features[initial_indices].copy()

        assignments = np.zeros(len(features), dtype=np.int64)
        for _ in range(self.max_iter):
            distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_assignments = distances.argmin(axis=1).astype(np.int64)
            if np.array_equal(assignments, new_assignments):
                break
            assignments = new_assignments
            for cluster_index in range(cluster_count):
                mask = assignments == cluster_index
                if mask.any():
                    centers[cluster_index] = features[mask].mean(axis=0)

        self.centers_ = centers
        self.fallback_config_ = int(np.bincount(best_configs, minlength=self.n_configs).argmax())

        cluster_best_configs = []
        for cluster_index in range(cluster_count):
            mask = assignments == cluster_index
            if mask.any():
                mean_runtimes = runtimes[mask].mean(axis=0)
                cluster_best_configs.append(int(np.argmin(mean_runtimes)))
            else:
                cluster_best_configs.append(self.fallback_config_)
        self.cluster_best_configs_ = np.array(cluster_best_configs, dtype=np.int64)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        distances = ((features[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        nearest_clusters = distances.argmin(axis=1).astype(np.int64)
        return self.cluster_best_configs_[nearest_clusters]
