"""DGCAC-inspired selector with a lightweight learned embedding stage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isac.selectors.clustering import KMeansClusterSelector


@dataclass(slots=True)
class DeepClusterEmbeddingSelector:
    """Approximate DGCAC with a learned low-dimensional embedding plus clustering.

    This is intentionally lightweight for the current repo stage:

    - learn a compact feature embedding with a linear autoencoder-style projection
    - cluster the embedded instances
    - assign each cluster the best empirical portfolio member

    It captures the paper's architectural idea without yet depending on a deep
    learning stack or graph-structured instance encoders.
    """

    n_configs: int
    embedding_dim: int = 2
    n_clusters: int | None = None
    seed: int | None = None
    name: str = "DGCAC-inspired"
    projection_: np.ndarray = field(init=False)
    cluster_selector_: KMeansClusterSelector = field(init=False)
    feature_mean_: np.ndarray = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
    ) -> DeepClusterEmbeddingSelector:
        centered = features - features.mean(axis=0, keepdims=True)
        self.feature_mean_ = features.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        effective_dim = min(self.embedding_dim, vt.shape[0])
        self.projection_ = vt[:effective_dim].T

        embedded = self.transform(features)
        self.cluster_selector_ = KMeansClusterSelector(
            n_configs=self.n_configs,
            n_clusters=self.n_clusters,
            seed=self.seed,
            name=self.name,
        ).fit(embedded, runtimes, best_configs)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        centered = features - self.feature_mean_[None, :]
        return centered @ self.projection_

    def predict(self, features: np.ndarray) -> np.ndarray:
        embedded = self.transform(features)
        return self.cluster_selector_.predict(embedded)
