"""Classification-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class NearestCentroidClassifierSelector:
    """Predict the best config by nearest class centroid in feature space."""

    n_configs: int
    name: str = "Classifier"
    centroids_: np.ndarray = field(init=False)
    fallback_config_: int = field(init=False, default=0)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
    ) -> NearestCentroidClassifierSelector:
        del runtimes
        self.fallback_config_ = int(np.bincount(best_configs, minlength=self.n_configs).argmax())
        centroids = []
        for config_index in range(self.n_configs):
            mask = best_configs == config_index
            if mask.any():
                centroids.append(features[mask].mean(axis=0))
            else:
                centroids.append(features.mean(axis=0))
        self.centroids_ = np.stack(centroids, axis=0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        distances = ((features[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1).astype(np.int64)
