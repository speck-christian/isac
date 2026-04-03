"""Classification-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(slots=True)
class NearestCentroidClassifierSelector:
    """Predict the best config by nearest class centroid in feature space."""

    n_configs: int
    name: str = "Classifier"
    centroids_: torch.Tensor = field(init=False)
    fallback_config_: int = field(init=False, default=0)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
    ) -> NearestCentroidClassifierSelector:
        del runtimes
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        labels = torch.as_tensor(best_configs, dtype=torch.long)
        self.fallback_config_ = int(np.bincount(best_configs, minlength=self.n_configs).argmax())

        centroids: list[torch.Tensor] = []
        for config_index in range(self.n_configs):
            mask = labels == config_index
            if bool(mask.any()):
                centroids.append(feature_tensor[mask].mean(dim=0))
            else:
                centroids.append(feature_tensor.mean(dim=0))
        self.centroids_ = torch.stack(centroids, dim=0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        distances = torch.cdist(feature_tensor, self.centroids_, p=2) ** 2
        return distances.argmin(dim=1).cpu().numpy().astype(np.int64)
