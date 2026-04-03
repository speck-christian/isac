"""Classification-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.selectors.portfolio_learning import assign_to_portfolio, derive_kmeans_portfolio


@dataclass(slots=True)
class NearestCentroidClassifierSelector:
    """Privileged comparator that predicts the oracle best config by centroid."""

    n_configs: int
    max_portfolio_size: int = 12
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

    def predict(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        distances = torch.cdist(feature_tensor, self.centroids_, p=2) ** 2
        return distances.argmin(dim=1).cpu().numpy().astype(np.int64)
