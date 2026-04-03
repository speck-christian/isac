"""Clustering-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(slots=True)
class KMeansClusterSelector:
    """Cluster instances and assign the empirically best config to each cluster."""

    n_configs: int
    n_clusters: int | None = None
    max_iter: int = 25
    seed: int | None = None
    name: str = "Cluster ISAC"
    centers_: torch.Tensor = field(init=False)
    cluster_best_configs_: torch.Tensor = field(init=False)
    fallback_config_: int = field(init=False, default=0)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
    ) -> KMeansClusterSelector:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        runtime_tensor = torch.as_tensor(runtimes, dtype=torch.float32)
        cluster_count = self.n_clusters or self.n_configs

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

    def predict(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        distances = torch.cdist(feature_tensor, self.centers_, p=2) ** 2
        nearest_clusters = distances.argmin(dim=1)
        return self.cluster_best_configs_[nearest_clusters].cpu().numpy().astype(np.int64)
