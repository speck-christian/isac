"""Synthetic instance generators for early ISAC experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isac.core.normalization import ZScoreNormalizer


@dataclass(slots=True)
class SyntheticInstance:
    """Single problem instance with features and configuration runtimes."""

    features: np.ndarray
    runtimes: np.ndarray
    best_config: int
    cluster_id: int


@dataclass(slots=True)
class SyntheticBenchmark:
    """Generates instances where each configuration performs best near its centroid."""

    n_features: int = 6
    n_configs: int = 4
    feature_noise: float = 0.6
    runtime_noise: float = 0.2
    seed: int | None = None
    rng: np.random.Generator = field(init=False)
    centroids: np.ndarray = field(init=False)
    runtime_offsets: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.centroids = self.rng.normal(loc=0.0, scale=2.0, size=(self.n_configs, self.n_features))
        self.runtime_offsets = self.rng.uniform(0.5, 2.0, size=self.n_configs)

    def sample_instance(self) -> SyntheticInstance:
        cluster_id = int(self.rng.integers(0, self.n_configs))
        features = self.centroids[cluster_id] + self.rng.normal(
            loc=0.0,
            scale=self.feature_noise,
            size=self.n_features,
        )

        distances = ((self.centroids - features) ** 2).sum(axis=1)
        runtimes = self.runtime_offsets + distances + self.rng.normal(
            loc=0.0,
            scale=self.runtime_noise,
            size=self.n_configs,
        )
        runtimes = np.clip(runtimes, a_min=0.01, a_max=None)
        best_config = int(np.argmin(runtimes))
        return SyntheticInstance(
            features=features.astype(np.float64),
            runtimes=runtimes.astype(np.float64),
            best_config=best_config,
            cluster_id=cluster_id,
        )

    def sample_batch(self, n_instances: int, normalize: bool = True) -> list[SyntheticInstance]:
        instances = [self.sample_instance() for _ in range(n_instances)]
        if normalize:
            feature_matrix = np.stack([instance.features for instance in instances], axis=0)
            normalizer = ZScoreNormalizer.fit(feature_matrix)
            normalized = normalizer.transform(feature_matrix)
            for instance, normalized_features in zip(instances, normalized, strict=True):
                instance.features = normalized_features
        return instances

    def estimate_global_best_config(self, n_instances: int = 512) -> int:
        instances = self.sample_batch(n_instances=n_instances, normalize=False)
        mean_runtimes = np.stack([instance.runtimes for instance in instances], axis=0).mean(axis=0)
        return int(np.argmin(mean_runtimes))
