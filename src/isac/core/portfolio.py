"""Synthetic portfolio benchmark for fixed parameter selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isac.core.normalization import ZScoreNormalizer


@dataclass(slots=True)
class ParameterConfig:
    """One fixed parameter setting in the selectable portfolio."""

    name: str
    values: np.ndarray


@dataclass(slots=True)
class PortfolioInstance:
    """One problem instance with observable features and latent ideal parameters."""

    features: np.ndarray
    ideal_params: np.ndarray
    runtimes: np.ndarray
    best_config: int
    regime_id: int


@dataclass(slots=True)
class PortfolioBenchmark:
    """Synthetic benchmark for selecting among precomputed parameter settings.

    The setup is intentionally simple:

    - each instance belongs to a latent regime
    - the regime shapes both observable features and the ideal parameter vector
    - the agent only sees features
    - each selectable portfolio item is a fixed parameter vector
    - runtime worsens as the chosen parameter vector moves away from the ideal one
    """

    n_features: int = 5
    feature_noise: float = 0.4
    parameter_noise: float = 0.08
    runtime_noise: float = 0.03
    seed: int | None = None
    n_parameter_dims: int = 2
    rng: np.random.Generator = field(init=False)
    regime_centers: np.ndarray = field(init=False)
    regime_ideal_params: np.ndarray = field(init=False)
    portfolio: tuple[ParameterConfig, ...] = field(init=False)
    difficulty_weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.regime_centers = np.array(
            [
                [-2.0, -1.2, 0.4, 1.8, 0.5],
                [1.6, 1.1, -0.8, -1.4, 0.2],
                [0.2, -2.1, 1.7, 0.4, -1.8],
            ],
            dtype=np.float64,
        )[:, : self.n_features]
        self.regime_ideal_params = np.array(
            [
                [0.15, 0.85],
                [0.85, 0.25],
                [0.55, 0.6],
            ],
            dtype=np.float64,
        )[:, : self.n_parameter_dims]
        self.portfolio = (
            ParameterConfig("conservative", np.array([0.1, 0.9], dtype=np.float64)),
            ParameterConfig("aggressive", np.array([0.9, 0.2], dtype=np.float64)),
            ParameterConfig("balanced", np.array([0.55, 0.55], dtype=np.float64)),
            ParameterConfig("robust", np.array([0.35, 0.7], dtype=np.float64)),
        )
        self.difficulty_weights = self.rng.uniform(0.4, 1.1, size=self.n_features)

    @property
    def n_configs(self) -> int:
        return len(self.portfolio)

    def sample_instance(self) -> PortfolioInstance:
        regime_id = int(self.rng.integers(0, len(self.regime_centers)))
        center = self.regime_centers[regime_id]

        features = center + self.rng.normal(0.0, self.feature_noise, size=self.n_features)
        feature_signal = np.tanh(features[: self.n_parameter_dims]) * 0.08
        ideal_params = np.clip(
            self.regime_ideal_params[regime_id]
            + feature_signal
            + self.rng.normal(0.0, self.parameter_noise, size=self.n_parameter_dims),
            0.0,
            1.0,
        )

        base_difficulty = 1.0 + float(np.abs(features * self.difficulty_weights).mean())
        runtimes = []
        for config in self.portfolio:
            mismatch = float(((config.values - ideal_params) ** 2).sum())
            runtime = base_difficulty + 4.0 * mismatch
            runtime += float(self.rng.normal(0.0, self.runtime_noise))
            runtimes.append(max(runtime, 0.01))

        runtimes_array = np.array(runtimes, dtype=np.float64)
        best_config = int(np.argmin(runtimes_array))
        return PortfolioInstance(
            features=features.astype(np.float64),
            ideal_params=ideal_params.astype(np.float64),
            runtimes=runtimes_array,
            best_config=best_config,
            regime_id=regime_id,
        )

    def sample_batch(self, n_instances: int, normalize: bool = True) -> list[PortfolioInstance]:
        instances = [self.sample_instance() for _ in range(n_instances)]
        if normalize:
            feature_matrix = np.stack([instance.features for instance in instances], axis=0)
            normalized = ZScoreNormalizer.fit(feature_matrix).transform(feature_matrix)
            for instance, normalized_features in zip(instances, normalized, strict=True):
                instance.features = normalized_features
        return instances
