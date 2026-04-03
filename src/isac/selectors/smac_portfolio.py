"""Optional SMAC3-based portfolio construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from isac.core import AlgorithmicEpisode, AlgorithmicPortfolioBenchmark
from isac.selectors.portfolio_learning import derive_kmeans_portfolio


def _split_builder_episodes(
    episodes: list[AlgorithmicEpisode],
) -> tuple[list[AlgorithmicEpisode], list[AlgorithmicEpisode]]:
    if len(episodes) < 4:
        return episodes, episodes
    split_index = max(1, int(round(0.7 * len(episodes))))
    if split_index >= len(episodes):
        split_index = len(episodes) - 1
    optimize_episodes = episodes[:split_index]
    validation_episodes = episodes[split_index:]
    return optimize_episodes, validation_episodes


def _tune_portfolio_from_incumbents(
    benchmark: AlgorithmicPortfolioBenchmark,
    incumbents: np.ndarray,
    validation_episodes: list[AlgorithmicEpisode],
    *,
    max_portfolio_size: int,
    seed: int | None,
) -> np.ndarray:
    best_portfolio: np.ndarray | None = None
    best_score = float("inf")
    max_size = min(max_portfolio_size, incumbents.shape[0])
    for portfolio_size in range(1, max_size + 1):
        candidate_portfolio = derive_kmeans_portfolio(
            incumbents,
            max_portfolio_size=portfolio_size,
            seed=seed,
        )
        validation_score = float(
            np.mean(
                [
                    benchmark.evaluate_portfolio(episode, candidate_portfolio).min()
                    for episode in validation_episodes
                ]
            )
        )
        if (
            validation_score < best_score - 1e-9
            or (
                abs(validation_score - best_score) <= 1e-9
                and best_portfolio is not None
                and candidate_portfolio.shape[0] < best_portfolio.shape[0]
            )
        ):
            best_score = validation_score
            best_portfolio = candidate_portfolio

    if best_portfolio is None:
        raise ValueError("Portfolio tuning produced no candidate portfolio.")
    return best_portfolio


def _require_smac() -> tuple[Any, Any, Any]:
    """Import SMAC3 and ConfigSpace lazily so they remain optional."""

    try:
        from ConfigSpace import ConfigurationSpace
        from smac import HyperparameterOptimizationFacade, Scenario
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "SMAC3 support requires optional dependencies. Install with "
            '`pip install -e ".[smac]"` or `pip install smac`.'
        ) from exc
    return ConfigurationSpace, HyperparameterOptimizationFacade, Scenario


@dataclass(slots=True)
class SMAC3PortfolioBuilder:
    """Build a compact parameter portfolio from SMAC3 incumbents.

    The intended workflow is:

    1. optimize the concrete algorithm separately on diverse training episodes,
    2. collect the resulting incumbents,
    3. compress those incumbents to a capped portfolio with k-means-style
       prototype extraction.
    """

    benchmark: AlgorithmicPortfolioBenchmark
    n_trials: int = 40
    max_portfolio_size: int = 12
    seed: int | None = None

    def _configspace(self) -> Any:
        ConfigurationSpace, _, _ = _require_smac()
        return ConfigurationSpace(
            {
                "alpha": (0.02, 0.98),
                "beta": (0.02, 0.98),
                "gate": (0.0, 6.0),
            }
        )

    @staticmethod
    def _config_to_array(config: Any) -> np.ndarray:
        return np.array(
            [float(config["alpha"]), float(config["beta"]), float(config["gate"])],
            dtype=np.float64,
        )

    def optimize_episode(self, episode: AlgorithmicEpisode) -> np.ndarray:
        """Optimize one training episode and return the incumbent parameter vector."""

        _, HyperparameterOptimizationFacade, Scenario = _require_smac()
        configspace = self._configspace()
        scenario = Scenario(
            configspace,
            deterministic=True,
            n_trials=self.n_trials,
            seed=0 if self.seed is None else self.seed,
        )

        def target_function(config: Any, seed: int = 0) -> float:
            del seed
            parameter_values = self._config_to_array(config)
            return self.benchmark.evaluate_parameters(episode, parameter_values)

        smac = HyperparameterOptimizationFacade(
            scenario,
            target_function,
            overwrite=True,
        )
        incumbent = smac.optimize()
        return self._config_to_array(incumbent)

    def build_portfolio(self, episodes: list[AlgorithmicEpisode]) -> np.ndarray:
        """Optimize diverse episodes and compress the incumbents into one portfolio."""

        if not episodes:
            raise ValueError("At least one training episode is required.")

        optimize_episodes, validation_episodes = _split_builder_episodes(episodes)
        incumbents = np.stack(
            [self.optimize_episode(episode) for episode in optimize_episodes],
            axis=0,
        )
        return _tune_portfolio_from_incumbents(
            self.benchmark,
            incumbents,
            validation_episodes,
            max_portfolio_size=self.max_portfolio_size,
            seed=self.seed,
        )


@dataclass(slots=True)
class RandomSearchPortfolioBuilder:
    """Lightweight offline portfolio builder using random parameter search.

    This is the default algorithmic portfolio builder when SMAC3 is not being
    used. It preserves the intended workflow of optimizing the concrete
    algorithm on diverse episodes and then compressing those incumbents into a
    capped portfolio.
    """

    benchmark: AlgorithmicPortfolioBenchmark
    n_trials: int = 64
    max_portfolio_size: int = 12
    seed: int | None = None

    def _sample_parameter_vector(self, rng: np.random.Generator) -> np.ndarray:
        return np.array(
            [
                float(rng.uniform(0.02, 0.98)),
                float(rng.uniform(0.02, 0.98)),
                float(rng.uniform(0.0, 6.0)),
            ],
            dtype=np.float64,
        )

    def optimize_episode(self, episode: AlgorithmicEpisode) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        best_params = self._sample_parameter_vector(rng)
        best_score = self.benchmark.evaluate_parameters(episode, best_params)
        for _ in range(self.n_trials - 1):
            candidate = self._sample_parameter_vector(rng)
            score = self.benchmark.evaluate_parameters(episode, candidate)
            if score < best_score:
                best_score = score
                best_params = candidate
        return best_params

    def build_portfolio(self, episodes: list[AlgorithmicEpisode]) -> np.ndarray:
        if not episodes:
            raise ValueError("At least one training episode is required.")
        optimize_episodes, validation_episodes = _split_builder_episodes(episodes)
        incumbents = np.stack(
            [self.optimize_episode(episode) for episode in optimize_episodes],
            axis=0,
        )
        return _tune_portfolio_from_incumbents(
            self.benchmark,
            incumbents,
            validation_episodes,
            max_portfolio_size=self.max_portfolio_size,
            seed=self.seed,
        )
