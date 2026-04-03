"""Algorithm-driven ISAC environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isac.core import AlgorithmicEpisode, AlgorithmicPortfolioBenchmark, AlgorithmicState


@dataclass(slots=True)
class ISACAlgorithmicEnv:
    """Environment around a concrete adaptive forecasting algorithm."""

    horizon: int = 24
    n_features: int = 5
    observation_noise: float = 0.08
    regime_switch_prob: float = 0.14
    seed: int | None = None
    benchmark: AlgorithmicPortfolioBenchmark = field(init=False)
    episode: AlgorithmicEpisode | None = field(init=False, default=None)
    current_state: AlgorithmicState | None = field(init=False, default=None)
    algorithm_state: Any = field(init=False, default=None)
    step_count: int = field(init=False, default=0)
    previous_action: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.benchmark = AlgorithmicPortfolioBenchmark(
            horizon=self.horizon,
            n_features=self.n_features,
            observation_noise=self.observation_noise,
            regime_switch_prob=self.regime_switch_prob,
            seed=self.seed,
        )

    @property
    def n_configs(self) -> int:
        return self.benchmark.n_configs

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.seed = seed
            self.__post_init__()

        self.step_count = 0
        self.previous_action = None
        self.episode = self.benchmark.sample_episode()
        self.current_state = self.episode.states[0]
        self.algorithm_state = self.benchmark._initialize_algorithm_state(self.episode)
        info = {
            "n_configs": self.n_configs,
            "n_features": self.n_features,
            "horizon": self.horizon,
            "algorithm": "adaptive_forecaster",
            "parameter_names": ["alpha", "beta", "gate"],
            "regime_switch_prob": self.benchmark.regime_switch_prob,
        }
        return self.current_state.features.copy(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_state is None or self.episode is None or self.algorithm_state is None:
            raise RuntimeError("Call reset() before step().")
        if not 0 <= action < self.n_configs:
            raise ValueError(f"Action must be in [0, {self.n_configs - 1}], got {action}.")

        state = self.current_state
        parameter_values = self.benchmark.portfolio[action].values
        loss, self.algorithm_state = self.benchmark.algorithm_step(
            self.algorithm_state,
            state,
            parameter_values,
        )
        best_loss = float(state.runtimes[state.best_config])
        reward = -(loss - best_loss)
        info = {
            "loss": float(loss),
            "best_loss": best_loss,
            "best_config": state.best_config,
            "selected_params": parameter_values.copy(),
            "regret": float(loss - best_loss),
            "regime_id": state.regime_id,
            "target": state.target,
            "observed_value": state.observed_value,
            "timestep": state.timestep,
        }

        self.previous_action = action
        self.step_count += 1
        terminated = self.step_count >= self.horizon
        truncated = False
        if terminated:
            next_observation = np.zeros(self.n_features, dtype=np.float64)
        else:
            self.current_state = self.episode.states[self.step_count]
            next_observation = self.current_state.features.copy()
        return next_observation, float(reward), terminated, truncated, info
