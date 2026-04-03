"""Dynamic ISAC-inspired environment with evolving instance state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isac.core import DynamicPortfolioBenchmark, DynamicPortfolioEpisode, DynamicPortfolioState


@dataclass(slots=True)
class ISACDynamicEnv:
    """Environment where a single instance evolves and actions can be revised."""

    horizon: int = 16
    n_features: int = 5
    observation_noise: float = 0.10
    missing_feature_prob: float = 0.22
    multimodal_surface_scale: float = 0.30
    seed: int | None = None
    benchmark: DynamicPortfolioBenchmark = field(init=False)
    episode: DynamicPortfolioEpisode | None = field(init=False, default=None)
    step_count: int = field(init=False, default=0)
    current_state: DynamicPortfolioState | None = field(init=False, default=None)
    previous_action: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.benchmark = DynamicPortfolioBenchmark(
            horizon=self.horizon,
            n_features=self.n_features,
            observation_noise=self.observation_noise,
            missing_feature_prob=self.missing_feature_prob,
            multimodal_surface_scale=self.multimodal_surface_scale,
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
        info = {
            "n_configs": self.n_configs,
            "n_features": self.n_features,
            "horizon": self.horizon,
            "switching_cost": self.benchmark.switching_cost,
            "observation_noise": self.benchmark.observation_noise,
            "missing_feature_prob": self.benchmark.missing_feature_prob,
            "multimodal_surface_scale": self.benchmark.multimodal_surface_scale,
            "partial_observability": True,
            "dynamic": True,
        }
        return self.current_state.features.copy(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_state is None or self.episode is None:
            raise RuntimeError("Call reset() before step().")
        if not 0 <= action < self.n_configs:
            raise ValueError(f"Action must be in [0, {self.n_configs - 1}], got {action}.")

        state = self.current_state
        selected_runtime = float(state.runtimes[action])
        best_runtime = float(state.runtimes[state.best_config])
        switch_cost = 0.0
        if self.previous_action is not None and action != self.previous_action:
            switch_cost = self.benchmark.switching_cost
        reward = -(selected_runtime - best_runtime + switch_cost)

        info = {
            "selected_runtime": selected_runtime,
            "best_runtime": best_runtime,
            "best_config": state.best_config,
            "regime_id": state.regime_id,
            "timestep": state.timestep,
            "regret": selected_runtime - best_runtime,
            "switch_cost": switch_cost,
            "total_penalty": selected_runtime - best_runtime + switch_cost,
            "observation_mask": state.observation_mask.copy(),
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
        return next_observation, reward, terminated, truncated, info
