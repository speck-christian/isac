"""First ISAC-inspired simulation environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isac.core.synthetic import SyntheticBenchmark, SyntheticInstance


@dataclass(slots=True)
class ISACSimpleEnv:
    """Synthetic environment for choosing a configuration from instance features.

    Each timestep presents one synthetic problem instance. The agent observes a
    feature vector and chooses one of the available configurations. Reward is the
    negative regret relative to the best configuration for that instance.
    """

    n_features: int = 6
    n_configs: int = 4
    horizon: int = 32
    seed: int | None = None
    benchmark: SyntheticBenchmark = field(init=False)
    step_count: int = field(init=False, default=0)
    current_instance: SyntheticInstance | None = field(init=False, default=None)
    global_best_config: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.benchmark = SyntheticBenchmark(
            n_features=self.n_features,
            n_configs=self.n_configs,
            seed=self.seed,
        )
        self.global_best_config = self.benchmark.estimate_global_best_config()

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.seed = seed
            self.__post_init__()

        self.step_count = 0
        self.current_instance = self.benchmark.sample_batch(n_instances=1)[0]
        info = {
            "global_best_config": self.global_best_config,
            "n_configs": self.n_configs,
            "n_features": self.n_features,
        }
        return self.current_instance.features.copy(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_instance is None:
            raise RuntimeError("Call reset() before step().")
        if not 0 <= action < self.n_configs:
            raise ValueError(f"Action must be in [0, {self.n_configs - 1}], got {action}.")

        instance = self.current_instance
        selected_runtime = float(instance.runtimes[action])
        best_runtime = float(instance.runtimes[instance.best_config])
        global_runtime = float(instance.runtimes[self.global_best_config])
        reward = -(selected_runtime - best_runtime)

        self.step_count += 1
        terminated = self.step_count >= self.horizon
        truncated = False

        next_instance = self.benchmark.sample_batch(n_instances=1)[0]
        self.current_instance = next_instance
        info = {
            "selected_runtime": selected_runtime,
            "best_runtime": best_runtime,
            "global_runtime": global_runtime,
            "best_config": instance.best_config,
            "cluster_id": instance.cluster_id,
            "regret": selected_runtime - best_runtime,
            "fallback_regret": global_runtime - best_runtime,
        }
        return next_instance.features.copy(), reward, terminated, truncated, info
