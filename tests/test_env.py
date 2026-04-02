from __future__ import annotations

import numpy as np

from isac import make
from isac.core.synthetic import SyntheticBenchmark


def test_make_returns_registered_environment() -> None:
    env = make("isac-simple-v0", seed=1)
    observation, info = env.reset()

    assert observation.shape == (env.n_features,)
    assert info["n_configs"] == env.n_configs


def test_step_reward_matches_negative_regret() -> None:
    env = make("isac-simple-v0", seed=2, horizon=1)
    env.reset()
    current = env.current_instance
    assert current is not None

    _, reward, terminated, truncated, info = env.step(current.best_config)

    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert info["regret"] == 0.0


def test_global_best_config_is_valid() -> None:
    benchmark = SyntheticBenchmark(seed=3)
    best_config = benchmark.estimate_global_best_config(n_instances=64)

    assert 0 <= best_config < benchmark.n_configs


def test_reset_seed_reproducibility() -> None:
    env = make("isac-simple-v0", seed=4)
    obs_a, _ = env.reset(seed=11)
    obs_b, _ = env.reset(seed=11)

    assert np.allclose(obs_a, obs_b)
