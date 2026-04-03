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


def test_dynamic_env_returns_registered_environment() -> None:
    env = make("isac-dynamic-v0", seed=5, horizon=6)
    observation, info = env.reset()

    assert observation.shape == (env.n_features,)
    assert info["dynamic"] is True
    assert info["horizon"] == 6
    assert info["partial_observability"] is True
    assert info["multimodal_surface_scale"] == env.benchmark.multimodal_surface_scale


def test_dynamic_env_features_change_over_time() -> None:
    env = make("isac-dynamic-v0", seed=6, horizon=4)
    first_observation, _ = env.reset()
    second_observation, _, terminated, _, _ = env.step(0)

    assert terminated is False
    assert not np.allclose(first_observation, second_observation)


def test_dynamic_env_applies_switching_cost() -> None:
    env = make("isac-dynamic-v0", seed=7, horizon=4)
    env.reset()
    _, _, _, _, first_info = env.step(0)
    _, _, _, _, second_info = env.step(1)

    assert first_info["switch_cost"] == 0.0
    assert second_info["switch_cost"] == env.benchmark.switching_cost


def test_dynamic_env_observation_is_partially_observed() -> None:
    env = make(
        "isac-dynamic-v0",
        seed=8,
        horizon=4,
        observation_noise=0.25,
        missing_feature_prob=0.5,
    )
    observation, _ = env.reset()
    current = env.current_state

    assert current is not None
    assert not np.allclose(observation, current.latent_features)
    assert np.any(current.observation_mask < 1.0)
