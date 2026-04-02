from __future__ import annotations

import numpy as np

from isac.core import PortfolioBenchmark


def test_portfolio_benchmark_returns_valid_instance() -> None:
    benchmark = PortfolioBenchmark(seed=5)
    instance = benchmark.sample_instance()

    assert instance.features.shape == (benchmark.n_features,)
    assert instance.ideal_params.shape == (benchmark.n_parameter_dims,)
    assert instance.runtimes.shape == (benchmark.n_configs,)
    assert 0 <= instance.best_config < benchmark.n_configs


def test_portfolio_batch_normalizes_features() -> None:
    benchmark = PortfolioBenchmark(seed=6)
    batch = benchmark.sample_batch(64, normalize=True)

    feature_matrix = np.stack([instance.features for instance in batch], axis=0)
    assert np.allclose(feature_matrix.mean(axis=0), 0.0, atol=1e-7)


def test_portfolio_best_config_matches_lowest_runtime() -> None:
    benchmark = PortfolioBenchmark(seed=7)
    instance = benchmark.sample_instance()

    assert instance.best_config == int(np.argmin(instance.runtimes))
