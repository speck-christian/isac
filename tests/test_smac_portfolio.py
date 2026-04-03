from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from isac.core import AlgorithmicPortfolioBenchmark
from isac.selectors.smac_portfolio import SMAC3PortfolioBuilder


def test_smac_builder_requires_optional_dependency_when_unavailable() -> None:
    if importlib.util.find_spec("smac") is not None:
        pytest.skip("SMAC is installed; dependency-guard behavior is not applicable.")

    benchmark = AlgorithmicPortfolioBenchmark(seed=21, horizon=8)
    builder = SMAC3PortfolioBuilder(benchmark=benchmark, n_trials=4, max_portfolio_size=3, seed=21)

    with pytest.raises(ImportError):
        builder._configspace()


def test_smac_builder_config_conversion() -> None:
    config = {"alpha": 0.25, "beta": 0.75, "gate": 2.5}
    values = SMAC3PortfolioBuilder._config_to_array(config)

    assert values.shape == (3,)
    assert np.allclose(values, np.array([0.25, 0.75, 2.5]))
