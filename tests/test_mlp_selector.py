from __future__ import annotations

import numpy as np

from isac.selectors import MLPClassifierSelector


def test_mlp_selector_predicts_valid_config_indices() -> None:
    features = np.array(
        [
            [-1.0, -0.8, -0.7],
            [-0.9, -0.6, -0.5],
            [0.2, 0.1, 0.0],
            [0.3, 0.2, 0.1],
            [1.2, 1.1, 0.8],
            [1.3, 1.0, 0.9],
        ]
    )
    ideal_params = np.array(
        [
            [0.10, 0.90],
            [0.14, 0.84],
            [0.52, 0.56],
            [0.58, 0.52],
            [0.90, 0.20],
            [0.86, 0.24],
        ]
    )
    runtimes = np.array(
        [
            [1.0, 2.2, 2.4],
            [1.1, 2.1, 2.3],
            [2.0, 1.0, 1.6],
            [1.9, 1.1, 1.7],
            [2.4, 2.0, 1.0],
            [2.3, 2.1, 1.1],
        ]
    )
    best_configs = np.argmin(runtimes, axis=1)

    selector = MLPClassifierSelector(n_configs=3, max_portfolio_size=4, seed=0, epochs=250).fit(
        features,
        runtimes,
        best_configs,
        ideal_params=ideal_params,
    )
    predictions = selector.predict(features)

    assert predictions.shape == (6,)
    assert np.all(predictions >= 0)
    assert np.all(predictions < selector.portfolio_values_.shape[0])
    assert selector.portfolio_values_.shape[0] <= 4
