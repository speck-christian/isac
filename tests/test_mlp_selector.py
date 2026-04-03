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

    selector = MLPClassifierSelector(n_configs=3, seed=0, epochs=250).fit(
        features,
        runtimes,
        best_configs,
    )
    predictions = selector.predict(features)

    assert predictions.shape == (6,)
    assert np.all(predictions >= 0)
    assert np.all(predictions < 3)
