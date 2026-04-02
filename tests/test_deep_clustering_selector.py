from __future__ import annotations

import numpy as np

from isac.selectors import DeepClusterEmbeddingSelector


def test_dgcac_inspired_selector_predicts_shape() -> None:
    features = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.2, 0.0, 0.1],
            [2.8, 3.0, 3.1],
            [3.2, 3.1, 2.9],
        ]
    )
    runtimes = np.array(
        [
            [1.0, 2.0],
            [1.1, 1.9],
            [2.2, 1.0],
            [2.0, 1.1],
        ]
    )
    best_configs = np.argmin(runtimes, axis=1)

    selector = DeepClusterEmbeddingSelector(
        n_configs=2,
        embedding_dim=2,
        n_clusters=2,
        seed=0,
    ).fit(features, runtimes, best_configs)
    predictions = selector.predict(features)

    assert predictions.shape == (4,)
