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
    ideal_params = np.array(
        [
            [0.12, 0.88],
            [0.16, 0.84],
            [0.86, 0.24],
            [0.82, 0.28],
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
        max_portfolio_size=2,
        embedding_dim=2,
        n_clusters=2,
        seed=0,
    ).fit(features, runtimes, best_configs, ideal_params=ideal_params)
    predictions = selector.predict(features)
    embedding = selector.transform(features)

    assert predictions.shape == (4,)
    assert embedding.shape == (4, 2)
    assert np.all(predictions >= 0)
    assert np.all(predictions < selector.portfolio_values_.shape[0])
    assert selector.portfolio_values_.shape[0] <= 2


def test_dgcac_inspired_selector_predicts_episode_actions() -> None:
    selector = DeepClusterEmbeddingSelector(
        n_configs=2,
        max_portfolio_size=3,
        embedding_dim=2,
        n_clusters=2,
        seed=0,
        encoder_epochs=40,
    )
    episode_a = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
        ]
    )
    episode_b = np.array(
        [
            [2.8, 3.0, 3.1],
            [3.0, 3.1, 3.0],
            [3.2, 3.1, 2.9],
        ]
    )
    features = np.vstack([episode_a, episode_b])
    ideal_params = np.array(
        [
            [0.12, 0.88],
            [0.16, 0.84],
            [0.20, 0.80],
            [0.86, 0.24],
            [0.84, 0.26],
            [0.82, 0.28],
        ]
    )
    runtimes = np.array(
        [
            [1.0, 2.0],
            [1.1, 1.9],
            [1.2, 1.8],
            [2.2, 1.0],
            [2.1, 1.1],
            [2.0, 1.2],
        ]
    )
    best_configs = np.argmin(runtimes, axis=1)

    selector.fit(features, runtimes, best_configs, ideal_params=ideal_params)
    episode_predictions = selector.predict_episode(episode_a, switching_cost=0.04)

    assert episode_predictions.shape == (3,)
    assert np.all(episode_predictions >= 0)
    assert np.all(episode_predictions < selector.portfolio_values_.shape[0])
