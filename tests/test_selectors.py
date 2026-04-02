from __future__ import annotations

import numpy as np

from isac.selectors import (
    KMeansClusterSelector,
    LinearRuntimeRegressorSelector,
    NearestCentroidClassifierSelector,
)


def test_classifier_selector_predicts_shape() -> None:
    features = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 1.1]])
    runtimes = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [1.1, 1.8],
            [1.9, 1.0],
        ]
    )
    best_configs = np.argmin(runtimes, axis=1)

    selector = NearestCentroidClassifierSelector(n_configs=2).fit(features, runtimes, best_configs)
    predictions = selector.predict(features)

    assert predictions.shape == (4,)


def test_regression_selector_predicts_shape() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    runtimes = np.array(
        [
            [1.0, 4.0],
            [1.5, 3.0],
            [2.0, 2.0],
            [2.5, 1.0],
        ]
    )
    best_configs = np.argmin(runtimes, axis=1)

    selector = LinearRuntimeRegressorSelector(n_configs=2).fit(features, runtimes, best_configs)
    predictions = selector.predict(features)

    assert predictions.shape == (4,)


def test_cluster_selector_predicts_shape() -> None:
    features = np.array([[0.0, 0.0], [0.2, 0.1], [3.0, 3.0], [3.1, 3.2]])
    runtimes = np.array(
        [
            [1.0, 2.0],
            [1.1, 1.9],
            [2.2, 1.0],
            [2.0, 1.1],
        ]
    )
    best_configs = np.argmin(runtimes, axis=1)

    selector = KMeansClusterSelector(n_configs=2, n_clusters=2, seed=0).fit(
        features,
        runtimes,
        best_configs,
    )
    predictions = selector.predict(features)

    assert predictions.shape == (4,)
