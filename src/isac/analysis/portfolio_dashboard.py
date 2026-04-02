"""Utilities for evaluating and visualizing the portfolio benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from isac.core import PortfolioBenchmark, PortfolioInstance, ZScoreNormalizer
from isac.selectors import (
    DeepClusterEmbeddingSelector,
    KMeansClusterSelector,
    LinearRuntimeRegressorSelector,
    NearestCentroidClassifierSelector,
)


def _stack_features(instances: list[PortfolioInstance]) -> np.ndarray:
    return np.stack([instance.features for instance in instances], axis=0)


def _stack_runtimes(instances: list[PortfolioInstance]) -> np.ndarray:
    return np.stack([instance.runtimes for instance in instances], axis=0)


def _stack_best_configs(instances: list[PortfolioInstance]) -> np.ndarray:
    return np.array([instance.best_config for instance in instances], dtype=np.int64)


def _normalize_splits(
    train_instances: list[PortfolioInstance],
    test_instances: list[PortfolioInstance],
) -> tuple[list[PortfolioInstance], list[PortfolioInstance]]:
    train_features = _stack_features(train_instances)
    test_features = _stack_features(test_instances)
    normalizer = ZScoreNormalizer.fit(train_features)
    normalized_train = normalizer.transform(train_features)
    normalized_test = normalizer.transform(test_features)

    for instance, normalized_features in zip(train_instances, normalized_train, strict=True):
        instance.features = normalized_features
    for instance, normalized_features in zip(test_instances, normalized_test, strict=True):
        instance.features = normalized_features
    return train_instances, test_instances


def make_instance_table(
    instances: list[PortfolioInstance],
    benchmark: PortfolioBenchmark,
) -> pd.DataFrame:
    """Create a row-wise table of instances and portfolio outcomes."""

    rows: list[dict[str, float | int | str]] = []
    for instance_id, instance in enumerate(instances):
        row: dict[str, float | int | str] = {
            "instance_id": instance_id,
            "regime_id": instance.regime_id,
            "best_config": benchmark.portfolio[instance.best_config].name,
            "best_config_index": instance.best_config,
            "ideal_param_1": float(instance.ideal_params[0]),
            "ideal_param_2": (
                float(instance.ideal_params[1]) if len(instance.ideal_params) > 1 else 0.0
            ),
        }
        for feature_index, value in enumerate(instance.features):
            row[f"feature_{feature_index + 1}"] = float(value)
        for config_index, runtime in enumerate(instance.runtimes):
            row[f"runtime_{benchmark.portfolio[config_index].name}"] = float(runtime)
        rows.append(row)
    return pd.DataFrame(rows)


def make_portfolio_table(benchmark: PortfolioBenchmark) -> pd.DataFrame:
    """Create a table describing the fixed parameter portfolio."""

    rows = []
    for config_index, config in enumerate(benchmark.portfolio):
        rows.append(
            {
                "config_index": config_index,
                "config_name": config.name,
                "param_1": float(config.values[0]),
                "param_2": float(config.values[1]) if len(config.values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def make_feature_embedding_table(instances: list[PortfolioInstance]) -> pd.DataFrame:
    """Create a simple 2D embedding table from feature vectors.

    A lightweight PCA-style projection keeps the dashboard dependency surface
    small while still exposing cluster structure visually.
    """

    features = _stack_features(instances)
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[:2].T

    rows = []
    for instance_id, _instance in enumerate(instances):
        rows.append(
            {
                "instance_id": instance_id,
                "x": float(projection[instance_id, 0]),
                "y": float(projection[instance_id, 1] if projection.shape[1] > 1 else 0.0),
            }
        )
    return pd.DataFrame(rows)


def _choose_global_best(instances: list[PortfolioInstance]) -> int:
    runtimes = _stack_runtimes(instances)
    return int(np.argmin(runtimes.mean(axis=0)))


def evaluate_selectors(
    benchmark: PortfolioBenchmark,
    n_instances: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train and evaluate a family of selectors on held-out synthetic instances."""

    local_benchmark = PortfolioBenchmark(
        n_features=benchmark.n_features,
        feature_noise=benchmark.feature_noise,
        parameter_noise=benchmark.parameter_noise,
        runtime_noise=benchmark.runtime_noise,
        seed=seed if seed is not None else benchmark.seed,
        n_parameter_dims=benchmark.n_parameter_dims,
    )
    train_instances = local_benchmark.sample_batch(n_instances=n_instances, normalize=False)
    test_instances = local_benchmark.sample_batch(n_instances=n_instances, normalize=False)
    train_instances, test_instances = _normalize_splits(train_instances, test_instances)

    train_features = _stack_features(train_instances)
    train_runtimes = _stack_runtimes(train_instances)
    train_best_configs = _stack_best_configs(train_instances)

    test_features = _stack_features(test_instances)
    test_runtimes = _stack_runtimes(test_instances)
    test_best_configs = _stack_best_configs(test_instances)

    instance_table = make_instance_table(test_instances, local_benchmark)
    embedding_table = make_feature_embedding_table(test_instances)

    global_best = _choose_global_best(train_instances)

    selectors: dict[str, np.ndarray] = {
        "Oracle": test_best_configs.copy(),
        "Global Best": np.full(n_instances, global_best, dtype=np.int64),
        "Random": local_benchmark.rng.integers(0, local_benchmark.n_configs, size=n_instances),
    }
    model_selectors = [
        KMeansClusterSelector(n_configs=local_benchmark.n_configs, seed=seed),
        DeepClusterEmbeddingSelector(n_configs=local_benchmark.n_configs, seed=seed),
        NearestCentroidClassifierSelector(n_configs=local_benchmark.n_configs),
        LinearRuntimeRegressorSelector(n_configs=local_benchmark.n_configs),
    ]
    for selector in model_selectors:
        selector.fit(train_features, train_runtimes, train_best_configs)
        selectors[selector.name] = selector.predict(test_features)

    summary_rows = []
    action_rows = []
    for selector_name, selected_configs in selectors.items():
        selected_runtimes = test_runtimes[np.arange(n_instances), selected_configs]
        oracle_runtimes = test_runtimes[np.arange(n_instances), test_best_configs]
        regrets = selected_runtimes - oracle_runtimes
        histogram = np.bincount(selected_configs, minlength=local_benchmark.n_configs)

        summary_rows.append(
            {
                "selector": selector_name,
                "avg_runtime": float(selected_runtimes.mean()),
                "avg_regret": float(regrets.mean()),
                "accuracy": float((selected_configs == test_best_configs).mean()),
            }
        )
        for config_index, count in enumerate(histogram):
            action_rows.append(
                {
                    "selector": selector_name,
                    "config_name": local_benchmark.portfolio[config_index].name,
                    "count": int(count),
                }
            )
        instance_table[f"choice_{selector_name.lower().replace(' ', '_')}"] = selected_configs
        instance_table[f"choice_name_{selector_name.lower().replace(' ', '_')}"] = [
            local_benchmark.portfolio[index].name for index in selected_configs
        ]
        instance_table[f"regret_{selector_name.lower().replace(' ', '_')}"] = regrets

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(action_rows),
        instance_table.merge(embedding_table, on="instance_id", how="left").assign(
            regime_label=lambda frame: frame["regime_id"].map(lambda value: f"Regime {value}")
        ),
    )
