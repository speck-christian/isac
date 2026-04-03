"""Utilities for evaluating and visualizing the portfolio benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from isac.core import (
    DynamicPortfolioBenchmark,
    DynamicPortfolioEpisode,
    DynamicPortfolioState,
    PortfolioBenchmark,
    PortfolioInstance,
    ZScoreNormalizer,
)
from isac.selectors import (
    DeepClusterEmbeddingSelector,
    KMeansClusterSelector,
    LinearRuntimeRegressorSelector,
    MLPClassifierSelector,
    NearestCentroidClassifierSelector,
    TemporalMixtureOfExpertsSelector,
    TemporalSoftClusterSelector,
)


def _stack_features(instances: list[PortfolioInstance]) -> np.ndarray:
    return np.stack([instance.features for instance in instances], axis=0)


def _stack_runtimes(instances: list[PortfolioInstance]) -> np.ndarray:
    return np.stack([instance.runtimes for instance in instances], axis=0)


def _stack_best_configs(instances: list[PortfolioInstance]) -> np.ndarray:
    return np.array([instance.best_config for instance in instances], dtype=np.int64)


def _stack_ideal_params(instances: list[PortfolioInstance]) -> np.ndarray:
    return np.stack([instance.ideal_params for instance in instances], axis=0)


def _stack_dynamic_features(states: list[DynamicPortfolioState]) -> np.ndarray:
    return np.stack([state.features for state in states], axis=0)


def _stack_dynamic_runtimes(states: list[DynamicPortfolioState]) -> np.ndarray:
    return np.stack([state.runtimes for state in states], axis=0)


def _stack_dynamic_best_configs(states: list[DynamicPortfolioState]) -> np.ndarray:
    return np.array([state.best_config for state in states], dtype=np.int64)


def _stack_dynamic_ideal_params(states: list[DynamicPortfolioState]) -> np.ndarray:
    return np.stack([state.ideal_params for state in states], axis=0)


def _flatten_episode_states(episodes: list[DynamicPortfolioEpisode]) -> list[DynamicPortfolioState]:
    return [state for episode in episodes for state in episode.states]


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


def _normalize_dynamic_splits(
    train_episodes: list[DynamicPortfolioEpisode],
    test_episodes: list[DynamicPortfolioEpisode],
) -> tuple[list[DynamicPortfolioEpisode], list[DynamicPortfolioEpisode]]:
    train_states = _flatten_episode_states(train_episodes)
    test_states = _flatten_episode_states(test_episodes)
    train_features = _stack_dynamic_features(train_states)
    test_features = _stack_dynamic_features(test_states)
    normalizer = ZScoreNormalizer.fit(train_features)
    normalized_train = normalizer.transform(train_features)
    normalized_test = normalizer.transform(test_features)

    for state, normalized_features in zip(train_states, normalized_train, strict=True):
        state.features = normalized_features
    for state, normalized_features in zip(test_states, normalized_test, strict=True):
        state.features = normalized_features
    return train_episodes, test_episodes


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


def _selector_key(selector_name: str) -> str:
    return selector_name.lower().replace(" ", "_")


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
    train_ideal_params = _stack_ideal_params(train_instances)

    test_features = _stack_features(test_instances)

    instance_table = make_instance_table(test_instances, local_benchmark)
    embedding_table = make_feature_embedding_table(test_instances)

    global_best_param = train_ideal_params.mean(axis=0)
    random_portfolio = train_ideal_params[
        local_benchmark.rng.choice(
            train_ideal_params.shape[0],
            size=min(12, train_ideal_params.shape[0]),
            replace=False,
        )
    ]

    selectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    model_selectors = [
        KMeansClusterSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        DeepClusterEmbeddingSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        NearestCentroidClassifierSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        LinearRuntimeRegressorSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        MLPClassifierSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        TemporalMixtureOfExpertsSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
    ]
    for selector in model_selectors:
        selector.fit(
            train_features,
            train_runtimes,
            train_best_configs,
            ideal_params=train_ideal_params,
        )
        selectors[selector.name] = (
            selector.predict(test_features),
            selector.portfolio_values_,
        )

    summary_rows = []
    action_rows = []
    oracle_selected_params = np.stack(
        [instance.ideal_params for instance in test_instances],
        axis=0,
    )
    selectors["Oracle"] = (
        np.zeros(n_instances, dtype=np.int64),
        np.zeros((1, train_ideal_params.shape[1])),
    )
    selectors["Global Best"] = (
        np.zeros(n_instances, dtype=np.int64),
        np.repeat(global_best_param[None, :], repeats=1, axis=0),
    )
    selectors["Random"] = (
        local_benchmark.rng.integers(0, len(random_portfolio), size=n_instances),
        random_portfolio,
    )

    selector_order = [
        "Oracle",
        "Global Best",
        "Random",
        *[selector.name for selector in model_selectors],
    ]
    for selector_name in selector_order:
        selected_configs, selector_portfolio = selectors[selector_name]
        if selector_name == "Oracle":
            selected_runtimes = np.array(
                [local_benchmark.optimal_runtime(instance) for instance in test_instances],
                dtype=np.float64,
            )
            best_runtimes = selected_runtimes.copy()
            selected_param_values = oracle_selected_params
            histogram = np.array([n_instances], dtype=np.int64)
        else:
            selected_param_values = selector_portfolio[selected_configs]
            selected_runtimes = np.array(
                [
                    local_benchmark.evaluate_parameters(instance, selected_param_values[index])
                    for index, instance in enumerate(test_instances)
                ],
                dtype=np.float64,
            )
            portfolio_runtimes = np.stack(
                [
                    local_benchmark.evaluate_portfolio(instance, selector_portfolio)
                    for instance in test_instances
                ],
                axis=0,
            )
            best_runtimes = portfolio_runtimes.min(axis=1)
            histogram = np.bincount(selected_configs, minlength=len(selector_portfolio))
        oracle_runtimes = np.array(
            [local_benchmark.optimal_runtime(instance) for instance in test_instances],
            dtype=np.float64,
        )
        regrets = selected_runtimes - oracle_runtimes
        accuracy = float(np.isclose(selected_runtimes, best_runtimes, atol=1e-8).mean())

        summary_rows.append(
            {
                "selector": selector_name,
                "avg_runtime": float(selected_runtimes.mean()),
                "avg_regret": float(regrets.mean()),
                "accuracy": accuracy,
            }
        )
        for config_index, count in enumerate(histogram):
            action_rows.append(
                {
                    "selector": selector_name,
                    "config_name": "oracle" if selector_name == "Oracle" else f"s{config_index}",
                    "param_1": (
                        float("nan")
                        if selector_name == "Oracle"
                        else float(selector_portfolio[config_index, 0])
                    ),
                    "param_2": (
                        float("nan")
                        if selector_name == "Oracle"
                        else (
                            float(selector_portfolio[config_index, 1])
                            if selector_portfolio.shape[1] > 1
                            else 0.0
                        )
                    ),
                    "count": int(count),
                }
            )
        selector_key = _selector_key(selector_name)
        instance_table[f"choice_{selector_key}"] = selected_configs
        instance_table[f"choice_name_{selector_key}"] = (
            ["oracle"] * n_instances
            if selector_name == "Oracle"
            else [f"s{index}" for index in selected_configs]
        )
        instance_table[f"selected_param_1_{selector_key}"] = selected_param_values[:, 0]
        instance_table[f"selected_param_2_{selector_key}"] = (
            selected_param_values[:, 1] if selected_param_values.shape[1] > 1 else 0.0
        )
        instance_table[f"regret_{selector_key}"] = regrets

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(action_rows),
        instance_table.merge(embedding_table, on="instance_id", how="left").assign(
            regime_label=lambda frame: frame["regime_id"].map(lambda value: f"Regime {value}")
        ),
    )


def evaluate_dynamic_selectors(
    benchmark: DynamicPortfolioBenchmark,
    n_episodes: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate selectors on evolving episodes with switching costs."""

    local_benchmark = DynamicPortfolioBenchmark(
        horizon=benchmark.horizon,
        n_features=benchmark.n_features,
        n_parameter_dims=benchmark.n_parameter_dims,
        feature_noise=benchmark.feature_noise,
        parameter_noise=benchmark.parameter_noise,
        runtime_noise=benchmark.runtime_noise,
        drift_scale=benchmark.drift_scale,
        regime_switch_prob=benchmark.regime_switch_prob,
        switching_cost=benchmark.switching_cost,
        observation_noise=benchmark.observation_noise,
        missing_feature_prob=benchmark.missing_feature_prob,
        multimodal_surface_scale=benchmark.multimodal_surface_scale,
        seed=seed if seed is not None else benchmark.seed,
    )
    train_episodes = [local_benchmark.sample_episode() for _ in range(n_episodes)]
    test_episodes = [local_benchmark.sample_episode() for _ in range(n_episodes)]
    train_episodes, test_episodes = _normalize_dynamic_splits(train_episodes, test_episodes)

    train_states = _flatten_episode_states(train_episodes)
    train_features = _stack_dynamic_features(train_states)
    train_runtimes = _stack_dynamic_runtimes(train_states)
    train_best_configs = _stack_dynamic_best_configs(train_states)
    train_ideal_params = _stack_dynamic_ideal_params(train_states)
    global_best_param = train_ideal_params.mean(axis=0)
    random_portfolio = train_ideal_params[
        local_benchmark.rng.choice(
            train_ideal_params.shape[0],
            size=min(12, train_ideal_params.shape[0]),
            replace=False,
        )
    ]

    model_selectors = [
        KMeansClusterSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        DeepClusterEmbeddingSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        NearestCentroidClassifierSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        LinearRuntimeRegressorSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        MLPClassifierSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        TemporalMixtureOfExpertsSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
        TemporalSoftClusterSelector(
            n_configs=local_benchmark.n_configs,
            max_portfolio_size=12,
            seed=seed,
        ),
    ]

    selector_models: dict[str, tuple[object, np.ndarray]] = {}
    for selector in model_selectors:
        if hasattr(selector, "fit_dynamic"):
            selector.fit_dynamic(train_episodes)
        else:
            selector.fit(
                train_features,
                train_runtimes,
                train_best_configs,
                ideal_params=train_ideal_params,
            )
        selector_models[selector.name] = (selector, selector.portfolio_values_)

    summary_rows: list[dict[str, float | str]] = []
    trace_rows: list[dict[str, float | int | str]] = []

    selector_names = [
        "Oracle",
        "Global Best",
        "Random",
        *selector_models.keys(),
    ]
    for selector_name in selector_names:
        total_runtime = 0.0
        total_regret = 0.0
        total_switch_cost = 0.0
        correct_count = 0
        switch_count = 0
        step_count = 0

        for episode_id, episode in enumerate(test_episodes):
            previous_action: int | None = None
            episode_features = _stack_dynamic_features(episode.states)
            if selector_name == "Oracle":
                episode_actions = np.zeros(len(episode.states), dtype=np.int64)
                selector_portfolio = np.stack(
                    [state.ideal_params for state in episode.states],
                    axis=0,
                )
            elif selector_name == "Global Best":
                episode_actions = np.zeros(len(episode.states), dtype=np.int64)
                selector_portfolio = np.repeat(global_best_param[None, :], repeats=1, axis=0)
            elif selector_name == "Random":
                episode_actions = local_benchmark.rng.integers(
                    0,
                    len(random_portfolio),
                    size=len(episode.states),
                )
                selector_portfolio = random_portfolio
            else:
                selector_model, selector_portfolio = selector_models[selector_name]
                if hasattr(selector_model, "predict_episode"):
                    episode_actions = selector_model.predict_episode(
                        episode_features,
                        switching_cost=local_benchmark.switching_cost,
                    )
                else:
                    episode_actions = selector_model.predict(episode_features)

            for state, action in zip(episode.states, episode_actions, strict=True):
                if selector_name == "Oracle":
                    selected_runtime = local_benchmark.optimal_runtime(state)
                    best_runtime = selected_runtime
                    selected_params = state.ideal_params
                else:
                    selected_params = selector_portfolio[int(action)]
                    selected_runtime = float(
                        local_benchmark.evaluate_parameters(state, selected_params)
                    )
                    best_runtime = float(
                        local_benchmark.evaluate_portfolio(state, selector_portfolio).min()
                    )
                switch_cost = 0.0
                if previous_action is not None and int(action) != previous_action:
                    switch_cost = local_benchmark.switching_cost
                    switch_count += 1
                oracle_runtime = local_benchmark.optimal_runtime(state)
                total_runtime += selected_runtime
                total_regret += selected_runtime - oracle_runtime
                total_switch_cost += switch_cost
                correct_count += int(np.isclose(selected_runtime, best_runtime, atol=1e-8))
                step_count += 1
                row = {
                    "selector": selector_name,
                    "episode_id": episode_id,
                    "timestep": state.timestep,
                    "regime_id": state.regime_id,
                    "best_config": state.best_config,
                    "action": int(action),
                    "runtime": selected_runtime,
                    "regret": selected_runtime - oracle_runtime,
                    "switch_cost": switch_cost,
                    "total_penalty": selected_runtime - oracle_runtime + switch_cost,
                    "selected_param_1": float(selected_params[0]),
                    "selected_param_2": (
                        float(selected_params[1]) if len(selected_params) > 1 else 0.0
                    ),
                }
                for feature_index, value in enumerate(state.features):
                    row[f"feature_{feature_index + 1}"] = float(value)
                for feature_index, value in enumerate(state.latent_features):
                    row[f"latent_feature_{feature_index + 1}"] = float(value)
                for feature_index, value in enumerate(state.observation_mask):
                    row[f"mask_{feature_index + 1}"] = float(value)
                for param_index, value in enumerate(state.ideal_params):
                    row[f"ideal_param_{param_index + 1}"] = float(value)
                trace_rows.append(row)
                previous_action = int(action)

        summary_rows.append(
            {
                "selector": selector_name,
                "avg_runtime": total_runtime / step_count,
                "avg_regret": total_regret / step_count,
                "avg_switch_cost": total_switch_cost / step_count,
                "avg_total_penalty": (total_regret + total_switch_cost) / step_count,
                "accuracy": correct_count / step_count,
                "switch_rate": switch_count / max(step_count - n_episodes, 1),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(trace_rows)
