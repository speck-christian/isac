"""Utilities for evaluating and visualizing the portfolio benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from isac.core import (
    AlgorithmicEpisode,
    AlgorithmicPortfolioBenchmark,
    AlgorithmicState,
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
    RandomSearchPortfolioBuilder,
    SMAC3PortfolioBuilder,
    TemporalMixtureOfExpertsSelector,
    TemporalSoftClusterSelector,
)
from isac.selectors.portfolio_learning import assign_to_portfolio, derive_kmeans_portfolio


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


def _flatten_algorithmic_states(episodes: list[AlgorithmicEpisode]) -> list[AlgorithmicState]:
    return [state for episode in episodes for state in episode.states]


def _stack_algorithmic_features(states: list[AlgorithmicState]) -> np.ndarray:
    return np.stack([state.features for state in states], axis=0)


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


def _normalize_algorithmic_splits(
    train_episodes: list[AlgorithmicEpisode],
    test_episodes: list[AlgorithmicEpisode],
) -> tuple[list[AlgorithmicEpisode], list[AlgorithmicEpisode]]:
    train_states = _flatten_algorithmic_states(train_episodes)
    test_states = _flatten_algorithmic_states(test_episodes)
    train_features = _stack_algorithmic_features(train_states)
    test_features = _stack_algorithmic_features(test_states)
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

    oracle_portfolio = derive_kmeans_portfolio(
        train_ideal_params,
        max_portfolio_size=12,
        seed=seed,
    )
    oracle_assignments = assign_to_portfolio(_stack_ideal_params(test_instances), oracle_portfolio)
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
    selectors["Oracle"] = (
        oracle_assignments,
        oracle_portfolio,
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
                    "config_name": f"s{config_index}",
                    "param_1": float(selector_portfolio[config_index, 0]),
                    "param_2": (
                        float(selector_portfolio[config_index, 1])
                        if selector_portfolio.shape[1] > 1
                        else 0.0
                    ),
                    "count": int(count),
                }
            )
        selector_key = _selector_key(selector_name)
        instance_table[f"choice_{selector_key}"] = selected_configs
        instance_table[f"choice_name_{selector_key}"] = [f"s{index}" for index in selected_configs]
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
    oracle_portfolio = derive_kmeans_portfolio(
        train_ideal_params,
        max_portfolio_size=12,
        seed=seed,
    )
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
                selector_portfolio = oracle_portfolio
                episode_ideal_params = np.stack(
                    [state.ideal_params for state in episode.states],
                    axis=0,
                )
                episode_actions = assign_to_portfolio(episode_ideal_params, selector_portfolio)
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


def _algorithmic_runtime_sequences(
    benchmark: AlgorithmicPortfolioBenchmark,
    episodes: list[AlgorithmicEpisode],
    portfolio_values: np.ndarray,
) -> list[np.ndarray]:
    runtime_sequences: list[np.ndarray] = []
    for episode in episodes:
        losses = np.stack(
            [benchmark.rollout_parameters(episode, params) for params in portfolio_values],
            axis=1,
        )
        runtime_sequences.append(losses.astype(np.float64))
    return runtime_sequences


def evaluate_algorithmic_selectors(
    benchmark: AlgorithmicPortfolioBenchmark,
    n_episodes: int,
    seed: int | None = None,
    *,
    portfolio_source: str = "random_search",
    portfolio_trials: int = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate selectors on the concrete algorithmic benchmark."""

    local_benchmark = AlgorithmicPortfolioBenchmark(
        horizon=benchmark.horizon,
        n_features=benchmark.n_features,
        observation_noise=benchmark.observation_noise,
        regime_switch_prob=benchmark.regime_switch_prob,
        seed=seed if seed is not None else benchmark.seed,
    )
    train_episodes = [local_benchmark.sample_episode() for _ in range(n_episodes)]
    test_episodes = [local_benchmark.sample_episode() for _ in range(n_episodes)]
    train_episodes, test_episodes = _normalize_algorithmic_splits(train_episodes, test_episodes)

    model_rows: list[dict[str, float | str]] = []
    trace_rows: list[dict[str, float | int | str]] = []

    selector_specs = [
        ("Privileged Classifier", NearestCentroidClassifierSelector),
        ("Regressor", LinearRuntimeRegressorSelector),
        ("MLP Selector", MLPClassifierSelector),
        ("DGCAC-inspired", DeepClusterEmbeddingSelector),
        ("Cluster ISAC", KMeansClusterSelector),
        ("Temporal Soft Cluster ISAC", TemporalSoftClusterSelector),
    ]

    for selector_index, (_name, selector_cls) in enumerate(selector_specs):
        selector_seed = (seed or 0) + selector_index
        if portfolio_source == "smac":
            builder = SMAC3PortfolioBuilder(
                benchmark=local_benchmark,
                n_trials=portfolio_trials,
                max_portfolio_size=12,
                seed=selector_seed,
            )
        elif portfolio_source == "random_search":
            builder = RandomSearchPortfolioBuilder(
                benchmark=local_benchmark,
                n_trials=portfolio_trials,
                max_portfolio_size=12,
                seed=selector_seed,
            )
        else:
            raise ValueError("portfolio_source must be 'random_search' or 'smac'.")

        selector_portfolio = builder.build_portfolio(train_episodes)
        train_runtime_sequences = _algorithmic_runtime_sequences(
            local_benchmark,
            train_episodes,
            selector_portfolio,
        )
        train_states = _flatten_algorithmic_states(train_episodes)
        train_features = _stack_algorithmic_features(train_states)
        train_runtimes = np.concatenate(train_runtime_sequences, axis=0)
        train_best_configs = train_runtimes.argmin(axis=1).astype(np.int64)

        selector = selector_cls(
            n_configs=len(selector_portfolio),
            max_portfolio_size=len(selector_portfolio),
            seed=selector_seed,
        )
        if hasattr(selector, "fit_algorithmic_with_portfolio"):
            selector.fit_algorithmic_with_portfolio(
                train_episodes,
                portfolio_values=selector_portfolio,
                runtime_sequences=train_runtime_sequences,
            )
        else:
            selector.fit(
                train_features,
                train_runtimes,
                train_best_configs,
                ideal_params=None,
            )
            selector.portfolio_values_ = selector_portfolio

        total_loss = 0.0
        total_regret = 0.0
        step_count = 0
        for episode_id, episode in enumerate(test_episodes):
            portfolio_step_losses = _algorithmic_runtime_sequences(
                local_benchmark,
                [episode],
                selector_portfolio,
            )[0]
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            if hasattr(selector, "predict_episode"):
                episode_actions = selector.predict_episode(episode_features, switching_cost=0.0)
            else:
                episode_actions = selector.predict(episode_features)

            algorithm_state = local_benchmark._initialize_algorithm_state(episode)
            for state_index, (state, action) in enumerate(
                zip(episode.states, episode_actions, strict=True)
            ):
                parameter_values = selector_portfolio[int(action)]
                loss, algorithm_state = local_benchmark.algorithm_step(
                    algorithm_state,
                    state,
                    parameter_values,
                )
                best_loss = float(portfolio_step_losses[state_index].min())
                total_loss += float(loss)
                total_regret += float(loss - best_loss)
                step_count += 1
                trace_rows.append(
                    {
                        "selector": selector.name,
                        "episode_id": episode_id,
                        "timestep": state.timestep,
                        "regime_id": state.regime_id,
                        "action": int(action),
                        "loss": float(loss),
                        "best_loss": best_loss,
                        "regret": float(loss - best_loss),
                        "selected_param_1": float(parameter_values[0]),
                        "selected_param_2": (
                            float(parameter_values[1]) if len(parameter_values) > 1 else 0.0
                        ),
                        "selected_param_3": (
                            float(parameter_values[2]) if len(parameter_values) > 2 else 0.0
                        ),
                    }
                )

        model_rows.append(
            {
                "selector": selector.name,
                "portfolio_source": portfolio_source,
                "portfolio_size": int(len(selector_portfolio)),
                "avg_loss": total_loss / step_count,
                "avg_regret": total_regret / step_count,
            }
        )

    return pd.DataFrame(model_rows), pd.DataFrame(trace_rows)
