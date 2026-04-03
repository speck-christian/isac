"""Algorithm-driven dynamic benchmark for portfolio learning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isac.core.portfolio import ParameterConfig


@dataclass(slots=True)
class ForecastAlgorithmState:
    """Internal state of the forecasting algorithm."""

    level: float
    trend: float


@dataclass(slots=True)
class AlgorithmicState:
    """One timestep in an algorithm-driven episode."""

    features: np.ndarray
    target: float
    observed_value: float
    runtimes: np.ndarray
    best_config: int
    regime_id: int
    timestep: int


@dataclass(slots=True)
class AlgorithmicEpisode:
    """Full sequence for the algorithmic benchmark."""

    states: list[AlgorithmicState]


@dataclass(slots=True)
class AlgorithmicPortfolioBenchmark:
    """Dynamic benchmark built around one concrete parameterized algorithm.

    The underlying algorithm is a simple adaptive forecaster with three tunable
    parameters:

    - alpha: level adaptation rate
    - beta: trend adaptation rate
    - gate: volatility sensitivity that dampens overreaction under noisy input

    Episodes are generated from latent regimes with different trend, volatility,
    mean reversion, and shock characteristics. Observable features are rolling
    statistics computed from the observed stream itself.
    """

    horizon: int = 24
    n_features: int = 5
    observation_noise: float = 0.08
    regime_switch_prob: float = 0.14
    seed: int | None = None
    rng: np.random.Generator = field(init=False)
    portfolio: tuple[ParameterConfig, ...] = field(init=False)
    regime_parameters: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.portfolio = (
            ParameterConfig("stable", np.array([0.16, 0.10, 3.0], dtype=np.float64)),
            ParameterConfig("responsive", np.array([0.72, 0.32, 0.4], dtype=np.float64)),
            ParameterConfig("trend_aware", np.array([0.48, 0.82, 1.1], dtype=np.float64)),
            ParameterConfig("shock_robust", np.array([0.28, 0.22, 4.2], dtype=np.float64)),
        )
        # anchor, drift, seasonal_amp, noise_scale, shock_prob, shock_scale, mean_reversion
        self.regime_parameters = np.array(
            [
                [0.2, 0.015, 0.08, 0.05, 0.04, 0.18, 0.22],
                [0.9, 0.065, 0.03, 0.08, 0.10, 0.30, 0.08],
                [-0.4, -0.025, 0.16, 0.12, 0.14, 0.45, 0.14],
            ],
            dtype=np.float64,
        )

    @property
    def n_configs(self) -> int:
        return len(self.portfolio)

    def _advance_signal(
        self,
        signal: float,
        timestep: int,
        regime_id: int,
    ) -> float:
        anchor, drift, seasonal_amp, noise_scale, shock_prob, shock_scale, mean_reversion = (
            self.regime_parameters[regime_id]
        )
        seasonal = seasonal_amp * np.sin((2.0 * np.pi * timestep) / max(self.horizon, 2))
        shock = 0.0
        if float(self.rng.random()) < shock_prob:
            shock = float(self.rng.normal(0.0, shock_scale))
        innovation = float(self.rng.normal(0.0, noise_scale))
        return float(
            signal
            + drift
            + mean_reversion * (anchor - signal)
            + seasonal
            + shock
            + innovation
        )

    def _build_features(self, history: list[float]) -> np.ndarray:
        window = np.asarray(history[-5:], dtype=np.float64)
        diffs = np.diff(window) if window.shape[0] > 1 else np.zeros(1, dtype=np.float64)
        slope = float(window[-1] - window[0]) / max(window.shape[0] - 1, 1)
        volatility = float(diffs.std()) if diffs.size > 0 else 0.0
        jump = float(np.abs(diffs[-1])) if diffs.size > 0 else 0.0
        mean_abs_diff = float(np.abs(diffs).mean()) if diffs.size > 0 else 0.0
        feature_vector = np.array(
            [
                float(window.mean()),
                volatility,
                slope,
                jump,
                mean_abs_diff,
            ],
            dtype=np.float64,
        )
        return feature_vector[: self.n_features]

    def _initialize_algorithm_state(self, episode: AlgorithmicEpisode) -> ForecastAlgorithmState:
        first_value = episode.states[0].observed_value
        return ForecastAlgorithmState(level=float(first_value), trend=0.0)

    def algorithm_step(
        self,
        algorithm_state: ForecastAlgorithmState,
        state: AlgorithmicState,
        parameter_values: np.ndarray,
    ) -> tuple[float, ForecastAlgorithmState]:
        """Run one forecasting step with the given parameter vector."""

        alpha = float(np.clip(parameter_values[0], 0.02, 0.98))
        beta = float(np.clip(parameter_values[1], 0.02, 0.98))
        gate = float(np.clip(parameter_values[2], 0.0, 6.0)) if len(parameter_values) > 2 else 0.0

        prediction = algorithm_state.level + algorithm_state.trend
        error = state.target - prediction
        volatility = max(float(state.features[1]), 1e-3)
        slope_signal = abs(float(state.features[2]))

        effective_alpha = float(np.clip(alpha / (1.0 + gate * volatility), 0.02, 0.98))
        effective_beta = float(np.clip(beta * (1.0 + 0.35 * np.tanh(slope_signal)), 0.02, 0.98))

        new_level = algorithm_state.level + effective_alpha * error
        new_trend = (1.0 - effective_beta) * algorithm_state.trend + effective_beta * (
            new_level - algorithm_state.level
        )
        loss = float(error**2 + 0.015 * abs(new_trend))
        return loss, ForecastAlgorithmState(level=float(new_level), trend=float(new_trend))

    def rollout_parameters(
        self,
        episode: AlgorithmicEpisode,
        parameter_values: np.ndarray,
    ) -> np.ndarray:
        """Evaluate one parameter vector across an episode."""

        algorithm_state = self._initialize_algorithm_state(episode)
        losses: list[float] = []
        for state in episode.states:
            loss, algorithm_state = self.algorithm_step(algorithm_state, state, parameter_values)
            losses.append(loss)
        return np.asarray(losses, dtype=np.float64)

    def evaluate_parameters(
        self,
        episode: AlgorithmicEpisode,
        parameter_values: np.ndarray,
    ) -> float:
        """Return mean loss across the full episode."""

        return float(self.rollout_parameters(episode, parameter_values).mean())

    def evaluate_portfolio(
        self,
        episode: AlgorithmicEpisode,
        portfolio_values: np.ndarray,
    ) -> np.ndarray:
        """Evaluate all portfolio members on a shared episode."""

        return np.asarray(
            [self.evaluate_parameters(episode, params) for params in portfolio_values],
            dtype=np.float64,
        )

    def sample_episode(self) -> AlgorithmicEpisode:
        regime_id = int(self.rng.integers(0, len(self.regime_parameters)))
        signal = float(self.regime_parameters[regime_id, 0] + self.rng.normal(0.0, 0.06))

        observed_history: list[float] = []
        states: list[AlgorithmicState] = []
        for timestep in range(self.horizon):
            if timestep > 0 and float(self.rng.random()) < self.regime_switch_prob:
                choices = [
                    index for index in range(len(self.regime_parameters)) if index != regime_id
                ]
                regime_id = int(self.rng.choice(choices))

            signal = self._advance_signal(signal, timestep, regime_id)
            observed_value = float(signal + self.rng.normal(0.0, self.observation_noise))
            observed_history.append(observed_value)
            features = self._build_features(observed_history)
            states.append(
                AlgorithmicState(
                    features=features,
                    target=float(signal),
                    observed_value=observed_value,
                    runtimes=np.empty(0, dtype=np.float64),
                    best_config=0,
                    regime_id=regime_id,
                    timestep=timestep,
                )
            )

        episode = AlgorithmicEpisode(states=states)
        portfolio_values = np.stack([config.values for config in self.portfolio], axis=0)
        config_step_losses = np.stack(
            [self.rollout_parameters(episode, params) for params in portfolio_values],
            axis=1,
        )

        finalized_states: list[AlgorithmicState] = []
        for state, losses in zip(states, config_step_losses, strict=True):
            finalized_states.append(
                AlgorithmicState(
                    features=state.features.copy(),
                    target=state.target,
                    observed_value=state.observed_value,
                    runtimes=losses.astype(np.float64),
                    best_config=int(np.argmin(losses)),
                    regime_id=state.regime_id,
                    timestep=state.timestep,
                )
            )
        return AlgorithmicEpisode(states=finalized_states)
