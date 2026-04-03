"""Dynamic synthetic portfolio benchmark with evolving instance state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isac.core.portfolio import ParameterConfig


@dataclass(slots=True)
class DynamicPortfolioState:
    """One timestep of a drifting instance sequence."""

    features: np.ndarray
    latent_features: np.ndarray
    observation_mask: np.ndarray
    ideal_params: np.ndarray
    base_difficulty: float
    runtimes: np.ndarray
    best_config: int
    regime_id: int
    timestep: int


@dataclass(slots=True)
class DynamicPortfolioEpisode:
    """A full evolving instance trajectory."""

    states: list[DynamicPortfolioState]


@dataclass(slots=True)
class DynamicPortfolioBenchmark:
    """Benchmark where a single instance evolves over time.

    Each episode starts in a latent regime, then drifts through feature and
    parameter space over time. Regimes can occasionally switch, making adaptive
    reconfiguration meaningful.
    """

    horizon: int = 16
    n_features: int = 5
    n_parameter_dims: int = 2
    feature_noise: float = 0.15
    parameter_noise: float = 0.04
    runtime_noise: float = 0.02
    drift_scale: float = 0.18
    regime_switch_prob: float = 0.12
    switching_cost: float = 0.04
    observation_noise: float = 0.10
    missing_feature_prob: float = 0.22
    multimodal_surface_scale: float = 0.30
    seed: int | None = None
    rng: np.random.Generator = field(init=False)
    regime_centers: np.ndarray = field(init=False)
    regime_ideal_params: np.ndarray = field(init=False)
    portfolio: tuple[ParameterConfig, ...] = field(init=False)
    difficulty_weights: np.ndarray = field(init=False)
    config_modes: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.regime_centers = np.array(
            [
                [-2.0, -1.2, 0.4, 1.8, 0.5],
                [1.6, 1.1, -0.8, -1.4, 0.2],
                [0.2, -2.1, 1.7, 0.4, -1.8],
            ],
            dtype=np.float64,
        )[:, : self.n_features]
        self.regime_ideal_params = np.array(
            [
                [0.15, 0.85],
                [0.85, 0.25],
                [0.55, 0.60],
            ],
            dtype=np.float64,
        )[:, : self.n_parameter_dims]
        self.portfolio = (
            ParameterConfig("conservative", np.array([0.1, 0.9], dtype=np.float64)),
            ParameterConfig("aggressive", np.array([0.9, 0.2], dtype=np.float64)),
            ParameterConfig("balanced", np.array([0.55, 0.55], dtype=np.float64)),
            ParameterConfig("robust", np.array([0.35, 0.7], dtype=np.float64)),
        )
        self.difficulty_weights = self.rng.uniform(0.4, 1.1, size=self.n_features)
        base_modes = np.array(
            [
                [[0.10, 0.90], [0.32, 0.74]],
                [[0.90, 0.18], [0.72, 0.42]],
                [[0.56, 0.56], [0.22, 0.82]],
                [[0.36, 0.70], [0.84, 0.26]],
            ],
            dtype=np.float64,
        )
        self.config_modes = base_modes[:, :, : self.n_parameter_dims]

    @property
    def n_configs(self) -> int:
        return len(self.portfolio)

    def _state_modes(self, state: DynamicPortfolioState) -> np.ndarray:
        regime_anchor = self.regime_ideal_params[state.regime_id]
        mirrored = 1.0 - state.ideal_params
        secondary_mode = np.clip(0.55 * regime_anchor + 0.45 * mirrored, 0.0, 1.0)
        return np.stack([state.ideal_params, secondary_mode], axis=0)

    def evaluate_parameters(
        self,
        state: DynamicPortfolioState,
        parameter_values: np.ndarray,
        *,
        add_noise: bool = False,
    ) -> float:
        """Evaluate an arbitrary parameter vector on a dynamic state."""

        parameter_values = np.asarray(parameter_values, dtype=np.float64)
        state_modes = self._state_modes(state)
        modal_mismatch = float(
            np.min(((state_modes - parameter_values[None, :]) ** 2).sum(axis=1))
        )
        latent_signature = np.sin(state.latent_features[: self.n_parameter_dims] * 1.7)
        phase = float(
            np.sin(
                float(np.dot(latent_signature, parameter_values))
                + 0.7 * (state.regime_id + 1)
            )
            ** 2
        )
        runtime = (
            state.base_difficulty
            + 2.8 * modal_mismatch
            + self.multimodal_surface_scale * phase
        )
        if add_noise:
            runtime += float(self.rng.normal(0.0, self.runtime_noise))
        return max(runtime, 0.01)

    def evaluate_portfolio(
        self,
        state: DynamicPortfolioState,
        portfolio_values: np.ndarray,
        *,
        add_noise: bool = False,
    ) -> np.ndarray:
        """Evaluate a learned portfolio on a dynamic state."""

        return np.array(
            [
                self.evaluate_parameters(state, parameter_values, add_noise=add_noise)
                for parameter_values in portfolio_values
            ],
            dtype=np.float64,
        )

    def optimal_runtime(self, state: DynamicPortfolioState) -> float:
        """Continuous oracle runtime under the state's multimodal surface."""

        return float(
            min(self.evaluate_parameters(state, mode) for mode in self._state_modes(state))
        )

    def sample_episode(self) -> DynamicPortfolioEpisode:
        regime_id = int(self.rng.integers(0, len(self.regime_centers)))
        latent_features = self.regime_centers[regime_id] + self.rng.normal(
            0.0,
            self.feature_noise,
            size=self.n_features,
        )
        ideal_params = np.clip(
            self.regime_ideal_params[regime_id]
            + self.rng.normal(0.0, self.parameter_noise, size=self.n_parameter_dims),
            0.0,
            1.0,
        )

        states: list[DynamicPortfolioState] = []
        previous_observation: np.ndarray | None = None
        for timestep in range(self.horizon):
            if timestep > 0 and float(self.rng.random()) < self.regime_switch_prob:
                choices = [index for index in range(len(self.regime_centers)) if index != regime_id]
                regime_id = int(self.rng.choice(choices))

            center_pull = self.regime_centers[regime_id] - latent_features
            latent_features = (
                latent_features
                + self.drift_scale * center_pull
                + self.rng.normal(0.0, self.feature_noise, size=self.n_features)
            )

            parameter_pull = self.regime_ideal_params[regime_id] - ideal_params
            feature_signal = np.tanh(latent_features[: self.n_parameter_dims]) * 0.06
            ideal_params = np.clip(
                ideal_params
                + self.drift_scale * parameter_pull
                + feature_signal
                + self.rng.normal(0.0, self.parameter_noise, size=self.n_parameter_dims),
                0.0,
                1.0,
            )

            observed_features = latent_features + self.rng.normal(
                0.0,
                self.observation_noise,
                size=self.n_features,
            )
            observation_mask = (
                self.rng.random(self.n_features) >= self.missing_feature_prob
            ).astype(np.float64)
            if previous_observation is None:
                previous_observation = observed_features.copy()
            observed_features = np.where(
                observation_mask > 0.5,
                observed_features,
                previous_observation,
            )
            previous_observation = observed_features.copy()

            base_difficulty = 1.0 + float(np.abs(latent_features * self.difficulty_weights).mean())
            scaffold_state = DynamicPortfolioState(
                features=observed_features.astype(np.float64).copy(),
                latent_features=latent_features.astype(np.float64).copy(),
                observation_mask=observation_mask.astype(np.float64).copy(),
                ideal_params=ideal_params.astype(np.float64).copy(),
                base_difficulty=float(base_difficulty),
                runtimes=np.empty(0, dtype=np.float64),
                best_config=0,
                regime_id=regime_id,
                timestep=timestep,
            )
            runtimes_array = self.evaluate_portfolio(
                scaffold_state,
                np.stack([config.values for config in self.portfolio], axis=0),
                add_noise=True,
            )
            states.append(
                DynamicPortfolioState(
                    features=observed_features.astype(np.float64).copy(),
                    latent_features=latent_features.astype(np.float64).copy(),
                    observation_mask=observation_mask.astype(np.float64).copy(),
                    ideal_params=ideal_params.astype(np.float64).copy(),
                    base_difficulty=float(base_difficulty),
                    runtimes=runtimes_array,
                    best_config=int(np.argmin(runtimes_array)),
                    regime_id=regime_id,
                    timestep=timestep,
                )
            )
        return DynamicPortfolioEpisode(states=states)
