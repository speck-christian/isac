"""Temporal soft-routing Cluster ISAC selector."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.core import DynamicPortfolioEpisode


@dataclass(slots=True)
class TemporalSoftClusterSelector:
    """History-aware soft cluster routing with local linear runtime models."""

    n_configs: int
    max_portfolio_size: int = 12
    assignment_temperature: float = 1.2
    history_blend: float = 0.72
    trend_blend: float = 0.45
    switching_penalty_weight: float = 0.75
    max_iter: int = 30
    seed: int | None = None
    name: str = "Temporal Soft Cluster ISAC"
    centers_: torch.Tensor = field(init=False)
    cluster_coefficients_: torch.Tensor = field(init=False)
    portfolio_values_: np.ndarray = field(init=False)
    feature_mean_: torch.Tensor = field(init=False)
    feature_scale_: torch.Tensor = field(init=False)

    def fit_dynamic(
        self,
        episodes: list[DynamicPortfolioEpisode],
    ) -> TemporalSoftClusterSelector:
        """Fit on dynamic episodes using temporalized observations."""

        temporal_features: list[np.ndarray] = []
        ideal_params: list[np.ndarray] = []
        for episode in episodes:
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            temporal_features.extend(self._temporalize_episode_features(episode_features))
            ideal_params.extend([state.ideal_params for state in episode.states])

        feature_array = np.stack(temporal_features, axis=0).astype(np.float64)
        ideal_param_array = np.stack(ideal_params, axis=0).astype(np.float64)
        cluster_count = min(self.max_portfolio_size, ideal_param_array.shape[0])

        self.portfolio_values_ = self._fit_portfolio_and_clusters(
            feature_array,
            ideal_param_array,
            cluster_count=cluster_count,
        )
        return self

    def _fit_portfolio_and_clusters(
        self,
        feature_array: np.ndarray,
        ideal_param_array: np.ndarray,
        *,
        cluster_count: int,
    ) -> np.ndarray:
        feature_tensor = torch.as_tensor(feature_array, dtype=torch.float32)
        self.feature_mean_ = feature_tensor.mean(dim=0)
        feature_scale = feature_tensor.std(dim=0, unbiased=False)
        self.feature_scale_ = torch.where(
            feature_scale < 1e-6,
            torch.ones_like(feature_scale),
            feature_scale,
        )
        normalized = self._normalize_tensor(feature_tensor)

        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        initial_indices = torch.randperm(normalized.shape[0], generator=generator)[:cluster_count]
        centers = normalized[initial_indices].clone()

        for _ in range(self.max_iter):
            weights = self._soft_assignments(normalized, centers)
            denominator = weights.sum(dim=0, keepdim=True).T.clamp_min(1e-8)
            new_centers = (weights.T @ normalized) / denominator
            if torch.allclose(new_centers, centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        weights = self._soft_assignments(normalized, centers).cpu().numpy()
        self.centers_ = centers

        portfolio_values: list[np.ndarray] = []
        coefficient_rows: list[np.ndarray] = []
        design = np.concatenate(
            [feature_array, np.ones((feature_array.shape[0], 1), dtype=np.float64)],
            axis=1,
        )

        for cluster_index in range(cluster_count):
            cluster_weights = weights[:, cluster_index]
            total_weight = np.maximum(cluster_weights.sum(), 1e-8)
            weighted_ideal = (
                (cluster_weights[:, None] * ideal_param_array).sum(axis=0) / total_weight
            )
            portfolio_values.append(weighted_ideal.astype(np.float64))

        portfolio_array = np.stack(portfolio_values, axis=0)
        surrogate_runtimes = (
            (ideal_param_array[:, None, :] - portfolio_array[None, :, :]) ** 2
        ).sum(axis=2)

        for cluster_index in range(cluster_count):
            cluster_weights = weights[:, cluster_index]
            weighted_design = design * np.sqrt(cluster_weights[:, None])
            weighted_targets = surrogate_runtimes * np.sqrt(cluster_weights[:, None])
            coefficients = np.linalg.lstsq(weighted_design, weighted_targets, rcond=None)[0]
            coefficient_rows.append(coefficients.astype(np.float64))

        self.cluster_coefficients_ = torch.as_tensor(
            np.stack(coefficient_rows, axis=0),
            dtype=torch.float32,
        )
        return portfolio_array

    def _normalize_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean_) / self.feature_scale_

    def _soft_assignments(self, features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        squared_distances = torch.cdist(features, centers, p=2) ** 2
        scaled_logits = -self.assignment_temperature * squared_distances
        return torch.softmax(scaled_logits, dim=1)

    def _predict_expected_runtimes(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        normalized = self._normalize_tensor(feature_tensor)
        assignments = self._soft_assignments(normalized, self.centers_)
        design = torch.cat(
            [feature_tensor, torch.ones((feature_tensor.shape[0], 1), dtype=feature_tensor.dtype)],
            dim=1,
        )
        expert_predictions = torch.einsum("nd,kdp->nkp", design, self.cluster_coefficients_)
        expected_runtimes = (assignments[:, :, None] * expert_predictions).sum(dim=1)
        return expected_runtimes.cpu().numpy()

    def predict_episode(
        self,
        episode_features: np.ndarray,
        *,
        switching_cost: float = 0.0,
    ) -> np.ndarray:
        contextual_features = self._temporalize_episode_features(episode_features)
        expected_runtimes = self._predict_expected_runtimes(contextual_features)
        actions: list[int] = []
        previous_action: int | None = None
        for timestep in range(expected_runtimes.shape[0]):
            penalized = expected_runtimes[timestep].copy()
            if previous_action is not None:
                switch_penalty = self.switching_penalty_weight * switching_cost
                penalized += switch_penalty
                penalized[previous_action] -= switch_penalty
            action = int(np.argmin(penalized))
            actions.append(action)
            previous_action = action
        return np.array(actions, dtype=np.int64)

    def _temporalize_episode_features(self, episode_features: np.ndarray) -> np.ndarray:
        history = np.zeros(episode_features.shape[1], dtype=np.float64)
        previous = np.zeros(episode_features.shape[1], dtype=np.float64)
        contextual_rows: list[np.ndarray] = []
        for timestep, current in enumerate(episode_features):
            current = np.asarray(current, dtype=np.float64)
            if timestep == 0:
                delta = np.zeros_like(current)
            else:
                delta = current - previous
            history = self.history_blend * history + (1.0 - self.history_blend) * current
            trend = self.trend_blend * delta + (1.0 - self.trend_blend) * (current - history)
            contextual_rows.append(np.concatenate([current, history, delta, trend], axis=0))
            previous = current
        return np.stack(contextual_rows, axis=0)
