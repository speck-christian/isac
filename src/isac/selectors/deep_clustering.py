"""DGCAC-inspired selector with a learned nonlinear embedding stage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.core import AlgorithmicEpisode, DynamicPortfolioEpisode
from isac.selectors.portfolio_learning import (
    assign_to_portfolio,
    derive_kmeans_portfolio,
    portfolio_regret_targets,
)


@dataclass(slots=True)
class DeepClusterEmbeddingSelector:
    """Approximate DGCAC with denoising, cost-aware embedding, and soft routing."""

    n_configs: int
    max_portfolio_size: int = 12
    embedding_dim: int = 4
    hidden_dim: int = 16
    n_clusters: int | None = None
    encoder_epochs: int = 300
    learning_rate: float = 0.02
    denoising_noise: float = 0.08
    assignment_temperature: float = 1.5
    history_blend: float = 0.72
    trend_blend: float = 0.45
    switching_penalty_weight: float = 0.75
    cluster_runtime_weight: float = 0.40
    local_runtime_weight: float = 0.35
    head_runtime_weight: float = 0.20
    fallback_runtime_weight: float = 0.05
    ridge_penalty: float = 0.06
    algorithmic_validation_fraction: float = 0.25
    min_validation_episodes: int = 2
    reconstruction_weight: float = 0.6
    runtime_weight: float = 0.3
    classification_weight: float = 0.1
    seed: int | None = None
    name: str = "DGCAC-inspired"
    feature_mean_: torch.Tensor = field(init=False)
    feature_scale_: torch.Tensor = field(init=False)
    encoder_: torch.nn.Module = field(init=False)
    decoder_: torch.nn.Module = field(init=False)
    runtime_head_: torch.nn.Module = field(init=False)
    classifier_head_: torch.nn.Module = field(init=False)
    cluster_centers_: torch.Tensor = field(init=False)
    cluster_runtime_means_: torch.Tensor = field(init=False)
    cluster_runtime_coefficients_: torch.Tensor = field(init=False)
    fallback_runtime_: torch.Tensor = field(init=False)
    portfolio_values_: np.ndarray = field(init=False)
    temporal_mode_: bool = field(init=False, default=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
        ideal_params: np.ndarray | None = None,
    ) -> DeepClusterEmbeddingSelector:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if ideal_params is not None:
            self.portfolio_values_ = derive_kmeans_portfolio(
                ideal_params,
                max_portfolio_size=self.max_portfolio_size,
                seed=self.seed,
            )
            runtimes = portfolio_regret_targets(ideal_params, self.portfolio_values_)
            best_configs = assign_to_portfolio(ideal_params, self.portfolio_values_)
        else:
            self.portfolio_values_ = np.eye(self.n_configs, dtype=np.float64)

        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        runtime_tensor = torch.as_tensor(runtimes, dtype=torch.float32)
        label_tensor = torch.as_tensor(best_configs, dtype=torch.long)
        self.feature_mean_ = feature_tensor.mean(dim=0)
        feature_scale = feature_tensor.std(dim=0, unbiased=False)
        self.feature_scale_ = torch.where(
            feature_scale < 1e-6,
            torch.ones_like(feature_scale),
            feature_scale,
        )
        normalized = self._normalize_tensor(feature_tensor)

        input_dim = normalized.shape[1]
        embedding_dim = min(self.embedding_dim, input_dim)
        hidden_dim = max(self.hidden_dim, embedding_dim + 2)

        self.encoder_ = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder_ = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )
        self.runtime_head_ = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(self.portfolio_values_)),
        )
        self.classifier_head_ = torch.nn.Linear(embedding_dim, len(self.portfolio_values_))

        optimizer = torch.optim.Adam(
            list(self.encoder_.parameters())
            + list(self.decoder_.parameters())
            + list(self.runtime_head_.parameters())
            + list(self.classifier_head_.parameters()),
            lr=self.learning_rate,
        )
        mse_loss = torch.nn.MSELoss()
        cross_entropy = torch.nn.CrossEntropyLoss()

        self.encoder_.train()
        self.decoder_.train()
        self.runtime_head_.train()
        self.classifier_head_.train()
        for _ in range(self.encoder_epochs):
            optimizer.zero_grad()
            corrupted = normalized + torch.randn_like(normalized) * self.denoising_noise
            embedded = self.encoder_(corrupted)
            reconstructed = self.decoder_(embedded)
            runtime_predictions = self.runtime_head_(embedded)
            class_logits = self.classifier_head_(embedded)

            loss = (
                self.reconstruction_weight * mse_loss(reconstructed, normalized)
                + self.runtime_weight * mse_loss(runtime_predictions, runtime_tensor)
                + self.classification_weight * cross_entropy(class_logits, label_tensor)
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            embedded_features = self._transform_tensor(feature_tensor)
        self._fit_soft_clusters(embedded_features, runtime_tensor)
        self.fallback_runtime_ = runtime_tensor.mean(dim=0)
        self.temporal_mode_ = False
        return self

    def fit_dynamic(
        self,
        episodes: list[DynamicPortfolioEpisode],
    ) -> DeepClusterEmbeddingSelector:
        """Fit the selector on dynamic episodes using short-horizon temporal context."""

        temporal_features: list[np.ndarray] = []
        runtimes: list[np.ndarray] = []
        best_configs: list[int] = []
        ideal_params: list[np.ndarray] = []
        for episode in episodes:
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            contextual_features = self._temporalize_episode_features(episode_features)
            temporal_features.extend(contextual_features)
            runtimes.extend([state.runtimes for state in episode.states])
            best_configs.extend([state.best_config for state in episode.states])
            ideal_params.extend([state.ideal_params for state in episode.states])

        self.fit(
            np.stack(temporal_features, axis=0),
            np.stack(runtimes, axis=0),
            np.array(best_configs, dtype=np.int64),
            ideal_params=np.stack(ideal_params, axis=0),
        )
        self.temporal_mode_ = True
        return self

    def fit_algorithmic_with_portfolio(
        self,
        episodes: list[AlgorithmicEpisode],
        *,
        portfolio_values: np.ndarray,
        runtime_sequences: list[np.ndarray],
    ) -> DeepClusterEmbeddingSelector:
        self.portfolio_values_ = np.asarray(portfolio_values, dtype=np.float64)
        if len(episodes) < 4:
            feature_array, runtime_array = self._stack_algorithmic_training_rows(
                episodes,
                runtime_sequences,
            )
            self.fit(
                feature_array,
                runtime_array,
                runtime_array.argmin(axis=1).astype(np.int64),
                ideal_params=None,
            )
            self.portfolio_values_ = np.asarray(portfolio_values, dtype=np.float64)
            self.temporal_mode_ = True
            return self

        validation_count = max(
            self.min_validation_episodes,
            int(round(len(episodes) * self.algorithmic_validation_fraction)),
        )
        validation_count = min(validation_count, len(episodes) - 1)
        train_count = len(episodes) - validation_count
        train_episodes = episodes[:train_count]
        validation_episodes = episodes[train_count:]
        train_runtime_sequences = runtime_sequences[:train_count]
        validation_runtime_sequences = runtime_sequences[train_count:]

        train_feature_array, train_runtime_array = self._stack_algorithmic_training_rows(
            train_episodes,
            train_runtime_sequences,
        )
        validation_feature_rows = [
            self._temporalize_episode_features(
                np.stack([state.features for state in episode.states], axis=0)
            )
            for episode in validation_episodes
        ]

        original_cluster_runtime_weight = self.cluster_runtime_weight
        original_local_runtime_weight = self.local_runtime_weight
        original_head_runtime_weight = self.head_runtime_weight
        original_fallback_runtime_weight = self.fallback_runtime_weight
        original_assignment_temperature = self.assignment_temperature
        original_n_clusters = self.n_clusters

        candidate_cluster_counts = self._algorithmic_candidate_cluster_counts(
            len(self.portfolio_values_)
        )
        candidate_temperatures = [0.9, original_assignment_temperature, 2.1]
        candidate_blends = [
            (0.55, 0.20, 0.20, 0.05),
            (0.45, 0.30, 0.20, 0.05),
            (0.35, 0.40, 0.20, 0.05),
        ]

        best_score = float("inf")
        best_snapshot: dict[str, object] | None = None
        for cluster_count in candidate_cluster_counts:
            self.n_clusters = int(cluster_count)
            for assignment_temperature in candidate_temperatures:
                self.assignment_temperature = float(assignment_temperature)
                for (
                    cluster_weight,
                    local_weight,
                    head_weight,
                    fallback_weight,
                ) in candidate_blends:
                    self.cluster_runtime_weight = cluster_weight
                    self.local_runtime_weight = local_weight
                    self.head_runtime_weight = head_weight
                    self.fallback_runtime_weight = fallback_weight
                    self.fit(
                        train_feature_array,
                        train_runtime_array,
                        train_runtime_array.argmin(axis=1).astype(np.int64),
                        ideal_params=None,
                    )
                    validation_score = self._score_algorithmic_validation(
                        validation_feature_rows,
                        validation_runtime_sequences,
                    )
                    if validation_score < best_score - 1e-9:
                        best_score = validation_score
                        best_snapshot = self._snapshot_algorithmic_state()

        if best_snapshot is not None:
            self.cluster_runtime_weight = float(best_snapshot["cluster_runtime_weight"])
            self.local_runtime_weight = float(best_snapshot["local_runtime_weight"])
            self.head_runtime_weight = float(best_snapshot["head_runtime_weight"])
            self.fallback_runtime_weight = float(best_snapshot["fallback_runtime_weight"])
            self.assignment_temperature = float(best_snapshot["assignment_temperature"])
            self.n_clusters = int(best_snapshot["n_clusters"])

        full_feature_array, full_runtime_array = self._stack_algorithmic_training_rows(
            episodes,
            runtime_sequences,
        )
        self.fit(
            full_feature_array,
            full_runtime_array,
            full_runtime_array.argmin(axis=1).astype(np.int64),
            ideal_params=None,
        )
        self.portfolio_values_ = np.asarray(portfolio_values, dtype=np.float64)
        self.temporal_mode_ = True
        if best_snapshot is None:
            self.cluster_runtime_weight = original_cluster_runtime_weight
            self.local_runtime_weight = original_local_runtime_weight
            self.head_runtime_weight = original_head_runtime_weight
            self.fallback_runtime_weight = original_fallback_runtime_weight
            self.assignment_temperature = original_assignment_temperature
            self.n_clusters = original_n_clusters
        return self

    def _stack_algorithmic_training_rows(
        self,
        episodes: list[AlgorithmicEpisode],
        runtime_sequences: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        temporal_features: list[np.ndarray] = []
        runtime_rows: list[np.ndarray] = []
        for episode, runtime_sequence in zip(episodes, runtime_sequences, strict=True):
            episode_features = np.stack([state.features for state in episode.states], axis=0)
            temporal_features.extend(self._temporalize_episode_features(episode_features))
            runtime_rows.extend(runtime_sequence)
        return (
            np.stack(temporal_features, axis=0).astype(np.float64),
            np.stack(runtime_rows, axis=0).astype(np.float64),
        )

    def _algorithmic_candidate_cluster_counts(self, portfolio_size: int) -> list[int]:
        raw_counts = [2, min(4, portfolio_size), portfolio_size]
        counts = sorted({count for count in raw_counts if 1 <= count <= portfolio_size})
        return counts or [min(self.max_portfolio_size, portfolio_size)]

    def _score_algorithmic_validation(
        self,
        validation_feature_rows: list[np.ndarray],
        validation_runtime_sequences: list[np.ndarray],
    ) -> float:
        regrets: list[float] = []
        for episode_features, runtime_sequence in zip(
            validation_feature_rows,
            validation_runtime_sequences,
            strict=True,
        ):
            actions = self.predict_episode(episode_features, switching_cost=0.0)
            chosen = runtime_sequence[np.arange(len(actions)), actions]
            best = runtime_sequence.min(axis=1)
            regrets.append(float((chosen - best).mean()))
        return float(np.mean(regrets))

    def _snapshot_algorithmic_state(self) -> dict[str, object]:
        return {
            "feature_mean": self.feature_mean_.clone(),
            "feature_scale": self.feature_scale_.clone(),
            "cluster_centers": self.cluster_centers_.clone(),
            "cluster_runtime_means": self.cluster_runtime_means_.clone(),
            "cluster_runtime_coefficients": self.cluster_runtime_coefficients_.clone(),
            "fallback_runtime": self.fallback_runtime_.clone(),
            "encoder_state": {
                key: value.detach().clone() for key, value in self.encoder_.state_dict().items()
            },
            "decoder_state": {
                key: value.detach().clone() for key, value in self.decoder_.state_dict().items()
            },
            "runtime_head_state": {
                key: value.detach().clone()
                for key, value in self.runtime_head_.state_dict().items()
            },
            "classifier_head_state": {
                key: value.detach().clone()
                for key, value in self.classifier_head_.state_dict().items()
            },
            "cluster_runtime_weight": self.cluster_runtime_weight,
            "local_runtime_weight": self.local_runtime_weight,
            "head_runtime_weight": self.head_runtime_weight,
            "fallback_runtime_weight": self.fallback_runtime_weight,
            "assignment_temperature": self.assignment_temperature,
            "n_clusters": (
                self.n_clusters
                if self.n_clusters is not None
                else len(self.portfolio_values_)
            ),
        }

    def _normalize_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean_) / self.feature_scale_

    def _transform_tensor(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_tensor(features)
        self.encoder_.eval()
        return self.encoder_(normalized)

    def _fit_soft_clusters(
        self,
        embedded_features: torch.Tensor,
        runtimes: torch.Tensor,
    ) -> None:
        cluster_count = self.n_clusters or len(self.portfolio_values_)
        initial_indices = torch.randperm(embedded_features.shape[0])[:cluster_count]
        centers = embedded_features[initial_indices].clone()

        for _ in range(30):
            weights = self._soft_assignments(embedded_features, centers)
            denominator = weights.sum(dim=0, keepdim=True).T.clamp_min(1e-8)
            new_centers = (weights.T @ embedded_features) / denominator
            if torch.allclose(new_centers, centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        weights = self._soft_assignments(embedded_features, centers)
        runtime_means = []
        design = torch.cat(
            [
                embedded_features,
                torch.ones((embedded_features.shape[0], 1), dtype=embedded_features.dtype),
            ],
            dim=1,
        )
        runtime_coefficients = []
        for cluster_index in range(cluster_count):
            cluster_weights = weights[:, cluster_index]
            total_weight = cluster_weights.sum().clamp_min(1e-8)
            weighted_runtime = (cluster_weights[:, None] * runtimes).sum(dim=0) / total_weight
            runtime_means.append(weighted_runtime)
            weight_root = torch.sqrt(cluster_weights[:, None].clamp_min(1e-8))
            weighted_design = design * weight_root
            weighted_targets = runtimes * weight_root
            gram = weighted_design.T @ weighted_design
            regularizer = self.ridge_penalty * torch.eye(
                gram.shape[0],
                dtype=gram.dtype,
                device=gram.device,
            )
            regularizer[-1, -1] = 0.0
            rhs = weighted_design.T @ weighted_targets
            runtime_coefficients.append(torch.linalg.solve(gram + regularizer, rhs))

        self.cluster_centers_ = centers
        self.cluster_runtime_means_ = torch.stack(runtime_means, dim=0)
        self.cluster_runtime_coefficients_ = torch.stack(runtime_coefficients, dim=0)

    def _soft_assignments(
        self,
        embedded_features: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        squared_distances = torch.cdist(embedded_features, centers, p=2) ** 2
        scaled_logits = -self.assignment_temperature * squared_distances
        return torch.softmax(scaled_logits, dim=1)

    def transform(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            embedded = self._transform_tensor(feature_tensor)
        return embedded.cpu().numpy()

    def _predict_expected_runtimes(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            embedded = self._transform_tensor(feature_tensor)
            assignments = self._soft_assignments(embedded, self.cluster_centers_)
            embedded_design = torch.cat(
                [embedded, torch.ones((embedded.shape[0], 1), dtype=embedded.dtype)],
                dim=1,
            )
            local_runtime_predictions = torch.einsum(
                "nd,kdp->nkp",
                embedded_design,
                self.cluster_runtime_coefficients_,
            )
            cluster_expected_runtimes = assignments @ self.cluster_runtime_means_
            head_expected_runtimes = self.runtime_head_(embedded)
            expected_runtimes = (
                self.cluster_runtime_weight * cluster_expected_runtimes
                + self.local_runtime_weight
                * (assignments[:, :, None] * local_runtime_predictions).sum(dim=1)
                + self.head_runtime_weight * head_expected_runtimes
                + self.fallback_runtime_weight * self.fallback_runtime_[None, :]
            )
        return expected_runtimes.cpu().numpy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        expected_runtimes = self._predict_expected_runtimes(features)
        return expected_runtimes.argmin(axis=1).astype(np.int64)

    def predict_episode(
        self,
        episode_features: np.ndarray,
        *,
        switching_cost: float = 0.0,
    ) -> np.ndarray:
        """Predict a sequence of actions using temporal context and switch awareness."""

        contextual_features = (
            self._temporalize_episode_features(episode_features)
            if self.temporal_mode_
            else np.asarray(episode_features, dtype=np.float64)
        )
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
        """Build history-aware features from one normalized episode trace."""

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
