"""DGCAC-inspired selector with a learned nonlinear embedding stage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(slots=True)
class DeepClusterEmbeddingSelector:
    """Approximate DGCAC with a denoising encoder plus soft cluster routing."""

    n_configs: int
    embedding_dim: int = 3
    hidden_dim: int = 12
    n_clusters: int | None = None
    encoder_epochs: int = 250
    learning_rate: float = 0.02
    denoising_noise: float = 0.08
    assignment_temperature: float = 1.5
    seed: int | None = None
    name: str = "DGCAC-inspired"
    feature_mean_: torch.Tensor = field(init=False)
    feature_scale_: torch.Tensor = field(init=False)
    encoder_: torch.nn.Module = field(init=False)
    decoder_: torch.nn.Module = field(init=False)
    cluster_centers_: torch.Tensor = field(init=False)
    cluster_runtime_means_: torch.Tensor = field(init=False)
    fallback_runtime_: torch.Tensor = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
    ) -> DeepClusterEmbeddingSelector:
        del best_configs
        if self.seed is not None:
            torch.manual_seed(self.seed)

        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        runtime_tensor = torch.as_tensor(runtimes, dtype=torch.float32)
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
        hidden_dim = max(self.hidden_dim, embedding_dim + 1)

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
        optimizer = torch.optim.Adam(
            list(self.encoder_.parameters()) + list(self.decoder_.parameters()),
            lr=self.learning_rate,
        )
        loss_fn = torch.nn.MSELoss()

        self.encoder_.train()
        self.decoder_.train()
        for _ in range(self.encoder_epochs):
            optimizer.zero_grad()
            corrupted = normalized + torch.randn_like(normalized) * self.denoising_noise
            embedded = self.encoder_(corrupted)
            reconstructed = self.decoder_(embedded)
            loss = loss_fn(reconstructed, normalized)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            embedded_features = self._transform_tensor(feature_tensor)
        self._fit_soft_clusters(embedded_features, runtime_tensor)
        self.fallback_runtime_ = runtime_tensor.mean(dim=0)
        return self

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
        cluster_count = self.n_clusters or self.n_configs
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
        for cluster_index in range(cluster_count):
            cluster_weights = weights[:, cluster_index]
            total_weight = cluster_weights.sum().clamp_min(1e-8)
            weighted_runtime = (cluster_weights[:, None] * runtimes).sum(dim=0) / total_weight
            runtime_means.append(weighted_runtime)

        self.cluster_centers_ = centers
        self.cluster_runtime_means_ = torch.stack(runtime_means, dim=0)

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

    def predict(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            embedded = self._transform_tensor(feature_tensor)
            assignments = self._soft_assignments(embedded, self.cluster_centers_)
            expected_runtimes = assignments @ self.cluster_runtime_means_
            expected_runtimes = 0.9 * expected_runtimes + 0.1 * self.fallback_runtime_[None, :]
        return expected_runtimes.argmin(dim=1).cpu().numpy().astype(np.int64)
