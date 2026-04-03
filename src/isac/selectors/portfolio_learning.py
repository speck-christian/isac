"""Helpers for deriving small learned parameter portfolios."""

from __future__ import annotations

import numpy as np
import torch


def derive_kmeans_portfolio(
    ideal_params: np.ndarray,
    *,
    max_portfolio_size: int,
    seed: int | None = None,
    max_iter: int = 30,
) -> np.ndarray:
    """Summarize ideal parameter vectors with up to ``max_portfolio_size`` centers."""

    if ideal_params.ndim != 2:
        raise ValueError("ideal_params must be a 2D array.")

    cluster_count = max(1, min(max_portfolio_size, ideal_params.shape[0]))
    param_tensor = torch.as_tensor(ideal_params, dtype=torch.float32)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    initial_indices = torch.randperm(param_tensor.shape[0], generator=generator)[:cluster_count]
    centers = param_tensor[initial_indices].clone()

    assignments = torch.zeros(param_tensor.shape[0], dtype=torch.long)
    for _ in range(max_iter):
        distances = torch.cdist(param_tensor, centers, p=2) ** 2
        new_assignments = distances.argmin(dim=1)
        if torch.equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_index in range(cluster_count):
            mask = assignments == cluster_index
            if bool(mask.any()):
                centers[cluster_index] = param_tensor[mask].mean(dim=0)

    return centers.cpu().numpy().astype(np.float64)


def assign_to_portfolio(ideal_params: np.ndarray, portfolio_values: np.ndarray) -> np.ndarray:
    """Assign each ideal parameter vector to its nearest derived portfolio member."""

    squared_distances = ((ideal_params[:, None, :] - portfolio_values[None, :, :]) ** 2).sum(axis=2)
    return squared_distances.argmin(axis=1).astype(np.int64)


def portfolio_regret_targets(ideal_params: np.ndarray, portfolio_values: np.ndarray) -> np.ndarray:
    """Distance-based surrogate runtimes for training learned-portfolio selectors."""

    return ((ideal_params[:, None, :] - portfolio_values[None, :, :]) ** 2).sum(axis=2).astype(
        np.float64
    )
