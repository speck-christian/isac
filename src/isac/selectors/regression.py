"""Regression-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isac.selectors.portfolio_learning import derive_kmeans_portfolio, portfolio_regret_targets


@dataclass(slots=True)
class LinearRuntimeRegressorSelector:
    """Fit one linear runtime model per configuration and choose the smallest prediction."""

    n_configs: int
    max_portfolio_size: int = 12
    seed: int | None = None
    name: str = "Regressor"
    coefficients_: torch.Tensor = field(init=False)
    portfolio_values_: np.ndarray = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
        ideal_params: np.ndarray | None = None,
    ) -> LinearRuntimeRegressorSelector:
        del best_configs
        if ideal_params is not None:
            self.portfolio_values_ = derive_kmeans_portfolio(
                ideal_params,
                max_portfolio_size=self.max_portfolio_size,
                seed=self.seed,
            )
            runtimes = portfolio_regret_targets(ideal_params, self.portfolio_values_)
        else:
            self.portfolio_values_ = np.eye(self.n_configs, dtype=np.float64)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        runtime_tensor = torch.as_tensor(runtimes, dtype=torch.float32)
        bias = torch.ones((feature_tensor.shape[0], 1), dtype=feature_tensor.dtype)
        design = torch.cat([feature_tensor, bias], dim=1)
        solution = torch.linalg.lstsq(design, runtime_tensor).solution
        self.coefficients_ = solution
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        bias = torch.ones((feature_tensor.shape[0], 1), dtype=feature_tensor.dtype)
        design = torch.cat([feature_tensor, bias], dim=1)
        predictions = design @ self.coefficients_
        return predictions.argmin(dim=1).cpu().numpy().astype(np.int64)
