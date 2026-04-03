"""Regression-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(slots=True)
class LinearRuntimeRegressorSelector:
    """Fit one linear runtime model per configuration and choose the smallest prediction."""

    n_configs: int
    name: str = "Regressor"
    coefficients_: torch.Tensor = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
    ) -> LinearRuntimeRegressorSelector:
        del best_configs
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
