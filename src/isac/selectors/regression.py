"""Regression-style selectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class LinearRuntimeRegressorSelector:
    """Fit one linear runtime model per configuration and choose the smallest prediction."""

    n_configs: int
    name: str = "Regressor"
    coefficients_: np.ndarray = field(init=False)

    def fit(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        best_configs: np.ndarray,
    ) -> LinearRuntimeRegressorSelector:
        del best_configs
        bias = np.ones((len(features), 1), dtype=features.dtype)
        design = np.concatenate([features, bias], axis=1)
        coefficients = []
        for config_index in range(self.n_configs):
            coef, *_ = np.linalg.lstsq(design, runtimes[:, config_index], rcond=None)
            coefficients.append(coef)
        self.coefficients_ = np.stack(coefficients, axis=0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        bias = np.ones((len(features), 1), dtype=features.dtype)
        design = np.concatenate([features, bias], axis=1)
        predictions = design @ self.coefficients_.T
        return predictions.argmin(axis=1).astype(np.int64)
