"""Feature normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ZScoreNormalizer:
    """Simple z-score normalizer with a NumPy-friendly API."""

    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> ZScoreNormalizer:
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(mean_=mean, scale_=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean_) / self.scale_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        fitted = self.fit(features)
        self.mean_ = fitted.mean_
        self.scale_ = fitted.scale_
        return self.transform(features)
