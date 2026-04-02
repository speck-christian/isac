"""Base selector protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Selector(Protocol):
    """Minimal trainable selector interface."""

    name: str

    def fit(self, features: np.ndarray, runtimes: np.ndarray, best_configs: np.ndarray) -> Selector:
        ...

    def predict(self, features: np.ndarray) -> np.ndarray:
        ...
