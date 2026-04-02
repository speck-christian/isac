"""Small baseline policies."""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Uniformly samples a configuration."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, _: np.ndarray) -> int:
        return int(self.rng.integers(0, self.n_actions))
