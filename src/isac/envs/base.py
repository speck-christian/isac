"""Small protocol for simulation environments."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Env(Protocol):
    """Minimal Gym-style environment protocol."""

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        ...
