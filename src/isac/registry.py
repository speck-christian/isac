"""Small environment registry with a Gym-like interface."""

from __future__ import annotations

from collections.abc import Callable

RegistryFactory = Callable[..., object]

_REGISTRY: dict[str, RegistryFactory] = {}


def register(env_id: str, factory: RegistryFactory) -> None:
    """Register an environment factory under a stable identifier."""
    _REGISTRY[env_id] = factory


def make(env_id: str, **kwargs: object) -> object:
    """Create a registered environment instance."""
    try:
        factory = _REGISTRY[env_id]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown environment '{env_id}'. Available: {available}") from exc
    return factory(**kwargs)


from isac.envs.isac_dynamic import ISACDynamicEnv  # noqa: E402
from isac.envs.isac_simple import ISACSimpleEnv  # noqa: E402

register("isac-simple-v0", ISACSimpleEnv)
register("isac-dynamic-v0", ISACDynamicEnv)
