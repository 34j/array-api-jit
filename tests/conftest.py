from typing import Any

import pytest


@pytest.fixture(scope="session", params=["numpy", "torch", "jax"])
def xp(request: pytest.FixtureRequest) -> Any:
    """Get the array namespace for the given backend."""
    backend = request.param
    if backend == "numpy":
        import numpy as xp
    elif backend == "torch":
        import torch as xp
    elif backend == "jax":
        import jax.numpy as xp
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return xp
