import shutil
from typing import Any

import pytest


@pytest.fixture(
    scope="session",
    params=[
        "numpy",
        pytest.param(
            "torch",
            marks=pytest.mark.skipif(
                shutil.which("g++") is None,
                reason="Torch JIT requires a C++ compiler",
            ),
        ),
        "jax",
    ],
)
def xp(request: pytest.FixtureRequest) -> Any:
    """Get the array namespace for the given backend.

    Returns
    -------
    Any
        The selected array namespace.

    Raises
    ------
    ValueError
        If the requested backend is unknown.

    """
    backend = request.param
    if backend == "numpy":
        import numpy as np

        return np
    if backend == "torch":
        import torch as xp

        return xp
    if backend == "jax":
        import jax.numpy as xp

        return xp
    raise ValueError(f"Unknown backend: {backend}")
