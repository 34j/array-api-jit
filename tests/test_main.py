from typing import Any

import numba
import numpy as np
from array_api_compat import array_namespace
from cm_time import timer
from numba import prange
from numba.extending import overload

from array_api_jit.main import jit


@overload(np.stack)
def stack(arrays, axis=0):
    def inner(arrays, axis=0):
        if axis == 0:
            shape = (len(arrays),) + arrays[0].shape  # noqa: RUF005
            stacked_array = np.empty(shape, dtype=arrays[0].dtype)
            for j in prange(len(arrays)):
                stacked_array[j] = arrays[j]
        elif axis == -1:
            shape = arrays[0].shape + (len(arrays),)  # noqa: RUF005
            stacked_array = np.empty(shape, dtype=arrays[0].dtype)
            for j in prange(len(arrays)):
                stacked_array[..., j] = arrays[j]
        return stacked_array

    return inner


def legendre(x: Any, n_end: int) -> Any:
    """
    Legendre polynomial of order 0 to n_end-1.

    Parameters
    ----------
    x : Any
        The points at which to evaluate the polynomial.
    n_end : int
        The number of orders to compute, starting from 0.

    Returns
    -------
    Any
        Array of shape (*x.shape, n_end),
        where [..., i] is the value of the i-th order polynomial at x.

    """
    xp = array_namespace(x)
    prevprev = xp.zeros_like(x)
    prev = xp.ones_like(x)
    result = [prevprev, prev]
    for n in range(2, n_end):
        prevprev, prev = prev, (((2 * n - 1) * x * prev - (n - 1) * prevprev) / n)
        result.append(prev)
    return xp.stack(result[:n_end], axis=-1)


def legendre_assign(x: Any, n_end: int) -> Any:
    """
    Legendre polynomial of order 0 to n_end-1.

    Parameters
    ----------
    x : Any
        The points at which to evaluate the polynomial.
    n_end : int
        The number of orders to compute, starting from 0.

    Returns
    -------
    Any
        Array of shape (*x.shape, n_end),
        where [..., i] is the value of the i-th order polynomial at x.

    """
    xp = array_namespace(x)
    prevprev = xp.zeros_like(x)
    prev = xp.ones_like(x)
    result = xp.empty((*x.shape, n_end), dtype=x.dtype)
    if n_end > 0:
        result[..., 0] = prevprev
    if n_end > 1:
        result[..., 1] = prev
    for n in range(2, n_end):
        prevprev, prev = prev, (((2 * n - 1) * x * prev - (n - 1) * prevprev) / n)
        result[..., n] = prev
    return result


legendre_jit = jit(
    {"numpy": numba.jit(nogil=True, parallel=True)},
    decorator_kwargs={"jax": {"static_argnames": ["n_end"]}},
)(legendre)
legendre_assign_jit = jit({"numpy": numba.jit(nogil=True, parallel=True)})(legendre_assign)


def test_jit(xp: Any) -> None:
    t = {}
    for name, func in [("nojit", legendre), ("jit", legendre_jit)] + (
        [
            ("assign-nojit", legendre_assign),
            ("assign-jit", legendre_assign_jit),
        ]
        if "jax" not in xp.__name__
        else []
    ):
        print(xp.__name__, name)
        for i in range(20):
            with timer() as timer_:
                x = xp.arange(1000000, dtype=xp.float32)
                p = func(x, 10)  # type: ignore
                assert p.shape == (1000000, 10)
            t[name] = timer_.elapsed
            if i == 0:
                print(f"First call: {timer_.elapsed:g}s")
        print(f"Last call: {timer_.elapsed:g}s")
    if "numpy" not in xp.__name__:
        assert t["jit"] < t["nojit"], (
            f"JIT time {t['jit']} should be less than non-JIT time {t['nojit']}"
        )
        assert t["assign-jit"] < t["assign-nojit"], (
            f"JIT assign time {t['assign-jit']} should be "
            f"less than non-JIT assign time {t['assign-nojit']}"
        )
