# https://github.com/search?q=gumerov+translation+language%3APython&type=code&l=Python
import importlib.util
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import cache, wraps
from types import ModuleType
from typing import Any, ParamSpec, TypeVar

from array_api_compat import (
    array_namespace,
    is_cupy_namespace,
    is_dask_namespace,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)
from frozendict import frozendict

if importlib.util.find_spec("numba"):
    import numpy as np
    from numba.extending import overload

    @overload(array_namespace)
    def _array_namespace_overload(*args: Any) -> Any:
        def inner(*args: Any) -> Any:
            return np

        return inner


P = ParamSpec("P")
T = TypeVar("T")
Pin = ParamSpec("Pin")
Tin = TypeVar("Tin")
Pinner = ParamSpec("Pinner")
Tinner = TypeVar("Tinner")
STR_TO_IS_NAMESPACE = {
    "numpy": is_numpy_namespace,
    "jax": is_jax_namespace,
    "cupy": is_cupy_namespace,
    "torch": is_torch_namespace,
    "dask": is_dask_namespace,
}


def _default_decorator(
    module: ModuleType,
    /,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    if is_jax_namespace(module):
        import jax

        return jax.jit
    elif is_numpy_namespace(module) or is_cupy_namespace(module):
        # import numba

        # return numba.jit()
        # The success rate of numba.jit is low
        return lambda x: x
    elif is_torch_namespace(module):
        import torch

        return torch.compile
    elif is_dask_namespace(module):
        return lambda x: x
    else:
        return getattr(module, "jit", lambda x: x)


Decorator = Callable[[Callable[Pin, Tin]], Callable[Pin, Tin]]


def jit(
    decorator: Mapping[str, Decorator[..., Any]] | None = None,
    /,
    *,
    fail_on_error: bool = False,
    rerun_on_error: bool = False,
    decorator_args: Mapping[str, Sequence[Any]] | None = None,
    decorator_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Just-in-time compilation decorator with multiple backends.

    Parameters
    ----------
    decorator : Mapping[str, Callable[[Callable[P, T]], Callable[P, T]]] | None, optional
        The JIT decorator to use for each array namespace, by default None
    fail_on_error : bool, optional
        If True, raise an error if the JIT decorator fails to apply.
        If False, just warn and return the original function, by default False
    rerun_on_error : bool, optional
        If True, rerun the function without JIT if the function
        with JIT applied fails, by default False
    decorator_args : Mapping[str, Sequence[Any]] | None, optional
        Additional positional arguments to be passed along with the function
        to the decorator for each array namespace, by default None
    decorator_kwargs : Mapping[str, Mapping[str, Any]] | None, optional
        Additional keyword arguments to be passed along with the function
        to the decorator for each array namespace, by default None

    Returns
    -------
    Callable[[Callable[P, T]], Callable[P, T]]
        The JIT decorator that can be applied to a function.

    Example
    -------
    >>> from array_api_jit import jit
    >>> from array_api_compat import array_namespace
    >>> from typing import Any
    >>> import numba
    >>> @jit(
    ...     {"numpy": numba.jit()},  # numba.jit is not used by default
    ...     decorator_kwargs={"jax": {"static_argnames": ["n"]}},  # jax requires static_argnames
    ... )
    ... def sin_n_times(x: Any, n: int) -> Any:
    ...     xp = array_namespace(x)
    ...     for i in range(n):
    ...         x = xp.sin(x)
    ...     return x

    """

    def new_decorator(f: Callable[Pinner, Tinner]) -> Callable[Pinner, Tinner]:
        decorator_args_ = frozendict(decorator_args or {})
        decorator_kwargs_ = frozendict(decorator_kwargs or {})
        decorator_ = decorator or {}

        @cache
        def jit_cached(xp: ModuleType) -> Callable[Pinner, Tinner]:
            for name_, is_namespace in STR_TO_IS_NAMESPACE.items():
                if is_namespace(xp):
                    name = name_
            else:
                name = xp.__name__.split(".")[0]
            decorator_args__ = decorator_args_.get(name, ())
            decorator_kwargs__ = decorator_kwargs_.get(name, {})
            if name in decorator_:
                decorator_current = decorator_[name]
            else:
                decorator_current = _default_decorator(xp)
            try:
                return decorator_current(f, *decorator_args__, **decorator_kwargs__)
            except Exception as e:
                if fail_on_error:
                    raise RuntimeError(f"Failed to apply JIT decorator for {name}") from e
                warnings.warn(
                    f"Failed to apply JIT decorator for {name}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return f

        @wraps(f)
        def inner(*args_inner: Pinner.args, **kwargs_inner: Pinner.kwargs) -> Tinner:
            try:
                xp = array_namespace(*args_inner)
            except TypeError as e:
                if e.args[0] == "Unrecognized array input":
                    return f(*args_inner, **kwargs_inner)
                raise
            f_jit = jit_cached(xp)
            try:
                return f_jit(*args_inner, **kwargs_inner)
            except Exception as e:
                if rerun_on_error:
                    warnings.warn(
                        f"JIT failed for {xp.__name__}: {e}. Rerunning without JIT.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return f(*args_inner, **kwargs_inner)
                raise RuntimeError(f"Failed to run JIT function for {xp.__name__}") from e

        return inner

    return new_decorator
