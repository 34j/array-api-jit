# array API JIT

<p align="center">
  <a href="https://github.com/34j/array-api-jit/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/array-api-jit/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://array-api-jit.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/array-api-jit.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/array-api-jit">
    <img src="https://img.shields.io/codecov/c/github/34j/array-api-jit.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/array-api-jit/">
    <img src="https://img.shields.io/pypi/v/array-api-jit.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/array-api-jit.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/array-api-jit.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://array-api-jit.readthedocs.io" target="_blank">https://array-api-jit.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/array-api-jit" target="_blank">https://github.com/34j/array-api-jit </a>

---

JIT decorator supporting multiple array API compatible libraries

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install array-api-jit
```

## Usage

Simply decorate your function with `@jit()`:

```python
from array_api_jit import jit


@jit()
def my_function(x: Any) -> Any:
    xp = array_namespace(x)
    return xp.sin(x) + xp.cos(x)
```

## Advanced Usage

You can specify the decorator, arguments, and keyword arguments for each library.

```python
from array_api_jit import jit
from array_api_compat import array_namespace
from typing import Any
import numba


@jit(
    {"numpy": numba.jit()},  # numba.jit is not used by default because it may not succeed
    decorator_kwargs={
        "jax": {"static_argnames": ["n"]}
    },  # jax requires for-loop variable to be "static_argnames"
    # fail_on_error: bool = False, # do not raise an error if the decorator fails (Default)
    # rerun_on_error: bool = True, # re-run the original function if the wrapped function fails (NOT Default)
)
def sin_n_times(x: Any, n: int) -> Any:
    xp = array_namespace(x)
    for i in range(n):
        x = xp.sin(x)
    return x
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
