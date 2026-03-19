"""Optional CuPy compatibility helpers.

This module allows importing the package on CPU-only environments.
GPU-specific paths should call ``require_cupy`` before using CuPy features.
"""

from __future__ import annotations

import numpy as np
from typing import Any

try:
    import cupy as _cupy  # type: ignore
    import cupyx.scipy.signal as csp  # type: ignore
    CUPY_AVAILABLE = True
    _CUPY_NDARRAY_TYPE = _cupy.ndarray
except Exception:  # pragma: no cover - depends on environment
    CUPY_AVAILABLE = False
    csp = None
    _CUPY_NDARRAY_TYPE = tuple()


def require_cupy(feature: str = "GPU functionality") -> None:
    if not CUPY_AVAILABLE:
        raise ImportError(
            f"{feature} requires CuPy, but CuPy is not installed. "
            "Install an appropriate extra, for example: pip install -e '.[gpu-cuda12]'"
        )


def is_cupy_array(value) -> bool:
    return bool(CUPY_AVAILABLE and isinstance(value, _CUPY_NDARRAY_TYPE))


def asnumpy(value: Any):
    if CUPY_AVAILABLE and is_cupy_array(value):
        import cupy as cp
        return cp.asnumpy(value)
    return value


def free_cupy_memory() -> None:
    if CUPY_AVAILABLE:
        import cupy as cp
        cp._default_memory_pool.free_all_blocks()
