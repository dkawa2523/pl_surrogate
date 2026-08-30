"""Optional torch backend loader with explicit opt-in."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Any

_TORCH_RUNTIME_CACHE: dict[str, bool] = {}


def torch_backend_enabled() -> bool:
    flag = os.environ.get("PLASMA_SURROGATE_ENABLE_TORCH", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def torch_runtime_available(*, refresh: bool = False) -> bool:
    """
    Return True when torch import is likely safe in this process.

    We probe importability in a short-lived subprocess first, because some
    environments abort the whole process at import-time (OpenMP/SHM mismatch).
    """

    flag = os.environ.get("PLASMA_SURROGATE_ENABLE_TORCH", "0").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        _TORCH_RUNTIME_CACHE[flag] = False
        return False
    if (not refresh) and (flag in _TORCH_RUNTIME_CACHE):
        return bool(_TORCH_RUNTIME_CACHE[flag])
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import importlib; importlib.import_module('torch')"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        _TORCH_RUNTIME_CACHE[flag] = False
        return False
    ok = int(proc.returncode) == 0
    _TORCH_RUNTIME_CACHE[flag] = ok
    return ok


def require_torch() -> Any:
    if not torch_backend_enabled():
        raise RuntimeError(
            "Torch backend is disabled. Set PLASMA_SURROGATE_ENABLE_TORCH=1 "
            "to enable Torch-backed models."
        )
    if not torch_runtime_available():
        raise RuntimeError(
            "Torch runtime is unavailable in this environment. "
            "Keep PLASMA_SURROGATE_ENABLE_TORCH=0 or fix torch/OpenMP runtime first."
        )
    return importlib.import_module("torch")
