from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest

from plasma_surrogate.core.torch_backend import torch_backend_enabled, torch_runtime_available

_SKIP_TORCH_BACKEND_DISABLED = "torch backend disabled for this environment"
_SKIP_TORCH_RUNTIME_UNAVAILABLE = "torch runtime unavailable in this environment"
_SKIP_OPTUNA_MISSING = "optuna missing"


def require_torch_backend(*, enable_backend: bool = False) -> None:
    if enable_backend:
        os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    if not torch_backend_enabled():
        pytest.skip(_SKIP_TORCH_BACKEND_DISABLED)


def require_torch_runtime(*, enable_backend: bool = False, refresh: bool = True) -> None:
    require_torch_backend(enable_backend=enable_backend)
    if not torch_runtime_available(refresh=refresh):
        pytest.skip(_SKIP_TORCH_RUNTIME_UNAVAILABLE)


def require_optuna() -> ModuleType:
    try:
        return importlib.import_module("optuna")
    except ImportError:
        pytest.skip(_SKIP_OPTUNA_MISSING)
