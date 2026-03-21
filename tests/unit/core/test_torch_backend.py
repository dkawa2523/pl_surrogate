from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from plasma_surrogate.core import torch_backend


def test_torch_runtime_available_false_when_disabled(monkeypatch):
    monkeypatch.setenv("PLASMA_SURROGATE_ENABLE_TORCH", "0")
    assert torch_backend.torch_runtime_available(refresh=True) is False


def test_torch_runtime_available_true_when_subprocess_succeeds(monkeypatch):
    monkeypatch.setenv("PLASMA_SURROGATE_ENABLE_TORCH", "1")
    with patch("plasma_surrogate.core.torch_backend.subprocess.run", return_value=SimpleNamespace(returncode=0)):
        assert torch_backend.torch_runtime_available(refresh=True) is True


def test_torch_runtime_available_false_when_subprocess_fails(monkeypatch):
    monkeypatch.setenv("PLASMA_SURROGATE_ENABLE_TORCH", "1")
    with patch("plasma_surrogate.core.torch_backend.subprocess.run", return_value=SimpleNamespace(returncode=1)):
        assert torch_backend.torch_runtime_available(refresh=True) is False


def test_torch_runtime_available_refresh_updates_cache(monkeypatch):
    monkeypatch.setenv("PLASMA_SURROGATE_ENABLE_TORCH", "1")
    torch_backend._TORCH_RUNTIME_CACHE.clear()
    with patch("plasma_surrogate.core.torch_backend.subprocess.run", return_value=SimpleNamespace(returncode=1)):
        assert torch_backend.torch_runtime_available(refresh=True) is False
    with patch("plasma_surrogate.core.torch_backend.subprocess.run", return_value=SimpleNamespace(returncode=0)):
        assert torch_backend.torch_runtime_available(refresh=True) is True
    with patch("plasma_surrogate.core.torch_backend.subprocess.run", side_effect=AssertionError("cache miss")):
        assert torch_backend.torch_runtime_available() is True


def test_require_torch_raises_when_runtime_unavailable(monkeypatch):
    monkeypatch.setenv("PLASMA_SURROGATE_ENABLE_TORCH", "1")
    with patch("plasma_surrogate.core.torch_backend.torch_runtime_available", return_value=False):
        with pytest.raises(RuntimeError, match="Torch runtime is unavailable"):
            torch_backend.require_torch()
