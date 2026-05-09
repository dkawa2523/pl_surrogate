from __future__ import annotations
from typing import Any
from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.core.input_modes import input_mode_metadata_keys


@pytest.fixture
def assert_input_mode_metadata_keys():
    """Return assertion helper that verifies canonical input-mode metadata keys."""

    def _assert(payload: dict[str, Any]) -> None:
        missing = [key for key in input_mode_metadata_keys() if key not in payload]
        assert not missing, f"missing input-mode metadata keys: {missing}"

    return _assert


@pytest.fixture
def geometry_root(tmp_path: Path) -> Path:
    """Create minimal geometry dataset root used by infer unit tests."""

    root = tmp_path / "dataset"
    geom = root / "geometry"
    geom.mkdir(parents=True, exist_ok=True)
    mask = np.ones((8, 8), dtype=np.float32)
    mask[0, :] = 0.0
    np.save(geom / "mask_plasma.npy", mask)
    np.save(geom / "eps.npy", np.ones_like(mask, dtype=np.float32))
    np.save(geom / "wafer_mask.npy", np.zeros_like(mask, dtype=np.float32))
    return root
