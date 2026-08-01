from __future__ import annotations

import pytest

from plasma_surrogate.models.checkpoint_grid import load_grid_checkpoint_model


def test_grid_checkpoint_spec_rejects_legacy_backend_message() -> None:
    with pytest.raises(ValueError, match="legacy numpy FNO checkpoints are no longer supported"):
        load_grid_checkpoint_model({"model_type": "fno"})


def test_grid_checkpoint_spec_rejects_legacy_impl_message() -> None:
    with pytest.raises(ValueError, match="legacy FNO checkpoint format is not supported"):
        load_grid_checkpoint_model(
            {
                "model_type": "fno",
                "backend": "torch",
                "fno_impl_version": "spectral_v1",
            }
        )
