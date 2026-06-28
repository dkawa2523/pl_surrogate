from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate import models
from plasma_surrogate.models import checkpoint
from plasma_surrogate.models.checkpoint_global import load_global_mlp_checkpoint_weights
from plasma_surrogate.models.mlp.global_mlp import GlobalMLP


def test_checkpoint_facade_exposes_generic_public_names() -> None:
    assert callable(checkpoint.load_checkpoint)
    assert callable(checkpoint.save_checkpoint)
    assert callable(checkpoint.build_model_from_name)
    assert models.load_checkpoint is checkpoint.load_checkpoint
    assert models.save_checkpoint is checkpoint.save_checkpoint


def test_global_mlp_checkpoint_rejects_legacy_w_b_weights() -> None:
    model = GlobalMLP(input_dim=2, grid_shape=(4, 4), out_channels=1, hidden=[4], dropout=0.0)
    with pytest.raises(ValueError, match="legacy global_mlp checkpoint format is not supported"):
        load_global_mlp_checkpoint_weights(
            model,
            {"W": np.zeros((2, 16), dtype=np.float32), "b": np.zeros((16,), dtype=np.float32)},
        )
