from __future__ import annotations

from plasma_surrogate import models
from plasma_surrogate.models import checkpoint


def test_checkpoint_facade_exposes_generic_public_names() -> None:
    assert callable(checkpoint.load_checkpoint)
    assert callable(checkpoint.save_checkpoint)
    assert callable(checkpoint.build_model_from_name)
    assert models.load_checkpoint is checkpoint.load_checkpoint
    assert models.save_checkpoint is checkpoint.save_checkpoint
