"""Checkpoint save/load orchestration.

Family-specific checkpoint details live in ``checkpoint_*`` modules.  This
module keeps the public checkpoint API small and predictable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.models.checkpoint_deeponet_plasma import (
    DEEPONET_PLASMA_CHECKPOINT_MODEL_TYPES,
    load_deeponet_plasma_checkpoint_model,
    load_deeponet_plasma_checkpoint_weights,
    make_deeponet_plasma_checkpoint_meta,
)
from plasma_surrogate.models.checkpoint_global import (
    GLOBAL_MLP_CHECKPOINT_MODEL_TYPES,
    load_global_mlp_checkpoint_model,
    load_global_mlp_checkpoint_weights,
    make_global_mlp_checkpoint_meta,
)
from plasma_surrogate.models.checkpoint_grid import (
    GRID_CHECKPOINT_MODEL_TYPES,
    load_grid_checkpoint_model,
    make_grid_checkpoint_meta,
)
from plasma_surrogate.models.checkpoint_pod import (
    POD_DEEPONET_CHECKPOINT_MODEL_TYPES,
    load_pod_deeponet_checkpoint_model,
    make_pod_deeponet_checkpoint_meta,
)


def _validate_checkpoint_extra_meta(extra_meta: dict[str, Any]) -> None:
    from plasma_surrogate.core.input_modes import input_mode_metadata_keys

    required = tuple(input_mode_metadata_keys())
    if not any(key in extra_meta for key in required):
        return
    missing = [key for key in required if key not in extra_meta]
    if missing:
        raise ValueError(f"checkpoint extra_meta missing required input-mode keys: {missing}")
    invalid: list[str] = []
    for key in required:
        value = extra_meta.get(key)
        if value is None:
            invalid.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            invalid.append(key)
    if invalid:
        raise ValueError(f"checkpoint extra_meta has empty required input-mode keys: {invalid}")


def _make_checkpoint_meta(model: Any) -> dict[str, Any]:
    for make_meta in (
        make_global_mlp_checkpoint_meta,
        make_grid_checkpoint_meta,
        make_pod_deeponet_checkpoint_meta,
        make_deeponet_plasma_checkpoint_meta,
    ):
        meta = make_meta(model)
        if meta is not None:
            return dict(meta)
    raise TypeError(f"Unsupported model type for checkpoint: {type(model)}")


def _load_checkpoint_model(meta: dict[str, Any], weights: Any) -> Any:
    for load_model in (
        load_global_mlp_checkpoint_model,
        lambda meta_: load_grid_checkpoint_model(meta_, weights),
        lambda meta_: load_pod_deeponet_checkpoint_model(meta_, weights),
        load_deeponet_plasma_checkpoint_model,
    ):
        model = load_model(meta)
        if model is not None:
            return model
    raise ValueError(f"Unknown model type in checkpoint: {str(meta.get('model_type', '')).strip().lower()}")


def _load_checkpoint_weights(model: Any, model_type: str, weights: Any) -> None:
    if model_type in DEEPONET_PLASMA_CHECKPOINT_MODEL_TYPES:
        load_deeponet_plasma_checkpoint_weights(model, weights)
        return

    if model_type in GLOBAL_MLP_CHECKPOINT_MODEL_TYPES:
        load_global_mlp_checkpoint_weights(model, weights)
        return

    if model_type in GRID_CHECKPOINT_MODEL_TYPES | POD_DEEPONET_CHECKPOINT_MODEL_TYPES:
        if not hasattr(model, "load_state_dict_numpy"):
            raise TypeError(f"{model_type} checkpoint model does not support load_state_dict_numpy")
        state = {k: np.asarray(weights[k], dtype=np.float32) for k in weights.files}
        model.load_state_dict_numpy(state)
        return

    model.W = np.asarray(weights["W"], dtype=np.float32)
    model.b = np.asarray(weights["b"], dtype=np.float32)


def save_checkpoint(
    model: Any,
    ckpt_dir: str | Path,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    meta = _make_checkpoint_meta(model)
    if extra_meta:
        _validate_checkpoint_extra_meta(dict(extra_meta))
        overlap = sorted(set(meta.keys()) & set(extra_meta.keys()))
        if overlap:
            raise ValueError(f"checkpoint extra_meta overlaps with model meta keys: {overlap}")
        meta = {**meta, **dict(extra_meta)}

    if hasattr(model, "state_dict_numpy"):
        np.savez_compressed(ckpt_path / "weights.npz", **model.state_dict_numpy())
    else:
        np.savez_compressed(ckpt_path / "weights.npz", W=model.W, b=model.b)

    with (ckpt_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return ckpt_path


def load_checkpoint(ckpt_dir: str | Path) -> Any:
    ckpt_path = Path(ckpt_dir)
    with (ckpt_path / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    model_type = str(meta.get("model_type", "")).strip().lower()
    with np.load(ckpt_path / "weights.npz") as weights:
        model = _load_checkpoint_model(meta, weights)
        _load_checkpoint_weights(model, model_type, weights)
    return model
