"""Utilities for deterministic fixed-length vector pack contracts."""

from __future__ import annotations

from typing import Any

import numpy as np


def load_vector_from_pack(
    *,
    pack: dict[str, Any] | None,
    pack_name: str,
) -> tuple[np.ndarray, list[str]]:
    """Load a vector pack with strict finite/dimension checks."""

    payload = dict(pack or {})
    if "vector" not in payload:
        raise ValueError(f"{pack_name} must include 'vector' array")
    vec = np.asarray(payload["vector"], dtype=np.float32).reshape(-1)
    if vec.size < 1:
        raise ValueError(f"{pack_name} vector must be non-empty")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{pack_name} vector must contain finite values")
    names: list[str] = []
    if "feature_names" in payload:
        names = [str(v) for v in np.asarray(payload["feature_names"]).reshape(-1).tolist()]
        if names and len(names) != int(vec.shape[0]):
            raise ValueError(
                f"{pack_name} feature_names length mismatch: "
                f"len(names)={len(names)}, dim={int(vec.shape[0])}"
            )
    return vec.astype(np.float32), names


__all__ = ["load_vector_from_pack"]
