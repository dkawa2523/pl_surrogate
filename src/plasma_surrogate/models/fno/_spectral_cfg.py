"""Private helpers for shared spectral model configuration."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_common_spectral_cfg(
    spectral_cfg: dict[str, Any] | None,
    *,
    cfg_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize the common spectral config shared by FNO and FFNO."""

    cfg = dict(spectral_cfg or {})
    width = int(max(int(cfg.get("width", 64)), 8))
    n_layers = int(max(int(cfg.get("n_layers", 4)), 1))
    dropout = float(np.clip(float(cfg.get("dropout", 0.0)), 0.0, 0.9))
    dealias_ratio = float(np.clip(float(cfg.get("dealias_ratio", 1.0)), 0.05, 1.0))
    taper_alpha = float(max(float(cfg.get("taper_alpha", 0.0)), 0.0))
    skip_filter = str(cfg.get("skip_filter", "none")).strip().lower()
    if skip_filter not in {"none", "match_spectral"}:
        raise ValueError(f"{cfg_prefix}.model_cfg.spectral_cfg.skip_filter must be one of: none, match_spectral")

    cfg["width"] = int(width)
    cfg["n_layers"] = int(n_layers)
    cfg["dropout"] = float(dropout)
    cfg["dealias_ratio"] = float(dealias_ratio)
    cfg["taper_alpha"] = float(taper_alpha)
    cfg["skip_filter"] = str(skip_filter)

    normalized = {
        "width": int(width),
        "n_layers": int(n_layers),
        "dropout": float(dropout),
        "dealias_ratio": float(dealias_ratio),
        "taper_alpha": float(taper_alpha),
        "skip_filter": str(skip_filter),
    }
    return cfg, normalized
