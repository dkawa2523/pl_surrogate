"""Inference output writing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore


def diagnostics_maps_enabled(ood_cfg: dict[str, Any]) -> bool:
    diagnostics_cfg = dict(dict(ood_cfg or {}).get("diagnostics", {}) or {})
    maps_cfg = dict(diagnostics_cfg.get("maps", {}) or {})
    return bool(maps_cfg.get("enabled", False))


def write_single_run_outputs(
    *,
    store: ArtifactStore,
    case_key: str,
    fields_model: dict[str, np.ndarray],
    fields_phys: dict[str, np.ndarray],
    derived: dict[str, np.ndarray],
    qoi: dict[str, Any],
    diagnostics: dict[str, Any],
    diagnostics_maps: dict[str, np.ndarray],
    save_diagnostics_maps: bool,
) -> None:
    base = f"single/{case_key}"
    store.save_npz(f"{base}/fields_model.npz", **fields_model)
    store.save_npz(f"{base}/fields_phys.npz", **fields_phys)
    store.save_npz(f"{base}/derived.npz", **derived)
    store.save_json(f"{base}/qoi.json", qoi)
    store.save_json(f"{base}/diagnostics.json", diagnostics)
    if save_diagnostics_maps and diagnostics_maps:
        store.save_npz(f"{base}/diagnostics_maps.npz", **diagnostics_maps)


__all__ = ["diagnostics_maps_enabled", "write_single_run_outputs"]
