#!/usr/bin/env python
"""Fast tensor/gradient/checkpoint contract test for every isolated v46 UNO."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plasma_surrogate.cli.main import main as _cli_import_guard  # noqa: E402,F401
from run_study import VARIANTS, _build_fresh_model, _load_source_config  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(1237)
    source = _load_source_config()
    for variant, spec in VARIANTS.items():
        model = _build_fresh_model(variant, spec, source)
        model.grid_shape = (32, 48)
        model.coord = np.stack(
            np.meshgrid(
                np.linspace(0, 1, 48, dtype=np.float32),
                np.linspace(0, 1, 32, dtype=np.float32),
                indexing="xy",
            ),
            axis=-1,
        )
        cond = rng.normal(size=(3, 2)).astype(np.float32)
        spatial = rng.normal(size=(3, 32, 48, len(spec["features"]))).astype(np.float32)
        spatial[..., spec["features"].index("mask_plasma")] = 1.0
        if "sdf_coil_01" in spec["features"]:
            for name in (name for name in spec["features"] if name.startswith("sdf_coil_")):
                channel = spec["features"].index(name)
                spatial[:, 12:20, 16:26, channel] = -1.0
        prediction = model.forward_raw(cond, training=True, spatial_features=spatial)
        if prediction.shape != (3, 4, 32, 48) or not np.all(np.isfinite(prediction)):
            raise AssertionError(f"{variant} invalid prediction {prediction.shape}")
        diagnostics = model.backward_raw(
            np.full_like(prediction, 1.0 / prediction.size, dtype=np.float32),
            lr=1.0e-5,
            apply_step=True,
            target_raw=rng.normal(size=prediction.shape).astype(np.float32),
            supervised_mask=np.ones((3, 32, 48), dtype=np.float32),
        )
        state = model.state_dict_numpy()
        clone = _build_fresh_model(variant, spec, source)
        clone.load_state_dict_numpy(state)
        if not np.isfinite(float(diagnostics.get("loss_aux_total", 0.0))):
            raise AssertionError(f"{variant} invalid auxiliary loss")
        print(
            f"PASS {variant}: params={sum(value.size for value in state.values()):,} "
            f"aux={diagnostics.get('loss_aux_total', 0.0):.6g}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
