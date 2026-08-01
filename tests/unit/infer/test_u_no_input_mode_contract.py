from __future__ import annotations

import pytest
from pathlib import Path


from plasma_surrogate.core.input_modes import build_input_mode_effective_metadata
from tests._runtime_requirements import require_torch_runtime
from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.uno.simple_uno import UNOBaseline
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema

pytestmark = pytest.mark.torch_runtime


def test_u_no_rejects_table_only_inference_mode(tmp_path: Path, geometry_root: Path) -> None:
    require_torch_runtime()
    model = UNOBaseline(
        input_dim=3,
        grid_shape=(8, 8),
        input_feature_channels=["x", "y", "mask_plasma", "distance_signed", "distance_any"],
        n_modes=3,
        uno_cfg={"width": 16, "n_layers": 2, "dropout": 0.0},
        backend="torch",
    )
    meta = build_input_mode_effective_metadata(
        {
            "runtime": {
                "input_mode": "table_only",
                "structure": {
                    "feature_profile": "none",
                    "descriptor_profile": "none",
                    "latent_profile": "none",
                    "adapter_mode": "none",
                    "provider_mode": "fixed",
                },
            }
        }
    )
    with pytest.raises(ValueError, match="table_only"):
        InferenceEngine(
            model=model,
            cond_schema=CondSchema(order=["c0", "c1", "c2"]),
            axis_schema=AxisSchema(mode="steady"),
            geometry_provider=FixedGeometryProvider(geometry_root),
            output_dir=tmp_path / "inference",
            input_mode="table_only",
            input_mode_meta=meta,
        )
