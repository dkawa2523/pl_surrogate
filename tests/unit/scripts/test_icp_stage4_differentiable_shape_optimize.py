from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.torch_runtime
torch = pytest.importorskip("torch")


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "icp_stage4" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_icp_stage4_differentiable_shape_optimize import (  # noqa: E402
    Bounds,
    _plasma_midplane_core_band,
    _wafer_top_band,
    decode_design,
    differentiable_part_features,
    hard_part_features,
)


def test_decode_design_always_satisfies_gap_and_bounds() -> None:
    for count in range(2, 7):
        for raw in (-8.0, 0.0, 8.0):
            design = decode_design(torch.full((4,), raw), count=count, bounds=Bounds())
            ll = float(design["llcoil"])
            rrc = float(design["rrc"])
            rrce = float(design["rrce"])
            assert 0.5 <= ll <= 1.5
            assert 2.0 <= rrc <= 10.0
            assert 20.0 <= rrce <= 30.0
            assert (rrce - rrc) / count - ll >= 0.2 * ll - 1.0e-5


def test_straight_through_features_match_hard_manhattan_features() -> None:
    r = np.linspace(0.0, 30.0, 61, dtype=np.float32)
    z = np.linspace(0.0, 22.0, 45, dtype=np.float32)
    r_grid, z_grid = np.meshgrid(r, z, indexing="xy")
    logits = torch.nn.Parameter(torch.zeros(4))
    design = decode_design(logits, count=3, bounds=Bounds())
    soft = differentiable_part_features(
        r_grid=torch.from_numpy(r_grid),
        z_grid=torch.from_numpy(z_grid),
        dr=float(r[1] - r[0]),
        dz=float(z[1] - z[0]),
        design=design,
        count=3,
    )
    geom = {key: float(value.detach()) for key, value in design.items()}
    hard = hard_part_features(r_grid=r_grid, z_grid=z_grid, geom=geom, count=3)

    for key in hard:
        np.testing.assert_allclose(soft[key].detach().numpy(), hard[key], atol=1.0e-6, rtol=0.0)
    loss = soft["part_sdf_nearest"].mean() + soft["part_gap_proxy"].mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert float(torch.linalg.vector_norm(logits.grad)) > 0.0


def test_wafer_top_band_selects_requested_top_layers_per_column() -> None:
    wafer = np.zeros((6, 4), dtype=np.float32)
    wafer[:4, :] = 1.0
    plasma = np.ones_like(wafer)

    selected = _wafer_top_band(wafer, plasma, layers=2)

    expected = np.zeros_like(selected)
    expected[2:4, :] = True
    np.testing.assert_array_equal(selected, expected)


def test_plasma_midplane_core_band_uses_middle_rows_and_radial_cutoff() -> None:
    r = np.broadcast_to(np.arange(6, dtype=np.float32), (8, 6))
    plasma = np.zeros((8, 6), dtype=np.float32)
    plasma[1:7] = 1.0

    selected, meta = _plasma_midplane_core_band(plasma, r, band_px=1, radial_fraction=0.8)

    expected = np.zeros_like(plasma, dtype=bool)
    expected[2:5, :5] = True
    np.testing.assert_array_equal(selected, expected)
    assert meta["mid_row"] == 3.0
    assert meta["sample_count"] == 15.0
