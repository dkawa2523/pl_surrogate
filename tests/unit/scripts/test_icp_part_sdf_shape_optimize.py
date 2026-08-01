from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests._runtime_requirements import require_torch_runtime


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "icp_stage4" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_icp_part_sdf_shape_optimize import (  # noqa: E402
    _canonical_series_signature,
    _coil_series_layout,
    _resolve_checkpoint_dir,
    _series_structure_descriptor,
    _sobol_unit,
    _write_structure_diversity_candidates,
)


def _geom() -> dict[str, float]:
    return {
        "series.nncoil": 3.0,
        "series.llcoil": 1.0,
        "series.rrc": 3.0,
        "series.rrce": 24.0,
        "series.zzc": 2.0,
    }


def test_structure_descriptor_is_fixed_scaled_geometry_only() -> None:
    descriptor, _ = _series_structure_descriptor(geom_param=_geom(), min_gap_frac=0.2, length_scale=1.5)

    assert descriptor.shape == (36,)
    assert np.all(np.isfinite(descriptor))
    assert np.all((descriptor >= 0.0) & (descriptor <= 1.0))


def test_canonical_signature_ignores_inactive_rows_and_row_order() -> None:
    rows, _ = _coil_series_layout(
        nncoil=3,
        llcoil=1.0,
        rrc=3.0,
        rrce=24.0,
        zzc=2.0,
        part_ids=[f"coil_{idx:02d}" for idx in range(1, 7)],
        min_gap_frac=0.2,
        chamber_bbox={"r_min": 0.0, "r_max": 30.0, "z_min": 0.0, "z_max": 22.0},
    )
    modified = [dict(row) for row in reversed(rows)]
    for row in modified:
        if int(row["active"]) == 0:
            row["r_center"] = 12345.0

    assert _canonical_series_signature(rows) == _canonical_series_signature(modified)


@pytest.mark.torch_runtime
def test_scrambled_sobol_is_seeded_and_bounded() -> None:
    require_torch_runtime(enable_backend=True)
    first = _sobol_unit(n=16, d=4, seed=17)
    second = _sobol_unit(n=16, d=4, seed=17)

    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0.0) & (first < 1.0))
    assert np.unique(first, axis=0).shape[0] == 16


@pytest.mark.torch_runtime
def test_structure_sampler_writes_unique_geometry_candidates(tmp_path: Path) -> None:
    require_torch_runtime(enable_backend=True)
    path = tmp_path / "candidates.csv"
    args = SimpleNamespace(
        space_mode="coil_series",
        seed=17,
        n_trials=4,
        feature_pool_size=16,
        feature_dedupe_bins=64,
        min_gap_frac=0.2,
        llcoil_range=(0.5, 1.5),
    )
    summary = _write_structure_diversity_candidates(
        path,
        space={"pp": (1000.0, 1000.0), "pp0": (0.02, 0.02)},
        series_space={
            "series.nncoil": (2.0, 2.0),
            "series.llcoil": (0.7, 1.1),
            "series.rrc": (2.0, 4.0),
            "series.rrce": (22.0, 26.0),
            "series.zzc": (0.5, 3.0),
        },
        args=args,
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert len({row["sampler_structure_signature"] for row in rows}) == 4
    assert summary["sampler"] == "scrambled_sobol"
    assert summary["descriptor_scaling"] == "fixed_physical"
    assert summary["condition_features_in_descriptor"] is False


def test_checkpoint_resolver_prefers_structure_holdout(tmp_path: Path) -> None:
    for protocol in ("extrap", "structure_holdout"):
        checkpoint = tmp_path / "models" / "u_no" / "eval_protocol" / protocol / "checkpoints"
        checkpoint.mkdir(parents=True)
        (checkpoint / "meta.json").write_text("{}", encoding="utf-8")

    checkpoint, protocol = _resolve_checkpoint_dir(tmp_path, "u_no", "auto")

    assert protocol == "structure_holdout"
    assert checkpoint.name == "checkpoints"
