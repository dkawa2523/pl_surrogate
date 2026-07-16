from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments" / "icp_stage4" / "scripts" / "plot_icp_stage4_geometry_overview.py"
SPEC = importlib.util.spec_from_file_location("plot_icp_stage4_geometry_overview", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT
SPEC.loader.exec_module(PLOT)


def test_expected_masks_match_fixed_comsol_regions() -> None:
    spec = PLOT.GeometrySpec()
    r = np.asarray([5.0, 16.0, 21.0, 29.0])
    z = np.asarray([1.0, 3.0, 13.0, 20.0])

    plasma, valid, outside, substrate = PLOT._expected_masks(r, z, spec)

    assert substrate[0].tolist() == [True, False, False, False]
    assert outside[0].tolist() == [True, True, False, False]
    assert plasma[0].tolist() == [False, False, True, True]
    assert plasma[1].tolist() == [True, True, True, True]
    assert not np.any(plasma[2:])
    assert np.array_equal(valid, ~outside)


def test_active_coil_reader_ignores_inactive_slots(tmp_path: Path) -> None:
    path = tmp_path / "layout.csv"
    fieldnames = [
        "case_id",
        "coil_index",
        "active",
        "order",
        "r_min",
        "r_max",
        "z_min",
        "z_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case_a",
                "coil_index": 2,
                "active": 1,
                "order": 2,
                "r_min": 8,
                "r_max": 9,
                "z_min": 15,
                "z_max": 16,
            }
        )
        writer.writerow(
            {
                "case_id": "case_a",
                "coil_index": 1,
                "active": 1,
                "order": 1,
                "r_min": 4,
                "r_max": 5,
                "z_min": 15,
                "z_max": 16,
            }
        )
        writer.writerow({"case_id": "case_a", "coil_index": 3, "active": 0, "order": ""})

    coils = PLOT._read_active_coils(path, "case_a")

    assert [coil.index for coil in coils] == [1, 2]
    assert coils[0].width == pytest.approx(1.0)


def test_geometry_validation_rejects_coil_mask_drift(tmp_path: Path) -> None:
    spec = PLOT.GeometrySpec()
    r = np.arange(0.5, 30.0, 1.0)
    z = np.arange(0.5, 22.0, 1.0)
    plasma, valid, outside, _ = PLOT._expected_masks(r, z, spec)
    coil = PLOT.CoilRectangle(index=1, order=1, r_min=4.0, r_max=5.0, z_min=15.0, z_max=16.0)
    case = PLOT.CaseGeometry(
        case_id="case_a",
        r_coords=r,
        z_coords=z,
        plasma_mask=plasma,
        coil_mask=np.zeros_like(plasma),
        valid_field_mask=valid,
        outside_mask=outside,
        coils=(coil,),
        structure_path=tmp_path / "case_a.npz",
        layout_path=tmp_path / "layout.csv",
        labels_path=tmp_path / "labels.csv",
        summary_path=tmp_path / "summary.json",
    )

    with pytest.raises(ValueError, match="mask_coil differs"):
        PLOT._validate_case_geometry(case, spec)


def test_metadata_records_geometry_provenance(tmp_path: Path) -> None:
    spec = PLOT.GeometrySpec()
    case = PLOT.CaseGeometry(
        case_id="case_a",
        r_coords=np.asarray([0.5, 1.5]),
        z_coords=np.asarray([0.5, 1.5]),
        plasma_mask=np.ones((2, 2)),
        coil_mask=np.zeros((2, 2)),
        valid_field_mask=np.ones((2, 2)),
        outside_mask=np.zeros((2, 2)),
        coils=(PLOT.CoilRectangle(1, 1, 4.0, 5.0, 15.0, 16.0),),
        structure_path=Path("structure.npz"),
        layout_path=Path("layout.csv"),
        labels_path=Path("labels.csv"),
        summary_path=Path("summary.json"),
    )
    output = tmp_path / "overview.png"

    path = PLOT._write_metadata(
        out_dir=tmp_path,
        prefix="overview",
        case=case,
        spec=spec,
        outputs=(output,),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["case_id"] == "case_a"
    assert payload["length_unit"] == "cm"
    assert payload["active_coil_count"] == 1
    assert payload["sources"]["comsol_model_definition"] == PLOT.MODEL_DEFINITION_URL
    assert "integration segment" in payload["omitted_nonphysical_geometry"].lower()


def test_outline_variant_has_only_unfilled_component_outlines(tmp_path: Path) -> None:
    spec = PLOT.GeometrySpec()
    case = PLOT.CaseGeometry(
        case_id="case_a",
        r_coords=np.asarray([0.5, 1.5]),
        z_coords=np.asarray([0.5, 1.5]),
        plasma_mask=np.ones((2, 2)),
        coil_mask=np.zeros((2, 2)),
        valid_field_mask=np.ones((2, 2)),
        outside_mask=np.zeros((2, 2)),
        coils=(PLOT.CoilRectangle(1, 1, 4.0, 5.0, 15.0, 16.0),),
        structure_path=tmp_path / "structure.npz",
        layout_path=tmp_path / "layout.csv",
        labels_path=tmp_path / "labels.csv",
        summary_path=tmp_path / "summary.json",
    )

    figure = PLOT.create_geometry_figure(case, spec, variant="outline")
    try:
        axis = figure.axes[0]
        assert axis.get_legend() is None
        assert all(patch.get_facecolor()[3] == pytest.approx(0.0) for patch in axis.patches)
        visible_edges = {
            tuple(np.round(patch.get_edgecolor(), 6))
            for patch in axis.patches
            if patch.get_edgecolor()[3] > 0.0
        }
        assert len(visible_edges) == 1
        assert all(patch.get_linestyle() == "solid" for patch in axis.patches)
        assert axis.lines[-1].get_linestyle() != "-"
    finally:
        PLOT.plt.close(figure)


def test_outline_prefix_replaces_overview_suffix() -> None:
    assert PLOT._variant_prefix("gec_icp_geometry_overview", "color") == "gec_icp_geometry_overview"
    assert PLOT._variant_prefix("gec_icp_geometry_overview", "outline") == "gec_icp_geometry_outline"
