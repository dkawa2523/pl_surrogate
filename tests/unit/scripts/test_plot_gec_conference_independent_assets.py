from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments" / "conference" / "scripts" / "plot_gec_conference_independent_assets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("plot_gec_conference_independent_assets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ASSETS = _load_script()


def _close(figure) -> None:
    ASSETS.plt.close(figure)


def test_registry_contains_only_standalone_materials() -> None:
    assert len(ASSETS.FIGURES) == 29
    stems = [stem for stem, _builder in ASSETS.FIGURES.values()]
    assert len(stems) == len(set(stems))
    assert set(ASSETS.FIGURES) == {
        "ccp_geometry_diagram",
        "ccp_mesh",
        "icp_geometry_diagram",
        "icp_mesh",
        "ccp_ne_2d",
        "ccp_ne_3d",
        "ccp_ni_2d",
        "ccp_ni_3d",
        "ccp_te_2d",
        "ccp_te_3d",
        "ccp_phi_2d",
        "ccp_phi_3d",
        "icp_ne_2d",
        "icp_ne_3d",
        "icp_ni_2d",
        "icp_ni_3d",
        "icp_te_2d",
        "icp_te_3d",
        "icp_phi_2d",
        "icp_phi_3d",
        "ccp_response",
        "icp_geometry",
        "icp_operating",
        "icp_response",
        "ccp_zscore_before",
        "ccp_zscore_after",
        "icp_dimension",
        "icp_single_sdf",
        "icp_union_sdf",
    }


def test_geometry_mesh_and_density_materials_match_source_artifacts() -> None:
    figures = []
    try:
        figure, ccp_geometry = ASSETS.create_ccp_geometry_asset()
        figures.append(figure)
        figure, ccp_mesh = ASSETS.create_ccp_mesh_asset()
        figures.append(figure)
        figure, icp_geometry = ASSETS.create_icp_geometry_asset()
        figures.append(figure)
        figure, icp_mesh = ASSETS.create_icp_mesh_asset()
        figures.append(figure)
        figure, ccp_2d = ASSETS.create_ccp_electron_density_2d()
        figures.append(figure)
        figure, ccp_3d = ASSETS.create_ccp_electron_density_perspective()
        figures.append(figure)
        figure, icp_2d = ASSETS.create_icp_electron_density_2d()
        figures.append(figure)
        figure, icp_3d = ASSETS.create_icp_electron_density_perspective()
        figures.append(figure)

        assert ccp_geometry["geometry_alternative"] == "base4"
        assert ccp_mesh["mesh_kind"] == "COMSOL quadrilateral mesh"
        assert ccp_mesh["mesh_vertices"] == 1520
        assert ccp_mesh["mesh_elements"] == ccp_mesh["displayed_elements"] == 1425
        assert icp_geometry["active_coils"] == 6
        assert icp_mesh["grid_shape_zr"] == [440, 600]
        assert icp_mesh["display_line_stride"] == 10
        assert ccp_2d["case_id"] == ccp_3d["case_id"] == ASSETS.simple.detail.CCP_CASE_ID
        assert icp_2d["case_id"] == icp_3d["case_id"] == ASSETS.simple.detail.ICP_CASE_ID
        assert ccp_3d["view"] == icp_3d["view"] == "flat 2D map tilted in depth"
        assert ccp_3d["height_encoding"] == icp_3d["height_encoding"] == "none; the plate is flat"
        assert ccp_3d["color_encoding"] == icp_3d["color_encoding"] == "logarithmic ne [m^-3]"
        assert ccp_2d["display_limits_m3"][0] < ccp_2d["display_limits_m3"][1]
        assert icp_2d["display_limits_m3"][0] < icp_2d["display_limits_m3"][1]
    finally:
        for figure in figures:
            _close(figure)


@pytest.mark.parametrize("system", ["ccp", "icp"])
@pytest.mark.parametrize(
    ("field_name", "unit", "scale"),
    [("ne", "m^-3", "logarithmic"), ("ni", "m^-3", "logarithmic"), ("Te", "eV", "linear"), ("phi", "V", "linear")],
)
def test_physical_field_views_share_color_encoding(system: str, field_name: str, unit: str, scale: str) -> None:
    figures = []
    try:
        figure, map_2d = ASSETS._physical_field_2d(system, field_name)
        figures.append(figure)
        figure, tilted = ASSETS._physical_field_perspective(system, field_name)
        figures.append(figure)

        assert map_2d["shown_field"] == tilted["shown_field"] == field_name
        assert map_2d["unit"] == tilted["unit"] == unit
        assert map_2d["display_limits"] == tilted["display_limits"]
        assert map_2d["color_encoding"] == tilted["color_encoding"]
        assert map_2d["normalization"] == f"{scale} color scale"
        assert tilted["height_encoding"] == "none; the plate is flat"
        assert tilted["view"] == "flat 2D map tilted in depth"
    finally:
        for figure in figures:
            _close(figure)


def test_potential_colormap_matches_sign_structure() -> None:
    ccp = ASSETS._field_display("ccp", "phi")
    icp = ASSETS._field_display("icp", "phi")
    assert ccp["vmin"] < 0.0 < ccp["vmax"]
    assert ccp["cmap_name"] == "coolwarm"
    assert icp["vmin"] > 0.0
    assert icp["cmap_name"] == "cividis"


def test_ccp_quality_materials_match_current_dataset() -> None:
    figures = []
    try:
        figure, response = ASSETS.create_ccp_response_coverage()
        figures.append(figure)

        assert response["max_over_min"] == pytest.approx(8.738, rel=2.0e-3)
    finally:
        for figure in figures:
            _close(figure)


def test_icp_quality_materials_match_current_structure_holdout() -> None:
    figures = []
    try:
        figure, geometry = ASSETS.create_icp_geometry_coverage()
        figures.append(figure)
        figure, operating = ASSETS.create_icp_operating_coverage()
        figures.append(figure)
        figure, response = ASSETS.create_icp_response_coverage()
        figures.append(figure)

        assert geometry["parameters"]["nncoil"]["min"] == 2.0
        assert geometry["parameters"]["nncoil"]["max"] == 6.0
        assert geometry["parameters"]["nncoil"]["unique"] == 5
        assert operating["unique_pairs"] == 360
        assert operating["surface_case_count"] == 359
        assert operating["scatter_case_count"] == 360
        assert operating["excluded_from_surface_case_id"] == "case_g012_op01"
        assert operating["surface_extrapolation"].startswith("none")
        assert set(response["by_split"]) == {"train", "val", "test"}
        assert response["minimum"] == pytest.approx(2.79e8, rel=2.0e-3)
    finally:
        for figure in figures:
            _close(figure)


def test_split_zscore_and_sdf_materials_keep_source_contracts() -> None:
    figures = []
    try:
        figure, before = ASSETS.create_ccp_zscore_before()
        figures.append(figure)
        figure, after = ASSETS.create_ccp_zscore_after()
        figures.append(figure)
        figure, dimension = ASSETS.create_icp_dimension_vector()
        figures.append(figure)
        figure, single = ASSETS.create_icp_single_coil_sdf()
        figures.append(figure)
        figure, union = ASSETS.create_icp_union_sdf()
        figures.append(figure)

        assert before["stage"] == "before"
        assert after["stage"] == "after"
        assert before["samples"] == after["samples"] == 1_321_920
        assert after["z_mean"] == pytest.approx(0.0, abs=2.0e-6)
        assert after["z_std_ddof0"] == pytest.approx(1.0, abs=2.0e-6)
        assert dimension["dimension_vector"] == ["llcoil", "rrc", "nncoil", "rrce", "zzc"]
        assert single["kind"] == "single_coil"
        assert union["kind"] == "union"
        assert single["strict_sign_contradiction_cells"] == 0
        assert union["strict_sign_contradiction_cells"] == 0
    finally:
        for figure in figures:
            _close(figure)


def test_selected_generation_writes_editable_outputs_and_metadata(tmp_path: Path) -> None:
    outputs = ASSETS.generate(
        out_dir=tmp_path,
        figures=("ccp_response", "icp_response"),
        formats=("png", "pdf", "svg"),
        dpi=72,
    )
    assert len(outputs) == 8
    for stem in ("ccp_dataset_response_coverage", "icp_dataset_response_coverage"):
        svg = (tmp_path / f"{stem}.svg").read_text(encoding="utf-8")
        pdf = (tmp_path / f"{stem}.pdf").read_bytes()
        metadata = json.loads((tmp_path / f"{stem}_metadata.json").read_text(encoding="utf-8"))
        assert "<text" in svg
        assert "<image" not in svg
        assert b"/Subtype /Type3" not in pdf
        assert metadata["presentation_revision"] == "standalone_single_claim_v1"
        assert len(metadata["outputs"]) == 3
