from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments" / "gec_ccp" / "scripts" / "plot_gec_ccp_geometry_overview.py"
SPEC = importlib.util.spec_from_file_location("plot_gec_ccp_geometry_overview", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT
SPEC.loader.exec_module(PLOT)


def _base4_boundaries() -> tuple[object, ...]:
    values = {
        0: (0.0, 0.0, 0.0, 0.0254),
        1: (0.0, 0.0508, 0.0, 0.0),
        2: (0.0, 0.0508, 0.0254, 0.0254),
        3: (0.0508, 0.0538, 0.0, 0.0),
        4: (0.0508, 0.0538, 0.0254, 0.0254),
        5: (0.0538, 0.0538, -0.0381, 0.0),
        6: (0.0538, 0.1016, -0.0381, -0.0381),
        7: (0.0538, 0.0538, 0.0254, 0.0635),
        8: (0.0538, 0.1016, 0.0635, 0.0635),
        9: (0.1016, 0.1016, -0.0381, 0.0),
        10: (0.1016, 0.1016, 0.0, 0.0254),
        11: (0.1016, 0.1016, 0.0254, 0.0635),
    }
    return tuple(PLOT.BoundarySegment(entity_id, *coordinates) for entity_id, coordinates in values.items())


def _geometry(tmp_path: Path) -> object:
    boundaries = _base4_boundaries()
    dimensions, domain_vertices = PLOT._validate_boundary_topology(boundaries, base_name="base4")
    return PLOT.GecCcpGeometry(
        base_name="base4",
        td_value=0.03,
        boundaries=boundaries,
        dimensions=dimensions,
        domain_vertices_m=domain_vertices,
        mphtxt_path=tmp_path / "base4.mphtxt",
        structure_index_path=tmp_path / "structure_file_index.csv",
        manifest_path=tmp_path / "parts_manifest.json",
    )


def test_base4_topology_and_role_partition_are_exact() -> None:
    dimensions, vertices = PLOT._validate_boundary_topology(_base4_boundaries(), base_name="base4")

    assert dimensions.discharge_gap == pytest.approx(0.0254)
    assert dimensions.inner_radius == pytest.approx(0.0538)
    assert dimensions.outer_radius == pytest.approx(0.1016)
    assert dimensions.chamber_height == pytest.approx(0.1016)
    assert dimensions.powered_side_dielectric_break == pytest.approx(0.003)
    assert vertices[2] == pytest.approx((0.0538, -0.0381))
    role_ids = [entity_id for ids in PLOT.ROLE_ENTITY_IDS.values() for entity_id in ids]
    assert sorted(role_ids) == list(range(12))
    assert len(role_ids) == len(set(role_ids))


def test_outline_is_unfilled_and_uses_four_boundary_legend_items(tmp_path: Path) -> None:
    figure = PLOT.create_geometry_figure(_geometry(tmp_path), variant="outline")
    try:
        axis = figure.axes[0]
        assert axis.patches[0].get_facecolor()[3] == pytest.approx(0.0)
        legend = axis.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "Driven electrode",
            "Dielectric-contact boundary",
            "Grounded electrode / walls",
            "Axis of symmetry",
        ]
        assert len(axis.lines) == 12
    finally:
        PLOT.plt.close(figure)


def test_color_legend_identifies_plasma_and_physical_boundaries(tmp_path: Path) -> None:
    figure = PLOT.create_geometry_figure(_geometry(tmp_path), variant="color")
    try:
        legend = figure.axes[0].get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels[0] == "Plasma domain"
        assert "Dielectric-contact boundary" in labels
        assert len(labels) == 5
    finally:
        PLOT.plt.close(figure)


def test_metadata_records_boundary_only_dielectric_and_sources(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    output = tmp_path / "overview.svg"

    path = PLOT._write_metadata(
        out_dir=tmp_path,
        prefix="overview",
        geometry=geometry,
        variant="color",
        outputs=(output,),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["representative_alternative"] == "base4"
    assert payload["geometry_dimensions_cm"]["powered_side_dielectric_break"] == pytest.approx(0.3)
    assert payload["boundary_roles_zero_based_mphtxt_entity_id"]["dielectric_contact"] == [3, 5]
    assert "not a solid/material" in payload["dielectric_representation"]
    assert payload["sources"]["comsol_model_definition"] == PLOT.MODEL_DEFINITION_URL
    assert "not treated as dThick" in payload["dataset_condition_record"]["interpretation_note"]
