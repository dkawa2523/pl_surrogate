from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_script(name: str):
    path = ROOT / "experiments" / "conference" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATA = _load_script("plot_gec_data_and_features")
LOSS = _load_script("plot_gec_loss_design")
QUALITY = _load_script("plot_gec_dataset_quality")
SIMPLE = _load_script("plot_gec_conference_simple")
CALC = _load_script("plot_gec_conference_dataset_and_formula")


def test_masked_statistics_use_only_finite_selected_values() -> None:
    values = np.asarray([[1.0, np.nan, 99.0], [3.0, 5.0, np.inf]])
    mask = np.asarray([[True, True, False], [True, True, False]])

    stats = DATA._finite_masked_stats(values, mask)

    assert stats["count"] == 3
    assert stats["min"] == pytest.approx(1.0)
    assert stats["median"] == pytest.approx(3.0)
    assert stats["max"] == pytest.approx(5.0)


def test_ccp_zscore_matches_saved_train_plasma_scalers() -> None:
    figure, metadata = DATA.create_ccp_zscore_figure()
    try:
        assert metadata["train_case_count"] == 54
        assert metadata["plasma_sample_count_per_target"] == 1_321_920
        assert metadata["preprocessing_contract"]["value_transform"] == "identity for all four targets"
        assert metadata["preprocessing_contract"]["clip"] == "none"
        for target in DATA.TARGETS:
            verified = metadata["verification"][target]
            assert verified["z_mean"] == pytest.approx(0.0, abs=2.0e-6)
            assert verified["z_std_ddof0"] == pytest.approx(1.0, abs=2.0e-6)
        assert len(figure.axes) == 8
    finally:
        DATA.plt.close(figure)


def test_icp_feature_figure_records_dimension_and_sdf_contracts() -> None:
    figure, metadata = DATA.create_icp_geometry_features_figure()
    try:
        dimension = metadata["dimension_parameter_encoding"]
        spatial = metadata["spatial_structure_encoding"]
        validation = metadata["validation"]
        assert metadata["case_id"] == "case_g002_op01"
        assert dimension["geometry_vector_order"] == ["llcoil", "rrc", "nncoil", "rrce", "zzc"]
        assert dimension["pitch_value_cm"] == pytest.approx(4.3256666667)
        assert spatial["sign_convention"].startswith("negative inside")
        assert spatial["order_invariant_profile"]["source_tau_cm"] == pytest.approx(0.2, abs=1.0e-6)
        assert spatial["trained_profile"] == "part_source_v1"
        assert validation["active_coil_count"] == 6
        assert validation["union_sdf_strict_sign_contradiction_cells"] == 0
    finally:
        DATA.plt.close(figure)


def test_loss_metadata_is_the_trained_contract_not_the_untrained_alternative() -> None:
    ccp = LOSS.build_ccp_metadata()
    icp = LOSS.build_icp_metadata()

    assert ccp["per_target_loss"]["boundary_weight"] == pytest.approx(0.25)
    assert ccp["field_family_weighting"]["density"]["targets"] == {
        "ne": pytest.approx(1.0 / 6.0),
        "ni": pytest.approx(1.0 / 6.0),
    }
    assert icp["status"] == "trained_part_source_v1_baseline"
    assert icp["per_target_loss"]["gradient_weight"] == pytest.approx(0.02)
    assert icp["target_weights"] == {"ne": 1.0, "ni": 1.0, "Te": 1.2, "phi": 0.5}
    assert icp["alternative_contract_not_plotted"]["status"] == "configuration_only_not_yet_trained"


def test_quality_audits_expose_missing_coverage_without_hiding_integrity() -> None:
    ccp = QUALITY._audit_ccp(
        ROOT / "data" / "outputs_merged_td_all_success_pa_ext0520",
        ROOT / "data" / "outputs_merged_td_csv_periodic_ext0520_v2",
    )
    icp = QUALITY._audit_icp(
        ROOT / "data" / "outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2",
        ROOT / "reports" / "icp_stage4_major_fixes" / "dataset_audit_part_lite_v2.json",
        ROOT / "data" / "outputs_icp_stage4_enriched_360" / "learning_summary.json",
    )

    assert ccp["source_success_cases"] == 78
    assert ccp["finite_rate"] == pytest.approx(1.0)
    assert ccp["missing_case_count"] == 3
    assert ccp["coverage_by_td"]["0.16"][2][0] == 0
    assert icp["processed_cases"] == 360
    assert icp["structure_groups"] == 60
    assert icp["cross_split_leakage_count"] == 0
    assert icp["test_missing_ncoil"] == [2, 6]


def test_simple_quality_figures_are_scoped_to_representative_subsets() -> None:
    ccp_figure, ccp = SIMPLE.create_ccp_quality_simple()
    icp_figure, icp = SIMPLE.create_icp_quality_simple()
    try:
        assert len(ccp_figure.axes) == 3
        assert ccp["base_name"] == "base4"
        assert ccp["case_count"] == 27
        assert ccp["unique_operating_combinations"] == 27
        assert ccp["finite_rate"] == pytest.approx(1.0)
        assert "Broader 78-case" in ccp["scope_note"]

        assert len(icp_figure.axes) == 3
        assert icp["base_case_id"] == "case_g002"
        assert icp["operating_cases"] == 6
        assert icp["field_packs_valid"] == 6
        assert icp["structure_packs_valid"] == 6
        assert icp["finite_rate_eight_fields"] == pytest.approx(1.0)
        assert "Broader 360-case" in icp["scope_note"]
    finally:
        SIMPLE.plt.close(ccp_figure)
        SIMPLE.plt.close(icp_figure)


def test_simple_main_figures_keep_requested_panel_counts_and_single_target() -> None:
    ccp_field, ccp_field_meta = SIMPLE.create_ccp_field_simple()
    zscore, zscore_meta = SIMPLE.create_ccp_zscore_simple()
    icp_field, icp_field_meta = SIMPLE.create_icp_field_simple()
    sdf, sdf_meta = SIMPLE.create_icp_sdf_simple()
    try:
        assert len(ccp_field.axes) == 2  # one map plus one colorbar
        assert ccp_field_meta["shown_field"] == "ne"
        assert len(zscore.axes) == 2
        assert zscore_meta["shown_target"] == "ne"
        assert zscore_meta["z_mean"] == pytest.approx(0.0, abs=2.0e-6)
        assert zscore_meta["z_std_ddof0"] == pytest.approx(1.0, abs=2.0e-6)
        assert len(icp_field.axes) == 2  # one map plus one colorbar
        assert icp_field_meta["shown_field"] == "ne"
        assert len(sdf.axes) == 4  # dimension, one SDF, union SDF, shared colorbar
        assert sdf_meta["shown_sdf_slot"] == "sdf_coil_03"
        assert sdf_meta["strict_sign_contradiction_cells"] == 0
    finally:
        for figure in (ccp_field, zscore, icp_field, sdf):
            SIMPLE.plt.close(figure)


def test_simple_loss_figures_keep_only_two_equations_and_main_ideas() -> None:
    ccp_figure, ccp = SIMPLE.create_ccp_loss_simple()
    icp_figure, icp = SIMPLE.create_icp_loss_simple()
    try:
        assert len(ccp_figure.axes) == 1
        assert ccp["presentation_revision"] == "two_equations_four_ideas_v2"
        assert ccp["per_target_loss"]["boundary_weight"] == pytest.approx(0.25)
        assert len(icp_figure.axes) == 1
        assert icp["presentation_revision"] == "two_equations_three_ideas_v2"
        assert icp["target_weights"] == {"ne": 1.0, "ni": 1.0, "Te": 1.2, "phi": 0.5}
    finally:
        SIMPLE.plt.close(ccp_figure)
        SIMPLE.plt.close(icp_figure)


def test_simple_exports_keep_editable_svg_text_and_non_type3_pdf(tmp_path: Path) -> None:
    figure, _ = SIMPLE.create_ccp_quality_simple()
    try:
        SIMPLE._save(
            figure,
            out_dir=tmp_path,
            stem="font_contract",
            formats=("svg", "pdf"),
            dpi=72,
        )
        svg = (tmp_path / "font_contract.svg").read_text(encoding="utf-8")
        pdf = (tmp_path / "font_contract.pdf").read_bytes()

        assert "<text" in svg
        assert b"/Subtype /Type3" not in pdf
    finally:
        SIMPLE.plt.close(figure)


def test_ccp_zscore_calculation_uses_train_plasma_population_statistics() -> None:
    figure, metadata = CALC.create_ccp_zscore_calculation()
    try:
        assert metadata["fit_case_count"] == 54
        assert metadata["fit_region"] == "plasma_only"
        assert metadata["fit_pixel_count"] == 1_321_920
        assert metadata["value_transform"] == "identity"
        assert metadata["clip"] == "none"
        assert metadata["std_ddof"] == 0
        assert metadata["mean_m3"] == pytest.approx(1.2915238680752868e15)
        assert metadata["std_m3"] == pytest.approx(1.8712640036294472e15)
        assert metadata["example"]["z"] == pytest.approx(0.912999, rel=2.0e-4)
    finally:
        CALC.plt.close(figure)


def test_calculation_registry_excludes_number_only_dataset_cards() -> None:
    assert set(CALC.BUILDERS) == {"ccp_zscore_calc", "ccp_loss_calc", "icp_loss_calc"}
    assert all("dataset_composition" not in stem for stem, _builder in CALC.BUILDERS.values())


def test_calculation_sheets_keep_the_trained_loss_contracts() -> None:
    ccp_figure, ccp = CALC.create_ccp_loss_calculation()
    icp_figure, icp = CALC.create_icp_loss_calculation()
    try:
        assert ccp["explicit_huber_piecewise"] is True
        assert ccp["per_target_loss"]["boundary_weight"] == pytest.approx(0.25)
        assert ccp["per_target_loss"]["gradient_weight"] == pytest.approx(0.10)
        assert ccp["per_target_loss"]["multiscale_weight"] == pytest.approx(0.05)
        assert ccp["effective_target_weights"] == pytest.approx(
            {"ne": 1 / 6, "ni": 1 / 6, "Te": 1 / 3, "phi": 1 / 3}
        )

        assert icp["status"] == "trained_part_source_v1_baseline"
        assert icp["explicit_huber_piecewise"] is True
        assert icp["per_target_loss"]["boundary_weight"] == pytest.approx(0.0)
        assert icp["per_target_loss"]["gradient_weight"] == pytest.approx(0.02)
        assert icp["per_target_loss"]["multiscale_weight"] == pytest.approx(0.05)
        assert icp["target_weights"] == {"ne": 1.0, "ni": 1.0, "Te": 1.2, "phi": 0.5}
        assert icp["target_weight_sum_normalization"] is False
    finally:
        CALC.plt.close(ccp_figure)
        CALC.plt.close(icp_figure)
