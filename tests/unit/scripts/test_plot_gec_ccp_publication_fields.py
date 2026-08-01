from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "plot_gec_ccp_all_best_publication_fields.py"
SPEC = importlib.util.spec_from_file_location("plot_gec_ccp_all_best_publication_fields", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT
SPEC.loader.exec_module(PLOT)


def _write_geometry(root: Path) -> Path:
    geometry = root / "geometry"
    geometry.mkdir(parents=True)
    mask = np.ones((5, 6), dtype=np.float32)
    mask[0, :] = 0.0
    np.save(geometry / "mask_plasma.npy", mask)
    np.save(geometry / "r_coords.npy", np.linspace(0.0, 0.05, 6, dtype=np.float32))
    np.save(geometry / "z_coords.npy", np.linspace(-0.02, 0.02, 5, dtype=np.float32))
    return geometry


def test_publication_masks_load_case_specific_entity_slots_without_union(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_geometry(dataset)
    structure_dir = dataset / "structure_features"
    structure_dir.mkdir()
    mask = np.ones((5, 6), dtype=np.float32)
    mask[0, :] = 0.0
    stack = np.zeros((2, 5, 6), dtype=np.float32)
    stack[0, 2, 1:5] = 1.0
    stack[1, 1:4, 4] = 1.0
    structure_path = structure_dir / "base2.npz"
    np.savez_compressed(
        structure_path,
        mask_plasma=mask,
        valid_field_mask=mask,
        outside_mask=1.0 - mask,
        part_mask_stack=stack,
        part_ids=np.asarray(["boundary_00", "boundary_01"], dtype=object),
    )

    masks = PLOT._load_masks(dataset, case={"case_id": "a", "structure_npz": str(structure_path)})

    assert masks.part_ids == ("boundary_00", "boundary_01")
    assert len(masks.part_masks) == 2
    assert np.array_equal(masks.chamber_parts, np.maximum.reduce(stack, axis=0) > 0.5)
    # Boundary-line overlays do not remove simulation-valid plasma target pixels.
    assert np.array_equal(masks.target, mask > 0.5)


def test_publication_masks_select_one_manifest_alternative_by_td(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    geometry = _write_geometry(dataset)
    base2 = np.zeros((5, 6), dtype=np.float32)
    base3 = np.zeros((5, 6), dtype=np.float32)
    base2[1, 1] = 1.0
    base3[3, 4] = 1.0
    np.savez_compressed(
        geometry / "parts_pack.npz",
        mask_stack=np.stack([base2, base3], axis=0),
        part_ids=np.asarray(["base2", "base3"], dtype=object),
    )
    (geometry / "parts_manifest.json").write_text(
        json.dumps(
            {
                "part_semantics": "alternatives",
                "part_ids": ["base2", "base3"],
                "alternative_conditions": {"base2": {"Td": 0.30}, "base3": {"Td": 0.16}},
            }
        ),
        encoding="utf-8",
    )

    masks = PLOT._load_masks(dataset, case={"case_id": "b", "cond": {"Td": 0.16}})

    assert masks.part_ids == ("base3",)
    assert len(masks.part_masks) == 1
    assert masks.chamber_parts[3, 4]
    assert not masks.chamber_parts[1, 1]


def test_publication_selection_reader_requires_v2_validation_provenance(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="legacy v1/test-selected.*not an implicit fallback"):
        PLOT._read_best_runs(missing)

    manifest = tmp_path / "best_by_model.csv"
    manifest.write_text(
        "model_id,validation_selection_value,validation_selection_mode,validation_selection_reliable,"
        "leaderboard,run_root,selection_split,protocol_variant,scaler_fit_split\n"
        "fno,0.25,min,true,runs/v2/fno/leaderboard.csv,runs/v2/fno,interp,gec_ccp_v2,interp\n",
        encoding="utf-8",
    )
    runs = PLOT._read_best_runs(manifest)

    assert runs["fno"]["run_root"] == Path("runs/v2/fno")
    assert runs["fno"]["validation_selection_value"] == 0.25
    assert "quality" not in runs["fno"]
    assert runs["fno"]["selection_split"] == "interp"
    assert runs["fno"]["protocol"] == "gec_ccp_v2"
    assert runs["fno"]["scaler_fit_split"] == "interp"


def test_publication_engine_uses_checkpoint_spatial_scaler_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_meta = {"scaler_fit_split": "interp"}
    lane_coord = {"contract_version": 3, "channels": {"x": {"mean": 1.0}}}
    lane_distance = {"enabled": True, "signed_tanh_tau_auto": 2.5}
    calls: list[tuple[str, dict[str, object]]] = []

    class _Bundle:
        schemas = {
            "cond_stats": {},
            "coord_feature_pack": None,
            "target_role_schema": {},
            "structure_descriptor_pack": None,
            "latent_feature_pack": None,
        }
        transforms = {"coord_scaler": {}}
        geometry_store = None

        @staticmethod
        def cond_schema_obj() -> object:
            return object()

        @staticmethod
        def axis_schema_obj() -> object:
            return object()

        @staticmethod
        def transform_bundle_for_checkpoint(meta: dict[str, object]) -> object:
            calls.append(("field", meta))
            return object()

        @staticmethod
        def spatial_transform_artifacts_for_checkpoint(meta: dict[str, object]) -> dict[str, object]:
            calls.append(("spatial", meta))
            return {
                "coord_feature_scaler": lane_coord,
                "distance_transform_stats": lane_distance,
            }

    captured: dict[str, object] = {}

    monkeypatch.setattr(PLOT, "load_checkpoint", lambda _path: object())
    monkeypatch.setattr(
        PLOT,
        "load_checkpoint_metadata_with_input_mode",
        lambda _path: (checkpoint_meta, {"input_mode_effective": "table_plus_structure"}),
    )
    monkeypatch.setattr(PLOT.RunBundleLoader, "load", lambda _path, model: _Bundle())
    monkeypatch.setattr(PLOT, "build_geometry_provider", lambda *_args, **_kwargs: object())

    def _capture_engine(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(PLOT, "InferenceEngine", _capture_engine)

    run_cfg = {
        "inference": {"ood": {"source": "root"}},
        "benchmark": {
            "inference": {
                "ood": {"source": "benchmark"},
                "qoi": {"uniformity": {"target": "ne"}},
            }
        },
    }
    PLOT._make_engine(tmp_path, "interp", "deeponet_pod", run_cfg, tmp_path)

    assert calls == [("spatial", checkpoint_meta), ("field", checkpoint_meta)]
    assert captured["coord_feature_scaler"] is lane_coord
    assert captured["coord_distance_transform_stats"] is lane_distance
    assert captured["ood_cfg"] == {
        "source": "benchmark",
        "qoi": {"uniformity": {"target": "ne"}},
    }


def test_publication_split_filter_preserves_default_and_supports_interp_only() -> None:
    assert PLOT._requested_split_files(["interp", "extrap"]) == PLOT.SPLIT_FILES
    assert PLOT._requested_split_files(["interp"]) == {
        "interp": PLOT.SPLIT_FILES["interp"]
    }
    assert PLOT._requested_split_files(["interp", "interp"]) == {
        "interp": PLOT.SPLIT_FILES["interp"]
    }
    with pytest.raises(ValueError, match="unknown evaluation splits"):
        PLOT._requested_split_files(["unknown"])


def test_publication_cli_keeps_both_splits_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    args = PLOT._parse_args()
    assert args.splits == ["interp", "extrap"]


def test_publication_interp_only_summaries_do_not_create_extrap_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_files = {"interp": PLOT.SPLIT_FILES["interp"]}
    summary_rows = [
        {
            "model": "fno",
            "split": "interp",
            "rel_rmse_ne": 0.1,
            "mean_signed_peak_percent_error_ne": 1.0,
            "mean_abs_peak_percent_error_ne": 2.0,
        },
        {
            "model": "fno",
            "split": "extrap",
            "rel_rmse_ne": 9.9,
            "mean_signed_peak_percent_error_ne": 99.0,
            "mean_abs_peak_percent_error_ne": 99.0,
        },
    ]
    plot_rows = [
        {
            "model": "fno",
            "split": "interp",
            "field": "ne",
            "field_label": "ne",
            "units": "m^-3",
            "case_label": "median",
            "case_id": "case_interp",
            "rel_rmse": "0.1",
            "png_path": "publication_by_field/fno/ne/interp/median.png",
            "pdf_path": "publication_by_field/fno/ne/interp/median.pdf",
        },
        {
            "model": "fno",
            "split": "extrap",
            "field": "ne",
            "field_label": "ne",
            "units": "m^-3",
            "case_label": "median",
            "case_id": "case_extrap",
            "rel_rmse": "9.9",
            "png_path": "publication_by_field/fno/ne/extrap/median.png",
            "pdf_path": "publication_by_field/fno/ne/extrap/median.pdf",
        },
    ]

    field_summaries = PLOT._field_summary_rows(
        summary_rows,
        plot_rows,
        split_files=split_files,
    )
    assert {(row["model"], row["split"]) for row in field_summaries} == {
        ("fno", "interp")
    }
    scope_text = "\n".join(PLOT._split_scope_lines(split_files))
    assert "interp" in scope_text
    assert "extrap" not in scope_text
    assert "PP0=5" not in scope_text

    monkeypatch.setattr(PLOT, "OUT_ROOT", tmp_path)
    pages = PLOT._write_model_summary_pages(
        runs={
            "fno": {
                "seed": "412",
                "run_root": tmp_path / "fno",
            }
        },
        plot_rows=plot_rows,
        field_summary_rows=field_summaries,
        split_files=split_files,
    )
    page = pages["fno"].read_text(encoding="utf-8")
    assert "## interp:" in page
    assert "## extrap:" not in page
    assert "case_extrap" not in page
