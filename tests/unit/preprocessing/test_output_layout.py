from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plasma_surrogate.core.synthetic_data import build_synthetic_dataset
from plasma_surrogate.preprocessing.runner import PreprocessRunner
from tests._config_presets import (
    default_target_transforms_four_field_example,
    runtime_table_plus_structure,
)


def test_preprocess_saves_output_layout_and_train_only_cond_stats(tmp_path: Path):
    run_dir = tmp_path / "run"
    dataset = build_synthetic_dataset(
        {"n_cases": 9, "height": 8, "width": 8, "cond_dim": 3, "seed": 11},
        run_dir,
    )
    pre = PreprocessRunner(
        {
            "split": {"seed": 1, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {"target_transforms": default_target_transforms_four_field_example()},
        },
        run_dir / "preprocessing",
        runtime_cfg=runtime_table_plus_structure(),
    )
    output = pre.run(
        cases=dataset.cases,
        geometry_root=dataset.geometry_root,
        target_metadata=dataset.target_metadata,
    )

    with (run_dir / "preprocessing" / "schema" / "output_layout.json").open("r", encoding="utf-8") as f:
        layout = json.load(f)
    assert layout == {"order": "C", "shape": [4, 8, 8], "vars": ["ne", "ni", "Te", "phi"]}
    with (run_dir / "preprocessing" / "schema" / "field_layout.json").open("r", encoding="utf-8") as f:
        field_layout = json.load(f)
    assert field_layout == {
        "version": 1,
        "layout_type": "grid2d",
        "vars": layout["vars"],
        "shape": layout["shape"],
        "order": layout["order"],
        "axes": ["channel", "y", "x"],
    }
    with (run_dir / "preprocessing" / "schema" / "target_role_schema.json").open("r", encoding="utf-8") as f:
        role_schema = json.load(f)
    assert role_schema["vars"] == ["ne", "ni", "Te", "phi"]
    assert role_schema["role_to_targets"]["density_electron"] == ["ne"]
    assert role_schema["role_to_targets"]["potential"] == ["phi"]
    assert set(role_schema["positive_targets"]) == {"ne", "ni", "Te"}
    targets_by_id = {str(item["id"]): item for item in role_schema["targets"]}
    assert targets_by_id["ne"]["role"] == "density_electron"
    assert targets_by_id["ne"]["field_family"] == "density"
    assert targets_by_id["Te"]["role"] == "temperature_electron"
    assert targets_by_id["phi"]["role"] == "potential"
    assert targets_by_id["phi"]["positive"] is False
    assert (run_dir / "preprocessing" / "split" / "split_extrap_v1.json").exists()
    assert not (run_dir / "preprocessing" / "split" / "split_pressure_extrap_v1.json").exists()
    assert (run_dir / "preprocessing" / "scalers" / "xgrid_channel_scalers.json").exists()
    assert (run_dir / "preprocessing" / "sampling" / "pairs" / "phase_wrap_pairs.json").exists()
    assert (run_dir / "preprocessing" / "sampling" / "pairs" / "time_adj_pairs.json").exists()
    assert (run_dir / "preprocessing" / "stats" / "y_stats.json").exists()
    assert (run_dir / "preprocessing" / "validation" / "repro_hashes.json").exists()
    with (run_dir / "preprocessing" / "validation" / "runtime_schema_hashes.json").open("r", encoding="utf-8") as f:
        runtime_schema_hashes = json.load(f)
    assert runtime_schema_hashes["target_schema_hash"]
    assert runtime_schema_hashes["feature_schema_hash"]
    assert output.hashes["target_schema_hash"] == runtime_schema_hashes["target_schema_hash"]
    assert output.hashes["feature_schema_hash"] == runtime_schema_hashes["feature_schema_hash"]

    with (run_dir / "preprocessing" / "stats" / "cond_stats.json").open("r", encoding="utf-8") as f:
        cond_stats = json.load(f)

    train_ids = set(output.split["train"])
    train_rows = np.array(
        [[case["cond"]["c0"], case["cond"]["c1"], case["cond"]["c2"]] for case in dataset.cases if case["case_id"] in train_ids],
        dtype=np.float32,
    )
    assert np.isclose(cond_stats["c0"]["mean"], float(np.mean(train_rows[:, 0])))
    assert np.isclose(cond_stats["c1"]["mean"], float(np.mean(train_rows[:, 1])))
    assert np.isclose(cond_stats["c2"]["mean"], float(np.mean(train_rows[:, 2])))


def test_preprocess_y_stats_reduce_large_density_fields_in_float64(tmp_path: Path):
    run_dir = tmp_path / "large_density"
    dataset = build_synthetic_dataset(
        {"n_cases": 9, "height": 8, "width": 8, "cond_dim": 3, "seed": 17},
        run_dir,
    )
    for idx, case in enumerate(dataset.cases):
        case["y"]["ne"] = np.full((8, 8), float(idx + 1) * 1.0e18, dtype=np.float32)
        case["y"]["ni"] = np.full((8, 8), float(idx + 2) * 1.0e18, dtype=np.float32)
    pre = PreprocessRunner(
        {
            "split": {"seed": 2, "ratios": [0.6, 0.2, 0.2]},
            "scalers": {"target_transforms": default_target_transforms_four_field_example()},
        },
        run_dir / "preprocessing",
        runtime_cfg=runtime_table_plus_structure(),
    )
    output = pre.run(cases=dataset.cases, geometry_root=dataset.geometry_root)

    y_stats = json.loads(
        (run_dir / "preprocessing" / "stats" / "y_stats.json").read_text(encoding="utf-8")
    )
    assert np.isfinite(float(y_stats["ne"]["std"]))
    train_ids = set(output.split["train"])
    expected = np.concatenate(
        [
            np.asarray(case["y"]["ne"], dtype=np.float64).reshape(-1)
            for case in dataset.cases
            if case["case_id"] in train_ids
        ]
    )
    assert float(y_stats["ne"]["std"]) == float(np.std(expected, dtype=np.float64))


def test_preprocess_target_scaler_uses_case_structure_masks_in_train_rows_only(tmp_path: Path):
    run_dir = tmp_path / "case_masks"
    dataset = build_synthetic_dataset(
        {"n_cases": 9, "height": 8, "width": 8, "cond_dim": 3, "seed": 23},
        run_dir,
    )
    structure_dir = run_dir / "case_structures"
    structure_dir.mkdir(parents=True)
    active_value_by_id: dict[str, float] = {}
    for index, case in enumerate(dataset.cases):
        mask = np.zeros((8, 8), dtype=np.float32)
        row, col = index % 8, (index * 3) % 8
        mask[row, col] = 1.0
        active_value = float(2**index)
        active_value_by_id[str(case["case_id"])] = active_value
        field = np.full((8, 8), 100_000.0 + float(index), dtype=np.float32)
        field[row, col] = active_value
        case["y"]["ne"] = field
        structure_path = structure_dir / f"{case['case_id']}.npz"
        np.savez_compressed(
            structure_path,
            mask_plasma=mask,
            valid_field_mask=np.ones((8, 8), dtype=np.float32),
            outside_mask=1.0 - mask,
        )
        case["structure_npz"] = str(structure_path)

    preprocess_dir = run_dir / "preprocessing"
    output = PreprocessRunner(
        {
            "split": {"seed": 19, "ratios": [0.6, 0.2, 0.2]},
            "coord_features": {"enabled": False},
            "scalers": {"target_transforms": default_target_transforms_four_field_example()},
        },
        preprocess_dir,
        runtime_cfg=runtime_table_plus_structure(),
    ).run(
        cases=dataset.cases,
        geometry_root=dataset.geometry_root,
        target_metadata=dataset.target_metadata,
    )

    y_scalers = json.loads((preprocess_dir / "scalers" / "y_scalers.json").read_text(encoding="utf-8"))
    train_values = np.asarray(
        [active_value_by_id[case_id] for case_id in output.split["train"]],
        dtype=np.float64,
    )
    assert float(y_scalers["ne"]["mean"][0]) == float(np.mean(train_values))
    assert float(y_scalers["ne"]["std"][0]) == float(np.std(train_values))
    assert not np.isclose(
        float(y_scalers["ne"]["mean"][0]),
        float(np.mean(list(active_value_by_id.values()))),
    )
    fit_policy = json.loads((preprocess_dir / "scalers" / "fit_policy.json").read_text(encoding="utf-8"))
    assert fit_policy["mask_applied"] is True
    assert fit_policy["mask_active_ratio"] == 1.0 / 64.0
