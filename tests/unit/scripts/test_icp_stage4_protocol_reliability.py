from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "icp_stage4" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from icp_stage4_protocol_reliability import (  # noqa: E402
    assess_protocol_reliability,
    enforce_structure_study_protocol,
    read_validation_selection,
    row_is_reliably_selectable,
    validation_row_is_selectable,
)
from generate_icp_stage4_core4_benchmark_configs import _load_template, _mutate_config  # noqa: E402
from run_icp_stage4_uno_tuning_plan import (  # noqa: E402
    UNO_RECIPES,
    _collect_uno_summary,
    _make_uno_cfg,
    _make_uno_configs,
)
from run_icp_stage4_struct_suite_plan import _select_top_models  # noqa: E402
from summarize_icp_stage4_core4_benchmarks import _summary_row  # noqa: E402


def _write_protocol_case(
    root: Path,
    *,
    primary_split: str,
    split: dict[str, list[str]],
    groups: dict[str, str],
) -> Path:
    data_root = root / "data"
    data_root.mkdir(parents=True)
    with (data_root / "index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "split_group"])
        writer.writeheader()
        for case_id, group_id in groups.items():
            writer.writerow({"case_id": case_id, "split_group": group_id})

    split_dir = root / "preprocessing" / "split"
    split_dir.mkdir(parents=True)
    filenames = {
        "random": "split_random_v1.json",
        "interp": "split_interp_v1.json",
        "extrap": "split_extrap_v1.json",
        "structure_holdout": "split_structure_holdout_v1.json",
    }
    empty = {"train": ["tr"], "val": ["va"], "test": ["te0", "te1", "te2"]}
    for name, filename in filenames.items():
        payload = split if name == primary_split else empty
        (split_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    resolved = {
        "dataset": {
            "root": str(data_root),
            "index_csv": "index.csv",
            "case_id_column": "case_id",
            "split_group_column": "split_group",
        },
        "eval_protocol": {
            "mode": "primary_axis",
            "primary_split": primary_split,
            "min_primary_test_cases": 3,
        },
        "preprocessing": {"scalers": {"fit_split": primary_split}},
    }
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    return root


def test_generic_metric_uses_resolved_interp_split_and_rejects_one_case(tmp_path: Path) -> None:
    output = _write_protocol_case(
        tmp_path / "run",
        primary_split="interp",
        split={"train": ["a", "b"], "val": ["c"], "test": ["d"]},
        groups={"a": "g1", "b": "g2", "c": "g3", "d": "g4"},
    )

    result = assess_protocol_reliability(
        output,
        leaderboard_row={
            "primary_metric": "test_r2_group_default_plasma",
            "primary_metric_reliable": "true",
        },
    )

    assert result["primary_split_effective"] == "interp"
    assert result["primary_metric_protocol_reliable"] == "false"
    assert "interp_test_too_small:1<3" in result["eval_protocol_issue"]

    leaderboard = output / "leaderboard.csv"
    with leaderboard.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "primary_metric",
                "primary_metric_value",
                "primary_metric_reliable",
                "primary_metric_protocol_reliable",
                "test_rmse_ne_plasma",
                "test_rmse_ni_plasma",
                "test_r2_ne_plasma",
                "test_r2_ni_plasma",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "primary_metric": "test_r2_group_default_plasma",
                "primary_metric_value": "0.99",
                "primary_metric_reliable": "true",
                "primary_metric_protocol_reliable": "true",
                "test_rmse_ne_plasma": "1",
                "test_rmse_ni_plasma": "1",
                "test_r2_ne_plasma": "0.99",
                "test_r2_ni_plasma": "0.99",
            }
        )
    summary = _summary_row(size="full", model="u_no", leaderboard=leaderboard)
    assert summary["primary_metric_reliable"] == "false"
    assert summary["primary_metric_protocol_reliable"] == "false"


def test_structure_holdout_requires_disjoint_groups(tmp_path: Path) -> None:
    output = _write_protocol_case(
        tmp_path / "run",
        primary_split="structure_holdout",
        split={"train": ["a1", "a2"], "val": ["b1"], "test": ["c1", "c2", "c3"]},
        groups={"a1": "ga", "a2": "ga", "b1": "gb", "c1": "gc", "c2": "gd", "c3": "ge"},
    )
    assert assess_protocol_reliability(output)["primary_metric_protocol_reliable"] == "true"

    with (output / "preprocessing" / "split" / "split_structure_holdout_v1.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump({"train": ["a1"], "val": ["b1"], "test": ["a2", "c1", "c2"]}, f)
    result = assess_protocol_reliability(output)
    assert result["primary_metric_protocol_reliable"] == "false"
    assert "structure_group_overlap_train_test:1" in result["eval_protocol_issue"]


def test_selection_requires_explicit_metric_and_protocol_reliability() -> None:
    assert row_is_reliably_selectable(
        {"primary_metric_reliable": "true", "primary_metric_protocol_reliable": "true"}
    )
    assert not row_is_reliably_selectable({"primary_metric_reliable": "true"})
    assert not row_is_reliably_selectable(
        {"primary_metric_reliable": "true", "primary_metric_protocol_reliable": "false"}
    )
    assert validation_row_is_selectable(
        {"primary_metric_protocol_reliable": "true", "validation_selection_reliable": "true"}
    )
    assert not validation_row_is_selectable(
        {"primary_metric_protocol_reliable": "true", "validation_selection_reliable": "false"}
    )


def test_validation_selection_reads_selected_epoch_not_test_metric(tmp_path: Path) -> None:
    metrics = (
        tmp_path
        / "models"
        / "u_no"
        / "eval_protocol"
        / "structure_holdout"
        / "train"
        / "scalars"
        / "metrics.csv"
    )
    metrics.parent.mkdir(parents=True)
    with metrics.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "val_balance_score",
                "selected_epoch_flag",
                "selection_valid_flag",
                "selection_score_ne",
                "selection_score_ni",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "3",
                "val_balance_score": "0.91",
                "selected_epoch_flag": "1",
                "selection_valid_flag": "1",
                "selection_score_ne": "0.92",
                "selection_score_ni": "0.93",
            }
        )
        writer.writerow(
            {
                "epoch": "4",
                "val_balance_score": "0.10",
                "selected_epoch_flag": "0",
                "selection_valid_flag": "1",
                "selection_score_ne": "0.11",
                "selection_score_ni": "0.12",
            }
        )
    result = read_validation_selection(
        tmp_path,
        model_id="u_no",
        primary_split="structure_holdout",
    )
    assert result["validation_selection_reliable"] == "true"
    assert result["validation_selection_score"] == "0.91"
    assert result["validation_selected_epoch"] == "3"


def test_struct_model_selection_uses_validation_not_test_metric(tmp_path: Path) -> None:
    summary = tmp_path / "comparison.csv"
    fields = [
        "dataset_size",
        "model_id",
        "status",
        "primary_metric_value",
        "primary_metric_reliable",
        "primary_metric_protocol_reliable",
        "validation_selection_reliable",
        "validation_selection_score",
    ]
    with summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "dataset_size": "full",
                "model_id": "fno",
                "status": "ok",
                "primary_metric_value": "0.20",
                "primary_metric_reliable": "true",
                "primary_metric_protocol_reliable": "true",
                "validation_selection_reliable": "true",
                "validation_selection_score": "0.95",
            }
        )
        writer.writerow(
            {
                "dataset_size": "full",
                "model_id": "ffno",
                "status": "ok",
                "primary_metric_value": "0.99",
                "primary_metric_reliable": "true",
                "primary_metric_protocol_reliable": "true",
                "validation_selection_reliable": "true",
                "validation_selection_score": "0.80",
            }
        )
    assert _select_top_models(summary, top_n=1) == ["fno"]


def test_structure_plan_rejects_interp_primary_before_training() -> None:
    config = {
        "benchmark": {
            "runtime": {"input_mode": "table_plus_structure"},
            "eval_protocol": {"mode": "primary_axis", "primary_split": "interp"},
            "preprocessing": {"scalers": {"fit_split": "interp"}},
        }
    }
    with pytest.raises(ValueError, match="primary_split=structure_holdout"):
        enforce_structure_study_protocol(config)


def test_structural_generator_defaults_to_group_holdout_for_generic_metric() -> None:
    template_root = Path(__file__).resolve().parents[3] / "configs" / "benchmarkrun_ext0520" / "templates"
    config = _mutate_config(
        _load_template(template_root, "fno"),
        size="full",
        model="fno",
        smoke_epochs=1,
        full_epochs=1,
        run_root=Path("runs/test"),
        primary_metric="test_r2_group_default_plasma",
        primary_mode="max",
        primary_split_override="",
        dataset_root="data/test",
        structure_spatial_v1=False,
        part_sdf_lite_v1=True,
        part_lite_v1=False,
        linear_target_preprocessing=False,
        enable_optimize=False,
        dual_axis=False,
    )
    benchmark = config["benchmark"]
    assert benchmark["eval_protocol"]["primary_split"] == "structure_holdout"
    assert benchmark["preprocessing"]["scalers"]["fit_split"] == "structure_holdout"


def test_uno_tuning_overrides_legacy_interp_base_to_structure_holdout() -> None:
    base = {
        "benchmark": {
            "eval": {},
            "eval_protocol": {"mode": "primary_axis", "primary_split": "interp"},
            "preprocessing": {"scalers": {"fit_split": "interp"}},
            "train": {
                "fno": {"optimizer": {}},
                "unet_like": {},
                "loss": {"supervised": {}, "multitask": {}},
            },
        }
    }
    config = _make_uno_cfg(
        base,
        recipe=UNO_RECIPES[0],
        run_root=Path("runs/test"),
        epochs=2,
        seed=412,
    )
    benchmark = config["benchmark"]
    assert benchmark["eval_protocol"]["primary_split"] == "structure_holdout"
    assert benchmark["preprocessing"]["scalers"]["fit_split"] == "structure_holdout"
    assert benchmark["seed"] == 412
    assert benchmark["output_dir"].endswith("__seed412")


def test_uno_seed_matrix_writes_collision_free_configs(tmp_path: Path) -> None:
    base_path = tmp_path / "base" / "full" / "benchmark_icp_stage4_core4_full_fno.yaml"
    base_path.parent.mkdir(parents=True)
    base_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "eval": {},
                    "eval_protocol": {"mode": "primary_axis", "primary_split": "interp"},
                    "preprocessing": {"scalers": {"fit_split": "interp"}},
                    "train": {
                        "fno": {"optimizer": {}},
                        "unet_like": {},
                        "loss": {"supervised": {}, "multitask": {}},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    recipe = str(UNO_RECIPES[0]["recipe"])
    items = _make_uno_configs(
        SimpleNamespace(
            config_root=str(tmp_path / "base"),
            uno_config_root=str(tmp_path / "generated"),
            only_recipe=[recipe],
            seeds=[411, 412, 411],
            run_root=str(tmp_path / "runs"),
            epochs=2,
        )
    )
    assert [item["seed"] for item in items] == [411, 412]
    assert len({str(item["config"]) for item in items}) == 2
    output_dirs = []
    for item in items:
        config = yaml.safe_load(Path(item["config"]).read_text(encoding="utf-8"))
        output_dirs.append(config["benchmark"]["output_dir"])
    assert len(set(output_dirs)) == 2


def test_uno_summary_does_not_rank_unreliable_candidate(tmp_path: Path) -> None:
    output = _write_protocol_case(
        tmp_path / "run",
        primary_split="interp",
        split={"train": ["a", "b"], "val": ["c"], "test": ["d"]},
        groups={"a": "g1", "b": "g2", "c": "g3", "d": "g4"},
    )
    with (output / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "primary_metric",
            "primary_metric_value",
            "primary_metric_reliable",
            "test_r2_group_default_plasma",
            "test_r2_ne_plasma",
            "test_r2_ni_plasma",
            "test_r2_Te_plasma",
            "test_r2_phi_plasma",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "primary_metric": "test_r2_group_default_plasma",
                "primary_metric_value": "0.99",
                "primary_metric_reliable": "true",
                "test_r2_group_default_plasma": "0.99",
                "test_r2_ne_plasma": "0.99",
                "test_r2_ni_plasma": "0.99",
                "test_r2_Te_plasma": "0.99",
                "test_r2_phi_plasma": "0.99",
            }
        )
    config_path = tmp_path / "uno.yaml"
    config_path.write_text(
        yaml.safe_dump({"benchmark": {"output_dir": str(output)}}),
        encoding="utf-8",
    )
    out_csv = tmp_path / "summary.csv"
    report = tmp_path / "report.md"
    _collect_uno_summary(
        [
            {
                "order": 1,
                "stage": "uno_tuning",
                "group": "allvars",
                "model": "u_no",
                "recipe": "candidate",
                "config": config_path,
            }
        ],
        out_csv=out_csv,
        report_md=report,
    )
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["status"] == "unreliable_selection"
    assert row["primary_metric_reliable"] == "false"
    assert "reliable_candidates: 0" in report.read_text(encoding="utf-8")


def test_uno_adoption_uses_validation_not_test_score(tmp_path: Path) -> None:
    items: list[dict[str, object]] = []
    specs = [
        ("high_validation", 0.95, 0.20),
        ("high_test", 0.80, 0.99),
    ]
    for order, (recipe, validation_score, test_score) in enumerate(specs, start=1):
        output = _write_protocol_case(
            tmp_path / recipe,
            primary_split="structure_holdout",
            split={"train": ["a1", "a2"], "val": ["b1"], "test": ["c1", "c2", "c3"]},
            groups={"a1": "ga", "a2": "ga", "b1": "gb", "c1": "gc", "c2": "gd", "c3": "ge"},
        )
        with (output / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
            fields = [
                "model_id",
                "primary_metric",
                "primary_metric_value",
                "primary_metric_reliable",
                "test_r2_ne_plasma",
                "test_r2_ni_plasma",
                "test_r2_Te_plasma",
                "test_r2_phi_plasma",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "model_id": "u_no",
                    "primary_metric": "test_r2_group_default_plasma",
                    "primary_metric_value": test_score,
                    "primary_metric_reliable": "true",
                    "test_r2_ne_plasma": test_score,
                    "test_r2_ni_plasma": test_score,
                    "test_r2_Te_plasma": test_score,
                    "test_r2_phi_plasma": test_score,
                }
            )
        metrics = (
            output
            / "models"
            / "u_no"
            / "eval_protocol"
            / "structure_holdout"
            / "train"
            / "scalars"
            / "metrics.csv"
        )
        metrics.parent.mkdir(parents=True)
        with metrics.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "val_balance_score",
                    "selected_epoch_flag",
                    "selection_valid_flag",
                    "selection_score_ne",
                    "selection_score_ni",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "epoch": 5,
                    "val_balance_score": validation_score,
                    "selected_epoch_flag": 1,
                    "selection_valid_flag": 1,
                    "selection_score_ne": validation_score,
                    "selection_score_ni": validation_score,
                }
            )
        cfg = tmp_path / f"{recipe}.yaml"
        cfg.write_text(yaml.safe_dump({"benchmark": {"output_dir": str(output)}}), encoding="utf-8")
        items.append(
            {
                "order": order,
                "stage": "uno_tuning",
                "group": "allvars",
                "model": "u_no",
                "recipe": recipe,
                "run_id": f"{recipe}__seed411",
                "seed": 411,
                "config": cfg,
            }
        )

    report = tmp_path / "validation_report.md"
    _collect_uno_summary(items, out_csv=tmp_path / "validation_summary.csv", report_md=report)
    text = report.read_text(encoding="utf-8")
    assert "adopted_recipe: high_validation" in text
    assert "final_test_metrics_used_for_ranking: false" in text
