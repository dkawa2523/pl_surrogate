from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from icp_stage4_protocol_reliability import (
    assess_protocol_reliability,
    read_validation_selection,
    validation_row_is_selectable,
)

from run_icp_stage4_struct_suite_plan import (
    _leaderboard_for_config,
    _limit_threads,
    _run_config,
    _set_below_normal_priority,
    _set_lr,
    _wait_until_idle,
)


RUN_ROOT_DEFAULT = Path("runs/icp_stage4_struct_model_suite_part_sdf_lite_v1_e80")
CONFIG_ROOT_DEFAULT = Path("configs/experimental/icp_stage4/generated_struct_model_suite_part_sdf_lite_v1_e80")
UNO_CONFIG_ROOT_DEFAULT = Path(
    "configs/experimental/icp_stage4/generated_struct_model_suite_part_sdf_lite_v1_e80_uno_tuning"
)

DENSITY_SELECTION = {"ne": 0.4, "ni": 0.4, "Te": 0.1, "phi": 0.1}
DENSITY_LOSS = {"ne": 2.0, "ni": 2.0, "Te": 0.8, "phi": 0.3}

UNO_RECIPES: list[dict[str, Any]] = [
    {
        "group": "allvars",
        "recipe": "u_no_m10_w48_l4_lr3e4",
        "lr": 3.0e-4,
        "n_modes": 10,
        "width": 48,
        "n_layers": 4,
        "warmup_epochs": 0,
    },
    {
        "group": "allvars",
        "recipe": "u_no_m12_w64_l4_lr2e4",
        "lr": 2.0e-4,
        "n_modes": 12,
        "width": 64,
        "n_layers": 4,
        "warmup_epochs": 0,
    },
    {
        "group": "allvars",
        "recipe": "u_no_m16_w64_l5_lr2e4_warmup5",
        "lr": 2.0e-4,
        "n_modes": 16,
        "width": 64,
        "n_layers": 5,
        "warmup_epochs": 5,
    },
    {
        "group": "density",
        "recipe": "u_no_density_m12_w64_l4_lr2e4_dw2",
        "lr": 2.0e-4,
        "n_modes": 12,
        "width": 64,
        "n_layers": 4,
        "warmup_epochs": 0,
        "selection_weights": DENSITY_SELECTION,
        "loss_weights": DENSITY_LOSS,
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICP Stage4 UNO tuning behind an idle GPU gate.")
    parser.add_argument("--config-root", default=str(CONFIG_ROOT_DEFAULT))
    parser.add_argument("--uno-config-root", default=str(UNO_CONFIG_ROOT_DEFAULT))
    parser.add_argument("--run-root", default=str(RUN_ROOT_DEFAULT))
    parser.add_argument("--status-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--plan-json", default=None)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[411],
        help="Independent training seeds; output/config names include the seed to prevent collisions.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--max-gpu-util", type=float, default=10.0)
    parser.add_argument("--max-gpu-mem-mib", type=float, default=2200.0)
    parser.add_argument("--wait-command-absent", action="append", default=[])
    parser.add_argument("--only-recipe", action="append", default=[])
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def _as_float(raw: Any) -> float:
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _read_csv_first(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return dict(rows[0])


def _format_float(value: float) -> str:
    return f"{value:.12g}" if math.isfinite(value) else ""


def _apply_density_objective(cfg: dict[str, Any], recipe: dict[str, Any]) -> None:
    train_cfg = cfg.setdefault("benchmark", {}).setdefault("train", {})
    family_cfg = train_cfg.setdefault("u_no", {})
    selection = family_cfg.setdefault("selection", {})
    selection["mode"] = "best_val_allvars_balance"
    selection["weights"] = dict(recipe.get("selection_weights") or DENSITY_SELECTION)

    weights = dict(recipe.get("loss_weights") or DENSITY_LOSS)
    loss_cfg = train_cfg.setdefault("loss", {})
    supervised = loss_cfg.setdefault("supervised", {})
    supervised["target_weights"] = dict(weights)
    if isinstance(loss_cfg.get("multitask"), dict):
        loss_cfg["multitask"]["fixed_weights_by_var"] = dict(weights)


def _make_uno_cfg(
    base_cfg: dict[str, Any],
    *,
    recipe: dict[str, Any],
    run_root: Path,
    epochs: int,
    seed: int = 411,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    bench = cfg.setdefault("benchmark", {})
    recipe_name = str(recipe["recipe"])
    group = str(recipe["group"])
    run_id = f"{recipe_name}__seed{int(seed)}"
    bench["output_dir"] = str(run_root / "uno_tuning" / f"u_no__{run_id}").replace("\\", "/")
    bench["profile"] = "m7_u_no_experimental"
    bench["seed"] = int(seed)

    eval_cfg = bench.setdefault("eval", {})
    eval_cfg["protocol_variant"] = f"icp_stage4_uno_tuning_{recipe_name}"
    eval_cfg["target_family_for_score"] = "allvars"
    eval_cfg["target_vars_for_score"] = ["ne", "ni", "Te", "phi"]
    eval_cfg["primary_metric"] = "test_r2_group_default_plasma"

    eval_protocol = bench.setdefault("eval_protocol", {})
    eval_protocol["mode"] = "primary_axis"
    eval_protocol["primary_split"] = "structure_holdout"
    eval_protocol["min_primary_test_cases"] = max(
        int(eval_protocol.get("min_primary_test_cases", 3)),
        3,
    )
    eval_protocol["min_primary_test_groups"] = max(
        int(eval_protocol.get("min_primary_test_groups", 3)),
        3,
    )
    eval_protocol["scope"] = "u_no_isolated"
    bench.setdefault("preprocessing", {}).setdefault("scalers", {})["fit_split"] = "structure_holdout"

    train_cfg = bench.setdefault("train", {})
    source_fno_cfg = deepcopy(train_cfg.get("fno", {}))
    if not source_fno_cfg:
        raise ValueError("Base FNO config is missing benchmark.train.fno")
    train_cfg.pop("fno", None)
    train_cfg["u_no"] = source_fno_cfg
    train_cfg.setdefault("unet_like", {})["batch_size_cases"] = 2

    family_cfg = train_cfg["u_no"]
    family_cfg["epochs"] = int(epochs)
    family_cfg["batch_size_cases"] = 2
    family_cfg["model_cfg"] = {
        "backend": "torch",
        "n_modes": int(recipe["n_modes"]),
        "fno_n_modes": int(recipe["n_modes"]),
        "uno_cfg": {
            "width": int(recipe["width"]),
            "n_layers": int(recipe["n_layers"]),
            "dropout": float(recipe.get("dropout", 0.0)),
        },
    }
    _set_lr(cfg, "u_no", float(recipe["lr"]))
    optimizer = family_cfg.setdefault("optimizer", {})
    if isinstance(optimizer, dict):
        optimizer["schedule"] = str(optimizer.get("schedule", "cosine"))
        optimizer["warmup_epochs"] = int(recipe.get("warmup_epochs", 0))

    if group == "density":
        _apply_density_objective(cfg, recipe)
    return cfg


def _make_uno_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    config_root = Path(args.config_root)
    base_path = config_root / "full" / "benchmark_icp_stage4_core4_full_fno.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base FNO config: {base_path}")
    with base_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    out_root = Path(args.uno_config_root) / "uno_tuning"
    out_root.mkdir(parents=True, exist_ok=True)
    selected = set(str(name) for name in args.only_recipe)
    generated: list[dict[str, Any]] = []
    order = 0
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    for recipe in UNO_RECIPES:
        recipe_name = str(recipe["recipe"])
        if selected and recipe_name not in selected:
            continue
        for seed in seeds:
            order += 1
            run_id = f"{recipe_name}__seed{seed}"
            cfg = _make_uno_cfg(
                base_cfg,
                recipe=recipe,
                run_root=Path(args.run_root),
                epochs=int(args.epochs),
                seed=seed,
            )
            out_path = out_root / f"benchmark_icp_stage4_uno_tuning__{run_id}.yaml"
            with out_path.open("w", encoding="utf-8", newline="\n") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
            generated.append(
                {
                    "order": order,
                    "stage": "uno_tuning",
                    "group": str(recipe["group"]),
                    "model": "u_no",
                    "recipe": recipe_name,
                    "run_id": run_id,
                    "seed": seed,
                    "config": out_path,
                }
            )
            print(f"[UNO CONFIG] {out_path}", flush=True)
    return generated


def _collect_uno_summary(items: list[dict[str, Any]], *, out_csv: Path, report_md: Path) -> None:
    rows: list[dict[str, str]] = []
    for item in items:
        cfg_path = Path(item["config"])
        leaderboard = _leaderboard_for_config(cfg_path)
        row = _read_csv_first(leaderboard)
        base = {
            "stage_order": str(item["order"]),
            "stage": str(item["stage"]),
            "group": str(item["group"]),
            "model_id": str(item["model"]),
            "recipe": str(item["recipe"]),
            "run_id": str(item.get("run_id", item["recipe"])),
            "seed": str(item.get("seed", "")),
            "leaderboard": str(leaderboard),
            "config": str(cfg_path),
        }
        if row is None:
            rows.append({**base, "status": "missing_or_empty_leaderboard"})
            continue
        ne = _as_float(row.get("test_r2_ne_plasma"))
        ni = _as_float(row.get("test_r2_ni_plasma"))
        te = _as_float(row.get("test_r2_Te_plasma"))
        phi = _as_float(row.get("test_r2_phi_plasma"))
        primary = _as_float(row.get("primary_metric_value") or row.get("test_r2_group_default_plasma"))
        assessment = assess_protocol_reliability(
            leaderboard.parent,
            leaderboard_row=row,
            config_path=cfg_path,
        )
        validation = read_validation_selection(
            leaderboard.parent,
            model_id=str(row.get("model_id") or item["model"]),
            primary_split=str(assessment.get("primary_split_effective", "")),
        )
        protocol_reliable = str(assessment.get("primary_metric_protocol_reliable", "")).lower() == "true"
        primary_reliable = str(row.get("primary_metric_reliable", "")).strip().lower() == "true"
        validation_ne = _as_float(validation.get("validation_selection_score_ne"))
        validation_ni = _as_float(validation.get("validation_selection_score_ni"))
        selection_probe = {**assessment, **validation}
        rows.append(
            {
                **base,
                "status": "ok" if validation_row_is_selectable(selection_probe) else "unreliable_selection",
                "primary_metric": str(row.get("primary_metric", "test_r2_group_default_plasma")),
                "primary_metric_value": _format_float(primary),
                "primary_metric_reliable": "true" if primary_reliable and protocol_reliable else "false",
                **assessment,
                **validation,
                "validation_density_r2_mean": _format_float(_mean_finite([validation_ne, validation_ni])),
                "final_test_density_r2_mean": _format_float(_mean_finite([ne, ni])),
                "final_test_r2_ne_plasma": _format_float(ne),
                "final_test_r2_ni_plasma": _format_float(ni),
                "final_test_r2_Te_plasma": _format_float(te),
                "final_test_r2_phi_plasma": _format_float(phi),
            }
        )

    fieldnames = [
        "stage_order",
        "stage",
        "group",
        "model_id",
        "recipe",
        "run_id",
        "seed",
        "status",
        "validation_selection_reliable",
        "validation_selection_issue",
        "validation_selection_score",
        "validation_selected_epoch",
        "validation_density_r2_mean",
        "validation_selection_score_ne",
        "validation_selection_score_ni",
        "validation_selection_score_Te",
        "validation_selection_score_phi",
        "validation_metrics",
        "primary_metric",
        "primary_metric_value",
        "primary_metric_reliable",
        "primary_metric_protocol_reliable",
        "eval_protocol_reliable",
        "eval_protocol_issue",
        "eval_protocol_mode_effective",
        "primary_split_effective",
        "min_primary_test_cases",
        "min_primary_test_groups",
        "primary_test_groups",
        "interp_test_cases",
        "extrap_test_cases",
        "structure_holdout_test_cases",
        "scaler_fit_split",
        "final_test_density_r2_mean",
        "final_test_r2_ne_plasma",
        "final_test_r2_ni_plasma",
        "final_test_r2_Te_plasma",
        "final_test_r2_phi_plasma",
        "leaderboard",
        "config",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row.get("status") == "ok" and validation_row_is_selectable(row)]
    validation_ranked = sorted(
        ok_rows,
        key=lambda row: _as_float(row.get("validation_selection_score")),
        reverse=True,
    )
    expected_by_recipe: dict[str, int] = {}
    for item in items:
        recipe = str(item["recipe"])
        expected_by_recipe[recipe] = expected_by_recipe.get(recipe, 0) + 1
    aggregate_rows: list[dict[str, Any]] = []
    for recipe, expected_count in expected_by_recipe.items():
        recipe_rows = [row for row in ok_rows if row.get("recipe") == recipe]
        values = [_as_float(row.get("validation_selection_score")) for row in recipe_rows]
        values = [value for value in values if math.isfinite(value)]
        if len(values) != expected_count:
            continue
        aggregate_rows.append(
            {
                "recipe": recipe,
                "group": recipe_rows[0].get("group", "") if recipe_rows else "",
                "n_seeds": len(values),
                "validation_mean": statistics.fmean(values),
                "validation_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        )
    aggregate_rows.sort(key=lambda row: float(row["validation_mean"]), reverse=True)
    adopted_recipe = str(aggregate_rows[0]["recipe"]) if aggregate_rows else ""
    lines = [
        "# ICP Stage4 UNO Tuning Report",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- summary_csv: {out_csv}",
        f"- reliable_candidates: {len(ok_rows)}",
        f"- adoption_basis: validation_selection_score_only",
        f"- adopted_recipe: {adopted_recipe or 'none'}",
        "- final_test_metrics_used_for_ranking: false",
        "",
        "## Recipe Ranking by Validation (seed aggregate)",
        "",
        "| rank | group | recipe | seeds | validation mean | validation std |",
        "|---:|---|---|---:|---:|---:|",
    ]
    if not aggregate_rows:
        lines.extend(
            [
                "No recipe is eligible for adoption because its complete seed set did not have a reliable validation protocol.",
                "",
            ]
        )
    for idx, row in enumerate(aggregate_rows, start=1):
        lines.append(
            "| {rank} | {group} | {recipe} | {n_seeds} | {mean:.12g} | {std:.12g} |".format(
                rank=idx,
                group=row.get("group", ""),
                recipe=row.get("recipe", ""),
                n_seeds=row.get("n_seeds", 0),
                mean=float(row.get("validation_mean", float("nan"))),
                std=float(row.get("validation_std", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Seed-level Validation Ranking",
            "",
            "| rank | group | recipe | seed | validation | density validation | selected epoch |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(validation_ranked, start=1):
        lines.append(
            "| {rank} | {group} | {recipe} | {seed} | {score} | {density} | {epoch} |".format(
                rank=idx,
                group=row.get("group", ""),
                recipe=row.get("recipe", ""),
                seed=row.get("seed", ""),
                score=row.get("validation_selection_score", ""),
                density=row.get("validation_density_r2_mean", ""),
                epoch=row.get("validation_selected_epoch", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Final Test Metrics (reporting only)",
            "",
            "These values are not used for recipe ranking or adoption.",
            "",
            "| recipe | seed | primary | density | ne | ni | Te | phi |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validation_ranked:
        lines.append(
            "| {recipe} | {seed} | {primary} | {density} | {ne} | {ni} | {te} | {phi} |".format(
                recipe=row.get("recipe", ""),
                seed=row.get("seed", ""),
                primary=row.get("primary_metric_value", ""),
                density=row.get("final_test_density_r2_mean", ""),
                ne=row.get("final_test_r2_ne_plasma", ""),
                ni=row.get("final_test_r2_ni_plasma", ""),
                te=row.get("final_test_r2_Te_plasma", ""),
                phi=row.get("final_test_r2_phi_plasma", ""),
            )
        )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[UNO SUMMARY] {out_csv}", flush=True)
    print(f"[UNO REPORT] {report_md}", flush=True)


def main() -> int:
    args = _parse_args()
    root = Path.cwd()
    for path in (root / "src", root, Path(__file__).resolve().parent):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)

    _set_below_normal_priority()
    _limit_threads()

    run_root = Path(args.run_root)
    logs = run_root / "logs"
    status_csv = Path(args.status_csv or (run_root / "uno_tuning_status.csv"))
    summary_csv = Path(args.summary_csv or (run_root / "summary" / "uno_tuning_summary.csv"))
    report_md = Path(args.report_md or (run_root / "summary" / "uno_tuning_report.md"))
    plan_json = Path(args.plan_json or (run_root / "summary" / "uno_tuning_plan.json"))

    generated = _make_uno_configs(args)
    plan_json.parent.mkdir(parents=True, exist_ok=True)
    plan_json.write_text(
        json.dumps([{**item, "config": str(item["config"])} for item in generated], ensure_ascii=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"[UNO PLAN] {plan_json}", flush=True)

    for item in generated:
        cfg_path = Path(item["config"])
        recipe = str(item["recipe"])
        run_id = str(item.get("run_id", recipe))
        if not _leaderboard_for_config(cfg_path).exists():
            _wait_until_idle(args, label=f"uno_tuning:u_no:{run_id}")
        passed = _run_config(
            cfg_path,
            stage="uno_tuning",
            model="u_no",
            recipe=run_id,
            status_csv=status_csv,
            log_dir=logs,
            stop_on_failure=bool(args.stop_on_failure),
        )
        _collect_uno_summary(generated, out_csv=summary_csv, report_md=report_md)
        if not passed and args.stop_on_failure:
            return 1

    _collect_uno_summary(generated, out_csv=summary_csv, report_md=report_md)
    print("[UNO DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
