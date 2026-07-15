from __future__ import annotations

import argparse
import csv
import json
import math
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
    _set_epochs,
    _set_lr,
    _wait_until_idle,
)


RUN_ROOT_DEFAULT = Path("runs/icp_stage4_struct_model_suite_part_sdf_lite_v1_e80")
CONFIG_ROOT_DEFAULT = Path("configs/experimental/icp_stage4/generated_struct_model_suite_part_sdf_lite_v1_e80")
FOLLOWUP_CONFIG_ROOT_DEFAULT = Path(
    "configs/experimental/icp_stage4/generated_struct_model_suite_part_sdf_lite_v1_e80_followup_tuning"
)

DENSITY_SELECTION = {"ne": 0.4, "ni": 0.4, "Te": 0.1, "phi": 0.1}
DENSITY_SELECTION_STRONG = {"ne": 0.45, "ni": 0.45, "Te": 0.05, "phi": 0.05}
DENSITY_LOSS = {"ne": 2.0, "ni": 2.0, "Te": 0.8, "phi": 0.3}
DENSITY_LOSS_STRONG = {"ne": 3.0, "ni": 3.0, "Te": 0.6, "phi": 0.25}


FOLLOWUP_STAGES: list[dict[str, Any]] = [
    {
        "stage": "followup_1_cno_density",
        "group": "cno_density",
        "items": [
            {
                "model": "cno",
                "recipe": "cno_density_w48_l4_lr3e4_dw2",
                "lr": 3.0e-4,
                "width": 48,
                "n_layers": 4,
                "loss_weights": DENSITY_LOSS,
                "selection_weights": DENSITY_SELECTION,
            },
            {
                "model": "cno",
                "recipe": "cno_density_w64_l4_lr25e4_dw2",
                "lr": 2.5e-4,
                "width": 64,
                "n_layers": 4,
                "loss_weights": DENSITY_LOSS,
                "selection_weights": DENSITY_SELECTION,
            },
            {
                "model": "cno",
                "recipe": "cno_density_w48_l4_lr2e4_dw3",
                "lr": 2.0e-4,
                "width": 48,
                "n_layers": 4,
                "loss_weights": DENSITY_LOSS_STRONG,
                "selection_weights": DENSITY_SELECTION_STRONG,
            },
        ],
    },
    {
        "stage": "followup_2_unet_family",
        "group": "unet_family",
        "items": [
            {
                "model": "unet",
                "recipe": "unet_density_ch32_d2_lr3e4",
                "lr": 3.0e-4,
                "base_channels": 32,
                "depth": 2,
            },
            {
                "model": "unet",
                "recipe": "unet_density_ch48_d2_lr2e4",
                "lr": 2.0e-4,
                "base_channels": 48,
                "depth": 2,
            },
            {
                "model": "unetpp",
                "recipe": "unetpp_density_ch32_lr3e4",
                "lr": 3.0e-4,
                "base_channels": 32,
            },
            {
                "model": "unetpp",
                "recipe": "unetpp_density_ch48_lr2e4",
                "lr": 2.0e-4,
                "base_channels": 48,
            },
            {
                "model": "unetpp_attn",
                "recipe": "unetppattn_density_ch32_red4_lr3e4",
                "lr": 3.0e-4,
                "base_channels": 32,
                "reduction": 4,
            },
            {
                "model": "unetpp_attn",
                "recipe": "unetppattn_density_ch48_red4_lr2e4",
                "lr": 2.0e-4,
                "base_channels": 48,
                "reduction": 4,
            },
        ],
        "loss_weights": DENSITY_LOSS,
        "selection_weights": DENSITY_SELECTION,
    },
    {
        "stage": "followup_3_deeponet_pod",
        "group": "deeponet_pod",
        "items": [
            {
                "model": "deeponet_pod",
                "recipe": "pod_density_r48_h160_lr2e4_dw2",
                "lr": 2.0e-4,
                "hidden_dim": 160,
                "latent_dim": 160,
                "rank": 48,
                "coeff_loss_weight": 0.08,
            },
            {
                "model": "deeponet_pod",
                "recipe": "pod_density_r64_h192_lr15e4_dw2",
                "lr": 1.5e-4,
                "hidden_dim": 192,
                "latent_dim": 192,
                "rank": 64,
                "coeff_loss_weight": 0.05,
            },
        ],
        "loss_weights": DENSITY_LOSS,
        "selection_weights": DENSITY_SELECTION,
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run follow-up ICP structural tuning in adoption-decision order.")
    parser.add_argument("--config-root", default=str(CONFIG_ROOT_DEFAULT))
    parser.add_argument("--followup-config-root", default=str(FOLLOWUP_CONFIG_ROOT_DEFAULT))
    parser.add_argument("--run-root", default=str(RUN_ROOT_DEFAULT))
    parser.add_argument("--status-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--max-gpu-util", type=float, default=10.0)
    parser.add_argument("--max-gpu-mem-mib", type=float, default=2200.0)
    parser.add_argument("--wait-command-absent", action="append", default=[])
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--only-stage", choices=[str(stage["stage"]) for stage in FOLLOWUP_STAGES], default=None)
    return parser.parse_args()


def _apply_density_objective(cfg: dict[str, Any], model: str, recipe: dict[str, Any], stage: dict[str, Any]) -> None:
    train_cfg = cfg.setdefault("benchmark", {}).setdefault("train", {})
    family_cfg = train_cfg.setdefault(model, {})
    selection = family_cfg.setdefault("selection", {})
    selection["mode"] = "best_val_allvars_balance"
    selection["weights"] = dict(recipe.get("selection_weights") or stage.get("selection_weights") or DENSITY_SELECTION)

    loss_cfg = train_cfg.setdefault("loss", {})
    weights = dict(recipe.get("loss_weights") or stage.get("loss_weights") or DENSITY_LOSS)
    supervised = loss_cfg.setdefault("supervised", {})
    supervised["target_weights"] = weights
    if "multitask" in loss_cfg and isinstance(loss_cfg["multitask"], dict):
        loss_cfg["multitask"]["fixed_weights_by_var"] = weights


def _apply_followup_recipe(cfg: dict[str, Any], *, model: str, recipe: dict[str, Any], stage: dict[str, Any]) -> None:
    train_cfg = cfg.setdefault("benchmark", {}).setdefault("train", {})
    family_cfg = train_cfg.setdefault(model, {})
    model_cfg = family_cfg.setdefault("model_cfg", {})
    _set_lr(cfg, model, float(recipe["lr"]))
    _apply_density_objective(cfg, model, recipe, stage)

    if model == "cno":
        cno_cfg = model_cfg.setdefault("cno_cfg", {})
        cno_cfg["width"] = int(recipe["width"])
        cno_cfg["n_layers"] = int(recipe["n_layers"])
        cno_cfg.setdefault("dropout", 0.0)
        cno_cfg.setdefault("kernel_size", 3)
    elif model in {"unet", "unetpp", "unetpp_attn"}:
        train_cfg.setdefault("unet_like", {})["batch_size_cases"] = 4
        conv = model_cfg.setdefault("conv_cfg", {})
        conv["base_channels"] = int(recipe["base_channels"])
        if "depth" in recipe:
            conv["depth"] = int(recipe["depth"])
        if model == "unetpp_attn":
            attention_cfg = conv.setdefault("attention_cfg", {})
            attention_cfg["enabled"] = True
            attention_cfg["reduction"] = int(recipe["reduction"])
    elif model == "deeponet_pod":
        family_cfg["batch_size_cases"] = 4
        train_cfg.setdefault("unet_like", {})["batch_size_cases"] = 4
        model_cfg["hidden_dim"] = int(recipe["hidden_dim"])
        model_cfg["latent_dim"] = int(recipe["latent_dim"])
        model_cfg["coeff_loss_weight"] = float(recipe["coeff_loss_weight"])
        model_cfg.setdefault("basis", {})["rank"] = int(recipe["rank"])
        loss_cfg = train_cfg.setdefault("loss", {})
        loss_cfg.setdefault("deeponet_pod", {})["coeff_loss_weight"] = float(recipe["coeff_loss_weight"])
    else:
        raise ValueError(f"Unsupported follow-up model: {model}")


def _make_followup_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    config_root = Path(args.config_root)
    out_root = Path(args.followup_config_root) / "followup_tuning"
    out_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.run_root)
    generated: list[dict[str, Any]] = []
    selected_stages = [stage for stage in FOLLOWUP_STAGES if args.only_stage in {None, stage["stage"]}]
    for order, stage in enumerate(selected_stages, start=1):
        for recipe in stage["items"]:
            model = str(recipe["model"])
            recipe_name = str(recipe["recipe"])
            base_path = config_root / "full" / f"benchmark_icp_stage4_core4_full_{model}.yaml"
            if not base_path.exists():
                print(f"[FOLLOWUP CONFIG SKIP] missing base config for {model}: {base_path}", flush=True)
                continue
            with base_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            bench = cfg.setdefault("benchmark", {})
            bench["output_dir"] = str(run_root / "followup_tuning" / f"{model}__{recipe_name}").replace("\\", "/")
            eval_cfg = bench.setdefault("eval", {})
            eval_cfg["protocol_variant"] = f"icp_stage4_followup_{model}_{recipe_name}"
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
            bench.setdefault("preprocessing", {}).setdefault("scalers", {})[
                "fit_split"
            ] = "structure_holdout"
            _set_epochs(cfg, model, int(args.epochs))
            _apply_followup_recipe(cfg, model=model, recipe=recipe, stage=stage)
            out_path = out_root / f"benchmark_icp_stage4_followup_{model}__{recipe_name}.yaml"
            with out_path.open("w", encoding="utf-8", newline="\n") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
            generated.append(
                {
                    "order": order,
                    "stage": stage["stage"],
                    "group": stage["group"],
                    "model": model,
                    "recipe": recipe_name,
                    "config": out_path,
                }
            )
            print(f"[FOLLOWUP CONFIG] {out_path}", flush=True)
    return generated


def _read_csv_first(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return dict(rows[0])


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


def _collect_followup_summary(items: list[dict[str, Any]], *, out_csv: Path, report_md: Path) -> None:
    rows: list[dict[str, str]] = []
    for item in items:
        cfg_path = Path(item["config"])
        leaderboard = _leaderboard_for_config(cfg_path)
        row = _read_csv_first(leaderboard)
        if row is None:
            rows.append(
                {
                    "stage_order": str(item["order"]),
                    "stage": str(item["stage"]),
                    "group": str(item["group"]),
                    "model_id": str(item["model"]),
                    "recipe": str(item["recipe"]),
                    "status": "missing_or_empty_leaderboard",
                    "leaderboard": str(leaderboard),
                    "config": str(cfg_path),
                }
            )
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
        enriched = {
            "stage_order": str(item["order"]),
            "stage": str(item["stage"]),
            "group": str(item["group"]),
            "model_id": str(item["model"]),
            "recipe": str(item["recipe"]),
            "status": "ok" if validation_row_is_selectable(selection_probe) else "unreliable_selection",
            "primary_metric": str(row.get("primary_metric", "test_r2_group_default_plasma")),
            "primary_metric_value": f"{primary:.12g}" if math.isfinite(primary) else "",
            "primary_metric_reliable": "true" if primary_reliable and protocol_reliable else "false",
            **assessment,
            **validation,
            "validation_density_r2_mean": f"{_mean_finite([validation_ne, validation_ni]):.12g}",
            "final_test_density_r2_mean": f"{_mean_finite([ne, ni]):.12g}",
            "final_test_r2_ne_plasma": f"{ne:.12g}" if math.isfinite(ne) else "",
            "final_test_r2_ni_plasma": f"{ni:.12g}" if math.isfinite(ni) else "",
            "final_test_r2_Te_plasma": f"{te:.12g}" if math.isfinite(te) else "",
            "final_test_r2_phi_plasma": f"{phi:.12g}" if math.isfinite(phi) else "",
            "leaderboard": str(leaderboard),
            "config": str(cfg_path),
        }
        rows.append(enriched)

    fieldnames: list[str] = []
    preferred = [
        "stage_order",
        "stage",
        "group",
        "model_id",
        "recipe",
        "status",
        "primary_metric",
        "primary_metric_value",
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
    for key in preferred:
        if key not in fieldnames:
            fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

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
    lines = [
        "# ICP Stage4 Follow-Up Tuning Report",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- summary_csv: {out_csv}",
        f"- reliable_candidates: {len(ok_rows)}",
        "- adoption_basis: validation_selection_score_only",
        "- final_test_metrics_used_for_ranking: false",
        "",
        "## Validation Ranking",
        "",
        "| rank | group | model | recipe | validation | density validation | selected epoch |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    if not ok_rows:
        lines.extend(
            [
                "No candidate is eligible for adoption because no explicitly reliable primary protocol was found.",
                "",
            ]
        )
    for idx, row in enumerate(validation_ranked, start=1):
        lines.append(
            "| {rank} | {group} | {model_id} | {recipe} | {score} | {density} | {epoch} |".format(
                rank=idx,
                group=row.get("group", ""),
                model_id=row.get("model_id", ""),
                recipe=row.get("recipe", ""),
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
            "These values are not used for tuning order or adoption.",
            "",
            "| group | model | recipe | primary | density | ne | ni | Te | phi |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validation_ranked:
        lines.append(
            "| {group} | {model_id} | {recipe} | {primary} | {density} | {ne} | {ni} | {te} | {phi} |".format(
                group=row.get("group", ""),
                model_id=row.get("model_id", ""),
                recipe=row.get("recipe", ""),
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
    print(f"[FOLLOWUP SUMMARY] {out_csv}", flush=True)
    print(f"[FOLLOWUP REPORT] {report_md}", flush=True)


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
    status_csv = Path(args.status_csv or (run_root / "followup_tuning_status.csv"))
    summary_csv = Path(args.summary_csv or (run_root / "summary" / "followup_tuning_summary.csv"))
    report_md = Path(args.report_md or (run_root / "summary" / "followup_tuning_report.md"))

    generated = _make_followup_configs(args)
    plan_json = run_root / "summary" / "followup_tuning_plan.json"
    plan_json.parent.mkdir(parents=True, exist_ok=True)
    plan_json.write_text(
        json.dumps(
            [
                {**item, "config": str(item["config"])}
                for item in generated
            ],
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[FOLLOWUP PLAN] {plan_json}", flush=True)

    for item in generated:
        cfg_path = Path(item["config"])
        model = str(item["model"])
        recipe = str(item["recipe"])
        stage = str(item["stage"])
        if not _leaderboard_for_config(cfg_path).exists():
            _wait_until_idle(args, label=f"{stage}:{model}:{recipe}")
        passed = _run_config(
            cfg_path,
            stage=stage,
            model=model,
            recipe=recipe,
            status_csv=status_csv,
            log_dir=logs,
            stop_on_failure=bool(args.stop_on_failure),
        )
        _collect_followup_summary(generated, out_csv=summary_csv, report_md=report_md)
        if not passed and args.stop_on_failure:
            return 1

    _collect_followup_summary(generated, out_csv=summary_csv, report_md=report_md)
    print("[FOLLOWUP DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
