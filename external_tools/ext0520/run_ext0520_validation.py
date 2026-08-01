#!/usr/bin/env python3
"""Run ext0520 benchmark validation for all implemented models.

This script is intentionally outside `src/plasma_surrogate` and serves as an
operator workflow utility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
CLI_MAIN = "plasma_surrogate.cli.main"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_selected_models.py"
PLOT_SCRIPT = ROOT / "scripts" / "plot_spatial_distribution_summary.py"

PERIODIC_DATASET_ROOT = "data/outputs_merged_td_csv_periodic_ext0520"
TRANSIENT_DATASET_ROOT = "data/outputs_merged_td_csv_transient_ext0520"

TARGETS = [
    {"id": "ne", "source_key": "log_ne", "value_transform": "pow10", "units": "m^-3", "dtype": "float32"},
    {"id": "ni", "source_key": "log_ni", "value_transform": "pow10", "units": "m^-3", "dtype": "float32"},
    {"id": "Te", "source_key": "Te", "value_transform": "identity", "units": "eV", "dtype": "float32"},
    {"id": "phi", "source_key": "phi", "value_transform": "identity", "units": "V", "dtype": "float32"},
]

RUNTIME_TABLE_ONLY = {
    "input_mode": "table_only",
    "strict_input_mode": "error",
    "allow_mode_fallback": False,
    "structure": {
        "feature_profile": "none",
        "descriptor_profile": "none",
        "latent_profile": "none",
        "adapter_mode": "auto",
        "provider_mode": "fixed",
    },
}

RUNTIME_PLUS_GRID = {
    "input_mode": "table_plus_structure",
    "strict_input_mode": "error",
    "allow_mode_fallback": False,
    "structure": {
        "feature_profile": "geom_v1_mainline",
        "descriptor_profile": "none",
        "latent_profile": "none",
        "adapter_mode": "auto",
        "provider_mode": "fixed",
    },
}

RUNTIME_PLUS_HYBRID = {
    "input_mode": "table_plus_structure",
    "strict_input_mode": "error",
    "allow_mode_fallback": False,
    "structure": {
        "feature_profile": "geom_v1_mainline",
        "descriptor_profile": "struct_desc_v1",
        "latent_profile": "none",
        "adapter_mode": "hybrid_pack_descriptor",
        "provider_mode": "parametric_parts",
    },
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    model: str
    fixture: str
    runtime: dict[str, Any]
    dataset_root: str
    stage_epochs: dict[str, int]
    axis_mode: str = "steady"
    harmonics: int = 1
    lane: str = "plus_grid"


PERIODIC_SPECS: list[RunSpec] = [
    RunSpec("global_mlp", "global_mlp", "benchmark_periodic_real_m7_global_frozen_ref.yaml", RUNTIME_TABLE_ONLY, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 80}, lane="table_only"),
    RunSpec("deeponet_pod_table_only", "deeponet_pod", "benchmark_periodic_real_m7_deeponet_pod_experimental.yaml", RUNTIME_TABLE_ONLY, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 120}, lane="table_only"),
    RunSpec("unet", "unet", "benchmark_periodic_real_m7_unet_isolated_mainline.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 80}),
    RunSpec("unetpp", "unetpp", "benchmark_periodic_real_m7_unetpp_isolated_mainline.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 80}),
    RunSpec("unetpp_attn", "unetpp_attn", "benchmark_periodic_real_m7_unetpp_attn_isolated_mainline.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 80}),
    RunSpec("fno", "fno", "benchmark_periodic_real_m7_fno_isolated_mainline.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 80}),
    RunSpec("ffno", "ffno", "benchmark_periodic_real_m7_ffno_isolated_mainline.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 80}),
    RunSpec("coord_mlp_fourier", "coord_mlp_fourier", "benchmark_periodic_real_m7_coord_mlp_fourier_experimental.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 120}),
    RunSpec("coord_mlp_siren", "coord_mlp_siren", "benchmark_periodic_real_m7_coord_mlp_siren_experimental.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 120}),
    RunSpec("deeponet_plasma", "deeponet_plasma", "benchmark_periodic_real_m7_deeponet_isolated_mainline.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 120}),
    RunSpec("u_no", "u_no", "benchmark_dual_mode_table_plus_structure_u_no_smoke.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 80}),
    RunSpec("cno", "cno", "benchmark_dual_mode_table_plus_structure_cno_smoke.yaml", RUNTIME_PLUS_GRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 80}),
    RunSpec("deeponet_pod_hybrid", "deeponet_pod", "benchmark_periodic_real_m7_deeponet_pod_experimental.yaml", RUNTIME_PLUS_HYBRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 120}, lane="plus_hybrid"),
    RunSpec("geom_deeponet_siren", "geom_deeponet_siren", "benchmark_dual_mode_table_plus_structure_geom_deeponet_siren_smoke.yaml", RUNTIME_PLUS_HYBRID, PERIODIC_DATASET_ROOT, {"smoke": 2, "full": 80}, lane="plus_hybrid"),
]

TRANSIENT_REP_SPECS: list[RunSpec] = [
    RunSpec("global_mlp_transient", "global_mlp", "benchmark_periodic_real_m7_global_frozen_ref.yaml", RUNTIME_TABLE_ONLY, TRANSIENT_DATASET_ROOT, {"smoke": 1, "full": 40}, axis_mode="time", lane="table_only"),
    RunSpec("fno_transient", "fno", "benchmark_periodic_real_m7_fno_isolated_mainline.yaml", RUNTIME_PLUS_GRID, TRANSIENT_DATASET_ROOT, {"smoke": 1, "full": 40}, axis_mode="time"),
    RunSpec("deeponet_plasma_transient", "deeponet_plasma", "benchmark_periodic_real_m7_deeponet_isolated_mainline.yaml", RUNTIME_PLUS_GRID, TRANSIENT_DATASET_ROOT, {"smoke": 1, "full": 40}, axis_mode="time"),
]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _apply_runtime(cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> None:
    cfg["runtime"] = copy.deepcopy(runtime_cfg)
    bench = cfg.setdefault("benchmark", {})
    bench["runtime"] = copy.deepcopy(runtime_cfg)


def _apply_dataset(cfg: dict[str, Any], dataset_root: str) -> None:
    bench = cfg.setdefault("benchmark", {})
    dataset = bench.setdefault("dataset", {})
    dataset["type"] = "csv_npz"
    dataset["root"] = dataset_root
    dataset["index_csv"] = "index.csv"
    dataset["targets"] = copy.deepcopy(TARGETS)
    dataset.setdefault("cond_columns", ["PP0", "Td", "gamma"])
    dataset.setdefault("axis_column", "axis")
    dataset.setdefault("fields_npz_column", "fields_npz")
    dataset.setdefault("case_id_column", "case_id")
    dataset.setdefault("base_case_id_column", "base_case_id")
    dataset.setdefault("split_group_column", "split_group")
    dataset.setdefault("geometry_root", "geometry")


def _apply_axis_mode(cfg: dict[str, Any], axis_mode: str, harmonics: int) -> None:
    bench = cfg.setdefault("benchmark", {})
    pre = bench.setdefault("preprocessing", {})
    axis = pre.setdefault("axis_schema", {})
    axis["mode"] = axis_mode
    axis["harmonics"] = int(harmonics)
    infer = bench.setdefault("inference", {})
    axis_infer = infer.setdefault("axis", {})
    axis_infer["mode"] = axis_mode
    axis_infer["value"] = 0.5 if axis_mode == "time" else 0.0


def _set_model_epochs(cfg: dict[str, Any], model_name: str, epochs: int) -> None:
    bench = cfg.setdefault("benchmark", {})
    train = bench.setdefault("train", {})
    model_cfg = train.setdefault(model_name, {})
    model_cfg["epochs"] = int(epochs)


def _set_model_batch_size(cfg: dict[str, Any], model_name: str, batch_size_cases: int) -> None:
    bench = cfg.setdefault("benchmark", {})
    train = bench.setdefault("train", {})
    model_cfg = train.setdefault(model_name, {})
    model_cfg["batch_size_cases"] = int(batch_size_cases)


def _override_model_optimizer(
    cfg: dict[str, Any],
    model_name: str,
    *,
    lr: float | None,
    weight_decay: float | None,
) -> None:
    if lr is None and weight_decay is None:
        return
    bench = cfg.setdefault("benchmark", {})
    train = bench.setdefault("train", {})
    model_cfg = train.setdefault(model_name, {})
    opt_cfg = dict(model_cfg.get("optimizer", {}))
    if lr is not None:
        lr_v = float(lr)
        model_cfg["lr"] = lr_v
        opt_cfg["lr"] = lr_v
    if weight_decay is not None:
        opt_cfg["weight_decay"] = float(weight_decay)
    model_cfg["optimizer"] = opt_cfg


def _ensure_preprocess_scalers(cfg: dict[str, Any]) -> None:
    bench = cfg.setdefault("benchmark", {})
    pre = bench.setdefault("preprocessing", {})
    scalers = pre.setdefault("scalers", {})
    if "target_transforms" not in scalers:
        scalers["target_transforms"] = {
            "ne": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
            "ni": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
            "Te": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
            "phi": {"value_transform": "identity", "scaler": "zscore", "fit_scope": "plasma_only", "clip": {"mode": "none"}},
        }
    coord_features = pre.setdefault("coord_features", {})
    if cfg.get("runtime", {}).get("input_mode") == "table_plus_structure":
        if "channels" in coord_features:
            coord_features.pop("channels", None)
        coord_features.setdefault("enabled", True)
        coord_features.setdefault("channels_from_profile", "geom_v1_mainline")
        stats = coord_features.setdefault("distance_transform_stats", {})
        stats.setdefault("enabled", True)
        stats.setdefault("fit_scope", "train_split")
        stats.setdefault("mask_scope", "plasma_plus_band")


def _ensure_eval_protocol_defaults(cfg: dict[str, Any]) -> None:
    bench = cfg.setdefault("benchmark", {})
    eval_cfg = bench.setdefault("eval", {})
    eval_cfg.setdefault("mask_metrics", "plasma_only")
    eval_cfg.setdefault("interp_mode", "overlap")
    eval_cfg.setdefault("target_vars_for_score", ["ne", "ni", "Te", "phi"])
    eval_cfg.setdefault("primary_metric", "test_r2_plasma_mean_dual")
    eval_cfg.setdefault("primary_mode", "max")
    aggregate_cfg = eval_cfg.setdefault("aggregate_score", {})
    aggregate_cfg.setdefault("enabled", True)
    aggregate_cfg.setdefault("rmse_weight", 0.5)
    aggregate_cfg.setdefault("r2_weight", 0.4)
    aggregate_cfg.setdefault("boundary_penalty_weight", 0.1)
    aggregate_cfg.setdefault("use_plasma_metrics", True)

    ep_cfg = bench.setdefault("eval_protocol", {})
    ep_cfg.setdefault("mode", "dual_axis")
    ep_cfg.setdefault("scope", "common")
    ep_cfg.setdefault("primary_split", "interp")
    ep_cfg.setdefault("interp_weight", 0.5)
    ep_cfg.setdefault("extrap_weight", 0.5)
    ep_cfg.setdefault("frozen_ref_tag", "periodic_global_frozen_v2")


def _normalize_config_for_spec(
    cfg: dict[str, Any],
    spec: RunSpec,
    *,
    stage: str,
    stage_label: str,
    uniform_epochs: int | None,
    uniform_batch_size: int | None,
    override_lr: float | None,
    override_weight_decay: float | None,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    bench = out.setdefault("benchmark", {})
    _apply_runtime(out, spec.runtime)
    _apply_dataset(out, spec.dataset_root)
    _apply_axis_mode(out, axis_mode=spec.axis_mode, harmonics=spec.harmonics)
    _ensure_preprocess_scalers(out)
    _ensure_eval_protocol_defaults(out)
    model_epochs = int(uniform_epochs) if uniform_epochs is not None else int(spec.stage_epochs[stage])
    _set_model_epochs(out, spec.model, model_epochs)
    if uniform_batch_size is not None:
        _set_model_batch_size(out, spec.model, int(uniform_batch_size))
    _override_model_optimizer(
        out,
        spec.model,
        lr=override_lr,
        weight_decay=override_weight_decay,
    )
    bench["output_dir"] = f"runs/ext0520/{stage_label}/{spec.key}"
    return out


def _run_benchmark(python_exe: Path, cfg_path: Path) -> None:
    cmd = [str(python_exe), "-m", CLI_MAIN, "benchmark", "run", "--config", str(cfg_path)]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"
    subprocess.run(
        cmd,
        check=True,
        cwd=str(ROOT),
        env=env,
    )


def _read_single_row(leaderboard_csv: Path) -> dict[str, str]:
    with leaderboard_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"empty leaderboard: {leaderboard_csv}")
    return rows[0]


def _write_compare_config(path: Path, rows: list[dict[str, Any]], output_dir: Path) -> None:
    payload = {
        "compare": {
            "output_dir": str(output_dir),
            "objective_metric": "auto_primary",
            "objective_mode": "max",
            "require_same_input_mode": True,
            "global_reference_mode": "off",
            "rows": rows,
        }
    }
    _save_yaml(path, payload)


def _run_compare_and_plot(
    *,
    python_exe: Path,
    lane_name: str,
    row_specs: list[tuple[str, Path, str]],
    out_root: Path,
    dataset_root: str,
) -> None:
    if len(row_specs) == 0:
        return
    compare_cfg = out_root / f"compare_{lane_name}.yaml"
    compare_out = out_root / f"compare_{lane_name}"
    compare_rows = [
        {
            "name": display_name,
            "model_id": model_id,
            "leaderboard_csv": str(leaderboard_csv),
        }
        for display_name, leaderboard_csv, model_id in row_specs
    ]
    _write_compare_config(compare_cfg, compare_rows, compare_out)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [str(python_exe), str(COMPARE_SCRIPT), "--config", str(compare_cfg)],
        check=True,
        cwd=str(ROOT),
        env=env,
    )
    compare_csv = compare_out / "selected_models_comparison.csv"
    model_names = [name for name, _, _ in row_specs]
    plot_out = compare_out / "plots_interp"
    plot_env = os.environ.copy()
    plot_env["PYTHONPATH"] = "src"
    plot_base_cmd = [
        str(python_exe),
        str(PLOT_SCRIPT),
        "--compare-csv",
        str(compare_csv),
        "--out-dir",
        str(plot_out),
        "--protocol",
        "interp",
        "--vars",
        "ne",
        "ni",
        "Te",
        "phi",
        "--model-names",
        *model_names,
        "--dataset-root",
        dataset_root,
        "--index-csv",
        f"{dataset_root}/index.csv",
        "--axis-mode",
        "steady",
        "--geom-id",
        "default",
        "--cond-columns",
        "PP0",
        "Td",
        "gamma",
    ]
    try:
        subprocess.run(
            [*plot_base_cmd, "--include-ground-truth"],
            check=True,
            cwd=str(ROOT),
            env=plot_env,
        )
    except subprocess.CalledProcessError:
        # Hybrid runs can emit geometry-aware case hashes that do not map to dataset GT keys.
        subprocess.run(
            plot_base_cmd,
            check=True,
            cwd=str(ROOT),
            env=plot_env,
        )


def _collect_run_report(run_root: Path) -> dict[str, Any]:
    leaderboard_path = run_root / "leaderboard.csv"
    resolved_path = run_root / "resolved_benchmark.json"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"missing leaderboard: {leaderboard_path}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"missing resolved_benchmark: {resolved_path}")
    row = _read_single_row(leaderboard_path)
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    summary = {
        "run_root": str(run_root),
        "leaderboard": str(leaderboard_path),
        "resolved_benchmark": str(resolved_path),
        "model_id": str(row.get("model_id", "")),
        "input_mode_effective": str(row.get("input_mode_effective", "")),
        "primary_metric": str(row.get("primary_metric", "")),
        "primary_metric_value": str(row.get("primary_metric_value", "")),
        "has_input_mode_meta": all(k in row for k in [
            "input_mode_effective",
            "structure_feature_profile_effective",
            "structure_descriptor_profile_effective",
            "structure_latent_profile_effective",
            "structure_adapter_mode_effective",
            "geometry_provider_mode_effective",
        ]),
        "resolved_has_mode_meta": all(k in resolved for k in [
            "input_mode_effective",
            "structure_feature_profile_effective",
            "structure_descriptor_profile_effective",
            "structure_latent_profile_effective",
            "structure_adapter_mode_effective",
            "geometry_provider_mode_effective",
            "has_structure_inputs_effective",
        ]),
        "effective_steps_per_model": resolved.get("effective_steps_per_model", {}),
        "resolved_train_per_model": resolved.get("resolved_train_per_model", {}),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=str(ROOT / ".venv-torch" / "Scripts" / "python.exe"))
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--include-transient", action="store_true")
    parser.add_argument("--run-compare", action="store_true")
    parser.add_argument("--output-tag", default="", help="Suffix for run directory label, e.g. aligned_hc")
    parser.add_argument(
        "--uniform-epochs",
        type=int,
        default=0,
        help="If >0, force all models to use this epoch count.",
    )
    parser.add_argument(
        "--uniform-batch-size-cases",
        type=int,
        default=0,
        help="If >0, force all models to use this case batch size.",
    )
    parser.add_argument(
        "--override-lr",
        type=float,
        default=0.0,
        help="If >0, override train.<model>.lr and optimizer.lr for all selected specs.",
    )
    parser.add_argument(
        "--override-weight-decay",
        type=float,
        default=-1.0,
        help="If >=0, override train.<model>.optimizer.weight_decay for all selected specs.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated spec keys to run (default: run all selected specs)",
    )
    parser.add_argument("--out-report", default=str(ROOT / "runs" / "ext0520" / "validation_report.json"))
    args = parser.parse_args()

    python_exe = Path(args.python)
    if not python_exe.exists():
        raise FileNotFoundError(f"python executable not found: {python_exe}")

    stage_label = args.stage
    tag = str(args.output_tag).strip()
    if tag:
        stage_label = f"{args.stage}_{tag}"
    uniform_epochs = int(args.uniform_epochs) if int(args.uniform_epochs) > 0 else None
    uniform_batch_size = int(args.uniform_batch_size_cases) if int(args.uniform_batch_size_cases) > 0 else None
    override_lr = float(args.override_lr) if float(args.override_lr) > 0.0 else None
    override_weight_decay = float(args.override_weight_decay) if float(args.override_weight_decay) >= 0.0 else None

    configs_dir = ROOT / "external_tools" / "ext0520" / "generated_configs" / stage_label
    configs_dir.mkdir(parents=True, exist_ok=True)
    run_reports: list[dict[str, Any]] = []
    lane_rows: dict[str, list[tuple[str, Path, str]]] = {"table_only": [], "plus_grid": [], "plus_hybrid": []}

    specs = list(PERIODIC_SPECS)
    if args.include_transient:
        specs.extend(TRANSIENT_REP_SPECS)
    only = [s.strip() for s in str(args.only).split(",") if s.strip()]
    if only:
        allow = set(only)
        specs = [spec for spec in specs if spec.key in allow]
        missing = sorted(list(allow - {spec.key for spec in specs}))
        if missing:
            raise ValueError(f"unknown spec keys in --only: {missing}")

    for spec in specs:
        fixture_cfg = _load_yaml(FIXTURES / spec.fixture)
        cfg = _normalize_config_for_spec(
            fixture_cfg,
            spec,
            stage=args.stage,
            stage_label=stage_label,
            uniform_epochs=uniform_epochs,
            uniform_batch_size=uniform_batch_size,
            override_lr=override_lr,
            override_weight_decay=override_weight_decay,
        )
        cfg_path = configs_dir / f"{spec.key}.yaml"
        _save_yaml(cfg_path, cfg)
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        _run_benchmark(python_exe, cfg_path)
        elapsed = float(time.perf_counter() - t0)
        finished = datetime.now(timezone.utc).isoformat()
        run_root = ROOT / cfg["benchmark"]["output_dir"]
        report = _collect_run_report(run_root)
        report["spec_key"] = spec.key
        report["stage"] = args.stage
        report["stage_label"] = stage_label
        report["dataset_root"] = spec.dataset_root
        report["started_at_utc"] = started
        report["finished_at_utc"] = finished
        report["run_wall_seconds"] = elapsed
        report["uniform_epochs"] = uniform_epochs
        report["uniform_batch_size_cases"] = uniform_batch_size
        report["override_lr"] = override_lr
        report["override_weight_decay"] = override_weight_decay
        run_reports.append(report)
        if spec in PERIODIC_SPECS:
            lane_rows[spec.lane].append(
                (
                    spec.key,
                    run_root / "leaderboard.csv",
                    str(report["model_id"]),
                )
            )

    compare_reports: list[dict[str, Any]] = []
    if args.run_compare:
        compare_root = ROOT / "runs" / "ext0520" / stage_label / "compare"
        compare_root.mkdir(parents=True, exist_ok=True)
        for lane_name, rows in lane_rows.items():
            if len(rows) < 2:
                continue
            t0 = time.perf_counter()
            _run_compare_and_plot(
                python_exe=python_exe,
                lane_name=lane_name,
                row_specs=rows,
                out_root=compare_root,
                dataset_root=PERIODIC_DATASET_ROOT,
            )
            compare_elapsed = float(time.perf_counter() - t0)
            compare_reports.append(
                {
                    "lane": lane_name,
                    "compare_dir": str(compare_root / f"compare_{lane_name}"),
                    "wall_seconds": compare_elapsed,
                }
            )

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": args.stage,
        "stage_label": stage_label,
        "runs": run_reports,
        "compare": compare_reports,
        "n_runs": len(run_reports),
        "n_compare": len(compare_reports),
        "uniform_epochs": uniform_epochs,
        "uniform_batch_size_cases": uniform_batch_size,
        "override_lr": override_lr,
        "override_weight_decay": override_weight_decay,
    }
    out_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
