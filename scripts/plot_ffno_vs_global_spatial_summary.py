from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from plasma_surrogate.core.dataset_io import load_dataset
from plasma_surrogate.core.run_bundle import RunBundleLoader
from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.mlp.io import load_mlp_checkpoint


DISPLAY_VARS = ("ne", "ni", "Te", "phi")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data or {})


def _load_case_ids(split_path: Path) -> list[str]:
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [str(v) for v in payload.get("test", [])]


def _make_engine(
    *,
    run_root: Path,
    split: str,
    model_name: str,
    temp_root: Path,
    run_cfg: dict[str, Any],
    geometry_root: Path,
) -> InferenceEngine:
    ckpt_dir = run_root / "models" / model_name / "eval_protocol" / split / "checkpoints"
    model = load_mlp_checkpoint(ckpt_dir)
    bundle = RunBundleLoader.load(run_root, model=model)
    cond_schema = bundle.cond_schema_obj()
    axis_schema = bundle.axis_schema_obj()
    transforms = bundle.transform_bundle()
    cond_stats = dict(bundle.schemas.get("cond_stats", {}))
    geom_provider = FixedGeometryProvider(geometry_root)
    output_dir = temp_root / model_name / split
    coord_feature_pack = bundle.schemas.get("coord_feature_pack")
    coord_pack_meta = dict(bundle.schemas.get("coord_feature_pack_meta", {}))
    coord_scaler = dict(bundle.transforms.get("coord_scaler", {}))
    coord_feature_scaler = dict(bundle.transforms.get("coord_feature_scaler", {}))
    coord_distance_stats = dict(bundle.transforms.get("distance_transform_stats", {}))
    train_cfg = dict(run_cfg.get("benchmark", run_cfg).get("train", run_cfg.get("train", {})))
    per_model_cfg = dict(train_cfg.get(model_name, {}))
    input_features_cfg = dict(per_model_cfg.get("input_features", {}))
    input_scaling_cfg = (
        dict(input_features_cfg.get("scale_using_train_stats", {}))
        or dict(input_features_cfg.get("input_scaling", {}))
    )
    return InferenceEngine(
        model=model,
        cond_schema=cond_schema,
        axis_schema=axis_schema,
        geometry_provider=geom_provider,
        output_dir=output_dir,
        transform_bundle=transforms,
        cond_stats=cond_stats,
        phi_mode=str(run_cfg.get("benchmark", run_cfg).get("phi_mode", "direct")),
        feature_store=bundle.geometry_store,
        coord_scaler=coord_scaler,
        coord_feature_scaler=coord_feature_scaler,
        coord_feature_pack=coord_feature_pack,
        coord_distance_transform_stats=coord_distance_stats,
        coord_input_scaling_cfg=input_scaling_cfg,
        coord_input_features_cfg=input_features_cfg,
        grid_input_features_cfg=input_features_cfg,
    )


def _display_field(var_name: str, arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if var_name in {"ne", "ni"}:
        return np.log10(np.maximum(a, 1.0))
    return a


def _build_summary_fields(
    *,
    dataset_cases: dict[str, dict[str, Any]],
    case_ids: list[str],
    ffno_engine: InferenceEngine,
    global_engine: InferenceEngine,
) -> dict[str, dict[str, np.ndarray]]:
    truth_acc = {name: [] for name in DISPLAY_VARS}
    global_acc = {name: [] for name in DISPLAY_VARS}
    ffno_acc = {name: [] for name in DISPLAY_VARS}
    for case_id in case_ids:
        case = dataset_cases[case_id]
        cond = {str(k): float(v) for k, v in dict(case["cond"]).items()}
        axis = {"mode": "steady", "value": float(case["axis"])}
        geom = {"geom_id": "default"}
        ffno_res = ffno_engine.single_run(cond=cond, geom=geom, axis=axis)
        global_res = global_engine.single_run(cond=cond, geom=geom, axis=axis)
        for var_name in DISPLAY_VARS:
            truth_acc[var_name].append(_display_field(var_name, np.asarray(case["y"][var_name], dtype=np.float32)))
            global_acc[var_name].append(
                _display_field(var_name, np.asarray(global_res.fields_phys[var_name], dtype=np.float32).squeeze())
            )
            ffno_acc[var_name].append(
                _display_field(var_name, np.asarray(ffno_res.fields_phys[var_name], dtype=np.float32).squeeze())
            )
    return {
        "truth": {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in truth_acc.items()},
        "global": {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in global_acc.items()},
        "ffno": {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in ffno_acc.items()},
    }


def _make_plot(
    *,
    summary: dict[str, dict[str, dict[str, np.ndarray]]],
    ffno_row: dict[str, str],
    global_row: dict[str, str],
    out_path: Path,
) -> None:
    col_specs = [
        ("interp", "truth", "GT\ninterp"),
        ("interp", "global", "GLOBAL-MLP\ninterp"),
        ("interp", "ffno", "FFNO\ninterp"),
        ("interp", "global_err", "|GLOBAL-GT|\ninterp"),
        ("interp", "ffno_err", "|FFNO-GT|\ninterp"),
        ("extrap", "truth", "GT\nextrap"),
        ("extrap", "global", "GLOBAL-MLP\nextrap"),
        ("extrap", "ffno", "FFNO\nextrap"),
        ("extrap", "global_err", "|GLOBAL-GT|\nextrap"),
        ("extrap", "ffno_err", "|FFNO-GT|\nextrap"),
    ]
    fig, axes = plt.subplots(nrows=len(DISPLAY_VARS), ncols=len(col_specs), figsize=(30, 13), constrained_layout=True)
    for col_idx, (_, _, title) in enumerate(col_specs):
        axes[0, col_idx].set_title(title, fontsize=11)

    for row_idx, var_name in enumerate(DISPLAY_VARS):
        field_samples = []
        error_samples = []
        for split_name in ("interp", "extrap"):
            field_samples.extend(
                [
                    summary[split_name]["truth"][var_name],
                    summary[split_name]["global"][var_name],
                    summary[split_name]["ffno"][var_name],
                ]
            )
            error_samples.extend(
                [
                    np.abs(summary[split_name]["global"][var_name] - summary[split_name]["truth"][var_name]),
                    np.abs(summary[split_name]["ffno"][var_name] - summary[split_name]["truth"][var_name]),
                ]
            )
        field_concat = np.concatenate([x.ravel() for x in field_samples if np.isfinite(x).any()])
        field_lo, field_hi = np.percentile(field_concat[np.isfinite(field_concat)], [2, 98])
        if np.isclose(field_lo, field_hi):
            field_lo -= 1.0
            field_hi += 1.0
        err_concat = np.concatenate([x.ravel() for x in error_samples if np.isfinite(x).any()])
        err_hi = float(np.percentile(err_concat[np.isfinite(err_concat)], 98))
        if not np.isfinite(err_hi) or err_hi <= 0.0:
            err_hi = 1.0

        field_axes = []
        err_axes = []
        field_im = None
        err_im = None
        for col_idx, (split_name, mode, _) in enumerate(col_specs):
            ax = axes[row_idx, col_idx]
            truth = summary[split_name]["truth"][var_name]
            global_pred = summary[split_name]["global"][var_name]
            ffno_pred = summary[split_name]["ffno"][var_name]
            if mode == "truth":
                img = truth
                field_im = ax.imshow(img, origin="lower", cmap="cividis", vmin=field_lo, vmax=field_hi)
                field_axes.append(ax)
            elif mode == "global":
                img = global_pred
                field_im = ax.imshow(img, origin="lower", cmap="cividis", vmin=field_lo, vmax=field_hi)
                field_axes.append(ax)
            elif mode == "ffno":
                img = ffno_pred
                field_im = ax.imshow(img, origin="lower", cmap="cividis", vmin=field_lo, vmax=field_hi)
                field_axes.append(ax)
            elif mode == "global_err":
                img = np.abs(global_pred - truth)
                err_im = ax.imshow(img, origin="lower", cmap="magma", vmin=0.0, vmax=err_hi)
                err_axes.append(ax)
            else:
                img = np.abs(ffno_pred - truth)
                err_im = ax.imshow(img, origin="lower", cmap="magma", vmin=0.0, vmax=err_hi)
                err_axes.append(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ylabel = f"{var_name}"
                if var_name in {"ne", "ni"}:
                    ylabel += "\nlog10 scale"
                axes[row_idx, col_idx].set_ylabel(ylabel, fontsize=11)
        if field_im is not None:
            fig.colorbar(field_im, ax=field_axes, fraction=0.018, pad=0.01)
        if err_im is not None:
            fig.colorbar(err_im, ax=err_axes, fraction=0.018, pad=0.01)

    fig.suptitle(
        "FFNO vs GLOBAL-MLP Spatial Summary with Ground Truth\n"
        f"Primary metric: FFNO={ffno_row.get('primary_metric_value', '')} | "
        f"GLOBAL-MLP={global_row.get('primary_metric_value', '')}",
        fontsize=16,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ffno-run",
        default=r"runs/periodic_real_tuned_v83/benchmark_m7_ffno_isolated_mainline",
    )
    parser.add_argument(
        "--global-run",
        default=r"runs/periodic_real_tuned_v48/benchmark_m7_global_frozen_ref_v2",
    )
    parser.add_argument(
        "--ffno-config",
        default=r"tests/fixtures/benchmark_periodic_real_m7_ffno_isolated_mainline.yaml",
    )
    parser.add_argument(
        "--global-config",
        default=r"tests/fixtures/benchmark_periodic_real_m7_global_frozen_ref.yaml",
    )
    parser.add_argument(
        "--output",
        default=r"runs/periodic_real_tuned_v83/benchmark_m7_ffno_isolated_mainline/analysis/ffno_vs_global_mlp_spatial_summary_dual_with_gt.png",
    )
    args = parser.parse_args()

    ffno_root = Path(args.ffno_run)
    global_root = Path(args.global_run)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".json")
    temp_root = out_path.parent / "_tmp_compare_infer"

    ffno_cfg = _load_yaml(Path(args.ffno_config))
    global_cfg = _load_yaml(Path(args.global_config))
    dataset = load_dataset(ffno_cfg.get("benchmark", {}), run_dir=Path("."))
    dataset_cases = {str(case["case_id"]): case for case in dataset.cases}
    geometry_root = Path(dataset.geometry_root)

    ffno_row = None
    with (ffno_root / "leaderboard.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["model_id"] == "ffno":
                ffno_row = row
                break
    global_row = None
    with (global_root / "leaderboard.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["model_id"] == "global_mlp":
                global_row = row
                break
    if ffno_row is None or global_row is None:
        raise RuntimeError("leaderboard rows for ffno/global_mlp not found")

    split_map = {
        "interp": "split_interp_marginal_v1.json",
        "extrap": "split_extrap_v1.json",
    }
    summary_payload: dict[str, Any] = {
        "ffno_run": str(ffno_root),
        "global_run": str(global_root),
        "output": str(out_path),
        "splits": {},
    }
    summary_fields: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for split_name, split_file in split_map.items():
        ffno_cases = set(_load_case_ids(ffno_root / "preprocessing" / "split" / split_file))
        global_cases = set(_load_case_ids(global_root / "preprocessing" / "split" / split_file))
        case_ids = sorted(ffno_cases & global_cases)
        if not case_ids:
            raise RuntimeError(f"No common case_ids for split={split_name}")
        ffno_engine = _make_engine(
            run_root=ffno_root,
            split=split_name,
            model_name="ffno",
            temp_root=temp_root,
            run_cfg=ffno_cfg,
            geometry_root=geometry_root,
        )
        global_engine = _make_engine(
            run_root=global_root,
            split=split_name,
            model_name="global_mlp",
            temp_root=temp_root,
            run_cfg=global_cfg,
            geometry_root=geometry_root,
        )
        summary_fields[split_name] = _build_summary_fields(
            dataset_cases=dataset_cases,
            case_ids=case_ids,
            ffno_engine=ffno_engine,
            global_engine=global_engine,
        )
        summary_payload["splits"][split_name] = {
            "n_common_cases": len(case_ids),
            "case_ids": case_ids,
        }

    _make_plot(summary=summary_fields, ffno_row=ffno_row, global_row=global_row, out_path=out_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(out_path)
    print(meta_path)


if __name__ == "__main__":
    main()
