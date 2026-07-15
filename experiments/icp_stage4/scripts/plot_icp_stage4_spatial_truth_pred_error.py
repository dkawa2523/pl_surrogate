#!/usr/bin/env python3
"""Plot ICP Stage4 truth/prediction/error fields for trained benchmark models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FULL_MODELS = ("deeponet_pod", "unet", "unetpp", "unetpp_attn", "fno", "ffno", "cno")
TARGETS = ("ne", "ni", "Te", "phi")


@dataclass(frozen=True)
class RunSpec:
    label: str
    stage: str
    model_id: str
    recipe: str
    run_dir: Path
    config_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="runs/icp_stage4_struct_model_suite_part_sdf_lite_v1_e80",
        help="Root containing full/ and tuning/ run outputs.",
    )
    parser.add_argument(
        "--full-config-root",
        default="configs/experimental/icp_stage4/generated_struct_model_suite_part_sdf_lite_v1_e80/full",
    )
    parser.add_argument(
        "--tuning-status",
        default="runs/icp_stage4_struct_model_suite_part_sdf_lite_v1_e80/tuning_status.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/icp_stage4_struct_model_suite_part_sdf_lite_v1_e80/summary/spatial_truth_pred_error",
    )
    parser.add_argument("--stages", nargs="+", default=["full", "tuning"], choices=["full", "tuning"])
    parser.add_argument("--full-models", nargs="+", default=list(FULL_MODELS))
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--split", default="interp", help="Evaluation split to plot.")
    parser.add_argument(
        "--case-indices",
        nargs="+",
        type=int,
        default=None,
        help="Indices into the selected split test list. Defaults to all selected-split test cases.",
    )
    parser.add_argument("--no-mask", action="store_true", help="Do not mask non-plasma pixels.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _prepare_imports() -> None:
    root = Path.cwd()
    for path in (root / "src", root):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)


def _safe_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("_") or "case"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _full_specs(run_root: Path, config_root: Path, models: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for model_id in models:
        run_dir = run_root / "full" / model_id
        config_path = config_root / f"benchmark_icp_stage4_core4_full_{model_id}.yaml"
        if not run_dir.exists() or not config_path.exists():
            continue
        specs.append(
            RunSpec(
                label=model_id,
                stage="full",
                model_id=model_id,
                recipe="",
                run_dir=run_dir,
                config_path=config_path,
            )
        )
    return specs


def _tuning_specs(status_path: Path) -> list[RunSpec]:
    if not status_path.exists():
        return []
    specs: list[RunSpec] = []
    for row in _read_csv(status_path):
        if str(row.get("status", "")).strip().lower() != "passed":
            continue
        model_id = str(row.get("model_id", "")).strip()
        recipe = str(row.get("recipe", "")).strip()
        leaderboard = Path(str(row.get("leaderboard", "")).strip())
        config_path = Path(str(row.get("config", "")).strip())
        if not model_id or not recipe or not leaderboard.exists() or not config_path.exists():
            continue
        specs.append(
            RunSpec(
                label=f"{model_id}__{recipe}",
                stage="tuning",
                model_id=model_id,
                recipe=recipe,
                run_dir=leaderboard.parent,
                config_path=config_path,
            )
        )
    return specs


def _metric_mask(dataset: Any) -> np.ndarray:
    root = Path(dataset.geometry_root)
    for path in (root / "geometry" / "mask_plasma.npy", root / "mask_plasma.npy"):
        if path.exists():
            arr = np.asarray(np.load(path), dtype=bool)
            if arr.ndim == 3 and int(arr.shape[0]) == 1:
                arr = arr[0]
            return arr
    raise FileNotFoundError(f"mask_plasma.npy not found under {root}")


def _plot_extent(dataset: Any) -> tuple[float, float, float, float] | None:
    root = Path(dataset.geometry_root)
    r_path = root / "geometry" / "r_coords.npy"
    z_path = root / "geometry" / "z_coords.npy"
    if not r_path.exists():
        r_path = root / "r_coords.npy"
    if not z_path.exists():
        z_path = root / "z_coords.npy"
    if not r_path.exists() or not z_path.exists():
        return None
    r = np.asarray(np.load(r_path), dtype=float)
    z = np.asarray(np.load(z_path), dtype=float)
    return (float(np.nanmin(r)), float(np.nanmax(r)), float(np.nanmin(z)), float(np.nanmax(z)))


def _case_lookup(dataset: Any) -> dict[str, int]:
    return {str(case["case_id"]): idx for idx, case in enumerate(dataset.cases)}


def _case_conditions(dataset: Any, case_idx: int) -> str:
    case = dict(dataset.cases[int(case_idx)])
    parts: list[str] = []
    for key in ("pp", "pp0", "axis", "llcoil", "rrc", "nncoil", "rrce", "zzc"):
        if key in case:
            try:
                parts.append(f"{key}={float(case[key]):g}")
            except Exception:
                parts.append(f"{key}={case[key]}")
    return ", ".join(parts)


def _squeeze_field(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    while out.ndim > 2 and 1 in out.shape:
        out = np.squeeze(out)
    if out.ndim != 2:
        raise ValueError(f"expected a 2D field after squeeze, got {out.shape}")
    return out


def _finite_percentile(arrs: list[np.ndarray], qs: tuple[float, float]) -> tuple[float, float]:
    vals: list[np.ndarray] = []
    for arr in arrs:
        raw = np.asarray(arr, dtype=np.float64)
        finite = raw[np.isfinite(raw)]
        if finite.size:
            vals.append(finite.reshape(-1))
    if not vals:
        return 0.0, 1.0
    all_vals = np.concatenate(vals)
    lo, hi = np.percentile(all_vals, qs)
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or float(lo) == float(hi):
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
    if float(lo) == float(hi):
        lo = float(lo) - 1.0
        hi = float(hi) + 1.0
    return float(lo), float(hi)


def _mask_field(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    raw = np.asarray(arr, dtype=np.float64)
    if mask is None:
        return raw
    return np.where(mask, raw, np.nan)


def _plot_case(
    *,
    truth: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    targets: list[str],
    mask: np.ndarray | None,
    extent: tuple[float, float, float, float] | None,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(len(targets), 3, figsize=(12.2, max(2.8 * len(targets), 3.2)), constrained_layout=True)
    if len(targets) == 1:
        axes = np.asarray([axes])
    for row_idx, target in enumerate(targets):
        truth_arr = _mask_field(_squeeze_field(truth[target]), mask)
        pred_arr = _mask_field(_squeeze_field(pred[target]), mask)
        err_arr = _mask_field(pred_arr - truth_arr, mask)
        vmin, vmax = _finite_percentile([truth_arr, pred_arr], (1.0, 99.0))
        err_abs = np.nanpercentile(np.abs(err_arr), 99.0) if np.isfinite(err_arr).any() else 1.0
        if not math.isfinite(float(err_abs)) or float(err_abs) <= 0.0:
            err_abs = 1.0
        panels = (
            ("truth", truth_arr, "viridis", vmin, vmax),
            ("prediction", pred_arr, "viridis", vmin, vmax),
            ("prediction - truth", err_arr, "coolwarm", -float(err_abs), float(err_abs)),
        )
        for col_idx, (name, arr, cmap, lo, hi) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi, extent=extent, aspect="auto")
            ax.set_title(f"{target} {name}", fontsize=9)
            ax.set_xlabel("r")
            ax.set_ylabel("z")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _target_indices(y_order: list[str], targets: list[str]) -> list[int]:
    missing = [name for name in targets if name not in y_order]
    if missing:
        raise KeyError(f"targets missing from output layout: {missing}; available={y_order}")
    return [y_order.index(name) for name in targets]


def _input_features_cfg(bench: dict[str, Any], model_id: str) -> dict[str, Any]:
    train_cfg = dict(bench.get("train", {})).get(model_id, {})
    if not isinstance(train_cfg, dict):
        return {}
    raw = train_cfg.get("input_features", {})
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _spatial_features_for_indices(
    *,
    bench: dict[str, Any],
    model_id: str,
    bundle: Any,
    h: int,
    w: int,
    indices: np.ndarray,
) -> np.ndarray | None:
    input_cfg = _input_features_cfg(bench, model_id)
    if str(input_cfg.get("mode", "")).strip().lower() != "geom_feature_pack":
        return None

    from plasma_surrogate.preprocessing.spatial_features import (
        apply_coord_feature_scaling,
        apply_distance_transform,
        build_case_spatial_features,
        build_coord_feature_rows,
        materialize_case_spatial_batch,
        resolve_coord_feature_channels,
        resolve_distance_transform_cfg,
        resolve_distance_transform_effective,
    )

    channels = resolve_coord_feature_channels(input_cfg.get("features"))
    distance_cfg = resolve_distance_transform_cfg(dict(input_cfg.get("distance_transform") or {}))
    distance_cfg, _ = resolve_distance_transform_effective(
        distance_cfg,
        stats=dict(bundle.transforms.get("distance_transform_stats", {})),
    )
    source, _ = build_case_spatial_features(
        channels=channels,
        h=h,
        w=w,
        static_pack=bundle.schemas.get("static_spatial_feature_pack"),
        case_pack=bundle.schemas.get("case_structure_feature_pack"),
        distance_transform_cfg=distance_cfg,
        coord_feature_scaler_artifact=dict(bundle.transforms.get("coord_feature_scaler", {})),
    )
    if source is not None:
        return materialize_case_spatial_batch(source, np.asarray(indices, dtype=np.int64))

    rows, _ = build_coord_feature_rows(
        channels=channels,
        pack=bundle.schemas.get("coord_feature_pack"),
        geom_ctx=None,
        h=h,
        w=w,
    )
    rows, _ = apply_distance_transform(rows.astype(np.float32), channels=channels, cfg=distance_cfg)
    rows, _, _ = apply_coord_feature_scaling(
        rows.astype(np.float32),
        channels=channels,
        coord_feature_scaler_artifact=dict(bundle.transforms.get("coord_feature_scaler", {})),
    )
    return np.repeat(rows.reshape(1, h, w, len(channels)).astype(np.float32), len(indices), axis=0)


def _augment_condition_for_model(
    *,
    model_id: str,
    cond_scaled: np.ndarray,
    bundle: Any,
) -> np.ndarray:
    if str(model_id).strip().lower() != "deeponet_pod":
        return np.asarray(cond_scaled, dtype=np.float32)

    from plasma_surrogate.train.deeponet_contracts import resolve_pod_descriptor_and_latent_contract

    report = dict(bundle.schemas.get("preprocess_report", {}) or {})
    descriptor_vec, _ = resolve_pod_descriptor_and_latent_contract(
        input_mode=str(report.get("input_mode_effective", "table_plus_structure")),
        adapter_mode=str(report.get("structure_adapter_mode_effective", "descriptor_branch")),
        descriptor_profile=str(
            report.get(
                "structure_descriptor_profile_effective",
                report.get("descriptor_profile", "none"),
            )
        ),
        latent_profile=str(
            report.get(
                "structure_latent_profile_effective",
                report.get("latent_profile", "none"),
            )
        ),
        descriptor_pack=bundle.schemas.get("structure_descriptor_pack"),
        latent_pack=bundle.schemas.get("latent_feature_pack"),
    )
    out = np.asarray(cond_scaled, dtype=np.float32)
    if descriptor_vec is None:
        return out
    desc = np.repeat(np.asarray(descriptor_vec, dtype=np.float32).reshape(1, -1), out.shape[0], axis=0)
    return np.concatenate([out, desc], axis=1).astype(np.float32)


def _predict_case_batch(
    *,
    model_obj: Any,
    cond_scaled: np.ndarray,
    spatial_features: np.ndarray | None,
    targets: list[str],
) -> dict[str, np.ndarray]:
    if spatial_features is None:
        raw = model_obj.predict_fields(cond_scaled)
    else:
        raw = model_obj.predict_fields(cond_scaled, spatial_features=spatial_features)
    return {name: np.asarray(raw[name], dtype=np.float32) for name in targets if name in raw}


def _plot_run(spec: RunSpec, *, split: str, case_indices: list[int] | None, targets: list[str], out_dir: Path, mask_nonplasma: bool, dpi: int) -> list[dict[str, str]]:
    from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
    from plasma_surrogate.core.dataset_io import load_dataset
    from plasma_surrogate.core.run_bundle import RunBundleLoader
    from plasma_surrogate.models.checkpoint import load_checkpoint

    cfg = _load_yaml(spec.config_path)
    bench = dict(cfg.get("benchmark", {}))
    dataset = load_dataset({"dataset": bench["dataset"]}, spec.run_dir)
    bundle = RunBundleLoader.load(spec.run_dir)
    transforms = bundle.transform_bundle()
    y_order = [str(v) for v in bundle.schemas.get("output_layout", {}).get("vars", TARGETS)]
    y_target_indices = _target_indices(y_order, targets)
    cond = build_cond_matrix_with_axis(dataset.cases, bundle.cond_schema_obj(), bundle.axis_schema_obj())
    cond_scaled = transforms.transform_cond(cond)
    cond_scaled = _augment_condition_for_model(model_id=spec.model_id, cond_scaled=cond_scaled, bundle=bundle)
    y = np.stack(
        [np.stack([case["y"][name] for name in y_order], axis=0).astype(np.float32) for case in dataset.cases],
        axis=0,
    )
    case_id_to_idx = _case_lookup(dataset)
    split_path = spec.run_dir / "preprocessing" / "split" / f"split_{split}_v1.json"
    if not split_path.exists() and split == "interp":
        split_path = spec.run_dir / "preprocessing" / "split" / "split_interp_marginal_v1.json"
    split_payload = _read_json(split_path)
    test_ids = [str(v) for v in split_payload.get("test", [])]
    selected_case_positions = list(range(len(test_ids))) if case_indices is None else [int(v) for v in case_indices]
    selected_case_ids = [test_ids[pos] for pos in selected_case_positions if 0 <= pos < len(test_ids)]
    dataset_indices = np.asarray([case_id_to_idx[cid] for cid in selected_case_ids], dtype=np.int64)
    if dataset_indices.size == 0:
        return []

    ckpt = spec.run_dir / "models" / spec.model_id / "eval_protocol" / split / "checkpoints"
    if not ckpt.exists() and split == "interp":
        ckpt = spec.run_dir / "models" / spec.model_id / "eval_protocol" / "interp" / "checkpoints"
    model_obj = load_checkpoint(ckpt)
    h, w = int(dataset.shape[0]), int(dataset.shape[1])
    spatial = _spatial_features_for_indices(
        bench=bench,
        model_id=spec.model_id,
        bundle=bundle,
        h=h,
        w=w,
        indices=dataset_indices,
    )
    mask = None if not mask_nonplasma else _metric_mask(dataset)
    extent = _plot_extent(dataset)

    records: list[dict[str, str]] = []
    # Keep prediction batches tiny so plotting does not need large GPU memory.
    for local_idx, case_id in enumerate(selected_case_ids):
        ds_idx = int(dataset_indices[local_idx])
        cond_one = cond_scaled[ds_idx : ds_idx + 1]
        spatial_one = None if spatial is None else spatial[local_idx : local_idx + 1]
        pred_scaled = _predict_case_batch(
            model_obj=model_obj,
            cond_scaled=cond_one,
            spatial_features=spatial_one,
            targets=targets,
        )
        pred_phys_all = transforms.inverse_field_dict(pred_scaled)
        pred_phys = {name: np.asarray(pred_phys_all[name], dtype=np.float32)[0] for name in targets}
        truth = {
            name: np.asarray(y[ds_idx, y_target_indices[i]], dtype=np.float32)
            for i, name in enumerate(targets)
        }
        title = f"{spec.stage}/{spec.label} | {split} test | {case_id} | {_case_conditions(dataset, ds_idx)}"
        out_path = out_dir / spec.stage / spec.label / f"{_safe_name(case_id)}_truth_pred_error.png"
        _plot_case(
            truth=truth,
            pred=pred_phys,
            targets=targets,
            mask=mask,
            extent=extent,
            title=title,
            out_path=out_path,
            dpi=dpi,
        )
        print(out_path)
        records.append(
            {
                "stage": spec.stage,
                "model_id": spec.model_id,
                "recipe": spec.recipe,
                "label": spec.label,
                "split": split,
                "case_id": case_id,
                "case_index": str(selected_case_positions[local_idx]),
                "dataset_index": str(ds_idx),
                "conditions": _case_conditions(dataset, ds_idx),
                "plot": str(out_path),
                "run_dir": str(spec.run_dir),
                "config": str(spec.config_path),
            }
        )
    return records


def _write_manifest(out_dir: Path, rows: list[dict[str, str]]) -> Path:
    manifest = out_dir / "spatial_truth_pred_error_manifest.csv"
    fieldnames = [
        "stage",
        "model_id",
        "recipe",
        "label",
        "split",
        "case_id",
        "case_index",
        "dataset_index",
        "conditions",
        "plot",
        "run_dir",
        "config",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _write_index(out_dir: Path, rows: list[dict[str, str]]) -> Path:
    lines = [
        "# ICP Stage4 Spatial Truth/Prediction/Error",
        "",
        "Each plot shows truth, prediction, and prediction - truth for ne, ni, Te, and phi.",
        "Non-plasma pixels are masked unless the script is run with `--no-mask`.",
        "",
        "| Stage | Model | Recipe | Split | Case | Plot |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        rel = Path(row["plot"]).relative_to(out_dir).as_posix()
        recipe = row["recipe"] or "-"
        lines.append(
            f"| {row['stage']} | {row['model_id']} | {recipe} | {row['split']} | "
            f"{row['case_id']} | [{Path(rel).name}]({rel}) |"
        )
    index = out_dir / "index.md"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main() -> int:
    args = _parse_args()
    _prepare_imports()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    targets = [str(v) for v in args.targets]
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"unknown target(s): {unknown}; expected subset of {TARGETS}")

    specs: list[RunSpec] = []
    if "full" in args.stages:
        specs.extend(_full_specs(run_root, Path(args.full_config_root), [str(v) for v in args.full_models]))
    if "tuning" in args.stages:
        specs.extend(_tuning_specs(Path(args.tuning_status)))
    if not specs:
        raise RuntimeError("no runnable model specs found")

    all_rows: list[dict[str, str]] = []
    for spec in specs:
        try:
            all_rows.extend(
                _plot_run(
                    spec,
                    split=str(args.split),
                    case_indices=args.case_indices,
                    targets=targets,
                    out_dir=out_dir,
                    mask_nonplasma=not bool(args.no_mask),
                    dpi=int(args.dpi),
                )
            )
        except Exception as exc:
            print(f"[FAILED] {spec.stage}/{spec.label}: {exc!r}", file=sys.stderr)
            raise

    manifest = _write_manifest(out_dir, all_rows)
    index = _write_index(out_dir, all_rows)
    print(manifest)
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
