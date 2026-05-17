from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from plasma_surrogate.benchmark.runtime_context import resolve_effective_benchmark_cfg
from plasma_surrogate.core.input_modes import extract_input_mode_metadata, resolve_benchmark_runtime_controls
from plasma_surrogate.core.run_bundle import RunBundleLoader
from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.infer.engine_builder import build_inference_engine
from plasma_surrogate.models.checkpoint import load_checkpoint


PART_IDS = tuple(f"coil_{idx:02d}" for idx in range(1, 7))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICP part-SDF-lite shape optimization.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="Benchmark model run dir, e.g. runs/.../full/unet")
    parser.add_argument("--model", default="unet", choices=("unet", "ffno"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-case-id", default="case_g002_op01")
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--uniformity-target", default="ne")
    parser.add_argument("--uniformity-region", default="plasma")
    parser.add_argument("--uniformity-score-mode", choices=("relative", "cv"), default="relative")
    parser.add_argument("--mid-height-band-px", type=int, default=0)
    parser.add_argument("--space-mode", choices=("layout", "transform"), default="transform")
    parser.add_argument(
        "--condition-mode",
        choices=("fixed", "train_bounds", "dataset_bounds"),
        default="fixed",
        help="fixed keeps pp/pp0 at the reference case; train_bounds optimizes within preprocessing cond_stats.",
    )
    parser.add_argument(
        "--coil-layout-root",
        default="data/outputs_icp_stage4_enriched_360/structure/coil_layout",
        help="Directory containing {case_id}__coil_layout.csv for layout-space optimization.",
    )
    parser.add_argument("--layout-center-span", type=float, default=0.45)
    parser.add_argument("--layout-size-min-scale", type=float, default=0.75)
    parser.add_argument("--layout-size-max-scale", type=float, default=1.25)
    return parser.parse_args()


def _load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return resolve_effective_benchmark_cfg(raw)


def _reference_cond(dataset_root: Path, case_id: str) -> dict[str, float]:
    with (dataset_root / "index.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("case_id")) == case_id:
                return {"pp": float(row["pp"]), "pp0": float(row["pp0"])}
    raise KeyError(f"reference case not found in index.csv: {case_id}")


def _dataset_cond_bounds(dataset_root: Path) -> dict[str, tuple[float, float]]:
    vals: dict[str, list[float]] = {"pp": [], "pp0": []}
    with (dataset_root / "index.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for key in vals:
                vals[key].append(float(row[key]))
    return {key: (float(min(items)), float(max(items))) for key, items in vals.items()}


def _condition_space(
    *,
    mode: str,
    reference_cond: dict[str, float],
    cond_stats: dict[str, Any],
    dataset_root: Path,
) -> dict[str, tuple[float, float]]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "fixed":
        return {key: (float(value), float(value)) for key, value in reference_cond.items()}
    if mode_norm == "dataset_bounds":
        return _dataset_cond_bounds(dataset_root)
    if mode_norm == "train_bounds":
        out: dict[str, tuple[float, float]] = {}
        for key in ("pp", "pp0"):
            stats = dict(cond_stats.get(key, {}))
            out[key] = (float(stats["min"]), float(stats["max"]))
        return out
    raise ValueError(f"unsupported condition mode: {mode}")


def _read_reference_layout(coil_layout_root: Path, case_id: str) -> list[dict[str, float | str]]:
    path = coil_layout_root / f"{case_id}__coil_layout.csv"
    if not path.exists():
        raise FileNotFoundError(f"coil layout csv not found: {path}")
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            active = int(float(row.get("active", 1)))
            if active <= 0:
                continue
            idx = int(float(row["coil_index"]))
            rows.append(
                {
                    "part_id": f"coil_{idx:02d}",
                    "r_center": float(row["r_center"]),
                    "z_center": float(row["z_center"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                }
            )
    rows.sort(key=lambda r: str(r["part_id"]))
    if len(rows) == 0:
        raise ValueError(f"no active coils in layout csv: {path}")
    return rows


def _layout_param_specs(
    layout_rows: list[dict[str, float | str]],
    *,
    center_span: float = 0.45,
    size_min_scale: float = 0.75,
    size_max_scale: float = 1.25,
) -> dict[str, dict[str, float]]:
    specs: dict[str, dict[str, float]] = {}
    span = max(0.0, float(center_span))
    min_scale = max(1.0e-6, float(size_min_scale))
    max_scale = max(min_scale, float(size_max_scale))
    for row in layout_rows:
        part_id = str(row["part_id"])
        r_center = float(row["r_center"])
        z_center = float(row["z_center"])
        width = float(row["width"])
        height = float(row["height"])
        specs[f"layout.{part_id}.r_center"] = {"default": r_center, "min": r_center - span, "max": r_center + span}
        specs[f"layout.{part_id}.z_center"] = {"default": z_center, "min": z_center - span, "max": z_center + span}
        specs[f"layout.{part_id}.width"] = {"default": width, "min": min_scale * width, "max": max_scale * width}
        specs[f"layout.{part_id}.height"] = {"default": height, "min": min_scale * height, "max": max_scale * height}
    return specs


def _prepare_layout_dataset_overlay(
    dataset_root: Path,
    out_dir: Path,
    case_id: str,
    coil_layout_root: Path,
    *,
    center_span: float = 0.45,
    size_min_scale: float = 0.75,
    size_max_scale: float = 1.25,
) -> Path:
    layout_rows = _read_reference_layout(coil_layout_root, case_id)
    overlay_root = out_dir / "_layout_dataset"
    overlay_geom = overlay_root / "geometry"
    overlay_geom.mkdir(parents=True, exist_ok=True)
    src_geom = dataset_root / "geometry"
    for src in src_geom.iterdir():
        if src.is_file():
            shutil.copy2(src, overlay_geom / src.name)
    manifest = {
        "part_ids": [str(row["part_id"]) for row in layout_rows],
        "plasma_mode": "preserve",
        "reference_case_id": case_id,
        "optimization_space": "coil_layout_rectangles_v1",
        "layout_center_span": float(center_span),
        "layout_size_min_scale": float(size_min_scale),
        "layout_size_max_scale": float(size_max_scale),
        "param_specs": _layout_param_specs(
            layout_rows,
            center_span=center_span,
            size_min_scale=size_min_scale,
            size_max_scale=size_max_scale,
        ),
    }
    (overlay_geom / "parts_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return overlay_root


def _default_geom_params(space_mode: str, layout_specs: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    if space_mode == "layout":
        return {key: float(spec["default"]) for key, spec in sorted(dict(layout_specs or {}).items())}
    out: dict[str, float] = {}
    for part_id in PART_IDS:
        out[f"part.{part_id}.tx"] = 0.0
        out[f"part.{part_id}.ty"] = 0.0
        out[f"part.{part_id}.scale_x"] = 1.0
        out[f"part.{part_id}.scale_y"] = 1.0
    return out


def _geom_space(space_mode: str, layout_specs: dict[str, dict[str, float]] | None = None) -> dict[str, tuple[float, float]]:
    if space_mode == "layout":
        return {
            key: (float(spec["min"]), float(spec["max"]))
            for key, spec in sorted(dict(layout_specs or {}).items())
        }
    out: dict[str, tuple[float, float]] = {}
    for part_id in PART_IDS:
        out[f"part.{part_id}.tx"] = (-0.02, 0.02)
        out[f"part.{part_id}.ty"] = (-0.02, 0.02)
        out[f"part.{part_id}.scale_x"] = (0.9, 1.1)
        out[f"part.{part_id}.scale_y"] = (0.9, 1.1)
    return out


def _field2d(result: Any, key: str) -> np.ndarray:
    arr = np.asarray(result.fields_phys[key], dtype=np.float32)
    return np.squeeze(arr)


def _plot_base_best(base: Any, best: Any, out_dir: Path, key: str) -> None:
    base_field = _field2d(base, key)
    best_field = _field2d(best, key)
    diff = best_field - base_field
    vmin = float(np.nanmin([np.nanmin(base_field), np.nanmin(best_field)]))
    vmax = float(np.nanmax([np.nanmax(base_field), np.nanmax(best_field)]))
    dmax = float(np.nanmax(np.abs(diff)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(base_field, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"base {key}")
    im1 = axes[1].imshow(best_field, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"best {key}")
    im2 = axes[2].imshow(diff, origin="lower", cmap="coolwarm", vmin=-dmax, vmax=dmax)
    axes[2].set_title("best - base")
    fig.colorbar(im0, ax=axes[:2], shrink=0.82)
    fig.colorbar(im2, ax=axes[2], shrink=0.82)
    fig.savefig(out_dir / f"base_vs_best_{key}.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    engine_dir = out_dir / "_engine"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(config_path)
    dataset_root = Path(cfg["dataset"]["root"])
    layout_specs: dict[str, dict[str, float]] | None = None
    provider_dataset_root = dataset_root
    if args.space_mode == "layout":
        overlay_root = _prepare_layout_dataset_overlay(
            dataset_root=dataset_root,
            out_dir=out_dir,
            case_id=str(args.reference_case_id),
            coil_layout_root=Path(args.coil_layout_root),
            center_span=float(args.layout_center_span),
            size_min_scale=float(args.layout_size_min_scale),
            size_max_scale=float(args.layout_size_max_scale),
        )
        provider_dataset_root = overlay_root
        manifest = json.loads((overlay_root / "geometry" / "parts_manifest.json").read_text(encoding="utf-8"))
        layout_specs = {str(k): dict(v) for k, v in dict(manifest["param_specs"]).items()}
    model = load_checkpoint(run_dir / "models" / args.model / "eval_protocol" / "extrap" / "checkpoints")
    bundle = RunBundleLoader.load(run_dir, model=model)
    checkpoint_meta_path = run_dir / "models" / args.model / "eval_protocol" / "extrap" / "checkpoints" / "meta.json"
    checkpoint_meta = json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
    input_mode_meta = extract_input_mode_metadata(checkpoint_meta)
    strict_input_mode, allow_mode_fallback = resolve_benchmark_runtime_controls(cfg)
    geometry_provider = build_geometry_provider(provider_dataset_root, provider_mode="parametric_parts")

    engine = build_inference_engine(
        model=model,
        cond_schema=bundle.cond_schema_obj(),
        axis_schema=bundle.axis_schema_obj(),
        geometry_provider=geometry_provider,
        output_dir=engine_dir,
        transform_bundle=bundle.transform_bundle(),
        cond_stats=bundle.schemas.get("cond_stats", {}),
        phi_mode=str(cfg.get("phi_mode", "direct")),
        phi_hybrid_steps=int(cfg.get("phi_hybrid_steps", 1)),
        poisson_refine_iters=int(cfg.get("poisson_refine_iters", 0)),
        ood_cfg={
            "poisson_residual_limit": 1e2,
            "uniformity_target": str(args.uniformity_target),
            "uniformity_region": str(args.uniformity_region),
            "uniformity_score_mode": str(args.uniformity_score_mode),
            "mid_height_band_px": int(args.mid_height_band_px),
        },
        feature_store=None,
        coord_scaler=bundle.transforms.get("coord_scaler", {}),
        coord_feature_scaler=bundle.transforms.get("coord_feature_scaler", {}),
        coord_feature_pack=None,
        coord_distance_transform_stats=bundle.transforms.get("distance_transform_stats", {}),
        input_mode_meta=input_mode_meta,
        strict_input_mode=strict_input_mode,
        allow_mode_fallback=allow_mode_fallback,
        checkpoint_meta_path=checkpoint_meta_path,
        grid_input_features_cfg=dict(cfg.get("train", {}).get(args.model, {}).get("input_features", {})),
    )

    cond = _reference_cond(dataset_root, args.reference_case_id)
    space = _condition_space(
        mode=str(args.condition_mode),
        reference_cond=cond,
        cond_stats=dict(bundle.schemas.get("cond_stats", {})),
        dataset_root=dataset_root,
    )
    axis = {"mode": "steady", "value": 0.0}
    base_geom = {"geom_id": "default", "geom_param": _default_geom_params(str(args.space_mode), layout_specs)}

    base = engine.single_run_aggregated(cond=cond, geom=base_geom, axis=axis)
    best = engine.optimize_run(
        space=space,
        geom_space=_geom_space(str(args.space_mode), layout_specs),
        n_trials=int(args.n_trials),
        geom={"geom_id": "default"},
        axis=axis,
        seed=int(args.seed),
        backend="random",
    )
    best_geom = {"geom_id": "default", "geom_param": dict(best["best_geom_param"])}
    best_result = engine.single_run_aggregated(cond=dict(best["best_cond"]), geom=best_geom, axis=axis)

    for name in ("summary.json", "best.json"):
        shutil.copy2(engine_dir / "optimize" / name, out_dir / name)
    shutil.copy2(engine_dir / "optimize" / "trials.csv", out_dir / "trials.csv")
    (out_dir / "base_qoi.json").write_text(json.dumps(base.qoi, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "best_qoi.json").write_text(json.dumps(best_result.qoi, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "fixed_condition.json").write_text(
        json.dumps(
            {
                "reference_case_id": args.reference_case_id,
                "cond": cond,
                "condition_mode": str(args.condition_mode),
                "condition_space": {key: [float(v[0]), float(v[1])] for key, v in sorted(space.items())},
                "axis": axis,
                "uniformity_target": str(args.uniformity_target),
                "uniformity_region": str(args.uniformity_region),
                "uniformity_score_mode": str(args.uniformity_score_mode),
                "mid_height_band_px": int(args.mid_height_band_px),
                "space_mode": str(args.space_mode),
                "layout_center_span": float(args.layout_center_span),
                "layout_size_min_scale": float(args.layout_size_min_scale),
                "layout_size_max_scale": float(args.layout_size_max_scale),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        out_dir / "base_best_fields.npz",
        base_ne=_field2d(base, "ne"),
        best_ne=_field2d(best_result, "ne"),
        base_ni=_field2d(base, "ni"),
        best_ni=_field2d(best_result, "ni"),
    )
    _plot_base_best(base, best_result, out_dir, "ne")
    _plot_base_best(base, best_result, out_dir, "ni")
    print(json.dumps({"out_dir": str(out_dir), "best_value": best["best_value"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
