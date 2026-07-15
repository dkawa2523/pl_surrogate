from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "src"))
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

from plasma_surrogate.benchmark.runtime_context import resolve_effective_benchmark_cfg
from plasma_surrogate.core.input_modes import extract_input_mode_metadata
from plasma_surrogate.core.run_bundle import RunBundleLoader
from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.infer.engine_builder import build_inference_engine
from plasma_surrogate.models.checkpoint import load_checkpoint


PART_IDS = tuple(f"coil_{idx:02d}" for idx in range(1, 7))


def _cfg_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def resolve_benchmark_runtime_controls(cfg: dict[str, Any]) -> tuple[str, bool]:
    """Archive-local compatibility shim for historical ICP configs."""

    root = dict(cfg or {})
    benchmark = dict(root.get("benchmark", root) or {})
    strict = str(benchmark.get("strict_input_mode", root.get("strict_input_mode", "error"))).strip().lower()
    if strict not in {"error", "warn", "off"}:
        strict = "error"
    allow_fallback = _cfg_bool(
        benchmark.get("allow_mode_fallback", root.get("allow_mode_fallback", False)),
        default=False,
    )
    return strict, allow_fallback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICP part-SDF-lite shape optimization.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="Benchmark model run dir, e.g. runs/.../full/unet")
    parser.add_argument("--model", default="ffno", choices=("unet", "ffno", "u_no", "cno", "cno_operator_unet"))
    parser.add_argument(
        "--eval-protocol",
        choices=("auto", "structure_holdout", "extrap", "interp"),
        default="auto",
        help="Checkpoint protocol. auto prefers structure_holdout, then extrap and interp.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-case-id", default="case_g002_op01")
    parser.add_argument("--n-trials", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--uniformity-target", default="ni")
    parser.add_argument("--uniformity-region", default="wafer_near")
    parser.add_argument("--uniformity-score-mode", choices=("relative", "cv"), default="relative")
    parser.add_argument("--mid-height-band-px", type=int, default=2)
    parser.add_argument("--uniformity-row-index", type=int, default=None)
    parser.add_argument("--uniformity-col-start", type=int, default=None)
    parser.add_argument("--uniformity-col-end", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=("random", "optuna", "csv", "two_stage", "structure_diversity", "feature_archive"),
        default="two_stage",
    )
    parser.add_argument("--objective-key", default="bohm_flux_uniformity")
    parser.add_argument("--objective-density-key", default="uniformity_max_density")
    parser.add_argument("--n-initial", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--local-trials-per-seed", type=int, default=16)
    parser.add_argument("--local-radius-frac", type=float, default=0.25)
    parser.add_argument("--physics-term-weight", type=float, default=0.0)
    parser.add_argument("--boundary-term-weight", type=float, default=0.0)
    parser.add_argument("--positive-vars", nargs="*", default=["ne", "ni", "Te"])
    parser.add_argument("--positive-floor", type=float, default=1.0e-30)
    parser.add_argument("--space-mode", choices=("layout", "transform", "coil_series"), default="transform")
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
    parser.add_argument("--coil-count-min", type=int, default=2)
    parser.add_argument("--coil-count-max", type=int, default=6)
    parser.add_argument("--llcoil-range", type=float, nargs=2, default=(0.5, 1.5))
    parser.add_argument("--rrc-range", type=float, nargs=2, default=(2.0, 10.0))
    parser.add_argument("--rrce-range", type=float, nargs=2, default=(20.0, 30.0))
    parser.add_argument("--zzc-range", type=float, nargs=2, default=(0.0, 5.0))
    parser.add_argument("--min-gap-frac", type=float, default=0.2)
    parser.add_argument("--density-lower-ratio", type=float, default=0.95)
    parser.add_argument("--density-maximize-key", default="")
    parser.add_argument("--density-maximize-weight", type=float, default=0.0)
    parser.add_argument("--candidate-csv", default="", help="CSV backend candidate file.")
    parser.add_argument("--feature-pool-size", type=int, default=4096)
    parser.add_argument("--feature-dedupe-bins", type=int, default=64)
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


def _reference_series_params(dataset_root: Path, case_id: str) -> dict[str, float]:
    keys = ("llcoil", "rrc", "nncoil", "rrce", "zzc")
    with (dataset_root / "index.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("case_id")) == case_id:
                missing = [key for key in keys if key not in row or str(row[key]).strip() == ""]
                if missing:
                    raise KeyError(f"reference case lacks series columns {missing}: {case_id}")
                return {key: float(row[key]) for key in keys}
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


def _objective_cfg(
    args: argparse.Namespace,
    *,
    density_ref: float | None = None,
    objective_density_ref: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    physics_weight = float(args.physics_term_weight)
    boundary_weight = float(args.boundary_term_weight)
    density_weight = float(args.density_maximize_weight)
    for name, weight in {
        "physics_term_weight": physics_weight,
        "boundary_term_weight": boundary_weight,
        "density_maximize_weight": density_weight,
    }.items():
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    density_ratio = float(args.density_lower_ratio)
    if not np.isfinite(density_ratio) or density_ratio < 0.0:
        raise ValueError("density_lower_ratio must be finite and >= 0")
    objective_key = str(args.objective_key or "").strip() or "uniformity"
    objective_density_key = str(args.objective_density_key or "").strip() or "uniformity_max_density"
    terms: list[dict[str, Any]] = [{"key": objective_key, "direction": "min", "weight": 1.0}]
    if physics_weight > 0.0:
        terms.append(
            {
                "key": "poisson_residual_norm",
                "direction": "min",
                "weight": physics_weight,
                "transform": "log1p_abs",
                "scale": 1.0,
            }
        )
    if boundary_weight > 0.0:
        terms.append({"key": "boundary_gamma_uniformity", "direction": "min", "weight": boundary_weight})

    density_maximize_key = str(args.density_maximize_key or "").strip()
    if objective_key == "uniformity_over_density_max":
        density_scale = objective_density_ref if objective_density_ref is not None else 1.0
        density_scale = max(abs(float(density_scale)), 1.0e-30)
        terms[0] = {"key": "uniformity", "direction": "min", "weight": 1.0}
        terms.append(
            {
                "key": objective_density_key,
                "direction": "max",
                "weight": 1.0,
                "scale": density_scale,
            }
        )
    elif density_maximize_key and density_weight > 0.0:
        density_scale = objective_density_ref if objective_density_ref is not None else 1.0
        density_scale = max(abs(float(density_scale)), 1.0e-30)
        terms.append(
            {
                "key": density_maximize_key,
                "direction": "max",
                "weight": density_weight,
                "scale": density_scale,
            }
        )

    constraints: list[dict[str, Any]] = []
    if density_ref is not None and float(density_ref) > 0.0 and density_ratio > 0.0:
        constraints.append({"key": "uniformity_mean_density", "lower": float(density_ref) * density_ratio})
    return {"mode": "weighted_sum", "terms": terms}, constraints


def _backend_cfg(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.backend) == "csv":
        csv_path = str(args.candidate_csv or "").strip()
        if not csv_path:
            raise ValueError("--candidate-csv is required when --backend csv")
        return {"csv_path": csv_path, "deduplicate": True}
    if str(args.backend) != "two_stage":
        return {}
    return {
        "n_initial": int(args.n_initial),
        "top_k": int(args.top_k),
        "local_trials_per_seed": int(args.local_trials_per_seed),
        "local_radius_frac": float(args.local_radius_frac),
    }


def _uniformity_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "target": str(args.uniformity_target),
        "region": str(args.uniformity_region),
        "mid_height_band_px": int(args.mid_height_band_px),
    }
    if args.uniformity_row_index is not None:
        cfg["row_index"] = int(args.uniformity_row_index)
    if args.uniformity_col_start is not None:
        cfg["col_start"] = int(args.uniformity_col_start)
    if args.uniformity_col_end is not None:
        cfg["col_end"] = int(args.uniformity_col_end)
    return cfg


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


def _range_pair(raw: Any, *, name: str) -> tuple[float, float]:
    vals = list(raw)
    if len(vals) != 2:
        raise ValueError(f"{name} must have exactly two values")
    lo = float(vals[0])
    hi = float(vals[1])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo > hi:
        raise ValueError(f"{name} requires finite lo <= hi; got={raw!r}")
    return lo, hi


def _coil_series_layout(
    *,
    nncoil: float,
    llcoil: float,
    rrc: float,
    rrce: float,
    zzc: float,
    part_ids: list[str],
    min_gap_frac: float,
    chamber_bbox: dict[str, float],
) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    n_max = len(part_ids)
    if n_max <= 0:
        raise ValueError("coil series requires at least one part id")
    n = max(1, min(n_max, int(np.rint(float(nncoil)))))
    length = float(llcoil)
    r_start = float(rrc)
    r_end = float(rrce)
    z_min = 14.0 + float(zzc)
    if length <= 0.0:
        raise ValueError("llcoil must be > 0")
    if r_end <= r_start:
        raise ValueError("rrce must be > rrc")
    pitch = (r_end - r_start) / float(n)
    min_gap = max(0.0, float(min_gap_frac)) * length
    if (pitch - length) < min_gap:
        raise ValueError("coil series requires non-overlapping coils with the requested min gap")
    r_min_bound = float(chamber_bbox.get("r_min", 0.0))
    r_max_bound = float(chamber_bbox.get("r_max", 30.0))
    z_min_bound = float(chamber_bbox.get("z_min", 0.0))
    z_max_bound = float(chamber_bbox.get("z_max", 22.0))
    rows: list[dict[str, float | int | str]] = []
    for idx, part_id in enumerate(part_ids):
        active = idx < n
        if active:
            r_min = r_start + float(idx) * pitch
            r_max = r_min + length
            z_max = z_min + length
            if r_min < r_min_bound or r_max > r_max_bound or z_min < z_min_bound or z_max > z_max_bound:
                raise ValueError("coil series rectangle must stay within chamber bbox")
            row = {
                "part_id": str(part_id),
                "coil_index": int(idx + 1),
                "active": 1,
                "order": int(idx + 1),
                "r_min": float(r_min),
                "r_max": float(r_max),
                "z_min": float(z_min),
                "z_max": float(z_max),
                "r_center": float(0.5 * (r_min + r_max)),
                "z_center": float(0.5 * (z_min + z_max)),
                "width": float(length),
                "height": float(length),
                "pitch": float(pitch),
            }
        else:
            row = {
                "part_id": str(part_id),
                "coil_index": int(idx + 1),
                "active": 0,
                "order": int(idx + 1),
                "r_min": float("nan"),
                "r_max": float("nan"),
                "z_min": float("nan"),
                "z_max": float("nan"),
                "r_center": float("nan"),
                "z_center": float("nan"),
                "width": 0.0,
                "height": 0.0,
                "pitch": float(pitch),
            }
        rows.append(row)
    effective = {
        "series.nncoil": float(n),
        "series.llcoil": float(length),
        "series.rrc": float(r_start),
        "series.rrce": float(r_end),
        "series.zzc": float(zzc),
        "series.pitch": float(pitch),
        "series.min_gap": float(pitch - length),
    }
    return rows, effective


def _layout_param_values_from_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        part_id = str(row["part_id"])
        prefix = f"layout.{part_id}."
        active = int(row.get("active", 1)) > 0
        out[prefix + "r_center"] = float(row["r_center"]) if active else 0.0
        out[prefix + "z_center"] = float(row["z_center"]) if active else 0.0
        out[prefix + "width"] = float(row["width"]) if active else 0.0
        out[prefix + "height"] = float(row["height"]) if active else 0.0
    return out


def _layout_param_specs_from_rows(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    specs: dict[str, dict[str, float]] = {}
    defaults = _layout_param_values_from_rows(rows)
    for key, default in defaults.items():
        attr = key.rsplit(".", 1)[-1]
        if attr == "r_center":
            bounds = (0.0, 30.0)
        elif attr == "z_center":
            bounds = (0.0, 22.0)
        else:
            bounds = (0.0, 30.0)
        specs[key] = {"default": float(default), "min": bounds[0], "max": bounds[1]}
    return specs


def _series_param_specs(
    reference: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    n_min = max(1, int(args.coil_count_min))
    n_max = min(6, int(args.coil_count_max))
    if n_min > n_max:
        raise ValueError("coil-count range requires min <= max within [1, 6]")
    ll_lo, ll_hi = _range_pair(args.llcoil_range, name="llcoil-range")
    rrc_lo, rrc_hi = _range_pair(args.rrc_range, name="rrc-range")
    rrce_lo, rrce_hi = _range_pair(args.rrce_range, name="rrce-range")
    zzc_lo, zzc_hi = _range_pair(args.zzc_range, name="zzc-range")
    return {
        "series.nncoil": {"default": float(reference["nncoil"]), "min": float(n_min), "max": float(n_max)},
        "series.llcoil": {"default": float(reference["llcoil"]), "min": ll_lo, "max": ll_hi},
        "series.rrc": {"default": float(reference["rrc"]), "min": rrc_lo, "max": rrc_hi},
        "series.rrce": {"default": float(reference["rrce"]), "min": rrce_lo, "max": rrce_hi},
        "series.zzc": {"default": float(reference["zzc"]), "min": zzc_lo, "max": zzc_hi},
    }


def _prepare_series_dataset_overlay(
    dataset_root: Path,
    out_dir: Path,
    case_id: str,
    args: argparse.Namespace,
) -> Path:
    reference = _reference_series_params(dataset_root, case_id)
    rows, _ = _coil_series_layout(
        nncoil=float(reference["nncoil"]),
        llcoil=float(reference["llcoil"]),
        rrc=float(reference["rrc"]),
        rrce=float(reference["rrce"]),
        zzc=float(reference["zzc"]),
        part_ids=list(PART_IDS),
        min_gap_frac=float(args.min_gap_frac),
        chamber_bbox={"r_min": 0.0, "r_max": 30.0, "z_min": 0.0, "z_max": 22.0},
    )
    overlay_root = out_dir / "_series_dataset"
    overlay_geom = overlay_root / "geometry"
    overlay_geom.mkdir(parents=True, exist_ok=True)
    src_geom = dataset_root / "geometry"
    for src in src_geom.iterdir():
        if src.is_file():
            shutil.copy2(src, overlay_geom / src.name)
    manifest = {
        "part_ids": list(PART_IDS),
        "plasma_mode": "preserve",
        "reference_case_id": case_id,
        "optimization_space": "coil_series_external_v1",
        "source_series_param_specs": _series_param_specs(reference, args),
        "param_specs": _layout_param_specs_from_rows(rows),
    }
    (overlay_geom / "parts_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return overlay_root


def _default_geom_params(space_mode: str, layout_specs: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    if space_mode == "layout":
        return {key: float(spec["default"]) for key, spec in sorted(dict(layout_specs or {}).items())}
    if space_mode == "coil_series":
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
    if space_mode == "coil_series":
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


def _lhs_unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    if n <= 0 or d <= 0:
        return np.zeros((max(0, n), max(0, d)), dtype=np.float32)
    out = np.empty((int(n), int(d)), dtype=np.float32)
    base = np.arange(int(n), dtype=np.float32)
    for j in range(int(d)):
        out[:, j] = (rng.permutation(base) + rng.random(int(n), dtype=np.float32)) / float(n)
    return out


def _sobol_unit(*, n: int, d: int, seed: int) -> np.ndarray:
    """Return a deterministic scrambled Sobol design for this torch workflow."""

    if n <= 0 or d <= 0:
        return np.zeros((max(0, n), max(0, d)), dtype=np.float32)
    try:
        from torch.quasirandom import SobolEngine
    except ImportError as exc:  # pragma: no cover - inference also requires torch
        raise RuntimeError("scrambled Sobol sampling requires the torch optional dependency") from exc
    engine = SobolEngine(dimension=int(d), scramble=True, seed=int(seed))
    return engine.draw(int(n)).cpu().numpy().astype(np.float32, copy=False)


def _series_structure_descriptor(
    *,
    geom_param: dict[str, float],
    min_gap_frac: float,
    length_scale: float = 1.5,
) -> tuple[np.ndarray, dict[str, float | str]]:
    rows, effective = _coil_series_layout(
        nncoil=float(geom_param["series.nncoil"]),
        llcoil=float(geom_param["series.llcoil"]),
        rrc=float(geom_param["series.rrc"]),
        rrce=float(geom_param["series.rrce"]),
        zzc=float(geom_param["series.zzc"]),
        part_ids=list(PART_IDS),
        min_gap_frac=float(min_gap_frac),
        chamber_bbox={"r_min": 0.0, "r_max": 30.0, "z_min": 0.0, "z_max": 22.0},
    )
    active_rows = [row for row in rows if int(row.get("active", 0)) > 0]
    r_centers = np.asarray([float(row["r_center"]) for row in active_rows], dtype=np.float32)
    z_centers = np.asarray([float(row["z_center"]) for row in active_rows], dtype=np.float32)
    widths = np.asarray([float(row["width"]) for row in active_rows], dtype=np.float32)
    heights = np.asarray([float(row["height"]) for row in active_rows], dtype=np.float32)
    n = float(effective["series.nncoil"])
    span_r = float(geom_param["series.rrce"] - geom_param["series.rrc"])
    ll = float(geom_param["series.llcoil"])
    pitch = float(effective["series.pitch"])
    min_gap = float(effective["series.min_gap"])
    z_min = 14.0 + float(geom_param["series.zzc"])
    z_max = z_min + ll
    fill_ratio = float(n * ll / max(span_r, 1.0e-12))
    area_proxy = float(np.sum(widths * heights, dtype=np.float32))
    r_hist_edges = np.linspace(0.0, 30.0, 7, dtype=np.float32)
    r_hist = np.histogram(r_centers, bins=r_hist_edges)[0].astype(np.float32) / max(n, 1.0)
    # Fixed, dimensionless geometry fingerprint. Operating conditions are
    # deliberately excluded: pp/pp0 changes do not create a new structure.
    safe_length_scale = max(abs(float(length_scale)), 1.0e-12)
    descriptor_values: list[float] = []
    for row in rows:
        active = int(row.get("active", 0)) > 0
        descriptor_values.extend(
            [
                1.0 if active else 0.0,
                float(row["r_center"]) / 30.0 if active else 0.0,
                float(row["z_center"]) / 22.0 if active else 0.0,
                float(row["width"]) / safe_length_scale if active else 0.0,
                float(row["height"]) / safe_length_scale if active else 0.0,
            ]
        )
    descriptor = np.asarray([*descriptor_values, *[float(v) for v in r_hist.tolist()]], dtype=np.float32)
    meta = {
        "sampler_nncoil_effective": n,
        "sampler_pitch": pitch,
        "sampler_min_gap": min_gap,
        "sampler_span_r": span_r,
        "sampler_fill_ratio": fill_ratio,
        "sampler_area_proxy": area_proxy,
        "sampler_r_center_mean": float(np.mean(r_centers, dtype=np.float32)),
        "sampler_r_center_std": float(np.std(r_centers, dtype=np.float32)),
        "sampler_z_min": z_min,
        "sampler_z_max": z_max,
    }
    return descriptor, meta


def _signature_from_descriptor(desc: np.ndarray, *, bins: int) -> str:
    vals = np.asarray(desc, dtype=np.float32).reshape(-1)
    finite = np.where(np.isfinite(vals), vals, 0.0)
    rounded = np.rint(finite * float(max(1, int(bins)))).astype(np.int32)
    digest = hashlib.sha256(rounded.tobytes()).hexdigest()[:16]
    return digest


def _canonical_series_signature(rows: list[dict[str, float | int | str]], *, decimals: int = 8) -> str:
    """Hash active rectangles in canonical order, ignoring inactive slot data."""

    canonical: list[tuple[float, float, float, float]] = []
    for row in rows:
        if int(row.get("active", 0)) <= 0:
            continue
        canonical.append(
            tuple(round(float(row[key]), int(decimals)) for key in ("r_min", "r_max", "z_min", "z_max"))
        )
    canonical.sort()
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def _farthest_indices(features: np.ndarray, n_select: int, *, rng: np.random.Generator) -> list[int]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0 or n_select <= 0:
        return []
    n = int(x.shape[0])
    target = min(int(n_select), n)
    center = np.mean(x, axis=0, dtype=np.float32)
    first = int(np.argmax(np.sum((x - center.reshape(1, -1)) ** 2, axis=1)))
    selected = [first]
    min_dist = np.sum((x - x[first].reshape(1, -1)) ** 2, axis=1)
    while len(selected) < target:
        idx = int(np.argmax(min_dist))
        if idx in selected:
            remaining = [i for i in range(n) if i not in set(selected)]
            if not remaining:
                break
            idx = int(rng.choice(np.asarray(remaining, dtype=np.int64)))
        selected.append(idx)
        dist = np.sum((x - x[idx].reshape(1, -1)) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return selected


def _write_structure_diversity_candidates(
    path: Path,
    *,
    space: dict[str, tuple[float, float]],
    series_space: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if str(args.space_mode) != "coil_series":
        raise ValueError("structure_diversity sampler currently requires --space-mode coil_series")
    rng = np.random.default_rng(int(args.seed))
    n_trials = max(1, int(args.n_trials))
    n_pool = max(n_trials, int(args.feature_pool_size))
    count_min = int(round(float(series_space["series.nncoil"][0])))
    count_max = int(round(float(series_space["series.nncoil"][1])))
    counts = list(range(count_min, count_max + 1))
    n_per_count = int(np.ceil(float(n_pool) / float(max(len(counts), 1))))
    continuous_keys = [k for k in [*sorted(space), *sorted(series_space)] if k != "series.nncoil"]

    candidates: list[dict[str, Any]] = []
    descriptors: list[np.ndarray] = []
    exact_signatures: set[str] = set()
    near_signatures: set[str] = set()
    rejected_exact_duplicates = 0
    rejected_near_duplicates = 0
    length_scale = max(abs(float(v)) for v in args.llcoil_range)
    for count in counts:
        unit = _sobol_unit(n=n_per_count, d=len(continuous_keys), seed=int(args.seed) + int(count))
        for row_idx in range(n_per_count):
            cond: dict[str, float] = {}
            geom: dict[str, float] = {"series.nncoil": float(count)}
            for j, key in enumerate(continuous_keys):
                bounds = space.get(key, series_space.get(key))
                if bounds is None:
                    continue
                lo, hi = float(bounds[0]), float(bounds[1])
                value = lo + float(unit[row_idx, j]) * (hi - lo)
                if key in space:
                    cond[key] = value
                else:
                    geom[key] = value
            try:
                desc, meta = _series_structure_descriptor(
                    geom_param=geom,
                    min_gap_frac=float(args.min_gap_frac),
                    length_scale=length_scale,
                )
                rows, _ = _coil_series_layout(
                    nncoil=float(geom["series.nncoil"]),
                    llcoil=float(geom["series.llcoil"]),
                    rrc=float(geom["series.rrc"]),
                    rrce=float(geom["series.rrce"]),
                    zzc=float(geom["series.zzc"]),
                    part_ids=list(PART_IDS),
                    min_gap_frac=float(args.min_gap_frac),
                    chamber_bbox={"r_min": 0.0, "r_max": 30.0, "z_min": 0.0, "z_max": 22.0},
                )
            except ValueError:
                continue
            exact_sig = _canonical_series_signature(rows)
            if exact_sig in exact_signatures:
                rejected_exact_duplicates += 1
                continue
            near_sig = _signature_from_descriptor(desc, bins=int(args.feature_dedupe_bins))
            if near_sig in near_signatures:
                rejected_near_duplicates += 1
                continue
            exact_signatures.add(exact_sig)
            near_signatures.add(near_sig)
            descriptors.append(desc)
            candidates.append(
                {
                    "cond": cond,
                    "geom": _layout_param_values_from_rows(rows),
                    "series_meta": geom,
                    "sampler_structure_signature": exact_sig,
                    "sampler_near_signature": near_sig,
                    **meta,
                }
            )

    if not candidates:
        raise ValueError("structure_diversity sampler generated no valid candidates")

    desc_mat = np.vstack(descriptors).astype(np.float32)
    # Channels use fixed physical scales, so distances do not change with the
    # candidate pool or random seed.
    norm = desc_mat

    selected: list[int] = []
    selected_set: set[int] = set()
    quota = max(1, int(np.floor(float(n_trials) / float(max(len(counts), 1)))))
    for count in counts:
        idxs = [i for i, cand in enumerate(candidates) if int(round(float(cand["sampler_nncoil_effective"]))) == count]
        if not idxs:
            continue
        sub = norm[np.asarray(idxs, dtype=np.int64)]
        for local_idx in _farthest_indices(sub, quota, rng=rng):
            idx = int(idxs[local_idx])
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

    if len(selected) < n_trials:
        remaining = [i for i in range(len(candidates)) if i not in selected_set]
        if remaining:
            base = norm[np.asarray(remaining, dtype=np.int64)]
            for local_idx in _farthest_indices(base, n_trials - len(selected), rng=rng):
                idx = int(remaining[local_idx])
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= n_trials:
                    break

    selected = selected[:n_trials]
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected):
        cand = candidates[idx]
        row: dict[str, Any] = {
            "sampler_rank": int(rank),
            **cand["cond"],
            **cand["geom"],
            **cand.get("series_meta", {}),
            **{k: v for k, v in cand.items() if k not in {"cond", "geom", "series_meta"}},
        }
        rows.append(row)
    if not rows:
        raise ValueError("structure_diversity sampler selected no unique feature candidates")

    header = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "candidate_csv": str(path),
        "pool_size_requested": int(n_pool),
        "pool_size_valid": int(len(candidates)),
        "selected_count": int(len(rows)),
        "unique_signature_count": int(len(exact_signatures)),
        "unique_near_signature_count": int(len(near_signatures)),
        "rejected_exact_duplicates": int(rejected_exact_duplicates),
        "rejected_near_duplicates": int(rejected_near_duplicates),
        "coil_counts": counts,
        "continuous_keys": continuous_keys,
        "descriptor_dim": int(desc_mat.shape[1]),
        "dedupe_bins": int(args.feature_dedupe_bins),
        "sampler": "scrambled_sobol",
        "descriptor_scaling": "fixed_physical",
        "condition_features_in_descriptor": False,
    }


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
    axes[1].imshow(best_field, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"best {key}")
    im2 = axes[2].imshow(diff, origin="lower", cmap="coolwarm", vmin=-dmax, vmax=dmax)
    axes[2].set_title("best - base")
    fig.colorbar(im0, ax=axes[:2], shrink=0.82)
    fig.colorbar(im2, ax=axes[2], shrink=0.82)
    fig.savefig(out_dir / f"base_vs_best_{key}.png", dpi=160)
    plt.close(fig)


def _layout_rows_from_params(
    *,
    space_mode: str,
    geom_param: dict[str, float],
    layout_specs: dict[str, dict[str, float]] | None,
    min_gap_frac: float = 0.2,
) -> list[dict[str, float | int | str]]:
    if space_mode == "coil_series":
        if {"series.nncoil", "series.llcoil", "series.rrc", "series.rrce", "series.zzc"}.issubset(geom_param):
            rows, _ = _coil_series_layout(
                nncoil=float(geom_param["series.nncoil"]),
                llcoil=float(geom_param["series.llcoil"]),
                rrc=float(geom_param["series.rrc"]),
                rrce=float(geom_param["series.rrce"]),
                zzc=float(geom_param["series.zzc"]),
                part_ids=list(PART_IDS),
                min_gap_frac=float(min_gap_frac),
                chamber_bbox={"r_min": 0.0, "r_max": 30.0, "z_min": 0.0, "z_max": 22.0},
            )
            return rows
        space_mode = "layout"
    if space_mode == "layout":
        rows: list[dict[str, float | int | str]] = []
        ids = sorted({key.split(".")[1] for key in dict(layout_specs or {}).keys() if key.startswith("layout.")})
        for idx, part_id in enumerate(ids, start=1):
            prefix = f"layout.{part_id}."
            r_center = float(geom_param[prefix + "r_center"])
            z_center = float(geom_param[prefix + "z_center"])
            width = float(geom_param[prefix + "width"])
            height = float(geom_param[prefix + "height"])
            active = width > 0.0 and height > 0.0
            rows.append(
                {
                    "part_id": part_id,
                    "coil_index": idx,
                    "active": 1 if active else 0,
                    "order": idx,
                    "r_min": r_center - 0.5 * width if active else np.nan,
                    "r_max": r_center + 0.5 * width if active else np.nan,
                    "z_min": z_center - 0.5 * height if active else np.nan,
                    "z_max": z_center + 0.5 * height if active else np.nan,
                    "r_center": r_center,
                    "z_center": z_center,
                    "width": width,
                    "height": height,
                    "pitch": np.nan,
                }
            )
        return rows
    return []


def _reference_layout_rows(coil_layout_root: Path, case_id: str) -> list[dict[str, float | int | str]]:
    rows = []
    for row in _read_reference_layout(coil_layout_root, case_id):
        r_center = float(row["r_center"])
        z_center = float(row["z_center"])
        width = float(row["width"])
        height = float(row["height"])
        part_id = str(row["part_id"])
        idx = int(part_id.split("_")[-1])
        rows.append(
            {
                "part_id": part_id,
                "coil_index": idx,
                "active": 1,
                "order": idx,
                "r_min": r_center - 0.5 * width,
                "r_max": r_center + 0.5 * width,
                "z_min": z_center - 0.5 * height,
                "z_max": z_center + 0.5 * height,
                "r_center": r_center,
                "z_center": z_center,
                "width": width,
                "height": height,
                "pitch": float(row.get("pitch", np.nan)),
            }
        )
    return rows


def _write_layout_csv(path: Path, rows: list[dict[str, float | int | str]], *, case_id: str) -> None:
    fieldnames = [
        "case_id",
        "coil_index",
        "active",
        "order",
        "r_min",
        "r_max",
        "z_min",
        "z_max",
        "r_center",
        "z_center",
        "width",
        "height",
        "pitch",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in fieldnames}
            payload["case_id"] = case_id
            writer.writerow(payload)


def _active_by_index(rows: list[dict[str, float | int | str]]) -> dict[int, dict[str, float | int | str]]:
    return {int(row["coil_index"]): row for row in rows if int(row.get("active", 0)) > 0}


def _plot_geometry_outputs(
    *,
    out_dir: Path,
    base_rows: list[dict[str, float | int | str]],
    best_rows: list[dict[str, float | int | str]],
    base_qoi: dict[str, Any],
    best_qoi: dict[str, Any],
) -> None:
    import matplotlib.patches as patches

    base_active = _active_by_index(base_rows)
    best_active = _active_by_index(best_rows)
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    for idx, row in base_active.items():
        rect = patches.Rectangle(
            (float(row["r_min"]), float(row["z_min"])),
            float(row["width"]),
            float(row["height"]),
            edgecolor="#7a7f87",
            facecolor="#7a7f87",
            alpha=0.26,
            lw=1.8,
            label="before" if idx == min(base_active) else None,
        )
        ax.add_patch(rect)
        ax.text(float(row["r_center"]), float(row["z_center"]), str(idx), ha="center", va="center", fontsize=7)
    for idx, row in best_active.items():
        rect = patches.Rectangle(
            (float(row["r_min"]), float(row["z_min"])),
            float(row["width"]),
            float(row["height"]),
            edgecolor="#2f6fbb",
            facecolor="#2f6fbb",
            alpha=0.32,
            lw=1.8,
            label="after" if idx == min(best_active) else None,
        )
        ax.add_patch(rect)
        ax.text(float(row["r_center"]), float(row["z_center"]), str(idx), ha="center", va="center", fontsize=7)
        if idx in base_active:
            brow = base_active[idx]
            ax.annotate(
                "",
                xy=(float(row["r_center"]), float(row["z_center"])),
                xytext=(float(brow["r_center"]), float(brow["z_center"])),
                arrowprops={"arrowstyle": "->", "color": "#b84545", "lw": 1.2},
            )
    active = list(base_active.values()) + list(best_active.values())
    if active:
        ax.set_xlim(min(float(row["r_min"]) for row in active) - 1.5, max(float(row["r_max"]) for row in active) + 1.5)
        ax.set_ylim(min(float(row["z_min"]) for row in active) - 1.5, max(float(row["z_max"]) for row in active) + 1.5)
    base_u = float(base_qoi.get("uniformity", np.nan))
    best_u = float(best_qoi.get("uniformity", np.nan))
    ax.set_title(f"coil geometry: uniformity {base_u:.3f} -> {best_u:.3f}")
    ax.set_xlabel("r")
    ax.set_ylabel("z")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "geometry_before_after_arrows.png", dpi=160)
    plt.close(fig)

    records: list[dict[str, float | int]] = []
    for idx in sorted(set(base_active.keys()) | set(best_active.keys())):
        b = base_active.get(idx)
        a = best_active.get(idx)
        if b is None or a is None:
            continue
        records.append(
            {
                "coil": idx,
                "dr_center": float(a["r_center"]) - float(b["r_center"]),
                "dz_center": float(a["z_center"]) - float(b["z_center"]),
                "width_ratio": float(a["width"]) / float(b["width"]),
                "height_ratio": float(a["height"]) / float(b["height"]),
            }
        )
    if records:
        with (out_dir / "geometry_delta_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        x = np.arange(len(records))
        labels = [f"C{int(row['coil'])}" for row in records]
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
        axes[0].bar(x - 0.18, [float(row["dr_center"]) for row in records], 0.36, label="dr center", color="#2f6fbb")
        axes[0].bar(x + 0.18, [float(row["dz_center"]) for row in records], 0.36, label="dz center", color="#b06a2d")
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_title("center shift")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(frameon=False)
        axes[1].bar(x - 0.18, [float(row["width_ratio"]) for row in records], 0.36, label="width ratio", color="#2f6fbb")
        axes[1].bar(x + 0.18, [float(row["height_ratio"]) for row in records], 0.36, label="height ratio", color="#b06a2d")
        axes[1].axhline(1, color="black", lw=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_title("size ratio")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(frameon=False)
        fig.suptitle("geometry delta summary")
        fig.savefig(out_dir / "geometry_delta_summary.png", dpi=160)
        plt.close(fig)


def _plot_fields_log(base: Any, best: Any, out_dir: Path) -> None:
    pairs = [("ne", _field2d(base, "ne"), _field2d(best, "ne")), ("ni", _field2d(base, "ni"), _field2d(best, "ni"))]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), constrained_layout=True)
    for row_idx, (name, base_field, best_field) in enumerate(pairs):
        base_log = np.log10(np.clip(base_field, 1.0e12, None))
        best_log = np.log10(np.clip(best_field, 1.0e12, None))
        diff = best_log - base_log
        both = np.concatenate([base_log.ravel(), best_log.ravel()])
        vmin, vmax = np.nanpercentile(both, [1, 99.5])
        lim = float(np.nanpercentile(np.abs(diff), 99))
        if not np.isfinite(lim) or lim <= 0.0:
            lim = 1.0
        panels = [
            (base_log, f"before log10({name})", "viridis", vmin, vmax),
            (best_log, f"after log10({name})", "viridis", vmin, vmax),
            (diff, f"after - before log10({name})", "coolwarm", -lim, lim),
        ]
        for col_idx, (arr, title, cmap, lo, hi) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("fields before/after")
    fig.savefig(out_dir / "fields_log_before_after.png", dpi=160)
    plt.close(fig)


def _plot_trial_line_profiles(
    *,
    engine: Any,
    out_dir: Path,
    trials_csv: Path,
    space_keys: list[str],
    geom_keys: list[str],
    axis: dict[str, Any],
    base_result: Any,
    target: str,
    row_index: int | None,
    col_start: int | None,
    col_end: int | None,
) -> None:
    if row_index is None:
        return
    trials: list[dict[str, str]] = []
    with trials_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                value = float(row.get("objective_value", row.get("value", "")))
            except ValueError:
                continue
            if np.isfinite(value):
                trials.append(dict(row))
    if not trials:
        return

    row_idx = int(row_index)
    base_field = _field2d(base_result, target)
    if row_idx < 0 or row_idx >= base_field.shape[0]:
        raise ValueError(f"profile row_index must be within [0, {base_field.shape[0] - 1}]")
    c0 = 0 if col_start is None else int(col_start)
    c1 = (base_field.shape[1] - 1) if col_end is None else int(col_end)
    if c0 < 0 or c1 < c0 or c0 >= base_field.shape[1]:
        raise ValueError(f"profile column range must overlap [0, {base_field.shape[1] - 1}]")
    c1 = min(c1, base_field.shape[1] - 1)
    x = np.arange(c0, c1 + 1)

    profiles: list[tuple[float, np.ndarray, int]] = []
    for row in trials:
        cond = {key: float(row[key]) for key in space_keys if str(row.get(key, "")).strip()}
        geom_param = {key: float(row[key]) for key in geom_keys if str(row.get(key, "")).strip()}
        result = engine.single_run_aggregated(
            cond=cond,
            geom={"geom_id": "default", "geom_param": geom_param},
            axis=axis,
        )
        profile = _field2d(result, target)[row_idx, c0 : c1 + 1].astype(np.float64)
        profile_value = float(row.get("objective_value", row.get("value", "nan")))
        profiles.append((profile_value, profile, int(row.get("trial", len(profiles)))))

    if not profiles:
        return
    best_value, best_profile, best_trial = min(profiles, key=lambda item: item[0])
    worst_value, worst_profile, worst_trial = max(profiles, key=lambda item: item[0])
    base_profile = base_field[row_idx, c0 : c1 + 1].astype(np.float64)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for _, profile, _ in profiles:
        ax.plot(x, np.log10(np.maximum(profile, 1.0e-30)), color="#9aa0a6", alpha=0.08, lw=0.6)
    ax.plot(x, np.log10(np.maximum(base_profile, 1.0e-30)), color="#202124", lw=1.8, ls="--", label="base")
    ax.plot(
        x,
        np.log10(np.maximum(best_profile, 1.0e-30)),
        color="#2f6fbb",
        lw=2.2,
        label=f"best trial {best_trial} ({best_value:.4g})",
    )
    ax.plot(
        x,
        np.log10(np.maximum(worst_profile, 1.0e-30)),
        color="#c43c39",
        lw=2.0,
        label=f"worst trial {worst_trial} ({worst_value:.4g})",
    )
    ax.set_xlabel("r pixel")
    ax.set_ylabel(f"log10({target}) at z={row_idx}")
    ax.set_title(f"trial profiles at z={row_idx}, r={c0}..{c1}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"trial_profiles_z{row_idx}_r{c0}_{c1}.png", dpi=170)
    plt.close(fig)


def _plot_qoi_and_curve(out_dir: Path, base_qoi: dict[str, Any], best_qoi: dict[str, Any], trials_csv: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    base_density = float(base_qoi.get("uniformity_mean_density", np.nan))
    best_density = float(best_qoi.get("uniformity_mean_density", np.nan))
    density_ratio = best_density / base_density if np.isfinite(base_density) and base_density > 0.0 else np.nan
    base_max_density = float(base_qoi.get("uniformity_max_density", np.nan))
    best_max_density = float(best_qoi.get("uniformity_max_density", np.nan))
    max_density_ratio = (
        best_max_density / base_max_density if np.isfinite(base_max_density) and base_max_density > 0.0 else np.nan
    )
    x = np.arange(4)
    labels = ["uniformity", "boundary gamma CV", "mean density / base", "max density / base"]
    before = [
        float(base_qoi.get("uniformity", np.nan)),
        float(base_qoi.get("boundary_gamma_uniformity", np.nan)),
        1.0,
        1.0,
    ]
    after = [
        float(best_qoi.get("uniformity", np.nan)),
        float(best_qoi.get("boundary_gamma_uniformity", np.nan)),
        density_ratio,
        max_density_ratio,
    ]
    ax.bar(x - 0.18, before, 0.36, label="before", color="#7a7f87")
    ax.bar(x + 0.18, after, 0.36, label="after", color="#2f6fbb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("QoI before/after")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "qoi_before_after.png", dpi=160)
    plt.close(fig)

    vals: list[float] = []
    with trials_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                val = float(row.get("objective_value", row.get("value", "")))
            except ValueError:
                continue
            if np.isfinite(val):
                vals.append(val)
    if vals:
        best_so_far = np.minimum.accumulate(np.asarray(vals, dtype=np.float64))
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ax.plot(np.arange(1, len(best_so_far) + 1), best_so_far, color="#2f6fbb", lw=1.8)
        ax.set_xlabel("trial")
        ax.set_ylabel("best-so-far objective")
        ax.set_title("optimization curve")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "best_so_far_curve.png", dpi=160)
        plt.close(fig)


def _resolve_checkpoint_dir(run_dir: Path, model_name: str, protocol: str) -> tuple[Path, str]:
    protocol_norm = str(protocol).strip().lower()
    protocols = [protocol_norm] if protocol_norm != "auto" else ["structure_holdout", "extrap", "interp"]
    for candidate_protocol in protocols:
        candidate = run_dir / "models" / model_name / "eval_protocol" / candidate_protocol / "checkpoints"
        if candidate.is_dir() and (candidate / "meta.json").is_file():
            return candidate, candidate_protocol
    searched = [str(run_dir / "models" / model_name / "eval_protocol" / p / "checkpoints") for p in protocols]
    raise FileNotFoundError(f"no checkpoint directory found; searched: {searched}")


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
    series_specs: dict[str, dict[str, float]] | None = None
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
        series_specs = {str(k): dict(v) for k, v in dict(manifest.get("source_series_param_specs", {})).items()}
    elif args.space_mode == "coil_series":
        overlay_root = _prepare_series_dataset_overlay(
            dataset_root=dataset_root,
            out_dir=out_dir,
            case_id=str(args.reference_case_id),
            args=args,
        )
        provider_dataset_root = overlay_root
        manifest = json.loads((overlay_root / "geometry" / "parts_manifest.json").read_text(encoding="utf-8"))
        layout_specs = {str(k): dict(v) for k, v in dict(manifest["param_specs"]).items()}
    checkpoint_dir, resolved_protocol = _resolve_checkpoint_dir(run_dir, str(args.model), str(args.eval_protocol))
    model = load_checkpoint(checkpoint_dir)
    bundle = RunBundleLoader.load(run_dir, model=model)
    checkpoint_meta_path = checkpoint_dir / "meta.json"
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
            "qoi": {"uniformity": {"preferred_targets": [str(args.uniformity_target), "ne", "ni", "Te"], **_uniformity_cfg(args)}},
            "postprocess": {
                "positive_vars": [str(v) for v in list(args.positive_vars or []) if str(v).strip()],
                "positive_floor": float(args.positive_floor),
            },
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
    density_ref = base.qoi.get("uniformity_mean_density")
    objective_density_key = str(args.objective_density_key or "").strip() or "uniformity_max_density"
    if str(args.objective_key or "").strip() == "uniformity_over_density_max":
        density_scale_key = objective_density_key
    else:
        density_scale_key = str(args.density_maximize_key or "").strip() or objective_density_key
    objective_density_ref = base.qoi.get(density_scale_key)
    objective_cfg, constraints_cfg = _objective_cfg(
        args,
        density_ref=float(density_ref) if density_ref is not None else None,
        objective_density_ref=float(objective_density_ref) if objective_density_ref is not None else None,
    )
    geom_space_mode = "layout" if str(args.space_mode) == "coil_series" else str(args.space_mode)
    geom_space = _geom_space(geom_space_mode, layout_specs)
    backend = str(args.backend)
    backend_cfg = (
        {}
        if str(args.space_mode) == "coil_series" and backend == "csv" and not str(args.candidate_csv or "").strip()
        else _backend_cfg(args)
    )
    sampler_summary: dict[str, Any] = {}
    if str(args.space_mode) == "coil_series" and (backend != "csv" or not str(args.candidate_csv or "").strip()):
        candidate_csv = out_dir / (
            "structure_diversity_candidates.csv"
            if backend in {"feature_archive", "structure_diversity"}
            else "coil_series_candidates.csv"
        )
        sampler_summary = _write_structure_diversity_candidates(
            candidate_csv,
            space=space,
            series_space=_geom_space("coil_series", series_specs),
            args=args,
        )
        backend = "csv"
        backend_cfg = {"csv_path": str(candidate_csv), "deduplicate": True}
        (out_dir / "structure_sampler_summary.json").write_text(
            json.dumps(sampler_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    elif backend in {"feature_archive", "structure_diversity"}:
        raise ValueError("--backend structure_diversity currently requires --space-mode coil_series")
    best = engine.optimize_run(
        space=space,
        geom_space=geom_space,
        n_trials=int(args.n_trials),
        geom={"geom_id": "default"},
        axis=axis,
        seed=int(args.seed),
        backend=backend,
        backend_cfg=backend_cfg,
        objective_cfg=objective_cfg,
        constraints_cfg=constraints_cfg,
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
                "uniformity_cfg": _uniformity_cfg(args),
                "backend": backend,
                "backend_requested": str(args.backend),
                "eval_protocol": resolved_protocol,
                "backend_cfg": backend_cfg,
                "structure_sampler": sampler_summary,
                "objective": objective_cfg,
                "constraints": constraints_cfg,
                "positive_vars": [str(v) for v in list(args.positive_vars or []) if str(v).strip()],
                "positive_floor": float(args.positive_floor),
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
    if str(args.space_mode) in {"layout", "coil_series"}:
        base_rows = _reference_layout_rows(Path(args.coil_layout_root), str(args.reference_case_id))
        best_rows = _layout_rows_from_params(
            space_mode=str(args.space_mode),
            geom_param=dict(best["best_geom_param"]),
            layout_specs=layout_specs,
            min_gap_frac=float(args.min_gap_frac),
        )
        _write_layout_csv(out_dir / "best_coil_layout.csv", best_rows, case_id="best")
        _plot_geometry_outputs(
            out_dir=out_dir,
            base_rows=base_rows,
            best_rows=best_rows,
            base_qoi=dict(base.qoi),
            best_qoi=dict(best_result.qoi),
        )
    _plot_fields_log(base, best_result, out_dir)
    _plot_qoi_and_curve(out_dir, dict(base.qoi), dict(best_result.qoi), out_dir / "trials.csv")
    _plot_trial_line_profiles(
        engine=engine,
        out_dir=out_dir,
        trials_csv=out_dir / "trials.csv",
        space_keys=list(space.keys()),
        geom_keys=list(geom_space.keys()),
        axis=axis,
        base_result=base,
        target=str(args.uniformity_target),
        row_index=args.uniformity_row_index,
        col_start=args.uniformity_col_start,
        col_end=args.uniformity_col_end,
    )
    print(json.dumps({"out_dir": str(out_dir), "best_objective_value": best["best_objective_value"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
