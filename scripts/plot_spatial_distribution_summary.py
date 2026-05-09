#!/usr/bin/env python3
"""Create reusable spatial distribution summary plots from compare CSV.

This script compares spatial field maps across selected model rows in
`selected_models_comparison.csv` and produces:
1) per-variable map comparison for one case
2) per-variable map comparison for the mean over N cases

Usage example:
  python scripts/plot_spatial_distribution_summary.py \
    --compare-csv runs/.../selected_models_comparison.csv \
    --out-dir runs/.../plots \
    --protocol interp \
    --vars ne ni Te phi \
    --model-names s251_a_unet_allvars_v2_control \
                 s251_b_unet_allvars_v2_selection_density \
                 s251_c_unet_allvars_v2_density_head_weight \
                 global_v2_frozen_ref
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

POSITIVE_RANGE_VARS = {"ne", "ni", "Te"}


def _load_compare_rows(compare_csv: Path) -> List[dict]:
    with compare_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def _select_rows(rows: List[dict], model_names: Sequence[str]) -> List[dict]:
    by_name = {r["name"]: r for r in rows}
    selected: List[dict] = []
    missing: List[str] = []
    for name in model_names:
        row = by_name.get(name)
        if row is None:
            missing.append(name)
            continue
        selected.append(row)
    if missing:
        raise ValueError(f"model name(s) not found in compare CSV: {missing}")
    return selected


def _inference_single_dir(row: dict, protocol: str) -> Path:
    lb = row.get("source_leaderboard", "")
    if not lb:
        raise ValueError(f"row={row.get('name')} has empty source_leaderboard")
    run_dir = Path(lb).resolve().parent
    model_id = row["model_id"]
    p = run_dir / "models" / model_id / "eval_protocol" / protocol / "inference" / "single"
    if not p.exists():
        raise FileNotFoundError(f"inference/single not found: {p}")
    return p


def _common_case_hashes(single_dirs: Sequence[Path]) -> List[str]:
    sets = []
    for d in single_dirs:
        case_ids = {p.name for p in d.iterdir() if p.is_dir()}
        sets.append(case_ids)
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


def _load_field_npz(npz_path: Path, var_name: str) -> np.ndarray:
    arr = np.load(npz_path)[var_name]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32)


def _collect_case_maps(
    rows: Sequence[dict],
    single_dirs: Sequence[Path],
    case_hash: str,
    vars_list: Sequence[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    maps: Dict[str, Dict[str, np.ndarray]] = {}
    for row, d in zip(rows, single_dirs):
        npz_path = d / case_hash / "fields_phys.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"fields_phys.npz not found: {npz_path}")
        maps[row["name"]] = {v: _load_field_npz(npz_path, v) for v in vars_list}
    return maps


def _collect_mean_maps(
    rows: Sequence[dict],
    single_dirs: Sequence[Path],
    case_hashes: Sequence[str],
    vars_list: Sequence[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    acc: Dict[str, Dict[str, np.ndarray]] = {row["name"]: {} for row in rows}
    for row, d in zip(rows, single_dirs):
        for v in vars_list:
            stack = []
            for c in case_hashes:
                npz_path = d / c / "fields_phys.npz"
                if npz_path.exists():
                    stack.append(_load_field_npz(npz_path, v))
            if not stack:
                raise RuntimeError(f"no case data for row={row['name']} var={v}")
            acc[row["name"]][v] = np.mean(np.stack(stack, axis=0), axis=0)
    return acc


def _case_key(cond: dict[str, float], axis_value: float, axis_mode: str = "steady", geom_id: str = "default") -> str:
    payload = {
        "cond": {k: float(np.float32(v)) for k, v in sorted(cond.items())},
        "axis": {"mode": str(axis_mode), "value": float(axis_value)},
        "geom": {"geom_id": str(geom_id)},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _load_gt_from_fields_npz(npz_path: Path, var_name: str) -> np.ndarray:
    data = np.load(npz_path)
    if var_name in data.files:
        arr = data[var_name]
    elif var_name == "ne" and "log_ne" in data.files:
        arr = np.power(10.0, np.asarray(data["log_ne"], dtype=np.float32))
    elif var_name == "ni" and "log_ni" in data.files:
        arr = np.power(10.0, np.asarray(data["log_ni"], dtype=np.float32))
    else:
        raise KeyError(f"GT field {var_name} not found in {npz_path.name}")
    return np.asarray(arr, dtype=np.float32)


def _build_casehash_to_fieldspath(
    dataset_root: Path,
    index_csv: Path,
    cond_columns: Sequence[str],
    axis_column: str,
    fields_npz_column: str,
    axis_mode: str,
    geom_id: str,
) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    with index_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = {k: float(row[k]) for k in cond_columns}
            axis_value = float(row[axis_column])
            key = _case_key(cond, axis_value=axis_value, axis_mode=axis_mode, geom_id=geom_id)
            mapping[key] = (dataset_root / row[fields_npz_column]).resolve()
    return mapping


def _collect_gt_case_maps(
    case_hash: str,
    casehash_to_fields: Dict[str, Path],
    vars_list: Sequence[str],
) -> Dict[str, np.ndarray]:
    p = casehash_to_fields.get(case_hash)
    if p is None:
        raise KeyError(f"ground-truth mapping not found for case hash: {case_hash}")
    if not p.exists():
        raise FileNotFoundError(f"ground-truth fields npz not found: {p}")
    return {v: _load_gt_from_fields_npz(p, v) for v in vars_list}


def _collect_gt_mean_maps(
    case_hashes: Sequence[str],
    casehash_to_fields: Dict[str, Path],
    vars_list: Sequence[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for v in vars_list:
        stack = []
        for c in case_hashes:
            p = casehash_to_fields.get(c)
            if p is None:
                continue
            if not p.exists():
                continue
            stack.append(_load_gt_from_fields_npz(p, v))
        if not stack:
            raise RuntimeError(f"no GT data found for var={v} in selected mean cases")
        out[v] = np.mean(np.stack(stack, axis=0), axis=0)
    return out


def _robust_limits(arrays: Sequence[np.ndarray], low_q: float, high_q: float) -> tuple[float, float]:
    concat = np.concatenate([a.reshape(-1) for a in arrays])
    vmin = float(np.nanpercentile(concat, low_q))
    vmax = float(np.nanpercentile(concat, high_q))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = float(np.nanmin(concat))
        vmax = float(np.nanmax(concat))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def _plot_per_var_grid(
    maps_by_model: Dict[str, Dict[str, np.ndarray]],
    ordered_names: Sequence[str],
    title_prefix: str,
    out_path: Path,
    vars_list: Sequence[str],
    q_low: float,
    q_high: float,
    gt_maps: Dict[str, np.ndarray] | None = None,
) -> None:
    n_vars = len(vars_list)
    n_models = len(ordered_names) + (1 if gt_maps is not None else 0)
    fig, axes = plt.subplots(n_vars, n_models, figsize=(3.4 * n_models, 2.9 * n_vars), constrained_layout=True)
    if n_vars == 1:
        axes = np.array([axes])
    if n_models == 1:
        axes = axes[:, np.newaxis]

    for vi, var_name in enumerate(vars_list):
        arrays = [maps_by_model[m][var_name] for m in ordered_names]
        if gt_maps is not None:
            arrays = [gt_maps[var_name]] + arrays
        vmin, vmax = _robust_limits(arrays, q_low, q_high)
        if var_name in POSITIVE_RANGE_VARS:
            vmin = max(0.0, vmin)
            vmax = max(vmax, vmin + 1e-6)
        im = None
        col_names = (["Ground Truth"] if gt_maps is not None else []) + list(ordered_names)
        for mi, model_name in enumerate(col_names):
            ax = axes[vi, mi]
            if model_name == "Ground Truth":
                img = gt_maps[var_name] if gt_maps is not None else maps_by_model[ordered_names[0]][var_name]
            else:
                img = maps_by_model[model_name][var_name]
            im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
            if vi == 0:
                ax.set_title(model_name, fontsize=10)
            if mi == 0:
                ax.set_ylabel(var_name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes[vi, :], shrink=0.86, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(title_prefix, fontsize=12)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_diff_vs_reference(
    maps_by_model: Dict[str, Dict[str, np.ndarray]],
    ordered_names: Sequence[str],
    reference_name: str,
    title_prefix: str,
    out_path: Path,
    vars_list: Sequence[str],
    q_low: float,
    q_high: float,
) -> None:
    others = [n for n in ordered_names if n != reference_name]
    n_vars = len(vars_list)
    n_models = len(others)
    fig, axes = plt.subplots(n_vars, n_models, figsize=(3.4 * n_models, 2.9 * n_vars), constrained_layout=True)
    if n_vars == 1:
        axes = np.array([axes])
    if n_models == 1:
        axes = axes[:, np.newaxis]

    ref_maps = maps_by_model[reference_name]
    for vi, var_name in enumerate(vars_list):
        diffs = [maps_by_model[m][var_name] - ref_maps[var_name] for m in others]
        vabs = max(abs(_robust_limits(diffs, q_low, q_high)[0]), abs(_robust_limits(diffs, q_low, q_high)[1]))
        im = None
        for mi, model_name in enumerate(others):
            ax = axes[vi, mi]
            diff = maps_by_model[model_name][var_name] - ref_maps[var_name]
            im = ax.imshow(diff, origin="lower", vmin=-vabs, vmax=vabs, cmap="coolwarm")
            if vi == 0:
                ax.set_title(f"{model_name} - {reference_name}", fontsize=10)
            if mi == 0:
                ax.set_ylabel(var_name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes[vi, :], shrink=0.86, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(title_prefix, fontsize=12)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot spatial distribution summary from compare CSV.")
    p.add_argument("--compare-csv", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--protocol", default="interp", choices=["interp", "extrap"])
    p.add_argument("--vars", nargs="+", required=True)
    p.add_argument("--model-names", nargs="+", required=True)
    p.add_argument("--reference-name", default="global_v2_frozen_ref")
    p.add_argument("--case-hash", default="", help="If empty, auto-select first common case.")
    p.add_argument("--mean-case-count", type=int, default=4)
    p.add_argument("--q-low", type=float, default=2.0)
    p.add_argument("--q-high", type=float, default=98.0)
    p.add_argument("--include-ground-truth", action="store_true")
    p.add_argument("--dataset-root", type=Path, default=Path("data/outputs_merged_td_csv_periodic"))
    p.add_argument("--index-csv", type=Path, default=Path("data/outputs_merged_td_csv_periodic/index.csv"))
    p.add_argument("--axis-mode", default="steady")
    p.add_argument("--geom-id", default="default")
    p.add_argument("--cond-columns", nargs="+", default=["PP0", "Td", "gamma"])
    p.add_argument("--axis-column", default="axis")
    p.add_argument("--fields-npz-column", default="fields_npz")
    args = p.parse_args()

    rows = _load_compare_rows(args.compare_csv)
    selected_rows = _select_rows(rows, args.model_names)
    single_dirs = [_inference_single_dir(r, args.protocol) for r in selected_rows]
    common_cases = _common_case_hashes(single_dirs)
    if not common_cases:
        raise RuntimeError("No common inference case hashes found across selected models.")

    casehash_to_fields: Dict[str, Path] | None = None
    if args.include_ground_truth:
        casehash_to_fields = _build_casehash_to_fieldspath(
            dataset_root=args.dataset_root.resolve(),
            index_csv=args.index_csv.resolve(),
            cond_columns=args.cond_columns,
            axis_column=args.axis_column,
            fields_npz_column=args.fields_npz_column,
            axis_mode=args.axis_mode,
            geom_id=args.geom_id,
        )
        common_cases = [c for c in common_cases if c in casehash_to_fields]
        if not common_cases:
            raise RuntimeError("No common inference case hashes with ground-truth mapping.")

    case_hash = args.case_hash or common_cases[0]
    if case_hash not in common_cases:
        raise ValueError(f"case-hash={case_hash} not in common set ({len(common_cases)} cases).")
    mean_cases = common_cases[: max(1, min(args.mean_case_count, len(common_cases)))]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ordered_names = [r["name"] for r in selected_rows]

    case_maps = _collect_case_maps(selected_rows, single_dirs, case_hash, args.vars)
    mean_maps = _collect_mean_maps(selected_rows, single_dirs, mean_cases, args.vars)
    gt_case_maps: Dict[str, np.ndarray] | None = None
    gt_mean_maps: Dict[str, np.ndarray] | None = None
    if args.include_ground_truth:
        assert casehash_to_fields is not None
        gt_case_maps = _collect_gt_case_maps(case_hash, casehash_to_fields, args.vars)
        gt_mean_maps = _collect_gt_mean_maps(mean_cases, casehash_to_fields, args.vars)

    case_out = args.out_dir / f"spatial_fields_{args.protocol}_case_{case_hash}_grid.png"
    _plot_per_var_grid(
        case_maps,
        ordered_names,
        f"{args.protocol} | Case={case_hash} | Field Comparison",
        case_out,
        args.vars,
        args.q_low,
        args.q_high,
        gt_maps=None,
    )
    if gt_case_maps is not None:
        case_gt_out = args.out_dir / f"spatial_fields_{args.protocol}_case_{case_hash}_with_gt_grid.png"
        _plot_per_var_grid(
            case_maps,
            ordered_names,
            f"{args.protocol} | Case={case_hash} | Field Comparison (with GT)",
            case_gt_out,
            args.vars,
            args.q_low,
            args.q_high,
            gt_maps=gt_case_maps,
        )

    mean_out = args.out_dir / f"spatial_fields_{args.protocol}_mean_{len(mean_cases)}cases_grid.png"
    _plot_per_var_grid(
        mean_maps,
        ordered_names,
        f"{args.protocol} | Mean over {len(mean_cases)} cases | Field Comparison",
        mean_out,
        args.vars,
        args.q_low,
        args.q_high,
        gt_maps=None,
    )
    if gt_mean_maps is not None:
        mean_gt_out = args.out_dir / f"spatial_fields_{args.protocol}_mean_{len(mean_cases)}cases_with_gt_grid.png"
        _plot_per_var_grid(
            mean_maps,
            ordered_names,
            f"{args.protocol} | Mean over {len(mean_cases)} cases | Field Comparison (with GT)",
            mean_gt_out,
            args.vars,
            args.q_low,
            args.q_high,
            gt_maps=gt_mean_maps,
        )

    if args.reference_name in ordered_names and len(ordered_names) > 1:
        case_diff_out = args.out_dir / f"spatial_fields_{args.protocol}_case_{case_hash}_diff_vs_{args.reference_name}.png"
        _plot_diff_vs_reference(
            case_maps,
            ordered_names,
            args.reference_name,
            f"{args.protocol} | Case={case_hash} | Difference vs {args.reference_name}",
            case_diff_out,
            args.vars,
            args.q_low,
            args.q_high,
        )
        mean_diff_out = args.out_dir / f"spatial_fields_{args.protocol}_mean_{len(mean_cases)}cases_diff_vs_{args.reference_name}.png"
        _plot_diff_vs_reference(
            mean_maps,
            ordered_names,
            args.reference_name,
            f"{args.protocol} | Mean {len(mean_cases)} cases | Difference vs {args.reference_name}",
            mean_diff_out,
            args.vars,
            args.q_low,
            args.q_high,
        )

    manifest = {
        "compare_csv": str(args.compare_csv),
        "protocol": args.protocol,
        "vars": args.vars,
        "model_names": ordered_names,
        "reference_name": args.reference_name,
        "selected_case_hash": case_hash,
        "mean_case_hashes": mean_cases,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "include_ground_truth": bool(args.include_ground_truth),
    }
    with (args.out_dir / f"spatial_fields_{args.protocol}_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"saved plots to: {args.out_dir}")
    for fn in sorted(args.out_dir.glob(f"spatial_fields_{args.protocol}_*.png")):
        print(fn.name)


if __name__ == "__main__":
    main()
