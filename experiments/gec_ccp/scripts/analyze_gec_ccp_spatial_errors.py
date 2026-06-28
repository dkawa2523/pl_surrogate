#!/usr/bin/env python3
"""Compute physical-scale spatial error summaries for GEC-CCP benchmark predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import plot_gec_ccp_spatial_truth_pred as spatial_plots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GEC-CCP physical-scale spatial errors.")
    parser.add_argument("--run-root", default="runs/gec_ccp_trustworthy_v1")
    parser.add_argument("--dataset-root", default="data/outputs_merged_td_csv_periodic_ext0520")
    parser.add_argument("--out-dir", default="reports/gec_ccp_trustworthy_v1/spatial_truth_pred")
    parser.add_argument("--size", type=int, default=78)
    parser.add_argument("--models", nargs="+", default=list(spatial_plots.MODEL_ORDER))
    parser.add_argument("--splits", nargs="+", default=["interp", "extrap"])
    return parser.parse_args()


def _active_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float64)[mask]
    return vals[np.isfinite(vals)]


def _rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    diff = _active_values(np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64), mask)
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff * diff)))


def _mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(_active_values(np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64), mask))
    if diff.size == 0:
        return float("nan")
    return float(np.mean(diff))


def _p95_abs(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(_active_values(np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64), mask))
    if diff.size == 0:
        return float("nan")
    return float(np.percentile(diff, 95.0))


def _std(truth: np.ndarray, mask: np.ndarray) -> float:
    vals = _active_values(truth, mask)
    if vals.size < 2:
        return float("nan")
    value = float(np.std(vals))
    return value if np.isfinite(value) and value > 0.0 else float("nan")


def _negative_ratio(pred: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(pred, dtype=np.float64)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals < 0.0))


def _case_rows(
    *,
    run_root: Path,
    dataset_root: Path,
    size: int,
    models: list[str],
    splits: list[str],
    mask: np.ndarray,
) -> list[dict[str, object]]:
    index = spatial_plots._load_index(dataset_root, size)
    rows: list[dict[str, object]] = []
    for split in splits:
        for model in models:
            model_root = run_root / f"n{size}" / model
            split_json = model_root / "preprocessing" / "split" / f"split_{split}_v1.json"
            batch_csv = model_root / "models" / model / "eval_protocol" / split / "inference" / "batch" / "summary.csv"
            split_dir = model_root / "models" / model / "eval_protocol" / split
            if not split_json.exists() or not batch_csv.exists():
                continue
            split_payload = spatial_plots._read_json(split_json)
            test_cases = list(split_payload.get("test", []))
            batch_rows = spatial_plots._read_csv(batch_csv)
            single_by_row = spatial_plots._match_single_dirs(split_dir, batch_rows)
            for case_index, single_dir in sorted(single_by_row.items()):
                if case_index >= min(len(test_cases), len(batch_rows)):
                    continue
                case_id = str(test_cases[case_index])
                index_row = index.get(case_id)
                if index_row is None:
                    continue
                truth = spatial_plots._load_truth(dataset_root, index_row, list(spatial_plots.TARGETS))
                pred = spatial_plots._load_prediction(single_dir, list(spatial_plots.TARGETS))
                row: dict[str, object] = {
                    "size": size,
                    "split": split,
                    "model_id": model,
                    "case_index": case_index,
                    "case_id": case_id,
                }
                for target in spatial_plots.TARGETS:
                    truth_arr = truth[target]
                    pred_arr = pred[target]
                    std = _std(truth_arr, mask)
                    rmse = _rmse(pred_arr, truth_arr, mask)
                    row[f"rmse_{target}"] = rmse
                    row[f"nrmse_{target}"] = float(rmse / std) if np.isfinite(rmse) and np.isfinite(std) else float("nan")
                    row[f"mae_{target}"] = _mae(pred_arr, truth_arr, mask)
                    row[f"p95_abs_{target}"] = _p95_abs(pred_arr, truth_arr, mask)
                    row[f"neg_ratio_{target}"] = _negative_ratio(pred_arr, mask)
                rows.append(row)
    return rows


def _summary_rows(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for row in case_rows:
        key = (int(row["size"]), str(row["split"]), str(row["model_id"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    metric_keys = [
        f"{metric}_{target}"
        for target in spatial_plots.TARGETS
        for metric in ("rmse", "nrmse", "mae", "p95_abs", "neg_ratio")
    ]
    for (size, split, model), rows in sorted(groups.items()):
        out: dict[str, object] = {"size": size, "split": split, "model_id": model, "n_cases": len(rows)}
        for key in metric_keys:
            vals = np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out[f"mean_{key}"] = float(np.mean(vals)) if vals.size else float("nan")
            out[f"median_{key}"] = float(np.median(vals)) if vals.size else float("nan")
            out[f"max_{key}"] = float(np.max(vals)) if vals.size else float("nan")
        summaries.append(out)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["size", "split", "model_id", "case_index", "case_id", "n_cases"]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in set(preferred)]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir) / f"n{int(args.size)}"
    mask = np.asarray(np.load(dataset_root / "geometry" / "mask_plasma.npy"), dtype=bool)
    case_rows = _case_rows(
        run_root=Path(args.run_root),
        dataset_root=dataset_root,
        size=int(args.size),
        models=[str(v) for v in args.models],
        splits=[str(v) for v in args.splits],
        mask=mask,
    )
    summary_rows = _summary_rows(case_rows)
    case_path = out_dir / "spatial_physical_case_error_summary.csv"
    summary_path = out_dir / "spatial_physical_error_summary.csv"
    _write_csv(case_path, case_rows)
    _write_csv(summary_path, summary_rows)
    print(case_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
