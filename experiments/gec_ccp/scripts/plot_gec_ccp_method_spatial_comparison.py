#!/usr/bin/env python3
"""Plot cross-model spatial comparison grids for GEC-CCP benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_gec_ccp_spatial_truth_pred as spatial_base  # noqa: E402


MODEL_ORDER = spatial_base.MODEL_ORDER
TARGETS = spatial_base.TARGETS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-case spatial fields across GEC-CCP methods.")
    parser.add_argument("--run-root", default="runs/gec_ccp_spatial_huber_v1")
    parser.add_argument("--dataset-root", default="data/outputs_merged_td_csv_periodic_ext0520")
    parser.add_argument("--summary-csv", default="reports/gec_ccp_spatial_huber_v1/summary_trustworthy_27_54_78.csv")
    parser.add_argument("--out-dir", default="reports/gec_ccp_spatial_huber_v1/spatial_method_comparison")
    parser.add_argument("--size", type=int, default=78)
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument("--splits", nargs="+", default=["interp", "extrap"])
    parser.add_argument("--case-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--sort-by-summary", action="store_true", default=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _float(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _ordered_models(models: list[str], *, summary_csv: Path, size: int, sort_by_summary: bool) -> list[str]:
    requested = [str(model) for model in models]
    if not sort_by_summary or not summary_csv.exists():
        return requested
    rows = [
        row
        for row in _read_csv(summary_csv)
        if int(float(row.get("dataset_size", -1))) == int(size) and str(row.get("model_id", "")) in set(requested)
    ]
    score_by_model = {str(row["model_id"]): _float(row.get("surrogate_quality_score")) for row in rows}
    return sorted(
        requested,
        key=lambda model: (
            0 if math.isfinite(score_by_model.get(model, float("nan"))) else 1,
            score_by_model.get(model, float("inf")),
            requested.index(model),
        ),
    )


def _score_labels(models: list[str], *, summary_csv: Path, size: int) -> dict[str, str]:
    rows = [
        row
        for row in _read_csv(summary_csv)
        if int(float(row.get("dataset_size", -1))) == int(size) and str(row.get("model_id", "")) in set(models)
    ]
    out: dict[str, str] = {}
    for row in rows:
        score = _float(row.get("surrogate_quality_score"))
        r2 = _float(row.get("test_r2_plasma_mean_dual"))
        pieces = [str(row["model_id"])]
        if math.isfinite(score):
            pieces.append(f"Q={score:.3f}")
        if math.isfinite(r2):
            pieces.append(f"R2={r2:.3f}")
        out[str(row["model_id"])] = "\n".join(pieces)
    return out


def _split_payload_for_model(run_root: Path, size: int, model: str, split: str) -> tuple[list[str], dict[int, Path]]:
    model_root = run_root / f"n{int(size)}" / model
    split_json = model_root / "preprocessing" / "split" / f"split_{split}_v1.json"
    batch_csv = model_root / "models" / model / "eval_protocol" / split / "inference" / "batch" / "summary.csv"
    split_dir = model_root / "models" / model / "eval_protocol" / split
    if not split_json.exists() or not batch_csv.exists():
        return [], {}
    split_payload = spatial_base._read_json(split_json)
    test_cases = [str(v) for v in list(split_payload.get("test", []))]
    batch_rows = spatial_base._read_csv(batch_csv)
    return test_cases, spatial_base._match_single_dirs(split_dir, batch_rows)


def _case_ids_from_reference(
    *,
    run_root: Path,
    size: int,
    models: list[str],
    split: str,
    case_indices: list[int],
) -> list[tuple[int, str]]:
    for model in models:
        test_cases, _ = _split_payload_for_model(run_root, size, model, split)
        if test_cases:
            out: list[tuple[int, str]] = []
            for case_idx in case_indices:
                if 0 <= int(case_idx) < len(test_cases):
                    out.append((int(case_idx), str(test_cases[int(case_idx)])))
            return out
    return []


def _load_model_predictions(
    *,
    run_root: Path,
    size: int,
    models: list[str],
    split: str,
    case_id: str,
    target: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for model in models:
        test_cases, single_by_row = _split_payload_for_model(run_root, size, model, split)
        if not test_cases:
            continue
        try:
            row_idx = test_cases.index(str(case_id))
        except ValueError:
            continue
        single_dir = single_by_row.get(row_idx)
        if single_dir is None:
            continue
        try:
            pred = spatial_base._load_prediction(single_dir, [target])[target]
        except (FileNotFoundError, KeyError):
            continue
        out[model] = np.asarray(pred, dtype=np.float64)
    return out


def _finite_percentile(arrs: list[np.ndarray], qs: tuple[float, float]) -> tuple[float, float]:
    vals = [np.asarray(arr)[np.isfinite(arr)].ravel() for arr in arrs if np.isfinite(arr).any()]
    if not vals:
        return 0.0, 1.0
    flat = np.concatenate(vals)
    lo, hi = np.percentile(flat, qs)
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or float(lo) == float(hi):
        lo = float(np.nanmin(flat))
        hi = float(np.nanmax(flat))
    if float(lo) == float(hi):
        lo -= 1.0
        hi += 1.0
    return float(lo), float(hi)


def _grid_shape(n_panels: int) -> tuple[int, int]:
    ncols = min(5, max(1, int(n_panels)))
    nrows = int(math.ceil(float(n_panels) / float(ncols)))
    return nrows, ncols


def _format_axes(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _save_prediction_grid(
    *,
    truth: np.ndarray,
    preds: dict[str, np.ndarray],
    labels: dict[str, str],
    target: str,
    case_title: str,
    mask: np.ndarray | None,
    out_path: Path,
) -> None:
    truth_d, _, _, label = spatial_base._display_arrays(target, truth, truth)
    truth_d = spatial_base._masked(truth_d, mask)
    pred_display: dict[str, np.ndarray] = {}
    for model, pred in preds.items():
        _, pred_d, _, _ = spatial_base._display_arrays(target, truth, pred)
        pred_display[model] = spatial_base._masked(pred_d, mask)
    vmin, vmax = _finite_percentile([truth_d, *pred_display.values()], (1.0, 99.0))

    panels = [("truth", truth_d), *[(model, pred_display[model]) for model in preds]]
    nrows, ncols = _grid_shape(len(panels))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    last_im = None
    for idx, (name, arr) in enumerate(panels):
        ax = axes_arr[idx]
        last_im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(labels.get(name, name), fontsize=8)
        _format_axes(ax)
    for ax in axes_arr[len(panels) :]:
        ax.axis("off")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr[: len(panels)].tolist(), fraction=0.025, pad=0.01)
        cbar.set_label(label)
    fig.suptitle(f"{case_title} | {target} prediction comparison", fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_error_grid(
    *,
    truth: np.ndarray,
    preds: dict[str, np.ndarray],
    labels: dict[str, str],
    target: str,
    case_title: str,
    mask: np.ndarray | None,
    out_path: Path,
) -> None:
    error_display: dict[str, np.ndarray] = {}
    for model, pred in preds.items():
        _, _, err_d, label = spatial_base._display_arrays(target, truth, pred)
        error_display[model] = spatial_base._masked(err_d, mask)
    err_abs = np.nanpercentile(
        np.concatenate([np.abs(arr[np.isfinite(arr)]).ravel() for arr in error_display.values() if np.isfinite(arr).any()]),
        99.0,
    ) if error_display else 1.0
    if not math.isfinite(float(err_abs)) or float(err_abs) <= 0.0:
        err_abs = 1.0

    nrows, ncols = _grid_shape(len(error_display))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    last_im = None
    for idx, (model, arr) in enumerate(error_display.items()):
        ax = axes_arr[idx]
        last_im = ax.imshow(arr, origin="lower", cmap="coolwarm", vmin=-float(err_abs), vmax=float(err_abs), aspect="auto")
        ax.set_title(labels.get(model, model), fontsize=8)
        _format_axes(ax)
    for ax in axes_arr[len(error_display) :]:
        ax.axis("off")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr[: len(error_display)].tolist(), fraction=0.025, pad=0.01)
        cbar.set_label(f"prediction - truth ({label})")
    fig.suptitle(f"{case_title} | {target} error comparison", fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_readme(out_dir: Path, records: list[dict[str, str]]) -> None:
    lines = [
        "# Spatial Method Comparison",
        "",
        "Each prediction image uses a shared color scale across truth and all model predictions for the same case/target.",
        "Each error image uses a shared symmetric color scale across all model error maps for the same case/target.",
        "",
        "| Size | Split | Case | Target | Predictions | Errors |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rec in records:
        pred_rel = Path(rec["prediction_plot"]).relative_to(out_dir).as_posix()
        err_rel = Path(rec["error_plot"]).relative_to(out_dir).as_posix()
        lines.append(
            f"| n{rec['size']} | {rec['split']} | {rec['case_id']} | {rec['target']} | "
            f"[prediction]({pred_rel}) | [error]({err_rel}) |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)
    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir) / f"n{int(args.size)}"
    models = _ordered_models(
        [str(v) for v in args.models],
        summary_csv=summary_csv,
        size=int(args.size),
        sort_by_summary=bool(args.sort_by_summary),
    )
    labels = _score_labels(models, summary_csv=summary_csv, size=int(args.size))
    labels["truth"] = "truth"
    targets = [str(v) for v in args.targets]
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"unknown target: {target}; expected one of {TARGETS}")

    index = spatial_base._load_index(dataset_root, int(args.size))
    mask: np.ndarray | None = None
    if not args.no_mask:
        mask = np.asarray(np.load(dataset_root / "geometry" / "mask_plasma.npy"), dtype=bool)

    records: list[dict[str, str]] = []
    for split in [str(v) for v in args.splits]:
        selected_cases = _case_ids_from_reference(
            run_root=run_root,
            size=int(args.size),
            models=models,
            split=split,
            case_indices=[int(v) for v in args.case_indices],
        )
        for case_idx, case_id in selected_cases:
            index_row = index.get(str(case_id))
            if index_row is None:
                continue
            cond_text = spatial_base._conditions_text(index_row)
            for target in targets:
                truth = spatial_base._load_truth(dataset_root, index_row, [target])[target]
                preds = _load_model_predictions(
                    run_root=run_root,
                    size=int(args.size),
                    models=models,
                    split=split,
                    case_id=str(case_id),
                    target=target,
                )
                if not preds:
                    continue
                safe_case = spatial_base._safe_name(str(case_id))
                case_title = f"n{args.size} {split} case#{case_idx} | {case_id} | {cond_text}"
                pred_path = out_dir / split / safe_case / f"{target}_prediction_methods.png"
                err_path = out_dir / split / safe_case / f"{target}_error_methods.png"
                _save_prediction_grid(
                    truth=truth,
                    preds=preds,
                    labels=labels,
                    target=target,
                    case_title=case_title,
                    mask=mask,
                    out_path=pred_path,
                )
                _save_error_grid(
                    truth=truth,
                    preds=preds,
                    labels=labels,
                    target=target,
                    case_title=case_title,
                    mask=mask,
                    out_path=err_path,
                )
                records.append(
                    {
                        "size": str(args.size),
                        "split": split,
                        "case_index": str(case_idx),
                        "case_id": str(case_id),
                        "conditions": cond_text,
                        "target": target,
                        "models": ";".join(preds.keys()),
                        "prediction_plot": str(pred_path),
                        "error_plot": str(err_path),
                    }
                )
                print(pred_path)
                print(err_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "spatial_method_comparison_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "size",
            "split",
            "case_index",
            "case_id",
            "conditions",
            "target",
            "models",
            "prediction_plot",
            "error_plot",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    _write_readme(out_dir, records)
    print(manifest)
    print(out_dir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
