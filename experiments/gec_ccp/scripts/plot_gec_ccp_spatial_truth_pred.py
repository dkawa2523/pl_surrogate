#!/usr/bin/env python3
"""Plot truth/prediction/error spatial fields for GEC-CCP benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "global_mlp",
    "deeponet_pod",
    "unet",
    "unetpp",
    "unetpp_attn",
    "unet_operator_v2",
    "fno",
    "ffno",
    "coord_mlp_fourier",
    "coord_mlp_siren",
    "coord_mlp_pod_residual",
    "u_no",
    "cno",
    "cno_operator_unet",
    "geom_deeponet_pod",
    "geom_deeponet_siren",
    "deeponet_plasma",
)

TARGETS = ("ne", "ni", "Te", "phi")
TARGET_SOURCE = {
    "ne": ("log_ne", "pow10"),
    "ni": ("log_ni", "pow10"),
    "Te": ("Te", "identity"),
    "phi": ("phi", "identity"),
}
QOI_MATCH_KEYS = (
    "uniformity",
    "uniformity_max_density",
    "uniformity_mean_density",
    "boundary_gamma_mean",
    "boundary_gamma_uniformity",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GEC-CCP truth/prediction/error spatial fields.")
    parser.add_argument("--run-root", default="runs/gec_ccp_trustworthy_v1")
    parser.add_argument("--dataset-root", default="data/outputs_merged_td_csv_periodic_ext0520")
    parser.add_argument("--out-dir", default="reports/gec_ccp_trustworthy_v1/spatial_truth_pred")
    parser.add_argument("--size", type=int, default=78)
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument("--splits", nargs="+", default=["interp", "extrap"])
    parser.add_argument("--case-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--no-mask", action="store_true", help="Do not mask non-plasma pixels in the plots.")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _close_enough(left: Any, right: Any) -> bool:
    a = _float(left)
    b = _float(right)
    if not (math.isfinite(a) and math.isfinite(b)):
        return str(left) == str(right)
    return abs(a - b) <= 1e-5 * max(1.0, abs(a), abs(b))


def _safe_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("_")


def _load_index(dataset_root: Path, size: int) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in _read_csv(dataset_root / f"index_{size}.csv")}


def _load_truth(dataset_root: Path, index_row: dict[str, str], targets: list[str]) -> dict[str, np.ndarray]:
    path = dataset_root / index_row["fields_npz"]
    out: dict[str, np.ndarray] = {}
    with np.load(path) as z:
        for target in targets:
            source, transform = TARGET_SOURCE[target]
            arr = np.asarray(z[source], dtype=np.float64)
            if transform == "pow10":
                arr = np.power(10.0, arr)
            out[target] = arr.astype(np.float64, copy=False)
    return out


def _load_prediction(single_dir: Path, targets: list[str]) -> dict[str, np.ndarray]:
    path = single_dir / "fields_phys.npz"
    if not path.exists():
        path = single_dir / "fields_model.npz"
    out: dict[str, np.ndarray] = {}
    with np.load(path) as z:
        for target in targets:
            out[target] = np.asarray(z[target], dtype=np.float64).squeeze()
    return out


def _match_single_dirs(split_dir: Path, batch_rows: list[dict[str, str]]) -> dict[int, Path]:
    single_root = split_dir / "inference" / "single"
    if not single_root.exists():
        return {}
    qois: list[tuple[Path, dict[str, Any]]] = []
    for qoi_path in single_root.glob("*/qoi.json"):
        qois.append((qoi_path.parent, _read_json(qoi_path)))

    used: set[Path] = set()
    out: dict[int, Path] = {}
    for idx, row in enumerate(batch_rows):
        best: Path | None = None
        best_score = -1
        for single_dir, qoi in qois:
            if single_dir in used:
                continue
            score = sum(1 for key in QOI_MATCH_KEYS if key in row and key in qoi and _close_enough(row[key], qoi[key]))
            if score > best_score:
                best = single_dir
                best_score = score
        if best is not None and best_score >= 3:
            out[idx] = best
            used.add(best)
    return out


def _display_arrays(
    target: str,
    truth: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    return truth, pred, pred - truth, target


def _masked(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if mask is None:
        return out
    return np.where(mask, out, np.nan)


def _finite_percentile(arrs: list[np.ndarray], qs: tuple[float, float]) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(arr)[np.isfinite(arr)].ravel() for arr in arrs if np.isfinite(arr).any()])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, qs)
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return float(lo), float(hi)


def _plot_case(
    *,
    truth: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    targets: list[str],
    mask: np.ndarray | None,
    extent: tuple[float, float, float, float] | None,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(len(targets), 3, figsize=(11.6, 2.65 * len(targets)), constrained_layout=True)
    if len(targets) == 1:
        axes = np.asarray([axes])
    for row_idx, target in enumerate(targets):
        truth_d, pred_d, err_d, label = _display_arrays(
            target,
            truth[target],
            pred[target],
        )
        truth_d = _masked(truth_d, mask)
        pred_d = _masked(pred_d, mask)
        err_d = _masked(err_d, mask)
        vmin, vmax = _finite_percentile([truth_d, pred_d], (1.0, 99.0))
        err_abs = np.nanpercentile(np.abs(err_d), 99.0) if np.isfinite(err_d).any() else 1.0
        if not math.isfinite(float(err_abs)) or err_abs <= 0:
            err_abs = 1.0
        panels = (
            ("truth", truth_d, "viridis", vmin, vmax),
            ("prediction", pred_d, "viridis", vmin, vmax),
            ("prediction - truth", err_d, "coolwarm", -float(err_abs), float(err_abs)),
        )
        for col_idx, (name, arr, cmap, lo, hi) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi, aspect="auto", extent=extent)
            ax.set_title(f"{target} {name}", fontsize=9)
            ax.set_xlabel("r")
            ax.set_ylabel("z")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(label if col_idx < 2 else f"delta {label}", fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _conditions_text(row: dict[str, str]) -> str:
    return f"PP0={float(row['PP0']):g}, Td={float(row['Td']):g}, gamma={float(row['gamma']):g}"


def _make_markdown(out_dir: Path, records: list[dict[str, str]]) -> None:
    lines = [
        "# Spatial Truth/Prediction Plots",
        "",
        "Each image shows `truth`, `prediction`, and `prediction - truth` for ne, ni, Te, and phi.",
        "Density fields ne/ni are shown in physical linear scale. Te and phi are shown in their native units.",
        "By default, non-plasma pixels are masked because the benchmark metrics are plasma-focused.",
        "",
        "| Size | Split | Model | Case | Conditions | Plot |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rec in records:
        rel = Path(rec["plot"]).relative_to(out_dir).as_posix()
        lines.append(
            f"| n{rec['size']} | {rec['split']} | {rec['model_id']} | {rec['case_id']} | "
            f"{rec['conditions']} | [{Path(rel).name}]({rel}) |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir) / f"n{int(args.size)}"
    targets = [str(t) for t in args.targets]
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"unknown target: {target}; expected one of {TARGETS}")

    index = _load_index(dataset_root, int(args.size))
    mask: np.ndarray | None = None
    if not args.no_mask:
        mask = np.asarray(np.load(dataset_root / "geometry" / "mask_plasma.npy"), dtype=bool)
    r = np.asarray(np.load(dataset_root / "geometry" / "r_coords.npy"), dtype=float)
    z = np.asarray(np.load(dataset_root / "geometry" / "z_coords.npy"), dtype=float)
    extent = (float(np.nanmin(r)), float(np.nanmax(r)), float(np.nanmin(z)), float(np.nanmax(z)))

    records: list[dict[str, str]] = []
    for model in [str(v) for v in args.models]:
        model_root = run_root / f"n{int(args.size)}" / model
        for split in [str(v) for v in args.splits]:
            split_json = model_root / "preprocessing" / "split" / f"split_{split}_v1.json"
            batch_csv = model_root / "models" / model / "eval_protocol" / split / "inference" / "batch" / "summary.csv"
            split_dir = model_root / "models" / model / "eval_protocol" / split
            if not split_json.exists() or not batch_csv.exists():
                continue
            split_payload = _read_json(split_json)
            test_cases = list(split_payload.get("test", []))
            batch_rows = _read_csv(batch_csv)
            single_by_row = _match_single_dirs(split_dir, batch_rows)
            for case_idx in [int(v) for v in args.case_indices]:
                if case_idx < 0 or case_idx >= min(len(test_cases), len(batch_rows)):
                    continue
                single_dir = single_by_row.get(case_idx)
                if single_dir is None:
                    continue
                case_id = str(test_cases[case_idx])
                index_row = index.get(case_id)
                if index_row is None:
                    continue
                truth = _load_truth(dataset_root, index_row, targets)
                pred = _load_prediction(single_dir, targets)
                cond_text = _conditions_text(index_row)
                title = f"n{args.size} {split} | {model} | {case_id} | {cond_text}"
                out_path = out_dir / split / model / f"{_safe_name(case_id)}_truth_pred_error.png"
                _plot_case(
                    truth=truth,
                    pred=pred,
                    targets=targets,
                    mask=mask,
                    extent=extent,
                    title=title,
                    out_path=out_path,
                )
                rec = {
                    "size": str(args.size),
                    "split": split,
                    "model_id": model,
                    "case_index": str(case_idx),
                    "case_id": case_id,
                    "conditions": cond_text,
                    "single_dir": str(single_dir),
                    "plot": str(out_path),
                }
                records.append(rec)
                print(out_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "spatial_truth_pred_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["size", "split", "model_id", "case_index", "case_id", "conditions", "single_dir", "plot"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    _make_markdown(out_dir, records)
    print(manifest)
    print(out_dir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
