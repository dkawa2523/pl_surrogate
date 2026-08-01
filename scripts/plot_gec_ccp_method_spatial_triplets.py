#!/usr/bin/env python3
"""Plot GEC-CCP truth/prediction/error triplets per benchmark method."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
DEFAULT_TARGETS = ("ne", "ni", "Te", "phi")
def _build_truth_mapping(
    *,
    dataset_root: Path,
    index_csv: Path,
    cond_columns: list[str],
    axis_column: str,
    fields_npz_column: str,
    case_id_column: str,
    axis_mode: str,
    geom_id: str,
) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    geom_variants = [{"geom_id": str(geom_id)}, {"geom_id": str(geom_id), "geom_param": {}}]
    with index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cond = {name: float(row[name]) for name in cond_columns}
            axis_value = float(row[axis_column])
            fields_path = Path(row[fields_npz_column])
            if not fields_path.is_absolute():
                fields_path = dataset_root / fields_path
            case_id = str(row.get(case_id_column, "")).strip() or fields_path.stem
            meta = {
                "fields_path": fields_path.resolve(),
                "case_id": case_id,
                "cond": cond,
                "axis": axis_value,
            }
            for geom in geom_variants:
                key = _case_key_for_geom(cond, axis_value=axis_value, axis_mode=axis_mode, geom=geom)
                mapping[key] = dict(meta)
    return mapping


def _case_key_for_geom(cond: dict[str, float], axis_value: float, axis_mode: str, geom: dict[str, Any]) -> str:
    payload = {
        "cond": {k: float(np.float32(v)) for k, v in sorted(cond.items())},
        "axis": {"mode": str(axis_mode), "value": float(axis_value)},
        "geom": geom,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _single_dir(run_root: Path, size: int, model: str, protocol: str) -> Path:
    path = run_root / f"n{size}" / model / "models" / model / "eval_protocol" / protocol / "inference" / "single"
    if not path.exists():
        raise FileNotFoundError(f"inference single directory not found: {path}")
    return path


def _case_lookup(single_dir: Path, truth_mapping: dict[str, dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for child in single_dir.iterdir():
        if not child.is_dir() or not (child / "fields_phys.npz").exists():
            continue
        meta = truth_mapping.get(child.name)
        if meta is None:
            continue
        lookup[str(meta["case_id"])] = child.name
    return lookup


def _common_case_ids(lookups: list[dict[str, str]]) -> list[str]:
    sets = [set(lookup) for lookup in lookups]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def _load_pred(single_dir: Path, case_hash: str, target: str) -> np.ndarray:
    path = single_dir / case_hash / "fields_phys.npz"
    with np.load(path) as data:
        arr = np.asarray(data[target], dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{path}:{target} must be 2D or [1,H,W], got {arr.shape}")
    return arr


def _load_truth(fields_path: Path, target: str) -> np.ndarray:
    with np.load(fields_path) as data:
        if target in data.files:
            arr = np.asarray(data[target], dtype=np.float32)
        else:
            raise KeyError(f"target={target} not found in truth fields: {fields_path}")
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{fields_path}:{target} must be 2D or [1,H,W], got {arr.shape}")
    return arr


def _as_display(target: str, arr: np.ndarray) -> tuple[np.ndarray, str]:
    data = np.asarray(arr, dtype=np.float64)
    return data.astype(np.float32), target


def _finite_values(arrays: list[np.ndarray]) -> np.ndarray:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return vals


def _limits(arrays: list[np.ndarray], q_low: float, q_high: float) -> tuple[float, float]:
    vals = _finite_values(arrays)
    lo = float(np.percentile(vals, q_low))
    hi = float(np.percentile(vals, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return lo, hi


def _truth_by_case_id(truth_mapping: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for meta in truth_mapping.values():
        out[str(meta["case_id"])] = meta
    return out


def _mean_truth(truth_by_case: dict[str, dict[str, Any]], case_ids: list[str], target: str) -> np.ndarray:
    stack = [_load_truth(Path(truth_by_case[case_id]["fields_path"]), target) for case_id in case_ids]
    return np.mean(np.stack(stack, axis=0), axis=0).astype(np.float32)


def _mean_pred(single_dir: Path, lookup: dict[str, str], case_ids: list[str], target: str) -> np.ndarray:
    stack = [_load_pred(single_dir, lookup[case_id], target) for case_id in case_ids]
    return np.mean(np.stack(stack, axis=0), axis=0).astype(np.float32)


def _plot_triplet(
    *,
    model: str,
    protocol: str,
    size: int,
    label: str,
    case_label: str,
    targets: list[str],
    truth_by_target: dict[str, np.ndarray],
    pred_by_target: dict[str, np.ndarray],
    out_path: Path,
    q_low: float,
    q_high: float,
) -> None:
    nrows = len(targets)
    fig, axes = plt.subplots(nrows, 3, figsize=(9.6, max(3.0, 2.45 * nrows)), constrained_layout=True)
    if nrows == 1:
        axes = np.asarray([axes])

    for row_idx, target in enumerate(targets):
        truth_display, display_name = _as_display(target, truth_by_target[target])
        pred_display, _ = _as_display(target, pred_by_target[target])
        err_display = np.abs(pred_display - truth_display)

        field_vmin, field_vmax = _limits([truth_display, pred_display], q_low, q_high)
        err_vmin, err_vmax = _limits([err_display], 0.0, q_high)
        err_vmin = 0.0

        panels = (
            ("Truth", truth_display, "viridis", field_vmin, field_vmax),
            ("Prediction", pred_display, "viridis", field_vmin, field_vmax),
            ("Abs error", err_display, "magma", err_vmin, err_vmax),
        )
        for col_idx, (title, image, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(display_name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"n={size} {protocol} | {model} | {label} | {case_label}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_index(
    *,
    out_root: Path,
    generated: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# GEC-CCP Spatial Triplet Plots",
        "",
        "Each image shows Truth, Prediction, and absolute error.",
        "Density targets `ne` and `ni` are shown in physical linear scale; `Te` and `phi` are linear.",
        "",
        "## Manifest",
        "",
    ]
    for key in ("run_root", "sizes", "protocols", "models", "targets", "case_hash_by_size_protocol"):
        lines.append(f"- `{key}`: `{manifest.get(key)}`")
    lines.extend(["", "## Files", "", "| Size | Protocol | Model | Case plot | Mean plot |", "| --- | --- | --- | --- | --- |"])
    by_key: dict[tuple[int, str, str], dict[str, str]] = {}
    for item in generated:
        key = (int(item["size"]), str(item["protocol"]), str(item["model"]))
        by_key.setdefault(key, {})[str(item["label"])] = str(Path(item["path"]).relative_to(out_root))
    for key in sorted(by_key):
        size, protocol, model = key
        paths = by_key[key]
        case_path = paths.get("case", "")
        mean_path = paths.get("mean", "")
        case_link = f"[case]({case_path.replace(chr(92), '/')})" if case_path else ""
        mean_link = f"[mean]({mean_path.replace(chr(92), '/')})" if mean_path else ""
        lines.append(f"| {size} | {protocol} | `{model}` | {case_link} | {mean_link} |")
    (out_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/gec_ccp_200epoch_gpu_ne_20260622"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/outputs_merged_td_csv_periodic_ext0520"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=[78])
    parser.add_argument("--protocols", nargs="+", choices=["interp", "extrap"], default=["interp", "extrap"])
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--case-id", default="", help="Physical case_id to plot. If empty, use first common case.")
    parser.add_argument("--case-hash", default="")
    parser.add_argument("--mean-case-count", type=int, default=3)
    parser.add_argument("--q-low", type=float, default=2.0)
    parser.add_argument("--q-high", type=float, default=98.0)
    parser.add_argument("--cond-columns", nargs="+", default=["PP0", "Td", "gamma"])
    parser.add_argument("--axis-column", default="axis")
    parser.add_argument("--fields-npz-column", default="fields_npz")
    parser.add_argument("--case-id-column", default="case_id")
    parser.add_argument("--axis-mode", default="steady")
    parser.add_argument("--geom-id", default="default")
    args = parser.parse_args()

    out_root = args.out_dir or args.run_root / "summary" / "spatial_triplets"
    out_root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    case_hash_by_size_protocol: dict[str, dict[str, Any]] = {}

    for size in [int(v) for v in args.sizes]:
        truth_mapping = _build_truth_mapping(
            dataset_root=args.dataset_root.resolve(),
            index_csv=(args.dataset_root / f"index_{size}.csv").resolve(),
            cond_columns=[str(v) for v in args.cond_columns],
            axis_column=str(args.axis_column),
            fields_npz_column=str(args.fields_npz_column),
            case_id_column=str(args.case_id_column),
            axis_mode=str(args.axis_mode),
            geom_id=str(args.geom_id),
        )
        for protocol in [str(v) for v in args.protocols]:
            single_dirs = [_single_dir(args.run_root, size, str(model), protocol) for model in args.models]
            lookups = [_case_lookup(single_dir, truth_mapping) for single_dir in single_dirs]
            common_cases = _common_case_ids(lookups)
            if not common_cases:
                raise RuntimeError(f"no common physical cases found for size={size} protocol={protocol}")
            truth_by_case = _truth_by_case_id(truth_mapping)
            requested_case_id = str(args.case_id).strip()
            if not requested_case_id and str(args.case_hash).strip():
                hash_meta = truth_mapping.get(str(args.case_hash).strip())
                if hash_meta is None:
                    raise ValueError(f"case hash {args.case_hash} is not in truth mapping")
                requested_case_id = str(hash_meta["case_id"])
            case_id = requested_case_id or common_cases[0]
            if case_id not in common_cases:
                raise ValueError(f"case_id {case_id} is not common for size={size} protocol={protocol}")
            mean_cases = common_cases[: max(1, min(int(args.mean_case_count), len(common_cases)))]
            case_meta = truth_by_case[case_id]
            case_hash_by_model = {
                str(model): lookup[case_id] for model, lookup in zip([str(v) for v in args.models], lookups)
            }
            case_hash_by_size_protocol[f"n{size}_{protocol}"] = {
                "case_id": case_id,
                "case_hash_by_model": case_hash_by_model,
                "mean_case_ids": mean_cases,
                "common_case_count": len(common_cases),
            }

            truth_case = {
                target: _load_truth(Path(case_meta["fields_path"]), target) for target in [str(v) for v in args.targets]
            }
            truth_mean = {
                target: _mean_truth(truth_by_case, mean_cases, target) for target in [str(v) for v in args.targets]
            }
            for model, single_dir, lookup in zip([str(v) for v in args.models], single_dirs, lookups):
                pred_case = {
                    target: _load_pred(single_dir, lookup[case_id], target) for target in [str(v) for v in args.targets]
                }
                pred_mean = {
                    target: _mean_pred(single_dir, lookup, mean_cases, target) for target in [str(v) for v in args.targets]
                }

                case_out = out_root / f"n{size}" / protocol / model / f"{model}_{protocol}_case_triplet.png"
                _plot_triplet(
                    model=model,
                    protocol=protocol,
                    size=size,
                    label="case",
                    case_label=case_id,
                    targets=[str(v) for v in args.targets],
                    truth_by_target=truth_case,
                    pred_by_target=pred_case,
                    out_path=case_out,
                    q_low=float(args.q_low),
                    q_high=float(args.q_high),
                )
                generated.append({"size": size, "protocol": protocol, "model": model, "label": "case", "path": str(case_out)})

                mean_out = out_root / f"n{size}" / protocol / model / f"{model}_{protocol}_mean{len(mean_cases)}_triplet.png"
                _plot_triplet(
                    model=model,
                    protocol=protocol,
                    size=size,
                    label=f"mean{len(mean_cases)}",
                    case_label=";".join(mean_cases),
                    targets=[str(v) for v in args.targets],
                    truth_by_target=truth_mean,
                    pred_by_target=pred_mean,
                    out_path=mean_out,
                    q_low=float(args.q_low),
                    q_high=float(args.q_high),
                )
                generated.append({"size": size, "protocol": protocol, "model": model, "label": "mean", "path": str(mean_out)})

    manifest = {
        "run_root": str(args.run_root),
        "dataset_root": str(args.dataset_root),
        "out_dir": str(out_root),
        "sizes": [int(v) for v in args.sizes],
        "protocols": [str(v) for v in args.protocols],
        "models": [str(v) for v in args.models],
        "targets": [str(v) for v in args.targets],
        "mean_case_count": int(args.mean_case_count),
        "q_low": float(args.q_low),
        "q_high": float(args.q_high),
        "case_hash_by_size_protocol": case_hash_by_size_protocol,
        "generated_count": len(generated),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_index(out_root=out_root, generated=generated, manifest=manifest)
    print(f"generated_count={len(generated)}")
    print(f"out_dir={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
