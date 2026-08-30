#!/usr/bin/env python
"""Evaluate the frozen ICP v45 model ladder on unknown coil combinations.

The script is intentionally isolated from the training package.  It performs
physical-space inference on the frozen v43 G4 test cases and writes auditable
case-level, family-level and structural-response metrics.  COMSOL arrays are
read only from the frozen dataset; no response-derived field is used as an
input feature.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

DATASET_ROOT = ROOT / "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45"
RUN_ROOT = ROOT / "runs/icp_conference_continuity_v45/final"
OUT_ROOT = ROOT / "reports/icp_conference_materials/conference_continuity_v45"
SEED = 1237
TARGETS = ("ne", "ni", "Te", "phi")
UNKNOWN_FAMILIES = (
    "unknown_gap_topology",
    "unseen_rank_height_size",
    "coupled_transform",
)
VARIANTS = (
    "conference_dimension",
    "explicit_dimension",
    "conference_sdf",
    "sdf_vacuum",
    "sdf_vacuum_structure",
)
REPRESENTATIVE_CASES = (
    "v43_te_n4_variant_1__center",
    "v43_te_n4_variant_2__center",
    "v43_te_n4_variant_3__center",
)
E_CHARGE = 1.602176634e-19
ATOMIC_MASS = 1.66053906660e-27
ARGON_ION_MASS = 39.948 * ATOMIC_MASS


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _squeeze_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    while array.ndim > 2 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected 2-D field, got {array.shape}")
    return array


def _relative_l2(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    reference = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if mask is not None:
        active = np.asarray(mask, dtype=bool)
        reference = reference[active]
        estimate = estimate[active]
    finite = np.isfinite(reference) & np.isfinite(estimate)
    reference = reference[finite]
    estimate = estimate[finite]
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(estimate - reference) / denominator) if denominator > 0.0 else float("nan")


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


def _gain(reference_delta: np.ndarray, predicted_delta: np.ndarray) -> float:
    denominator = float(np.linalg.norm(np.asarray(reference_delta, dtype=np.float64)))
    return float(np.linalg.norm(np.asarray(predicted_delta, dtype=np.float64)) / denominator) if denominator > 0.0 else float("nan")


def _wafer_selector(mask: np.ndarray, *, layers: int = 10) -> np.ndarray:
    plasma = np.asarray(mask, dtype=bool)
    first = np.asarray(
        [np.flatnonzero(plasma[:, column])[0] if np.any(plasma[:, column]) else -1 for column in range(plasma.shape[1])]
    )
    global_first = int(np.min(first[first >= 0]))
    raised = first > global_first
    surface_rows = np.unique(first[raised])
    if surface_rows.size != 1:
        raise ValueError(f"wafer surface must occupy one row, got {surface_rows.tolist()}")
    row0 = int(surface_rows[0])
    selector = np.zeros_like(plasma, dtype=bool)
    for column in np.flatnonzero(raised):
        stop = min(row0 + max(1, int(layers)), plasma.shape[0])
        selector[row0:stop, column] = plasma[row0:stop, column]
    return selector


def _bohm_profile(
    ni: np.ndarray,
    te: np.ndarray,
    mask: np.ndarray,
    r_coords: np.ndarray,
    *,
    radial_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    density = np.maximum(np.asarray(ni, dtype=np.float64), 1.0e8)
    temperature = np.maximum(np.asarray(te, dtype=np.float64), 0.05)
    bohm = density * np.sqrt(temperature * (E_CHARGE / ARGON_ION_MASS))
    selector = _wafer_selector(mask)
    columns = np.any(selector, axis=0)
    counts = np.sum(selector[:, columns], axis=0).astype(np.float64)
    column_flux = np.sum(np.where(selector, bohm, 0.0), axis=0)[columns] / counts
    r_grid = np.asarray(r_coords, dtype=np.float64)
    radial = (r_grid[0] if r_grid.ndim == 2 else r_grid)[columns]
    lower = float(np.min(radial))
    upper = float(np.max(radial))
    edges = np.sqrt(np.linspace(lower**2, upper**2, int(radial_bins) + 1))
    bin_index = np.clip(np.digitize(radial, edges[1:-1]), 0, int(radial_bins) - 1)
    radial_weights = np.maximum(radial, 0.5 * float(np.median(np.diff(radial))))
    profile = np.empty(int(radial_bins), dtype=np.float64)
    for index in range(int(radial_bins)):
        selected = bin_index == index
        if not np.any(selected):
            raise ValueError(f"empty equal-area radial bin {index}")
        profile[index] = float(np.average(column_flux[selected], weights=radial_weights[selected]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_flux = float(np.mean(profile))
    uniformity = float(100.0 * np.max(np.abs(profile / max(mean_flux, 1.0) - 1.0)))
    return centers, profile, bohm, uniformity


def _case_spatial_features(cfg: dict[str, Any], bundle: Any, artifacts: dict[str, Any], dataset: Any) -> Any:
    from plasma_surrogate.preprocessing.spatial_features import (
        apply_coord_feature_scaling,
        apply_distance_transform,
        build_case_spatial_features,
        build_coord_feature_rows,
        resolve_coord_feature_channels,
        resolve_distance_transform_cfg,
        resolve_distance_transform_effective,
    )

    input_cfg = dict(dict(cfg.get("train", {})).get("u_no", {}).get("input_features", {}) or {})
    channels = resolve_coord_feature_channels(input_cfg.get("features"))
    distance_cfg = resolve_distance_transform_cfg(input_cfg.get("distance_transform"))
    distance_cfg, _ = resolve_distance_transform_effective(
        distance_cfg, stats=artifacts["distance_transform_stats"]
    )
    h, w = int(dataset.shape[0]), int(dataset.shape[1])
    expected_case_ids = [str(case["case_id"]) for case in dataset.cases]
    source, _ = build_case_spatial_features(
        channels=channels,
        static_pack=bundle.schemas.get("static_spatial_feature_pack"),
        case_pack=bundle.schemas.get("case_structure_feature_pack"),
        h=h,
        w=w,
        distance_transform_cfg=distance_cfg,
        coord_feature_scaler_artifact=artifacts["coord_feature_scaler"],
        expected_case_ids=expected_case_ids,
    )
    if source is not None:
        return source
    rows, _ = build_coord_feature_rows(
        channels=channels,
        pack=bundle.schemas.get("coord_feature_pack"),
        geom_ctx=None,
        h=h,
        w=w,
    )
    rows, _ = apply_distance_transform(rows, channels=channels, cfg=distance_cfg)
    rows, _, _ = apply_coord_feature_scaling(
        rows,
        channels=channels,
        coord_feature_scaler_artifact=artifacts["coord_feature_scaler"],
        distance_transform_cfg=distance_cfg,
    )
    return rows.reshape(1, h, w, len(channels)).astype(np.float32)


def _spatial_batch(source: Any, indices: np.ndarray) -> np.ndarray:
    if hasattr(source, "batch"):
        return source.batch(indices)
    return np.repeat(np.asarray(source, dtype=np.float32), int(indices.size), axis=0)


def _load_geometry(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = Path(str(case["structure_npz"]))
    path = raw if raw.is_absolute() else DATASET_ROOT / raw
    with np.load(path, allow_pickle=False) as pack:
        return (
            np.asarray(pack["mask_plasma"], dtype=np.float32) > 0.5,
            np.asarray(pack["r_coords"], dtype=np.float64),
            np.asarray(pack["z_coords"], dtype=np.float64),
        )


def _predict_variant(variant: str, selected_ids: set[str], *, batch_size: int) -> dict[str, dict[str, Any]]:
    from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
    from plasma_surrogate.core.dataset_io import load_dataset
    from plasma_surrogate.core.input_modes import load_checkpoint_metadata_with_input_mode
    from plasma_surrogate.core.run_bundle import RunBundleLoader
    from plasma_surrogate.models.checkpoint import load_checkpoint

    run = RUN_ROOT / variant / f"seed_{SEED}"
    checkpoint = run / "models/u_no/eval_protocol/interp/checkpoints"
    if not (run / "leaderboard.csv").is_file():
        raise FileNotFoundError(f"training is incomplete: {run}")
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    dataset = load_dataset({"dataset": cfg["dataset"]}, run)
    lookup = {str(case["case_id"]): index for index, case in enumerate(dataset.cases)}
    missing = selected_ids - set(lookup)
    if missing:
        raise KeyError(f"{variant} is missing cases: {sorted(missing)[:5]}")
    model = load_checkpoint(checkpoint)
    checkpoint_meta, _ = load_checkpoint_metadata_with_input_mode(checkpoint / "meta.json")
    bundle = RunBundleLoader.load(run, model=model)
    transforms = bundle.transform_bundle_for_checkpoint(checkpoint_meta)
    artifacts = bundle.spatial_transform_artifacts_for_checkpoint(checkpoint_meta)
    cond = build_cond_matrix_with_axis(dataset.cases, bundle.cond_schema_obj(), bundle.axis_schema_obj())
    cond_scaled = transforms.transform_cond(cond)
    spatial_source = _case_spatial_features(cfg, bundle, artifacts, dataset)
    ordered = sorted((lookup[case_id], case_id) for case_id in selected_ids)
    predictions: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ordered), int(batch_size)):
        chunk = ordered[start : start + int(batch_size)]
        indices = np.asarray([index for index, _ in chunk], dtype=np.int64)
        spatial = _spatial_batch(spatial_source, indices)
        raw = model.forward_features(cond_scaled[indices], spatial_features=spatial)
        physical = transforms.inverse_field_dict(
            {name: np.asarray(values, dtype=np.float32) for name, values in raw.items()}
        )
        for offset, (index, case_id) in enumerate(chunk):
            case = dict(dataset.cases[index])
            mask, r_coords, z_coords = _load_geometry(case)
            truth = {name: _squeeze_2d(case["y"][name]) for name in TARGETS}
            prediction = {name: _squeeze_2d(physical[name][offset]) for name in TARGETS}
            predictions[case_id] = {
                "truth": truth,
                "prediction": prediction,
                "mask": mask,
                "r_coords": r_coords,
                "z_coords": z_coords,
            }
        print(f"{variant}: {min(start + batch_size, len(ordered))}/{len(ordered)}", flush=True)
    del model, bundle, dataset, spatial_source
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return predictions


def _bootstrap_ci(values: list[float], *, seed: int, repeats: int = 2000) -> tuple[float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(int(repeats), array.size), replace=True)
    medians = np.median(sampled, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _summaries(case_rows: list[dict[str, Any]], response_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    joined = [("case", row) for row in case_rows] + [("response", row) for row in response_rows]
    metric_names = {
        "case": (
            "ne_rel_l2_pct",
            "ni_rel_l2_pct",
            "Te_rel_l2_pct",
            "phi_rel_l2_pct",
            "bohm_profile_rel_l2_pct",
            "bohm_uniformity_abs_error_pp",
        ),
        "response": (
            "ni_delta_rel_l2_pct",
            "ni_delta_cosine",
            "ni_delta_gain",
            "bohm_delta_rel_l2_pct",
            "bohm_delta_cosine",
            "bohm_delta_gain",
        ),
    }
    rows: list[dict[str, Any]] = []
    for source in ("case", "response"):
        source_rows = [row for row_source, row in joined if row_source == source]
        for model in VARIANTS:
            for family in (*UNKNOWN_FAMILIES, "all_unknown"):
                selected = [
                    row for row in source_rows
                    if row["model"] == model and (family == "all_unknown" or row["layout_kind"] == family)
                ]
                if not selected:
                    continue
                for metric in metric_names[source]:
                    values = [float(row[metric]) for row in selected if np.isfinite(float(row[metric]))]
                    low, high = _bootstrap_ci(values, seed=SEED + len(rows))
                    rows.append(
                        {
                            "source": source,
                            "model": model,
                            "layout_kind": family,
                            "metric": metric,
                            "n": len(values),
                            "median": float(np.median(values)),
                            "p90": float(np.quantile(values, 0.9)),
                            "mean": float(np.mean(values)),
                            "bootstrap_median_ci_low": low,
                            "bootstrap_median_ci_high": high,
                        }
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--variants", nargs="*", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = _read_csv(DATASET_ROOT / "design_metadata_v43.csv")
    metadata = {row["case_id"]: row for row in metadata_rows}
    membership = _read_json(DATASET_ROOT / "split_membership.json")
    test_ids = {str(value) for value in membership["test"]}
    selected_ids = {
        case_id for case_id, row in metadata.items()
        if case_id in test_ids and row["stage"] == "G4"
        and row["layout_kind"] in (*UNKNOWN_FAMILIES, "regular_anchor")
    }
    unknown_ids = {case_id for case_id in selected_ids if metadata[case_id]["layout_kind"] in UNKNOWN_FAMILIES}
    if len(unknown_ids) != 75 or len(selected_ids) != 100:
        raise ValueError(f"frozen G4 contract mismatch: unknown={len(unknown_ids)}, all={len(selected_ids)}")

    predictions_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in args.variants:
        predictions_by_model[variant] = _predict_variant(
            variant, selected_ids, batch_size=max(1, int(args.batch_size))
        )

    case_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    representative_payload: dict[str, np.ndarray] = {}
    for variant, case_predictions in predictions_by_model.items():
        for case_id in sorted(unknown_ids):
            payload = case_predictions[case_id]
            row: dict[str, Any] = {
                "model": variant,
                "case_id": case_id,
                "layout_kind": metadata[case_id]["layout_kind"],
                "operation_id": metadata[case_id]["operation_id"],
                "design_family_id": metadata[case_id]["design_family_id"],
            }
            for target in TARGETS:
                row[f"{target}_rel_l2_pct"] = 100.0 * _relative_l2(
                    payload["truth"][target], payload["prediction"][target], payload["mask"]
                )
            centers, truth_profile, truth_bohm, truth_uniformity = _bohm_profile(
                payload["truth"]["ni"], payload["truth"]["Te"], payload["mask"], payload["r_coords"]
            )
            _, prediction_profile, prediction_bohm, prediction_uniformity = _bohm_profile(
                payload["prediction"]["ni"], payload["prediction"]["Te"], payload["mask"], payload["r_coords"]
            )
            row["bohm_profile_rel_l2_pct"] = 100.0 * _relative_l2(truth_profile, prediction_profile)
            row["bohm_uniformity_truth_pct"] = truth_uniformity
            row["bohm_uniformity_prediction_pct"] = prediction_uniformity
            row["bohm_uniformity_abs_error_pp"] = abs(prediction_uniformity - truth_uniformity)
            case_rows.append(row)

            layout_id = metadata[case_id]["layout_id"]
            coil_count = int(layout_id.split("_n", 1)[1].split("_", 1)[0])
            anchor_id = f"v43_te_n{coil_count}_anchor__{metadata[case_id]['operation_id']}"
            anchor = case_predictions[anchor_id]
            _, truth_anchor_profile, truth_anchor_bohm, _ = _bohm_profile(
                anchor["truth"]["ni"], anchor["truth"]["Te"], anchor["mask"], anchor["r_coords"]
            )
            _, pred_anchor_profile, pred_anchor_bohm, _ = _bohm_profile(
                anchor["prediction"]["ni"], anchor["prediction"]["Te"], anchor["mask"], anchor["r_coords"]
            )
            truth_delta_ni = (payload["truth"]["ni"] - anchor["truth"]["ni"])[payload["mask"]]
            pred_delta_ni = (payload["prediction"]["ni"] - anchor["prediction"]["ni"])[payload["mask"]]
            truth_delta_profile = truth_profile - truth_anchor_profile
            pred_delta_profile = prediction_profile - pred_anchor_profile
            response_rows.append(
                {
                    "model": variant,
                    "case_id": case_id,
                    "anchor_case_id": anchor_id,
                    "layout_kind": metadata[case_id]["layout_kind"],
                    "operation_id": metadata[case_id]["operation_id"],
                    "design_family_id": metadata[case_id]["design_family_id"],
                    "ni_truth_delta_over_anchor_pct": 100.0
                    * _gain(anchor["truth"]["ni"][payload["mask"]], truth_delta_ni),
                    "ni_delta_rel_l2_pct": 100.0 * _relative_l2(truth_delta_ni, pred_delta_ni),
                    "ni_delta_cosine": _cosine(truth_delta_ni, pred_delta_ni),
                    "ni_delta_gain": _gain(truth_delta_ni, pred_delta_ni),
                    "bohm_truth_delta_over_anchor_pct": 100.0
                    * _gain(truth_anchor_profile, truth_delta_profile),
                    "bohm_delta_rel_l2_pct": 100.0 * _relative_l2(truth_delta_profile, pred_delta_profile),
                    "bohm_delta_cosine": _cosine(truth_delta_profile, pred_delta_profile),
                    "bohm_delta_gain": _gain(truth_delta_profile, pred_delta_profile),
                }
            )
            if case_id in REPRESENTATIVE_CASES:
                prefix = f"{variant}__{case_id}"
                representative_payload[f"{prefix}__ni_prediction"] = np.asarray(payload["prediction"]["ni"], dtype=np.float32)
                representative_payload[f"{prefix}__Te_prediction"] = np.asarray(payload["prediction"]["Te"], dtype=np.float32)
                representative_payload[f"{prefix}__bohm_profile_prediction"] = np.asarray(prediction_profile, dtype=np.float32)
                representative_payload[f"{prefix}__bohm_profile_truth"] = np.asarray(truth_profile, dtype=np.float32)
                representative_payload[f"{prefix}__radial_centers"] = np.asarray(centers, dtype=np.float32)
                representative_payload[f"{prefix}__ni_truth"] = np.asarray(payload["truth"]["ni"], dtype=np.float32)
                representative_payload[f"{prefix}__mask"] = np.asarray(payload["mask"], dtype=np.uint8)
                representative_payload[f"{prefix}__r_coords"] = np.asarray(payload["r_coords"], dtype=np.float32)
                representative_payload[f"{prefix}__z_coords"] = np.asarray(payload["z_coords"], dtype=np.float32)

    summaries = _summaries(case_rows, response_rows)
    _write_csv(out_dir / "case_metrics_unknown75.csv", case_rows)
    _write_csv(out_dir / "structural_response_metrics_unknown75.csv", response_rows)
    _write_csv(out_dir / "summary_metrics.csv", summaries)
    np.savez_compressed(out_dir / "representative_predictions.npz", **representative_payload)

    audit = {
        "study": "icp_conference_continuity_v45",
        "seed": SEED,
        "dataset": str(DATASET_ROOT.relative_to(ROOT)),
        "models": list(args.variants),
        "selected_case_count": len(selected_ids),
        "unknown_case_count": len(unknown_ids),
        "unknown_family_counts": {
            family: sum(metadata[case_id]["layout_kind"] == family for case_id in unknown_ids)
            for family in UNKNOWN_FAMILIES
        },
        "metrics": {
            "field_error": "plasma-masked physical relative L2",
            "bohm_profile": "10-layer wafer band, 20 equal-area radial bins",
            "response_anchor": "same coil count and operating condition regular_anchor",
            "uncertainty": "case-bootstrap 95% CI of family median, 2000 resamples",
        },
        "response_leakage": False,
        "magnetic_feature": "deterministic unit-current vacuum field only",
        "representative_cases": list(REPRESENTATIVE_CASES),
    }
    (out_dir / "evaluation_contract.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
