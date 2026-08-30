#!/usr/bin/env python
"""Evaluate formal ICP references and every isolated v46 UNO candidate.

The frozen v43 G4 contract is reused verbatim: 75 unknown structures plus the
25 matched regular anchors required to measure structural response.  Models
are evaluated one at a time so the candidate ladder does not retain all field
predictions in memory.  No COMSOL response is used as an input feature.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V45_DIR = HERE.parent / "conference_continuity_v45"
for entry in (HERE, V45_DIR, ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_unknown_structures as base  # noqa: E402
from run_entry import install_isolated_factory  # noqa: E402


DATASET_ROOT = ROOT / "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45"
FORMAL_ROOT = ROOT / "runs/icp_conference_continuity_v45/final"
EXPERIMENT_ROOT = ROOT / "runs/icp_uno_structure_em_v46/final"
OUT_ROOT = ROOT / "reports/icp_conference_materials/uno_structure_em_v46"
SEED = 1237

MODEL_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "formal_dimension",
            {
                "root": FORMAL_ROOT,
                "run_variant": "conference_dimension",
                "role": "formal_reference",
                "description": "Frozen conference dimension-parameter UNO",
            },
        ),
        (
            "formal_sdf",
            {
                "root": FORMAL_ROOT,
                "run_variant": "conference_sdf",
                "role": "formal_reference",
                "description": "Frozen conference union-SDF UNO",
            },
        ),
        *[
            (
                name,
                {
                    "root": EXPERIMENT_ROOT,
                    "run_variant": name,
                    "role": "isolated_candidate",
                    "description": description,
                },
            )
            for name, description in (
                ("A_multires", "True U-shaped multiresolution UNO"),
                ("AB_separate_fusion", "Separate process/geometry/EM lifting and fusion"),
                ("ABC_adaptive_mix", "Adaptive local/global mixing"),
                ("ABD_shared_coils", "Permutation-invariant shared per-coil SDF encoder"),
                ("ABE_em_shape_amplitude", "Vector EM shape/amplitude encoding"),
                ("ABF_structure_delta", "Structural-delta plus Sobolev/multiscale supervision"),
                ("ABG_sdf_em_aux", "Geometry-latent to EM auxiliary reconstruction"),
                ("ABH_target_decoders", "Target-specific output decoders"),
                ("ALL_combined", "Combined experimental UNO"),
            )
        ],
    ]
)


def _profile_channels(run: Path) -> tuple[str, ...]:
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    benchmark = dict(cfg.get("benchmark", cfg))
    return tuple(
        benchmark.get("metadata", {}).get(
            "experimental_profile_channels",
            benchmark["train"]["u_no"]["input_features"]["features"],
        )
    )


def _predict_model(model_name: str, selected_ids: set[str], batch_size: int) -> dict[str, dict[str, Any]]:
    spec = MODEL_SPECS[model_name]
    root = Path(spec["root"])
    run_variant = str(spec["run_variant"])
    run = root / run_variant / f"seed_{SEED}"
    if spec["role"] == "isolated_candidate":
        install_isolated_factory(_profile_channels(run))
    original_root = base.RUN_ROOT
    try:
        base.RUN_ROOT = root
        return base._predict_variant(run_variant, selected_ids, batch_size=batch_size)
    finally:
        base.RUN_ROOT = original_root


def _append_model_metrics(
    *,
    model_name: str,
    predictions: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, str]],
    unknown_ids: set[str],
    case_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    representative_payload: dict[str, np.ndarray],
) -> None:
    for case_id in sorted(unknown_ids):
        payload = predictions[case_id]
        row: dict[str, Any] = {
            "model": model_name,
            "case_id": case_id,
            "layout_kind": metadata[case_id]["layout_kind"],
            "operation_id": metadata[case_id]["operation_id"],
            "design_family_id": metadata[case_id]["design_family_id"],
        }
        for target in base.TARGETS:
            row[f"{target}_rel_l2_pct"] = 100.0 * base._relative_l2(
                payload["truth"][target], payload["prediction"][target], payload["mask"]
            )
        centers, truth_profile, _, truth_uniformity = base._bohm_profile(
            payload["truth"]["ni"], payload["truth"]["Te"], payload["mask"], payload["r_coords"]
        )
        _, prediction_profile, _, prediction_uniformity = base._bohm_profile(
            payload["prediction"]["ni"],
            payload["prediction"]["Te"],
            payload["mask"],
            payload["r_coords"],
        )
        row["bohm_profile_rel_l2_pct"] = 100.0 * base._relative_l2(truth_profile, prediction_profile)
        row["bohm_uniformity_truth_pct"] = truth_uniformity
        row["bohm_uniformity_prediction_pct"] = prediction_uniformity
        row["bohm_uniformity_abs_error_pp"] = abs(prediction_uniformity - truth_uniformity)
        case_rows.append(row)

        layout_id = metadata[case_id]["layout_id"]
        coil_count = int(layout_id.split("_n", 1)[1].split("_", 1)[0])
        anchor_id = f"v43_te_n{coil_count}_anchor__{metadata[case_id]['operation_id']}"
        anchor = predictions[anchor_id]
        _, truth_anchor_profile, _, _ = base._bohm_profile(
            anchor["truth"]["ni"], anchor["truth"]["Te"], anchor["mask"], anchor["r_coords"]
        )
        _, pred_anchor_profile, _, _ = base._bohm_profile(
            anchor["prediction"]["ni"],
            anchor["prediction"]["Te"],
            anchor["mask"],
            anchor["r_coords"],
        )
        truth_delta_ni = (payload["truth"]["ni"] - anchor["truth"]["ni"])[payload["mask"]]
        pred_delta_ni = (payload["prediction"]["ni"] - anchor["prediction"]["ni"])[payload["mask"]]
        truth_delta_profile = truth_profile - truth_anchor_profile
        pred_delta_profile = prediction_profile - pred_anchor_profile
        response_rows.append(
            {
                "model": model_name,
                "case_id": case_id,
                "anchor_case_id": anchor_id,
                "layout_kind": metadata[case_id]["layout_kind"],
                "operation_id": metadata[case_id]["operation_id"],
                "design_family_id": metadata[case_id]["design_family_id"],
                "ni_truth_delta_over_anchor_pct": 100.0
                * base._gain(anchor["truth"]["ni"][payload["mask"]], truth_delta_ni),
                "ni_delta_rel_l2_pct": 100.0 * base._relative_l2(truth_delta_ni, pred_delta_ni),
                "ni_delta_cosine": base._cosine(truth_delta_ni, pred_delta_ni),
                "ni_delta_gain": base._gain(truth_delta_ni, pred_delta_ni),
                "bohm_truth_delta_over_anchor_pct": 100.0
                * base._gain(truth_anchor_profile, truth_delta_profile),
                "bohm_delta_rel_l2_pct": 100.0
                * base._relative_l2(truth_delta_profile, pred_delta_profile),
                "bohm_delta_cosine": base._cosine(truth_delta_profile, pred_delta_profile),
                "bohm_delta_gain": base._gain(truth_delta_profile, pred_delta_profile),
            }
        )
        if case_id in base.REPRESENTATIVE_CASES:
            prefix = f"{model_name}__{case_id}"
            representative_payload[f"{prefix}__ni_prediction"] = np.asarray(
                payload["prediction"]["ni"], dtype=np.float32
            )
            representative_payload[f"{prefix}__Te_prediction"] = np.asarray(
                payload["prediction"]["Te"], dtype=np.float32
            )
            representative_payload[f"{prefix}__bohm_profile_prediction"] = np.asarray(
                prediction_profile, dtype=np.float32
            )
            representative_payload[f"{prefix}__bohm_profile_truth"] = np.asarray(
                truth_profile, dtype=np.float32
            )
            representative_payload[f"{prefix}__radial_centers"] = np.asarray(centers, dtype=np.float32)
            representative_payload[f"{prefix}__ni_truth"] = np.asarray(
                payload["truth"]["ni"], dtype=np.float32
            )
            representative_payload[f"{prefix}__mask"] = np.asarray(payload["mask"], dtype=np.uint8)
            representative_payload[f"{prefix}__r_coords"] = np.asarray(
                payload["r_coords"], dtype=np.float32
            )
            representative_payload[f"{prefix}__z_coords"] = np.asarray(
                payload["z_coords"], dtype=np.float32
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--models", nargs="*", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    selected_models = list(args.models)
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = base._read_csv(DATASET_ROOT / "design_metadata_v43.csv")
    metadata = {row["case_id"]: row for row in metadata_rows}
    membership = base._read_json(DATASET_ROOT / "split_membership.json")
    test_ids = {str(value) for value in membership["test"]}
    selected_ids = {
        case_id
        for case_id, row in metadata.items()
        if case_id in test_ids
        and row["stage"] == "G4"
        and row["layout_kind"] in (*base.UNKNOWN_FAMILIES, "regular_anchor")
    }
    unknown_ids = {
        case_id for case_id in selected_ids if metadata[case_id]["layout_kind"] in base.UNKNOWN_FAMILIES
    }
    if len(unknown_ids) != 75 or len(selected_ids) != 100:
        raise ValueError(f"frozen G4 contract mismatch: unknown={len(unknown_ids)}, all={len(selected_ids)}")

    case_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    representative_payload: dict[str, np.ndarray] = {}
    for model_name in selected_models:
        predictions = _predict_model(model_name, selected_ids, max(1, int(args.batch_size)))
        _append_model_metrics(
            model_name=model_name,
            predictions=predictions,
            metadata=metadata,
            unknown_ids=unknown_ids,
            case_rows=case_rows,
            response_rows=response_rows,
            representative_payload=representative_payload,
        )
        del predictions
        gc.collect()

    original_variants = base.VARIANTS
    try:
        base.VARIANTS = tuple(selected_models)
        summaries = base._summaries(case_rows, response_rows)
    finally:
        base.VARIANTS = original_variants
    base._write_csv(out_dir / "case_metrics_unknown75.csv", case_rows)
    base._write_csv(out_dir / "structural_response_metrics_unknown75.csv", response_rows)
    base._write_csv(out_dir / "summary_metrics.csv", summaries)
    np.savez_compressed(out_dir / "representative_predictions.npz", **representative_payload)

    audit = {
        "study": "icp_uno_structure_em_v46",
        "experimental_only": True,
        "formal_models_mutated": False,
        "seed": SEED,
        "dataset": str(DATASET_ROOT.relative_to(ROOT)),
        "models": {
            name: {
                key: str(value) if isinstance(value, Path) else value
                for key, value in MODEL_SPECS[name].items()
            }
            for name in selected_models
        },
        "selected_case_count": len(selected_ids),
        "unknown_case_count": len(unknown_ids),
        "unknown_family_counts": {
            family: sum(metadata[case_id]["layout_kind"] == family for case_id in unknown_ids)
            for family in base.UNKNOWN_FAMILIES
        },
        "metrics": {
            "field_error": "plasma-masked physical relative L2",
            "bohm_profile": "10-layer wafer band, 20 equal-area radial bins",
            "response_anchor": "same coil count and operating condition regular_anchor",
            "uncertainty": "case-bootstrap 95% CI of family median, 2000 resamples",
        },
        "response_leakage": False,
        "representative_cases": list(base.REPRESENTATIVE_CASES),
    }
    (out_dir / "evaluation_contract.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
