#!/usr/bin/env python
"""Generate warm-started configs and retrain the controlled ICP v45 ladder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data/outputs_icp_stage4_plus_v43_vacuum_q3_v45"
CONFIG_ROOT = ROOT / "configs/experimental/icp_stage4/conference_continuity_v45"
RUN_ROOT = ROOT / "runs/icp_conference_continuity_v45"
FORMAL_ROOT = ROOT / "runs/icp_axisymmetric_operator_case_v1/final"
SEED = 1237

STATIC_CHANNELS = ("x", "y", "mask_plasma", "distance_signed", "distance_any")
SDF_CHANNELS = (*STATIC_CHANNELS, "part_sdf_union", "part_source_mean")
VACUUM_CHANNELS = (
    *SDF_CHANNELS,
    "vacuum_aphi_unit",
    "vacuum_br_unit",
    "vacuum_bz_unit",
    "vacuum_bmag_unit",
)
STRUCTURE_CHANNELS = (
    *VACUUM_CHANNELS,
    "part_second_proximity",
    "part_competition",
    "solid_proximity",
)
FORMAL_DIM_COND = ("llcoil", "rrc", "nncoil", "rrce", "zzc", "pp", "pp0")
EXPLICIT_DIM_COND = tuple(
    [
        f"coil_{slot:02d}_{name}"
        for slot in range(1, 7)
        for name in ("active", "r_center", "z_center", "width", "height")
    ]
    + ["pp", "pp0"]
)

VARIANTS: dict[str, dict[str, Any]] = {
    "conference_dimension": {
        "base": "dimension",
        "index": "index.csv",
        "cond": FORMAL_DIM_COND,
        "channels": STATIC_CHANNELS,
        "profile": "geom_v1_mainline",
        "provider": "fixed",
        "role": "adopted seven-scalar Dimension continuity baseline",
    },
    "explicit_dimension": {
        "base": "dimension",
        "index": "index_explicit_dimension.csv",
        "cond": EXPLICIT_DIM_COND,
        "channels": STATIC_CHANNELS,
        "profile": "geom_v1_mainline",
        "provider": "fixed",
        "role": "full-information six-coil Dimension control",
    },
    "conference_sdf": {
        "base": "union_sdf",
        "index": "index.csv",
        "cond": ("pp", "pp0"),
        "channels": SDF_CHANNELS,
        "profile": "icp_coil_sdf_source_mean_v3",
        "provider": "parametric_parts",
        "role": "adopted union-SDF continuity baseline",
    },
    "sdf_vacuum": {
        "base": "union_sdf",
        "index": "index.csv",
        "cond": ("pp", "pp0"),
        "channels": VACUUM_CHANNELS,
        "profile": "icp_coil_sdf_source_mean_vacuum_v1",
        "provider": "parametric_parts",
        "role": "SDF plus deterministic vacuum electromagnetic carrier",
    },
    "sdf_vacuum_structure": {
        "base": "union_sdf",
        "index": "index.csv",
        "cond": ("pp", "pp0"),
        "channels": STRUCTURE_CHANNELS,
        "profile": "icp_coil_sdf_source_mean_vacuum_structure_v1",
        "provider": "parametric_parts",
        "role": "SDF electromagnetic carrier plus order-invariant multi-coil summaries",
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _base_config(name: str) -> dict[str, Any]:
    path = FORMAL_ROOT / name / "u_no" / f"seed_{SEED}" / "resolved_config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or "train" not in payload:
        raise ValueError(f"invalid formal resolved config: {path}")
    return payload


def _combine_moments(
    count: int,
    mean: float,
    m2: float,
    values: np.ndarray,
) -> tuple[int, float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return count, mean, m2
    local_count = int(array.size)
    local_mean = float(np.mean(array, dtype=np.float64))
    local_m2 = float(np.sum((array - local_mean) ** 2, dtype=np.float64))
    if count == 0:
        return local_count, local_mean, local_m2
    delta = local_mean - mean
    total = count + local_count
    return (
        total,
        mean + delta * local_count / total,
        m2 + local_m2 + delta * delta * count * local_count / total,
    )


def _density_stats() -> dict[str, dict[str, float]]:
    cached = CONFIG_ROOT / "combined_train_density_stats.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    rows = _read_csv(DATASET / "index.csv")
    membership = json.loads((DATASET / "split_membership.json").read_text(encoding="utf-8"))
    train_ids = set(str(value) for value in membership["train"])
    state = {name: (0, 0.0, 0.0) for name in ("ne", "ni")}
    for row_index, row in enumerate((row for row in rows if row["case_id"] in train_ids), start=1):
        with np.load(DATASET / row["structure_npz"], allow_pickle=False) as structure:
            mask = np.asarray(structure["mask_plasma"], dtype=np.float32) > 0.5
        with np.load(DATASET / row["fields_npz"], allow_pickle=False) as fields:
            for name in state:
                state[name] = _combine_moments(*state[name], np.asarray(fields[name], dtype=np.float32)[mask])
        if row_index % 120 == 0:
            print(f"density statistics {row_index}/{len(train_ids)}", flush=True)
    result = {
        name: {
            "count": int(values[0]),
            "mean": float(values[1]),
            "std": float(np.sqrt(values[2] / max(values[0], 1))),
        }
        for name, values in state.items()
    }
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _source_checkpoint(base: str) -> Path:
    return (
        FORMAL_ROOT
        / base
        / "u_no"
        / f"seed_{SEED}"
        / "models/u_no/eval_protocol/structure_holdout/checkpoints/weights.npz"
    )


def _input_labels(base: str) -> tuple[str, ...]:
    if base == "dimension":
        return (*FORMAL_DIM_COND, *STATIC_CHANNELS)
    return ("pp", "pp0", *SDF_CHANNELS)


def _adapt_checkpoint(variant: str, spec: dict[str, Any], density_stats: dict[str, dict[str, float]]) -> Path:
    out = CONFIG_ROOT / "initial_weights" / f"{variant}_seed{SEED}.npz"
    meta_path = out.with_suffix(".json")
    source = _source_checkpoint(str(spec["base"]))
    source_labels = _input_labels(str(spec["base"]))
    target_labels = (*tuple(spec["cond"]), *tuple(spec["channels"]))
    with np.load(source, allow_pickle=False) as checkpoint:
        state = {str(name): np.asarray(checkpoint[name]).copy() for name in checkpoint.files}
    key = "torch::backbone.in_proj.weight"
    old = np.asarray(state[key], dtype=np.float32)
    if old.shape[1] != len(source_labels):
        raise ValueError(f"source input-label mismatch: {old.shape[1]} != {len(source_labels)}")
    new = np.zeros((old.shape[0], len(target_labels), *old.shape[2:]), dtype=np.float32)
    copied: list[str] = []
    for target_index, label in enumerate(target_labels):
        if label in source_labels:
            new[:, target_index] = old[:, source_labels.index(label)]
            copied.append(label)
    state[key] = new
    means = np.asarray([density_stats["ne"]["mean"], density_stats["ni"]["mean"]], dtype=np.float32)
    scales = np.asarray([density_stats["ne"]["std"], density_stats["ni"]["std"]], dtype=np.float32)
    state["torch::density_means"] = means.reshape(1, 2, 1, 1)
    state["torch::density_scales"] = scales.reshape(1, 2, 1, 1)
    state["torch::density_reference_scale"] = np.asarray(float(np.mean(scales)), dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **state)
    meta = {
        "variant": variant,
        "source_checkpoint": str(source.relative_to(ROOT)),
        "source_input_labels": list(source_labels),
        "target_input_labels": list(target_labels),
        "copied_input_labels": copied,
        "zero_initialized_input_labels": [label for label in target_labels if label not in copied],
        "all_layers_trainable": True,
        "density_stats": density_stats,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _config_for(
    variant: str,
    spec: dict[str, Any],
    *,
    epochs: int,
    run_dir: Path,
    initial_weights: Path,
) -> dict[str, Any]:
    cfg = deepcopy(_base_config(str(spec["base"])))
    cfg["output_dir"] = run_dir.relative_to(ROOT).as_posix()
    cfg["seed"] = SEED
    cfg["dataset"]["root"] = DATASET.relative_to(ROOT).as_posix()
    cfg["dataset"]["index_csv"] = str(spec["index"])
    cfg["dataset"]["cond_columns"] = list(spec["cond"])
    cfg["dataset"].setdefault("validation", {})["reject_same_input_different_output"] = False
    cfg["split"]["interp_mode"] = "marginal"
    cfg["split"]["fixed_interp"] = {
        "path": (DATASET / "split_membership.json").relative_to(ROOT).as_posix(),
        "allow_unassigned": False,
    }
    cfg["split"]["extrapolation"] = {
        "key": "pp",
        "direction": "high",
        "holdout_ratio": 0.2,
        "val_ratio": 0.2,
    }
    cfg["split"]["structure_holdout"] = {"enabled": False, "required": False}
    cfg["eval_protocol"]["primary_split"] = "interp"
    cfg["eval_protocol"]["scope"] = "u_no_isolated"
    cfg["eval_protocol"]["min_primary_test_cases"] = 100
    cfg["eval_protocol"]["min_primary_test_groups"] = 20
    cfg["preprocessing"]["scalers"]["fit_split"] = "interp"
    coord = cfg["preprocessing"]["coord_features"]
    coord["channels_from_profile"] = str(spec["profile"])
    if len(spec["channels"]) > len(STATIC_CHANNELS):
        coord["static_output"] = "features/static_spatial_feature_pack.npz"
        coord["case_structure_output"] = "features/case_structure_feature_pack.npz"
        coord["require_case_variation"] = True
    else:
        coord.pop("static_output", None)
        coord.pop("case_structure_output", None)
        coord.pop("require_case_variation", None)
    uno = cfg["train"]["u_no"]
    uno["epochs"] = int(epochs)
    uno["initial_weights_path"] = initial_weights.relative_to(ROOT).as_posix()
    uno["input_features"]["features"] = list(spec["channels"])
    uno["selection"]["warmup_epochs"] = min(5, max(0, int(epochs) // 4))
    cfg["eval"]["protocol_variant"] = f"icp_conference_continuity_v45_{variant}_seed{SEED}"
    cfg["runtime"]["structure"]["feature_profile"] = str(spec["profile"])
    cfg["runtime"]["structure"]["provider_mode"] = str(spec["provider"])
    cfg.setdefault("metadata", {}).update(
        {
            "study": "icp_conference_continuity_v45",
            "variant": variant,
            "comparison_role": str(spec["role"]),
            "formal_axis_seed": SEED,
            "current_dataset_case_count": 957,
            "frozen_test_contract": "v43_G4_plus_legacy_source_test",
            "magnetic_input_policy": "deterministic_unit_current_vacuum_only",
            "response_field_leakage": False,
            "epochs": int(epochs),
        }
    )
    return {"benchmark": cfg, "runtime": deepcopy(cfg["runtime"])}


def generate(*, epochs: int, smoke: bool) -> list[dict[str, str]]:
    if not (DATASET / "dataset_view_summary.json").is_file():
        raise FileNotFoundError(f"prepare dataset view first: {DATASET}")
    density_stats = _density_stats()
    manifest: list[dict[str, str]] = []
    # Keep the first smoke artifacts immutable: they diagnosed the unbounded
    # structural-distance failure.  The v2 smoke uses the bounded 0--1 maps.
    suffix = "smoke_bounded_v2" if smoke else "final"
    effective_epochs = 2 if smoke else int(epochs)
    for variant, spec in VARIANTS.items():
        weights = _adapt_checkpoint(variant, spec, density_stats)
        run_dir = RUN_ROOT / suffix / variant / f"seed_{SEED}"
        config_path = CONFIG_ROOT / suffix / variant / f"seed_{SEED}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _config_for(
            variant,
            spec,
            epochs=effective_epochs,
            run_dir=run_dir,
            initial_weights=weights,
        )
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        manifest.append(
            {
                "variant": variant,
                "config": str(config_path.relative_to(ROOT)),
                "run": str(run_dir.relative_to(ROOT)),
                "epochs": str(effective_epochs),
            }
        )
    manifest_path = CONFIG_ROOT / f"manifest_{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run(manifest: list[dict[str, str]], variants: set[str]) -> None:
    env = dict(os.environ, PLASMA_SURROGATE_ENABLE_TORCH="1")
    executable = ROOT / ".venv-torch/Scripts/plasma-surrogate.exe"
    for item in manifest:
        if variants and item["variant"] not in variants:
            continue
        run_dir = ROOT / item["run"]
        if (run_dir / "leaderboard.csv").is_file():
            print(f"skip completed {item['variant']}: {run_dir}", flush=True)
            continue
        command = [str(executable), "benchmark", "run", "--config", str(ROOT / item["config"])]
        print("RUN", item["variant"], flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "run", "all"), default="generate")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS), default=[])
    args = parser.parse_args()
    manifest = generate(epochs=int(args.epochs), smoke=bool(args.smoke))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.stage in {"run", "all"}:
        run(manifest, set(args.variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
