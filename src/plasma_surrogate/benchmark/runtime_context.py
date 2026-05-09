"""Benchmark runtime context builder to keep runner orchestration thin."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
from plasma_surrogate.core.dataset_io import load_dataset
from plasma_surrogate.core.feature_cache import prepare_feature_cache
from plasma_surrogate.core.input_modes import (
    build_input_mode_effective_metadata,
    normalize_input_mode_cfg,
    resolve_benchmark_runtime_controls,
    validate_input_mode_cfg,
)
from plasma_surrogate.core.run_bundle import RunBundle, RunBundleLoader, ensure_preprocess_contract
from plasma_surrogate.data.geometry_provider import GeometryProviderLike, build_geometry_provider
from plasma_surrogate.features.geometry_feature_store import GeometryFeatureStore, hash_json
from plasma_surrogate.preprocessing.runner import PreprocessRunner


def resolve_effective_benchmark_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve benchmark-local effective config with runtime precedence and validation."""

    root_cfg = copy.deepcopy(dict(cfg or {}))
    benchmark_cfg_raw = dict(root_cfg.get("benchmark", root_cfg))
    runtime_raw: dict[str, Any] | None = None
    benchmark_runtime = benchmark_cfg_raw.get("runtime")
    root_runtime = root_cfg.get("runtime")
    if isinstance(benchmark_runtime, dict):
        if isinstance(root_runtime, dict):
            norm_bench_runtime = dict(normalize_input_mode_cfg({"runtime": benchmark_runtime}).get("runtime", {}))
            norm_root_runtime = dict(normalize_input_mode_cfg({"runtime": root_runtime}).get("runtime", {}))
            if norm_bench_runtime != norm_root_runtime:
                raise ValueError(
                    "benchmark.runtime conflicts with top-level runtime; "
                    "benchmark runs require a single runtime contract"
                )
        runtime_raw = dict(benchmark_runtime)
    else:
        if isinstance(root_runtime, dict):
            runtime_raw = dict(root_runtime)
    if runtime_raw is not None:
        benchmark_cfg_raw["runtime"] = copy.deepcopy(runtime_raw)
    effective = normalize_input_mode_cfg(benchmark_cfg_raw)
    validate_input_mode_cfg(effective)
    resolve_benchmark_runtime_controls(effective)
    return effective


@dataclass
class BenchmarkDataContext:
    dataset: Any
    bundle: RunBundle
    split: dict[str, list[str]]
    cond: np.ndarray
    y: np.ndarray
    cond_scaled: np.ndarray
    y_scaled: np.ndarray
    geom_provider: GeometryProviderLike
    feature_store: GeometryFeatureStore
    feature_meta: dict[str, Any]
    profile_lock: dict[str, Any]
    split_seed: int
    split_ratios: tuple[float, float, float]
    global_seed: int
    requested_axis_mode: str
    n_cases: int
    h: int
    w: int
    cond_order: list[str]
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray
    transforms: Any
    deeponet_index: dict[str, Any]
    deeponet_index_meta: dict[str, Any]
    deeponet_poisson_index: dict[str, Any]
    deeponet_poisson_meta: dict[str, Any]
    deeponet_boundary_index: dict[str, Any]
    deeponet_boundary_meta: dict[str, Any]
    coord_feature_scaler: dict[str, Any] | None
    coord_feature_pack: dict[str, Any] | None
    coord_distance_transform_stats: dict[str, Any] | None
    lock_hash: str
    resolved: dict[str, Any]


def build_benchmark_data_context(
    cfg: dict[str, Any],
    output_root: Path,
    *,
    profile_lock: dict[str, Any],
) -> BenchmarkDataContext:
    cfg = resolve_effective_benchmark_cfg(cfg)
    input_mode_meta = build_input_mode_effective_metadata(cfg)
    requested_axis_mode = str(
        cfg.get(
            "axis_mode",
            cfg.get("preprocessing", {}).get("axis_schema", {}).get("mode", "steady"),
        )
    )
    ds_cfg = dict(cfg.get("dataset", {}))
    ds_cfg.setdefault("axis_mode", requested_axis_mode)
    dataset = load_dataset({"dataset": ds_cfg}, output_root)
    split_cfg = cfg.get("split", {})
    split_seed = int(split_cfg.get("seed", 0))
    split_ratios = tuple(split_cfg.get("ratios", [0.7, 0.15, 0.15]))
    eval_protocol_cfg = dict(cfg.get("eval_protocol", {}))
    eval_cfg = dict(cfg.get("eval", {}))
    interp_mode = str(eval_cfg.get("interp_mode", eval_protocol_cfg.get("interp_mode", "marginal"))).strip().lower()
    if interp_mode not in {"marginal", "overlap"}:
        raise ValueError("benchmark.eval.interp_mode must be one of: marginal, overlap")
    global_seed = int(cfg.get("seed", 0))

    n_cases = len(dataset.cases)
    h, w = dataset.shape
    cond_order = dataset.cond_order
    provider_mode = str(input_mode_meta.get("geometry_provider_mode_effective", "fixed"))
    geom_provider = build_geometry_provider(dataset.geometry_root, provider_mode=provider_mode)
    feature_store, feature_meta = prepare_feature_cache(
        run_root=output_root,
        geometry_provider=geom_provider,
        features_cfg=cfg.get("features", {}),
        axis_mode=requested_axis_mode,
        axis_value=0.0,
        geom_ref={"geom_id": "default"},
    )
    axis_harmonics = int(cfg.get("preprocessing", {}).get("axis_schema", {}).get("harmonics", 1))
    pre_cfg: dict[str, Any] = {
        "split": {"seed": split_seed, "ratios": list(split_ratios), "interp_mode": interp_mode},
        "cond_order": cond_order,
        "axis_schema": {
            "mode": requested_axis_mode,
            "harmonics": axis_harmonics,
        },
        "featurization_root": str(output_root / "featurization"),
    }
    pre_root_cfg = dict(cfg.get("preprocessing", {}))
    scalers_cfg = dict(cfg.get("preprocessing", {}).get("scalers", {}))
    if scalers_cfg:
        pre_cfg["scalers"] = scalers_cfg
    coord_features_cfg = dict(pre_root_cfg.get("coord_features", {}))
    if coord_features_cfg:
        pre_cfg["coord_features"] = coord_features_cfg
    if "coord_grid_source" in pre_root_cfg:
        pre_cfg["coord_grid_source"] = str(pre_root_cfg.get("coord_grid_source", "coord_grid"))
    distance_contract_cfg = dict(pre_root_cfg.get("distance_contract", {}))
    if distance_contract_cfg:
        pre_cfg["distance_contract"] = distance_contract_cfg
    coord_grid_contract_cfg = dict(pre_root_cfg.get("coord_grid_contract", {}))
    if coord_grid_contract_cfg:
        pre_cfg["coord_grid_contract"] = coord_grid_contract_cfg
    deeponet_sampling_cfg = cfg.get("preprocessing", {}).get("sampling", {}).get("deeponet", {})
    if (not deeponet_sampling_cfg) and ("deeponet_plasma" in profile_lock["models"]):
        deeponet_sampling_cfg = {
            "enabled": True,
            "n_sensors": min(32, h * w),
            "n_queries": min(64, h * w),
            "seed": global_seed,
        }
    if "deeponet_plasma" in profile_lock["models"]:
        deeponet_sampling_cfg = dict(deeponet_sampling_cfg or {})
        tasks = dict(deeponet_sampling_cfg.get("tasks", {}))
        base_seed = int(deeponet_sampling_cfg.get("seed", global_seed))
        if "poisson_head" not in tasks:
            tasks["poisson_head"] = {
                "n_sensors": min(32, h * w),
                "n_queries": int(h * w),
                "seed": base_seed,
            }
        if "boundary_operator" not in tasks:
            tasks["boundary_operator"] = {
                "n_sensors": min(64, h * w),
                "n_queries": min(64, h * w),
                "seed": base_seed + 1,
                "primary_qoi_key": str(cfg.get("boundary_operator", {}).get("primary_qoi_key", "Gamma_i")),
            }
        deeponet_sampling_cfg["tasks"] = tasks
    if deeponet_sampling_cfg:
        pre_cfg["sampling"] = {"deeponet": deeponet_sampling_cfg}

    pre = PreprocessRunner(
        pre_cfg,
        output_root / "preprocessing",
        runtime_input_mode_meta=input_mode_meta,
        runtime_cfg=dict(cfg.get("runtime", {})),
    )
    pre.run(cases=dataset.cases, geometry_root=dataset.geometry_root)
    ensure_preprocess_contract(output_root)
    bundle = RunBundleLoader.load(output_root)

    split = bundle.split_random()
    repro_hashes = ArtifactStore(output_root / "preprocessing" / "validation").load_json("repro_hashes.json")
    deeponet_task_hashes = dict(bundle.schemas.get("deeponet_task_hashes", {})) or dict(
        repro_hashes.get("deeponet_task_hashes", {})
    )
    deeponet_index = dict(bundle.schemas.get("deeponet_index", {}))
    deeponet_index_meta = dict(bundle.schemas.get("deeponet_index_meta", {}))
    deeponet_poisson_index = dict(bundle.schemas.get("deeponet_poisson_head_index", {}))
    deeponet_poisson_meta = dict(bundle.schemas.get("deeponet_poisson_head_meta", {}))
    deeponet_boundary_index = dict(bundle.schemas.get("deeponet_boundary_operator_index", {}))
    deeponet_boundary_meta = dict(bundle.schemas.get("deeponet_boundary_operator_meta", {}))
    deeponet_index_hash = str(repro_hashes.get("deeponet_index_hash", ""))
    profile_lock["sensor_query_seed"] = deeponet_index.get("seed")
    profile_lock["sensor_query_spec_hash"] = hash_json(deeponet_task_hashes) if deeponet_task_hashes else deeponet_index_hash
    profile_lock["deeponet_task_hashes"] = deeponet_task_hashes
    profile_lock["primary_qoi_key"] = str(
        deeponet_boundary_index.get(
            "primary_qoi_key",
            cfg.get("boundary_operator", {}).get("primary_qoi_key", "Gamma_i"),
        )
    )
    if ("deeponet_plasma" in profile_lock["models"] or profile_lock["phi_mode"] == "deeponet_poisson") and not deeponet_index:
        raise ValueError(
            "Benchmark profile requires DeepONet sensor/query artifact, but "
            "preprocessing/sampling/deeponet/sensor_query_index.json is missing."
        )
    if "deeponet_plasma" in profile_lock["models"] and (not deeponet_poisson_index or not deeponet_boundary_index):
        raise ValueError(
            "Benchmark deeponet_plasma requires deeponet/poisson_head and deeponet/boundary_operator artifacts"
        )
    required_tasks = set(profile_lock.get("required_deeponet_tasks", []))
    if required_tasks:
        missing_tasks = sorted([task for task in required_tasks if task not in deeponet_task_hashes])
        if missing_tasks:
            raise ValueError(f"Benchmark profile lock violation: missing deeponet task hash for {missing_tasks}")
    for lock_key in list(profile_lock.get("required_lock_keys", [])):
        value = profile_lock.get(lock_key)
        if value is None:
            raise ValueError(f"Benchmark profile lock violation: required key '{lock_key}' is missing")
        if isinstance(value, str) and (value.strip() == ""):
            raise ValueError(f"Benchmark profile lock violation: required key '{lock_key}' is empty")
        if isinstance(value, dict) and (len(value) == 0):
            raise ValueError(f"Benchmark profile lock violation: required key '{lock_key}' is empty")

    lock_hash = hash_json(
        {
            "feature_hash": feature_meta["feature_hash"],
            "split_hash": repro_hashes["split_hash"],
            "sampling_hash": repro_hashes["sampling_hash"],
            "deeponet_index_hash": deeponet_index_hash,
            "deeponet_task_hashes": deeponet_task_hashes,
        }
    )
    cond_schema = bundle.cond_schema_obj()
    axis_schema = bundle.axis_schema_obj()
    if axis_schema.mode != requested_axis_mode:
        raise ValueError(
            f"Preprocessing axis_schema.mode mismatch: expected={requested_axis_mode} actual={axis_schema.mode}"
        )
    cond = build_cond_matrix_with_axis(dataset.cases, cond_schema, axis_schema)
    y_vars = list(bundle.schemas.get("output_layout", {}).get("vars", ["ne", "ni", "Te", "phi"]))
    y = np.stack(
        [np.stack([c["y"][name] for name in y_vars], axis=0).astype(np.float32) for c in dataset.cases],
        axis=0,
    )
    case_ids = [c["case_id"] for c in dataset.cases]
    id_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    tr = np.array([id_to_idx[c] for c in split["train"]], dtype=np.int64)
    va = np.array([id_to_idx[c] for c in split["val"]], dtype=np.int64)
    te = np.array([id_to_idx[c] for c in split["test"]], dtype=np.int64)
    transforms = bundle.transform_bundle()
    cond_scaled = transforms.transform_cond(cond)
    y_scaled = transforms.transform_fields(y)

    resolved = {
        "profile": str(cfg.get("profile", "m7_global_frozen_ref")),
        "profile_lock": profile_lock,
        "seed": global_seed,
        "split": {"seed": split_seed, "ratios": list(split_ratios), "sizes": {k: len(v) for k, v in split.items()}},
        "dataset": {
            "n_cases": n_cases,
            "height": int(h),
            "width": int(w),
            "cond_order": cond_order,
            "seed": int(ds_cfg.get("seed", 0)),
        },
        "inference": {
            "axis": cfg.get("inference", {}).get(
                "axis",
                {"mode": requested_axis_mode, "value": 0.0},
            )
        },
        "artifact_hashes": {
            "feature_hash": feature_meta["feature_hash"],
            "split_hash": repro_hashes["split_hash"],
            "sampling_hash": repro_hashes["sampling_hash"],
            "deeponet_index_hash": deeponet_index_hash,
            "lock_hash": lock_hash,
        },
        "deeponet": {
            "enabled": bool(deeponet_index),
            "index_meta": deeponet_index_meta,
            "sensor_query_seed": deeponet_index.get("seed"),
            "sensor_query_spec_hash": profile_lock["sensor_query_spec_hash"],
            "task_hashes": deeponet_task_hashes,
            "poisson_head": {"index": deeponet_poisson_index, "meta": deeponet_poisson_meta},
            "boundary_operator": {"index": deeponet_boundary_index, "meta": deeponet_boundary_meta},
            "primary_qoi_key": profile_lock["primary_qoi_key"],
        },
        **input_mode_meta,
    }

    return BenchmarkDataContext(
        dataset=dataset,
        bundle=bundle,
        split=split,
        cond=cond,
        y=y,
        cond_scaled=cond_scaled,
        y_scaled=y_scaled,
        geom_provider=geom_provider,
        feature_store=feature_store,
        feature_meta=feature_meta,
        profile_lock=profile_lock,
        split_seed=split_seed,
        split_ratios=split_ratios,
        global_seed=global_seed,
        requested_axis_mode=requested_axis_mode,
        n_cases=n_cases,
        h=h,
        w=w,
        cond_order=cond_order,
        tr=tr,
        va=va,
        te=te,
        transforms=transforms,
        deeponet_index=deeponet_index,
        deeponet_index_meta=deeponet_index_meta,
        deeponet_poisson_index=deeponet_poisson_index,
        deeponet_poisson_meta=deeponet_poisson_meta,
        deeponet_boundary_index=deeponet_boundary_index,
        deeponet_boundary_meta=deeponet_boundary_meta,
        coord_feature_scaler=dict(bundle.transforms.get("coord_feature_scaler", {})),
        coord_feature_pack=bundle.schemas.get("coord_feature_pack"),
        coord_distance_transform_stats=dict(bundle.transforms.get("distance_transform_stats", {})),
        lock_hash=lock_hash,
        resolved=resolved,
    )
