"""CLI workflows for preprocess/train/infer/evaluate/viz and pipeline composition."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.model_families import (
    COND_ONLY_TORCH_MODELS,
    GLOBAL_MLP_FAMILY_MODELS,
    GRID_TORCH_MODELS,
)
from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
from plasma_surrogate.core.data_cleaning_audit import run_data_audit
from plasma_surrogate.core.input_modes import (
    DESCRIPTOR_PROFILE_KEY,
    GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY,
    INPUT_MODE_EFFECTIVE_KEY,
    STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY,
    LATENT_PROFILE_KEY,
    attach_runtime_schema_hashes,
    build_input_mode_effective_metadata,
    load_checkpoint_metadata_with_input_mode,
    merge_effective_runtime_metadata,
)
from plasma_surrogate.core.model_input_policy import resolve_effective_input_mode_metadata_for_model
from plasma_surrogate.core.physics_contract import build_physics_cfg
from plasma_surrogate.core.target_roles import resolve_physics_symbol_keys
from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.eval.core_metrics import build_region_metrics
from plasma_surrogate.eval.payloads import build_eval_metrics_payload, build_viz_tables_payload
from plasma_surrogate.eval.physics_metrics import build_single_case_physics_metrics
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.infer.engine_builder import build_inference_engine
from plasma_surrogate.infer.cases import InferenceCase, parse_batch_cases, parse_single_case
from plasma_surrogate.infer.optimize import cond_space_from_stats, validate_optimize_geom_contract
from plasma_surrogate.models.heads.plasma_head import PlasmaHead
from plasma_surrogate.models.checkpoint import save_checkpoint
from plasma_surrogate.pipeline.runtime_context import (
    build_infer_context,
    build_preprocess_context,
    build_train_context,
)
from plasma_surrogate.preprocessing.runner import PreprocessRunner
from plasma_surrogate.preprocessing.spatial_features import resolve_shared_distance_transform_cfg
from plasma_surrogate.train.model_dispatch import (
    TrainDispatchContext,
    normalize_model_name,
    run_model_train_predict,
    validate_runtime_model_policy,
)
from plasma_surrogate.train.loss_protocols import resolve_loss_protocol
from plasma_surrogate.train.model_artifacts import merge_checkpoint_dispatch_metadata
from plasma_surrogate.viz.runner import VizRunner

DEFAULT_TASK_SPEC: dict[str, Any] = {}


def _write_stage_manifest(
    *,
    run_dir: Path,
    stage: str,
    config_path: str | Path,
    produced: list[str] | None = None,
    depends_on: list[str] | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    upstream = [str(run_dir / "artifacts" / dep / "manifest.json") for dep in list(depends_on or [])]
    payload_core = {
        "stage": str(stage),
        "run_id": str(run_dir.name),
        "inputs": list(depends_on or []),
        "outputs": list(produced or []),
        "upstream_manifests": upstream,
    }
    schema_hash = hashlib.sha256(
        json.dumps(payload_core, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        **payload_core,
        "schema_hash": schema_hash,
        "config_path": str(Path(config_path)),
    }
    if extras:
        payload["extras"] = dict(extras)
    return ArtifactStore(run_dir / "artifacts" / str(stage)).save_json("manifest.json", payload)


def _build_inference_engine_from_bundle(
    *,
    bundle: Any,
    dataset_root: Path,
    feature_store: Any,
    output_dir: Path,
    input_mode_meta: dict[str, Any] | None = None,
    checkpoint_input_mode_meta: dict[str, Any] | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> InferenceEngine:
    cfg = dict(bundle.cfg or {})
    effective_input_mode_meta = attach_runtime_schema_hashes(
        dict(input_mode_meta or build_input_mode_effective_metadata(cfg)),
        schemas=dict(bundle.schemas or {}),
        schema_hashes=dict(bundle.schemas.get("runtime_schema_hashes", {}) or {}),
    )
    effective_checkpoint_input_mode_meta = dict(checkpoint_input_mode_meta or {})
    model_cfg = cfg.get("model", {})
    inf_cfg = cfg.get("inference", {})
    ood_cfg = dict(inf_cfg.get("ood", {"poisson_residual_limit": 1e2}) or {})
    for key in ("qoi", "postprocess", "diagnostics", "derived_fields", "derived_fields_strict"):
        if key in inf_cfg:
            value = inf_cfg.get(key)
            ood_cfg[key] = dict(value or {}) if isinstance(value, dict) else value
    resolved_model_name = str(model_name or model_cfg.get("name", "unet"))
    provider_mode = str(effective_input_mode_meta.get(GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY, "fixed")).strip().lower() or "fixed"
    geometry_provider = build_geometry_provider(dataset_root, provider_mode=provider_mode)
    spatial_transform_artifacts = bundle.spatial_transform_artifacts_for_checkpoint(checkpoint_meta)
    return build_inference_engine(
        model=bundle.model,
        cond_schema=bundle.cond_schema_obj(),
        axis_schema=bundle.axis_schema_obj(),
        geometry_provider=geometry_provider,
        output_dir=output_dir,
        transform_bundle=bundle.transform_bundle_for_checkpoint(checkpoint_meta),
        cond_stats=bundle.schemas.get("cond_stats", {}),
        phi_mode=str(model_cfg.get("phi_mode", "direct")),
        phi_hybrid_steps=int(model_cfg.get("phi_hybrid_steps", 1)),
        poisson_refine_iters=int(inf_cfg.get("poisson_refine", {}).get("iters", 0)),
        ood_cfg=ood_cfg,
        feature_store=feature_store,
        coord_scaler=bundle.transforms.get("coord_scaler", {}),
        coord_feature_scaler=spatial_transform_artifacts["coord_feature_scaler"],
        coord_feature_pack=bundle.schemas.get("coord_feature_pack"),
        coord_distance_transform_stats=spatial_transform_artifacts["distance_transform_stats"],
        input_mode_meta=effective_input_mode_meta,
        checkpoint_meta=dict(checkpoint_meta or {}),
        checkpoint_input_mode_meta=effective_checkpoint_input_mode_meta,
        target_role_schema=bundle.schemas.get("target_role_schema", {}),
        structure_descriptor_pack=bundle.schemas.get("structure_descriptor_pack"),
        latent_feature_pack=bundle.schemas.get("latent_feature_pack"),
        deeponet_head=(
            getattr(bundle.model, "poisson_head", None)
            if str(model_cfg.get("phi_mode", "direct")) == "deeponet_poisson"
            else None
        )
        or (bundle.model if str(model_cfg.get("phi_mode", "direct")) == "deeponet_poisson" else None),
        grid_input_features_cfg=dict(cfg.get("train", {}).get(resolved_model_name, {}).get("input_features", {})),
    )


def _resolve_y_vars(cases: list[dict[str, Any]]) -> list[str]:
    keys = list(cases[0]["y"].keys())
    if len(keys) == 0:
        raise ValueError("Dataset fields must include at least one target variable")
    return [str(k) for k in keys]


def _write_run_metadata(run_dir: Path, cfg: dict[str, Any], shape: tuple[int, int], y_vars: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(cfg)
    resolved.setdefault("task", {})
    resolved["task"]["spec_path"] = "task_spec.yaml"
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=True)

    task_spec = dict(DEFAULT_TASK_SPEC)
    task_spec["outputs"] = list(cfg.get("task_spec", {}).get("outputs", [{"name": name} for name in y_vars]))
    task_spec["grid_spec"] = {
        "axes_order": ["y", "x"],
        "coord_components": ["x", "y"],
        "shape": [int(shape[0]), int(shape[1])],
        "coord_system": "cartesian",
    }
    if cfg.get("task_spec"):
        task_spec.update(cfg["task_spec"])
    outputs = list(task_spec.get("outputs", []))
    transforms = dict(task_spec.get("transforms", {}))
    units = dict(task_spec.get("units", {}))
    for output in outputs:
        name = str(dict(output).get("name", "")).strip()
        if not name:
            continue
        transforms.setdefault(name, str(dict(output).get("transform", "zscore")))
        units.setdefault(name, str(dict(output).get("units", "")))
    task_spec["transforms"] = transforms
    task_spec["units"] = units

    with (run_dir / "task_spec.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(task_spec, f, sort_keys=True)


def _build_fields(cases: list[dict[str, Any]], y_vars: list[str]) -> np.ndarray:
    return np.stack(
        [np.stack([np.asarray(case["y"][var], dtype=np.float32) for var in y_vars], axis=0) for case in cases],
        axis=0,
    ).astype(np.float32)


def _validate_output_layout(layout: dict[str, Any], y_shape: tuple[int, int, int], y_vars: list[str]) -> dict[str, Any]:
    expected = {"order": "C", "shape": [int(y_shape[0]), int(y_shape[1]), int(y_shape[2])], "vars": y_vars}
    if layout != expected:
        raise ValueError(f"output_layout mismatch: expected={expected}, actual={layout}")
    return layout


def _resolve_physics_cfg(cfg: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    train_cfg = cfg.get("train", {})
    phys = train_cfg.get("physics", {})
    input_mode_meta = build_input_mode_effective_metadata(cfg)
    provider_mode = str(input_mode_meta.get(GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY, "fixed")).strip().lower() or "fixed"
    geom_ctx = build_geometry_provider(dataset_root, provider_mode=provider_mode).get()
    return build_physics_cfg(
        raw_cfg=phys,
        geom_ctx=geom_ctx,
        default_enabled=False,
        default_primary_qoi_key="Gamma_i",
    )


def _resolve_optimize_backend(opt_cfg: dict[str, Any]) -> str:
    if "sampler" in opt_cfg:
        raise ValueError("inference.optimize.sampler is removed; use inference.optimize.backend")
    backend = opt_cfg.get("backend")
    if backend is not None:
        return str(backend)
    return "random"


def _load_target_role_schema(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "preprocessing" / "schema" / "target_role_schema.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return dict(json.load(f) or {})


def _combined_inference_symbols(cfg: dict[str, Any]) -> dict[str, Any]:
    inf_cfg = dict(cfg.get("inference", {}) or {})
    ood_cfg = dict(inf_cfg.get("ood", {}) or {})
    symbols: dict[str, Any] = {}
    for section_name in ("physics", "boundary_operator"):
        section = dict(ood_cfg.get(section_name, {}) or {})
        raw = section.get("symbols")
        if isinstance(raw, dict):
            symbols.update({str(k): v for k, v in raw.items()})
    return symbols


def _resolve_viz_density_key(
    *,
    field_keys: list[str],
    target_role_schema: dict[str, Any],
    cfg: dict[str, Any],
) -> str | None:
    symbols = _combined_inference_symbols(cfg)
    try:
        resolved = resolve_physics_symbol_keys(
            field_keys,
            symbols=symbols,
            target_role_schema=target_role_schema,
            required=("density",),
            context="viz region metrics",
        )
    except ValueError:
        if symbols.get("density") is not None:
            raise
        return None
    return resolved["density"]


def _resolve_train_workflow_potential_key(
    *,
    field_keys: list[str],
    physics_cfg: dict[str, Any],
    context: str,
) -> str | None:
    symbols = dict(dict(physics_cfg or {}).get("symbols", {}) or {})
    try:
        resolved = resolve_physics_symbol_keys(
            [str(key) for key in field_keys],
            symbols=symbols,
            target_role_schema=dict(dict(physics_cfg or {}).get("target_role_schema", {}) or {}),
            required=("potential",),
            context=context,
        )
    except ValueError:
        if symbols.get("potential") is not None:
            raise
        return None
    return resolved["potential"]


def _parse_box_space(raw_space: dict[str, Any], *, cfg_key: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key, bounds in dict(raw_space or {}).items():
        name = str(key).strip()
        if not name:
            raise ValueError(f"{cfg_key} keys must be non-empty strings")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"{cfg_key}.{name} must be [lo, hi]")
        lo = float(bounds[0])
        hi = float(bounds[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"{cfg_key}.{name} bounds must be finite; got={bounds!r}")
        if lo > hi:
            raise ValueError(f"{cfg_key}.{name} requires lo <= hi; got={bounds!r}")
        out[name] = (lo, hi)
    return {k: out[k] for k in sorted(out.keys())}


def _inference_case_summary_row(*, role: str, case: InferenceCase, result: Any) -> dict[str, Any]:
    qoi = dict(getattr(result, "qoi", {}) or {})
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    return {
        "role": str(role),
        "case_id": str(case.case_id),
        "case_key": str(getattr(result, "case_key", "") or ""),
        "geom_id": str(case.geom.get("geom_id", "default")),
        "axis_mode": str(case.axis.get("mode", "")),
        "axis_value": float(case.axis.get("value", 0.0)),
        "uniformity": _summary_float(qoi.get("uniformity")),
        "boundary_gamma_uniformity": _summary_float(qoi.get("boundary_gamma_uniformity")),
        "poisson_residual_norm": _summary_float(diagnostics.get("poisson_residual_norm")),
        "bc_potential_mae": _summary_float(diagnostics.get("bc_potential_mae")),
        "boundary_operator_proxy_loss": _summary_float(diagnostics.get("boundary_operator_proxy_loss")),
        "physics_diagnostics_available": bool(diagnostics.get("physics_diagnostics_available", False)),
        "cond": dict(case.cond),
        "geom": dict(case.geom),
        "axis": dict(case.axis),
        "qoi": qoi,
        "diagnostics": diagnostics,
    }


def _summary_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _write_inference_case_summaries(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path] | None:
    if not rows:
        return None
    store = ArtifactStore(run_dir / "inference")
    csv_header = [
        "role",
        "case_id",
        "case_key",
        "geom_id",
        "axis_mode",
        "axis_value",
        "uniformity",
        "boundary_gamma_uniformity",
        "poisson_residual_norm",
        "bc_potential_mae",
        "boundary_operator_proxy_loss",
        "physics_diagnostics_available",
    ]
    csv_rows = [["" if row.get(key) is None else row.get(key, "") for key in csv_header] for row in rows]
    csv_path = store.save_csv("cases_summary.csv", csv_header, csv_rows)
    json_path = store.save_json("cases_summary.json", {"cases": rows})
    return csv_path, json_path


def _write_batch_summary_from_case_rows(engine: InferenceEngine, rows: list[dict[str, Any]]) -> None:
    batch_rows: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row.get("cond", {}) or {})
        out.update(dict(row.get("qoi", {}) or {}))
        batch_rows.append(out)
    if not batch_rows:
        return
    header = sorted({key for row in batch_rows for key in row.keys()})
    engine.store.save_csv("batch/summary.csv", header, [[row.get(key, "") for key in header] for row in batch_rows])


def run_preprocess(config_path: str | Path) -> dict[str, Any]:
    ctx = build_preprocess_context(config_path)
    cfg = ctx.cfg
    run_dir = ctx.run_dir
    dataset = ctx.dataset
    axis_mode = ctx.axis_mode
    feat_meta = ctx.feature_meta
    y_vars = _resolve_y_vars(dataset.cases)

    _write_run_metadata(run_dir, cfg, dataset.shape, y_vars)
    pre_cfg = cfg.get("preprocessing", {})
    pre_cfg.setdefault("cond_order", dataset.cond_order)
    pre_cfg.setdefault("axis_schema", {"mode": "steady", "harmonics": 1})
    pre_cfg.setdefault("featurization_root", str(run_dir / "featurization"))
    input_mode_meta = build_input_mode_effective_metadata(cfg)
    preprocess_model_name = normalize_model_name(dict(cfg.get("model", {})).get("name", "global_mlp"))
    coord_distance_transform_cfg = resolve_shared_distance_transform_cfg(
        train_cfg=dict(cfg.get("train", {})),
        model_names=[preprocess_model_name],
        scaling_enabled=bool(
            dict(dict(pre_cfg.get("coord_features", {})).get("scaling", {})).get("enabled", False)
        ),
    )

    pre = PreprocessRunner(
        pre_cfg,
        run_dir / "preprocessing",
        runtime_input_mode_meta=input_mode_meta,
        runtime_cfg=dict(cfg.get("runtime", {})),
        coord_distance_transform_cfg=coord_distance_transform_cfg,
    )
    output = pre.run(
        cases=dataset.cases,
        geometry_root=dataset.geometry_root,
        target_metadata=getattr(dataset, "target_metadata", []),
    )
    audit = run_data_audit(
        cases=dataset.cases,
        cond_order=dataset.cond_order,
        axis_mode=axis_mode,
        required_outputs=y_vars,
    )
    audit_path = ArtifactStore(run_dir / "data_cleaning").save_json("report.json", audit)
    _write_stage_manifest(
        run_dir=run_dir,
        stage="preprocess",
        config_path=config_path,
        depends_on=["cleanse", "feature"],
        produced=[
            str(run_dir / "preprocessing"),
            str(audit_path),
        ],
        extras={"split_sizes": {k: len(v) for k, v in output.split.items()}},
    )

    return {
        "run_dir": str(run_dir),
        "n_cases": len(dataset.cases),
        "split_sizes": {k: len(v) for k, v in output.split.items()},
        "feature_hash": feat_meta["feature_hash"],
    }


def run_cleanse(config_path: str | Path) -> dict[str, Any]:
    ctx = build_preprocess_context(config_path)
    cfg = ctx.cfg
    run_dir = ctx.run_dir
    dataset = ctx.dataset
    y_vars = _resolve_y_vars(dataset.cases)
    _write_run_metadata(run_dir, cfg, dataset.shape, y_vars)
    audit = run_data_audit(
        cases=dataset.cases,
        cond_order=dataset.cond_order,
        axis_mode=ctx.axis_mode,
        required_outputs=y_vars,
    )
    report_path = ArtifactStore(run_dir / "data_cleaning").save_json("report.json", audit)
    _write_stage_manifest(
        run_dir=run_dir,
        stage="cleanse",
        config_path=config_path,
        produced=[str(report_path)],
        extras={"n_cases": int(len(dataset.cases)), "required_outputs": list(y_vars)},
    )
    return {
        "run_dir": str(run_dir),
        "n_cases": int(len(dataset.cases)),
        "report": str(report_path),
    }


def run_feature(config_path: str | Path) -> dict[str, Any]:
    ctx = build_preprocess_context(config_path)
    run_dir = ctx.run_dir
    cache_index = run_dir / "featurization" / "geometry_cache_index.json"
    payload = {
        "feature_hash": str(ctx.feature_meta.get("feature_hash", "")),
        "cache_index": str(cache_index) if cache_index.exists() else "",
    }
    out_path = ArtifactStore(run_dir / "features").save_json("feature_meta.json", payload)
    _write_stage_manifest(
        run_dir=run_dir,
        stage="feature",
        config_path=config_path,
        produced=[str(out_path)],
        extras={"feature_hash": payload["feature_hash"]},
    )
    return {
        "run_dir": str(run_dir),
        "feature_hash": payload["feature_hash"],
        "feature_meta": str(out_path),
    }


def run_train(config_path: str | Path) -> dict[str, Any]:
    ctx = build_train_context(
        config_path,
        on_missing_preprocess=lambda: run_preprocess(config_path),
    )
    cfg = ctx.cfg
    run_dir = ctx.run_dir
    dataset = ctx.dataset
    axis_mode = ctx.axis_mode
    feat_store = ctx.feature_store
    bundle = ctx.bundle
    y_vars = _resolve_y_vars(dataset.cases)
    if bundle is None:
        raise RuntimeError("RuntimeContext missing RunBundle for train")
    input_mode_meta = attach_runtime_schema_hashes(
        build_input_mode_effective_metadata(cfg),
        schemas=dict(bundle.schemas or {}),
        schema_hashes=dict(bundle.schemas.get("runtime_schema_hashes", {}) or {}),
    )

    _write_run_metadata(run_dir, cfg, dataset.shape, y_vars)
    split = bundle.split_random()
    cond_schema = bundle.cond_schema_obj()
    axis_schema = bundle.axis_schema_obj()
    transforms = bundle.transform_bundle()

    y_phys = _build_fields(dataset.cases, y_vars)
    _validate_output_layout(bundle.schemas.get("output_layout", {}), y_phys.shape[1:], y_vars)
    cond = transforms.transform_cond(build_cond_matrix_with_axis(dataset.cases, cond_schema, axis_schema))
    y = transforms.transform_fields(y_phys)
    case_to_idx = {c["case_id"]: i for i, c in enumerate(dataset.cases)}
    tr = np.array([case_to_idx[cid] for cid in split["train"]], dtype=np.int64)
    va = np.array([case_to_idx[cid] for cid in split["val"]], dtype=np.int64)
    te = np.array([case_to_idx[cid] for cid in split["test"]], dtype=np.int64)

    model_cfg = cfg.get("model", {})
    model_name = normalize_model_name(model_cfg.get("name", "global_mlp"))
    input_mode_meta_effective = resolve_effective_input_mode_metadata_for_model(
        model_name=model_name,
        input_mode_meta=input_mode_meta,
    )
    provider_mode_effective = str(
        input_mode_meta_effective.get(GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY, "fixed")
    ).strip().lower() or "fixed"
    geometry_provider = build_geometry_provider(dataset.geometry_root, provider_mode=provider_mode_effective)
    input_mode_effective = str(input_mode_meta_effective.get(INPUT_MODE_EFFECTIVE_KEY, "")).strip().lower()
    phi_mode = str(model_cfg.get("phi_mode", "direct"))
    phi_hybrid_steps = int(model_cfg.get("phi_hybrid_steps", 1))
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 1e-2))
    loss_cfg = resolve_loss_protocol(
        dict(train_cfg.get("loss", {})),
        target_role_schema=bundle.schemas.get("target_role_schema", {}),
    )
    curriculum_cfg = dict(train_cfg.get("curriculum", {}))
    physics_cfg = _resolve_physics_cfg(cfg, dataset_root=dataset.geometry_root)
    physics_cfg["target_role_schema"] = bundle.schemas.get("target_role_schema", {})
    geom_ctx = None
    use_plasma_mask = str(loss_cfg.get("supervised", {}).get("mask", "none")).strip().lower() == "plasma_only"
    if phi_mode != "direct" or physics_cfg.get("enabled") or use_plasma_mask:
        geom_ctx = feat_store.get_context(
            geom_ref={"geom_id": "default"},
            axis_value=0.0,
            axis_mode=axis_mode,
            geometry_provider=geometry_provider,
        )
    if model_name == "deeponet_plasma" and geom_ctx is None:
        geom_ctx = feat_store.get_context(
            geom_ref={"geom_id": "default"},
            axis_value=0.0,
            axis_mode=axis_mode,
            geometry_provider=geometry_provider,
        )
    supervised_mask = np.asarray(geom_ctx.mask_plasma, dtype=np.float32) if (geom_ctx is not None and use_plasma_mask) else None
    supervised_distance = np.asarray(geom_ctx.distance_any, dtype=np.float32) if (geom_ctx is not None and use_plasma_mask) else None
    plasma_head = PlasmaHead(mode=phi_mode, jacobi_iters=phi_hybrid_steps)

    train_dispatch_cfg: dict[str, Any] = {"train": {}}
    if model_name in GLOBAL_MLP_FAMILY_MODELS:
        per_model_cfg = dict(train_cfg.get(model_name, {}))
        merged_model_cfg = dict(per_model_cfg.get("model_cfg", {}))
        if model_name == "global_mlp":
            # Preserve the legacy root-model configuration contract.
            merged_model_cfg.update(dict(model_cfg))
        else:
            routing_keys = {"name", "phi_mode", "phi_hybrid_steps", "seed"}
            misplaced_keys = sorted(set(model_cfg) - routing_keys)
            if misplaced_keys:
                raise ValueError(
                    f"model options for {model_name} must be placed under "
                    f"train.{model_name}.model_cfg; misplaced keys={misplaced_keys}"
                )
        train_dispatch_cfg["train"][model_name] = {
            **per_model_cfg,
            "epochs": int(per_model_cfg.get("epochs", epochs)),
            "lr": float(per_model_cfg.get("lr", lr)),
            "model_cfg": merged_model_cfg,
        }
    elif model_name in GRID_TORCH_MODELS or model_name in COND_ONLY_TORCH_MODELS:
        per_model_cfg = dict(train_cfg.get(model_name, {}))
        merged_model_cfg = dict(per_model_cfg.get("model_cfg", {}))
        merged_model_cfg.update(dict(model_cfg))
        train_dispatch_cfg["train"][model_name] = {
            **per_model_cfg,
            "epochs": epochs,
            "lr": lr,
            "model_cfg": merged_model_cfg,
        }
    elif model_name == "deeponet_plasma":
        train_dispatch_cfg["train"]["deeponet_plasma"] = {
            **dict(train_cfg.get("deeponet", {})),
            "epochs": epochs,
            "lr": lr,
            "model_cfg": dict(model_cfg),
        }
    else:
        raise ValueError(f"Unsupported model.name for mainline train workflow: {model_name}")

    dispatch = run_model_train_predict(
        TrainDispatchContext(
            run_cfg=train_dispatch_cfg,
            profile_lock={
                "phi_mode": phi_mode,
                "primary_qoi_key": str(physics_cfg.get("boundary_operator", {}).get("primary_qoi_key", "Gamma_i")),
            },
            model_idx=0,
            model_name=model_name,
            model_dir=run_dir,
            global_seed=int(model_cfg.get("seed", 0)),
            n_cases=len(dataset.cases),
            h=int(dataset.shape[0]),
            w=int(dataset.shape[1]),
            y_vars=y_vars,
            cond_scaled=cond,
            y=y_phys,
            y_scaled=y,
            tr=tr,
            va=va,
            te=te,
            transforms=transforms,
            physics_cfg=physics_cfg,
            geom_ctx=geom_ctx,
            deeponet_index=dict(bundle.schemas.get("deeponet_index", {})),
            deeponet_index_meta=dict(bundle.schemas.get("deeponet_index_meta", {})),
            deeponet_poisson_index=dict(bundle.schemas.get("deeponet_poisson_head_index", {})),
            deeponet_poisson_meta=dict(bundle.schemas.get("deeponet_poisson_head_meta", {})),
            deeponet_boundary_index=dict(bundle.schemas.get("deeponet_boundary_operator_index", {})),
            deeponet_boundary_meta=dict(bundle.schemas.get("deeponet_boundary_operator_meta", {})),
            config_base_dir=Path(config_path).resolve().parent,
            loss_cfg=loss_cfg,
            curriculum_cfg=curriculum_cfg,
            supervised_mask=supervised_mask,
            supervised_distance=supervised_distance,
            coord_scaler=dict(bundle.transforms.get("coord_scaler", {})),
            coord_feature_scaler=dict(bundle.transforms.get("coord_feature_scaler", {})),
            coord_feature_pack=bundle.schemas.get("coord_feature_pack"),
            static_spatial_feature_pack=bundle.schemas.get("static_spatial_feature_pack"),
            case_structure_feature_pack=bundle.schemas.get("case_structure_feature_pack"),
            coord_distance_transform_stats=dict(bundle.transforms.get("distance_transform_stats", {})),
            structure_descriptor_pack=bundle.schemas.get("structure_descriptor_pack"),
            latent_feature_pack=bundle.schemas.get("latent_feature_pack"),
            case_ids=[str(case["case_id"]) for case in dataset.cases],
            input_mode_effective=input_mode_effective,
            structure_adapter_mode_effective=str(
                input_mode_meta_effective.get(STRUCTURE_ADAPTER_MODE_EFFECTIVE_KEY, "")
            ),
            structure_descriptor_profile_effective=str(
                input_mode_meta_effective.get(DESCRIPTOR_PROFILE_KEY, "none")
            ),
            structure_latent_profile_effective=str(
                input_mode_meta_effective.get(LATENT_PROFILE_KEY, "none")
            ),
        )
    )
    model = dispatch.model
    history_len = len(dispatch.history)
    pred_eval = dict(dispatch.pred_eval)

    potential_key = _resolve_train_workflow_potential_key(
        field_keys=[str(key) for key in pred_eval.keys()],
        physics_cfg=physics_cfg,
        context="train workflow potential postprocess",
    )
    if geom_ctx is not None and plasma_head is not None and potential_key is not None:
        deeponet_head = None
        if phi_mode == "deeponet_poisson":
            deeponet_head = getattr(model, "poisson_head", None)
            if deeponet_head is None and hasattr(model, "predict_phi"):
                deeponet_head = model
            elif deeponet_head is None and hasattr(model, "predict_fields"):
                deeponet_head = model
        head_out, head_aux = plasma_head.apply(
            dict(pred_eval),
            geom_ctx=geom_ctx,
            refine_iters=0,
            deeponet_head=deeponet_head,
            cond_vec=cond[te],
            potential_key=potential_key,
        )
        pred_eval.update(head_out)
        pred_eval.update({k: v for k, v in head_aux.items() if k not in pred_eval})
    elif geom_ctx is not None and plasma_head is not None and phi_mode != "direct":
        raise ValueError(
            "train workflow potential postprocess requires a unique potential target. "
            "Set physics.symbols.potential or dataset.targets[].role."
        )

    true_eval = dispatch.true_eval
    eval_payload = build_eval_metrics_payload(
        true_eval=true_eval,
        pred_eval={k: pred_eval[k] for k in y_vars if k in pred_eval},
        eps=geom_ctx.eps if geom_ctx is not None else None,
        mask_plasma=np.asarray(geom_ctx.mask_plasma, dtype=np.float32) if geom_ctx is not None else None,
        potential_key=potential_key,
    )
    metrics = dict(eval_payload["rmse"])
    r2 = dict(eval_payload.get("r2", {}))
    metrics_plasma = dict(eval_payload.get("rmse_plasma", {}))
    r2_plasma = dict(eval_payload.get("r2_plasma", {}))

    ArtifactStore(run_dir / "eval").save_json("test_metrics.json", eval_payload)
    checkpoint_input_mode_meta = merge_checkpoint_dispatch_metadata(
        runtime_meta=input_mode_meta_effective,
        dispatch_meta=dict(dispatch.extra_artifacts or {}),
    )
    scaler_fit_split = str(
        dict(bundle.schemas.get("preprocess_report", {}) or {}).get("scaler_fit_split", "")
    ).strip().lower()
    protocol_scaler = dict(
        dict(bundle.transforms.get("protocol_transforms", {}) or {}).get(scaler_fit_split, {}) or {}
    )
    if scaler_fit_split and protocol_scaler.get("cond_scaler") and protocol_scaler.get("y_scalers"):
        checkpoint_input_mode_meta["scaler_fit_split"] = scaler_fit_split
        checkpoint_input_mode_meta["scaler_train_only"] = True
    save_checkpoint(model, run_dir / "checkpoints", extra_meta=checkpoint_input_mode_meta)
    _write_stage_manifest(
        run_dir=run_dir,
        stage="train",
        config_path=config_path,
        depends_on=["preprocess"],
        produced=[
            str(run_dir / "checkpoints"),
            str(run_dir / "eval" / "test_metrics.json"),
        ],
        extras={"model": model_name, "epochs": int(history_len)},
    )

    return {
        "run_dir": str(run_dir),
        "model": model_name,
        "epochs": history_len,
        "phi_mode": phi_mode,
        "test_rmse": metrics,
        "test_r2": r2,
        "test_rmse_plasma": metrics_plasma,
        "test_r2_plasma": r2_plasma,
    }


def run_infer(config_path: str | Path) -> dict[str, Any]:
    ctx = build_infer_context(
        config_path,
        on_missing_preprocess=lambda: run_preprocess(config_path),
        on_missing_checkpoint=lambda: run_train(config_path),
    )
    cfg = ctx.cfg
    run_dir = ctx.run_dir
    bundle = ctx.bundle
    dataset = ctx.dataset
    feat_store = ctx.feature_store
    if bundle is None:
        raise RuntimeError("RuntimeContext missing RunBundle for infer")
    input_mode_meta = attach_runtime_schema_hashes(
        build_input_mode_effective_metadata(cfg),
        schemas=dict(bundle.schemas or {}),
        schema_hashes=dict(bundle.schemas.get("runtime_schema_hashes", {}) or {}),
    )
    checkpoint_meta_path = run_dir / "checkpoints" / "meta.json"
    checkpoint_meta, checkpoint_input_mode_meta = load_checkpoint_metadata_with_input_mode(checkpoint_meta_path)

    model = bundle.model
    if model is None:
        raise FileNotFoundError(f"Missing model checkpoint metadata under {run_dir / 'checkpoints'}")
    model_cfg = cfg.get("model", {})
    model_name = validate_runtime_model_policy(
        input_mode_effective=str(input_mode_meta.get("input_mode_effective", "")),
        model_name=model_cfg.get("name", "global_mlp"),
    )
    input_mode_meta = resolve_effective_input_mode_metadata_for_model(
        model_name=model_name,
        input_mode_meta=input_mode_meta,
    )
    input_mode_effective = str(input_mode_meta.get(INPUT_MODE_EFFECTIVE_KEY, "")).strip().lower()
    provider_mode_effective = str(
        input_mode_meta.get(GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY, "fixed")
    ).strip().lower() or "fixed"
    inf_cfg = cfg.get("inference", {})
    cond_schema = bundle.cond_schema_obj()
    axis_schema = bundle.axis_schema_obj()
    engine = _build_inference_engine_from_bundle(
        bundle=bundle,
        dataset_root=dataset.geometry_root,
        feature_store=feat_store,
        output_dir=run_dir / "inference",
        input_mode_meta=input_mode_meta,
        checkpoint_input_mode_meta=checkpoint_input_mode_meta,
        checkpoint_meta=checkpoint_meta,
        model_name=model_name,
    )

    results: dict[str, Any] = {}
    case_summary_rows: list[dict[str, Any]] = []
    single_cfg = inf_cfg.get("single", {})
    if single_cfg.get("enabled", True):
        single_case = parse_single_case(single_cfg, cond_schema=cond_schema, axis_schema=axis_schema)
        single = engine.single_run_aggregated(cond=single_case.cond, geom=single_case.geom, axis=single_case.axis)
        results["single_qoi"] = single.qoi
        case_summary_rows.append(_inference_case_summary_row(role="single", case=single_case, result=single))

    batch_cfg = inf_cfg.get("batch", {})
    if batch_cfg.get("enabled", False):
        batch_cases = parse_batch_cases(
            batch_cfg,
            cond_schema=cond_schema,
            axis_schema=axis_schema,
            config_dir=Path(config_path).resolve().parent,
        )
        batch_summary_rows = []
        for case in batch_cases:
            result = engine.single_run_aggregated(cond=case.cond, geom=case.geom, axis=case.axis)
            row = _inference_case_summary_row(role="batch", case=case, result=result)
            batch_summary_rows.append(row)
            case_summary_rows.append(row)
        _write_batch_summary_from_case_rows(engine, batch_summary_rows)
        results["batch_count"] = len(batch_cases)

    opt_cfg = dict(inf_cfg.get("optimize", {}) or {})
    if opt_cfg.get("enabled", False):
        axis = opt_cfg.get("axis", {"mode": axis_schema.mode, "value": 0.0})
        space_cfg = opt_cfg.get("space")
        if space_cfg is None:
            space = cond_space_from_stats(
                list(cond_schema.order),
                bundle.schemas.get("cond_stats", {}),
                cfg_key="inference.optimize.space",
            )
        else:
            space = _parse_box_space(space_cfg, cfg_key="inference.optimize.space")
        geom_space_raw = _parse_box_space(
            dict(opt_cfg.get("geom_space", {}) or {}),
            cfg_key="inference.optimize.geom_space",
        )
        geom_cfg = dict(opt_cfg.get("geom", {"geom_id": "default"}))
        geom_space = validate_optimize_geom_contract(
            input_mode=input_mode_effective,
            provider_mode=provider_mode_effective,
            geom_space=geom_space_raw,
            geom_ref=geom_cfg,
            label_prefix="inference.optimize",
        )
        backend = _resolve_optimize_backend(opt_cfg)
        backend_cfg = dict(opt_cfg.get("backend_cfg", {}) or {})
        if backend == "csv" and "csv_path" in backend_cfg:
            csv_path = Path(str(backend_cfg["csv_path"]))
            if not csv_path.is_absolute():
                csv_path = Path(config_path).resolve().parent / csv_path
            backend_cfg["csv_path"] = str(csv_path)
        best = engine.optimize_run(
            space=space,
            geom_space=geom_space if geom_space else None,
            n_trials=int(opt_cfg.get("n_trials", 10)),
            geom=geom_cfg,
            axis=axis,
            seed=int(opt_cfg.get("seed", 0)),
            backend=backend,
            backend_cfg=backend_cfg,
            objective_cfg=dict(opt_cfg.get("objective", {}) or {}),
            constraints_cfg=opt_cfg.get("constraints", []) or [],
            output_cfg=dict(opt_cfg.get("output", {}) or {}),
        )
        results["optimize"] = best
    case_summary_paths = _write_inference_case_summaries(run_dir, case_summary_rows)
    if case_summary_paths is not None:
        results["cases_summary"] = {
            "csv": str(case_summary_paths[0]),
            "json": str(case_summary_paths[1]),
            "count": len(case_summary_rows),
        }
    inference_effective_meta = merge_effective_runtime_metadata(
        runtime_meta=input_mode_meta,
        dispatch_meta=checkpoint_meta,
    )
    infer_summary = {
        **inference_effective_meta,
        "result_keys": sorted(list(results.keys())),
        "inference_case_count": len(case_summary_rows),
    }
    infer_summary_path = ArtifactStore(run_dir / "inference").save_json("summary.json", infer_summary)

    _write_stage_manifest(
        run_dir=run_dir,
        stage="infer",
        config_path=config_path,
        depends_on=["train"],
        produced=[str(run_dir / "inference"), str(infer_summary_path)],
        extras={"keys": sorted(list(results.keys()))},
    )
    return {"run_dir": str(run_dir), "summary": str(infer_summary_path), **results}


def run_evaluate(config_path: str | Path) -> dict[str, Any]:
    ctx = build_train_context(
        config_path,
        on_missing_preprocess=lambda: run_preprocess(config_path),
    )
    run_dir = ctx.run_dir
    eval_dir = run_dir / "eval"
    metrics_path = eval_dir / "test_metrics.json"
    if not metrics_path.exists():
        run_train(config_path)
    payload: dict[str, Any] = {}
    if metrics_path.exists():
        payload = dict(json.loads(metrics_path.read_text(encoding="utf-8")))
    model_name = normalize_model_name(dict(ctx.cfg.get("model", {})).get("name", "global_mlp"))
    bundle = ctx.bundle
    input_mode_meta = build_input_mode_effective_metadata(ctx.cfg)
    if bundle is not None:
        input_mode_meta = attach_runtime_schema_hashes(
            input_mode_meta,
            schemas=dict(bundle.schemas or {}),
            schema_hashes=dict(bundle.schemas.get("runtime_schema_hashes", {}) or {}),
        )
    payload.update(
        resolve_effective_input_mode_metadata_for_model(
            model_name=model_name,
            input_mode_meta=input_mode_meta,
        )
    )
    out_path = ArtifactStore(eval_dir).save_json("summary.json", payload)
    _write_stage_manifest(
        run_dir=run_dir,
        stage="evaluate",
        config_path=config_path,
        depends_on=["train", "infer"],
        produced=[str(out_path)],
    )
    return {"run_dir": str(run_dir), "summary": str(out_path)}


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    stages = list(cfg.get("pipeline", {}).get("stages", ["cleanse", "feature", "preprocess", "train", "infer", "evaluate"]))
    stage_map: dict[str, Any] = {
        "cleanse": run_cleanse,
        "feature": run_feature,
        "preprocess": run_preprocess,
        "train": run_train,
        "infer": run_infer,
        "evaluate": run_evaluate,
    }
    outputs: dict[str, Any] = {}
    for stage in stages:
        key = str(stage).strip().lower()
        fn = stage_map.get(key)
        if fn is None:
            raise ValueError(f"pipeline.stages contains unsupported stage: {stage}")
        outputs[key] = fn(config_path)
    run_dir = Path(str(cfg.get("run_dir", "runs/mainline")))
    _write_stage_manifest(
        run_dir=run_dir,
        stage="pipeline",
        config_path=config_path,
        depends_on=[str(s) for s in stages],
        produced=[str(run_dir / "artifacts")],
        extras={"stages": [str(s) for s in stages]},
    )
    return {"run_dir": str(run_dir), "stages": [str(s) for s in stages], "results": outputs}


def run_viz(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    run_dir = Path(cfg.get("run_dir", "runs/mainline"))
    viz = VizRunner(run_dir / "viz")
    target_role_schema = _load_target_role_schema(run_dir)

    metrics_csv = run_dir / "train" / "scalars" / "metrics.csv"
    plots: list[str] = []

    if metrics_csv.exists():
        history: list[dict[str, float]] = []
        with metrics_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(
                    {
                        "epoch": float(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                    }
                )
        path = viz.plot_loss_curve(history)
        plots.append(str(path))

    batch_csv = run_dir / "inference" / "batch" / "summary.csv"
    if batch_csv.exists():
        with batch_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            keys = [k for k in rows[0].keys() if k != "uniformity"]
            x_key = keys[0] if keys else "uniformity"
            x = np.array([float(r[x_key]) for r in rows], dtype=np.float32)
            y = np.array([float(r["uniformity"]) for r in rows], dtype=np.float32)
            path = viz.plot_batch_qoi_scatter(x, y)
            plots.append(str(path))

    single_root = run_dir / "inference" / "single"
    if single_root.exists():
        diag_rows: list[dict[str, float | str]] = []
        region_rows: list[dict[str, float | str]] = []
        input_mode_meta = build_input_mode_effective_metadata(cfg)
        provider_mode = str(input_mode_meta.get(GEOMETRY_PROVIDER_MODE_EFFECTIVE_KEY, "fixed")).strip().lower() or "fixed"
        geom_ctx = build_geometry_provider(run_dir / "dataset", provider_mode=provider_mode).get()
        mask_plasma = geom_ctx.mask_plasma > 0.5
        mask_bulk = geom_ctx.regions.get("mask_bulk")
        if mask_bulk is None:
            mask_bulk = (geom_ctx.distance_any > 3.0) & mask_plasma
        else:
            mask_bulk = np.asarray(mask_bulk, dtype=np.float32) > 0.5
        mask_boundary = mask_plasma & (~mask_bulk)

        case_dirs = sorted([p for p in single_root.iterdir() if p.is_dir()])
        if len(case_dirs) == 0:
            return {"run_dir": str(run_dir), "plots": plots}
        for cdir in case_dirs:
            diag_rows.append(build_single_case_physics_metrics(cdir))

            with np.load(cdir / "fields_phys.npz") as fields:
                density_key = _resolve_viz_density_key(
                    field_keys=[str(k) for k in fields.files],
                    target_role_schema=target_role_schema,
                    cfg=cfg,
                )
                if density_key is not None:
                    density_map = np.asarray(fields[density_key], dtype=np.float32)[0]
                    region_rows.append(
                        build_region_metrics(
                            case_key=cdir.name,
                            density=density_map,
                            density_key=density_key,
                            mask_plasma=mask_plasma,
                            mask_bulk=mask_bulk,
                            mask_boundary=mask_boundary,
                        )
                )

        if diag_rows:
            table_payload = build_viz_tables_payload(diag_rows=diag_rows, region_rows=region_rows)
            ArtifactStore(run_dir / "viz").save_csv(
                "tables/physics_diagnostics.csv",
                table_payload["diag_header"],
                table_payload["diag_rows"],
            )
            plot_rows = [
                r
                for r in diag_rows[:20]
                if bool(r.get("physics_diagnostics_available", True))
                and np.isfinite(float(r.get("poisson_residual_norm", float("nan"))))
            ]
            if plot_rows:
                labels = [str(r["case_key"]) for r in plot_rows]
                resid_vals = [float(r["poisson_residual_norm"]) for r in plot_rows]
                bc_vals = [float(r["bc_potential_mae"]) for r in plot_rows]
                plots.append(
                    str(viz.plot_metric_bar(labels, resid_vals, "poisson_residual", rel_path="plots/poisson_residual_bar.png"))
                )
                plots.append(
                    str(viz.plot_metric_bar(labels, bc_vals, "bc_potential_mae", rel_path="plots/bc_potential_mae_bar.png"))
                )
                map_rows = [
                    r
                    for r in plot_rows
                    if np.isfinite(float(r.get("poisson_residual_map_l2", float("nan"))))
                ]
                if map_rows:
                    plots.append(
                        str(
                            viz.plot_metric_bar(
                                [str(r["case_key"]) for r in map_rows],
                                [float(r["poisson_residual_map_l2"]) for r in map_rows],
                                "poisson_residual_map_l2",
                                rel_path="plots/poisson_residual_map_l2_bar.png",
                            )
                        )
                    )

        if region_rows:
            table_payload = build_viz_tables_payload(diag_rows=diag_rows, region_rows=region_rows)
            ArtifactStore(run_dir / "viz").save_csv(
                "tables/region_metrics.csv",
                table_payload["region_header"],
                table_payload["region_rows"],
            )

    return {"run_dir": str(run_dir), "plots": plots}
