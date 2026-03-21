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
from plasma_surrogate.core.model_families import COND_ONLY_TORCH_MODELS, GRID_TORCH_MODELS
from plasma_surrogate.core.cond_utils import build_cond_matrix_with_axis
from plasma_surrogate.core.data_cleaning_audit import run_data_audit
from plasma_surrogate.core.physics_contract import build_physics_cfg
from plasma_surrogate.data.geometry_provider import FixedGeometryProvider
from plasma_surrogate.eval.metrics_builder import (
    build_eval_metrics_payload,
    build_region_metrics,
    build_single_case_physics_metrics,
    build_viz_tables_payload,
)
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.heads.plasma_head import PlasmaHead
from plasma_surrogate.models.mlp.io import save_mlp_checkpoint
from plasma_surrogate.pipeline.runtime_context import (
    build_infer_context,
    build_preprocess_context,
    build_train_context,
)
from plasma_surrogate.preprocessing.runner import PreprocessRunner
from plasma_surrogate.train.model_dispatch import TrainDispatchContext, run_model_train_predict
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
) -> InferenceEngine:
    cfg = dict(bundle.cfg or {})
    model_cfg = cfg.get("model", {})
    inf_cfg = cfg.get("inference", {})
    return InferenceEngine(
        model=bundle.model,
        cond_schema=bundle.cond_schema_obj(),
        axis_schema=bundle.axis_schema_obj(),
        geometry_provider=FixedGeometryProvider(dataset_root),
        output_dir=output_dir,
        transform_bundle=bundle.transform_bundle(),
        cond_stats=bundle.schemas.get("cond_stats", {}),
        phi_mode=str(model_cfg.get("phi_mode", "direct")),
        phi_hybrid_steps=int(model_cfg.get("phi_hybrid_steps", 1)),
        poisson_refine_iters=int(inf_cfg.get("poisson_refine", {}).get("iters", 0)),
        ood_cfg=inf_cfg.get("ood", {"poisson_residual_limit": 1e2}),
        feature_store=feature_store,
        coord_scaler=bundle.transforms.get("coord_scaler", {}),
        coord_feature_scaler=bundle.transforms.get("coord_feature_scaler", {}),
        coord_feature_pack=bundle.schemas.get("coord_feature_pack"),
        coord_distance_transform_stats=bundle.transforms.get("distance_transform_stats", {}),
        deeponet_head=(
            getattr(bundle.model, "poisson_head", None)
            if str(model_cfg.get("phi_mode", "direct")) == "deeponet_poisson"
            else None
        )
        or (bundle.model if str(model_cfg.get("phi_mode", "direct")) == "deeponet_poisson" else None),
        grid_input_features_cfg=dict(cfg.get("train", {}).get(str(model_cfg.get("name", "unet")), {}).get("input_features", {})),
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
    geom_ctx = FixedGeometryProvider(dataset_root).get()
    return build_physics_cfg(
        raw_cfg=phys,
        geom_ctx=geom_ctx,
        default_enabled=False,
        default_primary_qoi_key="Gamma_i",
        default_lambda_poisson=0.05,
        default_lambda_bc=0.02,
    )


def _resolve_optimize_backend(opt_cfg: dict[str, Any]) -> str:
    backend = opt_cfg.get("backend")
    if backend is not None:
        return str(backend)
    legacy_sampler = opt_cfg.get("sampler")
    if legacy_sampler is None:
        return "random"
    sampler = str(legacy_sampler).strip().lower()
    if sampler == "random":
        return "random"
    if sampler == "optuna_grid":
        return "optuna"
    if sampler == "csv":
        return "csv"
    raise ValueError(
        "Unsupported inference.optimize.sampler. "
        "Use backend='random|optuna|csv' (legacy sampler supports random|optuna_grid|csv)."
    )


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

    pre = PreprocessRunner(pre_cfg, run_dir / "preprocessing")
    output = pre.run(cases=dataset.cases, geometry_root=dataset.geometry_root)
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
    model_name = str(model_cfg.get("name", "global_mlp"))
    phi_mode = str(model_cfg.get("phi_mode", "direct"))
    phi_hybrid_steps = int(model_cfg.get("phi_hybrid_steps", 1))
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 1e-2))
    loss_cfg = dict(train_cfg.get("loss", {}))
    curriculum_cfg = dict(train_cfg.get("curriculum", {}))
    physics_cfg = _resolve_physics_cfg(cfg, dataset_root=dataset.geometry_root)
    geom_ctx = None
    use_plasma_mask = str(loss_cfg.get("supervised", {}).get("mask", "none")).strip().lower() == "plasma_only"
    if phi_mode != "direct" or physics_cfg.get("enabled") or use_plasma_mask:
        geom_ctx = feat_store.get_context(
            geom_ref={"geom_id": "default"},
            axis_value=0.0,
            axis_mode=axis_mode,
            geometry_provider=FixedGeometryProvider(dataset.geometry_root),
        )
    if model_name == "deeponet_plasma" and geom_ctx is None:
        geom_ctx = feat_store.get_context(
            geom_ref={"geom_id": "default"},
            axis_value=0.0,
            axis_mode=axis_mode,
            geometry_provider=FixedGeometryProvider(dataset.geometry_root),
        )
    supervised_mask = np.asarray(geom_ctx.mask_plasma, dtype=np.float32) if (geom_ctx is not None and use_plasma_mask) else None
    supervised_distance = np.asarray(geom_ctx.distance_any, dtype=np.float32) if (geom_ctx is not None and use_plasma_mask) else None
    plasma_head = PlasmaHead(mode=phi_mode, jacobi_iters=phi_hybrid_steps)

    train_dispatch_cfg: dict[str, Any] = {"train": {}}
    if model_name == "global_mlp":
        per_model_cfg = dict(train_cfg.get(model_name, {}))
        merged_model_cfg = dict(per_model_cfg.get("model_cfg", {}))
        merged_model_cfg.update(dict(model_cfg))
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
        raise ValueError(f"Unsupported model.name for cycle1: {model_name}")

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
            coord_distance_transform_stats=dict(bundle.transforms.get("distance_transform_stats", {})),
        )
    )
    model = dispatch.model
    history_len = len(dispatch.history)
    pred_eval = dict(dispatch.pred_eval)

    if geom_ctx is not None and plasma_head is not None and "phi" in pred_eval:
        deeponet_head = None
        if phi_mode == "deeponet_poisson":
            deeponet_head = getattr(model, "poisson_head", None)
            if deeponet_head is None and hasattr(model, "predict_phi"):
                deeponet_head = model
            elif deeponet_head is None and hasattr(model, "predict_fields"):
                deeponet_head = model
        pred_eval, head_aux = plasma_head.apply(
            pred_eval,
            geom_ctx=geom_ctx,
            refine_iters=0,
            deeponet_head=deeponet_head,
            cond_vec=cond[te],
        )
        pred_eval.update({k: v for k, v in head_aux.items() if k not in pred_eval})

    true_eval = dispatch.true_eval
    eval_payload = build_eval_metrics_payload(
        true_eval=true_eval,
        pred_eval={k: pred_eval[k] for k in y_vars if k in pred_eval},
        eps=geom_ctx.eps if geom_ctx is not None else None,
        mask_plasma=np.asarray(geom_ctx.mask_plasma, dtype=np.float32) if geom_ctx is not None else None,
    )
    metrics = dict(eval_payload["rmse"])
    r2 = dict(eval_payload.get("r2", {}))
    metrics_plasma = dict(eval_payload.get("rmse_plasma", {}))
    r2_plasma = dict(eval_payload.get("r2_plasma", {}))

    ArtifactStore(run_dir / "eval").save_json("test_metrics.json", eval_payload)
    save_mlp_checkpoint(model, run_dir / "checkpoints")
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

    model = bundle.model
    if model is None:
        raise FileNotFoundError(f"Missing model checkpoint metadata under {run_dir / 'checkpoints'}")
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "global_mlp"))
    inf_cfg = cfg.get("inference", {})
    cond_schema = bundle.cond_schema_obj()
    axis_schema = bundle.axis_schema_obj()
    transforms = bundle.transform_bundle()
    dataset_root = dataset.geometry_root
    engine = InferenceEngine(
        model=model,
        cond_schema=cond_schema,
        axis_schema=axis_schema,
        geometry_provider=FixedGeometryProvider(dataset_root),
        output_dir=run_dir / "inference",
        transform_bundle=transforms,
        cond_stats=bundle.schemas.get("cond_stats", {}),
        phi_mode=str(model_cfg.get("phi_mode", "direct")),
        phi_hybrid_steps=int(model_cfg.get("phi_hybrid_steps", 1)),
        poisson_refine_iters=int(inf_cfg.get("poisson_refine", {}).get("iters", 0)),
        ood_cfg=inf_cfg.get("ood", {"poisson_residual_limit": 1e2}),
        feature_store=feat_store,
        coord_scaler=bundle.transforms.get("coord_scaler", {}),
        coord_feature_scaler=bundle.transforms.get("coord_feature_scaler", {}),
        coord_feature_pack=bundle.schemas.get("coord_feature_pack"),
        coord_distance_transform_stats=bundle.transforms.get("distance_transform_stats", {}),
        deeponet_head=(
            getattr(model, "poisson_head", None)
            if str(model_cfg.get("phi_mode", "direct")) == "deeponet_poisson"
            else None
        )
        or (model if str(model_cfg.get("phi_mode", "direct")) == "deeponet_poisson" else None),
        grid_input_features_cfg=dict(cfg.get("train", {}).get(model_name, {}).get("input_features", {})),
    )

    results: dict[str, Any] = {}
    single_cfg = inf_cfg.get("single", {})
    if single_cfg.get("enabled", True):
        cond = single_cfg.get("cond", {k: 0.5 for k in cond_schema.order})
        axis = single_cfg.get("axis", {"mode": axis_schema.mode, "value": 0.0})
        geom = single_cfg.get("geom", {"geom_id": "default"})
        single = engine.single_run_aggregated(cond=cond, geom=geom, axis=axis)
        results["single_qoi"] = single.qoi
        results["single_warnings"] = list(single.warnings)

    batch_cfg = inf_cfg.get("batch", {})
    if batch_cfg.get("enabled", False):
        conds = batch_cfg.get("conds", [{k: 0.2 for k in cond_schema.order}, {k: 0.8 for k in cond_schema.order}])
        axis = batch_cfg.get("axis", {"mode": axis_schema.mode, "value": 0.0})
        geom = batch_cfg.get("geom", {"geom_id": "default"})
        rows = engine.batch_run(conds=conds, geom=geom, axis=axis)
        results["batch_count"] = len(rows)

    opt_cfg = inf_cfg.get("optimize", {})
    if opt_cfg.get("enabled", False):
        axis = opt_cfg.get("axis", {"mode": axis_schema.mode, "value": 0.0})
        space_cfg = opt_cfg.get("space", {k: [0.0, 1.0] for k in cond_schema.order})
        space = {k: (float(v[0]), float(v[1])) for k, v in space_cfg.items()}
        backend = _resolve_optimize_backend(opt_cfg)
        backend_cfg = dict(opt_cfg.get("backend_cfg", {}) or {})
        if backend == "csv" and "csv_path" in backend_cfg:
            csv_path = Path(str(backend_cfg["csv_path"]))
            if not csv_path.is_absolute():
                csv_path = Path(config_path).resolve().parent / csv_path
            backend_cfg["csv_path"] = str(csv_path)
        best = engine.optimize_run(
            space=space,
            n_trials=int(opt_cfg.get("n_trials", 10)),
            geom=opt_cfg.get("geom", {"geom_id": "default"}),
            axis=axis,
            seed=int(opt_cfg.get("seed", 0)),
            backend=backend,
            backend_cfg=backend_cfg,
        )
        results["optimize"] = best

    _write_stage_manifest(
        run_dir=run_dir,
        stage="infer",
        config_path=config_path,
        depends_on=["train"],
        produced=[str(run_dir / "inference")],
        extras={"keys": sorted(list(results.keys()))},
    )
    return {"run_dir": str(run_dir), **results}


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
    payload = {}
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
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
    run_dir = Path(str(cfg.get("run_dir", "runs/cycle1")))
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

    run_dir = Path(cfg.get("run_dir", "runs/cycle1"))
    viz = VizRunner(run_dir / "viz")

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
        geom_ctx = FixedGeometryProvider(run_dir / "dataset").get()
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

            fields = np.load(cdir / "fields_phys.npz")
            density_key = "ne" if "ne" in fields.files else "log_ne"
            if density_key not in fields.files:
                raise KeyError(f"fields_phys.npz must include ne or log_ne: {cdir}")
            density_map = np.asarray(fields[density_key], dtype=np.float32)[0]
            if density_key == "log_ne":
                density_map = np.power(10.0, density_map.astype(np.float64)).astype(np.float32)
            region_rows.append(
                build_region_metrics(
                    case_key=cdir.name,
                    density=density_map,
                    density_key="ne",
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
            labels = [str(r["case_key"]) for r in diag_rows[:20]]
            resid_vals = [float(r["poisson_residual_norm"]) for r in diag_rows[:20]]
            bc_vals = [float(r["bc_phi_mae"]) for r in diag_rows[:20]]
            resid_map_vals = [float(r["poisson_residual_map_l2"]) for r in diag_rows[:20]]
            plots.append(str(viz.plot_metric_bar(labels, resid_vals, "poisson_residual", rel_path="plots/poisson_residual_bar.png")))
            plots.append(str(viz.plot_metric_bar(labels, bc_vals, "bc_phi_mae", rel_path="plots/bc_phi_mae_bar.png")))
            plots.append(
                str(
                    viz.plot_metric_bar(
                        labels,
                        resid_map_vals,
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
