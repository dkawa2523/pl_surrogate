"""Benchmark runner for mainline model comparison."""

from __future__ import annotations

import copy
import gc
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from plasma_surrogate.core.boundary_distance import boundary_distance_channels_from_loss_cfg
from plasma_surrogate.benchmark.runtime_context import (
    build_benchmark_data_context,
    resolve_effective_benchmark_cfg,
)
from plasma_surrogate.benchmark.contract_summary import BenchmarkContractEmitter
from plasma_surrogate.benchmark.cv import CvBenchmarkRunner
from plasma_surrogate.benchmark.eval_diagnostics import (
    resolve_eval_diagnostics_cfg,
    write_eval_diagnostics,
)
from plasma_surrogate.benchmark.eval_geometry import materialize_evaluation_spatial_geometry
from plasma_surrogate.benchmark.manifest import BenchmarkManifestWriter
from plasma_surrogate.benchmark.metric_tables import save_leaderboard_metric_tables
from plasma_surrogate.benchmark.planning import BenchmarkPlanBuilder, build_leaderboard_header
from plasma_surrogate.benchmark.profiles import resolve_profile_lock
from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.contracts import RuntimeContract
from plasma_surrogate.core.input_modes import (
    attach_runtime_schema_hashes,
    build_input_mode_effective_metadata,
    input_mode_metadata_keys,
    runtime_metadata_keys,
)
from plasma_surrogate.core.model_families import (
    GRID_TORCH_MODELS,
    POD_DEEPONET_FAMILY_MODELS,
    SPECTRAL_FAMILY_MODELS,
    UNET_FAMILY_MODELS,
    resolve_single_family_model,
)
from plasma_surrogate.core.model_input_policy import resolve_effective_input_mode_metadata_for_model
from plasma_surrogate.core.model_specs import MODEL_SPECS, benchmark_scope_model_map
from plasma_surrogate.core.physics_contract import build_physics_cfg
from plasma_surrogate.core.physics_numeric import poisson_residual_loss
from plasma_surrogate.core.target_roles import resolve_physics_symbol_keys
from plasma_surrogate.eval.core_metrics import build_benchmark_eval_row
from plasma_surrogate.eval.quality_score import resolve_quality_score_protocol
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.infer.engine_builder import build_inference_engine
from plasma_surrogate.infer.optimize import cond_space_from_stats, validate_optimize_geom_contract
from plasma_surrogate.models.checkpoint import save_checkpoint
from plasma_surrogate.benchmark.model_dispatch import BenchmarkModelContext, run_model_train_eval
from plasma_surrogate.benchmark.tuning_selection import (
    VALIDATION_VALUE_KEY,
    require_validation_objective,
    validation_selection_from_history,
)
from plasma_surrogate.preprocessing.split import build_group_kfold_splits
from plasma_surrogate.train.loss_protocols import loss_protocol_metadata, resolve_loss_protocol
from plasma_surrogate.train.model_artifacts import merge_checkpoint_dispatch_metadata
from plasma_surrogate.viz.runner import VizRunner


def _benchmark_physics_symbols(ood_cfg: dict[str, Any]) -> dict[str, Any]:
    symbols: dict[str, Any] = {}
    for section_name in ("physics", "boundary_operator"):
        section = dict(dict(ood_cfg or {}).get(section_name, {}) or {})
        raw = section.get("symbols")
        if isinstance(raw, dict):
            symbols.update({str(key): value for key, value in raw.items()})
    return symbols


def _resolve_benchmark_potential_key(
    *,
    field_keys: list[str],
    target_role_schema: dict[str, Any] | None,
    ood_cfg: dict[str, Any],
    context: str,
) -> str | None:
    symbols = _benchmark_physics_symbols(ood_cfg)
    try:
        resolved = resolve_physics_symbol_keys(
            [str(key) for key in field_keys],
            symbols=symbols,
            target_role_schema=dict(target_role_schema or {}),
            required=("potential",),
            context=context,
        )
    except ValueError:
        if symbols.get("potential") is not None:
            raise
        return None
    return resolved["potential"]


@dataclass
class BenchmarkResult:
    leaderboard: list[dict[str, Any]]
    leaderboard_path: Path


@dataclass
class SweepResult:
    summary_path: Path
    best_trial: dict[str, Any]
    locked_config_path: Path
    locked_leaderboard_path: Path


@dataclass(frozen=True)
class BenchmarkModelResult:
    status: str
    skip_reason: str | None = None
    objective_value: float | None = None
    search_value: float | None = None
    feasible: bool = False
    objective_mode: str = "weighted_sum"
    violated_constraints: tuple[str, ...] = ()
    constraint_violation_total: float | None = None

    def as_summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "skip_reason": self.skip_reason,
            "objective_value": self.objective_value,
            "search_value": self.search_value,
            "feasible": bool(self.feasible),
            "objective_mode": self.objective_mode,
            "violated_constraints": list(self.violated_constraints),
            "constraint_violation_total": self.constraint_violation_total,
        }


@dataclass(frozen=True)
class BenchmarkProbeResult:
    qoi: dict[str, Any]
    diagnostics: dict[str, Any]
    batch_rows: list[dict[str, Any]]
    optimize: BenchmarkModelResult


@dataclass(frozen=True)
class SingleSplitResult:
    row: dict[str, Any]
    effective_steps: int
    contract_artifacts: dict[str, dict[str, Any]]

    def contract(self, name: str) -> dict[str, Any]:
        return dict(self.contract_artifacts.get(name, {}))


@dataclass
class BenchmarkContractSamples:
    unet: list[dict[str, Any]]
    fno: list[dict[str, Any]]
    ffno: list[dict[str, Any]]
    coord_mlp: list[dict[str, Any]]
    deeponet: list[dict[str, Any]]
    deeponet_pod: list[dict[str, Any]]

    @classmethod
    def empty(cls) -> "BenchmarkContractSamples":
        return cls(unet=[], fno=[], ffno=[], coord_mlp=[], deeponet=[], deeponet_pod=[])

    def add(self, result: SingleSplitResult) -> None:
        for name, bucket in [
            ("unet", self.unet),
            ("fno", self.fno),
            ("ffno", self.ffno),
            ("coord_mlp", self.coord_mlp),
            ("deeponet", self.deeponet),
            ("deeponet_pod", self.deeponet_pod),
        ]:
            sample = result.contract(name)
            if sample:
                bucket.append(sample)


_ISOLATED_SCOPE_MODEL_NAMES: dict[str, list[str]] = benchmark_scope_model_map()
_ISOLATED_SECTION_NAMES: tuple[str, ...] = tuple(MODEL_SPECS.keys())

_ISOLATED_SCOPE_FORBIDDEN_SECTIONS: dict[str, list[str]] = {
    scope: [name for name in _ISOLATED_SECTION_NAMES if name not in set(active)]
    for scope, active in _ISOLATED_SCOPE_MODEL_NAMES.items()
}
_EVAL_PROTOCOL_SCOPES: tuple[str, ...] = ("common", *_ISOLATED_SCOPE_MODEL_NAMES.keys())
_FROZEN_REFERENCE_SCOPES: set[str] = set(_ISOLATED_SCOPE_MODEL_NAMES.keys())
_UNET_CONTRACT_OPTIONAL_SCOPES: set[str] = {
    scope
    for scope, names in _ISOLATED_SCOPE_MODEL_NAMES.items()
    if not any(name in UNET_FAMILY_MODELS for name in names)
}


def _release_torch_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _evaluation_case_count(
    *,
    pred_eval: dict[str, Any],
    true_eval: dict[str, Any],
) -> int:
    """Return one fail-closed case count shared by evaluation geometry."""

    counts: set[int] = set()
    for source_name, fields in (("pred_eval", pred_eval), ("true_eval", true_eval)):
        for name, values in fields.items():
            arr = np.asarray(values)
            if arr.ndim < 1:
                raise ValueError(f"{source_name}[{name!r}] must include a case axis")
            counts.add(int(arr.shape[0]))
    if not counts:
        raise ValueError("benchmark evaluation requires at least one prediction/target field")
    if len(counts) != 1:
        raise ValueError(f"benchmark evaluation fields have inconsistent case counts: {sorted(counts)}")
    return int(next(iter(counts)))


def _inject_input_mode_metadata_into_row(
    *,
    row: dict[str, Any],
    input_mode_meta: dict[str, Any],
    case_spatial_pack_used: bool = False,
) -> None:
    contract = RuntimeContract.from_metadata(
        input_mode_meta,
        context="benchmark row runtime_contract",
        allow_pending_schema_hashes=False,
    )
    missing = [key for key in input_mode_metadata_keys() if key not in input_mode_meta]
    if missing:
        raise ValueError(
            "benchmark row metadata is missing required input_mode keys: "
            f"{missing}"
        )
    input_mode_meta = contract.to_metadata()
    for key in runtime_metadata_keys():
        if key in input_mode_meta:
            row[key] = input_mode_meta[key]
    input_mode = str(input_mode_meta.get("input_mode_effective", "")).strip().lower()
    provider_mode = str(input_mode_meta.get("geometry_provider_mode_effective", "fixed")).strip().lower()
    if input_mode == "table_only":
        structure_kind = "none"
    elif provider_mode == "parametric_parts" or bool(case_spatial_pack_used):
        structure_kind = "case_varying_geometry"
    else:
        structure_kind = "static_grid_features"
    row["structure_input_kind_effective"] = structure_kind
    row["has_case_varying_structure_inputs_effective"] = bool(structure_kind == "case_varying_geometry")


def _primary_metric_value(row: dict[str, Any], *, primary_metric: str, model_name: str) -> float:
    if primary_metric not in row:
        keys = sorted(str(key) for key in row.keys())
        preview = keys[:30]
        suffix = "" if len(keys) <= len(preview) else f", ... (+{len(keys) - len(preview)} more)"
        raise ValueError(
            f"benchmark.eval.primary_metric={primary_metric!r} is not present for model {model_name!r}; "
            f"available row keys include: {preview}{suffix}"
        )
    try:
        return float(row[primary_metric])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"benchmark.eval.primary_metric={primary_metric!r} for model {model_name!r} must be numeric; "
            f"got {row[primary_metric]!r}"
        ) from exc


def _target_metric_validity(row: dict[str, Any], *, target_vars: list[str]) -> tuple[bool, list[str]]:
    invalid: list[str] = []
    diagnostics = row.get("_diagnostics")
    diag_row = diagnostics if isinstance(diagnostics, dict) else {}
    for var_name in [str(v) for v in target_vars]:
        rmse_key = f"test_rmse_{var_name}_plasma" if f"test_rmse_{var_name}_plasma" in row else f"test_rmse_{var_name}"
        r2_key = f"test_r2_{var_name}_plasma" if f"test_r2_{var_name}_plasma" in row else f"test_r2_{var_name}"
        missing = [key for key in [rmse_key, r2_key] if key not in row]
        if missing:
            invalid.append(f"{var_name}:missing")
            continue
        try:
            rmse_val = float(row[rmse_key])
            r2_val = float(row[r2_key])
        except (TypeError, ValueError):
            invalid.append(str(var_name))
            continue
        nonfinite_count = float(diag_row.get(f"test_nonfinite_count_{var_name}_plasma", 0.0) or 0.0)
        if nonfinite_count > 0.0 or not np.isfinite(rmse_val) or not np.isfinite(r2_val):
            invalid.append(str(var_name))
    return len(invalid) == 0, invalid


def _attach_primary_metric_status(
    row: dict[str, Any],
    *,
    primary_metric: str,
    model_name: str,
    target_vars: list[str],
) -> None:
    row["primary_metric"] = primary_metric
    if primary_metric not in row:
        row["primary_metric_value"] = float("nan")
    else:
        row["primary_metric_value"] = _primary_metric_value(
            row,
            primary_metric=primary_metric,
            model_name=model_name,
        )
    target_valid, invalid_vars = _target_metric_validity(row, target_vars=target_vars)
    row["target_metrics_valid"] = bool(target_valid and bool(row.get("target_metrics_valid", True)))
    diagnostics = row.get("_diagnostics")
    diag_invalid = str(diagnostics.get("target_metrics_invalid_vars", "")) if isinstance(diagnostics, dict) else ""
    row["target_metrics_invalid_vars"] = "|".join(
        sorted(
            set(
                [
                    *invalid_vars,
                    *str(row.get("target_metrics_invalid_vars", "")).split("|"),
                    *diag_invalid.split("|"),
                ]
            )
            - {""}
        )
    )
    row["primary_metric_reliable"] = bool(
        np.isfinite(float(row["primary_metric_value"])) and row["target_metrics_valid"]
    )


def _build_benchmark_inference_engine(
    *,
    model: Any,
    cond_schema: Any,
    axis_schema: Any,
    geometry_provider: Any,
    output_dir: Path,
    transform_bundle: Any,
    cond_stats: dict[str, Any],
    phi_mode: str,
    phi_hybrid_steps: int,
    poisson_refine_iters: int,
    ood_cfg: dict[str, Any],
    feature_store: Any,
    coord_scaler: dict[str, Any],
    coord_feature_scaler: dict[str, Any],
    coord_feature_pack: dict[str, Any] | None,
    coord_distance_transform_stats: dict[str, Any],
    input_mode_meta: dict[str, Any],
    checkpoint_meta_path: Path,
    target_role_schema: dict[str, Any] | None = None,
    deeponet_head: Any | None = None,
    grid_input_features_cfg: dict[str, Any] | None = None,
) -> InferenceEngine:
    return build_inference_engine(
        model=model,
        cond_schema=cond_schema,
        axis_schema=axis_schema,
        geometry_provider=geometry_provider,
        output_dir=output_dir,
        transform_bundle=transform_bundle,
        cond_stats=cond_stats,
        phi_mode=phi_mode,
        phi_hybrid_steps=phi_hybrid_steps,
        poisson_refine_iters=poisson_refine_iters,
        ood_cfg=ood_cfg,
        feature_store=feature_store,
        coord_scaler=coord_scaler,
        coord_feature_scaler=coord_feature_scaler,
        coord_feature_pack=coord_feature_pack,
        coord_distance_transform_stats=coord_distance_transform_stats,
        input_mode_meta=input_mode_meta,
        checkpoint_meta_path=checkpoint_meta_path,
        target_role_schema=dict(target_role_schema or {}),
        deeponet_head=deeponet_head,
        grid_input_features_cfg=dict(grid_input_features_cfg or {}),
    )


@dataclass(frozen=True)
class BenchmarkProbe:
    """Optional inference/optimization smoke executor for one benchmark model."""

    benchmark_cfg: dict[str, Any]

    def _probe_cfg(self) -> dict[str, Any]:
        inference_cfg = dict(self.benchmark_cfg.get("inference", {}) or {})
        raw = inference_cfg.get("benchmark_probe", inference_cfg.get("smoke", {}))
        return dict(raw or {})

    def enabled(self) -> bool:
        return bool(self._probe_cfg().get("enabled", False))

    def objective_mode(self) -> str:
        optimize_cfg = dict(dict(self.benchmark_cfg.get("inference", {}) or {}).get("optimize", {}) or {})
        return str(dict(optimize_cfg.get("objective", {}) or {}).get("mode", "weighted_sum"))

    def skipped(self, reason: str) -> BenchmarkProbeResult:
        return BenchmarkProbeResult(
            qoi={"uniformity": 0.0, "boundary_gamma_uniformity": 0.0},
            diagnostics={"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0},
            batch_rows=[],
            optimize=BenchmarkModelResult(
                status="skipped",
                skip_reason=reason,
                objective_mode=self.objective_mode(),
            ),
        )

    def run(
        self,
        *,
        model: Any,
        model_name: str,
        model_idx: int,
        model_dir: Path,
        context: Any,
        profile_lock: dict[str, Any],
        train_cfg: dict[str, Any],
        effective_input_mode_meta: dict[str, Any],
        true_eval: dict[str, np.ndarray],
        pred_eval: dict[str, np.ndarray],
        metric_mask: np.ndarray | None,
        te_idx: np.ndarray,
        viz: VizRunner,
        transform_bundle: Any | None = None,
        coord_feature_scaler: dict[str, Any] | None = None,
        coord_distance_transform_stats: dict[str, Any] | None = None,
    ) -> BenchmarkProbeResult:
        if not self.enabled():
            return self.skipped("benchmark_probe_disabled")
        inference_ood_cfg = BenchmarkPlanBuilder(self.benchmark_cfg).inference_ood_cfg()
        target_role_schema = dict(context.bundle.schemas.get("target_role_schema", {}) or {})
        common_keys = sorted(set(str(key) for key in true_eval.keys()) & set(str(key) for key in pred_eval.keys()))
        potential_key = _resolve_benchmark_potential_key(
            field_keys=common_keys,
            target_role_schema=target_role_schema,
            ood_cfg=inference_ood_cfg,
            context="benchmark probe",
        )
        if potential_key is None:
            return self.skipped("missing_potential_target")
        case_spatial_inference_skip = bool(context.case_structure_feature_pack)
        if case_spatial_inference_skip:
            return self.skipped("case_varying_structure_inputs_not_supported_by_benchmark_inference")

        bundle = context.bundle
        viz.plot_parity(
            true_eval[potential_key].reshape(-1),
            pred_eval[potential_key].reshape(-1),
            rel_path="plots/parity_potential.png",
        )
        viz.plot_field_triplet(
            true_eval[potential_key][0, 0],
            pred_eval[potential_key][0, 0],
            rel_path="plots/potential_triplet.png",
            mask=metric_mask,
        )

        inference_cfg = dict(self.benchmark_cfg.get("inference", {}) or {})
        infer_engine = _build_benchmark_inference_engine(
            model=model,
            cond_schema=bundle.cond_schema_obj(),
            axis_schema=bundle.axis_schema_obj(),
            geometry_provider=context.geom_provider,
            output_dir=model_dir / "inference",
            transform_bundle=transform_bundle or context.transforms,
            cond_stats=bundle.schemas.get("cond_stats", {}),
            phi_mode=profile_lock["phi_mode"],
            phi_hybrid_steps=int(self.benchmark_cfg.get("phi_hybrid_steps", 1)),
            poisson_refine_iters=int(profile_lock["poisson_refine_iters"]),
            ood_cfg=inference_ood_cfg,
            feature_store=context.feature_store,
            coord_scaler=bundle.transforms.get("coord_scaler", {}),
            coord_feature_scaler=dict(
                coord_feature_scaler
                if coord_feature_scaler is not None
                else bundle.transforms.get("coord_feature_scaler", {})
            ),
            coord_feature_pack=bundle.schemas.get("coord_feature_pack"),
            coord_distance_transform_stats=dict(
                coord_distance_transform_stats
                if coord_distance_transform_stats is not None
                else bundle.transforms.get("distance_transform_stats", {})
            ),
            input_mode_meta=effective_input_mode_meta,
            checkpoint_meta_path=model_dir / "checkpoints" / "meta.json",
            target_role_schema=bundle.schemas.get("target_role_schema", {}),
            deeponet_head=(
                getattr(model, "poisson_head", None)
                if model_name == "deeponet_plasma"
                else None
            )
            or (model if model_name == "deeponet_plasma" else None),
            grid_input_features_cfg=dict(train_cfg.get(model_name, {}).get("input_features", {})),
        )
        infer_axis = inference_cfg.get(
            "axis",
            {"mode": context.requested_axis_mode, "value": 0.0},
        )
        cond_order = context.cond_order
        cond = context.cond
        cond_dict = {k: float(cond[te_idx[0], i]) for i, k in enumerate(cond_order)}
        single = infer_engine.single_run_aggregated(
            cond=cond_dict,
            geom={"geom_id": "default"},
            axis=infer_axis,
        )
        batch_rows = infer_engine.batch_run(
            conds=[
                {k: float(cond[idx, i]) for i, k in enumerate(cond_order)}
                for idx in te_idx[: min(3, len(te_idx))]
            ],
            geom={"geom_id": "default"},
            axis=infer_axis,
        )
        optimize_cfg = dict(inference_cfg.get("optimize", {}) or {})
        backend_cfg = dict(optimize_cfg.get("backend_cfg", {}) or {})
        best = infer_engine.optimize_run(
            space=cond_space_from_stats(
                list(cond_order),
                bundle.schemas.get("cond_stats", {}),
                cfg_key="benchmark.inference.optimize.space",
            ),
            geom_space=validate_optimize_geom_contract(
                input_mode=str(effective_input_mode_meta.get("input_mode_effective", "")),
                provider_mode=str(effective_input_mode_meta.get("geometry_provider_mode_effective", "fixed")),
                geom_space=None,
                geom_ref={"geom_id": "default"},
                label_prefix="inference.optimize",
            ),
            n_trials=int(optimize_cfg.get("n_trials", 8)),
            geom={"geom_id": "default"},
            axis=infer_axis,
            seed=context.global_seed + 100 + model_idx,
            backend=str(optimize_cfg.get("backend", "random")),
            backend_cfg=backend_cfg,
            objective_cfg=dict(optimize_cfg.get("objective", {}) or {}),
            constraints_cfg=optimize_cfg.get("constraints", []) or [],
            output_cfg=dict(optimize_cfg.get("output", {}) or {}),
        )
        objective_value = best.get("objective_value")
        search_value = best.get("search_value", objective_value)
        return BenchmarkProbeResult(
            qoi=dict(single.qoi),
            diagnostics=dict(single.diagnostics),
            batch_rows=batch_rows,
            optimize=BenchmarkModelResult(
                status=str(best.get("status", "succeeded" if objective_value is not None else "failed")),
                skip_reason=None,
                objective_value=(None if objective_value is None else float(objective_value)),
                search_value=(None if search_value is None else float(search_value)),
                feasible=bool(best.get("feasible", True)),
                objective_mode=str(best.get("objective_mode", "weighted_sum")),
                violated_constraints=tuple(str(v) for v in list(best.get("violated_constraints", []) or [])),
                constraint_violation_total=(
                    None
                    if best.get("constraint_violation_total") is None
                    else float(best.get("constraint_violation_total"))
                ),
            ),
        )


class BenchmarkRunner:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = copy.deepcopy(dict(cfg or {}))
        self.benchmark_cfg = resolve_effective_benchmark_cfg(self.cfg)
        self.input_mode_meta = build_input_mode_effective_metadata(self.benchmark_cfg)
        self.output_root = Path(self.benchmark_cfg.get("output_dir", "runs/benchmark_mainline"))
        self.store = ArtifactStore(self.output_root)
        self.manifest_writer = BenchmarkManifestWriter(self.store)
        self.contract_emitter = BenchmarkContractEmitter(
            unet_contract_optional_scopes=_UNET_CONTRACT_OPTIONAL_SCOPES
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkRunner":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})

    def run(self) -> BenchmarkResult:
        self.output_root.mkdir(parents=True, exist_ok=True)

        plan_builder = BenchmarkPlanBuilder(self.benchmark_cfg)
        profile = plan_builder.profile_name()
        profile_lock = self._resolve_profile_lock(profile)
        plan_builder.validate_profile_lock(profile_lock)
        context = build_benchmark_data_context(
            cfg=self.benchmark_cfg,
            output_root=self.output_root,
            profile_lock=profile_lock,
        )
        profile_lock = context.profile_lock
        resolved = context.resolved
        self.input_mode_meta = attach_runtime_schema_hashes(
            self.input_mode_meta,
            schemas=dict(context.bundle.schemas or {}),
            schema_hashes=dict(context.bundle.schemas.get("runtime_schema_hashes", {}) or {}),
        )
        resolved.update(self.input_mode_meta)
        global_seed = context.global_seed
        n_cases = context.n_cases
        h, w = context.h, context.w
        cond_order = context.cond_order
        case_id_to_idx = {str(c["case_id"]): i for i, c in enumerate(context.dataset.cases)}
        geom_provider = context.geom_provider
        bundle = context.bundle
        y = context.y
        cond_scaled = context.cond_scaled
        y_scaled = context.y_scaled
        y_vars = [str(v) for v in list(bundle.schemas.get("output_layout", {}).get("vars", []))]
        eval_cfg = dict(self.benchmark_cfg.get("eval", {}))
        eval_diagnostics_enabled = bool(resolve_eval_diagnostics_cfg(eval_cfg).get("enabled", False))
        eval_plan = plan_builder.eval_protocol_plan(y_vars=y_vars, scopes=_EVAL_PROTOCOL_SCOPES)
        eval_protocol_mode = eval_plan.mode
        eval_protocol_scope = eval_plan.scope
        primary_split = eval_plan.primary_split
        interp_mode = eval_plan.interp_mode
        interp_weight = eval_plan.interp_weight
        extrap_weight = eval_plan.extrap_weight
        primary_metric = eval_plan.primary_metric
        primary_mode = eval_plan.primary_mode
        target_family_for_score_raw = eval_plan.target_family_for_score
        target_vars_for_score_effective = list(eval_plan.target_vars_for_score)
        region_bands_effective = dict(eval_plan.region_bands)
        evaluation_protocol = eval_plan.as_contract()
        self._validate_eval_scope_models(scope=eval_protocol_scope, model_names=profile_lock["models"])
        resolved["eval_protocol"] = eval_plan.as_resolved_payload()
        resolved["active_model_scope"] = eval_protocol_scope
        resolved.setdefault("eval", {})
        resolved["eval"]["target_family_for_score_effective"] = target_family_for_score_raw
        resolved["eval"]["target_vars_for_score_effective"] = list(target_vars_for_score_effective)
        resolved["target_vars_effective"] = list(y_vars)
        resolved["eval"]["region_bands_effective"] = dict(region_bands_effective)
        resolved["eval"]["primary_metric_effective"] = primary_metric
        resolved["eval"]["primary_mode_effective"] = primary_mode
        resolved["eval"]["objective_mode_effective"] = primary_mode
        resolved["outer_selection_contract"] = {
            "source": "training_validation_history",
            "value_key": VALIDATION_VALUE_KEY,
            "mode_key": "validation_selection_mode",
            "reliability_key": "validation_selection_reliable",
            "test_metrics_used_for_selection": False,
            "test_metrics_role": "final_reporting_only",
        }
        resolved["eval_protocol_contract"] = evaluation_protocol.as_dict()
        protocol_variant = str(eval_cfg.get("protocol_variant", "")).strip()
        resolved["eval"]["protocol_variant_effective"] = protocol_variant if protocol_variant else "default"
        self._persist_resolved(split=context.split, resolved=resolved, include_manifest=True)
        tr = context.tr
        va = context.va
        te = context.te
        transforms = context.transforms
        deeponet_index = context.deeponet_index
        deeponet_index_meta = context.deeponet_index_meta
        deeponet_poisson_index = context.deeponet_poisson_index
        deeponet_poisson_meta = context.deeponet_poisson_meta
        deeponet_boundary_index = context.deeponet_boundary_index
        deeponet_boundary_meta = context.deeponet_boundary_meta
        coord_feature_pack = context.coord_feature_pack
        split_interp = self._load_split_json(
            self.output_root / "preprocessing" / "split" / f"split_interp_{interp_mode}_v1.json"
        )
        preprocess_report = self._load_json_if_exists(
            self.output_root / "preprocessing" / "validation" / "report.json"
        )
        split_extrap = self._load_split_json(
            self.output_root / "preprocessing" / "split" / "split_extrap_v1.json"
        )
        split_structure_holdout = self._load_split_json(
            self.output_root / "preprocessing" / "split" / "split_structure_holdout_v1.json"
        )
        structure_holdout_meta = dict(bundle.schemas.get("structure_holdout_meta", {}) or {})
        if primary_split == "structure_holdout" and not bool(
            structure_holdout_meta.get("is_real_structure_holdout", False)
        ):
            reason = str(structure_holdout_meta.get("fallback_reason", "missing_structure_metadata"))
            raise ValueError(
                "Evaluation requested primary_split=structure_holdout, but preprocessing did not create "
                f"a real structure-group holdout (reason={reason}). Configure split.structure_holdout."
            )
        split_by_name = {
            "interp": split_interp,
            "extrap": split_extrap,
            "structure_holdout": split_structure_holdout,
        }
        cond_tuple_by_case = {
            str(c["case_id"]): tuple(float(c["cond"][k]) for k in cond_order)
            for c in context.dataset.cases
        }
        resolved["tuple_overlap_ratio"] = {
            "interp": self._tuple_overlap_ratio(split_interp, cond_tuple_by_case),
            "extrap": self._tuple_overlap_ratio(split_extrap, cond_tuple_by_case),
            "structure_holdout": self._tuple_overlap_ratio(split_structure_holdout, cond_tuple_by_case),
        }
        test_holdout_ratio = float(len(split_extrap.get("test", [])) / float(max(n_cases, 1)))
        structure_holdout_ratio = float(len(split_structure_holdout.get("test", [])) / float(max(n_cases, 1)))
        interp_overlap_ratio = float(resolved["tuple_overlap_ratio"]["interp"])
        extrapolation_severity = float(np.clip((1.0 - interp_overlap_ratio) * 0.7 + test_holdout_ratio * 0.3, 0.0, 1.0))
        resolved["eval_protocol"]["extrapolation_severity"] = {
            "score": extrapolation_severity,
            "tuple_overlap_interp": interp_overlap_ratio,
            "tuple_overlap_extrap": float(resolved["tuple_overlap_ratio"]["extrap"]),
            "holdout_ratio": test_holdout_ratio,
        }
        resolved["eval_protocol"]["structure_holdout_test_ratio"] = structure_holdout_ratio
        self._persist_resolved(split=context.split, resolved=resolved)

        if len(profile_lock["models"]) == 0:
            header = build_leaderboard_header(target_vars=target_vars_for_score_effective)
            leaderboard_path = save_leaderboard_metric_tables(
                store=self.store,
                leaderboard=[],
                full_header=header,
            )
            return BenchmarkResult(leaderboard=[], leaderboard_path=leaderboard_path)

        leaderboard: list[dict[str, Any]] = []
        geom_ctx = geom_provider.get()
        train_cfg = dict(self.benchmark_cfg.get("train", {}))
        self._validate_eval_scope_train_sections(scope=eval_protocol_scope, train_cfg=train_cfg)
        active_unet_model = resolve_single_family_model(model_names=profile_lock["models"], family=UNET_FAMILY_MODELS)
        active_spectral_model = resolve_single_family_model(
            model_names=profile_lock["models"],
            family=SPECTRAL_FAMILY_MODELS,
        )
        if active_unet_model is not None and target_family_for_score_raw == "allvars":
            unet_model_key = str(active_unet_model)
            cfg_prefix = f"train.{unet_model_key}"
            unet_cfg = dict(train_cfg.get(unet_model_key, {}))
            unet_family = str(unet_cfg.get("target_family", "allvars")).strip().lower()
            if unet_family != target_family_for_score_raw:
                raise ValueError(
                    f"{cfg_prefix}.target_family must match benchmark.eval.target_family_for_score for isolated runs: "
                    f"{cfg_prefix}.target_family={unet_family}, "
                    f"benchmark.eval.target_family_for_score={target_family_for_score_raw}"
                )
        if active_spectral_model is not None and target_family_for_score_raw == "allvars":
            spectral_model_key = str(active_spectral_model)
            cfg_prefix = f"train.{spectral_model_key}"
            spectral_cfg = dict(train_cfg.get(spectral_model_key, {}))
            spectral_family = str(spectral_cfg.get("target_family", "allvars")).strip().lower()
            if spectral_family != target_family_for_score_raw:
                raise ValueError(
                    f"{cfg_prefix}.target_family must match benchmark.eval.target_family_for_score for isolated runs: "
                    f"{cfg_prefix}.target_family={spectral_family}, "
                    f"benchmark.eval.target_family_for_score={target_family_for_score_raw}"
                )
        resolved["optimizer_contract"] = dict(train_cfg.get("optimizer_contract", {}))
        resolved["resolved_train_per_model"] = self._resolve_train_per_model(
            train_cfg=train_cfg,
            model_names=profile_lock["models"],
        )
        self._persist_resolved(split=context.split, resolved=resolved)
        loss_cfg = resolve_loss_protocol(
            dict(train_cfg.get("loss", {})),
            target_role_schema=bundle.schemas.get("target_role_schema", {}),
        )
        resolved.update(loss_protocol_metadata(loss_cfg))
        resolved["eval"]["boundary_distance_channels_effective"] = list(
            boundary_distance_channels_from_loss_cfg(loss_cfg)
        )
        curriculum_cfg = dict(train_cfg.get("curriculum", {}))
        quality_protocol = resolve_quality_score_protocol(
            dict(self.benchmark_cfg.get("eval", {}).get("quality_score", {}))
        )
        quality_score_cfg = dict(quality_protocol["effective_config"])
        resolved["eval"]["quality_score_effective"] = dict(quality_score_cfg)
        resolved["eval"]["quality_score_protocol"] = str(quality_protocol["protocol"])
        resolved["eval"]["quality_score_protocol_version"] = int(quality_protocol["version"])
        resolved["eval"]["quality_score_definition_hash"] = str(quality_protocol["definition_hash"])
        sample_mean_group_mode = str(loss_cfg.get("supervised", {}).get("sample_mean_group_mode", "batch"))
        if sample_mean_group_mode != "batch":
            raise ValueError("supervised.sample_mean_group_mode must be one of: batch")
        use_plasma_mask = str(loss_cfg.get("supervised", {}).get("mask", "none")).strip().lower() == "plasma_only"
        eval_mask_mode = str(self.benchmark_cfg.get("eval", {}).get("mask_metrics", "none")).strip().lower()
        metric_mask = np.asarray(geom_ctx.mask_plasma, dtype=np.float32) if eval_mask_mode == "plasma_only" else None
        supervised_mask = np.asarray(geom_ctx.mask_plasma, dtype=np.float32) if use_plasma_mask else None
        supervised_distance = np.asarray(geom_ctx.distance_any, dtype=np.float32) if use_plasma_mask else None
        physics_cfg = build_physics_cfg(
            raw_cfg=self.benchmark_cfg.get("physics", {}),
            geom_ctx=geom_ctx,
            default_enabled=False,
            default_primary_qoi_key=str(profile_lock.get("primary_qoi_key", "Gamma_i")),
        )
        physics_cfg["target_role_schema"] = bundle.schemas.get("target_role_schema", {})
        resolved["physics_resolved_terms"] = list(physics_cfg.get("resolved_terms", []))
        effective_steps_per_model: dict[str, Any] = {}
        contract_samples = BenchmarkContractSamples.empty()
        self._evaluate_guardrails(
            phase="pre",
            train_cfg=train_cfg,
            model_names=profile_lock["models"],
            effective_steps_per_model=None,
            target_vars_for_score_effective=target_vars_for_score_effective,
        )
        self._persist_resolved(split=context.split, resolved=resolved, include_manifest=True)
        single_split_common = {
            "context": context,
            "profile_lock": profile_lock,
            "train_cfg": train_cfg,
            "loss_cfg": loss_cfg,
            "curriculum_cfg": curriculum_cfg,
            "physics_cfg": physics_cfg,
            "geom_ctx": geom_ctx,
            "quality_score_cfg": quality_score_cfg,
            "metric_mask": metric_mask,
            "supervised_mask": supervised_mask,
            "supervised_distance": supervised_distance,
            "target_vars_for_score": target_vars_for_score_effective,
            "region_band_cfg": region_bands_effective,
            "primary_metric": primary_metric,
            "preprocess_report": preprocess_report,
            "expected_lock_hash": str(resolved["artifact_hashes"]["lock_hash"]),
        }

        for model_idx, model_name in enumerate(profile_lock["models"]):
            model_dir = self.output_root / "models" / model_name
            if eval_protocol_mode == "single":
                out = self._run_single_split_model(
                    **single_split_common,
                    model_name=model_name,
                    model_idx=model_idx,
                    model_dir=model_dir,
                    tr_idx=tr,
                    va_idx=va,
                    te_idx=te,
                    scaler_split_name="random",
                )
                leaderboard.append(out.row)
                effective_steps_per_model[model_name] = int(out.effective_steps)
                contract_samples.add(out)
            elif eval_protocol_mode == "primary_axis":
                split_def = split_by_name[primary_split]
                tr_i, va_i, te_i = self._indices_from_split(split_def, case_id_to_idx)
                split_offset = {"interp": 1, "extrap": 2, "structure_holdout": 3}[primary_split]
                out = self._run_single_split_model(
                    **single_split_common,
                    model_name=model_name,
                    model_idx=(model_idx * 10 + split_offset),
                    model_dir=model_dir / "eval_protocol" / primary_split,
                    tr_idx=tr_i,
                    va_idx=va_i,
                    te_idx=te_i,
                    scaler_split_name=primary_split,
                )
                row = dict(out.row)
                _attach_primary_metric_status(
                    row,
                    primary_metric=primary_metric,
                    model_name=model_name,
                    target_vars=target_vars_for_score_effective,
                )
                leaderboard.append(row)
                effective_steps_per_model[model_name] = int(out.effective_steps)
                contract_samples.add(out)
            else:
                split_rows: dict[str, dict[str, Any]] = {}
                split_steps: dict[str, int] = {}
                for split_name, split_def in [("interp", split_interp), ("extrap", split_extrap)]:
                    tr_i, va_i, te_i = self._indices_from_split(split_def, case_id_to_idx)
                    split_out = self._run_single_split_model(
                        **single_split_common,
                        model_name=model_name,
                        model_idx=(model_idx * 10 + (1 if split_name == "interp" else 2)),
                        model_dir=model_dir / "eval_protocol" / split_name,
                        tr_idx=tr_i,
                        va_idx=va_i,
                        te_idx=te_i,
                        scaler_split_name=split_name,
                    )
                    split_rows[split_name] = split_out.row
                    split_steps[split_name] = int(split_out.effective_steps)
                    contract_samples.add(split_out)
                    _release_torch_cuda_cache()
                row = self._combine_dual_axis_rows(
                    split_rows=split_rows,
                    primary_split=primary_split,
                    interp_weight=interp_weight,
                    extrap_weight=extrap_weight,
                    target_vars=target_vars_for_score_effective,
                )
                _attach_primary_metric_status(
                    row,
                    primary_metric=primary_metric,
                    model_name=model_name,
                    target_vars=target_vars_for_score_effective,
                )
                leaderboard.append(row)
                effective_steps_per_model[model_name] = {
                    "interp": int(split_steps.get("interp", 0)),
                    "extrap": int(split_steps.get("extrap", 0)),
                }

        cv_cfg = dict(self.benchmark_cfg.get("cv", {}))
        if bool(cv_cfg.get("enabled", False)):
            case_ids = [str(c["case_id"]) for c in context.dataset.cases]
            group_key = str(cv_cfg.get("group_key", "base_case_id"))
            split_groups = [str(c.get(group_key, c["case_id"])) for c in context.dataset.cases]
            fold_splits = build_group_kfold_splits(
                case_ids=case_ids,
                split_groups=split_groups,
                n_folds=int(cv_cfg.get("n_folds", 5)),
                seed=int(cv_cfg.get("seed", global_seed)),
            )
            id_to_idx = {cid: i for i, cid in enumerate(case_ids)}
            cv_rows: dict[str, list[dict[str, float]]] = {m: [] for m in profile_lock["models"]}
            for fold_idx, split_fold in enumerate(fold_splits):
                tr_fold = np.array([id_to_idx[c] for c in split_fold["train"]], dtype=np.int64)
                va_fold = np.array([id_to_idx[c] for c in split_fold["val"]], dtype=np.int64)
                te_fold = np.array([id_to_idx[c] for c in split_fold["test"]], dtype=np.int64)
                for model_idx, model_name in enumerate(profile_lock["models"]):
                    fold_model_dir = self.output_root / "models" / model_name / "cv_folds" / f"fold_{fold_idx:02d}"
                    effective_input_mode_meta = resolve_effective_input_mode_metadata_for_model(
                        model_name=model_name,
                        input_mode_meta=self.input_mode_meta,
                    )
                    dispatch = run_model_train_eval(
                        BenchmarkModelContext(
                            benchmark_cfg=self.benchmark_cfg,
                            profile_lock=profile_lock,
                            model_idx=model_idx + 1000 + fold_idx,
                            model_name=model_name,
                            model_dir=fold_model_dir,
                            global_seed=global_seed + 777,
                            n_cases=n_cases,
                            h=h,
                            w=w,
                            y_vars=y_vars,
                            cond_scaled=cond_scaled,
                            y=y,
                            y_scaled=y_scaled,
                            tr=tr_fold,
                            va=va_fold,
                            te=te_fold,
                            transforms=transforms,
                            physics_cfg=physics_cfg,
                            loss_cfg=loss_cfg,
                            curriculum_cfg=curriculum_cfg,
                            supervised_mask=supervised_mask,
                            supervised_distance=supervised_distance,
                            geom_ctx=geom_ctx,
                            deeponet_index=deeponet_index,
                            deeponet_index_meta=deeponet_index_meta,
                            deeponet_poisson_index=deeponet_poisson_index,
                            deeponet_poisson_meta=deeponet_poisson_meta,
                            deeponet_boundary_index=deeponet_boundary_index,
                            deeponet_boundary_meta=deeponet_boundary_meta,
                            coord_feature_scaler=dict(bundle.transforms.get("coord_feature_scaler", {})),
                            coord_feature_pack=coord_feature_pack,
                            static_spatial_feature_pack=context.static_spatial_feature_pack,
                            case_structure_feature_pack=context.case_structure_feature_pack,
                            coord_distance_transform_stats=dict(bundle.transforms.get("distance_transform_stats", {})),
                            structure_descriptor_pack=bundle.schemas.get("structure_descriptor_pack"),
                            latent_feature_pack=bundle.schemas.get("latent_feature_pack"),
                            case_ids=case_ids,
                            input_mode_meta=effective_input_mode_meta,
                        )
                    )
                    pred_eval = dispatch["pred_eval"]
                    true_eval = dispatch["true_eval"]
                    fold_geometry = materialize_evaluation_spatial_geometry(
                        context=context,
                        geom_ctx=geom_ctx,
                        eval_indices=te_fold,
                        metric_mask_enabled=metric_mask is not None,
                        expected_eval_count=_evaluation_case_count(
                            pred_eval=pred_eval,
                            true_eval=true_eval,
                        ),
                        boundary_distance_channels=boundary_distance_channels_from_loss_cfg(loss_cfg),
                    )
                    fold_eval_row = build_benchmark_eval_row(
                        model_id=f"{model_name}_cv_fold_{fold_idx:02d}",
                        metrics=dict(dispatch["metrics"]),
                        r2_scores=dict(dispatch.get("r2_scores", {})),
                        pred_eval=pred_eval,
                        true_eval=true_eval,
                        mask_plasma=fold_geometry.metric_mask,
                        region_mask_plasma=fold_geometry.mask_plasma,
                        distance_any=fold_geometry.distance_any,
                        distance_signed=fold_geometry.distance_signed,
                        bc_dir_mask=geom_ctx.bc_dir_mask if geom_ctx is not None else None,
                        wafer_mask=(
                            np.asarray(geom_ctx.regions.get("wafer_mask"), dtype=np.float32)
                            if (
                                geom_ctx is not None
                                and getattr(geom_ctx, "regions", {}).get("wafer_mask") is not None
                            )
                            else None
                        ),
                        target_vars_for_score=target_vars_for_score_effective,
                        region_band_cfg=region_bands_effective,
                        quality_score_cfg=quality_score_cfg,
                        target_role_schema=bundle.schemas.get("target_role_schema", {}),
                        target_scalers=bundle.transforms.get("y_scalers", {}),
                        target_transforms=bundle.transforms.get("target_transforms", {}),
                        output_vars=y_vars,
                        single_diagnostics={},
                        extended_diagnostics_enabled=eval_diagnostics_enabled,
                    )
                    fold_eval_row["evaluation_geometry_source"] = fold_geometry.source
                    fold_eval_row["evaluation_boundary_distance_channels"] = list(
                        fold_geometry.boundary_distance_channels
                    )
                    row = {
                        str(key): float(value)
                        for key, value in fold_eval_row.items()
                        if isinstance(value, (int, float))
                        and (
                            str(key).startswith("test_rmse_")
                            or str(key).startswith("test_r2_")
                            or str(key).startswith("positive_violation_rate_")
                            or str(key).startswith("negative_min_")
                        )
                    }
                    cv_rows[model_name].append(row)
            CvBenchmarkRunner.attach_summary(leaderboard=leaderboard, cv_rows=cv_rows)
            resolved["cv"] = {
                "enabled": True,
                "n_folds": int(cv_cfg.get("n_folds", 5)),
                "group_key": group_key,
                "seed": int(cv_cfg.get("seed", global_seed)),
            }
            self._persist_resolved(split=context.split, resolved=resolved)

        leaderboard_path = self._finalize_benchmark_outputs(
            leaderboard=leaderboard,
            resolved=resolved,
            train_cfg=train_cfg,
            model_names=profile_lock["models"],
            effective_steps_per_model=effective_steps_per_model,
            target_vars_for_score=target_vars_for_score_effective,
            target_family_for_score=target_family_for_score_raw,
            y_vars=y_vars,
            active_unet_model=active_unet_model,
            eval_protocol_scope=eval_protocol_scope,
            contract_samples=contract_samples,
            region_bands_effective=region_bands_effective,
            split=context.split,
        )
        return BenchmarkResult(leaderboard=leaderboard, leaderboard_path=leaderboard_path)

    def _finalize_benchmark_outputs(
        self,
        *,
        leaderboard: list[dict[str, Any]],
        resolved: dict[str, Any],
        train_cfg: dict[str, Any],
        model_names: list[str],
        effective_steps_per_model: dict[str, Any],
        target_vars_for_score: list[str],
        target_family_for_score: str,
        y_vars: list[str],
        active_unet_model: str | None,
        eval_protocol_scope: str,
        contract_samples: BenchmarkContractSamples,
        region_bands_effective: dict[str, Any],
        split: dict[str, Any],
    ) -> Path:
        full_header = build_leaderboard_header(
            target_vars=target_vars_for_score,
            leaderboard=leaderboard,
        )
        self._evaluate_guardrails(
            phase="post",
            train_cfg=train_cfg,
            model_names=model_names,
            effective_steps_per_model=effective_steps_per_model,
        )
        resolved["effective_steps_per_model"] = effective_steps_per_model
        resolved["comparison_contract"] = {
            "global_reference_mode": "frozen"
            if eval_protocol_scope in _FROZEN_REFERENCE_SCOPES
            else "common",
            "active_model_scope": eval_protocol_scope,
            "target_family_mode": target_family_for_score,
        }
        self.contract_emitter.emit(
            resolved=resolved,
            train_cfg=train_cfg,
            y_vars=y_vars,
            active_unet_model=active_unet_model,
            eval_protocol_scope=eval_protocol_scope,
            contract_samples=contract_samples,
        )
        diagnostics_effective = resolve_eval_diagnostics_cfg(dict(self.benchmark_cfg.get("eval", {}) or {}))
        resolved["diagnostics_effective"] = diagnostics_effective
        resolved["spatial_error_audit_effective"] = bool(
            dict(diagnostics_effective.get("spatial", {})).get("enabled", False)
        )
        resolved["spatial_error_boundary_type_breakdown_effective"] = bool(
            dict(diagnostics_effective.get("spatial", {})).get("boundary_type_breakdown", False)
        )
        resolved["spatial_error_region_bands_effective"] = dict(region_bands_effective)
        self._persist_resolved(split=split, resolved=resolved, include_manifest=True)
        return save_leaderboard_metric_tables(
            store=self.store,
            leaderboard=leaderboard,
            full_header=full_header,
        )

    def _write_model_runtime_artifacts(
        self,
        *,
        model: Any,
        model_dir: Path,
        history: list[dict[str, Any]],
        effective_input_mode_meta: dict[str, Any],
        extra_artifacts: dict[str, Any],
        scaler_fit_split: str,
    ) -> VizRunner:
        checkpoint_meta = merge_checkpoint_dispatch_metadata(
            runtime_meta=effective_input_mode_meta,
            dispatch_meta=extra_artifacts,
        )
        checkpoint_meta["scaler_fit_split"] = str(scaler_fit_split)
        checkpoint_meta["scaler_train_only"] = True
        save_checkpoint(model, model_dir / "checkpoints", extra_meta=checkpoint_meta)

        viz = VizRunner(model_dir / "eval")
        eval_plot_cfg = dict(dict(self.benchmark_cfg.get("eval", {}) or {}).get("plots", {}) or {})
        viz_cfg = dict(self.benchmark_cfg.get("viz", {}) or {})
        viz.plot_loss_curve(
            history,
            rel_path="plots/loss_curve.png",
            yscale=str(viz_cfg.get("loss_yscale", eval_plot_cfg.get("loss_yscale", "linear"))),
        )
        return viz

    def _run_single_split_model(
        self,
        *,
        context: Any,
        profile_lock: dict[str, Any],
        train_cfg: dict[str, Any],
        loss_cfg: dict[str, Any],
        curriculum_cfg: dict[str, Any],
        physics_cfg: dict[str, Any],
        geom_ctx: Any,
        quality_score_cfg: dict[str, Any],
        metric_mask: np.ndarray | None,
        supervised_mask: np.ndarray | None,
        supervised_distance: np.ndarray | None,
        target_vars_for_score: list[str],
        region_band_cfg: dict[str, Any],
        primary_metric: str,
        preprocess_report: dict[str, Any],
        expected_lock_hash: str,
        model_name: str,
        model_idx: int,
        model_dir: Path,
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        te_idx: np.ndarray,
        scaler_split_name: str,
    ) -> SingleSplitResult:
        bundle = context.bundle
        y_vars = [str(v) for v in list(bundle.schemas.get("output_layout", {}).get("vars", []))]
        scaler_split_name = str(scaler_split_name).strip().lower()
        lane_transforms = bundle.transform_bundle(
            scaler_split_name,
            require_protocol=scaler_split_name in {"interp", "extrap", "structure_holdout"},
        )
        lane_cond_scaled = lane_transforms.transform_cond(context.cond)
        lane_y_scaled = lane_transforms.transform_fields(context.y)
        lane_artifacts = dict(
            dict(bundle.transforms.get("protocol_transforms", {}) or {}).get(scaler_split_name, {}) or {}
        )
        lane_coord_feature_scaler = dict(
            lane_artifacts.get("coord_feature_scaler", bundle.transforms.get("coord_feature_scaler", {})) or {}
        )
        lane_distance_transform_stats = dict(
            lane_artifacts.get(
                "distance_transform_stats",
                bundle.transforms.get("distance_transform_stats", {}),
            )
            or {}
        )
        if scaler_split_name in {"interp", "extrap", "structure_holdout"}:
            if not lane_coord_feature_scaler or not lane_distance_transform_stats:
                raise FileNotFoundError(
                    "Missing train-only spatial transform artifacts for evaluation lane "
                    f"{scaler_split_name!r} under preprocessing/scalers/by_split"
                )
        effective_input_mode_meta = resolve_effective_input_mode_metadata_for_model(
            model_name=model_name,
            input_mode_meta=self.input_mode_meta,
        )
        dispatch = run_model_train_eval(
            BenchmarkModelContext(
                benchmark_cfg=self.benchmark_cfg,
                profile_lock=profile_lock,
                model_idx=model_idx,
                model_name=model_name,
                model_dir=model_dir,
                global_seed=context.global_seed,
                n_cases=context.n_cases,
                h=context.h,
                w=context.w,
                y_vars=y_vars,
                cond_scaled=lane_cond_scaled,
                y=context.y,
                y_scaled=lane_y_scaled,
                tr=tr_idx,
                va=va_idx,
                te=te_idx,
                transforms=lane_transforms,
                physics_cfg=physics_cfg,
                loss_cfg=loss_cfg,
                curriculum_cfg=curriculum_cfg,
                supervised_mask=supervised_mask,
                supervised_distance=supervised_distance,
                geom_ctx=geom_ctx,
                deeponet_index=context.deeponet_index,
                deeponet_index_meta=context.deeponet_index_meta,
                deeponet_poisson_index=context.deeponet_poisson_index,
                deeponet_poisson_meta=context.deeponet_poisson_meta,
                deeponet_boundary_index=context.deeponet_boundary_index,
                deeponet_boundary_meta=context.deeponet_boundary_meta,
                coord_feature_scaler=lane_coord_feature_scaler,
                coord_feature_pack=context.coord_feature_pack,
                static_spatial_feature_pack=context.static_spatial_feature_pack,
                case_structure_feature_pack=context.case_structure_feature_pack,
                coord_distance_transform_stats=lane_distance_transform_stats,
                structure_descriptor_pack=bundle.schemas.get("structure_descriptor_pack"),
                latent_feature_pack=bundle.schemas.get("latent_feature_pack"),
                case_ids=[str(case["case_id"]) for case in context.dataset.cases],
                input_mode_meta=effective_input_mode_meta,
            )
        )
        model = dispatch["model"]
        history = dispatch["history"]
        pred_eval = dispatch["pred_eval"]
        true_eval = dispatch["true_eval"]
        metrics = dispatch["metrics"]
        r2_scores = dispatch.get("r2_scores", {})
        extra_artifacts = dict(dispatch.get("extra_artifacts", {}))
        eval_geometry = materialize_evaluation_spatial_geometry(
            context=context,
            geom_ctx=geom_ctx,
            eval_indices=te_idx,
            metric_mask_enabled=metric_mask is not None,
            expected_eval_count=_evaluation_case_count(
                pred_eval=pred_eval,
                true_eval=true_eval,
            ),
            boundary_distance_channels=boundary_distance_channels_from_loss_cfg(loss_cfg),
        )

        viz = self._write_model_runtime_artifacts(
            model=model,
            model_dir=model_dir,
            history=history,
            effective_input_mode_meta=effective_input_mode_meta,
            extra_artifacts=extra_artifacts,
            scaler_fit_split=scaler_split_name,
        )
        inference_ood_cfg = BenchmarkPlanBuilder(self.benchmark_cfg).inference_ood_cfg()
        potential_key = _resolve_benchmark_potential_key(
            field_keys=sorted(set(str(key) for key in true_eval.keys()) & set(str(key) for key in pred_eval.keys())),
            target_role_schema=bundle.schemas.get("target_role_schema", {}),
            ood_cfg=inference_ood_cfg,
            context="benchmark summary",
        )
        probe = BenchmarkProbe(self.benchmark_cfg).run(
            model=model,
            model_name=model_name,
            model_idx=model_idx,
            model_dir=model_dir,
            context=context,
            profile_lock=profile_lock,
            train_cfg=train_cfg,
            effective_input_mode_meta=effective_input_mode_meta,
            true_eval=true_eval,
            pred_eval=pred_eval,
            metric_mask=eval_geometry.metric_mask,
            te_idx=te_idx,
            viz=viz,
            transform_bundle=lane_transforms,
            coord_feature_scaler=lane_coord_feature_scaler,
            coord_distance_transform_stats=lane_distance_transform_stats,
        )

        lock_hash = str(context.lock_hash)
        summary = {
            "model_id": model_name,
            "rmse": metrics,
            "poisson_potential_rmse": (
                float(poisson_residual_loss(pred_eval[potential_key][:, 0])) if potential_key is not None else 0.0
            ),
            "single_qoi": probe.qoi,
            "single_diagnostics": probe.diagnostics,
            "batch_count": len(probe.batch_rows),
            "optimize": probe.optimize.as_summary(),
            "lock_hash": lock_hash,
        }
        ArtifactStore(model_dir).save_json("summary.json", summary)
        if summary["lock_hash"] != str(expected_lock_hash):
            raise ValueError("benchmark lock hash mismatch")

        eval_cfg = dict(self.benchmark_cfg.get("eval", {}) or {})
        eval_diagnostics_enabled = bool(resolve_eval_diagnostics_cfg(eval_cfg).get("enabled", False))
        row = build_benchmark_eval_row(
            model_id=model_name,
            metrics=metrics,
            r2_scores=r2_scores,
            pred_eval=pred_eval,
            true_eval=true_eval,
            mask_plasma=eval_geometry.metric_mask,
            region_mask_plasma=eval_geometry.mask_plasma,
            distance_any=eval_geometry.distance_any,
            distance_signed=eval_geometry.distance_signed,
            bc_dir_mask=geom_ctx.bc_dir_mask if geom_ctx is not None else None,
            wafer_mask=(
                np.asarray(geom_ctx.regions.get("wafer_mask"), dtype=np.float32)
                if (geom_ctx is not None and getattr(geom_ctx, "regions", {}).get("wafer_mask") is not None)
                else None
            ),
            target_vars_for_score=target_vars_for_score,
            region_band_cfg=region_band_cfg,
            quality_score_cfg=quality_score_cfg,
            target_role_schema=bundle.schemas.get("target_role_schema", {}),
            target_scalers=lane_transforms.to_dict().get("y_scalers", {}),
            target_transforms=lane_transforms.to_dict().get("target_transforms", {}),
            output_vars=y_vars,
            single_diagnostics=probe.diagnostics,
            extended_diagnostics_enabled=eval_diagnostics_enabled,
        )
        row["evaluation_geometry_source"] = eval_geometry.source
        row["evaluation_geometry_case_aligned"] = True
        row["evaluation_boundary_distance_channels"] = list(
            eval_geometry.boundary_distance_channels
        )
        _inject_input_mode_metadata_into_row(
            row=row,
            input_mode_meta=effective_input_mode_meta,
            case_spatial_pack_used=bool(context.case_structure_feature_pack),
        )
        row["scaler_fit_split"] = scaler_split_name
        row["scaler_train_only"] = True
        row["selection_split"] = scaler_split_name
        row["protocol_variant"] = str(
            dict(self.benchmark_cfg.get("eval", {}) or {}).get("protocol_variant", "default")
        ).strip() or "default"
        row.update(validation_selection_from_history(history))
        if eval_diagnostics_enabled:
            write_eval_diagnostics(
                eval_cfg=eval_cfg,
                model_dir=model_dir,
                row=row,
                context=context,
                geom_ctx=geom_ctx,
                viz=viz,
                pred_eval=pred_eval,
                true_eval=true_eval,
                metric_mask=eval_geometry.metric_mask,
                distance_any=eval_geometry.distance_any,
                distance_signed=eval_geometry.distance_signed,
                te_idx=te_idx,
                target_vars_for_score=target_vars_for_score,
                region_band_cfg=region_band_cfg,
                target_transforms=lane_transforms.to_dict().get("target_transforms", {}),
                target_scalers=lane_transforms.to_dict().get("y_scalers", {}),
            )
        _attach_primary_metric_status(
            row,
            primary_metric=primary_metric,
            model_name=model_name,
            target_vars=target_vars_for_score,
        )
        return SingleSplitResult(
            row=row,
            effective_steps=int(extra_artifacts.get("effective_steps", max(len(history), 0))),
            contract_artifacts={
                str(name): dict(contract)
                for name, contract in dict(extra_artifacts.get("model_contracts", {}) or {}).items()
                if isinstance(contract, dict)
            },
        )

    def run_sweep(self) -> SweepResult:
        sweep_cfg = self.benchmark_cfg.get("sweep", {})
        params_grid = sweep_cfg.get("params", {})
        if not isinstance(params_grid, dict) or not params_grid:
            raise ValueError("benchmark.sweep.params must be a non-empty dict")

        objective_cfg = sweep_cfg.get("objective", {})
        objective_model = str(objective_cfg.get("model_id", "global_mlp"))
        objective_metric = str(objective_cfg.get("metric", "auto_primary")).strip()
        if objective_metric not in {"auto_primary", VALIDATION_VALUE_KEY}:
            raise ValueError(
                "Outer benchmark sweep selection is validation-only; objective.metric must be "
                f"auto_primary or {VALIDATION_VALUE_KEY}. Test metrics are reporting-only."
            )
        objective_mode_requested = str(objective_cfg.get("mode", "auto")).strip().lower()
        if objective_mode_requested not in {"auto", "min", "max"}:
            raise ValueError("benchmark.sweep.objective.mode must be one of: auto, min, max")

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.store.save_json(
            "sweep/resolved_sweep.json",
            {
                "params": params_grid,
                "objective": {
                    "model_id": objective_model,
                    "metric": objective_metric,
                    "mode": objective_mode_requested,
                },
            },
        )

        keys = list(params_grid.keys())
        values = [params_grid[k] for k in keys]
        if any(not isinstance(v, list) or not v for v in values):
            raise ValueError("Each benchmark.sweep.params entry must be a non-empty list")

        trials: list[dict[str, Any]] = []
        objective_mode_effective: str | None = None
        for trial_idx, combo in enumerate(itertools.product(*values)):
            trial_values = {k: combo[i] for i, k in enumerate(keys)}
            trial_output = self.output_root / "sweep" / "trials" / f"trial_{trial_idx:03d}"
            trial_cfg = copy.deepcopy(self.benchmark_cfg)
            trial_cfg["output_dir"] = str(trial_output)
            trial_cfg.pop("sweep", None)
            for path_key, value in trial_values.items():
                self._set_nested_dict_value(trial_cfg, path_key, value)

            trial_runner = BenchmarkRunner({"benchmark": trial_cfg})
            trial_result = trial_runner.run()
            trial_row = self._select_trial_row(
                trial_result.leaderboard,
                model_id=objective_model,
                metric=objective_metric,
            )
            score_key = VALIDATION_VALUE_KEY
            raw_score, row_objective_mode = require_validation_objective(trial_row)
            if objective_mode_requested != "auto" and objective_mode_requested != row_objective_mode:
                raise ValueError(
                    "benchmark.sweep.objective.mode conflicts with the training validation selector: "
                    f"configured={objective_mode_requested}, validation={row_objective_mode}"
                )
            if objective_mode_effective is None:
                objective_mode_effective = row_objective_mode
            elif objective_mode_effective != row_objective_mode:
                raise ValueError("Benchmark sweep trials resolved inconsistent validation objective directions")
            score_reliable = True
            score = raw_score
            manifest = ArtifactStore(trial_output).load_json("manifest.json")
            artifact_hashes = dict(manifest.get("artifacts", {}).get("artifact_hashes", {}))
            trials.append(
                {
                    "trial_id": f"trial_{trial_idx:03d}",
                    "output_dir": str(trial_output),
                    "objective_model_id": objective_model,
                    "objective_metric": score_key,
                    "objective_score": score,
                    "objective_score_raw": raw_score,
                    "objective_reliable": score_reliable,
                    "objective_invalid_reason": "" if score_reliable else "unreliable_primary_metric",
                    "lock_hash": artifact_hashes["lock_hash"],
                    **trial_values,
                }
            )

        if objective_mode_effective is None:
            raise ValueError("benchmark sweep produced no validation objectives")
        reverse = objective_mode_effective == "max"
        reliable_trials = [row for row in trials if bool(row.get("objective_reliable", False))]
        if not reliable_trials:
            self.store.save_csv(
                "sweep/summary.csv",
                list(trials[0].keys()) if trials else [],
                [[row[h] for h in trials[0].keys()] for row in trials] if trials else [],
            )
            raise ValueError("benchmark sweep found no reliable trial rows for best selection")
        trials_sorted = sorted(trials, key=lambda r: float(r["objective_score"]), reverse=reverse)
        reliable_sorted = sorted(reliable_trials, key=lambda r: float(r["objective_score"]), reverse=reverse)
        best = reliable_sorted[0]
        best_params = {k: best[k] for k in keys}
        locked_cfg = copy.deepcopy(self.benchmark_cfg)
        locked_cfg.pop("sweep", None)
        locked_cfg["output_dir"] = str(self.output_root / "sweep" / "best_locked_run")
        for path_key, value in best_params.items():
            self._set_nested_dict_value(locked_cfg, path_key, value)
        locked_cfg_path = self.output_root / "sweep" / "locked_best_config.yaml"
        locked_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with locked_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"benchmark": locked_cfg}, f, sort_keys=True)
        locked_run = BenchmarkRunner({"benchmark": locked_cfg}).run()

        header = list(trials_sorted[0].keys())
        self.store.save_csv("sweep/summary.csv", header, [[row[h] for h in header] for row in trials_sorted])
        self.store.save_json("sweep/best_trial.json", best)
        self.store.save_json(
            "sweep/locked_best_result.json",
            {
                "locked_config": str(locked_cfg_path),
                "locked_leaderboard": str(locked_run.leaderboard_path),
                "best_trial": best,
            },
        )
        return SweepResult(
            summary_path=self.output_root / "sweep" / "summary.csv",
            best_trial=best,
            locked_config_path=locked_cfg_path,
            locked_leaderboard_path=locked_run.leaderboard_path,
        )

    @staticmethod
    def _resolve_profile_lock(profile: str) -> dict[str, Any]:
        return resolve_profile_lock(profile)

    @staticmethod
    def _set_nested_dict_value(root: dict[str, Any], dotted_key: str, value: Any) -> None:
        keys = str(dotted_key).split(".")
        ref = root
        for key in keys[:-1]:
            if key not in ref or not isinstance(ref[key], dict):
                ref[key] = {}
            ref = ref[key]
        ref[keys[-1]] = value

    @staticmethod
    def _select_trial_row(leaderboard: list[dict[str, Any]], model_id: str, metric: str) -> dict[str, Any]:
        for row in leaderboard:
            if str(row.get("model_id")) == model_id:
                if metric == "auto_primary":
                    return row
                if metric not in row:
                    raise KeyError(f"Metric '{metric}' not found in leaderboard row for model '{model_id}'")
                return row
        raise KeyError(f"Model '{model_id}' not found in leaderboard")

    @staticmethod
    def _load_split_json(path: Path) -> dict[str, list[str]]:
        if not path.exists():
            raise FileNotFoundError(f"Missing split artifact: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return {
            "train": [str(v) for v in raw.get("train", [])],
            "val": [str(v) for v in raw.get("val", [])],
            "test": [str(v) for v in raw.get("test", [])],
        }

    @staticmethod
    def _load_json_if_exists(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _indices_from_split(split: dict[str, list[str]], id_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def _to_idx(ids: list[str]) -> np.ndarray:
            idx: list[int] = []
            for cid in ids:
                if cid not in id_to_idx:
                    raise ValueError(f"split references unknown case_id: {cid}")
                idx.append(id_to_idx[cid])
            return np.asarray(idx, dtype=np.int64)

        tr = _to_idx(split.get("train", []))
        va = _to_idx(split.get("val", []))
        te = _to_idx(split.get("test", []))
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            raise ValueError("dual-axis split contains empty train/val/test")
        return tr, va, te

    @staticmethod
    def _tuple_overlap_ratio(split: dict[str, list[str]], cond_tuple_by_case: dict[str, tuple[float, ...]]) -> float:
        train_ids = [str(v) for v in split.get("train", [])]
        test_ids = [str(v) for v in split.get("test", [])]
        train_tuples = {cond_tuple_by_case[cid] for cid in train_ids if cid in cond_tuple_by_case}
        test_tuples = {cond_tuple_by_case[cid] for cid in test_ids if cid in cond_tuple_by_case}
        if len(test_tuples) == 0:
            return 0.0
        return float(len(train_tuples & test_tuples) / float(len(test_tuples)))

    @staticmethod
    def _r2_plasma_mean(row: dict[str, Any], *, target_vars: list[str] | None = None) -> float:
        value, _, _ = BenchmarkRunner._r2_plasma_mean_status(row, target_vars=target_vars)
        return float(value)

    @staticmethod
    def _dual_aggregatable_metric_key(key: str) -> bool:
        """Return whether a numeric lane value has a meaningful weighted dual value."""

        name = str(key)
        return name == "surrogate_quality_score" or name.startswith(
            ("test_rmse_", "test_r2_", "positive_violation_rate_")
        )

    @staticmethod
    def _combine_dual_axis_rows(
        *,
        split_rows: dict[str, dict[str, Any]],
        primary_split: str,
        interp_weight: float,
        extrap_weight: float,
        target_vars: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a top-level dual-axis leaderboard row from split rows.

        The split training/evaluation jobs already produce normal leaderboard rows.  The
        top-level dual row should therefore carry the weighted aggregate metrics while
        preserving split-specific diagnostics for reporting/debugging.
        """

        if primary_split not in split_rows:
            raise KeyError(f"dual-axis primary split row is missing: {primary_split!r}")
        row: dict[str, Any] = dict(split_rows[primary_split])
        weights = {"interp": float(interp_weight), "extrap": float(extrap_weight)}
        total_weight = sum(max(0.0, w) for w in weights.values())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            weights = {"interp": 0.5, "extrap": 0.5}

        numeric_metric_keys: set[str] = set()
        combined_diagnostics: dict[str, Any] = {}
        row.pop("_diagnostics", None)
        for split_name, split_row in split_rows.items():
            split_diagnostics = split_row.get("_diagnostics")
            if isinstance(split_diagnostics, dict):
                for diag_key, diag_value in split_diagnostics.items():
                    combined_diagnostics[f"{diag_key}_{split_name}"] = diag_value
            for key, value in split_row.items():
                if key == "_diagnostics":
                    continue
                if isinstance(value, (bool, np.bool_)):
                    row[f"{key}_{split_name}"] = bool(value)
                    continue
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    row[f"{key}_{split_name}"] = value
                    continue
                if np.isfinite(value_f):
                    if BenchmarkRunner._dual_aggregatable_metric_key(str(key)):
                        numeric_metric_keys.add(str(key))
                    row[f"{key}_{split_name}"] = value_f
                else:
                    row[f"{key}_{split_name}"] = value

        required_lanes = ("interp", "extrap")
        for key in sorted(numeric_metric_keys):
            weighted_values: list[tuple[float, float]] = []
            valid = True
            for split_name in required_lanes:
                split_row = split_rows.get(split_name, {})
                try:
                    value_f = float(split_row.get(key, float("nan")))
                except (TypeError, ValueError):
                    valid = False
                    break
                weight = max(0.0, float(weights.get(split_name, 0.0)))
                if not np.isfinite(value_f) or not np.isfinite(weight):
                    valid = False
                    break
                if weight > 0.0:
                    weighted_values.append((value_f, weight))
            if not valid or not weighted_values:
                row[key] = float("nan")
                continue
            denom = float(sum(weight for _, weight in weighted_values))
            if denom <= 0.0:
                row[key] = float("nan")
                continue
            row[key] = float(sum(value * weight for value, weight in weighted_values) / denom)

        lane_target_valid = all(
            bool(split_rows.get(split_name, {}).get("target_metrics_valid", False))
            for split_name in required_lanes
        )
        quality_hashes = {
            (
                ""
                if split_rows.get(split_name, {}).get("quality_score_definition_hash") is None
                else str(
                    split_rows.get(split_name, {}).get("quality_score_definition_hash", "")
                ).strip()
            )
            for split_name in required_lanes
        }
        quality_values_finite = True
        for split_name in required_lanes:
            try:
                quality_value = float(
                    split_rows.get(split_name, {}).get("surrogate_quality_score", float("nan"))
                )
            except (TypeError, ValueError):
                quality_values_finite = False
                break
            if not np.isfinite(quality_value):
                quality_values_finite = False
                break
        quality_dual_reliable = bool(
            lane_target_valid
            and len(quality_hashes) == 1
            and "" not in quality_hashes
            and quality_values_finite
        )
        row["target_metrics_valid"] = bool(lane_target_valid)
        row["quality_score_dual_reliable"] = quality_dual_reliable
        row["quality_score_dual_invalid_reason"] = (
            "" if quality_dual_reliable else "invalid_lane_metrics_or_quality_contract_mismatch"
        )
        if not quality_dual_reliable:
            row["surrogate_quality_score"] = float("nan")
        row["surrogate_quality_score_dual"] = float(
            row.get("surrogate_quality_score", float("nan"))
        )

        interp_r2, interp_ok, interp_invalid = BenchmarkRunner._r2_plasma_mean_status(
            split_rows.get("interp", {}),
            target_vars=target_vars,
        )
        extrap_r2, extrap_ok, extrap_invalid = BenchmarkRunner._r2_plasma_mean_status(
            split_rows.get("extrap", {}),
            target_vars=target_vars,
        )
        row["test_r2_plasma_mean_interp"] = interp_r2
        row["test_r2_plasma_mean_extrap"] = extrap_r2
        r2_dual_reliable = bool(interp_ok and extrap_ok)
        if r2_dual_reliable:
            valid_r2 = [
                (interp_r2, max(0.0, float(weights.get("interp", 0.0)))),
                (extrap_r2, max(0.0, float(weights.get("extrap", 0.0)))),
            ]
            denom = float(sum(weight for _, weight in valid_r2))
            row["test_r2_plasma_mean_dual"] = (
                float(sum(value * weight for value, weight in valid_r2) / denom)
                if denom > 0.0
                else float("nan")
            )
        else:
            row["test_r2_plasma_mean_dual"] = float("nan")
        row["test_r2_plasma_mean_dual_reliable"] = r2_dual_reliable
        row["test_r2_plasma_mean_interp_reliable"] = bool(interp_ok)
        row["test_r2_plasma_mean_extrap_reliable"] = bool(extrap_ok)
        row["test_r2_plasma_mean_interp_invalid_vars"] = ",".join(interp_invalid)
        row["test_r2_plasma_mean_extrap_invalid_vars"] = ",".join(extrap_invalid)
        row["eval_protocol_mode_effective"] = "dual_axis"
        row["primary_split_effective"] = str(primary_split)
        row["interp_weight_effective"] = float(weights.get("interp", 0.0))
        row["extrap_weight_effective"] = float(weights.get("extrap", 0.0))
        validation_modes = {
            str(split_row.get("validation_selection_mode", "")).strip().lower()
            for split_row in split_rows.values()
        }
        validation_reliable = all(
            bool(split_row.get("validation_selection_reliable", False))
            for split_row in split_rows.values()
        ) and len(validation_modes) == 1 and validation_modes <= {"min", "max"}
        validation_values: list[tuple[float, float]] = []
        if validation_reliable:
            for split_name in required_lanes:
                try:
                    value = float(split_rows.get(split_name, {}).get("validation_selection_value", float("nan")))
                except (TypeError, ValueError):
                    validation_reliable = False
                    break
                weight = max(0.0, float(weights.get(split_name, 0.0)))
                if not np.isfinite(value) or not np.isfinite(weight):
                    validation_reliable = False
                    break
                if weight > 0.0:
                    validation_values.append((value, weight))
        validation_denom = float(sum(weight for _, weight in validation_values))
        validation_value = (
            float(sum(value * weight for value, weight in validation_values) / validation_denom)
            if validation_reliable and validation_denom > 0.0
            else float("nan")
        )
        if not np.isfinite(validation_value):
            validation_reliable = False
        row["validation_selection_reliable"] = validation_reliable
        row["validation_selection_value"] = validation_value
        row["validation_selection_value_dual"] = validation_value
        row["validation_selection_mode"] = next(iter(validation_modes)) if len(validation_modes) == 1 else ""
        row["validation_selection_metric"] = "weighted_dual_validation_selection"
        row["validation_selection_invalid_reason"] = (
            "" if validation_reliable else "unreliable_or_mixed_dual_validation_selection"
        )
        row["selection_split"] = "dual_axis"
        row["scaler_fit_split"] = "per_lane"
        if combined_diagnostics:
            row["_diagnostics"] = combined_diagnostics
        return row

    @staticmethod
    def _r2_plasma_mean_status(
        row: dict[str, Any],
        *,
        target_vars: list[str] | None = None,
    ) -> tuple[float, bool, list[str]]:
        vars_effective = [str(v) for v in (target_vars or [])]
        keys = [f"test_r2_{name}_plasma" for name in vars_effective]
        if not keys:
            keys = [
                k
                for k in sorted(row.keys())
                if k.startswith("test_r2_") and k.endswith("_plasma") and "_interp" not in k and "_extrap" not in k
            ]
        missing = [key.replace("test_r2_", "").replace("_plasma", "") for key in keys if key not in row]
        vals: list[float] = []
        nonfinite: list[str] = []
        for key in keys:
            if key not in row:
                continue
            name = key.replace("test_r2_", "").replace("_plasma", "")
            val = float(row[key])
            if not np.isfinite(val):
                nonfinite.append(name)
                continue
            vals.append(val)
        invalid = [*missing, *nonfinite]
        if invalid or len(vals) != len(keys):
            return float("nan"), False, invalid
        if len(vals) == 0:
            return float("nan"), False, []
        return float(np.mean(np.asarray(vals, dtype=np.float64))), True, []

    def _persist_resolved(self, *, split: dict[str, Any], resolved: dict[str, Any], include_manifest: bool = False) -> None:
        self.manifest_writer.save_manifest(
            split=split,
            resolved=resolved,
            include_manifest=include_manifest,
        )

    @staticmethod
    def _to_mode(raw: Any, *, default: str, true_mode: str) -> str:
        if isinstance(raw, bool):
            return true_mode if raw else "off"
        mode = str(raw if raw is not None else default).strip().lower()
        if mode == "on":
            return true_mode
        return mode

    @staticmethod
    def _validate_eval_scope_models(*, scope: str, model_names: list[str]) -> None:
        names = [str(v) for v in list(model_names)]
        expected = _ISOLATED_SCOPE_MODEL_NAMES.get(scope)
        if expected is not None and names != expected:
            raise ValueError(
                f"benchmark.eval_protocol.scope={scope} requires profile models exactly {expected}"
            )

    @staticmethod
    def _validate_eval_scope_train_sections(*, scope: str, train_cfg: dict[str, Any]) -> None:
        if scope == "common":
            return

        def _is_nonempty(section: str) -> bool:
            return bool(dict(train_cfg.get(section, {})))

        violations = [
            f"train.{section}"
            for section in _ISOLATED_SCOPE_FORBIDDEN_SECTIONS.get(scope, [])
            if _is_nonempty(section)
        ]

        if violations:
            raise ValueError(
                f"benchmark.eval_protocol.scope={scope} forbids non-empty sections: {sorted(set(violations))}"
            )

    def _resolve_train_per_model(self, *, train_cfg: dict[str, Any], model_names: list[str]) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        unet_like_cfg = dict(train_cfg.get("unet_like", {}))
        for model_name in model_names:
            cfg = dict(train_cfg.get(model_name, {}))
            item: dict[str, Any] = {
                "epochs": int(cfg.get("epochs", train_cfg.get("epochs", 20))),
                "lr": float(cfg.get("lr", train_cfg.get("lr", 1e-3))),
            }
            if model_name == "global_mlp":
                grad_scale_cfg = dict(cfg.get("grad_scale", {}))
                item["grad_scale_mode"] = self._to_mode(
                    grad_scale_cfg.get("mode", "off"),
                    default="off",
                    true_mode="auto",
                )
                item["grad_scale_auto_ref"] = str(grad_scale_cfg.get("auto_ref", "active_pixels"))
                item["grad_scale_auto_power"] = float(grad_scale_cfg.get("auto_power", 0.5))
                out_mult = float(dict(cfg.get("layer_lr_multiplier", {})).get("output", 1.0))
                item["layer_lr_output"] = out_mult
                head_refresh = bool(dict(cfg.get("output_head_refresh", {})).get("enabled", False))
                item["output_head_refresh_enabled"] = head_refresh
                item["batch_size_cases"] = int(cfg.get("batch_size_cases", 0))
            elif model_name in GRID_TORCH_MODELS:
                item["batch_size_cases"] = int(unet_like_cfg.get("batch_size_cases", 0))
                item["shuffle_cases"] = bool(unet_like_cfg.get("shuffle_cases", True))
            elif model_name in POD_DEEPONET_FAMILY_MODELS:
                item["batch_size_cases"] = int(unet_like_cfg.get("batch_size_cases", 0))
                item["shuffle_cases"] = bool(unet_like_cfg.get("shuffle_cases", True))
            elif model_name == "deeponet_plasma":
                stage1 = dict(cfg.get("stage1", {}))
                stage2 = dict(cfg.get("stage2", {}))
                item["stage1_epochs"] = int(stage1.get("epochs", 0))
                item["stage2_epochs"] = int(stage2.get("epochs", 0))
            resolved[model_name] = item
        return resolved

    def _evaluate_guardrails(
        self,
        *,
        phase: str,
        train_cfg: dict[str, Any],
        model_names: list[str],
        effective_steps_per_model: dict[str, Any] | None,
        target_vars_for_score_effective: list[str] | None = None,
    ) -> None:
        guard_cfg = dict(self.benchmark_cfg.get("guardrails", {}))
        if not bool(guard_cfg.get("enabled", False)):
            return
        checks = dict(guard_cfg.get("checks", {}))
        violations: list[str] = []

        if phase == "pre":
            if bool(checks.get("global_disabled_boost", False)) and "global_mlp" in model_names:
                gcfg = dict(train_cfg.get("global_mlp", {}))
                grad_scale_mode = self._to_mode(
                    dict(gcfg.get("grad_scale", {})).get("mode", "off"),
                    default="off",
                    true_mode="auto",
                )
                head_refresh = bool(dict(gcfg.get("output_head_refresh", {})).get("enabled", False))
                out_mult = float(dict(gcfg.get("layer_lr_multiplier", {})).get("output", 1.0))
                if grad_scale_mode == "off" and (not head_refresh) and out_mult <= 1.0:
                    violations.append(
                        "global_disabled_boost: global_mlp has grad_scale=off, output_head_refresh=false, "
                        "layer_lr_multiplier.output<=1.0"
                    )

        if phase == "post" and bool(checks.get("effective_steps_floor", False)):
            floor_cfg = dict(self.benchmark_cfg.get("effective_steps_floor", {}))
            steps_map = dict(effective_steps_per_model or {})
            for model_name in model_names:
                if model_name not in floor_cfg:
                    continue
                floor = int(floor_cfg[model_name])
                steps = steps_map.get(model_name, 0)
                if isinstance(steps, dict):
                    for split_name, split_steps in steps.items():
                        if int(split_steps) < floor:
                            violations.append(
                                f"effective_steps_floor: {model_name}.{split_name}={int(split_steps)} (<{floor})"
                            )
                elif int(steps) < floor:
                    violations.append(f"effective_steps_floor: {model_name}={int(steps)} (<{floor})")
        if violations:
            raise ValueError("benchmark.guardrails violations:\n- " + "\n- ".join(violations))
        return
