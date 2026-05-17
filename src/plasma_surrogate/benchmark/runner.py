"""Benchmark runner for mainline model comparison."""

from __future__ import annotations

import copy
import csv
import gc
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from plasma_surrogate.benchmark.runtime_context import (
    build_benchmark_data_context,
    resolve_effective_benchmark_cfg,
)
from plasma_surrogate.benchmark.profiles import resolve_profile_lock
from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.density_contract import (
    resolve_allvars_order,
    resolve_density_key,
    resolve_family_vars,
)
from plasma_surrogate.core.input_modes import (
    build_input_mode_effective_metadata,
    input_mode_metadata_keys,
    merge_effective_runtime_metadata,
    resolve_benchmark_runtime_controls,
)
from plasma_surrogate.core.model_families import (
    COORD_MLP_FAMILY_MODELS,
    GRID_TORCH_MODELS,
    POD_DEEPONET_FAMILY_MODELS,
    SPECTRAL_FAMILY_MODELS,
    UNET_FAMILY_MODELS,
    resolve_single_family_model,
)
from plasma_surrogate.core.model_input_policy import resolve_effective_input_mode_metadata_for_model
from plasma_surrogate.core.physics_contract import build_physics_cfg
from plasma_surrogate.eval.metrics import r2_masked, rmse_masked
from plasma_surrogate.eval.metrics_builder import (
    build_benchmark_eval_row,
    build_spatial_distribution_by_case_rows,
    build_spatial_distribution_summary_rows,
    build_spatial_error_by_case_rows,
    build_spatial_error_summary_rows,
)
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.infer.engine_builder import build_inference_engine
from plasma_surrogate.infer.optimize import cond_space_from_stats, validate_optimize_geom_contract
from plasma_surrogate.models.deeponet.pod_deeponet_torch import normalize_pod_deeponet_model_cfg
from plasma_surrogate.models.checkpoint import save_checkpoint
from plasma_surrogate.models.mlp.coord_mlp_pod_residual import normalize_coord_mlp_pod_residual_cfg
from plasma_surrogate.models.mlp.coord_mlp_torch import _normalize_coord_mlp_model_cfg
from plasma_surrogate.benchmark.model_dispatch import BenchmarkModelContext, run_model_train_eval
from plasma_surrogate.preprocessing.split import build_group_kfold_splits
from plasma_surrogate.train.losses import poisson_residual_loss
from plasma_surrogate.viz.runner import VizRunner


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


_ISOLATED_SCOPE_MODEL_NAMES: dict[str, list[str]] = {
    "global_frozen": ["global_mlp"],
    "unet_isolated": ["unet"],
    "unetpp_isolated": ["unetpp"],
    "unetpp_attn_isolated": ["unetpp_attn"],
    "unet_operator_v2_isolated": ["unet_operator_v2"],
    "fno_isolated": [SPECTRAL_FAMILY_MODELS[0]],
    "ffno_isolated": [SPECTRAL_FAMILY_MODELS[1]],
    "deeponet_isolated": ["deeponet_plasma"],
    "u_no_isolated": ["u_no"],
    "cno_isolated": ["cno"],
    "cno_operator_unet_isolated": ["cno_operator_unet"],
    "geom_deeponet_siren_isolated": ["geom_deeponet_siren"],
    "coord_mlp_fourier_isolated": ["coord_mlp_fourier"],
    "coord_mlp_siren_isolated": ["coord_mlp_siren"],
    "coord_mlp_pod_residual_isolated": ["coord_mlp_pod_residual"],
}

_MAINLINE_ISOLATED_SECTION_NAMES: tuple[str, ...] = (
    "global_mlp",
    *UNET_FAMILY_MODELS,
    *SPECTRAL_FAMILY_MODELS,
    "deeponet_plasma",
    "u_no",
    "cno",
    "cno_operator_unet",
    "geom_deeponet_siren",
    *COORD_MLP_FAMILY_MODELS,
)

_ISOLATED_SCOPE_FORBIDDEN_SECTIONS: dict[str, list[str]] = {
    scope: [name for name in _MAINLINE_ISOLATED_SECTION_NAMES if name not in set(active)]
    for scope, active in _ISOLATED_SCOPE_MODEL_NAMES.items()
}
_EVAL_PROTOCOL_SCOPES: tuple[str, ...] = ("common", *_ISOLATED_SCOPE_MODEL_NAMES.keys())
_FROZEN_REFERENCE_SCOPES: set[str] = set(_ISOLATED_SCOPE_MODEL_NAMES.keys())
_UNET_CONTRACT_OPTIONAL_SCOPES: set[str] = {
    "fno_isolated",
    "ffno_isolated",
    "deeponet_isolated",
    "u_no_isolated",
    "cno_isolated",
    "cno_operator_unet_isolated",
    "unet_operator_v2_isolated",
    "geom_deeponet_siren_isolated",
    "coord_mlp_fourier_isolated",
    "coord_mlp_siren_isolated",
    "coord_mlp_pod_residual_isolated",
}


def _release_torch_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _inject_input_mode_metadata_into_row(
    *,
    row: dict[str, Any],
    input_mode_meta: dict[str, Any],
    case_spatial_pack_used: bool = False,
) -> None:
    missing = [key for key in input_mode_metadata_keys() if key not in input_mode_meta]
    if missing:
        raise ValueError(
            "benchmark row metadata is missing required input_mode keys: "
            f"{missing}"
        )
    for key in input_mode_metadata_keys():
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
        nonfinite_count = float(row.get(f"test_nonfinite_count_{var_name}_plasma", 0.0) or 0.0)
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
    row["target_metrics_invalid_vars"] = "|".join(
        sorted(set([*invalid_vars, *str(row.get("target_metrics_invalid_vars", "")).split("|")]) - {""})
    )
    protocol_reliable = bool(row.get("primary_metric_protocol_reliable", row.get("eval_protocol_reliable", True)))
    row["primary_metric_reliable"] = bool(
        np.isfinite(float(row["primary_metric_value"])) and row["target_metrics_valid"] and protocol_reliable
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
    strict_input_mode: str = "error",
    allow_mode_fallback: bool = False,
    checkpoint_meta_path: Path,
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
        strict_input_mode=str(strict_input_mode),
        allow_mode_fallback=bool(allow_mode_fallback),
        input_mode_meta=input_mode_meta,
        checkpoint_meta_path=checkpoint_meta_path,
        deeponet_head=deeponet_head,
        grid_input_features_cfg=dict(grid_input_features_cfg or {}),
    )


class BenchmarkRunner:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = copy.deepcopy(dict(cfg or {}))
        self.benchmark_cfg = resolve_effective_benchmark_cfg(self.cfg)
        self.input_mode_meta = build_input_mode_effective_metadata(self.benchmark_cfg)
        self.strict_input_mode, self.allow_mode_fallback = resolve_benchmark_runtime_controls(self.benchmark_cfg)
        self.output_root = Path(self.benchmark_cfg.get("output_dir", "runs/benchmark_mainline"))
        self.store = ArtifactStore(self.output_root)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkRunner":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})

    def run(self) -> BenchmarkResult:
        self.output_root.mkdir(parents=True, exist_ok=True)

        requested_axis_mode = str(
            self.benchmark_cfg.get(
                "axis_mode",
                self.benchmark_cfg.get("preprocessing", {}).get("axis_schema", {}).get("mode", "steady"),
            )
        )
        profile = str(self.benchmark_cfg.get("profile", "m7_global_frozen_ref"))
        profile_lock = self._resolve_profile_lock(profile)
        requested_phi_mode = self.benchmark_cfg.get("phi_mode")
        if requested_phi_mode is not None and str(requested_phi_mode) != str(profile_lock["phi_mode"]):
            raise ValueError(
                f"Benchmark profile lock violation: phi_mode={requested_phi_mode} "
                f"must be {profile_lock['phi_mode']}"
            )
        infer_refine = int(self.benchmark_cfg.get("inference", {}).get("poisson_refine", {}).get("iters", 0))
        if infer_refine != int(profile_lock["poisson_refine_iters"]):
            raise ValueError(
                f"Benchmark profile lock violation: poisson_refine.iters={infer_refine} "
                f"must be {profile_lock['poisson_refine_iters']}"
            )
        if requested_axis_mode != str(profile_lock.get("axis_mode", "steady")):
            raise ValueError(
                f"Benchmark profile lock violation: axis_mode={requested_axis_mode} "
                f"must be {profile_lock.get('axis_mode', 'steady')}"
            )
        requested_primary_qoi = str(
            self.benchmark_cfg.get("boundary_operator", {}).get(
                "primary_qoi_key",
                profile_lock.get("primary_qoi_key", "Gamma_i"),
            )
        )
        if str(profile_lock.get("primary_qoi_key", requested_primary_qoi)) != requested_primary_qoi:
            raise ValueError(
                f"Benchmark profile lock violation: boundary_operator.primary_qoi_key={requested_primary_qoi} "
                f"must be {profile_lock.get('primary_qoi_key')}"
            )
        context = build_benchmark_data_context(
            cfg=self.benchmark_cfg,
            output_root=self.output_root,
            profile_lock=profile_lock,
        )
        profile_lock = context.profile_lock
        resolved = context.resolved
        resolved.update(self.input_mode_meta)
        eval_protocol_cfg = dict(self.benchmark_cfg.get("eval_protocol", {}))
        eval_protocol_mode = str(eval_protocol_cfg.get("mode", "single")).strip().lower()
        if eval_protocol_mode not in {"single", "primary_axis", "dual_axis"}:
            raise ValueError("benchmark.eval_protocol.mode must be one of: single, primary_axis, dual_axis")
        eval_protocol_scope = str(eval_protocol_cfg.get("scope", "common")).strip().lower()
        if eval_protocol_scope not in set(_EVAL_PROTOCOL_SCOPES):
            raise ValueError(
                "benchmark.eval_protocol.scope must be one of: "
                f"{', '.join(_EVAL_PROTOCOL_SCOPES)}"
            )
        frozen_ref_tag = str(eval_protocol_cfg.get("frozen_ref_tag", "")).strip()
        self._validate_eval_scope_models(scope=eval_protocol_scope, model_names=profile_lock["models"])
        eval_cfg = dict(self.benchmark_cfg.get("eval", {}))
        region_bands_cfg = dict(eval_cfg.get("region_bands", {}))
        region_bands_mode = str(region_bands_cfg.get("mode", "fixed_px")).strip().lower()
        if region_bands_mode not in {"fixed_px", "signed_quantile"}:
            raise ValueError("benchmark.eval.region_bands.mode must be one of: fixed_px, signed_quantile")
        region_bands_effective = {
            "mode": region_bands_mode,
            "boundary_in_px": float(region_bands_cfg.get("boundary_in_px", 2.0)),
            "deep_plasma_px": float(region_bands_cfg.get("deep_plasma_px", 10.0)),
            "boundary_q": float(region_bands_cfg.get("boundary_q", 0.15)),
            "deep_q": float(region_bands_cfg.get("deep_q", 0.70)),
        }
        interp_mode = str(eval_cfg.get("interp_mode", eval_protocol_cfg.get("interp_mode", "marginal"))).strip().lower()
        if interp_mode not in {"marginal", "overlap"}:
            raise ValueError("benchmark.eval.interp_mode must be one of: marginal, overlap")
        primary_split = str(eval_protocol_cfg.get("primary_split", "interp")).strip().lower()
        if primary_split not in {"interp", "extrap", "structure_holdout"}:
            raise ValueError(
                "benchmark.eval_protocol.primary_split must be one of: interp, extrap, structure_holdout"
            )
        if eval_protocol_mode == "dual_axis" and primary_split == "structure_holdout":
            raise ValueError(
                "benchmark.eval_protocol.primary_split=structure_holdout requires "
                "benchmark.eval_protocol.mode=primary_axis"
            )
        interp_weight = float(eval_protocol_cfg.get("interp_weight", 0.5))
        extrap_weight = float(eval_protocol_cfg.get("extrap_weight", 0.5))
        resolved["eval_protocol"] = {
            "mode": eval_protocol_mode,
            "scope_effective": eval_protocol_scope,
            "frozen_ref_tag": frozen_ref_tag,
            "primary_split": primary_split,
            "interp_weight": interp_weight,
            "extrap_weight": extrap_weight,
            "interp_mode": interp_mode,
        }
        resolved["active_model_scope"] = eval_protocol_scope
        self._persist_resolved(split=context.split, resolved=resolved, include_manifest=True)

        global_seed = context.global_seed
        n_cases = context.n_cases
        h, w = context.h, context.w
        cond_order = context.cond_order
        case_id_to_idx = {str(c["case_id"]): i for i, c in enumerate(context.dataset.cases)}
        geom_provider = context.geom_provider
        bundle = context.bundle
        cond_schema = bundle.cond_schema_obj()
        axis_schema = bundle.axis_schema_obj()
        cond = context.cond
        y = context.y
        cond_scaled = context.cond_scaled
        y_scaled = context.y_scaled
        y_vars = resolve_allvars_order(
            list(bundle.schemas.get("output_layout", {}).get("vars", [])),
            prefer_linear=True,
        )
        target_family_for_score_raw = str(eval_cfg.get("target_family_for_score", "auto")).strip().lower()
        if target_family_for_score_raw not in {"auto", "allvars", "logpair", "field", "merged"}:
            raise ValueError(
                "benchmark.eval.target_family_for_score must be one of: auto, allvars, logpair, field, merged"
            )
        target_vars_for_score_raw = eval_cfg.get("target_vars_for_score", "auto")
        if target_family_for_score_raw in {"logpair", "field", "allvars", "merged"}:
            family_name = "allvars" if target_family_for_score_raw == "merged" else target_family_for_score_raw
            target_vars_for_score_effective = resolve_family_vars(
                family=family_name,
                available=y_vars,
                prefer_linear=True,
            )
        else:
            if target_vars_for_score_raw in (None, "auto", "all"):
                target_vars_for_score_effective = resolve_family_vars(
                    family="allvars",
                    available=y_vars,
                    prefer_linear=True,
                )
            else:
                if not isinstance(target_vars_for_score_raw, list) or len(target_vars_for_score_raw) == 0:
                    raise ValueError("benchmark.eval.target_vars_for_score must be a non-empty list or 'auto'")
                target_vars_for_score_effective = [str(v) for v in target_vars_for_score_raw]
                unknown = [v for v in target_vars_for_score_effective if v not in set(y_vars)]
                if unknown:
                    raise ValueError(
                        "benchmark.eval.target_vars_for_score contains vars not present in output layout: "
                        f"{unknown}; available={y_vars}"
                    )
                if len(set(target_vars_for_score_effective)) != len(target_vars_for_score_effective):
                    raise ValueError("benchmark.eval.target_vars_for_score must not contain duplicates")
        if target_family_for_score_raw != "auto" and target_vars_for_score_raw not in (None, "auto", "all"):
            family_name = "allvars" if target_family_for_score_raw == "merged" else target_family_for_score_raw
            family_expected = resolve_family_vars(
                family=family_name,
                available=y_vars,
                prefer_linear=True,
            )
            if list(target_vars_for_score_effective) != list(family_expected):
                raise ValueError(
                    "benchmark.eval.target_vars_for_score must match benchmark.eval.target_family_for_score="
                    f"{target_family_for_score_raw}: expected={family_expected}, got={target_vars_for_score_effective}"
                )
        resolved.setdefault("eval", {})
        resolved["eval"]["target_family_for_score_effective"] = target_family_for_score_raw
        resolved["eval"]["target_vars_for_score_effective"] = list(target_vars_for_score_effective)
        resolved["target_vars_effective"] = list(y_vars)
        resolved["eval"]["region_bands_effective"] = dict(region_bands_effective)
        density_ne_key = resolve_density_key(y_vars, canonical="ne", prefer_linear=True)
        density_ni_key = resolve_density_key(y_vars, canonical="ni", prefer_linear=True)
        resolved["density_contract_version_effective"] = "v2" if ("ne" in y_vars or "ni" in y_vars) else "v1"
        resolved["density_keys_effective"] = [k for k in ["ne", "ni"] if k in set(y_vars)]
        resolved["density_output_vars_effective"] = [v for v in [density_ne_key, density_ni_key] if v is not None]
        if eval_protocol_mode == "dual_axis":
            default_primary_metric = "score_total_dual"
        elif eval_protocol_mode == "primary_axis":
            default_primary_metric = f"score_total_{primary_split}"
        else:
            default_primary_metric = "score_total"
        primary_metric = str(eval_cfg.get("primary_metric", default_primary_metric)).strip()
        if primary_metric == "":
            raise ValueError("benchmark.eval.primary_metric must be a non-empty string")
        primary_mode = str(eval_cfg.get("primary_mode", "min")).strip().lower()
        if primary_mode not in {"min", "max"}:
            raise ValueError("benchmark.eval.primary_mode must be one of: min, max")
        resolved["eval"]["primary_metric_effective"] = primary_metric
        resolved["eval"]["primary_mode_effective"] = primary_mode
        protocol_variant = str(eval_cfg.get("protocol_variant", "")).strip()
        resolved["eval"]["protocol_variant_effective"] = protocol_variant if protocol_variant else "default"
        tr = context.tr
        va = context.va
        te = context.te
        transforms = context.transforms
        feature_store = context.feature_store
        deeponet_index = context.deeponet_index
        deeponet_index_meta = context.deeponet_index_meta
        deeponet_poisson_index = context.deeponet_poisson_index
        deeponet_poisson_meta = context.deeponet_poisson_meta
        deeponet_boundary_index = context.deeponet_boundary_index
        deeponet_boundary_meta = context.deeponet_boundary_meta
        coord_feature_pack = context.coord_feature_pack
        lock_hash = context.lock_hash
        split_interp = self._load_split_json(
            self.output_root / "preprocessing" / "split" / f"split_interp_{interp_mode}_v1.json",
            fallback=self._load_split_json(
                self.output_root / "preprocessing" / "split" / "split_interp_v1.json",
                fallback=context.split,
            ),
        )
        interp_overlap_status = self._load_json_if_exists(
            self.output_root / "preprocessing" / "split" / "split_interp_status_v1.json"
        )
        preprocess_report = self._load_json_if_exists(
            self.output_root / "preprocessing" / "validation" / "report.json"
        )
        if not interp_overlap_status:
            interp_overlap_status = {
                "requested_mode": interp_mode,
                "applied_mode": interp_mode,
                "feasible": True,
                "reason": "",
                "fallback_applied": False,
            }
        split_extrap = self._load_split_json(
            self.output_root / "preprocessing" / "split" / "split_extrap_v1.json",
            fallback=context.split,
        )
        split_structure_holdout = self._load_split_json(
            self.output_root / "preprocessing" / "split" / "split_structure_holdout_v1.json",
            fallback=context.split,
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
        strict_input_mode, allow_mode_fallback = self.strict_input_mode, self.allow_mode_fallback
        interp_overlap_ratio = float(resolved["tuple_overlap_ratio"]["interp"])
        extrapolation_severity = float(np.clip((1.0 - interp_overlap_ratio) * 0.7 + test_holdout_ratio * 0.3, 0.0, 1.0))
        resolved["eval_protocol"]["extrapolation_severity"] = {
            "score": extrapolation_severity,
            "tuple_overlap_interp": interp_overlap_ratio,
            "tuple_overlap_extrap": float(resolved["tuple_overlap_ratio"]["extrap"]),
            "holdout_ratio": test_holdout_ratio,
        }
        resolved["eval_protocol"]["structure_holdout_test_ratio"] = structure_holdout_ratio
        resolved["eval_protocol"]["interp_mode_effective"] = str(interp_overlap_status.get("applied_mode", interp_mode))
        resolved["interp_overlap_status"] = interp_overlap_status
        self._persist_resolved(split=context.split, resolved=resolved)

        if len(profile_lock["models"]) == 0:
            header = ["model_id"]
            for var_name in target_vars_for_score_effective:
                header.extend(
                    [
                        f"test_rmse_{var_name}",
                        f"test_rmse_{var_name}_plasma",
                        f"test_r2_{var_name}",
                        f"test_r2_{var_name}_plasma",
                    ]
                )
            header.extend(
                [
                    "test_poisson_phi",
                    "qoi_uniformity",
                    "qoi_boundary_gamma_uniformity",
                    "single_poisson_residual",
                    "single_poisson_residual_map_l2",
                    "single_boundary_operator_proxy_loss",
                    "single_boundary_residual_map_l2",
                    "opt_best_uniformity",
                ]
            )
            self.store.save_csv("leaderboard.csv", header, [])
            return BenchmarkResult(leaderboard=[], leaderboard_path=self.output_root / "leaderboard.csv")

        leaderboard: list[dict[str, Any]] = []
        geom_ctx = geom_provider.get()
        train_cfg = dict(self.benchmark_cfg.get("train", {}))
        self._validate_eval_scope_train_sections(scope=eval_protocol_scope, train_cfg=train_cfg)
        active_unet_model = resolve_single_family_model(model_names=profile_lock["models"], family=UNET_FAMILY_MODELS)
        active_spectral_model = resolve_single_family_model(
            model_names=profile_lock["models"],
            family=SPECTRAL_FAMILY_MODELS,
        )
        if active_unet_model is not None and target_family_for_score_raw in {"allvars", "logpair", "field"}:
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
        if active_spectral_model is not None and target_family_for_score_raw in {"allvars", "logpair", "field"}:
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
        loss_cfg = dict(train_cfg.get("loss", {}))
        curriculum_cfg = dict(train_cfg.get("curriculum", {}))
        aggregate_cfg = dict(self.benchmark_cfg.get("eval", {}).get("aggregate_score", {}))
        sample_mean_group_mode = str(loss_cfg.get("supervised", {}).get("sample_mean_group_mode", "batch"))
        if sample_mean_group_mode != "batch":
            raise ValueError("supervised.sample_mean_group_mode must be one of: batch")
        sample_mean_weight_denominator = str(
            loss_cfg.get("supervised", {}).get("sample_mean_weight_denominator", "weighted")
        )
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
            default_lambda_poisson=0.05,
            default_lambda_bc=0.02,
        )
        linear_density_enabled = ("ne" in set(y_vars)) and ("log_ne" not in set(y_vars))
        if bool(physics_cfg.get("enabled", False)) and linear_density_enabled:
            physics_cfg = {"enabled": False, "resolved_terms": []}
            resolved["physics_auto_disabled_reason"] = (
                "physics-aware is disabled when canonical linear density outputs are active in mainline"
            )
        resolved["physics_resolved_terms"] = list(physics_cfg.get("resolved_terms", []))
        effective_steps_per_model: dict[str, Any] = {}
        unet_contract_samples: list[dict[str, Any]] = []
        fno_contract_samples: list[dict[str, Any]] = []
        ffno_contract_samples: list[dict[str, Any]] = []
        coord_mlp_contract_samples: list[dict[str, Any]] = []
        deeponet_contract_samples: list[dict[str, Any]] = []
        deeponet_pod_contract_samples: list[dict[str, Any]] = []
        guardrail_warnings: list[str] = []
        if bool(interp_overlap_status.get("fallback_applied", False)):
            guardrail_warnings.append(
                "interp_overlap_fallback: requested overlap split is not feasible for this dataset; "
                "fallback to marginal interpolation split was applied"
            )
        guardrail_warnings.extend(
            self._evaluate_guardrails(
                phase="pre",
                train_cfg=train_cfg,
                model_names=profile_lock["models"],
                effective_steps_per_model=None,
                target_vars_for_score_effective=target_vars_for_score_effective,
            )
        )
        self._persist_resolved(split=context.split, resolved=resolved, include_manifest=True)

        def _run_single_split_model(
            *,
            model_name: str,
            model_idx: int,
            model_dir: Path,
            tr_idx: np.ndarray,
            va_idx: np.ndarray,
            te_idx: np.ndarray,
        ) -> dict[str, Any]:
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
                    global_seed=global_seed,
                    n_cases=n_cases,
                    h=h,
                    w=w,
                    y_vars=y_vars,
                    cond_scaled=cond_scaled,
                    y=y,
                    y_scaled=y_scaled,
                    tr=tr_idx,
                    va=va_idx,
                    te=te_idx,
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
                    case_spatial_feature_pack=context.case_spatial_feature_pack,
                    static_spatial_feature_pack=context.static_spatial_feature_pack,
                    case_structure_feature_pack=context.case_structure_feature_pack,
                    coord_distance_transform_stats=dict(bundle.transforms.get("distance_transform_stats", {})),
                    structure_descriptor_pack=bundle.schemas.get("structure_descriptor_pack"),
                    latent_feature_pack=bundle.schemas.get("latent_feature_pack"),
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

            checkpoint_meta = merge_effective_runtime_metadata(
                runtime_meta=effective_input_mode_meta,
                dispatch_meta=extra_artifacts,
            )
            save_checkpoint(model, model_dir / "checkpoints", extra_meta=checkpoint_meta)

            viz = VizRunner(model_dir / "eval")
            viz.plot_loss_curve(history, rel_path="plots/loss_curve.png")
            has_phi = "phi" in true_eval and "phi" in pred_eval
            single_qoi: dict[str, Any] = {"uniformity": 0.0, "boundary_gamma_uniformity": 0.0}
            single_diagnostics: dict[str, Any] = {"poisson_residual_norm": 0.0, "boundary_operator_proxy_loss": 0.0}
            best_uniformity = 0.0
            batch_rows: list[dict[str, Any]] = []
            case_spatial_inference_skip = bool(
                context.case_spatial_feature_pack or context.case_structure_feature_pack
            )
            if has_phi and not case_spatial_inference_skip:
                viz.plot_parity(true_eval["phi"].reshape(-1), pred_eval["phi"].reshape(-1), rel_path="plots/parity_phi.png")
                viz.plot_field_triplet(true_eval["phi"][0, 0], pred_eval["phi"][0, 0], rel_path="plots/phi_triplet.png")

                infer_engine = _build_benchmark_inference_engine(
                    model=model,
                    cond_schema=cond_schema,
                    axis_schema=axis_schema,
                    geometry_provider=geom_provider,
                    output_dir=model_dir / "inference",
                    transform_bundle=transforms,
                    cond_stats=bundle.schemas.get("cond_stats", {}),
                    phi_mode=profile_lock["phi_mode"],
                    phi_hybrid_steps=int(self.benchmark_cfg.get("phi_hybrid_steps", 1)),
                    poisson_refine_iters=int(profile_lock["poisson_refine_iters"]),
                    ood_cfg=self.benchmark_cfg.get("inference", {}).get("ood", {"poisson_residual_limit": 1e2}),
                    feature_store=feature_store,
                    coord_scaler=bundle.transforms.get("coord_scaler", {}),
                    coord_feature_scaler=bundle.transforms.get("coord_feature_scaler", {}),
                    coord_feature_pack=bundle.schemas.get("coord_feature_pack"),
                    coord_distance_transform_stats=bundle.transforms.get("distance_transform_stats", {}),
                    input_mode_meta=effective_input_mode_meta,
                    strict_input_mode=strict_input_mode,
                    allow_mode_fallback=allow_mode_fallback,
                    checkpoint_meta_path=model_dir / "checkpoints" / "meta.json",
                    deeponet_head=(
                        getattr(model, "poisson_head", None)
                        if model_name == "deeponet_plasma"
                        else None
                    )
                    or (model if model_name == "deeponet_plasma" else None),
                    grid_input_features_cfg=dict(train_cfg.get(model_name, {}).get("input_features", {})),
                )
                infer_axis = self.benchmark_cfg.get("inference", {}).get(
                    "axis",
                    {"mode": context.requested_axis_mode, "value": 0.0},
                )

                cond_dict = {k: float(cond[te_idx[0], i]) for i, k in enumerate(cond_order)}
                single = infer_engine.single_run_aggregated(
                    cond=cond_dict,
                    geom={"geom_id": "default"},
                    axis=infer_axis,
                )
                batch_rows = infer_engine.batch_run(
                    conds=[{k: float(cond[idx, i]) for i, k in enumerate(cond_order)} for idx in te_idx[: min(3, len(te_idx))]],
                    geom={"geom_id": "default"},
                    axis=infer_axis,
                )
                best = infer_engine.optimize_run(
                    # Keep benchmark optimize contract aligned with CLI/engine validation entrypoint.
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
                    n_trials=8,
                    geom={"geom_id": "default"},
                    axis=infer_axis,
                    seed=global_seed + 100 + model_idx,
                )
                single_qoi = dict(single.qoi)
                single_diagnostics = dict(single.diagnostics)
                best_uniformity = float(best["best_value"])

            summary = {
                "model_id": model_name,
                "rmse": metrics,
                "poisson_phi_rmse": float(poisson_residual_loss(pred_eval["phi"][:, 0])) if has_phi else 0.0,
                "single_qoi": single_qoi,
                "single_diagnostics": single_diagnostics,
                "batch_count": len(batch_rows),
                "optimize": {"best_value": best_uniformity},
                "lock_hash": lock_hash,
            }
            ArtifactStore(model_dir).save_json("summary.json", summary)
            if summary["lock_hash"] != resolved["artifact_hashes"]["lock_hash"]:
                raise ValueError("benchmark lock hash mismatch")
            row = build_benchmark_eval_row(
                model_id=model_name,
                metrics=metrics,
                r2_scores=r2_scores,
                pred_eval=pred_eval,
                true_eval=true_eval,
                mask_plasma=metric_mask,
                distance_any=geom_ctx.distance_any if geom_ctx is not None else None,
                distance_signed=geom_ctx.distance_signed if geom_ctx is not None else None,
                bc_dir_mask=geom_ctx.bc_dir_mask if geom_ctx is not None else None,
                wafer_mask=(
                    np.asarray(geom_ctx.regions.get("wafer_mask"), dtype=np.float32)
                    if (geom_ctx is not None and getattr(geom_ctx, "regions", {}).get("wafer_mask") is not None)
                    else None
                ),
                aggregate_cfg=aggregate_cfg,
                target_vars_for_score=target_vars_for_score_effective,
                region_band_cfg=region_bands_effective,
                single_qoi=single_qoi,
                single_diagnostics=single_diagnostics,
                opt_best_uniformity=float(best_uniformity),
            )
            _inject_input_mode_metadata_into_row(
                row=row,
                input_mode_meta=effective_input_mode_meta,
                case_spatial_pack_used=bool(context.case_spatial_feature_pack or context.case_structure_feature_pack),
            )
            row["scaler_fit_split"] = str(preprocess_report.get("scaler_fit_split", "unknown"))
            density_value_transform = dict(preprocess_report.get("density_value_transform_effective", {}))
            if density_value_transform:
                row["density_transform_ne"] = str(density_value_transform.get("ne", ""))
                row["density_transform_ni"] = str(density_value_transform.get("ni", ""))
            spatial_audit_cfg = dict(self.benchmark_cfg.get("eval", {}).get("spatial_error_audit", {}))
            boundary_type_breakdown = bool(spatial_audit_cfg.get("boundary_type_breakdown", False))
            pred_spatial = dict(pred_eval)
            true_spatial = dict(true_eval)
            for canonical in ("ne", "ni"):
                pred_key = resolve_density_key(list(pred_eval.keys()), canonical=canonical, prefer_linear=True)
                true_key = resolve_density_key(list(true_eval.keys()), canonical=canonical, prefer_linear=True)
                if pred_key is not None and true_key is not None:
                    pred_spatial[canonical] = np.asarray(pred_eval[pred_key], dtype=np.float32)
                    true_spatial[canonical] = np.asarray(true_eval[true_key], dtype=np.float32)
            spatial_common_keys = list(set(pred_spatial.keys()) & set(true_spatial.keys()))
            spatial_vars = [v for v in target_vars_for_score_effective if v in set(spatial_common_keys)]
            if not spatial_vars:
                spatial_vars = list(spatial_common_keys)
            eval_case_ids = [
                str(context.dataset.cases[int(i)].get("case_id", int(i)))
                for i in np.asarray(te_idx, dtype=np.int64).tolist()
            ]
            spatial_rows = build_spatial_error_summary_rows(
                pred_eval=pred_spatial,
                true_eval=true_spatial,
                mask_plasma=metric_mask,
                distance_signed=(geom_ctx.distance_signed if geom_ctx is not None else None),
                distance_any=(geom_ctx.distance_any if geom_ctx is not None else None),
                bc_dir_mask=(geom_ctx.bc_dir_mask if geom_ctx is not None else None),
                wafer_mask=(
                    np.asarray(geom_ctx.regions.get("wafer_mask"), dtype=np.float32)
                    if (geom_ctx is not None and getattr(geom_ctx, "regions", {}).get("wafer_mask") is not None)
                    else None
                ),
                boundary_type_breakdown=boundary_type_breakdown,
                vars_for_summary=[v for v in spatial_vars if v in set(true_spatial.keys())],
                region_band_cfg=region_bands_effective,
            )
            if spatial_rows:
                ArtifactStore(model_dir / "eval").save_csv(
                    "spatial_error_summary.csv",
                    ["var", "region", "boundary_type", "n_points", "rmse", "r2"],
                    [
                        [
                            r.get("var", ""),
                            r.get("region", ""),
                            r.get("boundary_type", "na"),
                            r.get("n_points", 0.0),
                            r.get("rmse", 0.0),
                            r.get("r2", 0.0),
                        ]
                        for r in spatial_rows
                    ],
                )
            spatial_case_rows = build_spatial_error_by_case_rows(
                pred_eval=pred_spatial,
                true_eval=true_spatial,
                mask_plasma=metric_mask,
                distance_signed=(geom_ctx.distance_signed if geom_ctx is not None else None),
                distance_any=(geom_ctx.distance_any if geom_ctx is not None else None),
                bc_dir_mask=(geom_ctx.bc_dir_mask if geom_ctx is not None else None),
                wafer_mask=(
                    np.asarray(geom_ctx.regions.get("wafer_mask"), dtype=np.float32)
                    if (geom_ctx is not None and getattr(geom_ctx, "regions", {}).get("wafer_mask") is not None)
                    else None
                ),
                boundary_type_breakdown=boundary_type_breakdown,
                vars_for_summary=[v for v in spatial_vars if v in set(true_spatial.keys())],
                case_ids=eval_case_ids,
                region_band_cfg=region_bands_effective,
            )
            if spatial_case_rows:
                ArtifactStore(model_dir / "eval").save_csv(
                    "spatial_error_by_case.csv",
                    ["case_id", "case_index", "var", "region", "boundary_type", "n_points", "rmse", "r2"],
                    [
                        [
                            r.get("case_id", ""),
                            r.get("case_index", 0.0),
                            r.get("var", ""),
                            r.get("region", ""),
                            r.get("boundary_type", "na"),
                            r.get("n_points", 0.0),
                            r.get("rmse", 0.0),
                            r.get("r2", 0.0),
                        ]
                        for r in spatial_case_rows
                    ],
            )
            distribution_cfg = dict(self.benchmark_cfg.get("eval", {}).get("spatial_distribution_audit", {}))
            if bool(distribution_cfg.get("enabled", True)):
                raw_distribution_vars = distribution_cfg.get("vars", "density")
                density_vars = [v for v in ["ne", "ni"] if v in set(true_spatial.keys()) and v in set(pred_spatial.keys())]
                if isinstance(raw_distribution_vars, list):
                    distribution_vars = [
                        str(v)
                        for v in raw_distribution_vars
                        if str(v) in set(true_spatial.keys()) and str(v) in set(pred_spatial.keys())
                    ]
                elif str(raw_distribution_vars).strip().lower() == "all":
                    distribution_vars = [v for v in spatial_vars if v in set(true_spatial.keys()) and v in set(pred_spatial.keys())]
                else:
                    distribution_vars = density_vars or [
                        v for v in spatial_vars if v in set(true_spatial.keys()) and v in set(pred_spatial.keys())
                    ]
                distribution_case_rows = build_spatial_distribution_by_case_rows(
                    pred_eval=pred_spatial,
                    true_eval=true_spatial,
                    mask_plasma=metric_mask,
                    vars_for_summary=distribution_vars,
                    case_ids=eval_case_ids,
                    top_fraction=float(distribution_cfg.get("top_fraction", 0.10)),
                )
                distribution_summary_rows = build_spatial_distribution_summary_rows(distribution_case_rows)
                if distribution_case_rows:
                    header = list(distribution_case_rows[0].keys())
                    ArtifactStore(model_dir / "eval").save_csv(
                        "spatial_distribution_by_case.csv",
                        header,
                        [[r.get(key, "") for key in header] for r in distribution_case_rows],
                    )
                if distribution_summary_rows:
                    header = list(distribution_summary_rows[0].keys())
                    ArtifactStore(model_dir / "eval").save_csv(
                        "spatial_distribution_summary.csv",
                        header,
                        [[r.get(key, "") for key in header] for r in distribution_summary_rows],
                    )
                    for summary_row in distribution_summary_rows:
                        var_name = str(summary_row.get("var", ""))
                        if not var_name:
                            continue
                        for metric_name in [
                            "integral_rel_error_mean",
                            "p99_rel_error_mean",
                            "center_of_mass_error_px_mean",
                            "peak_location_error_px_mean",
                            "distribution_error_score_mean",
                        ]:
                            row[f"dist_{var_name}_{metric_name}"] = float(
                                summary_row.get(metric_name, float("nan"))
                            )
                plot_worst_cases = int(distribution_cfg.get("plot_worst_cases", 0))
                if plot_worst_cases > 0 and distribution_case_rows:
                    worst_rows = sorted(
                        [
                            r
                            for r in distribution_case_rows
                            if np.isfinite(float(r.get("distribution_error_score", float("nan"))))
                        ],
                        key=lambda r: float(r.get("distribution_error_score", float("nan"))),
                        reverse=True,
                    )[:plot_worst_cases]
                    for worst in worst_rows:
                        var_name = str(worst.get("var", ""))
                        case_idx = int(float(worst.get("case_index", 0.0)))
                        if var_name not in true_spatial or var_name not in pred_spatial:
                            continue
                        true_arr = np.asarray(true_spatial[var_name], dtype=np.float32)
                        pred_arr = np.asarray(pred_spatial[var_name], dtype=np.float32)
                        if true_arr.ndim == 4:
                            true_field = true_arr[case_idx, 0]
                            pred_field = pred_arr[case_idx, 0]
                        elif true_arr.ndim == 3:
                            true_field = true_arr[case_idx]
                            pred_field = pred_arr[case_idx]
                        else:
                            continue
                        case_token = "".join(
                            ch if ch.isalnum() or ch in {"-", "_"} else "_"
                            for ch in str(worst.get("case_id", case_idx))
                        )[:80]
                        viz.plot_field_triplet(
                            true_field,
                            pred_field,
                            rel_path=f"plots/spatial_distribution_worst_{var_name}_{case_token}.png",
                        )
            row["protocol_variant"] = protocol_variant if protocol_variant else "default"
            _attach_primary_metric_status(
                row,
                primary_metric=primary_metric,
                model_name=model_name,
                target_vars=target_vars_for_score_effective,
            )
            return {
                "row": row,
                "effective_steps": int(extra_artifacts.get("effective_steps", max(len(history), 0))),
                "unet_contract_effective": dict(extra_artifacts.get("unet_contract_effective", {})),
                "fno_contract_effective": dict(extra_artifacts.get("fno_contract_effective", {})),
                "ffno_contract_effective": dict(extra_artifacts.get("ffno_contract_effective", {})),
                "coord_mlp_contract_effective": dict(extra_artifacts.get("coord_mlp_contract_effective", {})),
                "deeponet_contract_effective": dict(extra_artifacts.get("deeponet_contract_effective", {})),
                "deeponet_pod_contract_effective": dict(extra_artifacts.get("deeponet_pod_contract_effective", {})),
            }

        for model_idx, model_name in enumerate(profile_lock["models"]):
            model_dir = self.output_root / "models" / model_name
            if eval_protocol_mode == "single":
                out = _run_single_split_model(
                    model_name=model_name,
                    model_idx=model_idx,
                    model_dir=model_dir,
                    tr_idx=tr,
                    va_idx=va,
                    te_idx=te,
                )
                leaderboard.append(out["row"])
                effective_steps_per_model[model_name] = int(out.get("effective_steps", 0))
                if out.get("unet_contract_effective"):
                    unet_contract_samples.append(dict(out["unet_contract_effective"]))
                if out.get("fno_contract_effective"):
                    fno_contract_samples.append(dict(out["fno_contract_effective"]))
                if out.get("ffno_contract_effective"):
                    ffno_contract_samples.append(dict(out["ffno_contract_effective"]))
                if out.get("coord_mlp_contract_effective"):
                    coord_mlp_contract_samples.append(dict(out["coord_mlp_contract_effective"]))
                if out.get("deeponet_contract_effective"):
                    deeponet_contract_samples.append(dict(out["deeponet_contract_effective"]))
                if out.get("deeponet_pod_contract_effective"):
                    deeponet_pod_contract_samples.append(dict(out["deeponet_pod_contract_effective"]))
            elif eval_protocol_mode == "primary_axis":
                split_def = split_by_name[primary_split]
                tr_i, va_i, te_i = self._indices_from_split(split_def, case_id_to_idx)
                split_offset = {"interp": 1, "extrap": 2, "structure_holdout": 3}[primary_split]
                out = _run_single_split_model(
                    model_name=model_name,
                    model_idx=(model_idx * 10 + split_offset),
                    model_dir=model_dir / "eval_protocol" / primary_split,
                    tr_idx=tr_i,
                    va_idx=va_i,
                    te_idx=te_i,
                )
                row = dict(out["row"])
                for var_name in target_vars_for_score_effective:
                    row[f"test_r2_{var_name}_plasma_{primary_split}"] = float(
                        row.get(f"test_r2_{var_name}_plasma", 0.0)
                    )
                row[f"score_total_{primary_split}"] = float(row.get("score_total", 0.0))
                row[f"score_nrmse_plasma_mean_{primary_split}"] = float(
                    row.get("score_nrmse_plasma_mean", float("nan"))
                )
                r2_mean, r2_valid, r2_invalid = self._r2_plasma_mean_status(
                    row,
                    target_vars=target_vars_for_score_effective,
                )
                row[f"test_r2_plasma_mean_{primary_split}"] = float(r2_mean)
                row["primary_axis_split"] = primary_split
                row["interp_test_cases"] = float(len(split_interp.get("test", [])))
                row["extrap_test_cases"] = float(len(split_extrap.get("test", [])))
                row["structure_holdout_test_cases"] = float(len(split_structure_holdout.get("test", [])))
                row["interp_overlap_fallback_applied"] = bool(interp_overlap_status.get("fallback_applied", False))
                row["interp_mode_effective"] = str(interp_overlap_status.get("applied_mode", interp_mode))
                protocol_issues: list[str] = []
                if primary_split == "interp" and bool(interp_overlap_status.get("fallback_applied", False)):
                    protocol_issues.append(
                        f"interp_overlap_fallback:{interp_overlap_status.get('reason', 'unknown')}"
                    )
                if not bool(r2_valid):
                    protocol_issues.append(f"r2_invalid:{','.join(r2_invalid)}")
                row["eval_protocol_issue"] = "|".join(protocol_issues)
                row["eval_protocol_reliable"] = bool(len(protocol_issues) == 0)
                row["primary_metric_protocol_reliable"] = bool(len(protocol_issues) == 0)
                row["protocol_variant"] = protocol_variant if protocol_variant else "default"
                _attach_primary_metric_status(
                    row,
                    primary_metric=primary_metric,
                    model_name=model_name,
                    target_vars=target_vars_for_score_effective,
                )
                leaderboard.append(row)
                effective_steps_per_model[model_name] = int(out.get("effective_steps", 0))
                if out.get("unet_contract_effective"):
                    unet_contract_samples.append(dict(out["unet_contract_effective"]))
                if out.get("fno_contract_effective"):
                    fno_contract_samples.append(dict(out["fno_contract_effective"]))
                if out.get("ffno_contract_effective"):
                    ffno_contract_samples.append(dict(out["ffno_contract_effective"]))
                if out.get("coord_mlp_contract_effective"):
                    coord_mlp_contract_samples.append(dict(out["coord_mlp_contract_effective"]))
                if out.get("deeponet_contract_effective"):
                    deeponet_contract_samples.append(dict(out["deeponet_contract_effective"]))
                if out.get("deeponet_pod_contract_effective"):
                    deeponet_pod_contract_samples.append(dict(out["deeponet_pod_contract_effective"]))
            else:
                split_rows: dict[str, dict[str, Any]] = {}
                split_steps: dict[str, int] = {}
                for split_name, split_def in [("interp", split_interp), ("extrap", split_extrap)]:
                    tr_i, va_i, te_i = self._indices_from_split(split_def, case_id_to_idx)
                    split_out = _run_single_split_model(
                        model_name=model_name,
                        model_idx=(model_idx * 10 + (1 if split_name == "interp" else 2)),
                        model_dir=model_dir / "eval_protocol" / split_name,
                        tr_idx=tr_i,
                        va_idx=va_i,
                        te_idx=te_i,
                    )
                    split_rows[split_name] = split_out["row"]
                    split_steps[split_name] = int(split_out.get("effective_steps", 0))
                    if split_out.get("unet_contract_effective"):
                        unet_contract_samples.append(dict(split_out["unet_contract_effective"]))
                    if split_out.get("fno_contract_effective"):
                        fno_contract_samples.append(dict(split_out["fno_contract_effective"]))
                    if split_out.get("ffno_contract_effective"):
                        ffno_contract_samples.append(dict(split_out["ffno_contract_effective"]))
                    if split_out.get("coord_mlp_contract_effective"):
                        coord_mlp_contract_samples.append(dict(split_out["coord_mlp_contract_effective"]))
                    if split_out.get("deeponet_contract_effective"):
                        deeponet_contract_samples.append(dict(split_out["deeponet_contract_effective"]))
                    if split_out.get("deeponet_pod_contract_effective"):
                        deeponet_pod_contract_samples.append(dict(split_out["deeponet_pod_contract_effective"]))
                    _release_torch_cuda_cache()
                row = dict(split_rows[primary_split])
                for var_name in target_vars_for_score_effective:
                    row[f"test_r2_{var_name}_plasma_interp"] = float(
                        split_rows["interp"].get(f"test_r2_{var_name}_plasma", 0.0)
                    )
                    row[f"test_r2_{var_name}_plasma_extrap"] = float(
                        split_rows["extrap"].get(f"test_r2_{var_name}_plasma", 0.0)
                    )
                    row[f"test_r2_{var_name}_plasma_dual"] = float(
                        interp_weight * row[f"test_r2_{var_name}_plasma_interp"]
                        + extrap_weight * row[f"test_r2_{var_name}_plasma_extrap"]
                    )
                row["score_total_interp"] = float(split_rows["interp"].get("score_total", 0.0))
                row["score_total_extrap"] = float(split_rows["extrap"].get("score_total", 0.0))
                row["score_total_dual"] = float(
                    interp_weight * row["score_total_interp"] + extrap_weight * row["score_total_extrap"]
                )
                row["score_nrmse_plasma_mean_interp"] = float(
                    split_rows["interp"].get("score_nrmse_plasma_mean", float("nan"))
                )
                row["score_nrmse_plasma_mean_extrap"] = float(
                    split_rows["extrap"].get("score_nrmse_plasma_mean", float("nan"))
                )
                row["score_nrmse_plasma_mean_dual"] = float(
                    interp_weight * row["score_nrmse_plasma_mean_interp"]
                    + extrap_weight * row["score_nrmse_plasma_mean_extrap"]
                )
                interp_r2_mean, interp_r2_valid, interp_r2_invalid = self._r2_plasma_mean_status(
                    split_rows["interp"],
                    target_vars=target_vars_for_score_effective,
                )
                extrap_r2_mean, extrap_r2_valid, extrap_r2_invalid = self._r2_plasma_mean_status(
                    split_rows["extrap"],
                    target_vars=target_vars_for_score_effective,
                )
                row["test_r2_plasma_mean_interp"] = float(interp_r2_mean)
                row["test_r2_plasma_mean_extrap"] = float(extrap_r2_mean)
                row["test_r2_plasma_mean_dual"] = float(
                    interp_weight * row["test_r2_plasma_mean_interp"] + extrap_weight * row["test_r2_plasma_mean_extrap"]
                )
                row["test_r2_plasma_mean_invalid_vars"] = "|".join(
                    sorted(set([*interp_r2_invalid, *extrap_r2_invalid]))
                )
                interp_test_cases = int(len(split_interp.get("test", [])))
                extrap_test_cases = int(len(split_extrap.get("test", [])))
                protocol_issues: list[str] = []
                if bool(interp_overlap_status.get("fallback_applied", False)):
                    protocol_issues.append(
                        f"interp_overlap_fallback:{interp_overlap_status.get('reason', 'unknown')}"
                    )
                if interp_test_cases < 3:
                    protocol_issues.append(f"interp_test_too_small:{interp_test_cases}")
                row["interp_test_cases"] = float(interp_test_cases)
                row["extrap_test_cases"] = float(extrap_test_cases)
                row["interp_overlap_fallback_applied"] = bool(interp_overlap_status.get("fallback_applied", False))
                row["interp_mode_effective"] = str(interp_overlap_status.get("applied_mode", interp_mode))
                row["eval_protocol_issue"] = "|".join(protocol_issues)
                row["eval_protocol_reliable"] = bool(len(protocol_issues) == 0)
                primary_metric_lower = str(primary_metric).strip().lower()
                primary_uses_interp = ("interp" in primary_metric_lower) or ("dual" in primary_metric_lower)
                row["primary_metric_protocol_reliable"] = bool((not primary_uses_interp) or len(protocol_issues) == 0)
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
            row_by_model = {str(r["model_id"]): r for r in leaderboard}
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
                            case_spatial_feature_pack=context.case_spatial_feature_pack,
                            static_spatial_feature_pack=context.static_spatial_feature_pack,
                            case_structure_feature_pack=context.case_structure_feature_pack,
                            coord_distance_transform_stats=dict(bundle.transforms.get("distance_transform_stats", {})),
                            structure_descriptor_pack=bundle.schemas.get("structure_descriptor_pack"),
                            latent_feature_pack=bundle.schemas.get("latent_feature_pack"),
                            input_mode_meta=effective_input_mode_meta,
                        )
                    )
                    pred_eval = dispatch["pred_eval"]
                    true_eval = dispatch["true_eval"]
                    rmse_ne = float(dispatch["metrics"].get(str(density_ne_key), 0.0)) if density_ne_key else 0.0
                    rmse_ni = float(dispatch["metrics"].get(str(density_ni_key), 0.0)) if density_ni_key else 0.0
                    rmse_te = float(dispatch["metrics"]["Te"])
                    rmse_phi = float(dispatch["metrics"]["phi"])
                    r2_vals = dict(dispatch.get("r2_scores", {}))
                    row = {
                        "test_rmse_ne": rmse_ne,
                        "test_rmse_ni": rmse_ni,
                        "test_rmse_Te": rmse_te,
                        "test_rmse_phi": rmse_phi,
                        "test_r2_ne": float(r2_vals.get(str(density_ne_key), 0.0)) if density_ne_key else 0.0,
                        "test_r2_ni": float(r2_vals.get(str(density_ni_key), 0.0)) if density_ni_key else 0.0,
                        "test_r2_Te": float(r2_vals.get("Te", 0.0)),
                        "test_r2_phi": float(r2_vals.get("phi", 0.0)),
                        "test_poisson_phi": float(poisson_residual_loss(pred_eval["phi"][:, 0])),
                    }
                    if metric_mask is not None:
                        for name in [v for v in [density_ne_key, density_ni_key, "Te", "phi"] if v is not None]:
                            if name in true_eval and name in pred_eval:
                                row[f"test_rmse_{name}_plasma"] = float(
                                    rmse_masked(true_eval[name], pred_eval[name], metric_mask)
                                )
                                row[f"test_r2_{name}_plasma"] = float(
                                    r2_masked(true_eval[name], pred_eval[name], metric_mask)
                                )
                        if density_ne_key:
                            row["test_rmse_ne_plasma"] = float(row.get(f"test_rmse_{density_ne_key}_plasma", 0.0))
                            row["test_r2_ne_plasma"] = float(row.get(f"test_r2_{density_ne_key}_plasma", 0.0))
                        if density_ni_key:
                            row["test_rmse_ni_plasma"] = float(row.get(f"test_rmse_{density_ni_key}_plasma", 0.0))
                            row["test_r2_ni_plasma"] = float(row.get(f"test_r2_{density_ni_key}_plasma", 0.0))
                    cv_rows[model_name].append(row)
            for model_name, rows in cv_rows.items():
                if len(rows) == 0:
                    continue
                out_row = row_by_model.get(model_name)
                if out_row is None:
                    continue
                numeric_keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
                for key in numeric_keys:
                    vals = np.asarray([float(r.get(key, 0.0)) for r in rows], dtype=np.float64)
                    out_row[f"cv_mean_{key}"] = float(np.mean(vals))
                    out_row[f"cv_std_{key}"] = float(np.std(vals))
            resolved["cv"] = {
                "enabled": True,
                "n_folds": int(cv_cfg.get("n_folds", 5)),
                "group_key": group_key,
                "seed": int(cv_cfg.get("seed", global_seed)),
            }
            self._persist_resolved(split=context.split, resolved=resolved)

        header = ["model_id"]
        for var_name in target_vars_for_score_effective:
            header.extend(
                [
                    f"test_rmse_{var_name}",
                    f"test_rmse_{var_name}_plasma",
                    f"test_r2_{var_name}",
                    f"test_r2_{var_name}_plasma",
                    f"test_r2_{var_name}_plasma_interp",
                    f"test_r2_{var_name}_plasma_extrap",
                    f"test_r2_{var_name}_plasma_dual",
                ]
            )
        header.extend(
            [
                "test_poisson_phi",
                "qoi_uniformity",
                "qoi_boundary_gamma_uniformity",
                "single_poisson_residual",
                "single_poisson_residual_map_l2",
                "single_boundary_operator_proxy_loss",
                "single_boundary_residual_map_l2",
                "opt_best_uniformity",
                "score_rmse_plasma_mean",
                "score_nrmse_plasma_mean",
                "score_r2_plasma_mean",
                "score_boundary_penalty",
                "score_total",
                "score_nrmse_plasma_mean_interp",
                "score_nrmse_plasma_mean_extrap",
                "score_nrmse_plasma_mean_dual",
                "score_total_interp",
                "score_total_extrap",
                "score_total_dual",
                "test_r2_plasma_mean_interp",
                "test_r2_plasma_mean_extrap",
                "test_r2_plasma_mean_dual",
                "interp_test_cases",
                "extrap_test_cases",
                "interp_overlap_fallback_applied",
                "interp_mode_effective",
                "eval_protocol_reliable",
                "eval_protocol_issue",
                "primary_metric_protocol_reliable",
                "primary_metric",
                "primary_metric_value",
                "target_metrics_valid",
                "target_metrics_invalid_vars",
                "primary_metric_reliable",
            ]
        )
        extra_keys = sorted({k for row in leaderboard for k in row.keys() if k not in header})
        full_header = header + extra_keys
        guardrail_warnings.extend(
            self._evaluate_guardrails(
                phase="post",
                train_cfg=train_cfg,
                model_names=profile_lock["models"],
                effective_steps_per_model=effective_steps_per_model,
            )
        )
        resolved["guardrail_warnings"] = guardrail_warnings
        resolved["effective_steps_per_model"] = effective_steps_per_model
        resolved["comparison_contract"] = {
            "global_reference_mode": "frozen"
            if eval_protocol_scope in _FROZEN_REFERENCE_SCOPES
            else "common",
            "active_model_scope": eval_protocol_scope,
            "target_family_mode": target_family_for_score_raw,
        }
        guard_cfg = dict(self.benchmark_cfg.get("guardrails", {}))
        checks = dict(guard_cfg.get("checks", {}))
        guard_mode = str(guard_cfg.get("mode", "warn")).strip().lower()
        emit_unet_contract = bool(unet_contract_samples) or eval_protocol_scope not in _UNET_CONTRACT_OPTIONAL_SCOPES
        if emit_unet_contract:
            unet_contract_effective = self._aggregate_unet_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                model_key=str(active_unet_model or "unet"),
                unet_contract_samples=unet_contract_samples,
            )
            resolved["unet_contract_effective"] = unet_contract_effective
            resolved["unet_target_family_effective"] = str(unet_contract_effective.get("target_family_effective", "allvars"))
            resolved["unet_target_vars_effective"] = list(unet_contract_effective.get("target_vars_effective", y_vars))
            resolved["unet_backend_effective"] = str(unet_contract_effective.get("unet_backend_effective", "numpy"))
            resolved["unet_input_channels_effective"] = list(
                unet_contract_effective.get("unet_input_channels_effective", ["x", "y"])
            )
            resolved["unet_selection_mode_effective"] = str(
                unet_contract_effective.get("unet_selection_mode_effective", "last")
            )
            resolved["unet_spatial_consistency_effective"] = bool(
                unet_contract_effective.get("unet_spatial_consistency_effective", False)
            )
            resolved["boundary_bonus_weight_effective"] = float(
                unet_contract_effective.get("boundary_bonus_weight_effective", 0.0)
            )
            resolved["unet_optimizer_effective"] = dict(unet_contract_effective.get("unet_optimizer_effective", {}))
            resolved["unet_output_heads_mode_effective"] = str(
                unet_contract_effective.get("unet_output_heads_mode_effective", "shared")
            )
            resolved["unet_feature_contract_effective"] = dict(
                unet_contract_effective.get("unet_feature_contract_effective", {})
            )
        if fno_contract_samples:
            fno_contract_effective = self._aggregate_fno_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                fno_contract_samples=fno_contract_samples,
            )
            self._emit_spectral_contract_resolved(
                resolved=resolved,
                prefix="fno",
                contract_effective=fno_contract_effective,
                default_target_vars=y_vars,
            )
        if ffno_contract_samples:
            ffno_contract_effective = self._aggregate_ffno_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                ffno_contract_samples=ffno_contract_samples,
            )
            self._emit_spectral_contract_resolved(
                resolved=resolved,
                prefix="ffno",
                contract_effective=ffno_contract_effective,
                default_target_vars=y_vars,
            )
        if coord_mlp_contract_samples:
            resolved["coord_mlp_contract_effective"] = self._aggregate_coord_mlp_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                coord_mlp_contract_samples=coord_mlp_contract_samples,
            )
        if deeponet_pod_contract_samples:
            deeponet_pod_contract_effective = self._aggregate_deeponet_pod_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                deeponet_pod_contract_samples=deeponet_pod_contract_samples,
            )
            resolved["deeponet_pod_contract_effective"] = deeponet_pod_contract_effective
        if deeponet_contract_samples:
            deeponet_contract_effective = self._aggregate_deeponet_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                deeponet_contract_samples=deeponet_contract_samples,
            )
            resolved["deeponet_contract_effective"] = deeponet_contract_effective
            resolved["deeponet_target_family_effective"] = str(
                deeponet_contract_effective.get("target_family_effective", "allvars")
            )
            resolved["deeponet_target_vars_effective"] = list(
                deeponet_contract_effective.get("target_vars_effective", y_vars)
            )
            resolved["deeponet_selection_mode_effective"] = str(
                deeponet_contract_effective.get("selection_mode_effective", "last")
            )
            resolved["deeponet_feature_contract_effective"] = dict(
                deeponet_contract_effective.get("feature_contract_effective", {})
            )
            resolved["deeponet_feature_source_effective"] = str(
                dict(deeponet_contract_effective.get("feature_contract_effective", {})).get(
                    "feature_source_effective", "unknown"
                )
            )
            resolved["deeponet_distance_transform_effective"] = dict(
                deeponet_contract_effective.get("distance_transform_effective", {})
            )
            resolved["deeponet_operator_mode_effective"] = str(
                deeponet_contract_effective.get("operator_mode_effective", "pde_coupled")
            )
            resolved["deeponet_branch_mode_effective"] = str(
                deeponet_contract_effective.get("branch_mode_effective", "moments")
            )
            resolved["deeponet_trunk_cond_modulation_effective"] = str(
                deeponet_contract_effective.get("trunk_cond_modulation_effective", "none")
            )
            resolved["deeponet_trunk_cond_mod_hidden_effective"] = int(
                deeponet_contract_effective.get("trunk_cond_mod_hidden_effective", 64)
            )
            resolved["deeponet_residual_head_enabled_effective"] = bool(
                deeponet_contract_effective.get("residual_head_enabled_effective", False)
            )
            resolved["deeponet_residual_head_hidden_dim_effective"] = int(
                deeponet_contract_effective.get("residual_head_hidden_dim_effective", 64)
            )
            resolved["deeponet_residual_head_scale_init_effective"] = float(
                deeponet_contract_effective.get("residual_head_scale_init_effective", 0.0)
            )
            resolved["deeponet_residual_head_gain_mode_effective"] = str(
                deeponet_contract_effective.get("residual_head_gain_mode_effective", "learned")
            )
            resolved["deeponet_residual_head_gain_value_effective"] = float(
                deeponet_contract_effective.get("residual_head_gain_value_effective", 1.0)
            )
            resolved["deeponet_latent_layer_norm_effective"] = bool(
                deeponet_contract_effective.get("latent_layer_norm_effective", False)
            )
            resolved["deeponet_output_path_mode_effective"] = str(
                deeponet_contract_effective.get("output_path_mode_effective", "dot")
            )
            resolved["deeponet_output_path_dot_skip_effective"] = float(
                deeponet_contract_effective.get("output_path_dot_skip_effective", 0.25)
            )
            resolved["deeponet_output_path_dot_skip_mode_effective"] = str(
                deeponet_contract_effective.get("output_path_dot_skip_mode_effective", "fixed")
            )
            resolved["deeponet_output_path_fused_hidden_dim_effective"] = int(
                deeponet_contract_effective.get("output_path_fused_hidden_dim_effective", 96)
            )
            resolved["deeponet_output_path_global_local_enabled_effective"] = bool(
                deeponet_contract_effective.get("output_path_global_local_enabled_effective", False)
            )
            resolved["deeponet_output_path_global_hidden_dim_effective"] = int(
                deeponet_contract_effective.get("output_path_global_hidden_dim_effective", 64)
            )
            resolved["deeponet_missing_geom_feature_policy_effective"] = str(
                deeponet_contract_effective.get("missing_geom_feature_policy_effective", "warn_zero")
            )
            resolved["deeponet_trunk_fourier_mode_effective"] = str(
                deeponet_contract_effective.get("trunk_fourier_mode_effective", "legacy")
            )
            resolved["deeponet_trunk_fourier_n_freq_effective"] = int(
                deeponet_contract_effective.get("trunk_fourier_n_freq_effective", 1)
            )
            resolved["deeponet_optimizer_effective"] = dict(
                deeponet_contract_effective.get("optimizer_effective", {})
            )
        resolved["spatial_error_audit_effective"] = bool(
            self.benchmark_cfg.get("eval", {}).get("spatial_error_audit", {}).get("enabled", True)
        )
        resolved["spatial_error_boundary_type_breakdown_effective"] = bool(
            self.benchmark_cfg.get("eval", {}).get("spatial_error_audit", {}).get("boundary_type_breakdown", False)
        )
        resolved["spatial_error_region_bands_effective"] = dict(region_bands_effective)
        self._persist_resolved(split=context.split, resolved=resolved, include_manifest=True)
        self.store.save_csv("leaderboard.csv", full_header, [[row.get(h, 0.0) for h in full_header] for row in leaderboard])
        return BenchmarkResult(leaderboard=leaderboard, leaderboard_path=self.output_root / "leaderboard.csv")

    def run_sweep(self) -> SweepResult:
        sweep_cfg = self.benchmark_cfg.get("sweep", {})
        params_grid = sweep_cfg.get("params", {})
        if not isinstance(params_grid, dict) or not params_grid:
            raise ValueError("benchmark.sweep.params must be a non-empty dict")

        objective_cfg = sweep_cfg.get("objective", {})
        objective_model = str(objective_cfg.get("model_id", "global_mlp"))
        objective_metric = str(objective_cfg.get("metric", "auto_primary")).strip()
        objective_mode = str(objective_cfg.get("mode", "min"))
        if objective_mode not in {"min", "max"}:
            raise ValueError("benchmark.sweep.objective.mode must be 'min' or 'max'")

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.store.save_json(
            "sweep/resolved_sweep.json",
            {
                "params": params_grid,
                "objective": {
                    "model_id": objective_model,
                    "metric": objective_metric,
                    "mode": objective_mode,
                },
            },
        )

        keys = list(params_grid.keys())
        values = [params_grid[k] for k in keys]
        if any(not isinstance(v, list) or not v for v in values):
            raise ValueError("Each benchmark.sweep.params entry must be a non-empty list")

        trials: list[dict[str, Any]] = []
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
            score_key = "primary_metric_value" if objective_metric == "auto_primary" else objective_metric
            score = float(trial_row[score_key])
            resolved = ArtifactStore(trial_output).load_json("resolved_benchmark.json")
            trials.append(
                {
                    "trial_id": f"trial_{trial_idx:03d}",
                    "output_dir": str(trial_output),
                    "objective_model_id": objective_model,
                    "objective_metric": score_key,
                    "objective_score": score,
                    "lock_hash": resolved["artifact_hashes"]["lock_hash"],
                    **trial_values,
                }
            )

        reverse = objective_mode == "max"
        trials_sorted = sorted(trials, key=lambda r: float(r["objective_score"]), reverse=reverse)
        best = trials_sorted[0]
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
    def _load_split_json(path: Path, *, fallback: dict[str, list[str]]) -> dict[str, list[str]]:
        if not path.exists():
            return {
                "train": [str(v) for v in fallback.get("train", [])],
                "val": [str(v) for v in fallback.get("val", [])],
                "test": [str(v) for v in fallback.get("test", [])],
            }
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
        if include_manifest:
            self.store.save_json("manifest.json", {"split": split, "resolved": resolved})
        self.store.save_json("resolved_benchmark.json", resolved)

    @staticmethod
    def _first_str(samples: list[dict[str, Any]], key: str) -> str | None:
        for sample in samples:
            val = str(sample.get(key, ""))
            if val:
                return val
        return None

    @staticmethod
    def _first_dict(samples: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        for sample in samples:
            val = sample.get(key)
            if isinstance(val, dict) and len(val) > 0:
                return dict(val)
        return None

    @staticmethod
    def _first_list(samples: list[dict[str, Any]], key: str) -> list[Any] | None:
        for sample in samples:
            val = sample.get(key)
            if isinstance(val, list) and len(val) > 0:
                return list(val)
        return None

    @staticmethod
    def _mean_field(samples: list[dict[str, Any]], key: str) -> float | None:
        vals = [float(s.get(key, 0.0)) for s in samples if isinstance(s, dict)]
        if len(vals) == 0:
            return None
        return float(np.mean(np.asarray(vals, dtype=np.float64)))

    @staticmethod
    def _any_bool(samples: list[dict[str, Any]], key: str) -> bool:
        return bool(any(bool(s.get(key, False)) for s in samples if isinstance(s, dict)))

    def _aggregate_unet_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        model_key: str,
        unet_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        unet_cfg = dict(train_cfg.get(str(model_key), {}))
        unet_family = str(unet_cfg.get("target_family", "allvars")).strip().lower()
        if unet_family == "field":
            target_vars_default = [v for v in ["Te", "phi"] if v in set(y_vars)]
        elif unet_family == "logpair":
            target_vars_default = resolve_family_vars(
                family="logpair",
                available=y_vars,
                prefer_linear=True,
            )
        else:
            target_vars_default = resolve_allvars_order(list(y_vars), prefer_linear=True)
        unet_input_cfg = dict(unet_cfg.get("input_features", {}))
        raw_feat = unet_input_cfg.get("features", ["x", "y"])
        feat_list = [str(v) for v in raw_feat] if isinstance(raw_feat, list) and raw_feat else ["x", "y"]
        unet_selection_cfg = dict(unet_cfg.get("selection", {}))
        unet_optimizer_cfg = dict(unet_cfg.get("optimizer", {}))
        unet_model_cfg = dict(unet_cfg.get("model_cfg", {}))
        unet_operator_cfg = dict(unet_model_cfg.get("unet_operator_v2_cfg", {}))
        sup_cfg = dict(dict(train_cfg.get("loss", {})).get("supervised", {}))
        bt_cfg = dict(sup_cfg.get("boundary_type_weighting", {}))
        bp_cfg = dict(sup_cfg.get("boundary_profile_weighting", {}))
        rb_cfg = dict(sup_cfg.get("region_balance", {}))
        rb_schedule_cfg = dict(rb_cfg.get("schedule", {}))
        out = {
            "target_vars_effective": target_vars_default,
            "target_family_effective": unet_family,
            "merge_role": "single",
            "unet_backend_effective": str(unet_model_cfg.get("backend", "numpy")).strip().lower(),
            "unet_input_channels_effective": feat_list,
            "unet_selection_mode_effective": str(unet_selection_cfg.get("mode", "last")).strip().lower(),
            "selection_weights_effective": dict(unet_selection_cfg.get("weights", {})),
            "boundary_bonus_weight_effective": float(unet_selection_cfg.get("boundary_bonus_weight", 0.0)),
            "unet_optimizer_effective": {
                "type": str(unet_optimizer_cfg.get("type", "adamw")).strip().lower(),
                "lr": float(unet_optimizer_cfg.get("lr", unet_cfg.get("lr", train_cfg.get("lr", 1e-3)))),
                "weight_decay": float(unet_optimizer_cfg.get("weight_decay", 0.0)),
                "schedule": str(unet_optimizer_cfg.get("schedule", "none")).strip().lower(),
                "warmup_epochs": int(max(int(unet_optimizer_cfg.get("warmup_epochs", 0)), 0)),
            },
            "unet_output_heads_mode_effective": str(
                dict(unet_model_cfg.get("output_heads", {})).get(
                    "mode",
                    unet_operator_cfg.get("head_mode", "shared"),
                )
            ).strip().lower(),
            "unet_feature_contract_effective": {
                "input_features_mode": str(unet_input_cfg.get("mode", "geom_feature_pack")).strip().lower(),
                "input_feature_channels": feat_list,
                "upsample_mode": str(
                    dict(unet_model_cfg.get("conv_cfg", {})).get(
                        "upsample_mode",
                        unet_operator_cfg.get("upsample", unet_model_cfg.get("upsample_mode", "deconv")),
                    )
                ).strip().lower(),
                "distance_transform_mode": str(
                    dict(unet_input_cfg.get("distance_transform", {})).get("mode", "raw")
                ).strip().lower(),
            },
            "boundary_type_weighting_effective": {
                "enabled": bool(bt_cfg.get("enabled", False)),
                "vars": [str(v) for v in bt_cfg.get("vars", ["Te", "phi"])],
                "band_px": float(bt_cfg.get("band_px", 2.0)),
                "weights": {
                    "interface": float(dict(bt_cfg.get("weights", {})).get("interface", 1.0)),
                    "bc_dir": float(dict(bt_cfg.get("weights", {})).get("bc_dir", 1.0)),
                    "wafer": float(dict(bt_cfg.get("weights", {})).get("wafer", 1.0)),
                },
            },
            "boundary_profile_weighting_effective": {
                "enabled": bool(bp_cfg.get("enabled", False)),
                "vars": [str(v) for v in bp_cfg.get("vars", ["Te", "phi"])],
                "band_px": float(bp_cfg.get("band_px", 2.0)),
                "mode": str(bp_cfg.get("mode", "exp_decay")).strip().lower(),
                "alpha": float(bp_cfg.get("alpha", 0.35)),
                "tau_px": float(bp_cfg.get("tau_px", 0.8)),
            },
            "region_balance_bands_effective": {
                "enabled": bool(rb_cfg.get("enabled", False)),
                "boundary_in_px": float(rb_cfg.get("boundary_in_px", 2.0)),
                "mid_plasma_px": float(rb_cfg.get("mid_plasma_px", rb_cfg.get("deep_plasma_px", 10.0))),
                "deep_plasma_px": float(rb_cfg.get("deep_plasma_px", 10.0)),
                "weight_boundary_in": float(rb_cfg.get("weight_boundary_in", 0.6)),
                "weight_plasma_mid": float(rb_cfg.get("weight_plasma_mid", 0.0)),
                "weight_deep_plasma": float(rb_cfg.get("weight_deep_plasma", 0.4)),
            },
            "region_balance_schedule_effective": {
                "enabled": bool(rb_schedule_cfg.get("enabled", False)),
                "warmup_epochs": int(max(int(rb_schedule_cfg.get("warmup_epochs", 0)), 0)),
                "ramp_epochs": int(max(int(rb_schedule_cfg.get("ramp_epochs", 0)), 0)),
            },
            "density_head_loss_weights_effective": {
                "enabled": bool(dict(unet_model_cfg.get("output_heads", {}).get("loss_weights", {})).get("enabled", False)),
                "density": float(dict(unet_model_cfg.get("output_heads", {}).get("loss_weights", {})).get("density", 1.0)),
                "field": float(dict(unet_model_cfg.get("output_heads", {}).get("loss_weights", {})).get("field", 1.0)),
            },
            "unet_operator_v2_contract_effective": {
                "enabled": str(model_key) == "unet_operator_v2",
                "depth": int(unet_operator_cfg.get("depth", 0)),
                "width": int(unet_operator_cfg.get("width", 0)),
                "blocks_per_level": int(unet_operator_cfg.get("blocks_per_level", 0)),
                "downsample": str(unet_operator_cfg.get("downsample", "")).strip().lower(),
                "use_film": bool(unet_operator_cfg.get("use_film", False)),
                "head_mode": str(unet_operator_cfg.get("head_mode", "shared")).strip().lower(),
            },
            "unet_spatial_consistency_effective": bool(dict(sup_cfg.get("spatial_consistency", {})).get("enabled", False)),
        }
        v = self._first_str(unet_contract_samples, "target_family_effective")
        if v is not None:
            out["target_family_effective"] = v
        v = self._first_str(unet_contract_samples, "merge_role")
        if v is not None:
            out["merge_role"] = v
        v = self._first_str(unet_contract_samples, "unet_backend_effective")
        if v is not None:
            out["unet_backend_effective"] = v
        l = self._first_list(unet_contract_samples, "unet_input_channels_effective")
        if l is not None:
            out["unet_input_channels_effective"] = [str(x) for x in l]
        l = self._first_list(unet_contract_samples, "target_vars_effective")
        if l is not None:
            out["target_vars_effective"] = [str(x) for x in l]
        v = self._first_str(unet_contract_samples, "unet_selection_mode_effective")
        if v is not None:
            out["unet_selection_mode_effective"] = v
        d = self._first_dict(unet_contract_samples, "selection_weights_effective")
        if d is not None:
            out["selection_weights_effective"] = d
        for sample in unet_contract_samples:
            if isinstance(sample, dict) and "boundary_bonus_weight_effective" in sample:
                out["boundary_bonus_weight_effective"] = float(sample["boundary_bonus_weight_effective"])
                break
        d = self._first_dict(unet_contract_samples, "unet_optimizer_effective")
        if d is not None:
            merged_optimizer = dict(out.get("unet_optimizer_effective", {}))
            merged_optimizer.update(dict(d))
            out["unet_optimizer_effective"] = merged_optimizer
        if isinstance(out.get("unet_optimizer_effective"), dict):
            if "weight_decay" not in out["unet_optimizer_effective"]:
                out["unet_optimizer_effective"]["weight_decay"] = float(
                    dict(unet_cfg.get("optimizer", {})).get("weight_decay", 0.0)
                )
        v = self._first_str(unet_contract_samples, "unet_output_heads_mode_effective")
        if v is not None:
            out["unet_output_heads_mode_effective"] = v
        d = self._first_dict(unet_contract_samples, "unet_feature_contract_effective")
        if d is not None:
            out["unet_feature_contract_effective"] = d
        d = self._first_dict(unet_contract_samples, "boundary_type_weighting_effective")
        if d is not None:
            out["boundary_type_weighting_effective"] = d
        d = self._first_dict(unet_contract_samples, "boundary_profile_weighting_effective")
        if d is not None:
            out["boundary_profile_weighting_effective"] = d
        d = self._first_dict(unet_contract_samples, "region_balance_bands_effective")
        if d is not None:
            out["region_balance_bands_effective"] = d
        d = self._first_dict(unet_contract_samples, "region_balance_schedule_effective")
        if d is not None:
            out["region_balance_schedule_effective"] = d
        d = self._first_dict(unet_contract_samples, "density_head_loss_weights_effective")
        if d is not None:
            out["density_head_loss_weights_effective"] = d
        d = self._first_dict(unet_contract_samples, "unet_operator_v2_contract_effective")
        if d is not None:
            out["unet_operator_v2_contract_effective"] = d
        for sample in unet_contract_samples:
            if isinstance(sample, dict) and "unet_spatial_consistency_effective" in sample:
                out["unet_spatial_consistency_effective"] = bool(sample.get("unet_spatial_consistency_effective", False))
                break
        # Keep resolved_benchmark focused on effective contracts only.
        for key in [
            "boundary_type_weighting_effective",
            "boundary_profile_weighting_effective",
            "region_balance_bands_effective",
            "region_balance_schedule_effective",
            "density_head_loss_weights_effective",
            "unet_operator_v2_contract_effective",
        ]:
            payload = out.get(key)
            if isinstance(payload, dict) and not bool(payload.get("enabled", False)):
                out.pop(key, None)
        return out

    def _aggregate_spectral_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        model_key: str,
        prefix: str,
        contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        spectral_cfg = dict(train_cfg.get(model_key, {}))
        spectral_family = str(spectral_cfg.get("target_family", "allvars")).strip().lower()
        if spectral_family == "field":
            target_vars_default = [v for v in ["Te", "phi"] if v in set(y_vars)]
        elif spectral_family == "logpair":
            target_vars_default = resolve_family_vars(
                family="logpair",
                available=y_vars,
                prefer_linear=True,
            )
        else:
            target_vars_default = resolve_allvars_order(list(y_vars), prefer_linear=True)
        input_cfg = dict(spectral_cfg.get("input_features", {}))
        raw_feat = input_cfg.get("features", ["x", "y"])
        feat_list = [str(v) for v in raw_feat] if isinstance(raw_feat, list) and raw_feat else ["x", "y"]
        model_cfg = dict(spectral_cfg.get("model_cfg", {}))
        factorized_cfg = dict(dict(model_cfg.get("spectral_cfg", {})).get("factorized_cfg", {}))
        spectral_model_cfg = dict(model_cfg.get("spectral_cfg", {}))
        selection_cfg = dict(spectral_cfg.get("selection", {}))
        sup_cfg = dict(dict(train_cfg.get("loss", {})).get("supervised", {}))
        out = {
            "target_vars_effective": target_vars_default,
            "target_family_effective": spectral_family,
            "input_features_mode": str(input_cfg.get("mode", "geom_feature_pack")).strip().lower(),
            "input_feature_channels": feat_list,
            f"{prefix}_backend_effective": "torch",
            f"{prefix}_input_channels_effective": feat_list,
            f"{prefix}_input_features_mode_effective": str(input_cfg.get("mode", "geom_feature_pack")).strip().lower(),
            f"{prefix}_selection_mode_effective": str(selection_cfg.get("mode", "last")).strip().lower(),
            f"{prefix}_optimizer_effective": dict(spectral_cfg.get("optimizer", {})),
            f"{prefix}_feature_contract_effective": {
                "input_features_mode": str(input_cfg.get("mode", "geom_feature_pack")).strip().lower(),
                "input_feature_channels": feat_list,
                "distance_transform_mode": str(
                    dict(input_cfg.get("distance_transform", {})).get("mode", "raw")
                ).strip().lower(),
            },
            f"{prefix}_spectral_contract_effective": {
                "n_modes": int(model_cfg.get("n_modes", model_cfg.get("fno_n_modes", 2))),
                "dealias_ratio": float(spectral_model_cfg.get("dealias_ratio", 1.0)),
                "taper_alpha": float(spectral_model_cfg.get("taper_alpha", 0.0)),
                "skip_filter": str(spectral_model_cfg.get("skip_filter", "none")).strip().lower(),
            },
            f"{prefix}_spatial_consistency_effective": bool(
                dict(sup_cfg.get("spatial_consistency", {})).get("enabled", False)
            ),
        }
        if prefix == "ffno":
            out[f"{prefix}_spectral_contract_effective"]["factorized_cfg"] = {
                "enabled": bool(factorized_cfg.get("enabled", True)),
                "mode": str(factorized_cfg.get("mode", "separable_1d")).strip().lower(),
                "share_weights": bool(factorized_cfg.get("share_weights", False)),
            }
            local_skip_cfg = dict(spectral_model_cfg.get("local_skip_cfg", {}))
            out[f"{prefix}_spectral_contract_effective"]["local_skip_cfg"] = {
                "enabled": bool(local_skip_cfg.get("enabled", False)),
                "init_scale": float(local_skip_cfg.get("init_scale", 0.0)),
            }
            axis_mix_cfg = dict(spectral_model_cfg.get("axis_mix_cfg", {}))
            out[f"{prefix}_spectral_contract_effective"]["axis_mix_cfg"] = {
                "enabled": bool(axis_mix_cfg.get("enabled", False)),
                "init_h": float(axis_mix_cfg.get("init_h", 1.0)),
                "init_w": float(axis_mix_cfg.get("init_w", 1.0)),
            }
        l = self._first_list(contract_samples, "target_vars_effective")
        if l is not None:
            out["target_vars_effective"] = [str(x) for x in l]
        v = self._first_str(contract_samples, "target_family_effective")
        if v is not None:
            out["target_family_effective"] = v
        v = self._first_str(contract_samples, "input_features_mode")
        if v is not None:
            out["input_features_mode"] = v
        l = self._first_list(contract_samples, "input_feature_channels")
        if l is not None:
            out["input_feature_channels"] = [str(x) for x in l]
        v = self._first_str(contract_samples, f"{prefix}_backend_effective")
        if v is not None:
            out[f"{prefix}_backend_effective"] = v
        l = self._first_list(contract_samples, f"{prefix}_input_channels_effective")
        if l is not None:
            out[f"{prefix}_input_channels_effective"] = [str(x) for x in l]
        v = self._first_str(contract_samples, f"{prefix}_input_features_mode_effective")
        if v is not None:
            out[f"{prefix}_input_features_mode_effective"] = v
        v = self._first_str(contract_samples, f"{prefix}_selection_mode_effective")
        if v is not None:
            out[f"{prefix}_selection_mode_effective"] = v
        d = self._first_dict(contract_samples, f"{prefix}_optimizer_effective")
        if d is not None:
            out[f"{prefix}_optimizer_effective"] = d
        d = self._first_dict(contract_samples, f"{prefix}_feature_contract_effective")
        if d is not None:
            out[f"{prefix}_feature_contract_effective"] = d
        d = self._first_dict(contract_samples, f"{prefix}_spectral_contract_effective")
        if d is not None:
            out[f"{prefix}_spectral_contract_effective"] = d
        for sample in contract_samples:
            if isinstance(sample, dict) and f"{prefix}_spatial_consistency_effective" in sample:
                out[f"{prefix}_spatial_consistency_effective"] = bool(sample.get(f"{prefix}_spatial_consistency_effective", False))
                break
        return out

    def _aggregate_fno_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        fno_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._aggregate_spectral_contract_effective(
            train_cfg=train_cfg,
            y_vars=y_vars,
            model_key="fno",
            prefix="fno",
            contract_samples=fno_contract_samples,
        )

    def _aggregate_ffno_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        ffno_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._aggregate_spectral_contract_effective(
            train_cfg=train_cfg,
            y_vars=y_vars,
            model_key="ffno",
            prefix="ffno",
            contract_samples=ffno_contract_samples,
        )

    @staticmethod
    def _emit_spectral_contract_resolved(
        *,
        resolved: dict[str, Any],
        prefix: str,
        contract_effective: dict[str, Any],
        default_target_vars: list[str],
    ) -> None:
        resolved[f"{prefix}_contract_effective"] = dict(contract_effective)
        resolved[f"{prefix}_target_family_effective"] = str(contract_effective.get("target_family_effective", "allvars"))
        resolved[f"{prefix}_target_vars_effective"] = list(contract_effective.get("target_vars_effective", default_target_vars))
        resolved[f"{prefix}_backend_effective"] = str(contract_effective.get(f"{prefix}_backend_effective", "torch"))
        resolved[f"{prefix}_input_channels_effective"] = list(contract_effective.get(f"{prefix}_input_channels_effective", ["x", "y"]))
        resolved[f"{prefix}_selection_mode_effective"] = str(contract_effective.get(f"{prefix}_selection_mode_effective", "last"))
        resolved[f"{prefix}_spatial_consistency_effective"] = bool(
            contract_effective.get(f"{prefix}_spatial_consistency_effective", False)
        )
        resolved[f"{prefix}_optimizer_effective"] = dict(contract_effective.get(f"{prefix}_optimizer_effective", {}))
        resolved[f"{prefix}_feature_contract_effective"] = dict(
            contract_effective.get(f"{prefix}_feature_contract_effective", {})
        )
        resolved[f"{prefix}_spectral_contract_effective"] = dict(
            contract_effective.get(f"{prefix}_spectral_contract_effective", {})
        )

    def _aggregate_coord_mlp_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        coord_mlp_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        coord_model_key = resolve_single_family_model(
            model_names=[name for name in COORD_MLP_FAMILY_MODELS if bool(dict(train_cfg.get(name, {})))],
            family=COORD_MLP_FAMILY_MODELS,
        ) or "coord_mlp_fourier"
        cfg = dict(train_cfg.get(coord_model_key, {}))
        input_features_cfg = dict(cfg.get("input_features", {}))
        if coord_model_key == "coord_mlp_pod_residual":
            model_cfg = normalize_coord_mlp_pod_residual_cfg(dict(cfg.get("model_cfg", {})))
        else:
            _, model_cfg = _normalize_coord_mlp_model_cfg(
                model_name=str(coord_model_key),
                raw_cfg=dict(cfg.get("model_cfg", {})),
            )
        out = {
            "model_type_effective": str(coord_model_key),
            "backend_effective": "torch",
            "target_family_effective": str(cfg.get("target_family", "allvars")).strip().lower(),
            "target_vars_effective": list(cfg.get("target_vars", y_vars)),
            "input_features_mode": str(input_features_cfg.get("mode", "geom_feature_pack")).strip().lower(),
            "input_feature_channels": list(
                input_features_cfg.get("features", ["x", "y", "mask_plasma", "distance_signed", "distance_any"])
            ),
            "require_pack_effective": str(input_features_cfg.get("require_pack", "warn")).strip().lower(),
            "selection_mode_effective": str(dict(cfg.get("selection", {})).get("mode", "last")).strip().lower(),
            "selection_weights_effective": dict(dict(cfg.get("selection", {})).get("weights", {})),
            "embedding": dict(model_cfg.get("embedding", {})),
            "siren": dict(model_cfg.get("siren", {})),
        }
        if coord_model_key == "coord_mlp_pod_residual":
            out["pod_residual"] = {
                "basis": dict(model_cfg.get("basis", {})),
                "coeff_loss_weight": float(model_cfg.get("coeff_loss_weight", 0.0)),
                "residual_scale_init": float(model_cfg.get("residual_scale_init", 0.0)),
                "point_encoder": dict(model_cfg.get("point_encoder", {})),
            }
        sample = coord_mlp_contract_samples[0] if coord_mlp_contract_samples else {}
        for key in (
            "model_type_effective",
            "backend_effective",
            "target_family_effective",
            "target_vars_effective",
            "input_features_mode",
            "input_feature_channels",
            "feature_source_effective",
            "require_pack_effective",
            "selection_mode_effective",
            "selection_weights_effective",
            "embedding",
            "siren",
            "pod_residual",
        ):
            if key in sample:
                out[key] = sample[key]
        return out

    def _aggregate_deeponet_pod_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        deeponet_pod_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sample = deeponet_pod_contract_samples[0] if deeponet_pod_contract_samples else {}
        model_type = str(sample.get("model_type_effective", "deeponet_pod")).strip().lower() or "deeponet_pod"
        cfg = dict(train_cfg.get(model_type, train_cfg.get("deeponet_pod", {})))
        model_cfg = normalize_pod_deeponet_model_cfg(
            dict(cfg.get("model_cfg", {})),
            model_type=model_type,
        )
        basis_cfg = dict(model_cfg.get("basis", {}))
        out = {
            "model_type_effective": model_type,
            "target_family_effective": str(cfg.get("target_family", "allvars")).strip().lower(),
            "target_vars_effective": list(cfg.get("target_vars", y_vars)),
            "basis_rank_by_var": {str(v): int(basis_cfg.get("rank", 32)) for v in list(cfg.get("target_vars", y_vars))},
            "basis_fit_scope_effective": str(basis_cfg.get("fit_scope", "train_only")).strip().lower(),
            "basis_center_effective": bool(basis_cfg.get("center", True)),
            "basis_per_var_effective": bool(basis_cfg.get("per_var", True)),
            "selection_mode_effective": str(dict(cfg.get("selection", {})).get("mode", "last")).strip().lower(),
            "selection_weights_effective": dict(dict(cfg.get("selection", {})).get("weights", {})),
            "model_cfg_effective": dict(model_cfg),
        }
        for key in (
            "model_type_effective",
            "target_family_effective",
            "target_vars_effective",
            "basis_rank_by_var",
            "basis_fit_scope_effective",
            "basis_center_effective",
            "basis_per_var_effective",
            "selection_mode_effective",
            "selection_weights_effective",
            "model_cfg_effective",
        ):
            if key in sample:
                out[key] = sample[key]
        return out

    def _aggregate_deeponet_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        deeponet_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        deeponet_cfg = dict(train_cfg.get("deeponet_plasma", {}))
        out = {
            "target_family_effective": str(deeponet_cfg.get("target_family", "allvars")).strip().lower(),
            "target_vars_effective": resolve_allvars_order(list(y_vars), prefer_linear=True),
            "selection_mode_effective": str(dict(deeponet_cfg.get("selection", {})).get("mode", "last")).strip().lower(),
            "selection_weights_effective": dict(dict(deeponet_cfg.get("selection", {})).get("weights", {})),
            "operator_mode_effective": str(deeponet_cfg.get("operator_mode", "pde_coupled")).strip().lower(),
            "trunk_input_mode_effective": str(
                dict(deeponet_cfg.get("model_cfg", {})).get("trunk_input_mode", "legacy_xy_fourier")
            ),
            "trunk_fourier_n_freq_effective": int(dict(deeponet_cfg.get("model_cfg", {})).get("trunk_fourier_n_freq", 1)),
            "trunk_fourier_mode_effective": str(dict(deeponet_cfg.get("model_cfg", {})).get("trunk_fourier_mode", "legacy")),
            "trunk_cond_modulation_effective": str(
                dict(deeponet_cfg.get("model_cfg", {})).get("trunk_cond_modulation", "none")
            ).strip().lower(),
            "trunk_cond_mod_hidden_effective": int(
                dict(deeponet_cfg.get("model_cfg", {})).get("trunk_cond_mod_hidden", 64)
            ),
            "residual_head_enabled_effective": bool(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("residual_head", {})).get("enabled", False)
            ),
            "residual_head_hidden_dim_effective": int(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("residual_head", {})).get("hidden_dim", 64)
            ),
            "residual_head_scale_init_effective": float(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("residual_head", {})).get("scale_init", 0.0)
            ),
            "residual_head_gain_mode_effective": str(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("residual_head", {})).get("gain_mode", "learned")
            ),
            "residual_head_gain_value_effective": float(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("residual_head", {})).get("gain_value", 1.0)
            ),
            "output_path_mode_effective": str(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("output_path", {})).get("mode", "dot")
            ),
            "output_path_dot_skip_effective": float(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("output_path", {})).get("dot_skip", 0.25)
            ),
            "output_path_dot_skip_mode_effective": str(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("output_path", {})).get("dot_skip_mode", "fixed")
            ).strip().lower(),
            "output_path_fused_hidden_dim_effective": int(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("output_path", {})).get("fused_hidden_dim", 96)
            ),
            "output_path_global_local_enabled_effective": bool(
                dict(dict(dict(deeponet_cfg.get("model_cfg", {})).get("output_path", {})).get("global_local", {})).get(
                    "enabled", False
                )
            ),
            "output_path_global_hidden_dim_effective": int(
                dict(dict(deeponet_cfg.get("model_cfg", {})).get("output_path", {})).get("global_hidden_dim", 64)
            ),
            "sensor_pool_mode_effective": str(
                dict(deeponet_cfg.get("model_cfg", {})).get("sensor_pool_mode", "moments")
            ),
            "branch_mode_effective": str(dict(deeponet_cfg.get("model_cfg", {})).get("branch_mode", "moments")),
            "missing_geom_feature_policy_effective": str(
                dict(deeponet_cfg.get("model_cfg", {})).get("missing_geom_feature_policy", "warn_zero")
            ).strip().lower(),
            "latent_layer_norm_effective": bool(dict(deeponet_cfg.get("model_cfg", {})).get("latent_layer_norm", False)),
            "optimizer_effective": {
                "type": str(dict(deeponet_cfg.get("optimizer", {})).get("type", "adamw")).strip().lower(),
                "lr": float(dict(deeponet_cfg.get("optimizer", {})).get("lr", deeponet_cfg.get("lr", train_cfg.get("lr", 1e-3)))),
                "weight_decay": float(dict(deeponet_cfg.get("optimizer", {})).get("weight_decay", 0.0)),
                "schedule": str(dict(deeponet_cfg.get("optimizer", {})).get("schedule", "none")).strip().lower(),
                "warmup_epochs": int(max(int(dict(deeponet_cfg.get("optimizer", {})).get("warmup_epochs", 0)), 0)),
            },
            "distance_transform_effective": {},
            "feature_contract_effective": {
                "input_features_mode": str(dict(deeponet_cfg.get("input_features", {})).get("mode", "legacy_xy")).strip().lower(),
                "input_feature_channels": [str(v) for v in list(dict(deeponet_cfg.get("input_features", {})).get("features", ["x", "y"]))],
                "feature_source_effective": "unknown",
            },
            "strict_mainline_effective": bool(deeponet_cfg.get("strict_mainline", False)),
        }
        l = self._first_list(deeponet_contract_samples, "target_vars_effective")
        if l is not None:
            out["target_vars_effective"] = [str(v) for v in l]
        v = self._first_str(deeponet_contract_samples, "target_family_effective")
        if v is not None:
            out["target_family_effective"] = v
        v = self._first_str(deeponet_contract_samples, "selection_mode_effective")
        if v is not None:
            out["selection_mode_effective"] = v
        d = self._first_dict(deeponet_contract_samples, "selection_weights_effective")
        if d is not None:
            out["selection_weights_effective"] = d

        def _set_cast_value(key: str, caster: Any) -> None:
            raw = self._first_str(deeponet_contract_samples, key)
            if raw is None:
                return
            try:
                out[key] = caster(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid deeponet contract value for {key}: {raw!r}") from exc

        v = self._first_str(deeponet_contract_samples, "operator_mode_effective")
        if v is not None:
            out["operator_mode_effective"] = v
        v = self._first_str(deeponet_contract_samples, "trunk_input_mode_effective")
        if v is not None:
            out["trunk_input_mode_effective"] = v
        _set_cast_value("trunk_fourier_n_freq_effective", int)
        v = self._first_str(deeponet_contract_samples, "trunk_fourier_mode_effective")
        if v is not None:
            out["trunk_fourier_mode_effective"] = v
        v = self._first_str(deeponet_contract_samples, "trunk_cond_modulation_effective")
        if v is not None:
            out["trunk_cond_modulation_effective"] = v
        _set_cast_value("trunk_cond_mod_hidden_effective", int)
        for sample in deeponet_contract_samples:
            if isinstance(sample, dict) and "residual_head_enabled_effective" in sample:
                out["residual_head_enabled_effective"] = bool(sample.get("residual_head_enabled_effective", False))
                break
        _set_cast_value("residual_head_hidden_dim_effective", int)
        _set_cast_value("residual_head_scale_init_effective", float)
        v = self._first_str(deeponet_contract_samples, "residual_head_gain_mode_effective")
        if v is not None:
            out["residual_head_gain_mode_effective"] = v
        _set_cast_value("residual_head_gain_value_effective", float)
        v = self._first_str(deeponet_contract_samples, "output_path_mode_effective")
        if v is not None:
            out["output_path_mode_effective"] = v
        _set_cast_value("output_path_dot_skip_effective", float)
        v = self._first_str(deeponet_contract_samples, "output_path_dot_skip_mode_effective")
        if v is not None:
            out["output_path_dot_skip_mode_effective"] = v
        _set_cast_value("output_path_fused_hidden_dim_effective", int)
        for sample in deeponet_contract_samples:
            if isinstance(sample, dict) and "output_path_global_local_enabled_effective" in sample:
                out["output_path_global_local_enabled_effective"] = bool(
                    sample.get("output_path_global_local_enabled_effective", False)
                )
                break
        _set_cast_value("output_path_global_hidden_dim_effective", int)
        v = self._first_str(deeponet_contract_samples, "sensor_pool_mode_effective")
        if v is not None:
            out["sensor_pool_mode_effective"] = v
        v = self._first_str(deeponet_contract_samples, "branch_mode_effective")
        if v is not None:
            out["branch_mode_effective"] = v
        v = self._first_str(deeponet_contract_samples, "missing_geom_feature_policy_effective")
        if v is not None:
            out["missing_geom_feature_policy_effective"] = v
        for sample in deeponet_contract_samples:
            if isinstance(sample, dict) and "latent_layer_norm_effective" in sample:
                out["latent_layer_norm_effective"] = bool(sample.get("latent_layer_norm_effective", False))
                break
        d = self._first_dict(deeponet_contract_samples, "optimizer_effective")
        if d is not None:
            out["optimizer_effective"] = d
        d = self._first_dict(deeponet_contract_samples, "feature_contract_effective")
        if d is not None:
            out["feature_contract_effective"] = d
        d = self._first_dict(deeponet_contract_samples, "distance_transform_effective")
        if d is not None:
            out["distance_transform_effective"] = d
        for sample in deeponet_contract_samples:
            if isinstance(sample, dict) and "strict_mainline_effective" in sample:
                out["strict_mainline_effective"] = bool(sample.get("strict_mainline_effective", False))
                break
        return out

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
    ) -> list[str]:
        guard_cfg = dict(self.benchmark_cfg.get("guardrails", {}))
        if not bool(guard_cfg.get("enabled", False)):
            return []
        mode = str(guard_cfg.get("mode", "warn")).strip().lower()
        if mode not in {"warn", "error"}:
            raise ValueError("benchmark.guardrails.mode must be one of: warn, error")
        checks = dict(guard_cfg.get("checks", {}))
        warnings: list[str] = []

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
                    warnings.append(
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
                            warnings.append(
                                f"effective_steps_floor: {model_name}.{split_name}={int(split_steps)} (<{floor})"
                            )
                elif int(steps) < floor:
                    warnings.append(f"effective_steps_floor: {model_name}={int(steps)} (<{floor})")
        if warnings and mode == "error":
            raise ValueError("benchmark.guardrails violations:\n- " + "\n- ".join(warnings))
        return warnings
