#!/usr/bin/env python3
"""Audit and summarize the GEC-CCP NN/operator comparison v1 runs.

The suite is an interpolation-only, three-seed comparison.  This script fails
closed when the 8 x 3 matrix or its common spatial-learning contract is
incomplete.  Representative plotting seeds are selected only by the minimum
finite validation objective; test metrics are aggregated and reported after
that selection rule has been fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


DEFAULT_MODELS = (
    "global_mlp",
    "global_resmlp",
    "global_densemlp",
    "unet",
    "fno",
    "ffno",
    "u_no",
    "deeponet_pod",
)
DEFAULT_SEEDS = (411, 412, 413)
TARGETS = ("ne", "ni", "Te", "phi")
MODEL_FAMILIES = {
    "global_mlp": "vector_nn",
    "global_resmlp": "vector_nn",
    "global_densemlp": "vector_nn",
    "unet": "convolutional_nn",
    "fno": "neural_operator",
    "ffno": "neural_operator",
    "u_no": "neural_operator",
    "deeponet_pod": "pod_operator",
}

DATASET_INDEX = "index_78.csv"
LOSS_PROTOCOL = "plasma_surrogate_v3"
QUALITY_PROTOCOL = "spatial_huber_case_balanced_v2"
SELECTION_MODE = "best_val_spatial_objective"
SELECTION_OBJECTIVE_VERSION = "case_macro_spatial_rmse_v2"
BOUNDARY_CHANNELS = ("distance_any", "part_sdf_nearest")


def _metric_fields() -> tuple[str, ...]:
    fields = [
        "surrogate_quality_score",
        "test_r2_plasma_mean",
        "score_spatial_huber_component",
        "score_avgpool_huber_component",
        "score_gradient_huber_component",
    ]
    for target in TARGETS:
        fields.extend(
            [
                f"test_rmse_{target}_plasma",
                f"test_r2_{target}_plasma",
                f"score_total_{target}",
                f"score_physical_rel_l2_median_{target}",
                f"score_physical_rel_l2_p90_{target}",
                f"score_physical_rel_l2_worst_{target}",
                f"score_physical_gradient_rel_l2_median_{target}",
                f"score_physical_gradient_rel_l2_p90_{target}",
                f"score_physical_gradient_rel_l2_worst_{target}",
                f"test_rmse_{target}_boundary_to_deep_ratio",
            ]
        )
        if target in {"ne", "ni", "Te"}:
            fields.append(f"test_neg_ratio_{target}_plasma")
    return tuple(fields)


METRIC_FIELDS = _metric_fields()
VALIDATION_MANIFEST_FIELDS = (
    "dataset_size",
    "model_id",
    "family",
    "seed",
    "validation_selection_metric",
    "validation_selection_value",
    "validation_selection_mode",
    "validation_selection_reliable",
    "leaderboard",
    "run_root",
    "selection_split",
    "protocol_variant",
    "scaler_fit_split",
    "selection_source",
    "test_metrics_used_for_selection",
    "loss_protocol_definition_hash",
    "quality_score_definition_hash",
    "target_schema_hash",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/gec_ccp_nn_operator_comparison_v1")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dataset-size", type=int, default=78)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    return parser.parse_args()


def _read_csv(path: Path, *, exactly_one: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if exactly_one and len(rows) != 1:
        raise ValueError(f"CSV must contain exactly one row: {path}, rows={len(rows)}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return dict(payload)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(payload)


def _required_text(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"missing non-empty {field}: {context}")
    return value


def _finite_float(row: Mapping[str, Any], field: str, *, context: str) -> float:
    try:
        value = float(row.get(field, float("nan")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric: {context}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite: {context}")
    return value


def _as_bool(raw: Any, *, field: str, context: str) -> bool:
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"true", "1", "1.0", "yes"}:
        return True
    if normalized in {"false", "0", "0.0", "no"}:
        return False
    raise ValueError(f"{field} must be boolean-compatible: {context}, value={raw!r}")


def _require_true(row: Mapping[str, Any], field: str, *, context: str) -> None:
    if not _as_bool(row.get(field), field=field, context=context):
        raise ValueError(f"{field} must be true: {context}")


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _require_equal(actual: Any, expected: Any, *, field: str, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: {context}, expected={expected!r}, got={actual!r}")


def _audit_resolved_config(
    cfg: Mapping[str, Any], *, seed: int, model: str, size: int, context: str
) -> None:
    _require_equal(size, 78, field="dataset_size", context=context)
    _require_equal(int(cfg.get("seed", -1)), int(seed), field="seed", context=context)
    _require_equal(
        Path(str(_nested(cfg, "dataset", "index_csv"))).name,
        DATASET_INDEX,
        field="dataset.index_csv",
        context=context,
    )
    _require_equal(
        tuple(str(v) for v in (_nested(cfg, "dataset", "cond_columns") or [])),
        ("PP0", "Td", "gamma", "PA"),
        field="dataset.cond_columns",
        context=context,
    )
    targets = tuple(str(item.get("id", "")) for item in (_nested(cfg, "dataset", "targets") or []))
    _require_equal(targets, TARGETS, field="dataset.targets", context=context)
    for item in _nested(cfg, "dataset", "targets") or []:
        _require_equal(item.get("value_transform"), "identity", field="target.value_transform", context=context)
    _require_equal(int(_nested(cfg, "split", "seed") or -1), 7, field="split.seed", context=context)
    ratios = tuple(float(v) for v in (_nested(cfg, "split", "ratios") or []))
    _require_equal(ratios, (0.7, 0.15, 0.15), field="split.ratios", context=context)
    _require_equal(_nested(cfg, "eval_protocol", "mode"), "primary_axis", field="eval_protocol.mode", context=context)
    _require_equal(
        _nested(cfg, "eval_protocol", "primary_split"),
        "interp",
        field="eval_protocol.primary_split",
        context=context,
    )
    _require_equal(
        _nested(cfg, "eval_protocol", "interp_mode"),
        "marginal",
        field="eval_protocol.interp_mode",
        context=context,
    )
    _require_equal(
        tuple(str(v) for v in (_nested(cfg, "eval", "target_vars_for_score") or [])),
        TARGETS,
        field="eval.target_vars_for_score",
        context=context,
    )
    _require_equal(
        _nested(cfg, "preprocessing", "scalers", "y_fit_policy"),
        "plasma_only",
        field="scalers.y_fit_policy",
        context=context,
    )
    for target in TARGETS:
        transform = _nested(cfg, "preprocessing", "scalers", "target_transforms", target) or {}
        _require_equal(
            transform.get("value_transform"),
            "identity",
            field=f"{target}.value_transform",
            context=context,
        )
        _require_equal(transform.get("scaler"), "zscore", field=f"{target}.scaler", context=context)
        _require_equal(transform.get("fit_scope"), "plasma_only", field=f"{target}.fit_scope", context=context)

    loss = _nested(cfg, "train", "loss") or {}
    _require_equal(loss.get("protocol"), LOSS_PROTOCOL, field="train.loss.protocol", context=context)
    supervised = dict(loss.get("supervised", {}) or {})
    _require_equal(supervised.get("type"), "huber", field="loss.type", context=context)
    _require_equal(supervised.get("mask"), "plasma_only", field="loss.mask", context=context)
    _require_equal(supervised.get("normalization"), "sample_mean", field="loss.normalization", context=context)
    spatial = dict(supervised.get("spatial", {}) or {})
    for field, expected in (
        ("gradient_weight", 0.1),
        ("multiscale_weight", 0.05),
        ("boundary_weight", 0.25),
        ("boundary_band_px", 2.0),
    ):
        _require_equal(
            float(spatial.get(field, float("nan"))),
            expected,
            field=f"loss.spatial.{field}",
            context=context,
        )
    _require_equal(
        tuple(spatial.get("multiscale_scales", [])),
        (2, 4),
        field="loss.multiscale_scales",
        context=context,
    )
    _require_equal(
        tuple(spatial.get("boundary_distance_channels", [])),
        BOUNDARY_CHANNELS,
        field="loss.boundary_channels",
        context=context,
    )
    _require_equal(
        _nested(loss, "group_weighting", "mode"),
        "uniform_by_group",
        field="loss.group_weighting",
        context=context,
    )
    selection = _nested(cfg, "train", model, "selection") or {}
    _require_equal(selection.get("mode"), SELECTION_MODE, field=f"train.{model}.selection.mode", context=context)
    _require_equal(
        {str(key): float(value) for key, value in dict(selection.get("weights", {}) or {}).items()},
        {target: 0.25 for target in TARGETS},
        field="selection.weights",
        context=context,
    )


def _status_for_model(path: Path, *, size: int, model: str) -> dict[str, str]:
    rows = _read_csv(path)
    matches = [
        row
        for row in rows
        if str(row.get("dataset_size", "")) == str(size)
        and str(row.get("model_id", "")).strip() == model
    ]
    if len(matches) != 1:
        raise ValueError(f"run status must contain one row for n={size}, model={model}: {path}")
    status = str(matches[0].get("status", "")).strip()
    if status not in {"passed", "skipped_existing"}:
        raise ValueError(f"run did not pass: model={model}, status={status!r}, path={path}")
    _finite_float(matches[0], "seconds", context=f"status model={model}")
    return matches[0]


def _seed_row(*, run_root: Path, seed: int, size: int, model: str) -> dict[str, Any]:
    model_root = run_root / f"seed_{seed}" / f"n{size}" / model
    context = f"seed={seed}, model={model}"
    status = _status_for_model(
        run_root / f"seed_{seed}" / "run_status.csv",
        size=size,
        model=model,
    )
    cfg = _read_yaml(model_root / "resolved_config.yaml")
    _audit_resolved_config(cfg, seed=seed, model=model, size=size, context=context)

    leaderboard_path = model_root / "leaderboard.csv"
    leaderboard = _read_csv(leaderboard_path, exactly_one=True)[0]
    diagnostics = _read_csv(model_root / "diagnostics" / "diagnostics.csv", exactly_one=True)[0]
    for row, artifact in ((leaderboard, "leaderboard"), (diagnostics, "diagnostics")):
        _require_equal(str(row.get("model_id", "")).strip(), model, field=f"{artifact}.model_id", context=context)

    for field in (
        "target_metrics_valid",
        "primary_metric_reliable",
        "validation_selection_reliable",
        "scaler_train_only",
        "evaluation_geometry_case_aligned",
    ):
        _require_true(leaderboard, field, context=context)
    _require_equal(leaderboard.get("selection_split"), "interp", field="selection_split", context=context)
    _require_equal(leaderboard.get("scaler_fit_split"), "interp", field="scaler_fit_split", context=context)
    _require_equal(
        leaderboard.get("validation_selection_mode"),
        "min",
        field="validation_selection_mode",
        context=context,
    )
    _require_equal(
        leaderboard.get("validation_selection_metric"),
        "selected_epoch_score",
        field="validation_selection_metric",
        context=context,
    )
    _require_equal(
        leaderboard.get("validation_selection_objective_version"),
        SELECTION_OBJECTIVE_VERSION,
        field="validation_selection_objective_version",
        context=context,
    )
    validation_value = _finite_float(leaderboard, "validation_selection_value", context=context)
    quality = _finite_float(leaderboard, "surrogate_quality_score", context=context)
    _require_equal(
        leaderboard.get("quality_score_protocol"),
        QUALITY_PROTOCOL,
        field="quality_score_protocol",
        context=context,
    )
    quality_hash = _required_text(leaderboard, "quality_score_definition_hash", context=context)
    _require_equal(
        tuple(
            str(v)
            for v in yaml.safe_load(
                str(leaderboard.get("evaluation_boundary_distance_channels", "[]"))
            )
            or []
        ),
        BOUNDARY_CHANNELS,
        field="evaluation_boundary_distance_channels",
        context=context,
    )

    manifest = _read_json(model_root / "manifest.json")
    loss_hash = _required_text(
        dict(manifest.get("training", {}) or {}),
        "loss_protocol_definition_hash",
        context=context,
    )
    _require_equal(
        _nested(manifest, "training", "loss_protocol_effective"),
        LOSS_PROTOCOL,
        field="manifest.loss_protocol",
        context=context,
    )
    split_hash = _required_text(
        dict(_nested(manifest, "artifacts", "artifact_hashes") or {}),
        "split_hash",
        context=context,
    )

    checkpoint = _read_json(
        model_root / "models" / model / "eval_protocol" / "interp" / "checkpoints" / "meta.json"
    )
    _require_equal(checkpoint.get("scaler_fit_split"), "interp", field="checkpoint.scaler_fit_split", context=context)
    _require_true(checkpoint, "scaler_train_only", context=context)
    _require_equal(
        checkpoint.get("loss_protocol_definition_hash"),
        loss_hash,
        field="checkpoint.loss_hash",
        context=context,
    )
    target_hash = _required_text(checkpoint, "target_schema_hash", context=context)
    feature_hash = _required_text(checkpoint, "feature_schema_hash", context=context)

    merged_metrics = {**leaderboard, **diagnostics}
    r2_values = [
        _finite_float(leaderboard, f"test_r2_{target}_plasma", context=context)
        for target in TARGETS
    ]
    row: dict[str, Any] = {
        "seed": int(seed),
        "dataset_size": int(size),
        "model_id": model,
        "family": MODEL_FAMILIES.get(model, "neural_model"),
        "status": str(status["status"]),
        "seconds": _finite_float(status, "seconds", context=context),
        "leaderboard": leaderboard_path.as_posix(),
        "diagnostics": (model_root / "diagnostics" / "diagnostics.csv").as_posix(),
        "run_root": model_root.as_posix(),
        "validation_selection_metric": "selected_epoch_score",
        "validation_selection_value": validation_value,
        "validation_selection_mode": "min",
        "validation_selection_reliable": True,
        "selection_split": "interp",
        "protocol_variant": _required_text(leaderboard, "protocol_variant", context=context),
        "scaler_fit_split": "interp",
        "loss_protocol_definition_hash": loss_hash,
        "quality_score_definition_hash": quality_hash,
        "split_hash": split_hash,
        "target_schema_hash": target_hash,
        "feature_schema_hash": feature_hash,
        "surrogate_quality_score": quality,
        "test_r2_plasma_mean": statistics.fmean(r2_values),
    }
    for field in METRIC_FIELDS:
        if field not in row:
            row[field] = _finite_float(merged_metrics, field, context=context)
    return row


def _same(rows: Sequence[Mapping[str, Any]], field: str, *, scope: str) -> str:
    values = {str(row.get(field, "")).strip() for row in rows}
    if "" in values or len(values) != 1:
        raise ValueError(f"{field} differs or is missing for {scope}: {sorted(values)}")
    return next(iter(values))


def _audit_matrix(
    rows: Sequence[Mapping[str, Any]], *, seeds: Sequence[int], models: Sequence[str]
) -> None:
    if len(seeds) != 3 or len(set(int(seed) for seed in seeds)) != 3:
        raise ValueError("comparison v1 requires exactly three distinct seeds")
    if not models or len(set(str(model) for model in models)) != len(models):
        raise ValueError("models must be unique and non-empty")
    expected = {(int(seed), str(model)) for seed in seeds for model in models}
    actual = {(int(row["seed"]), str(row["model_id"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"seed/model matrix is incomplete or duplicated: expected={expected}, actual={actual}")
    for field in (
        "split_hash",
        "target_schema_hash",
        "loss_protocol_definition_hash",
        "quality_score_definition_hash",
    ):
        _same(rows, field, scope="all comparison-v1 runs")
    for model in models:
        model_rows = [row for row in rows if str(row["model_id"]) == str(model)]
        _same(model_rows, "feature_schema_hash", scope=f"model={model}")
        _same(model_rows, "validation_selection_metric", scope=f"model={model}")
        _same(model_rows, "validation_selection_mode", scope=f"model={model}")


def _aggregate(
    rows: Sequence[Mapping[str, Any]], *, models: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in rows if str(row["model_id"]) == str(model)]
        # The key is deliberately validation-only.  Test metrics do not occur here.
        selected = min(
            model_rows,
            key=lambda row: (float(row["validation_selection_value"]), int(row["seed"])),
        )
        aggregate: dict[str, Any] = {
            "model_id": model,
            "family": str(selected["family"]),
            "n_seeds": len(model_rows),
            "validation_selected_seed": int(selected["seed"]),
        }
        for field in ("seconds", "validation_selection_value", *METRIC_FIELDS):
            values = [float(row[field]) for row in model_rows]
            aggregate[f"{field}_mean"] = statistics.fmean(values)
            aggregate[f"{field}_std"] = statistics.stdev(values)
        aggregates.append(aggregate)
        selected_rows.append(
            {
                "dataset_size": int(selected["dataset_size"]),
                "model_id": model,
                "family": str(selected["family"]),
                "seed": int(selected["seed"]),
                "validation_selection_metric": str(selected["validation_selection_metric"]),
                "validation_selection_value": float(selected["validation_selection_value"]),
                "validation_selection_mode": "min",
                "validation_selection_reliable": True,
                "leaderboard": str(selected["leaderboard"]),
                "run_root": str(selected["run_root"]),
                "selection_split": "interp",
                "protocol_variant": str(selected["protocol_variant"]),
                "scaler_fit_split": "interp",
                "selection_source": "minimum_finite_training_validation_objective_across_seeds",
                "test_metrics_used_for_selection": False,
                "loss_protocol_definition_hash": str(selected["loss_protocol_definition_hash"]),
                "quality_score_definition_hash": str(selected["quality_score_definition_hash"]),
                "target_schema_hash": str(selected["target_schema_hash"]),
            }
        )
    aggregates.sort(key=lambda row: float(row["surrogate_quality_score_mean"]))
    return aggregates, selected_rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def _fmt(raw: Any, digits: int = 4) -> str:
    value = float(raw)
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def _report(
    *, rows: Sequence[Mapping[str, Any]], aggregates: Sequence[Mapping[str, Any]], size: int
) -> str:
    n_models = len({str(row["model_id"]) for row in rows})
    n_seeds = len({int(row["seed"]) for row in rows})
    lines = [
        "# GEC-CCP NN / Neural-Operator Comparison v1",
        "",
        f"n={size}; interpolation split 54/11/13; {n_seeds} independent training seeds.",
        f"All requested {n_models} x {n_seeds} artifacts passed the common-contract audit before aggregation.",
        "Representative plotting seeds use only the minimum finite validation objective.",
        "",
        "## Three-seed aggregate",
        "",
        "| Model | Quality mean ± SD | Plasma R2 mean ± SD | Validation mean ± SD | Plot seed | Runtime mean [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {model} | {quality} ± {quality_sd} | {r2} ± {r2_sd} | {val} ± {val_sd} | {seed} | {seconds} |".format(
                model=row["model_id"],
                quality=_fmt(row["surrogate_quality_score_mean"]),
                quality_sd=_fmt(row["surrogate_quality_score_std"]),
                r2=_fmt(row["test_r2_plasma_mean_mean"]),
                r2_sd=_fmt(row["test_r2_plasma_mean_std"]),
                val=_fmt(row["validation_selection_value_mean"]),
                val_sd=_fmt(row["validation_selection_value_std"]),
                seed=row["validation_selected_seed"],
                seconds=_fmt(row["seconds_mean"], 1),
            )
        )
    lines.extend(
        [
            "",
            "## Physical diagnostics (three-seed means)",
            "",
            "Negative ratios are evaluated only inside the plasma target. Gradient values are median physical relative L2 errors.",
            "",
            "| Model | ne negative [%] | ni negative [%] | Te negative [%] | ne gradient [%] | ni gradient [%] | Te gradient [%] | phi gradient [%] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(
            "| {model} | {neg_ne} | {neg_ni} | {neg_te} | {grad_ne} | {grad_ni} | {grad_te} | {grad_phi} |".format(
                model=row["model_id"],
                neg_ne=_fmt(100.0 * float(row["test_neg_ratio_ne_plasma_mean"])),
                neg_ni=_fmt(100.0 * float(row["test_neg_ratio_ni_plasma_mean"])),
                neg_te=_fmt(100.0 * float(row["test_neg_ratio_Te_plasma_mean"])),
                grad_ne=_fmt(100.0 * float(row["score_physical_gradient_rel_l2_median_ne_mean"]), 2),
                grad_ni=_fmt(100.0 * float(row["score_physical_gradient_rel_l2_median_ni_mean"]), 2),
                grad_te=_fmt(100.0 * float(row["score_physical_gradient_rel_l2_median_Te_mean"]), 2),
                grad_phi=_fmt(100.0 * float(row["score_physical_gradient_rel_l2_median_phi_mean"]), 2),
            )
        )
    lines.extend(
        [
            "",
            "## Seed-level audit",
            "",
            "| Model | Seed | Validation | Quality | Plasma R2 | Runtime [s] |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["model_id"]), int(item["seed"]))):
        lines.append(
            "| {model} | {seed} | {val} | {quality} | {r2} | {seconds} |".format(
                model=row["model_id"],
                seed=row["seed"],
                val=_fmt(row["validation_selection_value"]),
                quality=_fmt(row["surrogate_quality_score"]),
                r2=_fmt(row["test_r2_plasma_mean"]),
                seconds=_fmt(row["seconds"], 1),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The quality ranking is a test-set report, not a model-selection rule.",
            "The configured quality score does not penalize sign violations; use the physical-diagnostics table alongside it.",
            "Use the validation-selected manifest for publication plots and keep the PP0 stress lane separate.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_comparison(
    *, run_root: Path, out_dir: Path, size: int, seeds: Sequence[int], models: Sequence[str]
) -> tuple[Path, Path, Path, Path]:
    rows = [
        _seed_row(run_root=run_root, seed=int(seed), size=int(size), model=str(model))
        for seed in seeds
        for model in models
    ]
    _audit_matrix(rows, seeds=seeds, models=models)
    aggregates, selected = _aggregate(rows, models=models)

    seed_csv = out_dir / "seed_metrics.csv"
    aggregate_csv = out_dir / "model_seed_aggregate.csv"
    report_md = out_dir / "report.md"
    selected_csv = run_root / "validation_selected" / "best_by_model.csv"
    _write_csv(seed_csv, rows)
    _write_csv(aggregate_csv, aggregates)
    _write_csv(selected_csv, selected, fields=VALIDATION_MANIFEST_FIELDS)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(_report(rows=rows, aggregates=aggregates, size=int(size)), encoding="utf-8")
    return seed_csv, aggregate_csv, selected_csv, report_md


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "evaluation" / "seed_matrix"
    paths = summarize_comparison(
        run_root=run_root,
        out_dir=out_dir,
        size=int(args.dataset_size),
        seeds=[int(value) for value in args.seeds],
        models=[str(value) for value in args.models],
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
