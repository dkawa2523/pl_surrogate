#!/usr/bin/env python3
"""Summarize GEC-CCP trustworthy benchmark leaderboards."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


TARGET_VARS = ("ne", "ni", "Te", "phi")

CORE_MODELS = (
    "global_mlp",
    "deeponet_pod",
    "geom_deeponet_pod",
    "coord_mlp_pod_residual",
    "fno",
    "ffno",
    "unet_operator_v2",
    "cno_operator_unet",
)

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

MODEL_ROLE = {
    "global_mlp": "table-only baseline",
    "deeponet_pod": "POD operator baseline",
    "geom_deeponet_pod": "geometry descriptor POD operator",
    "coord_mlp_pod_residual": "POD plus coordinate residual",
    "fno": "spectral operator",
    "ffno": "factorized spectral operator",
    "unet_operator_v2": "CNN/operator hybrid",
    "cno_operator_unet": "CNO/operator hybrid",
}

SUMMARY_FIELDS = (
    "dataset_size",
    "model_id",
    "status",
    "core_model",
    "model_role",
    "input_mode_effective",
    "surrogate_quality_score",
    "primary_metric",
    "primary_metric_value",
    "primary_metric_reliable",
    "test_r2_plasma_mean_dual",
    "test_r2_plasma_mean_interp",
    "test_r2_plasma_mean_extrap",
    "score_spatial_huber_component",
    "score_avgpool_huber_component",
    "score_nrmse_component",
    "score_boundary_component",
    "score_continuity_component",
    "score_physics_component",
    "score_sign_component",
    "sdf_boundary_to_deep_rmse_ratio_mean",
    "sdf_boundary_minus_deep_r2_mean",
    "positive_target_negative_ratio_penalty",
    "failure_mode",
    "leaderboard_csv",
    "log",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GEC-CCP trustworthy benchmark outputs.")
    parser.add_argument("--run-root", default="runs/gec_ccp_trustworthy_v1")
    parser.add_argument("--out-dir", default="runs/gec_ccp_trustworthy_v1/summary")
    parser.add_argument("--status-csv", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=[27, 54, 78])
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    return parser.parse_args()


def _read_first_csv_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return dict(rows[0]) if rows else None


def _read_status_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        out: dict[tuple[str, str], dict[str, str]] = {}
        for row in csv.DictReader(f):
            key = (str(row.get("dataset_size", "")), str(row.get("model_id", "")))
            out[key] = {str(k): str(v) for k, v in row.items()}
        return out


def _coerce_float(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _coerce_value(raw: Any) -> Any:
    value = _coerce_float(raw)
    if math.isfinite(value):
        return value
    return "" if raw is None else str(raw)


def _mean_target_r2(row: dict[str, Any], *, split_suffix: str = "") -> float:
    vals: list[float] = []
    for target in TARGET_VARS:
        value = _coerce_float(row.get(f"test_r2_{target}_plasma{split_suffix}"))
        if not math.isfinite(value):
            return float("nan")
        vals.append(value)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _fill_r2_mean_fields(out: dict[str, Any], row: dict[str, Any]) -> None:
    if not math.isfinite(_coerce_float(out.get("test_r2_plasma_mean_dual"))):
        fallback = _mean_target_r2(row)
        if math.isfinite(fallback):
            out["test_r2_plasma_mean_dual"] = fallback
    if not math.isfinite(_coerce_float(out.get("test_r2_plasma_mean_interp"))):
        fallback = _mean_target_r2(row, split_suffix="_interp")
        if math.isfinite(fallback):
            out["test_r2_plasma_mean_interp"] = fallback
    if not math.isfinite(_coerce_float(out.get("test_r2_plasma_mean_extrap"))):
        fallback = _mean_target_r2(row, split_suffix="_extrap")
        if math.isfinite(fallback):
            out["test_r2_plasma_mean_extrap"] = fallback


def _failure_mode(row: dict[str, Any]) -> str:
    status = str(row.get("status", "")).strip()
    if status and status not in {"passed", "skipped_existing", "leaderboard_found"}:
        return status
    reliable = str(row.get("primary_metric_reliable", "")).strip().lower()
    if reliable in {"false", "0"}:
        invalid = str(row.get("target_metrics_invalid_vars", "")).strip()
        return f"unreliable target metrics: {invalid}" if invalid else "unreliable target metrics"
    if math.isfinite(_coerce_float(row.get("score_spatial_huber_component"))) or math.isfinite(
        _coerce_float(row.get("score_avgpool_huber_component"))
    ):
        components = {
            "spatial huber": _coerce_float(row.get("score_spatial_huber_component")),
            "avgpool huber": _coerce_float(row.get("score_avgpool_huber_component")),
        }
    else:
        components = {
            "boundary": _coerce_float(row.get("score_boundary_component")),
            "continuity": _coerce_float(row.get("score_continuity_component")),
            "physics": _coerce_float(row.get("score_physics_component")),
            "sign": _coerce_float(row.get("score_sign_component")),
        }
    finite = {key: value for key, value in components.items() if math.isfinite(value)}
    if not finite:
        return "not evaluated"
    worst_key, worst_value = max(finite.items(), key=lambda item: item[1])
    if worst_value <= 0.0:
        return "none flagged"
    return f"{worst_key} component"


def _summary_row(
    *,
    size: int,
    model: str,
    row: dict[str, Any] | None,
    status_row: dict[str, str] | None,
    leaderboard_path: Path,
) -> dict[str, Any]:
    status = str((status_row or {}).get("status", "")).strip()
    if row is None:
        out: dict[str, Any] = {
            "dataset_size": int(size),
            "model_id": model,
            "status": status or "missing_leaderboard",
            "core_model": model in set(CORE_MODELS),
            "model_role": MODEL_ROLE.get(model, "appendix candidate"),
            "leaderboard_csv": str(leaderboard_path),
            "log": str((status_row or {}).get("log", "")),
        }
        out["failure_mode"] = _failure_mode(out)
        return {field: out.get(field, "") for field in SUMMARY_FIELDS}

    out = {field: _coerce_value(row.get(field, "")) for field in SUMMARY_FIELDS}
    out.update(
        {
            "dataset_size": int(size),
            "model_id": model,
            "status": status or "leaderboard_found",
            "core_model": model in set(CORE_MODELS),
            "model_role": MODEL_ROLE.get(model, "appendix candidate"),
            "leaderboard_csv": str(leaderboard_path),
            "log": str((status_row or {}).get("log", "")),
        }
    )
    _fill_r2_mean_fields(out, row)
    out["failure_mode"] = _failure_mode({**row, **out})
    return {field: out.get(field, "") for field in SUMMARY_FIELDS}


def _sort_key_quality(row: dict[str, Any]) -> tuple[int, float, str]:
    score = _coerce_float(row.get("surrogate_quality_score"))
    if not math.isfinite(score):
        return (1, float("inf"), str(row.get("model_id", "")))
    return (0, score, str(row.get("model_id", "")))


def _sort_key_r2(row: dict[str, Any]) -> tuple[int, float, str]:
    score = _coerce_float(row.get("test_r2_plasma_mean_dual"))
    if not math.isfinite(score):
        return (1, 0.0, str(row.get("model_id", "")))
    return (0, -score, str(row.get("model_id", "")))


def _format_float(raw: Any, digits: int = 4) -> str:
    value = _coerce_float(raw)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], *, limit: int | None = None) -> str:
    selected = rows[:limit] if limit is not None else rows
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in selected:
        vals: list[str] = []
        for _, key in columns:
            raw = row.get(key, "")
            if isinstance(raw, float):
                vals.append(_format_float(raw))
            else:
                vals.append(str(raw))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, *, rows: list[dict[str, Any]], sizes: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latest_size = max(sizes)
    latest = [row for row in rows if int(row["dataset_size"]) == latest_size]
    core_latest = [row for row in latest if str(row.get("core_model", "")).lower() == "true"]
    quality_latest = sorted(core_latest, key=_sort_key_quality)
    r2_latest = sorted(core_latest, key=_sort_key_r2)
    failed = [
        row
        for row in rows
        if str(row.get("status", "")).startswith("failed") or str(row.get("status", "")) == "missing_leaderboard"
    ]

    trend_rows: list[dict[str, Any]] = []
    for model in CORE_MODELS:
        item: dict[str, Any] = {"model_id": model}
        for size in sizes:
            match = next((row for row in rows if int(row["dataset_size"]) == size and row["model_id"] == model), None)
            item[f"quality_n{size}"] = _format_float(match.get("surrogate_quality_score") if match else "")
            item[f"r2_n{size}"] = _format_float(match.get("test_r2_plasma_mean_dual") if match else "")
        trend_rows.append(item)

    trend_columns: list[tuple[str, str]] = [("Model", "model_id")]
    for size in sizes:
        trend_columns.extend([(f"Quality n{size}", f"quality_n{size}"), (f"Top R2 n{size}", f"r2_n{size}")])

    content = [
        "# GEC-CCP Trustworthy Benchmark Summary",
        "",
        f"Latest-size main table uses n={latest_size}. Lower `surrogate_quality_score` is better.",
        "",
        "## Main Reliability Ranking",
        "",
        _markdown_table(
            quality_latest,
            [
                ("Model", "model_id"),
                ("Role", "model_role"),
                ("Input", "input_mode_effective"),
                ("Quality", "surrogate_quality_score"),
                ("SpatialHuber", "score_spatial_huber_component"),
                ("Avgpool", "score_avgpool_huber_component"),
                ("Failure mode", "failure_mode"),
            ],
        ),
        "",
        "## Auxiliary Plasma R2 Ranking",
        "",
        _markdown_table(
            r2_latest,
            [
                ("Model", "model_id"),
                ("Top-level R2", "test_r2_plasma_mean_dual"),
                ("Interp R2", "test_r2_plasma_mean_interp"),
                ("Extrap R2", "test_r2_plasma_mean_extrap"),
                ("Quality", "surrogate_quality_score"),
            ],
        ),
        "",
        "## Data Size Dependence",
        "",
        _markdown_table(trend_rows, trend_columns),
        "",
        "## Failed Or Missing Runs",
        "",
        _markdown_table(
            failed,
            [
                ("Size", "dataset_size"),
                ("Model", "model_id"),
                ("Status", "status"),
                ("Log", "log"),
            ],
        )
        if failed
        else "No failed runs were found in the status CSV.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    sizes = [int(size) for size in args.sizes]
    models = [str(model) for model in args.models]
    status_path = Path(args.status_csv) if args.status_csv else run_root / "run_status.csv"
    status_rows = _read_status_rows(status_path)

    rows: list[dict[str, Any]] = []
    for size in sizes:
        for model in models:
            leaderboard = run_root / f"n{size}" / model / "leaderboard.csv"
            status_row = status_rows.get((str(size), model))
            rows.append(
                _summary_row(
                    size=size,
                    model=model,
                    row=_read_first_csv_row(leaderboard),
                    status_row=status_row,
                    leaderboard_path=leaderboard,
                )
            )

    summary_csv = out_dir / "summary_trustworthy_27_54_78.csv"
    core_csv = out_dir / "summary_trustworthy_core_27_54_78.csv"
    report_md = out_dir / "report_trustworthy_27_54_78.md"
    _write_csv(summary_csv, rows)
    _write_csv(core_csv, [row for row in rows if str(row.get("core_model", "")).lower() == "true"])
    _write_report(report_md, rows=rows, sizes=sizes)
    print(summary_csv)
    print(core_csv)
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
