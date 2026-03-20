#!/usr/bin/env python3
"""One-off compare utility for selected model rows from benchmark leaderboards."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import yaml


def _as_float(row: dict[str, Any], key: str) -> float:
    raw = row.get(key, 0.0)
    try:
        return float(raw)
    except Exception:
        return 0.0


def _coerce_value(raw: Any) -> Any:
    if raw is None:
        return ""
    if isinstance(raw, (int, float)):
        return raw
    text = str(raw).strip()
    if text == "":
        return ""
    try:
        val = float(text)
        if math.isfinite(val):
            return val
        return ""
    except Exception:
        return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path (compare: ...)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    compare_cfg = cfg.get("compare", cfg)
    out_dir = Path(str(compare_cfg.get("output_dir", "runs/compare_selected")))
    out_dir.mkdir(parents=True, exist_ok=True)
    objective_metric_raw = str(compare_cfg.get("objective_metric", "test_rmse_phi"))
    objective_mode = str(compare_cfg.get("objective_mode", "min"))
    primary_metric_cfg = str(compare_cfg.get("primary_metric", objective_metric_raw))
    global_reference_mode = str(compare_cfg.get("global_reference_mode", "off")).strip().lower()
    if global_reference_mode not in {"off", "frozen"}:
        raise ValueError("compare.global_reference_mode must be one of: off, frozen")

    rows_out: list[dict[str, Any]] = []
    dynamic_keys: set[str] = set()
    has_global_reference_row = False
    for item in compare_cfg.get("rows", []):
        lb = Path(str(item["leaderboard_csv"]))
        model_id = str(item["model_id"])
        with lb.open("r", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        row = next((r for r in data if str(r.get("model_id")) == model_id), None)
        if row is None:
            raise ValueError(f"model_id={model_id} not found in {lb}")
        row_primary_metric = str(row.get("primary_metric", primary_metric_cfg))
        if not row_primary_metric:
            row_primary_metric = primary_metric_cfg
        out_row: dict[str, Any] = {
            "name": str(item.get("name", model_id)),
            "model_id": model_id,
            "target_family": str(item.get("target_family", row.get("target_family_effective", ""))),
            "primary_metric": row_primary_metric,
            "primary_metric_value": _as_float(row, row_primary_metric),
            "merge_role": str(item.get("merge_role", row.get("merge_role", "single"))),
            "source_leaderboard": str(lb),
            "reference_type": "frozen" if global_reference_mode == "frozen" and model_id == "global_mlp" else "candidate",
            "protocol_variant": str(row.get("protocol_variant", "")),
        }
        for k, v in row.items():
            if k in {
                "rank",
                "name",
                "model_id",
                "target_family",
                "primary_metric",
                "primary_metric_value",
                "merge_role",
                "source_leaderboard",
                "reference_type",
                "protocol_variant",
            }:
                continue
            out_row[str(k)] = _coerce_value(v)
            dynamic_keys.add(str(k))
        rows_out.append(out_row)
        if global_reference_mode == "frozen" and model_id == "global_mlp":
            has_global_reference_row = True

    if global_reference_mode == "frozen" and not has_global_reference_row:
        raise ValueError("compare.global_reference_mode=frozen requires at least one row with model_id=global_mlp")

    if objective_metric_raw == "auto_primary":
        metric_candidates = [str(r.get("primary_metric", "")) for r in rows_out if str(r.get("primary_metric", ""))]
        unique_metrics = {m for m in metric_candidates if m}
        objective_metric = "primary_metric_value" if len(unique_metrics) > 1 else (metric_candidates[0] if metric_candidates else primary_metric_cfg)
    else:
        objective_metric = objective_metric_raw

    reverse = objective_mode == "max"
    rows_out.sort(key=lambda r: float(r.get(objective_metric, 0.0)), reverse=reverse)
    for i, row in enumerate(rows_out, start=1):
        row["rank"] = int(i)

    base_header = [
        "rank",
        "name",
        "model_id",
        "target_family",
        "primary_metric",
        "primary_metric_value",
    ]
    tail_header = [
        "protocol_variant",
        "merge_role",
        "reference_type",
        "source_leaderboard",
    ]
    dynamic_header = sorted(
        k
        for k in dynamic_keys
        if k not in set(base_header) and k not in set(tail_header)
    )
    header = base_header + dynamic_header + tail_header
    out_csv = out_dir / "selected_models_comparison.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows_out)
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
