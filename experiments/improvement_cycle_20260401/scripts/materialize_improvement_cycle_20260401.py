"""Materialize benchmark fixtures for improvement cycle 2026-04-01."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


_MAINLINE_RUNTIME_DEFAULT: dict[str, Any] = {
    "input_mode": "table_plus_structure",
    "strict_input_mode": "error",
    "structure": {
        "feature_profile": "geom_v1_mainline",
        "descriptor_profile": "none",
        "latent_profile": "none",
        "adapter_mode": "auto",
        "provider_mode": "fixed",
    },
}


def _deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for key, value in patch.items():
            if key in out:
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = value
        return out
    return patch


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data or {})


def _normalize_structure_runtime_cfg(merged: dict[str, Any]) -> None:
    runtime = dict(merged.get("runtime", {}))
    bench = dict(merged.get("benchmark", {}))
    bench_runtime = dict(bench.get("runtime", {}))
    mode = str(bench_runtime.get("input_mode", runtime.get("input_mode", ""))).strip().lower()
    feature_profile = str(
        bench_runtime.get("structure", {}).get(
            "feature_profile",
            runtime.get("structure", {}).get("feature_profile", ""),
        )
    ).strip().lower()
    if mode != "table_plus_structure" or feature_profile == "":
        merged["benchmark"] = bench
        return

    pre = dict(bench.get("preprocessing", {}))
    coord = dict(pre.get("coord_features", {}))
    if len(coord) == 0:
        merged["benchmark"] = bench
        return

    coord.pop("channels", None)
    coord["channels_from_profile"] = feature_profile
    pre["coord_features"] = coord
    bench["preprocessing"] = pre
    merged["benchmark"] = bench


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize improvement-cycle benchmark fixtures")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/experimental/improvement_cycle_20260401/experiment_matrix.yaml"),
        help="Path to experiment matrix YAML",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/fixtures/generated/improvement_cycle_20260401"),
        help="Output directory for generated benchmark YAML files",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root directory",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    matrix_path = (root / args.matrix).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = _load_yaml(matrix_path)
    experiments = list(matrix.get("experiments", []))
    if len(experiments) == 0:
        raise ValueError(f"no experiments defined in matrix: {matrix_path}")

    for exp in experiments:
        exp_cfg = dict(exp or {})
        exp_id = str(exp_cfg.get("id", "")).strip()
        if not exp_id:
            raise ValueError("experiment.id must be non-empty")
        base_fixture = str(exp_cfg.get("base_fixture", "")).strip()
        if not base_fixture:
            raise ValueError(f"experiment {exp_id}: base_fixture must be non-empty")

        base_path = (root / base_fixture).resolve()
        base_yaml = _load_yaml(base_path)
        patch = dict(exp_cfg.get("patch", {}))
        merged = _deep_merge(base_yaml, patch)
        if not isinstance(merged.get("runtime"), dict):
            merged["runtime"] = dict(_MAINLINE_RUNTIME_DEFAULT)
        merged_bench = dict(merged.get("benchmark", {}))
        if not isinstance(merged_bench.get("runtime"), dict):
            merged_bench["runtime"] = dict(_MAINLINE_RUNTIME_DEFAULT)
        if not str(merged_bench.get("output_dir", "")).strip():
            merged_bench["output_dir"] = f"runs/improvement_cycle_20260401/{exp_id}"
            merged["benchmark"] = merged_bench
        else:
            merged["benchmark"] = merged_bench
        _normalize_structure_runtime_cfg(merged)

        out_path = out_dir / f"{exp_id}.yaml"
        out_path.write_text(
            yaml.safe_dump(merged, allow_unicode=False, sort_keys=False),
            encoding="utf-8",
        )
        print(f"[materialize] {exp_id}: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
