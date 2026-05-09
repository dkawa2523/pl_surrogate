from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


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
    "deeponet_plasma_pod",
    "deeponet_plasma",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ext0520 benchmark configs from curated templates.")
    parser.add_argument("--template-dir", default="configs/benchmarkrun_ext0520/templates")
    parser.add_argument("--out-root", default="runs/benchmarkrun_ext0520/configs")
    parser.add_argument("--dataset-root", default="data/outputs_merged_td_csv_periodic_ext0520")
    parser.add_argument("--sizes", nargs="+", type=int, default=[27, 54, 78])
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "benchmark" not in payload:
        raise ValueError(f"{path} must contain a benchmark section")
    return payload


def main() -> int:
    args = _parse_args()
    template_dir = Path(args.template_dir)
    out_root = Path(args.out_root)
    for size in [int(v) for v in args.sizes]:
        for model in [str(v) for v in args.models]:
            src = template_dir / f"benchmark_ext0520_{model}.yaml"
            if not src.exists():
                raise FileNotFoundError(f"missing benchmark template: {src}")
            cfg = _load_yaml(src)
            benchmark = cfg["benchmark"]
            benchmark.setdefault("dataset", {})["root"] = str(args.dataset_root)
            benchmark["dataset"]["index_csv"] = f"index_{size}.csv"
            benchmark["output_dir"] = f"runs/benchmarkrun_ext0520/n{size}/{model}"
            benchmark.setdefault("eval", {})["protocol_variant"] = f"{model}_benchmarkrun_ext0520_n{size}"
            out = out_root / f"n{size}" / f"benchmark_ext0520_{model}_n{size}.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
