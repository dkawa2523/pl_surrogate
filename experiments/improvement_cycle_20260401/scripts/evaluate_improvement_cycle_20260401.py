"""Evaluate improvement-cycle acceptance gates from leaderboard CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"leaderboard has no rows: {path}")
    return dict(rows[0])


def _f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def _fmt(name: str, value: float, threshold: float, *, larger_is_better: bool = True) -> str:
    ok = value >= threshold if larger_is_better else value <= threshold
    state = "PASS" if ok else "FAIL"
    return f"{state:4s} {name}: value={value:.6f} threshold={threshold:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate improvement-cycle acceptance gates")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline leaderboard.csv")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate leaderboard.csv")
    parser.add_argument(
        "--track",
        type=str,
        choices=["deeponet", "unet_family", "coord_mlp_siren"],
        required=True,
        help="Gate set to evaluate",
    )
    args = parser.parse_args()

    base = _load_first_row(args.baseline)
    cand = _load_first_row(args.candidate)

    dual_delta = _f(cand, "test_r2_plasma_mean_dual") - _f(base, "test_r2_plasma_mean_dual")
    extrap_delta = _f(cand, "test_r2_plasma_mean_extrap") - _f(base, "test_r2_plasma_mean_extrap")
    te_extrap_delta = _f(cand, "test_r2_Te_plasma_extrap") - _f(base, "test_r2_Te_plasma_extrap")
    ne_extrap_delta = _f(cand, "test_r2_ne_plasma_extrap") - _f(base, "test_r2_ne_plasma_extrap")
    ni_extrap_delta = _f(cand, "test_r2_ni_plasma_extrap") - _f(base, "test_r2_ni_plasma_extrap")
    phi_extrap_delta = _f(cand, "test_r2_phi_plasma_extrap") - _f(base, "test_r2_phi_plasma_extrap")

    print(f"baseline : {args.baseline}")
    print(f"candidate: {args.candidate}")
    print(f"track    : {args.track}")
    print("")

    if args.track == "deeponet":
        checks = [
            _fmt("dual_delta", dual_delta, 0.15),
            _fmt("Te_extrap_delta", te_extrap_delta, 0.20),
        ]
    elif args.track == "unet_family":
        checks = [
            _fmt("ne_extrap_delta", ne_extrap_delta, 0.03),
            _fmt("ni_extrap_delta", ni_extrap_delta, 0.03),
            _fmt("dual_delta_min", dual_delta, -0.01),
        ]
    else:
        checks = [
            _fmt("extrap_delta", extrap_delta, 0.05),
            _fmt("phi_extrap_delta", phi_extrap_delta, 0.08),
        ]

    for line in checks:
        print(line)

    overall = all(line.startswith("PASS") for line in checks)
    print("")
    print(f"overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
