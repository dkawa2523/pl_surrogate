"""Evaluate Improvement Cycle R2 results and acceptance gates."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunScore:
    exp_id: str
    path: Path
    dual: float
    extrap: float
    ne_extrap: float
    ni_extrap: float
    te_extrap: float
    phi_extrap: float

    @property
    def min_var_extrap(self) -> float:
        return min(self.ne_extrap, self.ni_extrap, self.te_extrap, self.phi_extrap)

    @property
    def tephi_extrap_avg(self) -> float:
        return 0.5 * (self.te_extrap + self.phi_extrap)


def _read_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"leaderboard has no rows: {path}")
    return dict(rows[0])


def _f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def _load_score(exp_id: str, root: Path) -> RunScore:
    lb = root / exp_id / "leaderboard.csv"
    row = _read_first_row(lb)
    return RunScore(
        exp_id=exp_id,
        path=lb,
        dual=_f(row, "test_r2_plasma_mean_dual"),
        extrap=_f(row, "test_r2_plasma_mean_extrap"),
        ne_extrap=_f(row, "test_r2_ne_plasma_extrap"),
        ni_extrap=_f(row, "test_r2_ni_plasma_extrap"),
        te_extrap=_f(row, "test_r2_Te_plasma_extrap"),
        phi_extrap=_f(row, "test_r2_phi_plasma_extrap"),
    )


def _rank(scores: list[RunScore]) -> list[RunScore]:
    return sorted(
        scores,
        key=lambda s: (s.dual, s.extrap, s.min_var_extrap),
        reverse=True,
    )


def _fmt_score(s: RunScore) -> str:
    return (
        f"{s.exp_id:35s} dual={s.dual:.6f} extrap={s.extrap:.6f} "
        f"min_var_extrap={s.min_var_extrap:.6f} "
        f"(ne={s.ne_extrap:.6f}, ni={s.ni_extrap:.6f}, Te={s.te_extrap:.6f}, phi={s.phi_extrap:.6f})"
    )


def _print_track(track_name: str, scores: list[RunScore]) -> None:
    print(f"\n[{track_name}]")
    for s in _rank(scores):
        print(_fmt_score(s))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate improvement-cycle R2 runs")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/improvement_cycle_20260401_r2"),
        help="Directory containing per-experiment benchmark outputs",
    )
    args = parser.parse_args()

    root = Path(args.runs_root).resolve()
    track_coord_ids = [
        "C3_coord_siren_control",
        "C4_coord_siren_gated_affine",
        "C5_coord_siren_gated_affine_sel",
        "C6_coord_siren_gated_affine_sel_mt",
        "C7_coord_siren_gated_affine_w0_res",
    ]
    track_deeponet_ids = [
        "D7_deeponet_control",
        "D8_deeponet_tephi_sel",
        "D9_deeponet_tephi_sel_mt",
        "D10_deeponet_fused_learned_mix",
    ]
    track_unet_ids = [
        "U5_unetpp_attn_control",
        "U6_unetpp_attn_density_bias",
        "U7_unetpp_density_bias_port",
    ]

    failures: list[str] = []

    def _load_many(ids: list[str]) -> list[RunScore]:
        out: list[RunScore] = []
        for exp_id in ids:
            try:
                out.append(_load_score(exp_id, root))
            except Exception as exc:
                failures.append(f"{exp_id}: {exc}")
        return out

    coord_scores = _load_many(track_coord_ids)
    deeponet_scores = _load_many(track_deeponet_ids)
    unet_scores = _load_many(track_unet_ids)

    if failures:
        print("[missing_or_failed_runs]")
        for msg in failures:
            print(msg)

    if coord_scores:
        _print_track("coord_mlp_siren", coord_scores)
    if deeponet_scores:
        _print_track("deeponet_plasma", deeponet_scores)
    if unet_scores:
        _print_track("unet_family", unet_scores)

    print("\n[gates]")
    if coord_scores:
        best_coord = _rank(coord_scores)[0]
        coord_ok = (
            best_coord.dual >= 0.8262
            and best_coord.extrap >= 0.7154
            and best_coord.phi_extrap >= 0.683
        )
        print(
            f"coord stop gate: {'PASS' if coord_ok else 'FAIL'} "
            f"(dual={best_coord.dual:.6f}, extrap={best_coord.extrap:.6f}, phi_extrap={best_coord.phi_extrap:.6f})"
        )

    if deeponet_scores:
        by_id = {s.exp_id: s for s in deeponet_scores}
        baseline = by_id.get("D7_deeponet_control")
        best = _rank([s for s in deeponet_scores if s.exp_id != "D7_deeponet_control"] or deeponet_scores)[0]
        if baseline is not None:
            gate_dual = best.dual >= (baseline.dual + 0.02)
            gate_tephi = best.tephi_extrap_avg >= (baseline.tephi_extrap_avg + 0.05)
            gate_density_floor = min(best.ne_extrap, best.ni_extrap) >= (min(baseline.ne_extrap, baseline.ni_extrap) - 0.01)
            deep_ok = gate_dual and gate_tephi and gate_density_floor
            print(
                f"deeponet stop gate: {'PASS' if deep_ok else 'FAIL'} "
                f"(dual_delta={best.dual - baseline.dual:.6f}, "
                f"tephi_avg_delta={best.tephi_extrap_avg - baseline.tephi_extrap_avg:.6f}, "
                f"density_floor_delta={min(best.ne_extrap, best.ni_extrap) - min(baseline.ne_extrap, baseline.ni_extrap):.6f})"
            )
        else:
            print("deeponet stop gate: SKIP (missing D7 baseline)")

    if unet_scores:
        by_id = {s.exp_id: s for s in unet_scores}
        baseline = by_id.get("U5_unetpp_attn_control")
        candidates = [by_id[k] for k in ("U6_unetpp_attn_density_bias", "U7_unetpp_density_bias_port") if k in by_id]
        if baseline is not None and candidates:
            passes = []
            for c in candidates:
                ok = (
                    (c.ne_extrap - baseline.ne_extrap) >= 0.03
                    and (c.ni_extrap - baseline.ni_extrap) >= 0.03
                    and (c.dual - baseline.dual) >= -0.01
                )
                passes.append(ok)
                print(
                    f"{c.exp_id} gate: {'PASS' if ok else 'FAIL'} "
                    f"(ne_delta={c.ne_extrap - baseline.ne_extrap:.6f}, "
                    f"ni_delta={c.ni_extrap - baseline.ni_extrap:.6f}, "
                    f"dual_delta={c.dual - baseline.dual:.6f})"
                )
            print(f"unet stop gate: {'PASS' if any(passes) else 'FAIL'}")
        else:
            print("unet stop gate: SKIP (missing baseline/candidates)")

    print("\n[adoption]")
    if coord_scores:
        print(f"coord winner    : {_rank(coord_scores)[0].exp_id}")
    if deeponet_scores:
        print(f"deeponet winner : {_rank(deeponet_scores)[0].exp_id}")
    if unet_scores:
        print(f"unet winner     : {_rank(unet_scores)[0].exp_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
