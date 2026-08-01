"""Run improvement cycle R2 experiments in fixed order with gate reporting."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plasma_surrogate.benchmark.runner import BenchmarkRunner  # noqa: E402


@dataclass
class RunScore:
    exp_id: str
    leaderboard_path: str
    dual: float | None
    extrap: float | None
    ne_extrap: float | None
    ni_extrap: float | None
    te_extrap: float | None
    phi_extrap: float | None
    min_var_extrap: float | None
    tephi_extrap_avg: float | None
    status: str
    error: str = ""


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data or {})


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=False, sort_keys=False), encoding="utf-8")


def _load_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"leaderboard has no rows: {path}")
    return dict(rows[0])


def _f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def _score_from_leaderboard(exp_id: str, path: Path) -> RunScore:
    row = _load_first_row(path)
    ne_extrap = _f(row, "test_r2_ne_plasma_extrap")
    ni_extrap = _f(row, "test_r2_ni_plasma_extrap")
    te_extrap = _f(row, "test_r2_Te_plasma_extrap")
    phi_extrap = _f(row, "test_r2_phi_plasma_extrap")
    return RunScore(
        exp_id=exp_id,
        leaderboard_path=str(path),
        dual=_f(row, "test_r2_plasma_mean_dual"),
        extrap=_f(row, "test_r2_plasma_mean_extrap"),
        ne_extrap=ne_extrap,
        ni_extrap=ni_extrap,
        te_extrap=te_extrap,
        phi_extrap=phi_extrap,
        min_var_extrap=min(ne_extrap, ni_extrap, te_extrap, phi_extrap),
        tephi_extrap_avg=0.5 * (te_extrap + phi_extrap),
        status="ok",
    )


def _rank(scores: list[RunScore]) -> list[RunScore]:
    return sorted(scores, key=lambda s: (s.dual, s.extrap, s.min_var_extrap), reverse=True)


def _track_of(exp_id: str) -> str:
    if exp_id.startswith("C"):
        return "coord_mlp_siren"
    if exp_id.startswith("D"):
        return "deeponet_plasma"
    if exp_id.startswith("U"):
        return "unet_family"
    return "unknown"


def _override_epochs(cfg: dict[str, Any], epochs: int) -> dict[str, Any]:
    out = dict(cfg)
    bench = dict(out.get("benchmark", {}))
    train = dict(bench.get("train", {}))
    for key, section in list(train.items()):
        if isinstance(section, dict) and "epochs" in section:
            sec = dict(section)
            sec["epochs"] = int(epochs)
            train[key] = sec
    bench["train"] = train
    out["benchmark"] = bench
    return out


def _set_output_dir(cfg: dict[str, Any], output_dir: str) -> dict[str, Any]:
    out = dict(cfg)
    bench = dict(out.get("benchmark", {}))
    bench["output_dir"] = str(output_dir)
    out["benchmark"] = bench
    return out


def _materialize_if_requested(*, python_bin: str, enabled: bool) -> None:
    if not enabled:
        return
    script = REPO_ROOT / "experiments" / "improvement_cycle_20260401" / "scripts" / "materialize_improvement_cycle_20260401_r2.py"
    subprocess.run([python_bin, str(script)], check=True, cwd=str(REPO_ROOT))


def _gate_status(track_scores: list[RunScore], *, track: str) -> str:
    if len(track_scores) == 0:
        return "SKIP(no-scores)"
    best = _rank(track_scores)[0]
    if track == "coord_mlp_siren":
        ok = best.dual >= 0.8262 and best.extrap >= 0.7154 and best.phi_extrap >= 0.683
        return (
            f"{'PASS' if ok else 'FAIL'} "
            f"(best={best.exp_id}, dual={best.dual:.6f}, extrap={best.extrap:.6f}, phi={best.phi_extrap:.6f})"
        )
    if track == "deeponet_plasma":
        by_id = {s.exp_id: s for s in track_scores}
        baseline = by_id.get("D7_deeponet_control")
        if baseline is None:
            return "SKIP(missing-baseline:D7)"
        candidates = [s for s in track_scores if s.exp_id != "D7_deeponet_control"]
        if len(candidates) == 0:
            return "SKIP(no-candidates)"
        best_candidate = _rank(candidates)[0]
        ok = (
            best_candidate.dual >= baseline.dual + 0.02
            and best_candidate.tephi_extrap_avg >= baseline.tephi_extrap_avg + 0.05
            and min(best_candidate.ne_extrap, best_candidate.ni_extrap)
            >= min(baseline.ne_extrap, baseline.ni_extrap) - 0.01
        )
        return (
            f"{'PASS' if ok else 'FAIL'} "
            f"(best={best_candidate.exp_id}, "
            f"dual_delta={best_candidate.dual - baseline.dual:.6f}, "
            f"tephi_avg_delta={best_candidate.tephi_extrap_avg - baseline.tephi_extrap_avg:.6f}, "
            f"density_floor_delta="
            f"{min(best_candidate.ne_extrap, best_candidate.ni_extrap) - min(baseline.ne_extrap, baseline.ni_extrap):.6f})"
        )
    if track == "unet_family":
        by_id = {s.exp_id: s for s in track_scores}
        baseline = by_id.get("U5_unetpp_attn_control")
        if baseline is None:
            return "SKIP(missing-baseline:U5)"
        candidates = [by_id[k] for k in ("U6_unetpp_attn_density_bias", "U7_unetpp_density_bias_port") if k in by_id]
        if len(candidates) == 0:
            return "SKIP(no-candidates)"
        passed = [
            c
            for c in candidates
            if (c.ne_extrap - baseline.ne_extrap) >= 0.03
            and (c.ni_extrap - baseline.ni_extrap) >= 0.03
            and (c.dual - baseline.dual) >= -0.01
        ]
        return f"{'PASS' if len(passed) > 0 else 'FAIL'} (passed={[c.exp_id for c in passed]})"
    return "SKIP(unknown-track)"


def _select_winner(track_scores: list[RunScore], *, track: str) -> str | None:
    if len(track_scores) == 0:
        return None
    if track == "coord_mlp_siren":
        valid = [s for s in track_scores if s.dual >= 0.8262 and s.extrap >= 0.7154 and s.phi_extrap >= 0.683]
    elif track == "deeponet_plasma":
        by_id = {s.exp_id: s for s in track_scores}
        baseline = by_id.get("D7_deeponet_control")
        if baseline is None:
            return None
        valid = [
            s
            for s in track_scores
            if s.exp_id != "D7_deeponet_control"
            and s.dual >= baseline.dual + 0.02
            and s.tephi_extrap_avg >= baseline.tephi_extrap_avg + 0.05
            and min(s.ne_extrap, s.ni_extrap) >= min(baseline.ne_extrap, baseline.ni_extrap) - 0.01
        ]
    elif track == "unet_family":
        by_id = {s.exp_id: s for s in track_scores}
        baseline = by_id.get("U5_unetpp_attn_control")
        if baseline is None:
            return None
        valid = [
            s
            for s in track_scores
            if s.exp_id in {"U6_unetpp_attn_density_bias", "U7_unetpp_density_bias_port"}
            and (s.ne_extrap - baseline.ne_extrap) >= 0.03
            and (s.ni_extrap - baseline.ni_extrap) >= 0.03
            and (s.dual - baseline.dual) >= -0.01
        ]
    else:
        valid = []
    if len(valid) == 0:
        return None
    return _rank(valid)[0].exp_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Run improvement-cycle 20260401 R2 matrix")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/experimental/improvement_cycle_20260401_r2/experiment_matrix.yaml"),
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path("tests/fixtures/generated/improvement_cycle_20260401_r2"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/improvement_cycle_20260401_r2"),
        help="Root for run artifacts and summary output",
    )
    parser.add_argument(
        "--dry-run-epochs",
        type=int,
        default=0,
        help="If >0, override train.*.epochs for smoke runs",
    )
    parser.add_argument(
        "--max-attempts-per-id",
        type=int,
        default=2,
        help="Mark experiment invalid after this many failures",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize fixtures before running",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python binary used to run materialize helper",
    )
    args = parser.parse_args()

    matrix_path = (REPO_ROOT / args.matrix).resolve()
    fixtures_dir = (REPO_ROOT / args.fixtures_dir).resolve()
    runs_root = (REPO_ROOT / args.runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    tmp_cfg_dir = runs_root / "_tmp_configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    _materialize_if_requested(python_bin=str(args.python_bin), enabled=bool(args.materialize))

    matrix = _load_yaml(matrix_path)
    experiments = [dict(e) for e in list(matrix.get("experiments", []))]
    if len(experiments) != 12:
        raise ValueError(f"R2 matrix must define 12 experiments, got {len(experiments)}")

    scores: list[RunScore] = []
    failure_counts: dict[str, int] = {}
    invalid_ids: set[str] = set()

    for exp in experiments:
        exp_id = str(exp.get("id", "")).strip()
        if not exp_id:
            raise ValueError("experiment.id must be non-empty")
        fixture = fixtures_dir / f"{exp_id}.yaml"
        if not fixture.exists():
            raise FileNotFoundError(f"fixture not found: {fixture}")

        print(f"\n[run] {exp_id}")
        success = False
        last_error = ""
        for attempt in range(1, max(int(args.max_attempts_per_id), 1) + 1):
            try:
                cfg_path = fixture
                if int(args.dry_run_epochs) > 0:
                    cfg = _load_yaml(fixture)
                    cfg = _override_epochs(cfg, int(args.dry_run_epochs))
                    dry_root = runs_root / f"dryrun_e{int(args.dry_run_epochs)}"
                    cfg = _set_output_dir(cfg, str((dry_root / exp_id).as_posix()))
                    cfg_path = tmp_cfg_dir / f"{exp_id}.yaml"
                    _write_yaml(cfg_path, cfg)
                result = BenchmarkRunner.from_yaml(cfg_path).run()
                lb_path = Path(result.leaderboard_path).resolve()
                score = _score_from_leaderboard(exp_id, lb_path)
                scores.append(score)
                success = True
                print(
                    f"[ok] {exp_id} dual={score.dual:.6f} extrap={score.extrap:.6f} "
                    f"min_var_extrap={score.min_var_extrap:.6f}"
                )
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"[fail] {exp_id} attempt={attempt}: {last_error}")
                print(traceback.format_exc())

        if not success:
            failure_counts[exp_id] = int(failure_counts.get(exp_id, 0)) + max(int(args.max_attempts_per_id), 1)
            if failure_counts[exp_id] >= max(int(args.max_attempts_per_id), 1):
                invalid_ids.add(exp_id)
                scores.append(
                    RunScore(
                        exp_id=exp_id,
                        leaderboard_path="",
                        dual=None,
                        extrap=None,
                        ne_extrap=None,
                        ni_extrap=None,
                        te_extrap=None,
                        phi_extrap=None,
                        min_var_extrap=None,
                        tephi_extrap_avg=None,
                        status="invalid",
                        error=last_error,
                    )
                )
                print(f"[invalid] {exp_id} marked invalid after {failure_counts[exp_id]} failures")

        track_scores = {
            "coord_mlp_siren": [s for s in scores if s.status == "ok" and _track_of(s.exp_id) == "coord_mlp_siren"],
            "deeponet_plasma": [s for s in scores if s.status == "ok" and _track_of(s.exp_id) == "deeponet_plasma"],
            "unet_family": [s for s in scores if s.status == "ok" and _track_of(s.exp_id) == "unet_family"],
        }
        print("[gate] coord   :", _gate_status(track_scores["coord_mlp_siren"], track="coord_mlp_siren"))
        print("[gate] deeponet:", _gate_status(track_scores["deeponet_plasma"], track="deeponet_plasma"))
        print("[gate] unet    :", _gate_status(track_scores["unet_family"], track="unet_family"))

    by_track = {
        "coord_mlp_siren": [s for s in scores if s.status == "ok" and _track_of(s.exp_id) == "coord_mlp_siren"],
        "deeponet_plasma": [s for s in scores if s.status == "ok" and _track_of(s.exp_id) == "deeponet_plasma"],
        "unet_family": [s for s in scores if s.status == "ok" and _track_of(s.exp_id) == "unet_family"],
    }
    winners = {
        "coord_mlp_siren": _select_winner(by_track["coord_mlp_siren"], track="coord_mlp_siren"),
        "deeponet_plasma": _select_winner(by_track["deeponet_plasma"], track="deeponet_plasma"),
        "unet_family": _select_winner(by_track["unet_family"], track="unet_family"),
    }

    summary = {
        "matrix": str(matrix_path),
        "fixtures_dir": str(fixtures_dir),
        "runs_root": str(runs_root),
        "dry_run_epochs": int(args.dry_run_epochs),
        "invalid_ids": sorted(list(invalid_ids)),
        "scores": [asdict(s) for s in scores],
        "winners": winners,
    }
    summary_path = runs_root / (
        f"summary_dryrun_e{int(args.dry_run_epochs)}.json"
        if int(args.dry_run_epochs) > 0
        else "summary_full.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary] {summary_path}")
    print(f"[winners] {json.dumps(winners, ensure_ascii=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
