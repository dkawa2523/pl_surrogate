from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import time
import traceback
from pathlib import Path


MODEL_ORDER = (
    "global_mlp",
    "deeponet_pod",
    "unet",
    "unetpp",
    "unetpp_attn",
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

STATUS_FIELDS = (
    "dataset_size",
    "model_id",
    "status",
    "started_at",
    "finished_at",
    "seconds",
    "config",
    "log",
    "leaderboard",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ext0520 benchmark configs one at a time with low host load.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[78])
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument("--config-root", default="runs/benchmarkrun_ext0520/configs")
    parser.add_argument("--run-root", default="runs/benchmarkrun_ext0520")
    parser.add_argument("--status-csv", default="runs/benchmarkrun_ext0520/run_status.csv")
    parser.add_argument(
        "--load-profile",
        choices=("low", "normal"),
        default="low",
        help="Use low host load safeguards, or run normally while still executing models one at a time.",
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def _limit_host_threads() -> None:
    os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _set_below_normal_priority() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        below_normal = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, below_normal)
    except Exception:
        pass


def _read_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = []
        for raw in csv.DictReader(f):
            cleaned = {str(k).strip().strip('"'): str(v) for k, v in dict(raw).items() if k is not None}
            rows.append({field: cleaned.get(field, "") for field in STATUS_FIELDS})
        return rows


def _write_status(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(STATUS_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _upsert_status(path: Path, rows: list[dict[str, str]], row: dict[str, str]) -> None:
    key = (str(row["dataset_size"]), str(row["model_id"]))
    for idx, old in enumerate(rows):
        if (str(old.get("dataset_size", "")), str(old.get("model_id", ""))) == key:
            rows[idx] = row
            break
    else:
        rows.append(row)
    rows.sort(key=lambda item: (int(item.get("dataset_size", 0)), str(item.get("model_id", ""))))
    _write_status(path, rows)


def main() -> int:
    args = _parse_args()
    if args.load_profile == "low":
        _limit_host_threads()
        _set_below_normal_priority()

    from plasma_surrogate.benchmark.runner import BenchmarkRunner

    config_root = Path(args.config_root)
    run_root = Path(args.run_root)
    status_path = Path(args.status_csv)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_rows = _read_status(status_path)
    failures = 0

    for size in [int(v) for v in args.sizes]:
        for model in [str(v) for v in args.models]:
            cfg = config_root / f"n{size}" / f"benchmark_ext0520_{model}_n{size}.yaml"
            leaderboard = run_root / f"n{size}" / model / "leaderboard.csv"
            log = log_dir / f"n{size}_{model}.log"
            if leaderboard.exists():
                _upsert_status(
                    status_path,
                    status_rows,
                    {
                        "dataset_size": str(size),
                        "model_id": model,
                        "status": "skipped_existing",
                        "started_at": "",
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "seconds": "0.0",
                        "config": str(cfg),
                        "log": str(log),
                        "leaderboard": str(leaderboard),
                    },
                )
                print(f"[SKIP] n={size} model={model}", flush=True)
                continue
            if not cfg.exists():
                raise FileNotFoundError(f"missing benchmark config: {cfg}")

            started = time.strftime("%Y-%m-%d %H:%M:%S")
            start = time.time()
            _upsert_status(
                status_path,
                status_rows,
                {
                    "dataset_size": str(size),
                    "model_id": model,
                    "status": "running",
                    "started_at": started,
                    "finished_at": "",
                    "seconds": "",
                    "config": str(cfg),
                    "log": str(log),
                    "leaderboard": str(leaderboard),
                },
            )
            print(f"[RUN] n={size} model={model}", flush=True)
            status = "passed"
            with log.open("w", encoding="utf-8", errors="replace") as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    try:
                        BenchmarkRunner.from_yaml(cfg).run()
                    except BaseException:
                        traceback.print_exc()
                        status = "failed:exception"
            elapsed = time.time() - start
            if status == "passed" and not leaderboard.exists():
                status = "failed:no_leaderboard"
            if status.startswith("failed"):
                failures += 1
            _upsert_status(
                status_path,
                status_rows,
                {
                    "dataset_size": str(size),
                    "model_id": model,
                    "status": status,
                    "started_at": started,
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "seconds": f"{elapsed:.1f}",
                    "config": str(cfg),
                    "log": str(log),
                    "leaderboard": str(leaderboard),
                },
            )
            print(f"[{status.upper()}] n={size} model={model} seconds={elapsed:.1f}", flush=True)
            if status.startswith("failed") and bool(args.stop_on_failure):
                return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
