from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import time
import traceback
from pathlib import Path


CORE_MODELS = (
    "global_mlp",
    "deeponet_pod",
    "unet",
    "unetpp",
    "unetpp_attn",
    "fno",
    "ffno",
    "cno",
    "cno_operator_unet",
)
SIZE_ORDER = ("smoke", "full")
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
    parser = argparse.ArgumentParser(description="Run ICP Stage4 Core4 benchmark configs one model at a time.")
    parser.add_argument("--sizes", nargs="+", choices=SIZE_ORDER, default=["smoke"])
    parser.add_argument("--models", nargs="+", choices=CORE_MODELS, default=list(CORE_MODELS))
    parser.add_argument("--config-root", default="configs/experimental/icp_stage4/generated")
    parser.add_argument("--run-root", default="runs/icp_stage4_core4")
    parser.add_argument("--status-csv", default="runs/icp_stage4_core4/run_status.csv")
    parser.add_argument("--load-profile", choices=("low", "normal"), default="low")
    parser.add_argument("--force", action="store_true", help="Run even when the leaderboard already exists.")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def _prepare_imports() -> None:
    root = Path.cwd()
    for path in (root / "src", root):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)


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
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)
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


def _status_sort_key(row: dict[str, str]) -> tuple[int, int]:
    size = str(row.get("dataset_size", ""))
    model = str(row.get("model_id", ""))
    return (
        SIZE_ORDER.index(size) if size in SIZE_ORDER else len(SIZE_ORDER),
        CORE_MODELS.index(model) if model in CORE_MODELS else len(CORE_MODELS),
    )


def _upsert_status(path: Path, rows: list[dict[str, str]], row: dict[str, str]) -> None:
    key = (str(row["dataset_size"]), str(row["model_id"]))
    for idx, old in enumerate(rows):
        if (str(old.get("dataset_size", "")), str(old.get("model_id", ""))) == key:
            rows[idx] = row
            break
    else:
        rows.append(row)
    rows.sort(key=_status_sort_key)
    _write_status(path, rows)


def main() -> int:
    args = _parse_args()
    _prepare_imports()
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

    for size in args.sizes:
        for model in args.models:
            cfg = config_root / size / f"benchmark_icp_stage4_core4_{size}_{model}.yaml"
            leaderboard = run_root / size / model / "leaderboard.csv"
            log = log_dir / f"{size}_{model}.log"
            if leaderboard.exists() and not args.force:
                _upsert_status(
                    status_path,
                    status_rows,
                    {
                        "dataset_size": size,
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
                print(f"[SKIP] size={size} model={model}", flush=True)
                continue
            if not cfg.exists():
                raise FileNotFoundError(f"missing benchmark config: {cfg}")

            started = time.strftime("%Y-%m-%d %H:%M:%S")
            start = time.time()
            _upsert_status(
                status_path,
                status_rows,
                {
                    "dataset_size": size,
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
            print(f"[RUN] size={size} model={model}", flush=True)
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
                    "dataset_size": size,
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
            print(f"[{status.upper()}] size={size} model={model} seconds={elapsed:.1f}", flush=True)
            if status.startswith("failed") and args.stop_on_failure:
                return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
