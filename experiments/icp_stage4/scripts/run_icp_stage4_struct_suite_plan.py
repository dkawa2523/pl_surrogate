from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import runpy
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from icp_stage4_protocol_reliability import (
    assess_protocol_reliability,
    enforce_structure_study_protocol,
    read_validation_selection,
    validation_row_is_selectable,
)


GPU_MODELS = ("deeponet_pod", "unet", "unetpp", "unetpp_attn", "fno", "ffno", "cno")
STATUS_FIELDS = (
    "stage",
    "model_id",
    "recipe",
    "status",
    "started_at",
    "finished_at",
    "seconds",
    "config",
    "log",
    "leaderboard",
)


TUNING_RECIPES: dict[str, list[dict[str, Any]]] = {
    "deeponet_pod": [
        {"recipe": "pod_r48_h160_lr2e4", "lr": 2.0e-4, "hidden_dim": 160, "latent_dim": 160, "rank": 48},
        {"recipe": "pod_r64_h192_lr15e4", "lr": 1.5e-4, "hidden_dim": 192, "latent_dim": 192, "rank": 64},
    ],
    "unet": [
        {"recipe": "unet_ch24_lr3e4", "lr": 3.0e-4, "base_channels": 24},
        {"recipe": "unet_ch40_lr2e4", "lr": 2.0e-4, "base_channels": 40},
    ],
    "unetpp": [
        {"recipe": "unetpp_ch24_lr3e4", "lr": 3.0e-4, "base_channels": 24},
        {"recipe": "unetpp_ch40_lr2e4", "lr": 2.0e-4, "base_channels": 40},
    ],
    "unetpp_attn": [
        {"recipe": "unetppattn_ch24_red4_lr3e4", "lr": 3.0e-4, "base_channels": 24, "reduction": 4},
        {"recipe": "unetppattn_ch40_red2_lr2e4", "lr": 2.0e-4, "base_channels": 40, "reduction": 2},
    ],
    "fno": [
        {"recipe": "fno_m10_w40_lr4e4", "lr": 4.0e-4, "n_modes": 10, "width": 40},
        {"recipe": "fno_m14_w56_lr3e4", "lr": 3.0e-4, "n_modes": 14, "width": 56},
    ],
    "ffno": [
        {"recipe": "ffno_m12_w48_lr4e4", "lr": 4.0e-4, "n_modes": 12, "width": 48},
        {"recipe": "ffno_m18_w72_lr3e4", "lr": 3.0e-4, "n_modes": 18, "width": 72},
    ],
    "cno": [
        {"recipe": "cno_w48_l4_lr4e4", "lr": 4.0e-4, "width": 48, "n_layers": 4},
        {"recipe": "cno_w80_l4_lr3e4", "lr": 3.0e-4, "width": 80, "n_layers": 4},
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICP structural full, tuning, and evaluation behind an idle gate.")
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--models", nargs="+", default=list(GPU_MODELS))
    parser.add_argument("--status-csv", required=True)
    parser.add_argument("--summary-dir", default=None)
    parser.add_argument("--tuning-config-root", default=None)
    parser.add_argument("--tuning-status-csv", default=None)
    parser.add_argument("--tuning-top-n", type=int, default=3)
    parser.add_argument("--tuning-epochs", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--max-gpu-util", type=float, default=10.0)
    parser.add_argument("--max-gpu-mem-mib", type=float, default=2200.0)
    parser.add_argument("--wait-command-absent", action="append", default=[])
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def _set_below_normal_priority() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        below_normal = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)
    except Exception:
        pass


def _limit_threads() -> None:
    os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _gpu_state() -> tuple[float, float]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    first = out.strip().splitlines()[0]
    mem_raw, util_raw = [part.strip() for part in first.split(",", 1)]
    return float(mem_raw), float(util_raw)


def _process_rows() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return []
    if not out:
        return []
    try:
        raw = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    return [dict(row) for row in raw if isinstance(row, dict)]


def _matching_processes(patterns: list[str]) -> list[dict[str, Any]]:
    if not patterns:
        return []
    self_pid = os.getpid()
    self_script = Path(__file__).name.lower()
    needles = [str(pattern).lower() for pattern in patterns if str(pattern).strip()]
    matches: list[dict[str, Any]] = []
    for row in _process_rows():
        try:
            pid = int(row.get("ProcessId", -1))
        except Exception:
            pid = -1
        if pid == self_pid:
            continue
        cmd = str(row.get("CommandLine") or "")
        lower = cmd.lower()
        if self_script in lower:
            continue
        if any(needle in lower for needle in needles):
            matches.append({"pid": pid, "command": cmd})
    return matches


def _wait_until_idle(args: argparse.Namespace, *, label: str) -> None:
    consecutive = 0
    poll = max(int(args.poll_seconds), 5)
    needed = max(int(args.idle_checks), 1)
    while consecutive < needed:
        blocked = _matching_processes(list(args.wait_command_absent or []))
        mem_mib, util = _gpu_state()
        idle = not blocked and mem_mib <= float(args.max_gpu_mem_mib) and util <= float(args.max_gpu_util)
        block_text = ""
        if blocked:
            block_text = " wait_processes=" + ",".join(str(row["pid"]) for row in blocked[:6])
        status = "idle" if idle else "busy"
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] label={label} gate={status} "
            f"gpu_mem_mib={mem_mib:.0f} gpu_util={util:.0f}{block_text}",
            flush=True,
        )
        consecutive = consecutive + 1 if idle else 0
        if consecutive < needed:
            time.sleep(poll)


def _read_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{field: str(row.get(field, "")) for field in STATUS_FIELDS} for row in csv.DictReader(f)]


def _write_status(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(STATUS_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _upsert_status(path: Path, row: dict[str, str]) -> None:
    rows = _read_status(path)
    key = (str(row["stage"]), str(row["model_id"]), str(row["recipe"]))
    for idx, old in enumerate(rows):
        if (str(old.get("stage", "")), str(old.get("model_id", "")), str(old.get("recipe", ""))) == key:
            rows[idx] = row
            break
    else:
        rows.append(row)
    _write_status(path, rows)


def _leaderboard_for_config(cfg_path: Path) -> Path:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    output_dir = Path(str(dict(cfg.get("benchmark", {})).get("output_dir", "")))
    return output_dir / "leaderboard.csv"


def _run_config(
    cfg_path: Path,
    *,
    stage: str,
    model: str,
    recipe: str,
    status_csv: Path,
    log_dir: Path,
    stop_on_failure: bool,
) -> bool:
    from plasma_surrogate.benchmark.runner import BenchmarkRunner

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    output_dir = Path(str(dict(cfg.get("benchmark", {})).get("output_dir", "")))
    leaderboard = output_dir / "leaderboard.csv"
    log = log_dir / f"{stage}_{model}{('__' + recipe) if recipe else ''}.log"
    if leaderboard.exists():
        _upsert_status(
            status_csv,
            {
                "stage": stage,
                "model_id": model,
                "recipe": recipe,
                "status": "skipped_existing",
                "started_at": "",
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seconds": "0.0",
                "config": str(cfg_path),
                "log": str(log),
                "leaderboard": str(leaderboard),
            },
        )
        print(f"[SKIP] stage={stage} model={model} recipe={recipe}", flush=True)
        return True

    # This runner is specifically the unseen-structure study.  Reject legacy
    # interp-primary configs before an expensive training job starts.
    enforce_structure_study_protocol(cfg)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    start = time.time()
    _upsert_status(
        status_csv,
        {
            "stage": stage,
            "model_id": model,
            "recipe": recipe,
            "status": "running",
            "started_at": started,
            "finished_at": "",
            "seconds": "",
            "config": str(cfg_path),
            "log": str(log),
            "leaderboard": str(leaderboard),
        },
    )
    print(f"[RUN] stage={stage} model={model} recipe={recipe}", flush=True)
    status = "passed"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", errors="replace") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                BenchmarkRunner.from_yaml(cfg_path).run()
            except BaseException:
                traceback.print_exc()
                status = "failed:exception"
            finally:
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
    elapsed = time.time() - start
    if status == "passed" and not leaderboard.exists():
        status = "failed:no_leaderboard"
    _upsert_status(
        status_csv,
        {
            "stage": stage,
            "model_id": model,
            "recipe": recipe,
            "status": status,
            "started_at": started,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": f"{elapsed:.1f}",
            "config": str(cfg_path),
            "log": str(log),
            "leaderboard": str(leaderboard),
        },
    )
    print(f"[{status.upper()}] stage={stage} model={model} recipe={recipe} seconds={elapsed:.1f}", flush=True)
    if status.startswith("failed") and stop_on_failure:
        raise RuntimeError(f"{stage}/{model}/{recipe} failed; see {log}")
    return not status.startswith("failed")


def _run_summary(args: argparse.Namespace, *, sizes: list[str]) -> Path:
    summary_dir = Path(args.summary_dir or (Path(args.run_root) / "summary"))
    argv = [
        "experiments/icp_stage4/scripts/summarize_icp_stage4_core4_benchmarks.py",
        "--sizes",
        *sizes,
        "--models",
        *list(args.models),
        "--config-root",
        str(args.config_root),
        "--run-root",
        str(args.run_root),
        "--out-dir",
        str(summary_dir),
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        try:
            runpy.run_path(argv[0], run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv
    return summary_dir / "comparison_summary.csv"


def _as_float(raw: Any) -> float:
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _select_top_models(summary_csv: Path, *, top_n: int) -> list[str]:
    rows: list[dict[str, str]] = []
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("dataset_size", "")) != "full":
                continue
            if str(row.get("model_id", "")) not in GPU_MODELS:
                continue
            if str(row.get("status", "")).strip().lower() not in {"ok", "passed"}:
                continue
            value = _as_float(row.get("validation_selection_score"))
            if not math.isfinite(value):
                continue
            if not validation_row_is_selectable(row):
                continue
            rows.append(row)
    rows.sort(key=lambda row: _as_float(row.get("validation_selection_score")), reverse=True)
    return [str(row["model_id"]) for row in rows[: max(int(top_n), 1)]]


def _set_lr(model_cfg: dict[str, Any], model: str, lr: float) -> None:
    train_cfg = model_cfg.setdefault("benchmark", {}).setdefault("train", {})
    family_cfg = train_cfg.setdefault(model, {})
    family_cfg["lr"] = float(lr)
    optimizer = family_cfg.setdefault("optimizer", {})
    if isinstance(optimizer, dict):
        optimizer["lr"] = float(lr)


def _set_epochs(cfg: dict[str, Any], model: str, epochs: int) -> None:
    train_cfg = cfg.setdefault("benchmark", {}).setdefault("train", {})
    if model in train_cfg and isinstance(train_cfg[model], dict):
        train_cfg[model]["epochs"] = int(epochs)


def _apply_recipe(cfg: dict[str, Any], *, model: str, recipe: dict[str, Any]) -> None:
    train_cfg = cfg.setdefault("benchmark", {}).setdefault("train", {})
    family_cfg = train_cfg.setdefault(model, {})
    model_cfg = family_cfg.setdefault("model_cfg", {})
    if "lr" in recipe:
        _set_lr(cfg, model, float(recipe["lr"]))
    if model == "deeponet_pod":
        if "hidden_dim" in recipe:
            model_cfg["hidden_dim"] = int(recipe["hidden_dim"])
        if "latent_dim" in recipe:
            model_cfg["latent_dim"] = int(recipe["latent_dim"])
        if "rank" in recipe:
            model_cfg.setdefault("basis", {})["rank"] = int(recipe["rank"])
    elif model in {"unet", "unetpp", "unetpp_attn"}:
        conv = model_cfg.setdefault("conv_cfg", {})
        if "base_channels" in recipe:
            conv["base_channels"] = int(recipe["base_channels"])
        if model == "unetpp_attn":
            attn = conv.setdefault("attention_cfg", {})
            attn["enabled"] = True
            if "reduction" in recipe:
                attn["reduction"] = int(recipe["reduction"])
    elif model in {"fno", "ffno"}:
        if "n_modes" in recipe:
            model_cfg["n_modes"] = int(recipe["n_modes"])
            model_cfg["fno_n_modes"] = int(recipe["n_modes"])
        spectral = model_cfg.setdefault("spectral_cfg", {})
        if "width" in recipe:
            spectral["width"] = int(recipe["width"])
    elif model == "cno":
        cno = model_cfg.setdefault("cno_cfg", {})
        if "width" in recipe:
            cno["width"] = int(recipe["width"])
        if "n_layers" in recipe:
            cno["n_layers"] = int(recipe["n_layers"])


def _make_tuning_configs(args: argparse.Namespace, *, selected_models: list[str]) -> list[tuple[str, str, Path]]:
    config_root = Path(args.config_root)
    tuning_config_root = Path(args.tuning_config_root or (config_root.parent / f"{config_root.name}_tuning"))
    out_dir = tuning_config_root / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[str, str, Path]] = []
    for model in selected_models:
        base_path = config_root / "full" / f"benchmark_icp_stage4_core4_full_{model}.yaml"
        if not base_path.exists():
            print(f"[TUNE SKIP] missing base config for {model}: {base_path}", flush=True)
            continue
        with base_path.open("r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
        for recipe in TUNING_RECIPES.get(model, []):
            recipe_name = str(recipe["recipe"])
            cfg = deepcopy(base_cfg)
            bench = cfg.setdefault("benchmark", {})
            bench["output_dir"] = str(Path(args.run_root) / "tuning" / f"{model}__{recipe_name}").replace("\\", "/")
            eval_cfg = bench.setdefault("eval", {})
            eval_cfg["protocol_variant"] = f"icp_stage4_tuning_{model}_{recipe_name}"
            _set_epochs(cfg, model, int(args.tuning_epochs))
            _apply_recipe(cfg, model=model, recipe=recipe)
            out_path = out_dir / f"benchmark_icp_stage4_tuning_{model}__{recipe_name}.yaml"
            with out_path.open("w", encoding="utf-8", newline="\n") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
            generated.append((model, recipe_name, out_path))
            print(f"[TUNE CONFIG] {out_path}", flush=True)
    return generated


def _collect_tuning_summary(items: list[tuple[str, str, Path]], *, out_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for model, recipe, cfg_path in items:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        output_dir = Path(str(dict(cfg.get("benchmark", {})).get("output_dir", "")))
        leaderboard = output_dir / "leaderboard.csv"
        if not leaderboard.exists():
            rows.append({"model_id": model, "recipe": recipe, "status": "missing_leaderboard"})
            continue
        with leaderboard.open("r", encoding="utf-8-sig", newline="") as f:
            data = list(csv.DictReader(f))
        if not data:
            rows.append({"model_id": model, "recipe": recipe, "status": "empty_leaderboard"})
            continue
        row = dict(data[0])
        assessment = assess_protocol_reliability(
            output_dir,
            leaderboard_row=row,
            config_path=cfg_path,
        )
        validation = read_validation_selection(
            output_dir,
            model_id=str(row.get("model_id") or model),
            primary_split=str(assessment.get("primary_split_effective", "")),
        )
        row.update(assessment)
        row.update(validation)
        row["model_id"] = model
        row["recipe"] = recipe
        if not validation_row_is_selectable(row):
            row["status"] = "unreliable_selection"
        else:
            row["status"] = "ok"
        rows.append(row)
    keys: list[str] = []
    for key in ("model_id", "recipe", "status", "primary_metric", "primary_metric_value", "primary_metric_reliable"):
        if key not in keys:
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    root = Path.cwd()
    for path in (root / "src", root):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)
    _set_below_normal_priority()
    _limit_threads()

    run_root = Path(args.run_root)
    logs = run_root / "logs"
    plan_status = run_root / "plan_status.csv"
    baseline_status = Path(args.status_csv)
    models = [str(model) for model in args.models if str(model) in GPU_MODELS]

    print(f"[PLAN] full models={models}", flush=True)
    for model in models:
        cfg_path = Path(args.config_root) / "full" / f"benchmark_icp_stage4_core4_full_{model}.yaml"
        if not _leaderboard_for_config(cfg_path).exists():
            _wait_until_idle(args, label=f"full:{model}")
        _run_config(
            cfg_path,
            stage="full",
            model=model,
            recipe="",
            status_csv=plan_status,
            log_dir=logs,
            stop_on_failure=bool(args.stop_on_failure),
        )

    summary_csv = _run_summary(args, sizes=["full"])
    selected = _select_top_models(summary_csv, top_n=int(args.tuning_top_n))
    print(f"[PLAN] selected_for_tuning={selected}", flush=True)
    if not selected:
        print("[PLAN] no reliable full rows for tuning; stopping after full summary", flush=True)
        return 0

    tuning_items = _make_tuning_configs(args, selected_models=selected)
    tuning_status = Path(args.tuning_status_csv or (run_root / "tuning_status.csv"))
    for model, recipe, cfg_path in tuning_items:
        if not _leaderboard_for_config(cfg_path).exists():
            _wait_until_idle(args, label=f"tuning:{model}:{recipe}")
        _run_config(
            cfg_path,
            stage="tuning",
            model=model,
            recipe=recipe,
            status_csv=tuning_status,
            log_dir=logs,
            stop_on_failure=bool(args.stop_on_failure),
        )

    tuning_summary = run_root / "summary" / "tuning_summary.csv"
    _collect_tuning_summary(tuning_items, out_path=tuning_summary)
    print(f"[PLAN] tuning_summary={tuning_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
