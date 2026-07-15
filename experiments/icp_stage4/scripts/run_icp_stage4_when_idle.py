from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("unet", "unetpp", "unetpp_attn", "fno", "ffno", "cno")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for local GPU/process idle, then run ICP Stage4 benchmarks.")
    parser.add_argument("--sizes", nargs="+", default=["full"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--status-csv", required=True)
    parser.add_argument("--runner-script", default="experiments/icp_stage4/scripts/run_icp_stage4_core4_benchmarks.py")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--max-gpu-util", type=float, default=10.0)
    parser.add_argument("--max-gpu-mem-mib", type=float, default=2200.0)
    parser.add_argument(
        "--wait-command-absent",
        action="append",
        default=[],
        help="Do not start while any process command line contains this text.",
    )
    parser.add_argument("--load-profile", choices=("low", "normal"), default="low")
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


def _wait_until_idle(args: argparse.Namespace) -> None:
    consecutive = 0
    poll = max(int(args.poll_seconds), 5)
    needed = max(int(args.idle_checks), 1)
    while consecutive < needed:
        blocked = _matching_processes(list(args.wait_command_absent or []))
        mem_mib, util = _gpu_state()
        idle = not blocked and mem_mib <= float(args.max_gpu_mem_mib) and util <= float(args.max_gpu_util)
        status = "idle" if idle else "busy"
        block_text = ""
        if blocked:
            shown = ", ".join(str(row["pid"]) for row in blocked[:5])
            block_text = f" wait_processes={shown}"
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] gate={status} "
            f"gpu_mem_mib={mem_mib:.0f} gpu_util={util:.0f}{block_text}",
            flush=True,
        )
        consecutive = consecutive + 1 if idle else 0
        if consecutive < needed:
            time.sleep(poll)


def _run_benchmarks(args: argparse.Namespace) -> int:
    runner = Path(args.runner_script)
    if not runner.exists():
        raise FileNotFoundError(f"runner script not found: {runner}")
    argv = [
        str(runner),
        "--sizes",
        *[str(v) for v in args.sizes],
        "--models",
        *[str(v) for v in args.models],
        "--config-root",
        str(args.config_root),
        "--run-root",
        str(args.run_root),
        "--status-csv",
        str(args.status_csv),
        "--load-profile",
        str(args.load_profile),
    ]
    if bool(args.stop_on_failure):
        argv.append("--stop-on-failure")
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        try:
            runpy.run_path(str(runner), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            return int(code)
    finally:
        sys.argv = old_argv


def main() -> int:
    args = _parse_args()
    root = Path.cwd()
    for path in (root / "src", root):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)
    if str(args.load_profile) == "low":
        _set_below_normal_priority()
        _limit_threads()
    _wait_until_idle(args)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting ICP benchmarks", flush=True)
    return _run_benchmarks(args)


if __name__ == "__main__":
    raise SystemExit(main())
