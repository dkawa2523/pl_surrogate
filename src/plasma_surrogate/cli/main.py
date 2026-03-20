"""CLI skeleton for category pipeline tasks and benchmark/viz."""

from __future__ import annotations

import argparse
import json

from plasma_surrogate.pipeline.task_runner import TaskRunner


def _run_task(args: argparse.Namespace) -> int:
    result = TaskRunner().run(task_name=str(args.task_name), config_path=args.config)
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plasma_surrogate")
    sub = p.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess")
    preprocess.add_argument("--config", required=True)
    preprocess.set_defaults(func=_run_task, task_name="preprocess")

    cleanse = sub.add_parser("cleanse")
    cleanse.add_argument("--config", required=True)
    cleanse.set_defaults(func=_run_task, task_name="cleanse")

    feature = sub.add_parser("feature")
    feature.add_argument("--config", required=True)
    feature.set_defaults(func=_run_task, task_name="feature")

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.set_defaults(func=_run_task, task_name="train")

    infer = sub.add_parser("infer")
    infer.add_argument("--config", required=True)
    infer.set_defaults(func=_run_task, task_name="infer")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.set_defaults(func=_run_task, task_name="evaluate")

    pipeline = sub.add_parser("pipeline")
    pipeline.add_argument("--config", required=True)
    pipeline.set_defaults(func=_run_task, task_name="pipeline.run")

    viz = sub.add_parser("viz")
    viz.add_argument("--config", required=True)
    viz.set_defaults(func=_run_task, task_name="viz")

    benchmark = sub.add_parser("benchmark")
    bench_sub = benchmark.add_subparsers(dest="benchmark_cmd", required=True)
    run = bench_sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.set_defaults(func=_run_task, task_name="benchmark.run")
    sweep = bench_sub.add_parser("sweep")
    sweep.add_argument("--config", required=True)
    sweep.set_defaults(func=_run_task, task_name="benchmark.sweep")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
