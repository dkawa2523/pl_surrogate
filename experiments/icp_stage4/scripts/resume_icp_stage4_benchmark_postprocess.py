"""Finish benchmark evaluation from a completed grid-model checkpoint.

This is intentionally an experiment utility, not a second training path.  It
temporarily replaces the grid trainer call with checkpoint/history loading and
then lets ``BenchmarkRunner`` execute its normal inference, metrics, plots, and
leaderboard code unchanged.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from plasma_surrogate.benchmark.runner import BenchmarkRunner
from plasma_surrogate.models.checkpoint import load_checkpoint
from plasma_surrogate.train.trainer import TrainOutput, Trainer


def read_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    history: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, raw in row.items():
            try:
                parsed[str(key)] = float(raw)
            except (TypeError, ValueError):
                parsed[str(key)] = str(raw or "")
        history.append(parsed)
    if not history:
        raise ValueError(f"checkpoint postprocess requires non-empty history: {path}")
    return history


def run_postprocess(*, config_path: Path, checkpoint_dir: Path, history_path: Path) -> None:
    saved_model = load_checkpoint(checkpoint_dir)
    saved_state = saved_model.state_dict_numpy()
    history = read_history(history_path)
    original = Trainer.run_unet

    def load_completed_run(self: Trainer, model: Any, *_args: Any, **_kwargs: Any) -> TrainOutput:
        del self
        model.load_state_dict_numpy(saved_state)
        return TrainOutput(history=history, model=model)

    Trainer.run_unet = load_completed_run
    try:
        BenchmarkRunner.from_yaml(config_path).run()
    finally:
        Trainer.run_unet = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args()
    run_postprocess(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        history_path=args.history,
    )


if __name__ == "__main__":
    main()
