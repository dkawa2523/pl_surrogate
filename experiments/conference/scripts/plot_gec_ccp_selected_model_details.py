from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUT_DIR = Path("reports/gec_conference_materials/model_detail_assets")


@dataclass(frozen=True)
class ModelRun:
    model_id: str
    display_name: str
    run_root: Path
    color: str

    @property
    def metrics_path(self) -> Path:
        return (
            self.run_root
            / "models"
            / self.model_id
            / "eval_protocol"
            / "interp"
            / "train"
            / "scalars"
            / "metrics.csv"
        )

    @property
    def leaderboard_path(self) -> Path:
        return self.run_root / "leaderboard.csv"


MODEL_RUNS = (
    ModelRun(
        model_id="ffno",
        display_name="FFNO",
        run_root=Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/ffno"),
        color="#0072B2",
    ),
    ModelRun(
        model_id="deeponet_pod",
        display_name="DeepONet (POD)",
        run_root=Path("runs/gec_ccp_pod_branch_tuned_v2/final/seed_412/n78/deeponet_pod"),
        color="#D55E00",
    ),
    ModelRun(
        model_id="global_mlp",
        display_name="Global MLP",
        run_root=Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/global_mlp"),
        color="#009E73",
    ),
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _selected_epoch(run: ModelRun) -> int:
    rows = _read_csv(run.leaderboard_path)
    match = [row for row in rows if str(row.get("model_id", "")) == run.model_id]
    if len(match) != 1:
        raise ValueError(f"expected one leaderboard row for {run.model_id}, found {len(match)}")
    return int(float(match[0]["validation_selected_epoch"]))


def _history(run: ModelRun) -> dict[str, np.ndarray]:
    rows = _read_csv(run.metrics_path)
    required = ("epoch", "train_loss", "val_loss")
    for column in required:
        if not rows or column not in rows[0]:
            raise ValueError(f"{run.metrics_path} lacks {column}")
    values = {
        column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for column in required
    }
    valid = np.isfinite(values["epoch"])
    for column in ("train_loss", "val_loss"):
        valid &= np.isfinite(values[column]) & (values[column] > 0.0)
    if not np.any(valid):
        raise ValueError(f"no positive finite loss records in {run.metrics_path}")
    return {column: array[valid] for column, array in values.items()}


def _draw_history(ax: plt.Axes, run: ModelRun, history: dict[str, np.ndarray]) -> None:
    epoch = history["epoch"]
    train = history["train_loss"]
    val = history["val_loss"]
    selected = _selected_epoch(run)
    selected_idx = int(np.argmin(np.abs(epoch - selected)))
    ax.plot(epoch, train, color=run.color, linewidth=1.35, alpha=0.72, label="Training loss")
    ax.plot(epoch, val, color=run.color, linewidth=1.8, linestyle="--", label="Validation loss")
    ax.axvline(selected, color="#333333", linewidth=1.0, linestyle=":", label=f"Selected epoch ({selected})")
    ax.scatter(
        [epoch[selected_idx]],
        [val[selected_idx]],
        s=28,
        color=run.color,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective value")
    ax.set_title(run.display_name)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.6)
    ax.grid(True, which="minor", axis="y", color="#eeeeee", linewidth=0.4)
    ax.legend(frameon=False, loc="best")


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", pad_inches=0.08)


def _write_derived_data(
    out_dir: Path,
    histories: dict[str, dict[str, np.ndarray]],
) -> None:
    csv_path = out_dir / "learning_curves.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("model_id", "display_name", "epoch", "train_loss", "val_loss", "selected_epoch"),
        )
        writer.writeheader()
        for run in MODEL_RUNS:
            history = histories[run.model_id]
            selected = _selected_epoch(run)
            for epoch, train, val in zip(history["epoch"], history["train_loss"], history["val_loss"]):
                writer.writerow(
                    {
                        "model_id": run.model_id,
                        "display_name": run.display_name,
                        "epoch": f"{epoch:.12g}",
                        "train_loss": f"{train:.12g}",
                        "val_loss": f"{val:.12g}",
                        "selected_epoch": selected,
                    }
                )
    metadata = {
        "purpose": "conference plots for selected GEC-CCP models",
        "dataset_size": 78,
        "seed": 412,
        "selection_split": "interp validation",
        "curve_y_scale": "log",
        "curve_note": (
            "The plotted loss is each architecture's configured training objective; absolute "
            "loss values should not be used as a cross-architecture ranking."
        ),
        "models": [
            {
                "model_id": run.model_id,
                "display_name": run.display_name,
                "run_root": run.run_root.as_posix(),
                "metrics_csv": run.metrics_path.as_posix(),
                "leaderboard_csv": run.leaderboard_path.as_posix(),
                "selected_epoch": _selected_epoch(run),
            }
            for run in MODEL_RUNS
        ],
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    histories = {run.model_id: _history(run) for run in MODEL_RUNS}

    for run in MODEL_RUNS:
        fig, ax = plt.subplots(figsize=(4.8, 3.3), constrained_layout=True)
        _draw_history(ax, run, histories[run.model_id])
        _save(fig, out_dir / f"{run.model_id}_learning_curve")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.25), constrained_layout=True)
    for ax, run in zip(axes, MODEL_RUNS):
        _draw_history(ax, run, histories[run.model_id])
    fig.suptitle("GEC-CCP learning curves (78 conditions, seed 412)", fontsize=11)
    _save(fig, out_dir / "selected_models_learning_curves")
    plt.close(fig)
    _write_derived_data(out_dir, histories)


if __name__ == "__main__":
    main()
