"""Minimal visualization runner with a small plugin surface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class VizRunner:
    def __init__(self, output_dir: str | Path):
        self.store = ArtifactStore(output_dir)

    def plot_loss_curve(self, history: list[dict[str, float]], rel_path: str = "plots/loss_curve.png") -> Path:
        epochs = [int(h["epoch"]) for h in history]
        train = [h["train_loss"] for h in history]
        val = [h["val_loss"] for h in history]

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(epochs, train, label="train")
        ax.plot(epochs, val, label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(loc="best")
        path = self.store._path(rel_path)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_parity(self, y_true: np.ndarray, y_pred: np.ndarray, rel_path: str = "plots/parity.png") -> Path:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(y_true.reshape(-1), y_pred.reshape(-1), s=8)
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        path = self.store._path(rel_path)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_field_triplet(
        self,
        true_field: np.ndarray,
        pred_field: np.ndarray,
        rel_path: str = "plots/field_triplet.png",
    ) -> Path:
        err = np.abs(pred_field - true_field)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(true_field, origin="lower")
        axes[0].set_title("true")
        axes[1].imshow(pred_field, origin="lower")
        axes[1].set_title("pred")
        axes[2].imshow(err, origin="lower")
        axes[2].set_title("err")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        path = self.store._path(rel_path)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_batch_qoi_scatter(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        rel_path: str = "plots/batch_qoi_scatter.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(x_values, y_values, s=14)
        ax.set_xlabel("x")
        ax.set_ylabel("uniformity")
        path = self.store._path(rel_path)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_metric_bar(
        self,
        labels: list[str],
        values: list[float],
        ylabel: str,
        rel_path: str = "plots/metric_bar.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.6), 3))
        ax.bar(np.arange(len(labels)), values)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        path = self.store._path(rel_path)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
