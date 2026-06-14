"""Minimal visualization runner with a small plugin surface."""

from __future__ import annotations

import os
from pathlib import Path
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

    def plot_loss_curve(
        self,
        history: list[dict[str, float]],
        rel_path: str = "plots/loss_curve.png",
        *,
        yscale: str = "linear",
    ) -> Path:
        epochs = [int(h["epoch"]) for h in history]
        train = [h["train_loss"] for h in history]
        val = [h["val_loss"] for h in history]

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(epochs, train, label="train")
        ax.plot(epochs, val, label="val")
        yscale_eff = str(yscale or "linear").strip().lower()
        if yscale_eff not in {"linear", "log"}:
            raise ValueError("plot_loss_curve yscale must be 'linear' or 'log'")
        if yscale_eff == "log":
            ax.set_yscale("log")
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
        *,
        mask: np.ndarray | None = None,
        colorbar: bool = True,
    ) -> Path:
        true_arr = np.asarray(true_field, dtype=np.float32)
        pred_arr = np.asarray(pred_field, dtype=np.float32)
        if true_arr.shape != pred_arr.shape:
            raise ValueError(f"plot_field_triplet shape mismatch: {true_arr.shape} vs {pred_arr.shape}")
        if true_arr.ndim != 2:
            raise ValueError(f"plot_field_triplet expects [H,W] fields, got {true_arr.shape}")
        mask_arr: np.ndarray | None = None
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.ndim == 3 and int(mask_arr.shape[0]) == 1:
                mask_arr = mask_arr[0]
            if mask_arr.shape != true_arr.shape:
                raise ValueError(f"plot_field_triplet mask shape mismatch: {mask_arr.shape} vs {true_arr.shape}")
        finite = np.isfinite(true_arr) & np.isfinite(pred_arr)
        if mask_arr is not None:
            finite &= mask_arr
        values = np.concatenate([true_arr[finite].reshape(-1), pred_arr[finite].reshape(-1)])
        vmin = float(np.min(values)) if values.size else None
        vmax = float(np.max(values)) if values.size else None
        if vmin is not None and vmax is not None and np.isclose(vmin, vmax):
            vmax = float(vmin + 1.0)
        err = np.abs(pred_arr - true_arr)
        if mask_arr is not None:
            true_plot = np.ma.array(true_arr, mask=~mask_arr)
            pred_plot = np.ma.array(pred_arr, mask=~mask_arr)
            err_plot = np.ma.array(err, mask=~mask_arr)
        else:
            true_plot = true_arr
            pred_plot = pred_arr
            err_plot = err
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        im0 = axes[0].imshow(true_plot, origin="lower", vmin=vmin, vmax=vmax)
        axes[0].set_title("true")
        im1 = axes[1].imshow(pred_plot, origin="lower", vmin=vmin, vmax=vmax)
        axes[1].set_title("pred")
        im2 = axes[2].imshow(err_plot, origin="lower")
        axes[2].set_title("err")
        for ax, im in zip(axes, [im0, im1, im2]):
            ax.set_xticks([])
            ax.set_yticks([])
            if colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
