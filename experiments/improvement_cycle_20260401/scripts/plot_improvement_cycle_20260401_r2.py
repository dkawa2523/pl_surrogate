"""Plot improvement-cycle R2 benchmark outcomes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Score:
    exp_id: str
    track: str
    dual: float
    extrap: float
    ne_extrap: float
    ni_extrap: float
    te_extrap: float
    phi_extrap: float

    @property
    def min_var_extrap(self) -> float:
        return min(self.ne_extrap, self.ni_extrap, self.te_extrap, self.phi_extrap)


ORDER = [
    "C3_coord_siren_control",
    "C4_coord_siren_gated_affine",
    "C5_coord_siren_gated_affine_sel",
    "C6_coord_siren_gated_affine_sel_mt",
    "C7_coord_siren_gated_affine_w0_res",
    "D7_deeponet_control",
    "D8_deeponet_tephi_sel",
    "D9_deeponet_tephi_sel_mt",
    "D10_deeponet_fused_learned_mix",
    "U5_unetpp_attn_control",
    "U6_unetpp_attn_density_bias",
    "U7_unetpp_density_bias_port",
]


def _track_of(exp_id: str) -> str:
    if exp_id.startswith("C"):
        return "coord"
    if exp_id.startswith("D"):
        return "deeponet"
    return "unet"


def _read_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"leaderboard has no rows: {path}")
    return dict(rows[0])


def _f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def _load_scores(runs_root: Path) -> list[Score]:
    out: list[Score] = []
    for exp_id in ORDER:
        row = _read_first_row(runs_root / exp_id / "leaderboard.csv")
        out.append(
            Score(
                exp_id=exp_id,
                track=_track_of(exp_id),
                dual=_f(row, "test_r2_plasma_mean_dual"),
                extrap=_f(row, "test_r2_plasma_mean_extrap"),
                ne_extrap=_f(row, "test_r2_ne_plasma_extrap"),
                ni_extrap=_f(row, "test_r2_ni_plasma_extrap"),
                te_extrap=_f(row, "test_r2_Te_plasma_extrap"),
                phi_extrap=_f(row, "test_r2_phi_plasma_extrap"),
            )
        )
    return out


def _save_metrics_csv(scores: list[Score], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "exp_id",
                "track",
                "dual",
                "extrap",
                "ne_extrap",
                "ni_extrap",
                "te_extrap",
                "phi_extrap",
                "min_var_extrap",
            ]
        )
        for s in scores:
            writer.writerow(
                [
                    s.exp_id,
                    s.track,
                    f"{s.dual:.9f}",
                    f"{s.extrap:.9f}",
                    f"{s.ne_extrap:.9f}",
                    f"{s.ni_extrap:.9f}",
                    f"{s.te_extrap:.9f}",
                    f"{s.phi_extrap:.9f}",
                    f"{s.min_var_extrap:.9f}",
                ]
            )


def _plot_dual_extrap(scores: list[Score], out_path: Path) -> None:
    colors = {"coord": "#1b9e77", "deeponet": "#d95f02", "unet": "#7570b3"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in scores:
        ax.scatter(s.dual, s.extrap, color=colors[s.track], s=50)
        ax.text(s.dual + 0.001, s.extrap + 0.001, s.exp_id.split("_", 1)[0], fontsize=8)
    ax.set_title("R2 Cycle: dual vs extrap")
    ax.set_xlabel("test_r2_plasma_mean_dual")
    ax.set_ylabel("test_r2_plasma_mean_extrap")
    ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=v, label=k) for k, v in colors.items()]
    ax.legend(handles=handles, title="track", loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_track_bars(scores: list[Score], out_path: Path) -> None:
    labels = [s.exp_id for s in scores]
    x = np.arange(len(labels), dtype=np.float32)
    dual = np.asarray([s.dual for s in scores], dtype=np.float32)
    extrap = np.asarray([s.extrap for s in scores], dtype=np.float32)
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w / 2, dual, width=w, label="dual", color="#4daf4a")
    ax.bar(x + w / 2, extrap, width=w, label="extrap", color="#377eb8")
    ax.set_title("R2 Cycle: dual/extrap by experiment")
    ax.set_ylabel("R2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_var_extrap_heatmap(scores: list[Score], out_path: Path) -> None:
    mat = np.asarray(
        [[s.ne_extrap, s.ni_extrap, s.te_extrap, s.phi_extrap] for s in scores],
        dtype=np.float32,
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title("R2 Cycle: per-variable extrap")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["ne", "ni", "Te", "phi"])
    ax.set_yticks(np.arange(len(scores)))
    ax.set_yticklabels([s.exp_id for s in scores], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("R2 extrap")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot improvement-cycle R2 benchmark results")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/improvement_cycle_20260401_r2"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/improvement_cycle_20260401_r2/plots"),
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    scores = _load_scores(runs_root)
    _save_metrics_csv(scores, out_dir / "r2_metrics_table.csv")
    _plot_dual_extrap(scores, out_dir / "r2_dual_vs_extrap.png")
    _plot_track_bars(scores, out_dir / "r2_dual_extrap_by_experiment.png")
    _plot_var_extrap_heatmap(scores, out_dir / "r2_extrap_heatmap.png")
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
