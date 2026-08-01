#!/usr/bin/env python3
"""Plot field-wise case-macro relative RMSE for GEC-CCP n54 models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


DEFAULT_SOURCE = Path(
    "runs/gec_ccp_training_size_ablation_v1/summary_all_models/"
    "spatial/n54/spatial_plot_summary.csv"
)
DEFAULT_OUT_DIR = Path("reports/gec_conference_materials/training_size_ablation")
MODEL_ORDER = (
    "global_resmlp",
    "global_densemlp",
    "unet",
    "unetpp",
    "fno",
    "ffno",
    "cno",
    "deeponet_pod",
)
MODEL_LABELS = {
    "global_resmlp": "ResMLP",
    "global_densemlp": "DenseMLP",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "fno": "FNO",
    "ffno": "FFNO",
    "cno": "CNO",
    "deeponet_pod": "DeepONet",
}
FIELDS = ("ne", "ni", "Te", "phi")
FAMILY_STYLE = {
    "neural_network": {"label": "Neural Network", "color": "#2563A6", "marker": "o"},
    "neural_operator": {"label": "Neural Operator", "color": "#D65332", "marker": "D"},
}
PAIR_CONFIGS = {
    "ne_vs_te": {
        "stem": "ccp_n54_relative_rmse_ne_vs_te",
        "x_field": "ne",
        "y_field": "Te",
        "title": "Electron density–temperature relative RMSE",
        "x_label": r"Electron density $n_e$ relative RMSE (%)",
        "y_label": r"Electron temperature $T_e$ relative RMSE (%)",
        "main_xlim": (1.35, 7.75),
        "main_ylim": (2.0, 7.8),
        "offsets": {
            "global_resmlp": (-58, 9),
            "global_densemlp": (-75, -18),
            "unet": (8, -17),
            "unetpp": (8, 9),
            "fno": (-35, -19),
            "ffno": (-25, -20),
            "cno": (8, 9),
            "deeponet_pod": (8, 8),
        },
    },
    "phi_vs_ni": {
        "stem": "ccp_n54_relative_rmse_phi_vs_ni",
        "x_field": "phi",
        "y_field": "ni",
        "title": "Potential–ion density relative RMSE",
        "x_label": r"Potential $\phi$ relative RMSE (%)",
        "y_label": r"Ion density $n_i$ relative RMSE (%)",
        "main_xlim": (0.85, 4.65),
        "main_ylim": (1.25, 7.75),
        "offsets": {
            "global_resmlp": (-65, 8),
            "global_densemlp": (-75, -18),
            "unet": (8, 9),
            "unetpp": (-65, 9),
            "fno": (8, 9),
            "ffno": (-28, -20),
            "cno": (8, -18),
            "deeponet_pod": (-73, 8),
        },
    },
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 19,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no evaluation rows in {path}")
    return rows


def _aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    reference_cases: set[str] | None = None
    for model in MODEL_ORDER:
        selected = [row for row in rows if row["model"] == model and row["split"] == "interp"]
        if len(selected) != 13:
            raise ValueError(f"{model} must have 13 fixed interpolation test cases, found {len(selected)}")
        cases = {row["case_id"] for row in selected}
        if reference_cases is None:
            reference_cases = cases
        elif cases != reference_cases:
            raise ValueError(f"fixed test cases differ for {model}")
        families = {row["family"] for row in selected}
        if len(families) != 1:
            raise ValueError(f"family is not unique for {model}: {families}")
        family = next(iter(families))
        if family not in FAMILY_STYLE:
            raise ValueError(f"unknown family {family!r} for {model}")
        record: dict[str, Any] = {
            "model_id": model,
            "model_label": MODEL_LABELS[model],
            "family": family,
            "test_case_count": len(selected),
        }
        for field in FIELDS:
            values = np.asarray([float(row[f"rel_rmse_{field}"]) for row in selected], dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"invalid rel_rmse_{field} values for {model}")
            record[f"case_macro_relative_rmse_{field}_pct"] = float(100.0 * np.mean(values))
        records.append(record)
    return records


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=str(style["marker"]),
            linestyle="none",
            markersize=9,
            markerfacecolor=str(style["color"]),
            markeredgecolor="white",
            label=str(style["label"]),
        )
        for style in FAMILY_STYLE.values()
    ]


def _scatter(ax: plt.Axes, record: dict[str, Any], x_key: str, y_key: str, *, size: float) -> None:
    style = FAMILY_STYLE[str(record["family"])]
    ax.scatter(
        float(record[x_key]),
        float(record[y_key]),
        s=size,
        marker=str(style["marker"]),
        facecolor=str(style["color"]),
        edgecolor="white",
        linewidth=1.1,
        zorder=4,
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#E4E7EC", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4.5)
    ax.set_axisbelow(True)


def _plot_pair(records: list[dict[str, Any]], config: dict[str, Any], out: Path) -> None:
    x_key = f"case_macro_relative_rmse_{config['x_field']}_pct"
    y_key = f"case_macro_relative_rmse_{config['y_field']}_pct"
    main_xlim = tuple(config["main_xlim"])
    main_ylim = tuple(config["main_ylim"])
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.15)
    for record in records:
        _scatter(ax, record, x_key, y_key, size=125.0)
        model = str(record["model_id"])
        dx, dy = config["offsets"][model]
        color = str(FAMILY_STYLE[str(record["family"])]["color"])
        ax.annotate(
            str(record["model_label"]),
            (float(record[x_key]), float(record[y_key])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10.0,
            fontweight="semibold",
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.45, "linewidth": 0.8},
            zorder=5,
        )

    ax.set_xlim(*main_xlim)
    ax.set_ylim(*main_ylim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel(str(config["x_label"]))
    ax.set_ylabel(str(config["y_label"]))
    ax.set_title(str(config["title"]), pad=12, fontweight="semibold")
    _style_axis(ax)
    ax.legend(handles=_legend_handles(), loc="upper left", frameon=False, fontsize=10.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out.with_suffix(suffix))
    plt.close(fig)


def _write_data(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "model_id",
        "model_label",
        "family",
        "test_case_count",
        *(f"case_macro_relative_rmse_{field}_pct" for field in FIELDS),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields} for record in records)


def main() -> None:
    args = _parse_args()
    records = _aggregate(_read_rows(args.source))
    outputs: dict[str, list[str]] = {}
    for pair_id, config in PAIR_CONFIGS.items():
        out = args.out_dir / str(config["stem"])
        _plot_pair(records, config, out)
        outputs[pair_id] = [out.with_suffix(suffix).as_posix() for suffix in (".png", ".pdf", ".svg")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.out_dir / "ccp_n54_relative_rmse_pair_scatter_data.csv"
    _write_data(records, data_path)
    metadata = {
        "dataset": "GEC-CCP n78",
        "split": {"train": 54, "validation": 11, "test": 13},
        "evaluation_split": "fixed interpolation test cases",
        "metric": "100 * arithmetic mean across test cases of per-case relative RMSE",
        "fields": list(FIELDS),
        "models": [record["model_id"] for record in records],
        "excluded_models": ["global_mlp", "u_no", "deeponet_plasma"],
        "renamed_model": {"deeponet_pod": "DeepONet"},
        "axes": {
            "ne_vs_te": {"x": "electron-density relative RMSE", "y": "electron-temperature relative RMSE"},
            "phi_vs_ni": {"x": "potential relative RMSE", "y": "ion-density relative RMSE"},
        },
        "scale": "linear",
        "figure_height_to_width_ratio": 0.75,
        "source": args.source.as_posix(),
        "data": data_path.as_posix(),
        "outputs": outputs,
        "generation_command": (
            ".\\.venv-torch\\Scripts\\python.exe "
            "experiments/conference/scripts/plot_gec_ccp_n54_fieldwise_rmse.py"
        ),
    }
    (args.out_dir / "ccp_n54_relative_rmse_pair_scatter_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for paths in outputs.values():
        print(paths[0])
    print(data_path)


if __name__ == "__main__":
    main()
