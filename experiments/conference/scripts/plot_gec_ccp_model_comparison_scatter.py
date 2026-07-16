"""Create conference-ready GEC-CCP model accuracy scatter plots.

The comparison intentionally uses learning seed 412 for every model.  The
horizontal axis measures field-value fidelity and the vertical axis measures
spatial-gradient fidelity; both are physical relative L2 errors over the 13
interpolation test cases and are therefore directly understandable as errors
in percent (lower is better).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.conference.scripts import plot_gec_conference_simple as simple  # noqa: E402


DEFAULT_OUT_DIR = Path("reports/gec_conference_materials/independent_assets")
FORMATS = ("png", "pdf", "svg")
TARGETS = ("ne", "ni", "Te", "phi")
FAMILY_STYLE = {
    "Neural network": {"color": simple.COLORS["blue"], "marker": "o"},
    "Neural operator": {"color": simple.COLORS["red"], "marker": "D"},
}


@dataclass(frozen=True)
class ModelSource:
    model_id: str
    label: str
    family: str
    run_root: Path


MODEL_SOURCES = (
    ModelSource("global_mlp", "Global MLP", "Neural network", Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/global_mlp")),
    ModelSource("global_resmlp", "ResMLP", "Neural network", Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/global_resmlp")),
    ModelSource("global_densemlp", "DenseMLP", "Neural network", Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/global_densemlp")),
    ModelSource("unet", "U-Net", "Neural network", Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/unet")),
    ModelSource("unetpp", "U-Net++", "Neural network", Path("runs/gec_ccp_nn_operator_comparison_v2_single_run_additions/seed_412/n78/unetpp")),
    ModelSource("fno", "FNO", "Neural operator", Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/fno")),
    ModelSource("ffno", "FFNO", "Neural operator", Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/ffno")),
    ModelSource("u_no", "U-NO", "Neural operator", Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/u_no")),
    ModelSource("cno", "CNO", "Neural operator", Path("runs/gec_ccp_nn_operator_comparison_v2_single_run_additions/seed_412/n78/cno")),
    ModelSource("deeponet_pod", "POD-DeepONet", "Neural operator", Path("runs/gec_ccp_pod_branch_tuned_v2/final/seed_412/n78/deeponet_pod")),
    ModelSource("deeponet_plasma", "DeepONet", "Neural operator", Path("runs/gec_ccp_nn_operator_comparison_v2_single_run_additions/seed_412/n78/deeponet_plasma")),
)


LABEL_OFFSETS = {
    "Global MLP": (-82, 8),
    "ResMLP": (8, 8),
    "DenseMLP": (8, -17),
    "U-Net": (-50, 9),
    "U-Net++": (8, 7),
    "FNO": (-31, 9),
    "FFNO": (-24, 10),
    "U-NO": (8, -17),
    "CNO": (8, -16),
    "POD-DeepONet": (8, -17),
    "DeepONet": (-72, -16),
}

R2_DISPLAY_LABELS = {
    "global_mlp": "Global MLP",
    "global_resmlp": "ResMLP",
    "global_densemlp": "DenseMLP",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "fno": "FNO",
    "ffno": "FFNO",
    "u_no": "U-NO",
    "cno": "CNO",
    "deeponet_pod": "DeepONet",
}

R2_LABEL_OFFSETS = {
    "ne_te": {
        "global_resmlp": (8, 11),
        "global_densemlp": (8, -17),
        "unet": (8, -17),
        "unetpp": (8, -15),
        "fno": (-30, 11),
        "ffno": (-52, 10),
        "u_no": (-43, 9),
        "cno": (8, 9),
        "deeponet_pod": (-62, -18),
    },
    "ni_phi": {
        "global_resmlp": (-55, 9),
        "global_densemlp": (8, -17),
        "unet": (-49, 10),
        "unetpp": (8, 10),
        "fno": (-42, -27),
        "ffno": (-45, 9),
        "u_no": (-43, 9),
        "cno": (8, -17),
        "deeponet_pod": (-75, 2),
    },
}


def _read_single_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one diagnostics row in {path}, found {len(rows)}")
    return rows[0]


def _finite_metric(row: dict[str, str], key: str, *, source: Path) -> float:
    if key not in row:
        raise KeyError(f"{source} does not contain {key}")
    value = float(row[key])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid {key}={value!r} in {source}")
    return value


def _finite_r2(row: dict[str, str], key: str, *, source: Path) -> float:
    if key not in row:
        raise KeyError(f"{source} does not contain {key}")
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"invalid {key}={value!r} in {source}")
    return value


def _r2_display_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    displayed: list[dict[str, Any]] = []
    for record in records:
        model_id = str(record["model_id"])
        if model_id not in R2_DISPLAY_LABELS:
            continue
        displayed.append({**record, "model_label": R2_DISPLAY_LABELS[model_id]})
    if {str(record["model_id"]) for record in displayed} != set(R2_DISPLAY_LABELS):
        raise ValueError("R2 display model set does not match the configured labels")
    return displayed


def _collect() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in MODEL_SOURCES:
        diagnostics_path = source.run_root / "diagnostics" / "diagnostics.csv"
        leaderboard_path = source.run_root / "leaderboard.csv"
        row = _read_single_csv(diagnostics_path)
        leaderboard = _read_single_csv(leaderboard_path)
        record: dict[str, Any] = {
            "model_id": source.model_id,
            "model_label": source.label,
            "family": source.family,
            "learning_seed": 412,
            "test_case_count": 13,
            "run_root": source.run_root.as_posix(),
        }
        value_errors: list[float] = []
        gradient_errors: list[float] = []
        for target in TARGETS:
            value = 100.0 * _finite_metric(
                row,
                f"score_physical_rel_l2_median_{target}",
                source=diagnostics_path,
            )
            gradient = 100.0 * _finite_metric(
                row,
                f"score_physical_gradient_rel_l2_median_{target}",
                source=diagnostics_path,
            )
            record[f"value_error_{target}_pct"] = value
            record[f"gradient_error_{target}_pct"] = gradient
            record[f"test_r2_{target}_plasma"] = _finite_r2(
                leaderboard,
                f"test_r2_{target}_plasma",
                source=leaderboard_path,
            )
            value_errors.append(value)
            gradient_errors.append(gradient)
        record["mean_value_error_pct"] = float(np.mean(value_errors))
        record["mean_gradient_error_pct"] = float(np.mean(gradient_errors))
        records.append(record)
    return records


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=str(style["marker"]),
            linestyle="none",
            markerfacecolor=str(style["color"]),
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=10,
            label=family,
        )
        for family, style in FAMILY_STYLE.items()
    ]


def _scatter_point(ax: plt.Axes, record: dict[str, Any], *, x_key: str, y_key: str, size: float = 90.0) -> None:
    style = FAMILY_STYLE[str(record["family"])]
    ax.scatter(
        [float(record[x_key])],
        [float(record[y_key])],
        s=size,
        marker=str(style["marker"]),
        color=str(style["color"]),
        edgecolor="white",
        linewidth=1.1,
        zorder=4,
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#E4E7EC", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _plot_overall(records: list[dict[str, Any]]) -> plt.Figure:
    local_style = dict(simple.STYLE)
    local_style.update({"axes.titlesize": 19.0, "axes.labelsize": 14.5, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5})
    with plt.rc_context(local_style):
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.18)
        for record in records:
            _scatter_point(
                ax,
                record,
                x_key="mean_value_error_pct",
                y_key="mean_gradient_error_pct",
                size=115.0,
            )
            dx, dy = LABEL_OFFSETS[str(record["model_label"])]
            color = str(FAMILY_STYLE[str(record["family"])]["color"])
            ax.annotate(
                str(record["model_label"]),
                (float(record["mean_value_error_pct"]), float(record["mean_gradient_error_pct"])),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=10.5,
                fontweight="semibold",
                color=color,
                arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.45, "linewidth": 0.8},
                zorder=5,
            )
        ax.set_xlim(0.0, 21.5)
        ax.set_ylim(0.0, 69.0)
        ax.set_xlabel("Field-value relative error (%)")
        ax.set_ylabel("Spatial-gradient relative error (%)")
        ax.set_title("GEC-CCP model comparison", pad=13)
        ax.text(
            0.98,
            0.04,
            "Lower-left = better\nvalue and shape fidelity",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color=simple.COLORS["muted"],
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D5DD", "alpha": 0.95},
        )
        ax.annotate(
            "",
            xy=(0.055, 0.08),
            xytext=(0.20, 0.22),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": simple.COLORS["green"], "linewidth": 2.0},
        )
        ax.legend(handles=_legend_handles(), loc="center right", frameon=False, ncol=1, fontsize=10.5)
        fig.text(
            0.14,
            0.045,
            "Seed 412; 13 held-out cases. Axes show the mean of four per-field median physical relative L2 errors.",
            fontsize=8.8,
            color=simple.COLORS["muted"],
        )
        _style_axis(ax)
    return fig


def _plot_zoom(records: list[dict[str, Any]]) -> plt.Figure:
    visible = [
        record
        for record in records
        if float(record["mean_value_error_pct"]) <= 7.0
        and float(record["mean_gradient_error_pct"]) <= 35.0
    ]
    local_style = dict(simple.STYLE)
    local_style.update({"axes.titlesize": 19.0, "axes.labelsize": 14.5, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5})
    with plt.rc_context(local_style):
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.14)
        for record in visible:
            _scatter_point(ax, record, x_key="mean_value_error_pct", y_key="mean_gradient_error_pct", size=125.0)
            dx, dy = LABEL_OFFSETS[str(record["model_label"])]
            color = str(FAMILY_STYLE[str(record["family"])]["color"])
            ax.annotate(
                str(record["model_label"]),
                (float(record["mean_value_error_pct"]), float(record["mean_gradient_error_pct"])),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=11.0,
                fontweight="semibold",
                color=color,
                arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.45, "linewidth": 0.8},
            )
        ax.set_xlim(0.0, 7.0)
        ax.set_ylim(0.0, 35.0)
        ax.set_xlabel("Field-value relative error (%)")
        ax.set_ylabel("Spatial-gradient relative error (%)")
        ax.set_title("High-accuracy region (zoom)", pad=13)
        ax.legend(handles=_legend_handles(), loc="upper left", frameon=False, ncol=2, fontsize=11)
        ax.text(
            0.98,
            0.04,
            "Global MLP and DeepONet are outside this zoomed range",
            transform=ax.transAxes,
            ha="right",
            fontsize=9.5,
            color=simple.COLORS["muted"],
        )
        _style_axis(ax)
    return fig


def _plot_by_field(records: list[dict[str, Any]]) -> plt.Figure:
    title = {"ne": r"Electron density $n_e$", "ni": r"Ion density $n_i$", "Te": r"Electron temperature $T_e$", "phi": r"Potential $\phi$"}
    with plt.rc_context(simple.STYLE):
        local_style = dict(simple.STYLE)
        local_style.update({"axes.titlesize": 13.0, "axes.labelsize": 10.8, "xtick.labelsize": 9.2, "ytick.labelsize": 9.2})
        with plt.rc_context(local_style):
            fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
            fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.09, hspace=0.40, wspace=0.25)
            for ax, target in zip(axes.flat, TARGETS):
                x_key = f"value_error_{target}_pct"
                y_key = f"gradient_error_{target}_pct"
                for record in records:
                    _scatter_point(ax, record, x_key=x_key, y_key=y_key, size=62.0)
                    dx, dy = LABEL_OFFSETS[str(record["model_label"])]
                    ax.annotate(
                        str(record["model_label"]),
                        (float(record[x_key]), float(record[y_key])),
                        xytext=(0.55 * dx, 0.55 * dy),
                        textcoords="offset points",
                        fontsize=6.2,
                        color=str(FAMILY_STYLE[str(record["family"])]["color"]),
                    )
                ax.set_title(title[target])
                ax.set_xlabel("Field-value error (%)")
                ax.set_ylabel("Spatial-gradient error (%)")
                ax.set_xlim(left=0.0)
                ax.set_ylim(bottom=0.0)
                _style_axis(ax)
            fig.suptitle("GEC-CCP model accuracy by plasma quantity", fontsize=17.0, y=0.975)
            fig.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.915), frameon=False, ncol=2, fontsize=9.5)
    return fig


def _plot_r2_pair(
    records: list[dict[str, Any]],
    *,
    pair_id: str,
    x_target: str,
    y_target: str,
    title: str,
    x_label: str,
    y_label: str,
    main_xlim: tuple[float, float],
    main_ylim: tuple[float, float],
    inset_xlim: tuple[float, float],
    inset_ylim: tuple[float, float],
) -> plt.Figure:
    records = _r2_display_records(records)
    x_key = f"test_r2_{x_target}_plasma"
    y_key = f"test_r2_{y_target}_plasma"
    detailed = [
        record
        for record in records
        if main_xlim[0] <= float(record[x_key]) <= main_xlim[1]
        and main_ylim[0] <= float(record[y_key]) <= main_ylim[1]
    ]
    outliers = [record for record in records if record not in detailed]
    local_style = dict(simple.STYLE)
    local_style.update({"axes.titlesize": 18.0, "axes.labelsize": 14.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5})
    with plt.rc_context(local_style):
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.87, bottom=0.14)
        for record in detailed:
            _scatter_point(ax, record, x_key=x_key, y_key=y_key, size=115.0)
            model_id = str(record["model_id"])
            dx, dy = R2_LABEL_OFFSETS[pair_id][model_id]
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
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, pad=12)
        _style_axis(ax)
        ax.legend(handles=_legend_handles(), loc="upper left", frameon=False, fontsize=10.0)

        inset = ax.inset_axes([0.58, 0.08, 0.37, 0.36])
        for record in records:
            _scatter_point(inset, record, x_key=x_key, y_key=y_key, size=34.0)
        for index, record in enumerate(outliers):
            color = str(FAMILY_STYLE[str(record["family"])]["color"])
            inset.annotate(
                str(record["model_label"]),
                (float(record[x_key]), float(record[y_key])),
                xytext=(5, 7 if index % 2 == 0 else -12),
                textcoords="offset points",
                ha="left",
                fontsize=6.7,
                color=color,
            )
        inset.add_patch(
            Rectangle(
                (main_xlim[0], main_ylim[0]),
                main_xlim[1] - main_xlim[0],
                main_ylim[1] - main_ylim[0],
                fill=False,
                edgecolor=simple.COLORS["green"],
                linewidth=1.1,
                linestyle="--",
            )
        )
        inset.set_xlim(*inset_xlim)
        inset.set_ylim(*inset_ylim)
        inset.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        inset.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        inset.tick_params(labelsize=6.8, length=2.5)
        inset.set_title("Full range", fontsize=8.0, pad=2)
        inset.grid(True, color="#E4E7EC", linewidth=0.5)
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)

    return fig


def _save(fig: plt.Figure, *, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in dict.fromkeys(formats):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _write_data(out_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = out_dir / "ccp_model_accuracy_scatter_data.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def _write_r2_data(out_dir: Path, records: list[dict[str, Any]]) -> Path:
    records = _r2_display_records(records)
    fields = [
        "model_id",
        "model_label",
        "family",
        "learning_seed",
        "test_case_count",
        "test_r2_ne_plasma",
        "test_r2_ni_plasma",
        "test_r2_Te_plasma",
        "test_r2_phi_plasma",
        "run_root",
    ]
    path = out_dir / "ccp_model_r2_scatter_data.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: record[field] for field in fields} for record in records])
    return path


def _write_metadata(out_dir: Path, records: list[dict[str, Any]], outputs: dict[str, list[Path]]) -> Path:
    path = out_dir / "ccp_model_accuracy_scatter_metadata.json"
    payload = {
        "claim": "Model accuracy must be judged by both field values and spatial gradients; lower-left is better.",
        "dataset": "GEC-CCP n78",
        "split": {"train": 54, "validation": 11, "test": 13, "split_seed": 7},
        "learning_seed": 412,
        "x_axis": "Mean across ne, ni, Te, phi of the per-field median physical relative L2 value error over 13 test cases (%)",
        "y_axis": "Mean across ne, ni, Te, phi of the per-field median physical relative L2 spatial-gradient error over 13 test cases (%)",
        "lower_is_better": True,
        "family_styles": FAMILY_STYLE,
        "outputs": {name: [item.as_posix() for item in paths] for name, paths in outputs.items()},
        "r2_axes": {
            "ne_vs_Te": {"x": "test_r2_Te_plasma", "y": "test_r2_ne_plasma"},
            "ni_vs_phi": {"x": "test_r2_phi_plasma", "y": "test_r2_ni_plasma"},
        },
        "r2_figure_mode": {
            "coordinate_type": "measured_test_r2",
            "numeric_ticks": True,
            "excluded_source_model": "deeponet_plasma",
            "renamed_source_model": {"deeponet_pod": "DeepONet"},
            "data_csv": "ccp_model_r2_scatter_data.csv",
        },
        "models": records,
        "limitations": [
            "One learning seed is shown for model-to-model comparability.",
            "Model-specific epoch budgets differ because untuned existing reference recipes were requested.",
            "This is interpolation performance; it is not independent geometry generalization.",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    records = _collect()
    outputs = {
        "overall": _save(
            _plot_overall(records),
            out_dir=args.out_dir,
            stem="ccp_model_accuracy_scatter",
            formats=args.formats,
            dpi=int(args.dpi),
        ),
        "zoom": _save(
            _plot_zoom(records),
            out_dir=args.out_dir,
            stem="ccp_model_accuracy_scatter_zoom",
            formats=args.formats,
            dpi=int(args.dpi),
        ),
        "by_field": _save(
            _plot_by_field(records),
            out_dir=args.out_dir,
            stem="ccp_model_accuracy_by_field",
            formats=args.formats,
            dpi=int(args.dpi),
        ),
        "r2_ne_vs_Te": _save(
            _plot_r2_pair(
                records,
                pair_id="ne_te",
                x_target="Te",
                y_target="ne",
                title=r"Electron density vs. electron temperature: test $R^2$",
                x_label=r"Electron temperature $R^2$",
                y_label=r"Electron density $R^2$",
                main_xlim=(0.9938, 1.0003),
                main_ylim=(0.9958, 1.0002),
                inset_xlim=(0.89, 1.002),
                inset_ylim=(0.94, 1.002),
            ),
            out_dir=args.out_dir,
            stem="ccp_model_r2_ne_vs_te",
            formats=args.formats,
            dpi=int(args.dpi),
        ),
        "r2_ni_vs_phi": _save(
            _plot_r2_pair(
                records,
                pair_id="ni_phi",
                x_target="phi",
                y_target="ni",
                title=r"Ion density vs. potential: test $R^2$",
                x_label=r"Potential $R^2$",
                y_label=r"Ion density $R^2$",
                main_xlim=(0.9970, 1.00015),
                main_ylim=(0.9960, 1.00015),
                inset_xlim=(0.95, 1.002),
                inset_ylim=(0.94, 1.002),
            ),
            out_dir=args.out_dir,
            stem="ccp_model_r2_ni_vs_phi",
            formats=args.formats,
            dpi=int(args.dpi),
        ),
    }
    data_path = _write_data(args.out_dir, records)
    r2_data_path = _write_r2_data(args.out_dir, records)
    metadata_path = _write_metadata(args.out_dir, records, outputs)
    print(data_path)
    print(r2_data_path)
    print(metadata_path)
    for paths in outputs.values():
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
