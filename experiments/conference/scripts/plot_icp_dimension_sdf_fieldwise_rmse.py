#!/usr/bin/env python3
"""Create GEC-CCP-style field-wise RMSE plots for ICP Dimension and SDF models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


DEFAULT_OUT_DIR = Path("reports/icp_conference_materials/model_family_rmse")
FIELDS = ("ne", "ni", "Te", "phi")
FIELD_LABELS = {
    "ne": r"Electron density $n_e$ relative RMSE (%)",
    "ni": r"Ion density $n_i$ relative RMSE (%)",
    "Te": r"Electron temperature $T_e$ relative RMSE (%)",
    "phi": r"Potential $\phi$ relative RMSE (%)",
}
FAMILY_STYLE = {
    "neural_network": {"label": "Neural Network", "color": "#2563A6", "marker": "o"},
    "neural_operator": {"label": "Neural Operator", "color": "#D65332", "marker": "D"},
}


@dataclass(frozen=True)
class ModelSource:
    model_id: str
    label: str
    family: str
    representation: str
    run_root: Path

    @property
    def metrics_path(self) -> Path:
        return (
            self.run_root
            / "models"
            / self.model_id
            / "eval_protocol"
            / "structure_holdout"
            / "eval"
            / "spatial_distribution_by_case.csv"
        )


def _operator_source(representation: str, model_id: str, label: str) -> ModelSource:
    return ModelSource(
        model_id=model_id,
        label=label,
        family="neural_operator",
        representation=representation,
        run_root=(
            Path("runs/icp_axisymmetric_operator_case_v1/final")
            / representation
            / model_id
            / "seed_1237"
        ),
    )


SOURCES = (
    ModelSource(
        "unet",
        "U-Net",
        "neural_network",
        "dimension",
        Path("runs/icp_stage4_axisymmetric_dimension_v4"),
    ),
    ModelSource(
        "unetpp",
        "U-Net++",
        "neural_network",
        "dimension",
        Path("runs/icp_unetpp_bohm_case_v1/full/dimension/unetpp/ch32_lr3e4"),
    ),
    ModelSource(
        "unetpp_attn",
        "Attention U-Net++",
        "neural_network",
        "dimension",
        Path("runs/icp_unetpp_bohm_case_v1/full/dimension/unetpp_attn/ch32_red2_lr3e4"),
    ),
    _operator_source("dimension", "ffno", "FFNO"),
    _operator_source("dimension", "u_no", "UNO"),
    _operator_source("dimension", "cno", "CNO"),
    ModelSource(
        "unet",
        "U-Net",
        "neural_network",
        "union_sdf",
        Path("runs/icp_stage4_axisymmetric_neumann_v4"),
    ),
    ModelSource(
        "unetpp",
        "U-Net++",
        "neural_network",
        "union_sdf",
        Path("runs/icp_unetpp_bohm_case_v1/full/sdf/unetpp/ch32_lr3e4"),
    ),
    ModelSource(
        "unetpp_attn",
        "Attention U-Net++",
        "neural_network",
        "union_sdf",
        Path("runs/icp_unetpp_learned_stem_v1/full/sdf/unetpp_attn/ch32_red2_lr3e4"),
    ),
    _operator_source("union_sdf", "ffno", "FFNO"),
    _operator_source("union_sdf", "u_no", "UNO"),
    _operator_source("union_sdf", "cno", "CNO"),
)

PAIR_CONFIGS = (
    ("ne", "Te", r"$n_e$ vs $T_e$ relative RMSE"),
    ("phi", "ni", r"$\phi$ vs $n_i$ relative RMSE"),
)
LABEL_OFFSETS = {
    "Dimension": {
        "U-Net": (8, 9),
        "U-Net++": (8, -18),
        "Attention U-Net++": (-104, 10),
        "FFNO": (-38, -18),
        "UNO": (8, -18),
        "CNO": (-34, 10),
    },
    "Union SDF": {
        "U-Net": (-60, -20),
        "U-Net++": (8, -18),
        "Attention U-Net++": (-104, 10),
        "FFNO": (-38, -18),
        "UNO": (8, -18),
        "CNO": (-34, 10),
    },
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 18,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no evaluation rows in {path}")
    return rows


def _collect() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    reference_cases: dict[str, set[str]] = {}
    for source in SOURCES:
        rows = _read_rows(source.metrics_path)
        by_field: dict[str, list[dict[str, str]]] = {
            field: [
                row
                for row in rows
                if row.get("var") == field and row.get("target_region") == "plasma_only"
            ]
            for field in FIELDS
        }
        case_sets = [{row["case_id"] for row in by_field[field]} for field in FIELDS]
        if not case_sets[0] or any(cases != case_sets[0] for cases in case_sets[1:]):
            raise ValueError(f"field-wise test cases differ for {source.label}: {source.metrics_path}")
        expected = reference_cases.setdefault(source.representation, case_sets[0])
        if case_sets[0] != expected:
            raise ValueError(
                f"common test cases differ for {source.representation}/{source.label}: "
                f"{len(case_sets[0])} versus {len(expected)}"
            )
        record: dict[str, Any] = {
            "representation": source.representation,
            "model_id": source.model_id,
            "model_label": source.label,
            "family": source.family,
            "test_case_count": len(expected),
            "run_root": source.run_root.as_posix(),
            "source_metrics": source.metrics_path.as_posix(),
        }
        for field in FIELDS:
            values = np.asarray(
                [float(row["physical_rel_l2"]) for row in by_field[field]],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"invalid {field} physical_rel_l2 for {source.label}")
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


def _axis_limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = high - low
    pad = max(0.12 * span, 0.35)
    return max(0.0, low - pad), high + pad


def _plot_pair(
    records: list[dict[str, Any]],
    *,
    representation_label: str,
    x_field: str,
    y_field: str,
    title: str,
    out_stem: Path,
) -> None:
    x_key = f"case_macro_relative_rmse_{x_field}_pct"
    y_key = f"case_macro_relative_rmse_{y_field}_pct"
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.84, bottom=0.15)
    for record in records:
        style = FAMILY_STYLE[str(record["family"])]
        x, y = float(record[x_key]), float(record[y_key])
        ax.scatter(
            x,
            y,
            s=125,
            marker=str(style["marker"]),
            facecolor=str(style["color"]),
            edgecolor="white",
            linewidth=1.1,
            zorder=4,
        )
        dx, dy = LABEL_OFFSETS[representation_label][str(record["model_label"])]
        ax.annotate(
            str(record["model_label"]),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="semibold",
            color=str(style["color"]),
            arrowprops={
                "arrowstyle": "-",
                "color": str(style["color"]),
                "alpha": 0.45,
                "linewidth": 0.8,
            },
            zorder=5,
        )
    ax.set_xlim(*_axis_limits([float(row[x_key]) for row in records]))
    ax.set_ylim(*_axis_limits([float(row[y_key]) for row in records]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel(FIELD_LABELS[x_field])
    ax.set_ylabel(FIELD_LABELS[y_field])
    ax.set_title(f"{representation_label}: {title}", pad=12, fontweight="semibold")
    ax.grid(True, color="#E4E7EC", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4.5)
    ax.set_axisbelow(True)
    ax.legend(handles=_legend_handles(), loc="best", frameon=False, fontsize=10)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(suffix), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_representation_comparison(records: list[dict[str, Any]], out_stem: Path) -> None:
    model_order = ("unet", "unetpp", "unetpp_attn", "ffno", "u_no", "cno")
    labels = {
        "unet": "U-Net",
        "unetpp": "U-Net++",
        "unetpp_attn": "Attention U-Net++",
        "ffno": "FFNO",
        "u_no": "UNO",
        "cno": "CNO",
    }
    lookup = {
        (str(row["representation"]), str(row["model_id"])): row
        for row in records
    }
    def mean_rmse(representation: str, model: str) -> float:
        row = lookup[(representation, model)]
        return float(
            np.mean(
                [
                    float(row[f"case_macro_relative_rmse_{field}_pct"])
                    for field in FIELDS
                ]
            )
        )

    dimension = np.asarray([mean_rmse("dimension", model) for model in model_order])
    sdf = np.asarray([mean_rmse("union_sdf", model) for model in model_order])
    y = np.arange(len(model_order), dtype=np.float64)
    height = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    dimension_bars = ax.barh(
        y - height / 2,
        dimension,
        height=height,
        color="#2563A6",
        label="Dimension",
        zorder=3,
    )
    sdf_bars = ax.barh(
        y + height / 2,
        sdf,
        height=height,
        color="#D97706",
        label="Union SDF",
        zorder=3,
    )
    ax.bar_label(dimension_bars, fmt="%.1f%%", padding=4, fontsize=9, color="#174A7E")
    ax.bar_label(sdf_bars, fmt="%.1f%%", padding=4, fontsize=9, color="#9A4F00")
    ax.set_yticks(y, [labels[model] for model in model_order])
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean relative RMSE across $n_e$, $n_i$, $T_e$, and $\phi$ (%)")
    ax.set_title("Dimension vs SDF: overall prediction error", fontweight="semibold", pad=12)
    ax.grid(True, axis="x", color="#E4E7EC", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.axhline(2.5, color="#B8BEC7", linewidth=1.0)
    ax.set_xlim(0.0, 1.16 * float(max(np.max(dimension), np.max(sdf))))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(suffix), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _write_data(records: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "representation",
        "model_id",
        "model_label",
        "family",
        "test_case_count",
        *(f"case_macro_relative_rmse_{field}_pct" for field in FIELDS),
        "run_root",
        "source_metrics",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in records)


def _write_index(out_dir: Path) -> None:
    lines = [
        "# ICP Dimension / SDF モデルファミリー別RMSE",
        "",
        "GEC-CCP正式RMSE図と同じ色・記号規則で、NN系とNO系を比較します。",
        "",
        "- 青丸: Neural Network（U-Net、U-Net++、Attention U-Net++）",
        "- 赤菱形: Neural Operator（FFNO、UNO、CNO）",
        "- FNO: 正式図から除外",
        "- 指標: 共通36 structure-holdout test casesに対するcase-macro relative RMSE",
        "- 左下ほど両物性の誤差が小さい",
        "",
        "## Dimension",
        "",
        "[![Dimension: ne vs Te](dimension/dimension_relative_rmse_ne_vs_te.png)](dimension/dimension_relative_rmse_ne_vs_te.png)",
        "",
        "[電子密度–電子温度 PNG](dimension/dimension_relative_rmse_ne_vs_te.png) / "
        "[PDF](dimension/dimension_relative_rmse_ne_vs_te.pdf) / "
        "[SVG](dimension/dimension_relative_rmse_ne_vs_te.svg)",
        "",
        "[![Dimension: phi vs ni](dimension/dimension_relative_rmse_phi_vs_ni.png)](dimension/dimension_relative_rmse_phi_vs_ni.png)",
        "",
        "[電位–イオン密度 PNG](dimension/dimension_relative_rmse_phi_vs_ni.png) / "
        "[PDF](dimension/dimension_relative_rmse_phi_vs_ni.pdf) / "
        "[SVG](dimension/dimension_relative_rmse_phi_vs_ni.svg)",
        "",
        "## SDF",
        "",
        "[![SDF: ne vs Te](sdf/sdf_relative_rmse_ne_vs_te.png)](sdf/sdf_relative_rmse_ne_vs_te.png)",
        "",
        "[電子密度–電子温度 PNG](sdf/sdf_relative_rmse_ne_vs_te.png) / "
        "[PDF](sdf/sdf_relative_rmse_ne_vs_te.pdf) / "
        "[SVG](sdf/sdf_relative_rmse_ne_vs_te.svg)",
        "",
        "[![SDF: phi vs ni](sdf/sdf_relative_rmse_phi_vs_ni.png)](sdf/sdf_relative_rmse_phi_vs_ni.png)",
        "",
        "[電位–イオン密度 PNG](sdf/sdf_relative_rmse_phi_vs_ni.png) / "
        "[PDF](sdf/sdf_relative_rmse_phi_vs_ni.pdf) / "
        "[SVG](sdf/sdf_relative_rmse_phi_vs_ni.svg)",
        "",
        "## 数値と再現情報",
        "",
        "- [集約RMSE CSV](model_family_rmse.csv)",
        "- [metadata](metadata.json)",
        "- [生成スクリプト](../../../experiments/conference/scripts/plot_icp_dimension_sdf_fieldwise_rmse.py)",
        "",
        "各点はモデルごとのtest-case平均です。絶対RMSEではなく、物性スケールで正規化した相対RMSEを百分率表示しています。",
    ]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_index_clean(out_dir: Path) -> None:
    """Write the stable Japanese conference index without platform encoding artifacts."""
    lines = [
        "# ICP Dimension / SDF 学習精度比較",
        "",
        "共通36件のstructure-holdout test caseに対するcase-macro相対RMSEです。",
        "FNOは正式図から除外しています。RMSEが低いほど学習精度が高いことを示します。",
        "",
        "## Dimension / SDF直接比較",
        "",
        "**この図を学会発表の主図として正式採用します。**",
        "",
        "[![Dimension vs SDF](dimension_vs_sdf_relative_rmse_comparison.png)]"
        "(dimension_vs_sdf_relative_rmse_comparison.png)",
        "",
        "[PNG](dimension_vs_sdf_relative_rmse_comparison.png) / "
        "[PDF](dimension_vs_sdf_relative_rmse_comparison.pdf) / "
        "[SVG](dimension_vs_sdf_relative_rmse_comparison.svg)",
        "",
        "- 青棒: Dimension",
        "- 橙棒: Union SDF",
        "- 指標: 4物性のcase-macro相対RMSEの単純平均",
        "",
        "発表用の説明文、数値表、解釈上の注意は"
        "[学会正式採用ガイド](ADOPTION_GUIDE.md)を参照してください。",
        "",
        "## NN系 / NO系の色分け比較",
        "",
        "- [Dimension: ne vs Te](dimension/dimension_relative_rmse_ne_vs_Te.png)",
        "- [Dimension: phi vs ni](dimension/dimension_relative_rmse_phi_vs_ni.png)",
        "- [SDF: ne vs Te](sdf/sdf_relative_rmse_ne_vs_Te.png)",
        "- [SDF: phi vs ni](sdf/sdf_relative_rmse_phi_vs_ni.png)",
        "",
        "## 数値と再現情報",
        "",
        "- [集約RMSE CSV](model_family_rmse.csv)",
        "- [metadata](metadata.json)",
        "- [正式採用図カタログ](adopted_figures.csv)",
        "- [生成スクリプト](../../../experiments/conference/scripts/plot_icp_dimension_sdf_fieldwise_rmse.py)",
    ]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    records = _collect()
    outputs: dict[str, list[str]] = {}
    for representation, label, folder in (
        ("dimension", "Dimension", "dimension"),
        ("union_sdf", "Union SDF", "sdf"),
    ):
        selected = [row for row in records if row["representation"] == representation]
        for x_field, y_field, title in PAIR_CONFIGS:
            stem = out_dir / folder / f"{folder}_relative_rmse_{x_field}_vs_{y_field}"
            _plot_pair(
                selected,
                representation_label=label,
                x_field=x_field,
                y_field=y_field,
                title=title,
                out_stem=stem,
            )
            outputs[f"{folder}_{x_field}_vs_{y_field}"] = [
                path.relative_to(out_dir).as_posix()
                for path in (stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".svg"))
            ]
    data_path = out_dir / "model_family_rmse.csv"
    _write_data(records, data_path)
    comparison_stem = out_dir / "dimension_vs_sdf_relative_rmse_comparison"
    _plot_representation_comparison(records, comparison_stem)
    outputs["dimension_vs_sdf"] = [
        comparison_stem.with_suffix(suffix).relative_to(out_dir).as_posix()
        for suffix in (".png", ".pdf", ".svg")
    ]
    metadata = {
        "purpose": "ICP conference NN-versus-NO field-wise RMSE comparison",
        "layout_reference": "GEC-CCP adopted field-wise RMSE scatter plots",
        "metric": "100 * arithmetic mean across common test cases of per-case plasma physical_rel_l2",
        "evaluation": {"protocol": "structure_holdout", "test_case_count": 36},
        "family_style": FAMILY_STYLE,
        "representations": ["dimension", "union_sdf"],
        "fields": list(FIELDS),
        "models": [
            {
                "representation": row["representation"],
                "model_id": row["model_id"],
                "model_label": row["model_label"],
                "family": row["family"],
                "run_root": row["run_root"],
            }
            for row in records
        ],
        "excluded_models": {
            "fno": "excluded from the adopted figure at the user's request",
        },
        "conference_adoption": {
            "status": "adopted_for_icp_conference",
            "primary_figure_id": "icp_dimension_vs_sdf_overall_prediction_error",
            "primary_figure": "dimension_vs_sdf_relative_rmse_comparison.png",
            "adoption_guide": "ADOPTION_GUIDE.md",
            "catalog": "adopted_figures.csv",
        },
        "data": "model_family_rmse.csv",
        "outputs": outputs,
        "generator": "experiments/conference/scripts/plot_icp_dimension_sdf_fieldwise_rmse.py",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_index_clean(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
