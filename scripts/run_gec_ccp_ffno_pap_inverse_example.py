"""Conference-scale PAP inverse example using the trained GEC-CCP FFNO.

A radial electron-density profile is sampled from a selected COMSOL case at an
arbitrary physical height. Frozen multiplicative noise emulates a plasma
absorption probe (PAP) scan. CMA-ES or TPE then searches the configured input
conditions, optionally including gamma, while the reactor geometry stays
fixed. All residuals are evaluated in linear density space; the script never
applies a logarithmic target transform.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PLASMA_SURROGATE_ENABLE_TORCH"] = "1"

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import optuna
import torch
import yaml

from plasma_surrogate.core.dataset_io import load_dataset
from plasma_surrogate.core.input_modes import load_checkpoint_metadata_with_input_mode
from plasma_surrogate.core.run_bundle import RunBundleLoader
from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.infer.assimilation import LinearNeProfileAssimilationEngine, LinearNeProfileObservation
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.infer.optimize import OptimizeRunner
from plasma_surrogate.infer.profiles import extract_radial_profile
from plasma_surrogate.models.checkpoint import load_checkpoint


DEFAULT_RUN_ROOT = Path("runs/gec_ccp_nn_operator_comparison_v1/seed_412/n78/ffno")
DEFAULT_CASE_ID = "case_td003_pp0_3_gamma_004__steady"
DEFAULT_OUT_ROOT = Path("runs/gec_ccp_ffno_pap_inverse_example_v1")
BASE_SEARCH_SPACE = {"PP0": (1.0, 5.0), "PA": (0.05, 0.20)}
GAMMA_BOUNDS = (0.04, 0.10)
RESPONSE_COLOR_FLOOR_PERCENT = 1.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_ffno_engine(
    run_root: Path,
    cfg: dict[str, Any],
    geometry_root: Path,
    out_root: Path,
) -> InferenceEngine:
    checkpoint = run_root / "models" / "ffno" / "eval_protocol" / "interp" / "checkpoints"
    model = load_checkpoint(checkpoint)
    checkpoint_meta, input_meta = load_checkpoint_metadata_with_input_mode(checkpoint / "meta.json")
    bundle = RunBundleLoader.load(run_root, model=model)
    spatial = bundle.spatial_transform_artifacts_for_checkpoint(checkpoint_meta)
    benchmark_cfg = dict(cfg.get("benchmark", cfg) or {})
    train_cfg = dict(benchmark_cfg.get("train", cfg.get("train", {})))
    input_features_cfg = dict(dict(train_cfg.get("ffno", {})).get("input_features", {}))
    input_scaling_cfg = dict(input_features_cfg.get("scale_using_train_stats", {})) or dict(
        input_features_cfg.get("input_scaling", {})
    )
    engine = InferenceEngine(
        model=model,
        cond_schema=bundle.cond_schema_obj(),
        axis_schema=bundle.axis_schema_obj(),
        geometry_provider=build_geometry_provider(
            geometry_root,
            provider_mode=str(input_meta.get("geometry_provider_mode_effective", "fixed")),
        ),
        output_dir=out_root / "_engine_outputs",
        transform_bundle=bundle.transform_bundle_for_checkpoint(checkpoint_meta),
        cond_stats=dict(bundle.schemas.get("cond_stats", {})),
        phi_mode=str(benchmark_cfg.get("phi_mode", "direct")),
        ood_cfg={"qoi": {"uniformity": {"target": "ne"}}},
        feature_store=bundle.geometry_store,
        coord_scaler=dict(bundle.transforms.get("coord_scaler", {})),
        coord_feature_scaler=spatial["coord_feature_scaler"],
        coord_feature_pack=bundle.schemas.get("coord_feature_pack"),
        coord_distance_transform_stats=spatial["distance_transform_stats"],
        coord_input_scaling_cfg=input_scaling_cfg,
        coord_input_features_cfg=input_features_cfg,
        grid_input_features_cfg=input_features_cfg,
        input_mode=str(input_meta.get("input_mode_effective", "table_plus_structure")),
        input_mode_meta=dict(input_meta),
        checkpoint_input_mode_meta=dict(input_meta),
        checkpoint_meta=dict(checkpoint_meta),
        target_role_schema=dict(bundle.schemas.get("target_role_schema", {}) or {}),
        structure_descriptor_pack=bundle.schemas.get("structure_descriptor_pack"),
        latent_feature_pack=bundle.schemas.get("latent_feature_pack"),
    )
    if str(getattr(model, "device", "")) != "cpu":
        raise RuntimeError(f"CPU-only example requested; model device={getattr(model, 'device', None)!r}")
    return engine


def _geom(case: dict[str, Any]) -> dict[str, str]:
    geom_id = str(case.get("geom_id", case.get("base_name", "default"))).strip() or "default"
    return {"geom_id": geom_id}


def _axis(case: dict[str, Any]) -> dict[str, Any]:
    return {"mode": "steady", "value": float(case.get("axis", 0.0))}


def _profile(field: np.ndarray, geom_ctx: Any, z_m: float, r_m: np.ndarray) -> np.ndarray:
    result = extract_radial_profile(field, geom_ctx, z_m=z_m, r_m=r_m, require_plasma=True)
    if not np.all(result.valid):
        raise ValueError(f"PAP locations outside plasma: {result.r_m[~result.valid].tolist()}")
    return np.asarray(result.values, dtype=np.float64)


def _rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(ref)) / np.linalg.norm(np.asarray(ref)))


def _normalized_shape_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(pred_arr / np.linalg.norm(pred_arr) - ref_arr / np.linalg.norm(ref_arr)))


def _best_so_far(values: list[float]) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(values, dtype=np.float64))


def _pca_projection(
    trial_rows: list[dict[str, Any]],
    search_space: dict[str, tuple[float, float]],
    truth_cond: dict[str, float],
    best_cond: dict[str, float],
    initial_cond: dict[str, float],
) -> dict[str, Any]:
    variables = list(search_space)
    values = np.asarray(
        [[float(row[key]) for key in variables] for row in trial_rows],
        dtype=np.float64,
    )
    lower = np.asarray([float(search_space[key][0]) for key in variables], dtype=np.float64)
    span = np.asarray(
        [float(search_space[key][1]) - float(search_space[key][0]) for key in variables],
        dtype=np.float64,
    )
    scaled = (values - lower[None, :]) / span[None, :]
    mean_scaled = np.mean(scaled, axis=0)
    centered = scaled - mean_scaled[None, :]
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = np.asarray(right_vectors[:2], dtype=np.float64)
    for index in range(components.shape[0]):
        anchor = int(np.argmax(np.abs(components[index])))
        if components[index, anchor] < 0.0:
            components[index] *= -1.0
    scores = centered @ components.T
    variance = singular_values * singular_values
    explained_ratio = variance[:2] / np.sum(variance)

    def project(cond: dict[str, float]) -> np.ndarray:
        point = np.asarray([float(cond[key]) for key in variables], dtype=np.float64)
        return (((point - lower) / span) - mean_scaled) @ components.T

    return {
        "variables": variables,
        "lower": lower,
        "span": span,
        "mean_scaled": mean_scaled,
        "components": components,
        "explained_variance_ratio": explained_ratio,
        "scores": scores,
        "truth_score": project(truth_cond),
        "best_score": project(best_cond),
        "initial_score": project(initial_cond),
    }


def _pca_projection_from_reference(
    reference_trial_rows: list[dict[str, Any]],
    search_space: dict[str, tuple[float, float]],
    truth_cond: dict[str, float],
    best_cond: dict[str, float],
    initial_cond: dict[str, float],
    reference_meta: dict[str, Any],
) -> dict[str, Any]:
    variables = list(reference_meta["variables"])
    if variables != list(search_space):
        raise ValueError(
            f"PCA reference variables {variables!r} do not match search space {list(search_space)!r}"
        )
    lower = np.asarray([float(search_space[key][0]) for key in variables], dtype=np.float64)
    span = np.asarray(
        [float(search_space[key][1]) - float(search_space[key][0]) for key in variables],
        dtype=np.float64,
    )
    mean_scaled = np.asarray(reference_meta["mean_scaled"], dtype=np.float64)
    components = np.asarray(reference_meta["components"], dtype=np.float64)
    explained_ratio = np.asarray(reference_meta["explained_variance_ratio"], dtype=np.float64)

    def project(cond: dict[str, float]) -> np.ndarray:
        point = np.asarray([float(cond[key]) for key in variables], dtype=np.float64)
        return (((point - lower) / span) - mean_scaled) @ components.T

    reference_values = np.asarray(
        [[float(row[key]) for key in variables] for row in reference_trial_rows],
        dtype=np.float64,
    )
    reference_scores = ((reference_values - lower[None, :]) / span[None, :] - mean_scaled) @ components.T
    return {
        "variables": variables,
        "lower": lower,
        "span": span,
        "mean_scaled": mean_scaled,
        "components": components,
        "explained_variance_ratio": explained_ratio,
        "scores": reference_scores,
        "truth_score": project(truth_cond),
        "best_score": project(best_cond),
        "initial_score": project(initial_cond),
    }


def _plot_summary(
    out_root: Path,
    *,
    r_mm: np.ndarray,
    truth_profile: np.ndarray,
    observed: np.ndarray,
    initial_profile: np.ndarray,
    best_profile: np.ndarray,
    trial_rows: list[dict[str, Any]],
    response_trial_rows: list[dict[str, Any]] | None,
    truth_cond: dict[str, float],
    best_cond: dict[str, float],
    optimizer_name: str,
    include_gamma: bool,
    pca_projection: dict[str, Any] | None,
    output_stem: str = "pap_inverse_summary",
    selected_profile_label: str = "optimized FFNO",
    selected_marker_label: str | None = None,
    selected_trial: int | None = None,
) -> None:
    scale = 1.0e15
    losses = [float(row["loss"]) for row in trial_rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))

    ax = axes[0]
    ax.plot(r_mm, truth_profile / scale, color="black", linewidth=2.0, label="COMSOL truth")
    ax.scatter(r_mm, observed / scale, s=23, facecolor="white", edgecolor="black", label="PAP pseudo-data")
    ax.plot(r_mm, initial_profile / scale, color="#999999", linestyle="--", linewidth=1.5, label="initial FFNO")
    ax.plot(
        r_mm,
        best_profile / scale,
        color="#D55E00",
        linewidth=2.0,
        label=selected_profile_label,
    )
    ax.set_xlabel("Radius r [mm]")
    ax.set_ylabel(r"Electron density $n_e$ [$10^{15}$ m$^{-3}$]")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    trial_index = np.arange(1, len(losses) + 1)
    ax.scatter(trial_index, 100.0 * np.asarray(losses), s=13, alpha=0.28, color="#0072B2", label="trial")
    ax.plot(trial_index, 100.0 * _best_so_far(losses), color="#D55E00", linewidth=2.0, label="best so far")
    if selected_trial is not None:
        selected_index = int(selected_trial) - 1
        if not 0 <= selected_index < len(losses):
            raise ValueError(f"selected_trial outside trial history: {selected_trial}")
        ax.scatter(
            [selected_trial],
            [100.0 * losses[selected_index]],
            marker="X",
            s=58,
            color="#CC79A7",
            edgecolor="black",
            label=f"selected T{selected_trial}",
            zorder=6,
        )
    ax.set_yscale("log")
    ax.set_xlabel("FFNO evaluations")
    ax.set_ylabel("PAP profile relative L2 [%]")
    ax.grid(which="both", alpha=0.22)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    if include_gamma:
        if pca_projection is None:
            raise ValueError("PCA projection is required for the three-variable response surface")
        scores = np.asarray(pca_projection["scores"], dtype=np.float64)
        response_rows = response_trial_rows if response_trial_rows is not None else trial_rows
        response_losses = [float(row["loss"]) for row in response_rows]
        loss_percent = 100.0 * np.asarray(response_losses, dtype=np.float64)
        vmin = RESPONSE_COLOR_FLOOR_PERCENT
        vmax = float(np.max(loss_percent))
        levels = np.geomspace(vmin, vmax, 22)
        truth_score = np.asarray(pca_projection["truth_score"], dtype=np.float64)
        best_score = np.asarray(pca_projection["best_score"], dtype=np.float64)
        initial_score = np.asarray(pca_projection["initial_score"], dtype=np.float64)
        # Keep the map extent fixed by the response-map samples and the common
        # truth marker.  Run-specific initial/best points must not rescale or
        # distort the shared comparison axes.
        display_points = np.vstack([scores, truth_score[None, :]])
        display_min = np.min(display_points, axis=0)
        display_max = np.max(display_points, axis=0)
        display_padding = 0.07 * np.maximum(display_max - display_min, 1.0e-6)
        grid_x, grid_y = np.meshgrid(
            np.linspace(
                float(display_min[0] - display_padding[0]),
                float(display_max[0] + display_padding[0]),
                220,
            ),
            np.linspace(
                float(display_min[1] - display_padding[1]),
                float(display_max[1] + display_padding[1]),
                220,
            ),
        )
        grid_points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
        score_scale = np.std(scores, axis=0)
        normalized_scores = scores / score_scale[None, :]
        normalized_grid = grid_points / score_scale[None, :]
        log_loss = np.log(loss_percent)
        smoothed_log_loss = np.empty(normalized_grid.shape[0], dtype=np.float64)
        bandwidth = 0.18
        for start in range(0, normalized_grid.shape[0], 2048):
            stop = min(start + 2048, normalized_grid.shape[0])
            delta = normalized_grid[start:stop, None, :] - normalized_scores[None, :, :]
            distance_sq = np.sum(delta * delta, axis=2)
            distance_sq -= np.min(distance_sq, axis=1, keepdims=True)
            weights = np.exp(-0.5 * distance_sq / (bandwidth * bandwidth))
            smoothed_log_loss[start:stop] = (weights @ log_loss) / np.sum(weights, axis=1)
        surface = np.exp(smoothed_log_loss).reshape(grid_x.shape)
        image = ax.contourf(
            grid_x,
            grid_y,
            surface,
            levels=levels,
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap="viridis_r",
            extend="max",
        )
        explained = np.asarray(pca_projection["explained_variance_ratio"], dtype=np.float64)
        ax.scatter(
            initial_score[0],
            initial_score[1],
            marker="D",
            s=54,
            color="#56B4E9",
            edgecolor="black",
            label="initial",
            zorder=5,
        )
        ax.scatter(
            truth_score[0],
            truth_score[1],
            marker="*",
            s=145,
            color="white",
            edgecolor="black",
            label="truth",
            zorder=5,
        )
        ax.scatter(
            best_score[0],
            best_score[1],
            marker="X",
            s=70,
            color="#D55E00",
            edgecolor="black",
            label=selected_marker_label or f"{optimizer_name} best",
            zorder=5,
        )
        ax.set_xlabel(f"PC1 ({100.0 * explained[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({100.0 * explained[1]:.1f}% variance)")
        colorbar_label = "PAP profile relative L2 [%]"
        colorbar_ticks = [
            tick for tick in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0) if vmin <= tick <= vmax
        ]
    else:
        image = ax.scatter(
            [float(row["PP0"]) for row in trial_rows],
            [float(row["PA"]) for row in trial_rows],
            c=100.0 * np.asarray(losses),
            cmap="viridis_r",
            norm=matplotlib.colors.LogNorm(
                vmin=RESPONSE_COLOR_FLOOR_PERCENT, vmax=100.0 * max(losses)
            ),
            s=24,
            alpha=0.75,
        )
        colorbar_label = "profile relative L2 [%]"
        colorbar_ticks = None
        ax.scatter(
            truth_cond["PP0"],
            truth_cond["PA"],
            marker="*",
            s=145,
            color="white",
            edgecolor="black",
            label="truth",
        )
        ax.scatter(
            best_cond["PP0"],
            best_cond["PA"],
            marker="X",
            s=70,
            color="#D55E00",
            edgecolor="black",
            label=selected_marker_label or f"{optimizer_name} best",
        )
        ax.set_xlabel("PP0 [W]")
        ax.set_ylabel("PA [Torr]")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, frameon=False)
    colorbar = fig.colorbar(image, ax=ax, label=colorbar_label, pad=0.02, ticks=colorbar_ticks)
    if colorbar_ticks is not None:
        colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_root / f"{output_stem}.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(fig)


def _write_index(out_root: Path, summary: dict[str, Any]) -> None:
    truth = summary["truth_conditions"]
    best = summary["best_conditions"]
    initial = summary["initial_conditions"]
    optimizer = str(summary["optimizer"])
    include_gamma = "gamma" in summary["search_space"]
    pca_meta = dict(summary.get("pca_response_surface", {}) or {})
    pca_reference_run = pca_meta.get("reference_run")
    case_scope = "held-out test" if summary["case_is_held_out_interp_test"] else "non-held-out train"
    best_to_truth_ratio = float(summary["best_sensor_loss"]) / float(summary["truth_input_sensor_loss"])
    if best_to_truth_ratio <= 1.10:
        conclusion_lines = [
            "したがって、発表では「真のパラメータを一意同定した」ではなく、",
            "「PAP profileを真値入力時のFFNO誤差床と同程度まで高速に再現した」と結論づけます。",
        ]
    else:
        conclusion_lines = [
            f"{summary['n_trials']} trialで形状と主ピークは回復しましたが、PAP lossは真値入力時の {best_to_truth_ratio:.2f}倍です。",
            "したがってノイズ床への到達は主張せず、追加trialで改善余地がある難しい探索例として扱います。",
        ]
    search_text = "`PP0: 1–5 W`, `PA: 0.05–0.20 Torr`"
    fixed_text = f"`Td={truth['Td']}`、構造固定"
    if include_gamma:
        search_text += ", `gamma: 0.04–0.10`"
    else:
        fixed_text = f"`Td={truth['Td']}`, `gamma={truth['gamma']}`、構造固定"
    if include_gamma:
        result_header = "| 条件 | PP0 [W] | PA [Torr] | gamma | PAP loss [%] |"
        result_rule = "|---|---:|---:|---:|---:|"
        truth_row = (
            f"| 真値 | {truth['PP0']:.4g} | {truth['PA']:.4g} | {truth['gamma']:.4g} | "
            f"{100.0 * summary['truth_input_sensor_loss']:.3f} |"
        )
        initial_row = (
            f"| 初期値 | {initial['PP0']:.4g} | {initial['PA']:.4g} | {initial['gamma']:.4g} | "
            f"{100.0 * summary['initial_sensor_loss']:.3f} |"
        )
        best_row = (
            f"| {optimizer}最良 | {best['PP0']:.4g} | {best['PA']:.4g} | {best['gamma']:.4g} | "
            f"{100.0 * summary['best_sensor_loss']:.3f} |"
        )
    else:
        result_header = "| 条件 | PP0 [W] | PA [Torr] | PAP loss [%] |"
        result_rule = "|---|---:|---:|---:|"
        truth_row = (
            f"| 真値 | {truth['PP0']:.4g} | {truth['PA']:.4g} | "
            f"{100.0 * summary['truth_input_sensor_loss']:.3f} |"
        )
        initial_row = (
            f"| 初期値 | {initial['PP0']:.4g} | {initial['PA']:.4g} | "
            f"{100.0 * summary['initial_sensor_loss']:.3f} |"
        )
        best_row = (
            f"| {optimizer}最良 | {best['PP0']:.4g} | {best['PA']:.4g} | "
            f"{100.0 * summary['best_sensor_loss']:.3f} |"
        )
    lines = [
        "# FFNOによるPAP疑似計測の条件探索",
        "",
        f"{case_scope} COMSOL電子密度から作った単一高さのPAP半径方向疑似計測に対し、",
        f"学習済みFFNOをCPUで反復評価して {', '.join(f'`{key}`' for key in summary['search_space'])} を探索する学会発表用例題です。",
        "",
        "![PAP inverse summary](pap_inverse_summary.png)",
        "",
        "## 例題の目的",
        "",
        "PAPは吸収共鳴から局所電子密度を求める診断です。本例題ではプローブを一定高さで半径方向へ走査した",
        "19点の電子密度に、FFNO予測が最も合う運転条件を求めます。厳密な未知物理係数の同定ではなく、",
        f"観測した密度profileを再現する {', '.join(f'`{key}`' for key in summary['search_space'])} の条件チューニングを目的にします。",
        "",
        f"探索条件を $\\theta=({','.join(summary['search_space'])})$、PAP疑似計測を $n_e^{{\\mathrm{{PAP}}}}(r_i)$ とすると、目的関数は",
        "",
        "$$",
        "J(\\theta)=",
        "\\frac{\\sqrt{\\sum_i[ n_e^{\\mathrm{FFNO}}(r_i,z_{\\mathrm{PAP}};\\theta)-n_e^{\\mathrm{PAP}}(r_i)]^2}}",
        "{\\sqrt{\\sum_i[n_e^{\\mathrm{PAP}}(r_i)]^2}}.",
        "$$",
        "",
        "電子密度は物理単位の線形値を使い、対数変換や追加の補正係数は使いません。",
        "`--z-mm`で計測高さを変更でき、補間stencilがplasma外へ出る位置はエラーとして除外します。",
        "",
        "## 固定した問題設定",
        "",
        "| 項目 | 設定 |",
        "|---|---|",
        f"| 疑似真値 | {case_scope} `{summary['case_id']}` のCOMSOL電子密度 |",
        f"| PAP位置 | `z={summary['z_mm']:.1f} mm`, `r=0..{summary['r_max_mm']:.0f} mm`, {summary['sensor_points']}点 |",
        f"| ノイズ | 各点に固定seedの乗算Gaussianノイズ、標準偏差 {100.0 * summary['point_noise_rel']:.1f}%、±10%でclip |",
        f"| 探索変数 | {search_text} |",
        f"| 固定条件 | {fixed_text} |",
        f"| 探索 | {optimizer}、{summary['n_trials']} trial、CPU FFNO |",
        "| 目的関数 | 線形電子密度profileのrelative L2（対数変換なし） |",
        "",
        "本caseとPAP高さは、初期profileとの形状差を明瞭に示す発表例として選択しています。",
        "モデル精度の集約評価やcheckpoint選択には使用していません。",
        "",
        *(
            [
                (
                    f"右図は基準run `{pca_reference_run}` のPCA基底と"
                    f"{pca_meta['response_trials']}個の実評価log-lossを固定背景として再利用しています。"
                    if pca_reference_run
                    else f"右図は探索変数を各探索範囲で0–1正規化してPCAし、{pca_meta['response_trials']}個の実評価log-lossをPC1–PC2平面で"
                ),
                "Gaussian kernel平滑化した表示用応答曲面です。図を読みやすくするため、PC1–PC2表示範囲全体を推定値で塗りつぶしています。",
                "trial点が疎な端部は表示用の平滑化推定であり、最適化で直接評価した値ではありません。",
                f"PC1とPC2の累積寄与率は {100.0 * sum(pca_meta['explained_variance_ratio']):.1f}%です。",
                "",
            ]
            if pca_meta
            else []
        ),
        "`Td`は本データで構造IDと完全に対応しているため、固定構造のまま連続変化させていません。",
        "`Td`を探索する場合は対応構造も同時に切り替える離散・混合変数問題として別に扱う必要があります。",
        "",
        "## 結果",
        "",
        result_header,
        result_rule,
        truth_row,
        initial_row,
        best_row,
        "",
        f"最良値はtrial {summary['best_trial']}で得られ、真値入力時lossの10%以内へ初めて到達したのは"
        f"trial {summary['first_within_10pct_of_truth_input_trial']}です。",
        "",
        f"振幅を除いた正規化形状誤差は、初期 {100.0 * summary['initial_normalized_shape_l2']:.2f}%から "
        f"{optimizer}最良 {100.0 * summary['best_normalized_shape_l2']:.2f}%へ低下しました。",
        f"最大密度の半径位置は、COMSOL真値 {summary['truth_peak_radius_mm']:.1f} mm、"
        f"初期FFNO {summary['initial_peak_radius_mm']:.1f} mm、最適化後FFNO {summary['best_peak_radius_mm']:.1f} mmです。",
        "",
        f"{optimizer}最良profileのノイズなしCOMSOL真値に対するrelative L2は {100.0 * summary['best_profile_truth_rel_l2']:.3f}%です。",
        "最良条件が元のCOMSOL条件と完全には一致しないのは、観測ノイズ、FFNO近似誤差、",
        f"および単一高さの電子密度だけでは {', '.join(f'`{key}`' for key in summary['search_space'])} の組合せに等価性が残るためです。",
        *conclusion_lines,
        "",
        f"{optimizer}によるFFNOの{summary['n_trials']}回探索は {summary['optimization_wall_s']:.2f} sです。",
        f"同数のCOMSOL逐次評価は既存ケース平均から約 {summary['direct_comsol_estimated_hours']:.2f} hと推定され、",
        f"online探索の推定高速化は約 {summary['online_speedup_estimate']:.0f}倍です。",
        "COMSOL時間と高速化は既存ケース時間に基づく推定であり、新規COMSOL探索を実行した実測値ではありません。",
        "",
        "数値データ: [疑似PAP計測](pap_observation.csv) / [全trial](trials.csv) / [summary](summary.json)",
        "",
        "## 参考",
        "",
        "- [プラズマ吸収プローブによるプロセスプラズマの電子密度測定](https://doi.org/10.1585/jspf.78.998)",
        "- [COMSOL GEC-CCP example: mid-gap radial electron-density profile](https://doc.comsol.com/6.4/doc/com.comsol.help.models.plasma.argon_gec_ccp/argon_gec_ccp.html)",
    ]
    (out_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--z-mm", type=float, default=12.7)
    parser.add_argument("--r-max-mm", type=float, default=90.0)
    parser.add_argument("--r-step-mm", type=float, default=5.0)
    parser.add_argument("--point-noise-rel", type=float, default=0.05)
    parser.add_argument("--noise-seed", type=int, default=20260717)
    parser.add_argument("--optimizer-seed", type=int, default=412)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--optimizer", choices=("tpe", "cmaes"), default="tpe")
    parser.add_argument("--include-gamma", action="store_true")
    parser.add_argument(
        "--allow-non-held-out-case",
        action="store_true",
        help="Allow a train/validation COMSOL case for presentation-only inverse examples.",
    )
    parser.add_argument("--initial-pp0", type=float, default=1.5)
    parser.add_argument("--initial-pa", type=float, default=0.18)
    parser.add_argument("--initial-gamma", type=float, default=0.08)
    parser.add_argument(
        "--pca-reference-run",
        type=Path,
        default=None,
        help="Reuse this run's PCA basis, trial losses, and response map for comparable plots.",
    )
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 20))
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.trials < 20:
        raise ValueError("conference example requires at least 20 trials")
    if args.r_step_mm <= 0.0 or args.r_max_mm <= 0.0:
        raise ValueError("radial sampling limits must be positive")

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only example requested but CUDA remains visible")
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    with (args.run_root / "resolved_config.yaml").open("r", encoding="utf-8") as stream:
        cfg = dict(yaml.safe_load(stream) or {})
    dataset = load_dataset(cfg, args.run_root)
    cases = {str(case["case_id"]): case for case in dataset.cases}
    if args.case_id not in cases:
        raise KeyError(f"case not found: {args.case_id}")
    split_path = args.run_root / "preprocessing" / "split" / "split_interp_v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    case_is_held_out = args.case_id in set(split.get("test", []))
    if not case_is_held_out and not args.allow_non_held_out_case:
        raise ValueError(f"pseudo-truth must be a held-out interpolation test case: {args.case_id}")

    load_started = time.perf_counter()
    engine = _load_ffno_engine(args.run_root, cfg, Path(dataset.geometry_root), out_root)
    engine_load_s = time.perf_counter() - load_started
    case = cases[args.case_id]
    truth_cond = {str(key): float(value) for key, value in dict(case["cond"]).items()}
    geom = _geom(case)
    axis = _axis(case)
    geom_ctx = engine._get_geom_ctx(engine._validate_geom_ref(geom), axis=axis)
    z_m = 1.0e-3 * float(args.z_mm)
    r_m = 1.0e-3 * np.arange(0.0, args.r_max_mm + 0.5 * args.r_step_mm, args.r_step_mm)
    truth_profile = _profile(case["y"]["ne"], geom_ctx, z_m, r_m)

    rng = np.random.default_rng(int(args.noise_seed))
    noise_draw = np.clip(rng.normal(0.0, args.point_noise_rel, size=truth_profile.shape), -0.10, 0.10)
    observed = truth_profile * (1.0 + noise_draw)
    density_scale = float(np.sqrt(np.mean(observed * observed)))
    observation = LinearNeProfileObservation(
        r_m=r_m,
        z_m=z_m,
        ne_obs=observed,
        density_scale=density_scale,
        covariance_scaled=np.eye(r_m.size, dtype=np.float64),
        case_id=args.case_id,
    )
    search_space = dict(BASE_SEARCH_SPACE)
    fixed_cond = {"Td": truth_cond["Td"], "gamma": truth_cond["gamma"]}
    if args.include_gamma:
        search_space["gamma"] = GAMMA_BOUNDS
        fixed_cond.pop("gamma")
    assimilation = LinearNeProfileAssimilationEngine(
        engine,
        observation=observation,
        fixed_cond=fixed_cond,
        setpoints={},
        nuisance_rel_sigma={},
    )
    initial = {"PP0": float(args.initial_pp0), "PA": float(args.initial_pa)}
    if args.include_gamma:
        initial["gamma"] = float(args.initial_gamma)
    objective_cfg = {
        "mode": "weighted_sum",
        "terms": [{"key": "profile_linear_rel_l2", "direction": "min", "weight": 1.0}],
    }

    initial_result = assimilation.single_run_aggregated(cond=initial, geom=geom, axis=axis, save_outputs=False)
    initial_profile = _profile(initial_result.fields_phys["ne"], geom_ctx, z_m, r_m)
    truth_search = {key: truth_cond[key] for key in search_space}
    truth_result = assimilation.single_run_aggregated(cond=truth_search, geom=geom, axis=axis, save_outputs=False)
    truth_ffno_profile = _profile(truth_result.fields_phys["ne"], geom_ctx, z_m, r_m)

    started = time.perf_counter()
    if args.optimizer == "tpe":
        backend_cfg = {
            "sampler": "tpe",
            "n_startup_trials": 12,
            "multivariate": True,
            "initial_cond": initial,
        }
    else:
        backend_cfg = {
            "sampler": "cmaes",
            "n_startup_trials": 1,
            "sigma0": 0.25,
            "popsize": 9,
            "x0": initial,
            "initial_cond": initial,
        }
    optimization = OptimizeRunner(assimilation).run(
        space=search_space,
        geom_space={},
        n_trials=int(args.trials),
        geom_ref=geom,
        axis=axis,
        seed=int(args.optimizer_seed),
        backend="optuna",
        backend_cfg=backend_cfg,
        objective_cfg=objective_cfg,
        output_cfg={"save_fields": "none"},
    )
    optimization_wall_s = time.perf_counter() - started
    if optimization.status != "succeeded":
        raise RuntimeError(f"TPE failed: invalid_trials={optimization.invalid_trial_count}")
    best_cond = {str(key): float(value) for key, value in optimization.best_cond.items()}
    best_result = assimilation.single_run_aggregated(cond=best_cond, geom=geom, axis=axis, save_outputs=False)
    best_profile = _profile(best_result.fields_phys["ne"], geom_ctx, z_m, r_m)

    trial_rows: list[dict[str, Any]] = []
    for number, trial in enumerate(optimization.trials, start=1):
        cond = dict(trial.get("cond", {}) or {})
        row = {
            "trial": number,
            "PP0": float(cond["PP0"]),
            "PA": float(cond["PA"]),
        }
        if args.include_gamma:
            row["gamma"] = float(cond["gamma"])
        row.update(
            {
                "loss": float(trial["objective_value"]),
                "feasible": bool(trial.get("feasible", True)),
            }
        )
        trial_rows.append(row)
    observation_rows = [
        {
            "r_mm": float(1000.0 * radius),
            "z_mm": float(args.z_mm),
            "truth_ne_m-3": float(truth_value),
            "observed_ne_m-3": float(obs_value),
            "relative_noise": float(noise),
        }
        for radius, truth_value, obs_value, noise in zip(r_m, truth_profile, observed, noise_draw, strict=True)
    ]
    _write_csv(out_root / "pap_observation.csv", observation_rows)
    _write_csv(out_root / "trials.csv", trial_rows)

    source_conditions = Path("outputs_merged_td_all_success_pa_ext0520/conditions.csv")
    elapsed: list[float] = []
    with source_conditions.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("status") == "success" and row.get("elapsed_s"):
                elapsed.append(float(row["elapsed_s"]))
    mean_comsol_s = float(np.mean(elapsed))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "title": "FFNO PAP radial-profile inverse example",
        "model": "ffno",
        "device": "cpu",
        "run_root": str(args.run_root),
        "case_id": args.case_id,
        "case_is_held_out_interp_test": bool(case_is_held_out),
        "case_selection": (
            "presentation-only shape-contrast selection; not used for aggregate accuracy or checkpoint selection"
            if case_is_held_out
            else "presentation-only near-current-optimum train case; not valid as held-out accuracy evidence"
        ),
        "pseudo_truth_source": (
            "COMSOL held-out test field; not FFNO-generated"
            if case_is_held_out
            else "COMSOL non-held-out field; not FFNO-generated; presentation-only"
        ),
        "observation": "PAP local electron density sampled as a radial scan at one height",
        "objective": "linear electron-density profile relative L2; no logarithm",
        "z_mm": float(args.z_mm),
        "r_max_mm": float(args.r_max_mm),
        "r_step_mm": float(args.r_step_mm),
        "sensor_points": int(r_m.size),
        "point_noise_rel": float(args.point_noise_rel),
        "noise_clip_rel": 0.10,
        "noise_seed": int(args.noise_seed),
        "optimizer": args.optimizer.upper() if args.optimizer == "tpe" else "CMA-ES",
        "optimizer_config": dict(optimization.backend_cfg),
        "optimizer_seed": int(args.optimizer_seed),
        "requested_trials": int(args.trials),
        "n_trials": len(trial_rows),
        "search_space": {key: list(value) for key, value in search_space.items()},
        "fixed_conditions": {**fixed_cond, "geometry": geom},
        "truth_conditions": truth_cond,
        "initial_conditions": initial,
        "best_conditions": best_cond,
        "realized_noise_rel_l2": _rel_l2(observed, truth_profile),
        "initial_sensor_loss": _rel_l2(initial_profile, observed),
        "truth_input_sensor_loss": _rel_l2(truth_ffno_profile, observed),
        "best_sensor_loss": _rel_l2(best_profile, observed),
        "best_profile_truth_rel_l2": _rel_l2(best_profile, truth_profile),
        "initial_normalized_shape_l2": _normalized_shape_l2(initial_profile, truth_profile),
        "best_normalized_shape_l2": _normalized_shape_l2(best_profile, truth_profile),
        "truth_peak_radius_mm": float(1000.0 * r_m[int(np.argmax(truth_profile))]),
        "initial_peak_radius_mm": float(1000.0 * r_m[int(np.argmax(initial_profile))]),
        "best_peak_radius_mm": float(1000.0 * r_m[int(np.argmax(best_profile))]),
        "PP0_abs_error": abs(best_cond["PP0"] - truth_cond["PP0"]),
        "PA_abs_error": abs(best_cond["PA"] - truth_cond["PA"]),
        "engine_load_s": engine_load_s,
        "optimization_wall_s": optimization_wall_s,
        "mean_existing_comsol_case_s": mean_comsol_s,
        "direct_comsol_estimated_hours": mean_comsol_s * len(trial_rows) / 3600.0,
        "online_speedup_estimate": mean_comsol_s * len(trial_rows) / optimization_wall_s,
        "direct_cost_is_estimate": True,
    }
    summary["best_to_truth_input_loss_ratio"] = (
        float(summary["best_sensor_loss"]) / float(summary["truth_input_sensor_loss"])
    )
    best_trial_row = min(trial_rows, key=lambda row: float(row["loss"]))
    summary["best_trial"] = int(best_trial_row["trial"])
    within_threshold = 1.10 * float(summary["truth_input_sensor_loss"])
    first_within = next(
        (row for row in trial_rows if float(row["loss"]) <= within_threshold),
        None,
    )
    summary["first_within_10pct_of_truth_input_trial"] = (
        int(first_within["trial"]) if first_within is not None else None
    )
    if args.include_gamma:
        summary["gamma_abs_error"] = abs(best_cond["gamma"] - truth_cond["gamma"])
    pca_projection = None
    response_trial_rows = None
    if args.include_gamma:
        reference_run = args.pca_reference_run
        if reference_run is None:
            pca_projection = _pca_projection(
                trial_rows,
                search_space,
                truth_cond,
                best_cond,
                initial,
            )
        else:
            reference_summary = json.loads(
                (reference_run / "summary.json").read_text(encoding="utf-8")
            )
            reference_meta = dict(reference_summary.get("pca_response_surface", {}) or {})
            if not reference_meta:
                raise ValueError(f"PCA reference metadata is missing: {reference_run}")
            with (reference_run / "trials.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                response_trial_rows = list(csv.DictReader(stream))
            pca_projection = _pca_projection_from_reference(
                response_trial_rows,
                search_space,
                truth_cond,
                best_cond,
                initial,
                reference_meta,
            )
        summary["pca_response_surface"] = {
            "variables": list(pca_projection["variables"]),
            "input_normalization": "min-max by configured search bounds",
            "mean_scaled": np.asarray(pca_projection["mean_scaled"]).tolist(),
            "components": np.asarray(pca_projection["components"]).tolist(),
            "explained_variance_ratio": np.asarray(
                pca_projection["explained_variance_ratio"]
            ).tolist(),
            "truth_score": np.asarray(pca_projection["truth_score"]).tolist(),
            "best_score": np.asarray(pca_projection["best_score"]).tolist(),
            "initial_score": np.asarray(pca_projection["initial_score"]).tolist(),
            "reference_run": str(reference_run) if reference_run is not None else None,
            "response_trials": len(
                response_trial_rows if response_trial_rows is not None else trial_rows
            ),
            "surface": "Gaussian-kernel smoothed log-loss surface across the full rectangular PC1-PC2 plotting range; bandwidth=0.18 in score-standardized PCA coordinates; visualization only; sparse-edge regions are extrapolated estimates",
        }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_summary(
        out_root,
        r_mm=1000.0 * r_m,
        truth_profile=truth_profile,
        observed=observed,
        initial_profile=initial_profile,
        best_profile=best_profile,
        trial_rows=trial_rows,
        response_trial_rows=response_trial_rows,
        truth_cond=truth_cond,
        best_cond=best_cond,
        optimizer_name=str(summary["optimizer"]),
        include_gamma=bool(args.include_gamma),
        pca_projection=pca_projection,
    )
    _write_index(out_root, summary)


if __name__ == "__main__":
    main()
