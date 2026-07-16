from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import yaml

from plasma_surrogate.core.dataset_io import load_dataset
from plasma_surrogate.core.input_modes import load_checkpoint_metadata_with_input_mode
from plasma_surrogate.core.run_bundle import RunBundleLoader
from plasma_surrogate.data.geometry_provider import build_geometry_provider
from plasma_surrogate.infer.engine import InferenceEngine
from plasma_surrogate.models.checkpoint import load_checkpoint


OUT_ROOT = Path("runs/gec_ccp_all_models_best_spatial_truth_pred_error")
PUB_DIR = "publication_by_field"
DEFAULT_SELECTION_MANIFEST = Path("runs/gec_ccp_trustworthy_v2/validation_selected/best_by_model.csv")
SPLIT_FILES = {"interp": "split_interp_marginal_v1.json", "extrap": "split_extrap_v1.json"}
SELECTED_CASES = (("best", 0.0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("worst", 1.0))
ERR_VMIN = -30.0
ERR_VMAX = 30.0
ERROR_ABS_LIMIT_BY_SPLIT = {"interp": 30.0, "extrap": 100.0}
FIELD_DISPLAY_NAMES = {
    "density_electron": r"$n_e$",
    "density_ion": r"$n_i$",
    "temperature_electron": r"$T_e$",
    "potential": r"$\phi$",
}
TARGET_COLOR = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
OUTSIDE_COLOR = np.asarray([0.90, 0.90, 0.90, 1.0], dtype=np.float32)
PART_COLORS = (
    np.asarray([0.38, 0.38, 0.38, 1.0], dtype=np.float32),
    np.asarray([0.55, 0.47, 0.36, 1.0], dtype=np.float32),
    np.asarray([0.70, 0.61, 0.45, 1.0], dtype=np.float32),
    np.asarray([0.45, 0.54, 0.62, 1.0], dtype=np.float32),
)
PART_EDGES = ("#111111", "#7a4f00", "#b36b00", "#244f73")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class PlotMasks:
    target: np.ndarray
    chamber_parts: np.ndarray
    non_target: np.ndarray
    part_ids: tuple[str, ...]
    part_masks: tuple[np.ndarray, ...]
    r_mm: np.ndarray | None
    z_mm: np.ndarray | None

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        if self.r_mm is None or self.z_mm is None:
            return None
        return (float(self.r_mm[0]), float(self.r_mm[-1]), float(self.z_mm[0]), float(self.z_mm[-1]))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return dict(json.load(f))


def _read_case_ids(path: Path) -> list[str]:
    return [str(v) for v in _read_json(path).get("test", [])]


def _output_vars(run_root: Path) -> list[str]:
    raw = _read_json(run_root / "preprocessing" / "schema" / "output_layout.json").get("vars", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"missing output_layout.vars under {run_root}")
    return [str(v) for v in raw]


def _target_metadata(run_root: Path) -> dict[str, dict[str, Any]]:
    path = run_root / "preprocessing" / "schema" / "target_role_schema.json"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in list(_read_json(path).get("targets", []) or []):
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = dict(item)
    return out


def _field_label(name: str, metadata: dict[str, dict[str, Any]]) -> str:
    role = str(metadata.get(name, {}).get("role", "")).strip()
    return FIELD_DISPLAY_NAMES.get(role) or name


def _field_units(name: str, metadata: dict[str, dict[str, Any]]) -> str:
    return str(metadata.get(name, {}).get("units", "")).strip()


def _display_units(units: str) -> str:
    text = str(units).strip()
    if text == "m^-3":
        return r"m$^{-3}$"
    return text


def _resolve_alternative_id(
    *,
    case: dict[str, Any] | None,
    manifest: dict[str, Any],
    alternative_ids: tuple[str, ...],
) -> str | None:
    if case is None:
        return None
    for key in ("base_name", "structure_id", "alternative_id", "geom_id"):
        value = str(case.get(key, "")).strip()
        if value in set(alternative_ids):
            return value
    cond = dict(case.get("cond", {}) or {})
    td_raw = cond.get("Td")
    if td_raw is not None:
        td = float(td_raw)
        alternatives = manifest.get("alternative_conditions", {})
        if isinstance(alternatives, dict):
            matches = [
                str(name)
                for name, values in alternatives.items()
                if str(name) in set(alternative_ids)
                and isinstance(values, dict)
                and values.get("Td") is not None
                and np.isclose(float(values["Td"]), td, rtol=0.0, atol=1.0e-12)
            ]
            if len(matches) == 1:
                return matches[0]
        source = manifest.get("source", {})
        source_index = source.get("structure_file_index_csv") if isinstance(source, dict) else None
        if source_index:
            source_path = Path(str(source_index))
            if source_path.exists():
                with source_path.open("r", encoding="utf-8", newline="") as f:
                    matches = [
                        str(row.get("base_name", "")).strip()
                        for row in csv.DictReader(f)
                        if str(row.get("base_name", "")).strip() in set(alternative_ids)
                        and str(row.get("td_value", "")).strip()
                        and np.isclose(float(row["td_value"]), td, rtol=0.0, atol=1.0e-12)
                    ]
                if len(matches) == 1:
                    return matches[0]
    return None


def _load_masks(dataset_root: Path, *, case: dict[str, Any] | None = None) -> PlotMasks:
    root = dataset_root / "geometry" if (dataset_root / "geometry").exists() else dataset_root
    plasma = np.asarray(np.load(root / "mask_plasma.npy"), dtype=np.float32) > 0.5
    valid_field = plasma.copy()
    parts = np.zeros_like(plasma, dtype=bool)
    part_ids: tuple[str, ...] = ()
    part_masks: tuple[np.ndarray, ...] = ()
    structure_path_raw = None if case is None else case.get("structure_npz")
    if structure_path_raw:
        structure_path = Path(str(structure_path_raw))
        if not structure_path.exists():
            raise FileNotFoundError(f"case structure npz not found: {structure_path}")
        with np.load(structure_path, allow_pickle=True) as structure:
            if "mask_plasma" not in structure.files or "part_mask_stack" not in structure.files:
                raise ValueError(f"case structure npz must include mask_plasma and part_mask_stack: {structure_path}")
            plasma = np.asarray(structure["mask_plasma"], dtype=np.float32) > 0.5
            valid_field = (
                np.asarray(structure["valid_field_mask"], dtype=np.float32) > 0.5
                if "valid_field_mask" in structure.files
                else plasma.copy()
            )
            stack = np.asarray(structure["part_mask_stack"], dtype=np.float32)
            if stack.ndim != 3 or tuple(stack.shape[1:]) != tuple(plasma.shape):
                raise ValueError(f"invalid case part_mask_stack shape={stack.shape}: {structure_path}")
            part_masks = tuple(np.asarray(mask > 0.5, dtype=bool) for mask in stack)
            ids = structure["part_ids"] if "part_ids" in structure.files else np.arange(len(part_masks))
            part_ids = tuple(str(v) for v in np.asarray(ids).reshape(-1).tolist())
            if len(part_ids) != len(part_masks):
                raise ValueError(f"case part_ids do not match part_mask_stack slots: {structure_path}")
            parts = np.any(np.stack(part_masks, axis=0), axis=0) if part_masks else parts
    else:
        pack_path = root / "parts_pack.npz"
        manifest_path = root / "parts_manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        if not pack_path.exists():
            stack = None
        else:
            stack = np.asarray([], dtype=np.float32)
        if pack_path.exists():
            with np.load(pack_path, allow_pickle=True) as pack:
                stack = np.asarray(pack["mask_stack"], dtype=np.float32)
                ids = pack["part_ids"] if "part_ids" in pack.files else np.arange(stack.shape[0])
                all_ids = tuple(str(v) for v in np.asarray(ids).reshape(-1).tolist())
            if stack.ndim != 3 or tuple(stack.shape[1:]) != tuple(plasma.shape):
                raise ValueError(f"invalid geometry parts_pack mask_stack shape={stack.shape}: {pack_path}")
            part_semantics = str(manifest.get("part_semantics", "simultaneous")).strip().lower()
            selected_id = _resolve_alternative_id(
                case=case,
                manifest=manifest,
                alternative_ids=all_ids,
            )
            if part_semantics == "alternatives" or selected_id is not None:
                if selected_id is None:
                    raise ValueError(
                        "alternative structure mask selection requires case base_name/geom_id or a unique Td mapping"
                    )
                selected_idx = all_ids.index(selected_id)
                part_masks = (np.asarray(stack[selected_idx] > 0.5, dtype=bool),)
                part_ids = (selected_id,)
            else:
                part_masks = tuple(np.asarray(mask > 0.5, dtype=bool) for mask in stack)
                part_ids = all_ids
            parts = np.any(np.stack(part_masks, axis=0), axis=0) if part_masks else parts
    r_mm = z_mm = None
    if (root / "r_coords.npy").exists() and (root / "z_coords.npy").exists():
        r = np.asarray(np.load(root / "r_coords.npy"), dtype=np.float64)
        z = np.asarray(np.load(root / "z_coords.npy"), dtype=np.float64)
        if len(r) == plasma.shape[1] and len(z) == plasma.shape[0]:
            r_mm = 1000.0 * r
            z_mm = 1000.0 * z
    # edg entity masks are boundary-line features, not solid pixels.  Keep the
    # simulation-valid plasma target intact and use the masks only as overlays.
    target = plasma & valid_field
    return PlotMasks(target, parts, ~target, part_ids, part_masks, r_mm, z_mm)


def _manifest_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"selection manifest field {field!r} must be boolean-compatible")


def _read_best_runs(selection_manifest: Path = DEFAULT_SELECTION_MANIFEST) -> dict[str, dict[str, Any]]:
    if not selection_manifest.exists():
        raise FileNotFoundError(
            "validation-selected v2 manifest not found; legacy v1/test-selected results are not an "
            f"implicit fallback: {selection_manifest}"
        )
    runs: dict[str, dict[str, Any]] = {}
    with selection_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {
            "model_id",
            "validation_selection_value",
            "validation_selection_mode",
            "validation_selection_reliable",
            "leaderboard",
            "run_root",
            "scaler_fit_split",
        }
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"validation-selected manifest missing required columns: {missing}")
        if not ({"selection_split", "split"} & fieldnames):
            raise ValueError("validation-selected manifest requires selection_split or split provenance")
        if not ({"protocol_variant", "protocol"} & fieldnames):
            raise ValueError("validation-selected manifest requires protocol_variant or protocol provenance")
        for row in reader:
            model = str(row["model_id"]).strip()
            run_root_raw = str(row["run_root"]).strip()
            leaderboard_raw = str(row["leaderboard"]).strip()
            if not model or not run_root_raw or not leaderboard_raw:
                raise ValueError("validation-selected manifest contains an empty model/run/leaderboard reference")
            run_root = Path(run_root_raw)
            leaderboard = Path(leaderboard_raw)
            if not _manifest_bool(row["validation_selection_reliable"], field="validation_selection_reliable"):
                raise ValueError(f"validation-selected manifest marks model={model!r} as unreliable")
            value = float(row["validation_selection_value"])
            if not np.isfinite(value):
                raise ValueError(f"validation selection value must be finite for model={model!r}")
            if model in runs:
                raise ValueError(f"validation-selected manifest contains duplicated model_id={model!r}")
            selection_split = str(row.get("selection_split", row.get("split", ""))).strip()
            protocol = str(row.get("protocol_variant", row.get("protocol", ""))).strip()
            scaler_fit_split = str(row["scaler_fit_split"]).strip()
            if not selection_split or not protocol or not scaler_fit_split:
                raise ValueError(
                    f"validation-selected manifest has empty split/protocol/scaler provenance for model={model!r}"
                )
            runs[model] = {
                "model": model,
                "family": str(row.get("family", "")).strip() or "validation_selected_v2",
                "seed": str(row.get("seed", "")).strip(),
                "run_root": run_root,
                "validation_selection_value": value,
                "recipe": str(row["validation_selection_mode"]).strip(),
                "leaderboard": leaderboard,
                "selection_split": selection_split,
                "protocol": protocol,
                "scaler_fit_split": scaler_fit_split,
                "selection_manifest": str(selection_manifest),
            }
    if not runs:
        raise ValueError(f"validation-selected manifest has no rows: {selection_manifest}")
    return runs


def _make_engine(run_root: Path, split: str, model_name: str, run_cfg: dict[str, Any], geometry_root: Path) -> InferenceEngine:
    ckpt = run_root / "models" / model_name / "eval_protocol" / split / "checkpoints"
    model = load_checkpoint(ckpt)
    checkpoint_meta, checkpoint_input_meta = load_checkpoint_metadata_with_input_mode(ckpt / "meta.json")
    bundle = RunBundleLoader.load(run_root, model=model)
    spatial_transform_artifacts = bundle.spatial_transform_artifacts_for_checkpoint(checkpoint_meta)
    benchmark_cfg = dict(run_cfg.get("benchmark", run_cfg) or {})
    train_cfg = dict(benchmark_cfg.get("train", run_cfg.get("train", {})))
    input_features_cfg = dict(dict(train_cfg.get(model_name, {})).get("input_features", {}))
    input_scaling_cfg = dict(input_features_cfg.get("scale_using_train_stats", {})) or dict(
        input_features_cfg.get("input_scaling", {})
    )
    inference_cfg = dict(benchmark_cfg.get("inference", run_cfg.get("inference", {})) or {})
    ood_cfg = dict(inference_cfg.get("ood", {}) or {})
    for key in ("qoi", "postprocess", "diagnostics", "derived_fields", "derived_fields_strict"):
        if key in inference_cfg:
            value = inference_cfg[key]
            ood_cfg[key] = dict(value or {}) if isinstance(value, dict) else value
    return InferenceEngine(
        model=model,
        cond_schema=bundle.cond_schema_obj(),
        axis_schema=bundle.axis_schema_obj(),
        geometry_provider=build_geometry_provider(
            geometry_root,
            provider_mode=str(checkpoint_input_meta.get("geometry_provider_mode_effective", "fixed")),
        ),
        output_dir=OUT_ROOT / "_inference_disabled",
        transform_bundle=bundle.transform_bundle_for_checkpoint(checkpoint_meta),
        cond_stats=dict(bundle.schemas.get("cond_stats", {})),
        phi_mode=str(run_cfg.get("benchmark", run_cfg).get("phi_mode", "direct")),
        phi_hybrid_steps=int(run_cfg.get("benchmark", run_cfg).get("phi_hybrid_steps", 1) or 1),
        poisson_refine_iters=int(inference_cfg.get("poisson_refine", {}).get("iters", 0) or 0),
        ood_cfg=ood_cfg,
        feature_store=bundle.geometry_store,
        coord_scaler=dict(bundle.transforms.get("coord_scaler", {})),
        coord_feature_scaler=spatial_transform_artifacts["coord_feature_scaler"],
        coord_feature_pack=bundle.schemas.get("coord_feature_pack"),
        coord_distance_transform_stats=spatial_transform_artifacts["distance_transform_stats"],
        coord_input_scaling_cfg=input_scaling_cfg,
        coord_input_features_cfg=input_features_cfg,
        grid_input_features_cfg=input_features_cfg,
        input_mode=str(checkpoint_input_meta.get("input_mode_effective", "table_plus_structure")),
        input_mode_meta=dict(checkpoint_input_meta),
        checkpoint_input_mode_meta=dict(checkpoint_input_meta),
        checkpoint_meta=dict(checkpoint_meta),
        target_role_schema=dict(bundle.schemas.get("target_role_schema", {}) or {}),
        deeponet_head=(getattr(model, "poisson_head", None) if model_name == "deeponet_plasma" else None)
        or (model if model_name == "deeponet_plasma" else None),
        structure_descriptor_pack=bundle.schemas.get("structure_descriptor_pack"),
        latent_feature_pack=bundle.schemas.get("latent_feature_pack"),
    )


def _predict_case(engine: InferenceEngine, case: dict[str, Any], output_vars: list[str]) -> dict[str, np.ndarray]:
    cond = {str(k): float(v) for k, v in dict(case["cond"]).items()}
    result = engine.single_run(
        cond=cond,
        geom={"geom_id": str(case.get("geom_id", "default"))},
        axis={"mode": "steady", "value": float(case.get("axis", 0.0))},
        save_outputs=False,
    )
    out: dict[str, np.ndarray] = {}
    for name in output_vars:
        arr = np.asarray(result.fields_phys[name], dtype=np.float64)
        while arr.ndim > 2 and 1 in arr.shape:
            arr = np.squeeze(arr)
        out[name] = arr
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    finite = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else float("nan")


def _rel_rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    rmse = _rmse(truth, pred)
    scale = float(np.sqrt(np.mean(np.asarray(truth, dtype=np.float64) ** 2)))
    return float(rmse / scale) if np.isfinite(rmse) and scale > 0.0 else float("nan")


def _percent_error(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    truth_arr = np.asarray(truth, dtype=np.float64)
    peak = np.max(np.abs(truth_arr[mask][np.isfinite(truth_arr[mask])]))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.full_like(truth_arr, np.nan, dtype=np.float64)
    return 100.0 * (np.asarray(pred, dtype=np.float64) - truth_arr) / float(peak)


def _limits(arrs: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrs])
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        center = float(np.mean(finite))
        spread = float(np.std(finite)) or 1.0
        lo, hi = center - spread, center + spread
    return float(lo), float(hi)


def _image_kwargs(masks: PlotMasks) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"origin": "lower"}
    if masks.extent is not None:
        kwargs.update({"extent": masks.extent, "aspect": "equal"})
    return kwargs


def _masked(arr: np.ndarray, masks: PlotMasks) -> np.ma.MaskedArray:
    return np.ma.array(np.asarray(arr, dtype=np.float64), mask=~masks.target)


def _part_color(i: int) -> np.ndarray:
    return PART_COLORS[i % len(PART_COLORS)]


def _part_edge(i: int) -> str:
    return PART_EDGES[i % len(PART_EDGES)]


def _contour_args(mask: np.ndarray, masks: PlotMasks) -> tuple[Any, ...]:
    if masks.r_mm is not None and masks.z_mm is not None:
        return masks.r_mm, masks.z_mm, np.asarray(mask, dtype=np.float32)
    return (np.asarray(mask, dtype=np.float32),)


def _draw_background(ax: Any, masks: PlotMasks) -> None:
    bg = np.ones((*masks.target.shape, 4), dtype=np.float32)
    bg[masks.non_target] = OUTSIDE_COLOR
    bg[masks.target] = TARGET_COLOR
    covered = np.zeros_like(masks.chamber_parts, dtype=bool)
    for i, part_mask in enumerate(masks.part_masks):
        current = part_mask & ~covered
        bg[current] = _part_color(i)
        covered |= part_mask
    if not masks.part_masks:
        bg[masks.chamber_parts] = _part_color(0)
    ax.imshow(bg, **_image_kwargs(masks))


def _draw_contours(ax: Any, masks: PlotMasks) -> None:
    ax.contour(*_contour_args(masks.target, masks), levels=[0.5], colors="#00d5ff", linewidths=0.55)
    for i, part_mask in enumerate(masks.part_masks):
        if np.any(part_mask):
            ax.contour(*_contour_args(part_mask, masks), levels=[0.5], colors=_part_edge(i), linewidths=0.80)


def _format_axis(ax: Any, *, y_label: bool) -> None:
    ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.7, labelleft=bool(y_label))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel(r"$r$ [mm]")
    if y_label:
        ax.set_ylabel(r"$z$ [mm]")
    else:
        ax.set_ylabel("")


def _legend_handles(masks: PlotMasks) -> list[Any]:
    handles: list[Any] = [
        mpatches.Patch(facecolor=TARGET_COLOR, edgecolor="#00d5ff", label="plasma target"),
        mpatches.Patch(facecolor=OUTSIDE_COLOR, edgecolor="#999999", label="masked non-target"),
    ]
    if masks.part_ids:
        handles.append(
            mpatches.Patch(
                facecolor=_part_color(0),
                edgecolor=_part_edge(0),
                label=f"case structure ({len(masks.part_ids)} parts)",
            )
        )
    return handles


def _format_field_cbar(cbar: Any, units: str) -> None:
    if units:
        cbar.set_label(_display_units(units))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    cbar.formatter = formatter
    cbar.update_ticks()


def _plot_field(
    *,
    model: str,
    split: str,
    label: str,
    case_id: str,
    field: str,
    metadata: dict[str, dict[str, Any]],
    truth: np.ndarray,
    pred: np.ndarray,
    masks: PlotMasks,
    rel_rmse: float,
    png_path: Path,
    pdf_path: Path,
) -> None:
    err = _percent_error(truth, pred, masks.target)
    error_abs_limit = float(ERROR_ABS_LIMIT_BY_SPLIT.get(split, ERR_VMAX))
    field_lo, field_hi = _limits([np.where(masks.target, truth, np.nan), np.where(masks.target, pred, np.nan)])
    field_cmap = plt.get_cmap("jet").copy()
    field_cmap.set_bad((0, 0, 0, 0))
    err_cmap = plt.get_cmap("coolwarm").copy()
    err_cmap.set_bad((0, 0, 0, 0))

    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.8), constrained_layout=True, sharex=True, sharey=True)
    for ax, title in zip(axes, ("Geometry / Mask", "Truth", "Prediction", "Signed error")):
        ax.set_title(title, pad=4)
    for i, ax in enumerate(axes):
        _draw_background(ax, masks)
        if i == 1:
            ax.imshow(_masked(truth, masks), **_image_kwargs(masks), cmap=field_cmap, vmin=field_lo, vmax=field_hi)
        elif i == 2:
            pred_im = ax.imshow(_masked(pred, masks), **_image_kwargs(masks), cmap=field_cmap, vmin=field_lo, vmax=field_hi)
        elif i == 3:
            err_im = ax.imshow(
                _masked(err, masks),
                **_image_kwargs(masks),
                cmap=err_cmap,
                vmin=-error_abs_limit,
                vmax=error_abs_limit,
            )
        _draw_contours(ax, masks)
        _format_axis(ax, y_label=(i == 0))
    field_cbar = fig.colorbar(pred_im, ax=axes[1:3], fraction=0.045, pad=0.025, shrink=0.75)
    _format_field_cbar(field_cbar, _field_units(field, metadata))
    err_cbar = fig.colorbar(err_im, ax=axes[3], fraction=0.045, pad=0.025, shrink=0.75)
    err_cbar.set_label(r"$(\hat{y}-y)/\max(|y|)$ [%]")
    err_cbar.set_ticks([-error_abs_limit, -0.5 * error_abs_limit, 0, 0.5 * error_abs_limit, error_abs_limit])
    legend = fig.legend(
        handles=_legend_handles(masks),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    title = fig.suptitle(
        f"{model.upper()} | {split} {label} | {_field_label(field, metadata)} | rel. RMSE={rel_rmse:.4g}",
        y=1.01,
        fontsize=8.5,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", bbox_extra_artists=(legend, title), pad_inches=0.12)
    fig.savefig(pdf_path, bbox_inches="tight", bbox_extra_artists=(legend, title), pad_inches=0.12)
    plt.close(fig)


def _select(records: list[dict[str, Any]]) -> dict[int, str]:
    ordered = sorted(enumerate(records), key=lambda x: float(x[1]["score_mean_rel_rmse"]))
    out: dict[int, str] = {}
    n = len(ordered)
    for label, q in SELECTED_CASES:
        out.setdefault(ordered[int(round(q * (n - 1)))][0], label)
    return out


def _finite_metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _field_summary_rows(
    summary_rows: list[dict[str, Any]],
    plot_rows: list[dict[str, Any]],
    *,
    split_files: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    requested_splits = SPLIT_FILES if split_files is None else split_files
    model_order = list(dict.fromkeys(str(row["model"]) for row in plot_rows))
    field_order = list(dict.fromkeys(str(row["field"]) for row in plot_rows))
    field_meta = {
        (str(row["model"]), str(row["field"])): (str(row["field_label"]), str(row["units"]))
        for row in plot_rows
    }
    out: list[dict[str, Any]] = []
    for model in model_order:
        for split in requested_splits:
            case_rows = [row for row in summary_rows if row["model"] == model and row["split"] == split]
            for field in field_order:
                rel = _finite_metric_values(case_rows, f"rel_rmse_{field}")
                signed = _finite_metric_values(case_rows, f"mean_signed_peak_percent_error_{field}")
                absolute = _finite_metric_values(case_rows, f"mean_abs_peak_percent_error_{field}")
                if not rel:
                    continue
                label, units = field_meta.get((model, field), (field, ""))
                out.append(
                    {
                        "model": model,
                        "split": split,
                        "field": field,
                        "field_label": label,
                        "units": units,
                        "n_cases": len(rel),
                        "mean_rel_rmse": float(np.mean(rel)),
                        "std_rel_rmse": float(np.std(rel, ddof=1)) if len(rel) > 1 else 0.0,
                        "median_rel_rmse": float(np.median(rel)),
                        "min_rel_rmse": float(np.min(rel)),
                        "max_rel_rmse": float(np.max(rel)),
                        "mean_signed_peak_percent_error": float(np.mean(signed)) if signed else float("nan"),
                        "mean_abs_peak_percent_error": float(np.mean(absolute)) if absolute else float("nan"),
                    }
                )
    return out


def _metric_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.4f}" if np.isfinite(number) else ""


def _write_model_summary_pages(
    *,
    runs: dict[str, dict[str, Any]],
    plot_rows: list[dict[str, Any]],
    field_summary_rows: list[dict[str, Any]],
    split_files: dict[str, str] | None = None,
) -> dict[str, Path]:
    requested_splits = SPLIT_FILES if split_files is None else split_files
    out_dir = OUT_ROOT / "model_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_order = {label: i for i, (label, _quantile) in enumerate(SELECTED_CASES)}
    pages: dict[str, Path] = {}
    for model, spec in runs.items():
        page_path = out_dir / f"{model}.md"
        pages[model] = page_path
        lines = [
            f"# {model}: spatial truth / prediction / error",
            "",
            "[← 全モデルの集約へ](../index.md)",
            "",
            f"- validation-only representative seed: `{spec.get('seed', '') or 'not recorded'}`",
            f"- run: `{str(spec['run_root']).replace(chr(92), '/')}`",
            "- 代表ケースは4物性の平均相対RMSEで best / p25 / median / p75 / worst を選択",
            "- 各図は Structure / Mask、Truth、Prediction、Signed error の順",
            "- Truth と Prediction は同じカラースケール",
            "",
        ]
        for split in requested_splits:
            split_name = "marginal補間" if split == "interp" else "PP0条件外挿"
            limit = ERROR_ABS_LIMIT_BY_SPLIT.get(split, ERR_VMAX)
            lines.extend(
                [
                    f"## {split}: {split_name}",
                    "",
                    f"Signed error の表示範囲は `±{limit:g}%`。飽和色はこの値以上の誤差を表します。",
                    "",
                    "| 物性 | ケース数 | mean rel.RMSE | SD | median | min | max |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            summaries = [
                row for row in field_summary_rows if row["model"] == model and row["split"] == split
            ]
            for row in summaries:
                lines.append(
                    "| `{field}` | {n} | {mean} | {std} | {median} | {minv} | {maxv} |".format(
                        field=row["field"],
                        n=row["n_cases"],
                        mean=_metric_text(row["mean_rel_rmse"]),
                        std=_metric_text(row["std_rel_rmse"]),
                        median=_metric_text(row["median_rel_rmse"]),
                        minv=_metric_text(row["min_rel_rmse"]),
                        maxv=_metric_text(row["max_rel_rmse"]),
                    )
                )
            lines.append("")
            split_plots = [row for row in plot_rows if row["model"] == model and row["split"] == split]
            for summary in summaries:
                field = str(summary["field"])
                field_plots = sorted(
                    [row for row in split_plots if row["field"] == field],
                    key=lambda row: label_order.get(str(row["case_label"]), 999),
                )
                lines.extend([f"### {field}", ""])
                median = next((row for row in field_plots if row["case_label"] == "median"), None)
                if median is not None:
                    png = "../" + str(median["png_path"])
                    lines.extend(
                        [
                            f"[![{model} {split} {field} median]({png})]({png})",
                            "",
                        ]
                    )
                lines.extend(
                    [
                        "| 代表位置 | case | rel.RMSE | 図 |",
                        "| --- | --- | ---: | --- |",
                    ]
                )
                for row in field_plots:
                    png = "../" + str(row["png_path"])
                    pdf = "../" + str(row["pdf_path"])
                    lines.append(
                        f"| {row['case_label']} | `{row['case_id']}` | {_metric_text(row['rel_rmse'])} | "
                        f"[PNG]({png}) / [PDF]({pdf}) |"
                    )
                lines.append("")
        page_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
        help=(
            "Validation-selected run manifest. The v2 validation manifest is the only default; "
            "a legacy source must be passed explicitly and satisfy the provenance schema."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to the module-level OUT_ROOT for backward compatibility.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_FILES),
        default=list(SPLIT_FILES),
        help="Evaluation splits to plot. Defaults to both interp and extrap.",
    )
    return parser.parse_args()


def _requested_split_files(splits: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Return the requested split mapping once, preserving CLI order."""

    requested = tuple(dict.fromkeys(str(split) for split in splits))
    if not requested:
        raise ValueError("at least one evaluation split is required")
    unknown = [split for split in requested if split not in SPLIT_FILES]
    if unknown:
        raise ValueError(f"unknown evaluation splits: {unknown}")
    return {split: SPLIT_FILES[split] for split in requested}


def _split_scope_lines(split_files: dict[str, str]) -> list[str]:
    ranges = ", ".join(
        f"{split} `±{ERROR_ABS_LIMIT_BY_SPLIT.get(split, ERR_VMAX):g}%`"
        for split in split_files
    )
    lines = [f"- 誤差色域: {ranges}"]
    if "extrap" in split_files:
        lines.append(
            "- `extrap`は未見構造ではなく、PP0=1 train / PP0=3 validation / "
            "PP0=5 testの条件外挿"
        )
    return lines


def main() -> None:
    global OUT_ROOT
    args = _parse_args()
    split_files = _requested_split_files(args.splits)
    if args.out_root is not None:
        OUT_ROOT = Path(args.out_root)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    runs = _read_best_runs(Path(args.selection_manifest))
    first_cfg = _read_yaml(Path(next(iter(runs.values()))["run_root"]) / "resolved_config.yaml")
    dataset = load_dataset(first_cfg, run_dir=Path("."))
    cases = {str(case["case_id"]): case for case in dataset.cases}
    mask_cache: dict[str, PlotMasks] = {}
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, str]] = []
    plot_rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for model, spec in runs.items():
        run_root = Path(spec["run_root"])
        try:
            cfg = _read_yaml(run_root / "resolved_config.yaml")
            output_vars = _output_vars(run_root)
            metadata = _target_metadata(run_root)
            for split, split_file in split_files.items():
                engine = _make_engine(run_root, split, model, cfg, Path(dataset.geometry_root))
                preds: dict[str, dict[str, np.ndarray]] = {}
                records: list[dict[str, Any]] = []
                for case_id in _read_case_ids(run_root / "preprocessing" / "split" / split_file):
                    case = cases[case_id]
                    masks = mask_cache.setdefault(
                        case_id,
                        _load_masks(Path(dataset.geometry_root), case=case),
                    )
                    pred = _predict_case(engine, case, output_vars)
                    preds[case_id] = pred
                    rels: list[float] = []
                    row: dict[str, Any] = {
                        "model": model,
                        "family": str(spec["family"]),
                        "split": split,
                        "case_id": case_id,
                        "selected_label": "",
                        "validation_selection_value": float(spec["validation_selection_value"]),
                        "recipe": str(spec["recipe"]),
                        "run_root": str(run_root).replace("\\", "/"),
                        "validation_selection_split": str(spec["selection_split"]),
                        "validation_protocol": str(spec["protocol"]),
                        "validation_scaler_fit_split": str(spec["scaler_fit_split"]),
                        "selection_manifest": str(spec["selection_manifest"]).replace("\\", "/"),
                        "leaderboard": str(spec["leaderboard"]).replace("\\", "/"),
                    }
                    for field in output_vars:
                        truth = np.asarray(case["y"][field], dtype=np.float64)
                        rel = _rel_rmse(truth[masks.target], pred[field][masks.target])
                        row[f"rel_rmse_{field}"] = rel
                        row[f"rmse_{field}"] = _rmse(truth[masks.target], pred[field][masks.target])
                        row[f"mean_signed_peak_percent_error_{field}"] = float(np.nanmean(_percent_error(truth, pred[field], masks.target)[masks.target]))
                        row[f"mean_abs_peak_percent_error_{field}"] = float(np.nanmean(np.abs(_percent_error(truth, pred[field], masks.target)[masks.target])))
                        if np.isfinite(rel):
                            rels.append(rel)
                    row["score_mean_rel_rmse"] = float(np.mean(rels)) if rels else float("nan")
                    records.append(row)
                for idx, label in _select(records).items():
                    row = records[idx]
                    row["selected_label"] = label
                    case_id = str(row["case_id"])
                    case = cases[case_id]
                    masks = mask_cache[case_id]
                    selected_rows.append(
                        {
                            "model": model,
                            "family": str(spec["family"]),
                            "split": split,
                            "case_label": label,
                            "case_id": case_id,
                            "score_mean_rel_rmse": f"{float(row['score_mean_rel_rmse']):.8g}",
                            "validation_selection_value": f"{float(spec['validation_selection_value']):.8g}",
                            "recipe": str(spec["recipe"]),
                            "run_root": str(run_root).replace("\\", "/"),
                            "validation_selection_split": str(spec["selection_split"]),
                            "validation_protocol": str(spec["protocol"]),
                            "validation_scaler_fit_split": str(spec["scaler_fit_split"]),
                            "selection_manifest": str(spec["selection_manifest"]).replace("\\", "/"),
                            "leaderboard": str(spec["leaderboard"]).replace("\\", "/"),
                        }
                    )
                    for field in output_vars:
                        rel = float(row[f"rel_rmse_{field}"])
                        safe_field = str(field).replace("/", "_").replace("\\", "_")
                        png_rel = Path(PUB_DIR) / model / safe_field / split / f"{label}_{case_id}_{safe_field}.png"
                        pdf_rel = png_rel.with_suffix(".pdf")
                        _plot_field(
                            model=model,
                            split=split,
                            label=label,
                            case_id=case_id,
                            field=field,
                            metadata=metadata,
                            truth=np.asarray(case["y"][field], dtype=np.float64),
                            pred=preds[case_id][field],
                            masks=masks,
                            rel_rmse=rel,
                            png_path=OUT_ROOT / png_rel,
                            pdf_path=OUT_ROOT / pdf_rel,
                        )
                        plot_rows.append(
                            {
                                "model": model,
                                "family": str(spec["family"]),
                                "split": split,
                                "case_label": label,
                                "case_id": case_id,
                                "field": field,
                                "field_label": _field_label(field, metadata),
                                "units": _field_units(field, metadata),
                                "rel_rmse": f"{rel:.8g}" if np.isfinite(rel) else "",
                                "png_path": str(png_rel).replace("\\", "/"),
                                "pdf_path": str(pdf_rel).replace("\\", "/"),
                                "run_root": str(run_root).replace("\\", "/"),
                                "validation_selection_split": str(spec["selection_split"]),
                                "validation_protocol": str(spec["protocol"]),
                                "validation_scaler_fit_split": str(spec["scaler_fit_split"]),
                                "selection_manifest": str(spec["selection_manifest"]).replace("\\", "/"),
                                "leaderboard": str(spec["leaderboard"]).replace("\\", "/"),
                            }
                        )
                summary_rows.extend(records)
        except Exception as exc:
            skipped.append({"model": model, "family": str(spec.get("family", "")), "run_root": str(run_root), "reason": str(exc)})

    metric_cols = sorted({k for row in summary_rows for k in row if k.startswith(("rmse_", "rel_rmse_", "mean_"))})
    with (OUT_ROOT / "spatial_plot_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "model", "family", "split", "case_id", "selected_label", "score_mean_rel_rmse",
            "validation_selection_value", "recipe", "run_root", "validation_selection_split", "validation_protocol",
            "validation_scaler_fit_split",
            "selection_manifest", "leaderboard", *metric_cols,
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (OUT_ROOT / "selected_spatial_plots.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "model", "family", "split", "case_label", "case_id", "score_mean_rel_rmse",
            "validation_selection_value", "recipe", "run_root", "validation_selection_split", "validation_protocol",
            "validation_scaler_fit_split",
            "selection_manifest", "leaderboard",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)
    with (OUT_ROOT / "publication_field_plots.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "model", "family", "split", "case_label", "case_id", "field", "field_label", "units",
            "rel_rmse", "png_path", "pdf_path", "run_root", "validation_selection_split",
            "validation_protocol", "validation_scaler_fit_split", "selection_manifest", "leaderboard",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plot_rows)
    with (OUT_ROOT / "skipped_models.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "family", "run_root", "reason"])
        writer.writeheader()
        writer.writerows(skipped)

    field_summaries = _field_summary_rows(
        summary_rows,
        plot_rows,
        split_files=split_files,
    )
    field_summary_fields = [
        "model",
        "split",
        "field",
        "field_label",
        "units",
        "n_cases",
        "mean_rel_rmse",
        "std_rel_rmse",
        "median_rel_rmse",
        "min_rel_rmse",
        "max_rel_rmse",
        "mean_signed_peak_percent_error",
        "mean_abs_peak_percent_error",
    ]
    with (OUT_ROOT / "model_field_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_summary_fields)
        writer.writeheader()
        writer.writerows(field_summaries)

    model_pages = _write_model_summary_pages(
        runs=runs,
        plot_rows=plot_rows,
        field_summary_rows=field_summaries,
        split_files=split_files,
    )
    field_order = list(dict.fromkeys(str(row["field"]) for row in plot_rows))
    summary_by_key = {
        (str(row["model"]), str(row["split"]), str(row["field"])): row for row in field_summaries
    }
    lines = [
        "# GEC-CCP n78: Truth / Prediction / Spatial error",
        "",
        "Validation-onlyで選んだ各モデルについて、物性ごとの真値・予測値・空間誤差をまとめたページです。",
        "",
        "- 対象物性: `ne`, `ni`, `Te`, `phi`",
        "- 図の並び: Structure / Mask、Truth、Prediction、Signed error",
        "- Truth / Predictionはケース内で同一カラースケール",
        "- Signed error: `100 * (prediction - truth) / max(|truth|)`（plasma target内）",
        "- 代表ケース: 4物性平均rel.RMSEのbest / p25 / median / p75 / worst",
    ]
    lines.extend(_split_scope_lines(split_files))
    lines.extend(
        [
            "",
            "## 全テストケースの物性別平均相対RMSE",
            "",
            "値はvalidation-onlyで選んだ代表seedの全テストケースから計算しています。3 seed平均ではありません。",
            "",
            "| モデル | split | cases | "
            + " | ".join(f"`{field}`" for field in field_order)
            + " | 4物性平均 |",
            "| --- | --- | ---: | "
            + " | ".join("---:" for _ in field_order)
            + " | ---: |",
        ]
    )
    for model in runs:
        for split in split_files:
            rows = [summary_by_key[(model, split, field)] for field in field_order]
            means = [float(row["mean_rel_rmse"]) for row in rows]
            lines.append(
                f"| `{model}` | `{split}` | {rows[0]['n_cases']} | "
                + " | ".join(_metric_text(value) for value in means)
                + f" | {_metric_text(np.mean(means))} |"
            )
    lines.extend(
        [
            "",
            "## モデル別ページ",
            "",
        ]
    )
    for model, spec in runs.items():
        page_rel = model_pages[model].relative_to(OUT_ROOT).as_posix()
        lines.append(
            f"- [{model}]({page_rel}) — representative seed `{spec.get('seed', '') or 'not recorded'}`; "
            f"{' / '.join(split_files)} × 4物性 × 5代表ケース"
        )
    lines.extend(
        [
            "",
            "## 完全性",
            "",
            f"- requested / plotted models: `{len(runs)}` / `{len({row['model'] for row in plot_rows})}`",
            f"- all-case metric rows: `{len(summary_rows)}`",
            f"- selected model/split/case rows: `{len(selected_rows)}`",
            f"- PNG/PDF pairs: `{len(plot_rows)}`",
            f"- skipped models: `{len(skipped)}`",
            "",
            "## CSV",
            "",
            "- [model_field_summary.csv](model_field_summary.csv): モデル×split×物性の全ケース集約",
            f"- [publication_field_plots.csv](publication_field_plots.csv): 全{len(plot_rows)}代表図のパスとrel.RMSE",
            "- [selected_spatial_plots.csv](selected_spatial_plots.csv): 代表ケース選択",
            f"- [spatial_plot_summary.csv](spatial_plot_summary.csv): 全{len(summary_rows)}ケースの物性別誤差",
            "- [skipped_models.csv](skipped_models.csv)",
        ]
    )
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for row in skipped:
            lines.append(f"- `{row['model']}`: {row['reason']}")
    lines.extend(
        [
            "",
            "## 解釈上の注意",
            "",
            "- 各モデルのmedianケースは誤差順位から個別に選ぶため、モデル間で同一caseとは限りません。",
            "- 空間図はvalidation履歴だけで選んだ代表seedであり、テスト指標によるseed選択はしていません。",
            "- `Td`と構造IDが1対1対応するため、本図だけから構造入力の因果効果や未見構造汎化は評価できません。",
        ]
    )
    (OUT_ROOT / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_ROOT / "index.md")
    print(f"models={len(runs)} plotted={len({row['model'] for row in plot_rows})} skipped={len(skipped)} publication_pairs={len(plot_rows)}")


if __name__ == "__main__":
    main()
