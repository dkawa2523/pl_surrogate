"""Create conference-ready dataset-quality figures for GEC-CCP and GEC-ICP.

The script audits the processed datasets at execution time.  It intentionally
keeps model-performance metrics out of these figures: the panels describe
simulation success, design-space coverage, processed-pack integrity, finite
values, and split integrity only.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CCP_SOURCE_ROOT = Path("data/outputs_merged_td_all_success_pa_ext0520")
DEFAULT_CCP_DATASET_ROOT = Path("data/outputs_merged_td_csv_periodic_ext0520_v2")
DEFAULT_ICP_DATASET_ROOT = Path("data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2")
DEFAULT_ICP_AUDIT_JSON = Path("reports/icp_stage4_major_fixes/dataset_audit_part_lite_v2.json")
DEFAULT_ICP_SOURCE_SUMMARY = Path("data/outputs_icp_stage4_enriched_360/learning_summary.json")
DEFAULT_OUT_DIR = Path("reports/gec_conference_materials")

TARGETS = ("ne", "ni", "Te", "phi")
SPLIT_ORDER = ("train", "val", "test")

# Okabe-Ito-derived colors with light neutral companions.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
DARK = "#1F2937"
MID = "#64748B"
LIGHT = "#E5E7EB"
PALE = "#F8FAFC"
WHITE = "#FFFFFF"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")) or {})


def _truthy(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _as_float(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite {key} in row: {row}")
    return value


def _sorted_unique(rows: Sequence[Mapping[str, str]], key: str) -> list[float]:
    return sorted({_as_float(row, key) for row in rows})


def _is_binary(array: np.ndarray, *, atol: float = 1.0e-6) -> bool:
    values = np.asarray(array, dtype=np.float32)
    return bool(
        np.all(np.isfinite(values))
        and np.all((np.abs(values) <= atol) | (np.abs(values - 1.0) <= atol))
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return _relative_or_absolute(value)
    return value


def _format_decimal(value: float, *, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.edgecolor": DARK,
            "axes.linewidth": 0.8,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _draw_title(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_axis_off()
    ax.text(0.0, 0.76, title, ha="left", va="center", fontsize=19, weight="bold", color=DARK)
    ax.text(0.0, 0.20, subtitle, ha="left", va="center", fontsize=10.5, color=MID)


def _draw_card(ax: plt.Axes, value: str, label: str, *, accent: str) -> None:
    ax.set_axis_off()
    box = FancyBboxPatch(
        (0.01, 0.05),
        0.98,
        0.90,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.0,
        edgecolor=LIGHT,
        facecolor=WHITE,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.add_patch(
        Rectangle(
            (0.01, 0.05),
            0.018,
            0.90,
            transform=ax.transAxes,
            facecolor=accent,
            edgecolor="none",
        )
    )
    ax.text(0.09, 0.61, value, ha="left", va="center", fontsize=18, weight="bold", color=DARK)
    ax.text(0.09, 0.28, label, ha="left", va="center", fontsize=9.5, color=MID)


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {
            "facecolor": WHITE,
            "edgecolor": WHITE,
            "bbox_inches": "tight",
            "pad_inches": 0.08,
        }
        if suffix == "png":
            kwargs["dpi"] = int(dpi)
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _write_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _audit_ccp(source_root: Path, dataset_root: Path) -> dict[str, Any]:
    source_csv = source_root / "conditions_with_structure.csv"
    index_csv = dataset_root / "index.csv"
    source_rows = _read_csv(source_csv)
    rows = _read_csv(index_csv)
    if not rows:
        raise ValueError(f"CCP index is empty: {index_csv}")

    case_ids = [str(row.get("case_id", "")).strip() for row in rows]
    source_success = sum(
        str(row.get("status", "")).strip().lower() == "success"
        and _truthy(row.get("outputs_ok", ""))
        for row in source_rows
    )

    td_values = _sorted_unique(rows, "Td")
    pp0_values = _sorted_unique(rows, "PP0")
    pa_values = _sorted_unique(rows, "PA")
    gamma_values = _sorted_unique(rows, "gamma")
    observed: dict[tuple[float, float, float], set[float]] = {}
    observed_full: set[tuple[float, float, float, float]] = set()
    for row in rows:
        td = _as_float(row, "Td")
        pp0 = _as_float(row, "PP0")
        pa = _as_float(row, "PA")
        gamma = _as_float(row, "gamma")
        observed.setdefault((td, pp0, pa), set()).add(gamma)
        observed_full.add((td, pp0, pa, gamma))

    coverage_by_td: dict[str, list[list[int]]] = {}
    for td in td_values:
        matrix = [
            [len(observed.get((td, pp0, pa), set())) for pa in pa_values]
            for pp0 in pp0_values
        ]
        coverage_by_td[_format_decimal(td)] = matrix
    missing_combinations = [
        {"Td": td, "PP0": pp0, "PA": pa, "gamma": gamma}
        for td in td_values
        for pp0 in pp0_values
        for pa in pa_values
        for gamma in gamma_values
        if (td, pp0, pa, gamma) not in observed_full
    ]

    unique_structure_paths = sorted(
        {str(row.get("structure_npz", "")).strip() for row in rows if str(row.get("structure_npz", "")).strip()}
    )
    structure_masks: dict[str, np.ndarray] = {}
    structure_pack_valid = 0
    structure_pack_details: dict[str, dict[str, Any]] = {}
    expected_shape: tuple[int, int] | None = None
    for relative in unique_structure_paths:
        path = dataset_root / relative
        detail: dict[str, Any] = {"exists": path.is_file(), "valid": False}
        if not path.is_file():
            structure_pack_details[relative] = detail
            continue
        with np.load(path, allow_pickle=True) as data:
            required = {
                "mask_plasma",
                "mask_coil",
                "valid_field_mask",
                "outside_mask",
                "part_mask_stack",
                "r_coords",
                "z_coords",
            }
            keys_ok = required.issubset(data.files)
            if not keys_ok:
                detail["missing_keys"] = sorted(required - set(data.files))
                structure_pack_details[relative] = detail
                continue
            plasma = np.asarray(data["mask_plasma"], dtype=np.float32)
            coil = np.asarray(data["mask_coil"], dtype=np.float32)
            valid = np.asarray(data["valid_field_mask"], dtype=np.float32)
            outside = np.asarray(data["outside_mask"], dtype=np.float32)
            stack = np.asarray(data["part_mask_stack"], dtype=np.float32)
            r_coords = np.asarray(data["r_coords"], dtype=np.float32).reshape(-1)
            z_coords = np.asarray(data["z_coords"], dtype=np.float32).reshape(-1)
            shape = tuple(int(value) for value in plasma.shape)
            if expected_shape is None:
                expected_shape = shape
            shapes_ok = (
                plasma.ndim == 2
                and coil.shape == plasma.shape
                and valid.shape == plasma.shape
                and outside.shape == plasma.shape
                and stack.ndim == 3
                and tuple(stack.shape[1:]) == plasma.shape
                and r_coords.size == plasma.shape[1]
                and z_coords.size == plasma.shape[0]
                and shape == expected_shape
            )
            masks_binary = all(_is_binary(array) for array in (plasma, coil, valid, outside, stack))
            complement_ok = bool(np.allclose(plasma + outside, 1.0, rtol=0.0, atol=1.0e-6))
            coords_ok = bool(
                np.all(np.isfinite(r_coords))
                and np.all(np.isfinite(z_coords))
                and np.all(np.diff(r_coords) > 0.0)
                and np.all(np.diff(z_coords) > 0.0)
            )
            pack_valid = bool(shapes_ok and masks_binary and complement_ok and coords_ok)
            if pack_valid:
                structure_pack_valid += 1
                structure_masks[relative] = plasma > 0.5
            detail.update(
                {
                    "valid": pack_valid,
                    "shape": list(shape),
                    "part_slots": int(stack.shape[0]),
                    "masks_binary": masks_binary,
                    "plasma_outside_complement": complement_ok,
                    "coordinates_monotonic": coords_ok,
                }
            )
        structure_pack_details[relative] = detail

    total_values = 0
    finite_values = 0
    field_pack_valid = 0
    field_outside_mask_valid = 0
    field_shapes: set[tuple[int, int]] = set()
    missing_field_files: list[str] = []
    for row in rows:
        relative = str(row.get("fields_npz", "")).strip()
        path = dataset_root / relative
        if not path.is_file():
            missing_field_files.append(relative)
            continue
        with np.load(path, allow_pickle=False) as data:
            if not set(TARGETS).issubset(data.files):
                continue
            arrays = [np.asarray(data[name], dtype=np.float32) for name in TARGETS]
            shapes = {tuple(int(value) for value in array.shape) for array in arrays}
            if len(shapes) != 1:
                continue
            shape = next(iter(shapes))
            field_shapes.add(shape)
            finite_here = sum(int(np.count_nonzero(np.isfinite(array))) for array in arrays)
            total_here = sum(int(array.size) for array in arrays)
            finite_values += finite_here
            total_values += total_here
            if finite_here == total_here and (expected_shape is None or shape == expected_shape):
                field_pack_valid += 1
            structure_relative = str(row.get("structure_npz", "")).strip()
            plasma = structure_masks.get(structure_relative)
            if plasma is not None and plasma.shape == shape:
                outside_is_zero = all(
                    bool(np.allclose(array[~plasma], 0.0, rtol=0.0, atol=0.0))
                    for array in arrays
                )
                if outside_is_zero:
                    field_outside_mask_valid += 1

    finite_rate = float(finite_values / total_values) if total_values else 0.0
    base_counts: dict[str, int] = {}
    for row in rows:
        name = str(row.get("base_name", "")).strip()
        base_counts[name] = base_counts.get(name, 0) + 1

    return {
        "dataset": "GEC-CCP",
        "source_csv": source_csv,
        "index_csv": index_csv,
        "source_cases": len(source_rows),
        "source_success_cases": int(source_success),
        "processed_cases": len(rows),
        "unique_case_ids": len(set(case_ids)),
        "duplicate_case_ids": len(case_ids) - len(set(case_ids)),
        "td_values": td_values,
        "pp0_values": pp0_values,
        "pa_values": pa_values,
        "gamma_values": gamma_values,
        "coverage_by_td": coverage_by_td,
        "expected_factorial_cases": int(len(td_values) * len(pp0_values) * len(pa_values) * len(gamma_values)),
        "missing_case_count": len(missing_combinations),
        "missing_combinations": missing_combinations,
        "base_case_counts": base_counts,
        "field_pack_valid_cases": int(field_pack_valid),
        "field_pack_files_present": int(len(rows) - len(missing_field_files)),
        "missing_field_files": missing_field_files,
        "field_shapes": [list(shape) for shape in sorted(field_shapes)],
        "finite_values": int(finite_values),
        "total_values": int(total_values),
        "finite_rate": finite_rate,
        "field_outside_mask_valid_cases": int(field_outside_mask_valid),
        "structure_pack_count": len(unique_structure_paths),
        "structure_pack_valid_count": int(structure_pack_valid),
        "structure_pack_details": structure_pack_details,
    }


def _scan_field_finiteness(dataset_root: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    finite = 0
    total = 0
    valid_packs = 0
    for row in rows:
        relative = str(row.get("fields_npz", "")).strip()
        path = dataset_root / relative
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            if not set(TARGETS).issubset(data.files):
                continue
            arrays = [np.asarray(data[name]) for name in TARGETS]
            pack_finite = True
            for array in arrays:
                count = int(np.count_nonzero(np.isfinite(array)))
                finite += count
                total += int(array.size)
                pack_finite = pack_finite and count == int(array.size)
            if pack_finite:
                valid_packs += 1
    return {
        "finite_rate": float(finite / total) if total else 0.0,
        "finite_values": finite,
        "total_values": total,
        "finite_field_packs": valid_packs,
        "source": "direct_scan",
    }


def _audit_icp(
    dataset_root: Path,
    audit_json: Path,
    source_summary_json: Path,
) -> dict[str, Any]:
    index_csv = dataset_root / "index.csv"
    rows = _read_csv(index_csv)
    if not rows:
        raise ValueError(f"ICP index is empty: {index_csv}")
    audit = _read_json(audit_json)
    source_summary = _read_json(source_summary_json)

    source_manifest_raw = str(audit.get("source_manifest", "")).strip()
    source_manifest_path = (
        _resolve(source_manifest_raw)
        if source_manifest_raw
        else source_summary_json.parent / "learning_manifest.csv"
    )
    source_split_by_group: dict[str, set[str]] = {}
    if source_manifest_path.is_file():
        for source_row in _read_csv(source_manifest_path):
            group = str(
                source_row.get("case_group_id")
                or source_row.get("split_group")
                or source_row.get("base_case_id")
                or ""
            ).strip()
            split = str(source_row.get("split") or source_row.get("source_split") or "").strip().lower()
            if group and split:
                source_split_by_group.setdefault(group, set()).add(split)

    case_ids = [str(row.get("case_id", "")).strip() for row in rows]
    group_rows: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        group = str(row.get("split_group") or row.get("base_case_id") or "").strip()
        if not group:
            raise ValueError("ICP index row has no split_group/base_case_id")
        group_rows.setdefault(group, []).append(row)

    leakage_groups: list[str] = []
    unassigned_split_groups: list[str] = []
    inconsistent_coil_groups: list[str] = []
    group_records: list[tuple[str, str, int, int]] = []
    for group, members in sorted(group_rows.items()):
        splits = {str(row.get("source_split", "")).strip().lower() for row in members}
        splits.discard("")
        if not splits:
            splits = set(source_split_by_group.get(group, set()))
        coils = {int(round(_as_float(row, "nncoil"))) for row in members}
        if len(splits) > 1:
            leakage_groups.append(group)
        if not splits:
            unassigned_split_groups.append(group)
        if len(coils) != 1:
            inconsistent_coil_groups.append(group)
        split = next(iter(splits), "unknown")
        coil = next(iter(coils), -1)
        group_records.append((group, split, coil, len(members)))

    ncoil_values = sorted({int(round(_as_float(row, "nncoil"))) for row in rows})
    case_counts_by_ncoil = {
        str(coil): sum(int(round(_as_float(row, "nncoil"))) == coil for row in rows)
        for coil in ncoil_values
    }
    group_counts_by_split_ncoil = {
        split: {
            str(coil): sum(
                record_split == split and record_coil == coil
                for _, record_split, record_coil, _ in group_records
            )
            for coil in ncoil_values
        }
        for split in SPLIT_ORDER
    }
    split_case_counts = {
        split: sum(member_count for _, record_split, _, member_count in group_records if record_split == split)
        for split in SPLIT_ORDER
    }

    field_files_present = sum(
        (dataset_root / str(row.get("fields_npz", "")).strip()).is_file()
        for row in rows
    )
    structure_files_present = sum(
        (dataset_root / str(row.get("structure_npz", "")).strip()).is_file()
        for row in rows
    )

    audit_root_matches = False
    if audit.get("dataset_root"):
        audit_root_matches = _resolve(str(audit["dataset_root"])) == dataset_root.resolve()
    field_stats = dict(audit.get("field_stats", {}) or {}) if audit_root_matches else {}
    audit_has_all_targets = set(TARGETS).issubset(field_stats)
    audit_sampled_all_cases = audit_has_all_targets and all(
        int(dict(field_stats[name]).get("sampled_cases", 0)) == len(rows) for name in TARGETS
    )
    if audit_sampled_all_cases:
        finite_rate = min(float(dict(field_stats[name]).get("finite_rate", 0.0)) for name in TARGETS)
        finite_info = {
            "finite_rate": finite_rate,
            "field_stats": field_stats,
            "source": _relative_or_absolute(audit_json),
        }
    else:
        finite_info = _scan_field_finiteness(dataset_root, rows)
        finite_rate = float(finite_info["finite_rate"])

    structure_audit = dict(audit.get("structure", {}) or {}) if audit_root_matches else {}
    part_stack_cases = int(structure_audit.get("part_mask_stack_cases", 0))
    part_lite_ready = bool(structure_audit.get("part_lite_v1_ready", False))
    source_counts = dict(source_summary.get("run_summary_counts", {}) or {})
    source_success_cases = int(source_counts.get("success", source_summary.get("rows", 0)))
    source_failed_cases = int(source_counts.get("failed", 0))

    test_missing_ncoil = [
        coil
        for coil in ncoil_values
        if group_counts_by_split_ncoil.get("test", {}).get(str(coil), 0) == 0
    ]
    rows_per_group = sorted({len(members) for members in group_rows.values()})

    return {
        "dataset": "GEC-ICP",
        "index_csv": index_csv,
        "audit_json": audit_json,
        "source_summary_json": source_summary_json,
        "source_manifest": source_manifest_path,
        "processed_cases": len(rows),
        "unique_case_ids": len(set(case_ids)),
        "duplicate_case_ids": len(case_ids) - len(set(case_ids)),
        "structure_groups": len(group_rows),
        "rows_per_structure_group": rows_per_group,
        "source_success_cases": source_success_cases,
        "source_failed_cases": source_failed_cases,
        "ncoil_values": ncoil_values,
        "case_counts_by_ncoil": case_counts_by_ncoil,
        "group_counts_by_split_ncoil": group_counts_by_split_ncoil,
        "split_case_counts": split_case_counts,
        "field_files_present": int(field_files_present),
        "structure_files_present": int(structure_files_present),
        "finite_rate": finite_rate,
        "finite_audit": finite_info,
        "part_mask_stack_cases": part_stack_cases,
        "part_lite_v1_ready": part_lite_ready,
        "cross_split_leakage_groups": leakage_groups,
        "cross_split_leakage_count": len(leakage_groups),
        "unassigned_split_groups": unassigned_split_groups,
        "inconsistent_coil_groups": inconsistent_coil_groups,
        "test_missing_ncoil": test_missing_ncoil,
        "audit_root_matches_dataset": audit_root_matches,
    }


def _plot_ccp(metrics: Mapping[str, Any], out_dir: Path, *, dpi: int) -> tuple[list[Path], Path]:
    td_values = [float(value) for value in metrics["td_values"]]
    pp0_values = [float(value) for value in metrics["pp0_values"]]
    pa_values = [float(value) for value in metrics["pa_values"]]
    gamma_levels = len(metrics["gamma_values"])
    coverage_by_td = dict(metrics["coverage_by_td"])

    fig = plt.figure(figsize=(13.2, 7.7))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.08)
    grid = fig.add_gridspec(3, 12, height_ratios=(0.62, 1.18, 3.55), hspace=0.48, wspace=0.42)
    title_ax = fig.add_subplot(grid[0, :])
    _draw_title(
        title_ax,
        "GEC-CCP dataset coverage and integrity",
        (
            f"{metrics['processed_cases']} processed cases | {len(td_values)} geometries | "
            f"{gamma_levels} gamma levels per sampled PP0 x PA cell | "
            f"{metrics['missing_case_count']} missing cases shown explicitly"
        ),
    )

    cards = [
        (
            f"{metrics['source_success_cases']} / {metrics['source_cases']}",
            "Successful COMSOL exports",
            GREEN,
        ),
        (
            f"{metrics['field_pack_valid_cases']} / {metrics['processed_cases']}",
            "Valid processed field packs",
            BLUE,
        ),
        (
            f"{100.0 * float(metrics['finite_rate']):.1f}%",
            "Finite target values",
            SKY,
        ),
        (
            f"{metrics['structure_pack_valid_count']} / {metrics['structure_pack_count']}",
            "Valid structure and mask packs",
            ORANGE,
        ),
    ]
    for index, (value, label, accent) in enumerate(cards):
        _draw_card(fig.add_subplot(grid[1, index * 3 : (index + 1) * 3]), value, label, accent=accent)

    coverage_colors = ("#F3C46B", "#D9EAF3", "#88C4DF", BLUE)
    for td_index, td in enumerate(td_values):
        ax = fig.add_subplot(grid[2, td_index * 4 : (td_index + 1) * 4])
        key = _format_decimal(td)
        matrix = np.asarray(coverage_by_td[key], dtype=np.int64)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                count = int(matrix[row_index, col_index])
                color_index = min(max(count, 0), len(coverage_colors) - 1)
                ax.add_patch(
                    Rectangle(
                        (col_index - 0.5, row_index - 0.5),
                        1.0,
                        1.0,
                        facecolor=coverage_colors[color_index],
                        edgecolor=WHITE,
                        linewidth=0.8,
                    )
                )
                if count == 0:
                    label = f"0\n(-{gamma_levels} cases)"
                    color = DARK
                    weight = "bold"
                else:
                    label = str(count)
                    color = WHITE if count >= max(gamma_levels - 1, 1) else DARK
                    weight = "bold"
                ax.text(col_index, row_index, label, ha="center", va="center", color=color, weight=weight, fontsize=11)
        ax.set_title(rf"$T_d={td:.2f}$", pad=10, weight="bold")
        ax.set_xticks(np.arange(len(pa_values)), [_format_decimal(value) for value in pa_values])
        ax.set_yticks(np.arange(len(pp0_values)), [f"{value:g}" for value in pp0_values])
        ax.set_xlabel("PA")
        ax.set_ylabel("PP0" if td_index == 0 else "")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_color(LIGHT)
        ax.set_xlim(-0.5, len(pa_values) - 0.5)
        ax.set_ylim(len(pp0_values) - 0.5, -0.5)
        ax.set_aspect("equal", adjustable="box")

    stem = "gec_ccp_dataset_quality"
    output_files = _save_figure(fig, out_dir, stem, dpi=dpi)
    metadata_path = out_dir / f"{stem}_metadata.json"
    metadata = {
        "figure": "Conference-ready GEC-CCP dataset coverage and integrity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": metrics,
        "visual_encoding": {
            "coverage_cells": "number of unique gamma values available for each Td / PP0 / PA cell",
            "missing_cell_color": ORANGE,
            "complete_cell_color": BLUE,
        },
        "output_files": [_relative_or_absolute(path) for path in output_files],
    }
    _write_metadata(metadata_path, metadata)
    return output_files, metadata_path


def _plot_icp(metrics: Mapping[str, Any], out_dir: Path, *, dpi: int) -> tuple[list[Path], Path]:
    ncoil_values = [int(value) for value in metrics["ncoil_values"]]
    case_counts = [int(metrics["case_counts_by_ncoil"][str(value)]) for value in ncoil_values]
    split_matrix = np.asarray(
        [
            [int(metrics["group_counts_by_split_ncoil"][split][str(value)]) for value in ncoil_values]
            for split in SPLIT_ORDER
        ],
        dtype=np.int64,
    )
    test_missing = [int(value) for value in metrics["test_missing_ncoil"]]

    fig = plt.figure(figsize=(13.2, 7.7))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.08)
    grid = fig.add_gridspec(3, 12, height_ratios=(0.62, 1.18, 3.55), hspace=0.48, wspace=0.48)
    title_ax = fig.add_subplot(grid[0, :])
    missing_text = ", ".join(str(value) for value in test_missing) if test_missing else "none"
    _draw_title(
        title_ax,
        "GEC-ICP dataset coverage and integrity",
        (
            f"{metrics['processed_cases']} cases = {metrics['structure_groups']} coil structures x "
            f"{metrics['rows_per_structure_group'][0]} process settings | disjoint structure splits | "
            f"test coverage missing Ncoil={missing_text}"
        ),
    )

    cards = [
        (
            f"{metrics['source_success_cases']} / {metrics['processed_cases']}",
            "Successful COMSOL simulations",
            GREEN,
        ),
        (
            f"{metrics['field_files_present']} / {metrics['processed_cases']}",
            "Processed field packs present",
            BLUE,
        ),
        (
            f"{metrics['structure_files_present']} / {metrics['processed_cases']}",
            "Processed structure packs present",
            SKY,
        ),
        (
            f"{100.0 * float(metrics['finite_rate']):.1f}% | {metrics['cross_split_leakage_count']}",
            "Finite values | leaked structure groups",
            ORANGE,
        ),
    ]
    for index, (value, label, accent) in enumerate(cards):
        _draw_card(fig.add_subplot(grid[1, index * 3 : (index + 1) * 3]), value, label, accent=accent)

    bar_ax = fig.add_subplot(grid[2, :5])
    x = np.arange(len(ncoil_values))
    bars = bar_ax.bar(x, case_counts, color=BLUE, width=0.62, edgecolor="none")
    for bar, value in zip(bars, case_counts):
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(case_counts) * 0.035,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10.5,
            weight="bold",
            color=DARK,
        )
    bar_ax.set_title("Cases by active-coil count", loc="left", pad=10, weight="bold")
    bar_ax.set_xlabel(r"Active coils, $N_{coil}$")
    bar_ax.set_ylabel("Cases")
    bar_ax.set_xticks(x, [str(value) for value in ncoil_values])
    bar_ax.set_ylim(0, max(case_counts) * 1.24)
    bar_ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    bar_ax.set_axisbelow(True)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)

    heat_ax = fig.add_subplot(grid[2, 5:])
    max_split_count = max(int(np.max(split_matrix)), 1)
    blue_map = plt.get_cmap("Blues")
    for row_index in range(split_matrix.shape[0]):
        for col_index in range(split_matrix.shape[1]):
            value = int(split_matrix[row_index, col_index])
            highlight_missing = SPLIT_ORDER[row_index] == "test" and ncoil_values[col_index] in test_missing
            shade = 0.08 + 0.87 * (value / max_split_count)
            heat_ax.add_patch(
                Rectangle(
                    (col_index - 0.5, row_index - 0.5),
                    1.0,
                    1.0,
                    facecolor=blue_map(shade),
                    edgecolor=WHITE,
                    linewidth=0.8,
                )
            )
            if highlight_missing:
                heat_ax.add_patch(
                    Rectangle(
                        (col_index - 0.48, row_index - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor=ORANGE,
                        linewidth=2.4,
                    )
                )
            threshold = 0.56 * max_split_count
            color = WHITE if value >= threshold else (ORANGE if highlight_missing else DARK)
            heat_ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=11,
                weight="bold",
                color=color,
            )
    heat_ax.set_title("Structure groups by split and coil count", loc="left", pad=10, weight="bold")
    heat_ax.set_xlabel(r"Active coils, $N_{coil}$")
    heat_ax.set_xticks(np.arange(len(ncoil_values)), [str(value) for value in ncoil_values])
    heat_ax.set_yticks(np.arange(len(SPLIT_ORDER)), [value.capitalize() for value in SPLIT_ORDER])
    heat_ax.set_xlim(-0.5, len(ncoil_values) - 0.5)
    heat_ax.set_ylim(len(SPLIT_ORDER) - 0.5, -0.5)
    heat_ax.tick_params(length=0)
    for spine in heat_ax.spines.values():
        spine.set_color(LIGHT)
    stem = "gec_icp_dataset_quality"
    output_files = _save_figure(fig, out_dir, stem, dpi=dpi)
    metadata_path = out_dir / f"{stem}_metadata.json"
    metadata = {
        "figure": "Conference-ready GEC-ICP dataset coverage and integrity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": metrics,
        "visual_encoding": {
            "bar_chart": "case count by active-coil count",
            "heatmap": "structure-group count by source split and active-coil count",
            "orange_outlines": "coil counts absent from the test split",
        },
        "output_files": [_relative_or_absolute(path) for path in output_files],
    }
    _write_metadata(metadata_path, metadata)
    return output_files, metadata_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create GEC-CCP and GEC-ICP conference dataset-quality figures."
    )
    parser.add_argument("--only", choices=("all", "ccp", "icp"), default="all")
    parser.add_argument("--ccp-source-root", default=str(DEFAULT_CCP_SOURCE_ROOT))
    parser.add_argument("--ccp-dataset-root", default=str(DEFAULT_CCP_DATASET_ROOT))
    parser.add_argument("--icp-dataset-root", default=str(DEFAULT_ICP_DATASET_ROOT))
    parser.add_argument("--icp-audit-json", default=str(DEFAULT_ICP_AUDIT_JSON))
    parser.add_argument("--icp-source-summary", default=str(DEFAULT_ICP_SOURCE_SUMMARY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if int(args.dpi) < 72:
        raise ValueError("--dpi must be >= 72")
    _configure_matplotlib()
    out_dir = _resolve(args.out_dir)
    written: list[Path] = []

    if args.only in {"all", "ccp"}:
        ccp_metrics = _audit_ccp(
            _resolve(args.ccp_source_root),
            _resolve(args.ccp_dataset_root),
        )
        figure_paths, metadata_path = _plot_ccp(ccp_metrics, out_dir, dpi=int(args.dpi))
        written.extend([*figure_paths, metadata_path])
        print(
            "CCP audit: "
            f"success={ccp_metrics['source_success_cases']}/{ccp_metrics['source_cases']}, "
            f"processed={ccp_metrics['processed_cases']}, "
            f"missing={ccp_metrics['missing_case_count']}, "
            f"finite={100.0 * float(ccp_metrics['finite_rate']):.3f}%"
        )

    if args.only in {"all", "icp"}:
        icp_metrics = _audit_icp(
            _resolve(args.icp_dataset_root),
            _resolve(args.icp_audit_json),
            _resolve(args.icp_source_summary),
        )
        figure_paths, metadata_path = _plot_icp(icp_metrics, out_dir, dpi=int(args.dpi))
        written.extend([*figure_paths, metadata_path])
        print(
            "ICP audit: "
            f"success={icp_metrics['source_success_cases']}/{icp_metrics['processed_cases']}, "
            f"groups={icp_metrics['structure_groups']}, "
            f"leakage={icp_metrics['cross_split_leakage_count']}, "
            f"finite={100.0 * float(icp_metrics['finite_rate']):.3f}%"
        )

    for path in written:
        print(_relative_or_absolute(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
