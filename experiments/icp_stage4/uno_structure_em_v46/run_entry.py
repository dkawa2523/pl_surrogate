#!/usr/bin/env python
"""Run one benchmark with the isolated v46 UNO factory installed in-process."""

from __future__ import annotations

import sys
import os
import shutil
from pathlib import Path

import yaml

os.environ.setdefault("PLASMA_SURROGATE_ENABLE_TORCH", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from plasma_surrogate.cli.main import main as _cli_import_guard  # noqa: E402,F401
from experimental_uno import ExperimentalUNOBaseline, normalize_experimental_uno_cfg  # noqa: E402


BASE_PROFILE = (
    "x",
    "y",
    "mask_plasma",
    "distance_signed",
    "distance_any",
    "part_sdf_union",
    "part_source_mean",
    "vacuum_aphi_unit",
    "vacuum_br_unit",
    "vacuum_bz_unit",
    "vacuum_bmag_unit",
    "part_second_proximity",
    "part_competition",
    "solid_proximity",
)

PER_COIL = tuple(f"sdf_coil_{slot:02d}" for slot in range(1, 7))
PROFILE_BY_NAME = {
    "icp_uno_structure_em_v46_base": BASE_PROFILE,
    "icp_uno_structure_em_v46_percoil": (*BASE_PROFILE, *PER_COIL),
    "icp_uno_structure_em_v46_emshape": tuple(v for v in BASE_PROFILE if v != "vacuum_bmag_unit"),
    "icp_uno_structure_em_v46_all": tuple(v for v in (*BASE_PROFILE, *PER_COIL) if v != "vacuum_bmag_unit"),
}

ADOPTED_BASE_PROFILE = (
    "x",
    "y",
    "mask_plasma",
    "distance_signed",
    "distance_any",
    "part_sdf_union",
    "part_source_mean",
    "vacuum_aphi_unit",
    "vacuum_br_unit",
    "vacuum_bz_unit",
    "vacuum_bmag_unit",
    "part_second_proximity",
    "part_competition",
    "solid_proximity",
)


def _clone_preprocessing(source: Path, target: Path) -> None:
    """Hardlink an immutable deterministic preprocessing bundle."""

    source = source.resolve()
    target = target.resolve()
    allowed = (ROOT / "runs/icp_uno_structure_em_v46").resolve()
    if not str(target).lower().startswith(str(allowed).lower()):
        raise ValueError(f"refusing preprocessing target outside isolated run root: {target}")
    if not (source / "validation/report.json").is_file():
        raise FileNotFoundError(f"source preprocessing bundle is incomplete: {source}")
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        destination = target / source_file.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        try:
            os.link(source_file, destination)
        except OSError:
            shutil.copy2(source_file, destination)


def _install_preprocessing_reuse(benchmark: dict, profile_channels: tuple[str, ...]) -> None:
    """Reuse only byte-identical preprocessing contracts."""

    from plasma_surrogate.preprocessing import runner as preprocessing_runner

    run_dir = (ROOT / str(benchmark["output_dir"])).resolve()
    target = run_dir / "preprocessing"
    source: Path | None = None
    if (target / "validation/report.json").is_file():
        # A prior invocation may have completed preprocessing before a later
        # model-contract failure.  Treat that validated bundle as immutable
        # and do not build the multi-GB feature pack a second time.
        source = target
    elif tuple(profile_channels) == ADOPTED_BASE_PROFILE:
        source = ROOT / "runs/icp_conference_continuity_v45/final/sdf_vacuum_structure/seed_1237/preprocessing"
    elif "sdf_coil_01" in profile_channels:
        # ALL follows ABD in the manifest and shares the same superset pack.
        candidate = run_dir.parent.parent / "ABD_shared_coils/seed_1237/preprocessing"
        if candidate.resolve() != target.resolve() and (candidate / "validation/report.json").is_file():
            source = candidate
    if source is None:
        return
    if source.resolve() != target.resolve():
        _clone_preprocessing(source, target)

    def _reuse_run(self, *, cases, geometry_root, target_metadata=None):
        del self, cases, geometry_root, target_metadata
        return None

    preprocessing_runner.PreprocessRunner.run = _reuse_run


def install_isolated_factory(profile_channels: tuple[str, ...]) -> None:
    """Patch only this process; no source package files are changed."""

    from dataclasses import replace

    from plasma_surrogate.features import structure_feature_registry
    from plasma_surrogate.models import checkpoint_grid, factory
    from plasma_surrogate.preprocessing import runner as preprocessing_runner

    structure_feature_registry.FEATURE_PROFILE_CHANNELS.update(PROFILE_BY_NAME)
    # Reuse the already validated compact-pack implementation under its known
    # profile name, but change its channel contract only in this process.  The
    # builder already knows how to load per-coil SDFs, aggregate SDF/source
    # maps, structural summaries, and vacuum fields; the mainline limitation is
    # only its fixed profile enumeration.
    formal_profile = "icp_coil_sdf_source_mean_vacuum_structure_v1"
    structure_feature_registry.FEATURE_PROFILE_CHANNELS[formal_profile] = tuple(profile_channels)
    static = {"x", "y", "mask_plasma", "distance_signed", "distance_any"}
    preprocessing_runner.ICP_SDF_MEAN_VACUUM_STRUCTURE_CASE_CHANNELS = tuple(
        name for name in profile_channels if name not in static
    )
    factory.UNOBaseline = ExperimentalUNOBaseline
    factory.normalize_uno_cfg = normalize_experimental_uno_cfg
    checkpoint_grid.UNOBaseline = ExperimentalUNOBaseline
    checkpoint_grid.normalize_uno_cfg = normalize_experimental_uno_cfg
    # The mainline checkpoint registry intentionally accepts only the formal
    # ``uno_lite_v1`` implementation tag.  Permit the isolated tag in this
    # process so evaluation can deserialize the experimental checkpoints,
    # without widening the production loader contract on disk.
    uno_spec = checkpoint_grid._CHECKPOINT_SPECS["u_no"]
    isolated_impl = "isolated_icp_uno_structure_em_v46"
    allowed = tuple(dict.fromkeys((*uno_spec.allowed_impl_versions, isolated_impl)))
    checkpoint_grid._CHECKPOINT_SPECS["u_no"] = replace(
        uno_spec, allowed_impl_versions=allowed
    )


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        config_path = Path(cli_argv[cli_argv.index("--config") + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("isolated v46 entry requires --config") from exc
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = dict(payload.get("benchmark", payload))
    profile_channels = tuple(
        benchmark.get("metadata", {}).get(
            "experimental_profile_channels",
            benchmark["train"]["u_no"]["input_features"]["features"],
        )
    )
    install_isolated_factory(profile_channels)
    _install_preprocessing_reuse(benchmark, profile_channels)
    from plasma_surrogate.cli.main import main as cli_main

    return int(cli_main(cli_argv))


if __name__ == "__main__":
    raise SystemExit(main())
