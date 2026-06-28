"""Run bundle loader for reproducible inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.task_spec import TaskSpecV1
from plasma_surrogate.models.checkpoint import load_checkpoint
from plasma_surrogate.preprocessing.scalers import TransformBundle
from plasma_surrogate.preprocessing.schema import AxisSchema, CondSchema


@dataclass
class RunBundle:
    cfg: dict[str, Any]
    task_spec: TaskSpecV1
    model: Any
    transforms: dict[str, Any]
    schemas: dict[str, Any]
    geometry_store: Any
    run_dir: Path

    def cond_schema_obj(self) -> CondSchema:
        raw = self.schemas.get("cond_schema", {})
        if not raw:
            raise FileNotFoundError(f"Missing cond_schema.json under {self.run_dir / 'preprocessing' / 'schema'}")
        return CondSchema.from_dict(raw)

    def axis_schema_obj(self) -> AxisSchema:
        raw = self.schemas.get("axis_schema", {})
        if not raw:
            raise FileNotFoundError(f"Missing axis_schema.json under {self.run_dir / 'preprocessing' / 'schema'}")
        return AxisSchema.from_dict(raw)

    def transform_bundle(self) -> TransformBundle:
        cond_scaler = self.transforms.get("cond_scaler", {})
        y_scalers = self.transforms.get("y_scalers", {})
        if not cond_scaler or not y_scalers:
            raise FileNotFoundError(f"Missing scaler artifacts under {self.run_dir / 'preprocessing' / 'scalers'}")
        layout = self.schemas.get("output_layout", {})
        y_order = [str(v) for v in layout.get("vars", []) if str(v).strip()]
        if not y_order:
            raise FileNotFoundError(
                f"Missing output_layout vars under {self.run_dir / 'preprocessing' / 'schema' / 'output_layout.json'}"
            )
        return TransformBundle.from_dict(
            cond_scaler=cond_scaler,
            y_scalers=y_scalers,
            y_order=y_order,
        )

    def split_random(self) -> dict[str, list[str]]:
        split_path = self.run_dir / "preprocessing" / "split" / "split_random_v1.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split artifact: {split_path}")
        with split_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            "train": [str(v) for v in raw.get("train", [])],
            "val": [str(v) for v in raw.get("val", [])],
            "test": [str(v) for v in raw.get("test", [])],
        }


class RunBundleLoader:
    """Load bundle assets from a run directory."""

    @staticmethod
    def _load_json_if_exists(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_npz_if_exists(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        data = np.load(path, allow_pickle=True)
        return {k: np.asarray(data[k]) for k in data.files}

    @classmethod
    def _load_task_spec(
        cls,
        run_path: Path,
        cfg: dict[str, Any],
    ) -> TaskSpecV1:
        task_spec_path = run_path / "task_spec.yaml"
        if cfg.get("task", {}).get("spec_path"):
            task_spec_path = Path(cfg["task"]["spec_path"])
            if not task_spec_path.is_absolute():
                task_spec_path = run_path / task_spec_path
        if task_spec_path.exists():
            return TaskSpecV1.from_yaml(task_spec_path)
        raise FileNotFoundError(f"Missing task_spec.yaml under {run_path}")

    @classmethod
    def load(
        cls,
        run_dir: str | Path,
        model: Any = None,
        geometry_store: Any = None,
    ) -> RunBundle:
        run_path = Path(run_dir)
        cfg_path = run_path / "resolved_config.yaml"
        cfg: dict[str, Any] = {}
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        if model is None and (run_path / "checkpoints" / "meta.json").exists():
            model = load_checkpoint(run_path / "checkpoints")

        transforms = {
            "cond_scaler": cls._load_json_if_exists(run_path / "preprocessing" / "scalers" / "cond_scaler.json"),
            "y_scalers": cls._load_json_if_exists(run_path / "preprocessing" / "scalers" / "y_scalers.json"),
            "coord_scaler": cls._load_json_if_exists(run_path / "preprocessing" / "scalers" / "coord_scaler.json"),
            "coord_feature_scaler": cls._load_json_if_exists(
                run_path / "preprocessing" / "scalers" / "coord_feature_scaler.json"
            ),
            "distance_transform_stats": cls._load_json_if_exists(
                run_path / "preprocessing" / "scalers" / "distance_transform_stats.json"
            ),
            "xgrid_channel_scalers": cls._load_json_if_exists(run_path / "preprocessing" / "scalers" / "xgrid_channel_scalers.json"),
        }

        preprocess_report = cls._load_json_if_exists(run_path / "preprocessing" / "validation" / "report.json")
        coord_pack_rel = str(preprocess_report.get("coord_feature_pack_path", "features/coord_feature_pack.npz")).strip()
        coord_pack_path: Path | None = None
        coord_pack_meta_path: Path | None = None
        if coord_pack_rel:
            coord_pack_path = run_path / "preprocessing" / coord_pack_rel
            coord_pack_meta_path = coord_pack_path.parent / f"{coord_pack_path.stem}_meta.json"
        descriptor_pack_rel = str(
            preprocess_report.get("structure_descriptor_pack_path", "features/structure_descriptor_pack.npz")
        ).strip()
        descriptor_pack_path: Path | None = None
        descriptor_pack_meta_path: Path | None = None
        if descriptor_pack_rel:
            descriptor_pack_path = run_path / "preprocessing" / descriptor_pack_rel
            descriptor_pack_meta_path = descriptor_pack_path.parent / f"{descriptor_pack_path.stem}_meta.json"
        latent_pack_rel = str(preprocess_report.get("latent_feature_pack_path", "features/latent_feature_pack.npz")).strip()
        latent_pack_path: Path | None = None
        latent_pack_meta_path: Path | None = None
        if latent_pack_rel:
            latent_pack_path = run_path / "preprocessing" / latent_pack_rel
            latent_pack_meta_path = latent_pack_path.parent / f"{latent_pack_path.stem}_meta.json"
        case_spatial_pack_rel = str(preprocess_report.get("case_spatial_feature_pack_path", "")).strip()
        case_spatial_pack_path: Path | None = None
        case_spatial_pack_meta_path: Path | None = None
        if case_spatial_pack_rel:
            case_spatial_pack_path = run_path / "preprocessing" / case_spatial_pack_rel
            case_spatial_pack_meta_path = case_spatial_pack_path.parent / f"{case_spatial_pack_path.stem}_meta.json"
        static_spatial_pack_rel = str(preprocess_report.get("static_spatial_feature_pack_path", "")).strip()
        static_spatial_pack_path: Path | None = None
        static_spatial_pack_meta_path: Path | None = None
        if static_spatial_pack_rel:
            static_spatial_pack_path = run_path / "preprocessing" / static_spatial_pack_rel
            static_spatial_pack_meta_path = static_spatial_pack_path.parent / f"{static_spatial_pack_path.stem}_meta.json"
        case_structure_pack_rel = str(preprocess_report.get("case_structure_feature_pack_path", "")).strip()
        case_structure_pack_path: Path | None = None
        case_structure_pack_meta_path: Path | None = None
        if case_structure_pack_rel:
            case_structure_pack_path = run_path / "preprocessing" / case_structure_pack_rel
            case_structure_pack_meta_path = case_structure_pack_path.parent / f"{case_structure_pack_path.stem}_meta.json"
        deeponet_root = run_path / "preprocessing" / "sampling" / "deeponet"
        deeponet_default_index = cls._load_json_if_exists(deeponet_root / "default" / "sensor_query_index.json")
        deeponet_default_meta = cls._load_json_if_exists(deeponet_root / "default" / "index_meta.json")
        deeponet_poisson_index = cls._load_json_if_exists(deeponet_root / "poisson_head" / "sensor_query_index.json")
        deeponet_poisson_meta = cls._load_json_if_exists(deeponet_root / "poisson_head" / "index_meta.json")
        deeponet_boundary_index = cls._load_json_if_exists(deeponet_root / "boundary_operator" / "sensor_query_index.json")
        deeponet_boundary_meta = cls._load_json_if_exists(deeponet_root / "boundary_operator" / "index_meta.json")
        deeponet_index = deeponet_default_index or deeponet_poisson_index or deeponet_boundary_index
        deeponet_index_meta = deeponet_default_meta or deeponet_poisson_meta or deeponet_boundary_meta

        schemas = {
            "cond_schema": cls._load_json_if_exists(run_path / "preprocessing" / "schema" / "cond_schema.json"),
            "axis_schema": cls._load_json_if_exists(run_path / "preprocessing" / "schema" / "axis_schema.json"),
            "channel_map": cls._load_json_if_exists(run_path / "preprocessing" / "schema" / "channel_map.json"),
            "output_layout": cls._load_json_if_exists(run_path / "preprocessing" / "schema" / "output_layout.json"),
            "target_role_schema": cls._load_json_if_exists(
                run_path / "preprocessing" / "schema" / "target_role_schema.json"
            ),
            "deeponet_index": deeponet_index,
            "deeponet_index_meta": deeponet_index_meta,
            "deeponet_task_hashes": cls._load_json_if_exists(deeponet_root / "task_hashes.json"),
            "deeponet_poisson_head_index": deeponet_poisson_index,
            "deeponet_poisson_head_meta": deeponet_poisson_meta,
            "deeponet_boundary_operator_index": deeponet_boundary_index,
            "deeponet_boundary_operator_meta": deeponet_boundary_meta,
            "cond_stats": cls._load_json_if_exists(run_path / "preprocessing" / "stats" / "cond_stats.json"),
            "y_stats": cls._load_json_if_exists(run_path / "preprocessing" / "stats" / "y_stats.json"),
            "runtime_schema_hashes": cls._load_json_if_exists(
                run_path / "preprocessing" / "validation" / "runtime_schema_hashes.json"
            ),
            "geometry_cache_index": cls._load_json_if_exists(run_path / "featurization" / "geometry_cache_index.json"),
            "preprocess_report": preprocess_report,
            "coord_feature_pack_meta": cls._load_json_if_exists(coord_pack_meta_path) if coord_pack_meta_path else {},
            "coord_feature_pack": cls._load_npz_if_exists(coord_pack_path) if coord_pack_path else None,
            "structure_descriptor_pack_meta": cls._load_json_if_exists(descriptor_pack_meta_path) if descriptor_pack_meta_path else {},
            "structure_descriptor_pack": cls._load_npz_if_exists(descriptor_pack_path) if descriptor_pack_path else None,
            "latent_feature_pack_meta": cls._load_json_if_exists(latent_pack_meta_path) if latent_pack_meta_path else {},
            "latent_feature_pack": cls._load_npz_if_exists(latent_pack_path) if latent_pack_path else None,
            "case_spatial_feature_pack_meta": (
                cls._load_json_if_exists(case_spatial_pack_meta_path) if case_spatial_pack_meta_path else {}
            ),
            "case_spatial_feature_pack": (
                cls._load_npz_if_exists(case_spatial_pack_path) if case_spatial_pack_path else None
            ),
            "static_spatial_feature_pack_meta": (
                cls._load_json_if_exists(static_spatial_pack_meta_path) if static_spatial_pack_meta_path else {}
            ),
            "static_spatial_feature_pack": (
                cls._load_npz_if_exists(static_spatial_pack_path) if static_spatial_pack_path else None
            ),
            "case_structure_feature_pack_meta": (
                cls._load_json_if_exists(case_structure_pack_meta_path) if case_structure_pack_meta_path else {}
            ),
            "case_structure_feature_pack": (
                cls._load_npz_if_exists(case_structure_pack_path) if case_structure_pack_path else None
            ),
        }
        task_spec = cls._load_task_spec(run_path=run_path, cfg=cfg)

        ArtifactStore(run_path / "artifacts")

        return RunBundle(
            cfg=cfg,
            task_spec=task_spec,
            model=model,
            transforms=transforms,
            schemas=schemas,
            geometry_store=geometry_store,
            run_dir=run_path,
        )


def require_artifacts(run_dir: str | Path, rel_paths: list[str | Path]) -> None:
    root = Path(run_dir)
    missing: list[str] = []
    for rel in rel_paths:
        rel_path = Path(rel)
        full = root / rel_path
        if not full.exists():
            missing.append(str(rel_path))
    if missing:
        raise FileNotFoundError(f"Missing required artifacts under {root}: {missing}")


def required_preprocess_artifacts() -> list[str]:
    return [
        "preprocessing/split/split_random_v1.json",
        "preprocessing/schema/cond_schema.json",
        "preprocessing/schema/axis_schema.json",
        "preprocessing/schema/output_layout.json",
        "preprocessing/scalers/cond_scaler.json",
        "preprocessing/scalers/y_scalers.json",
    ]


def ensure_preprocess_contract(
    run_dir: str | Path,
    *,
    on_missing: Any | None = None,
) -> None:
    """Ensure preprocess artifacts exist; optionally run recovery callback once."""

    try:
        require_artifacts(run_dir, required_preprocess_artifacts())
    except FileNotFoundError:
        if on_missing is None:
            raise
        on_missing()
        require_artifacts(run_dir, required_preprocess_artifacts())
