"""Task spec contracts for v1 outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GridSpec:
    axes_order: tuple[str, ...]
    coord_components: tuple[str, ...]
    shape: tuple[int, ...]
    coord_system: str = "cartesian"

    def validate(self) -> None:
        if len(self.axes_order) != len(self.shape):
            raise ValueError("axes_order length must match shape dims")
        if len(set(self.axes_order)) != len(self.axes_order):
            raise ValueError("axes_order must be unique")
        if len(set(self.coord_components)) != len(self.coord_components):
            raise ValueError("coord_components must be unique")


@dataclass
class OutputSpec:
    name: str
    units: str = ""
    transform: str = "zscore"


@dataclass
class TaskSpecV1:
    outputs: list[OutputSpec]
    transforms: dict[str, str]
    units: dict[str, str]
    grid_spec: GridSpec
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.grid_spec.validate()
        output_names = [o.name for o in self.outputs]
        if len(output_names) == 0:
            raise ValueError("TaskSpec v1 requires at least one output")
        if len(set(output_names)) != len(output_names):
            raise ValueError(f"TaskSpec v1 outputs must be unique: {output_names}")
        for name in output_names:
            if name not in self.transforms:
                raise ValueError(f"Missing transform for output: {name}")
            if name not in self.units:
                raise ValueError(f"Missing units for output: {name}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TaskSpecV1":
        grid_raw = raw.get("grid_spec", {})
        grid = GridSpec(
            axes_order=tuple(grid_raw.get("axes_order", ("y", "x"))),
            coord_components=tuple(grid_raw.get("coord_components", ("x", "y"))),
            shape=tuple(grid_raw.get("shape", (16, 16))),
            coord_system=grid_raw.get("coord_system", "cartesian"),
        )

        outputs = [OutputSpec(**o) for o in raw.get("outputs", [])]
        spec = cls(
            outputs=outputs,
            transforms=dict(raw.get("transforms", {})),
            units=dict(raw.get("units", {})),
            grid_spec=grid,
            metadata=dict(raw.get("metadata", {})),
        )
        spec.validate()
        return spec

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskSpecV1":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": [o.__dict__ for o in self.outputs],
            "transforms": self.transforms,
            "units": self.units,
            "grid_spec": {
                "axes_order": list(self.grid_spec.axes_order),
                "coord_components": list(self.grid_spec.coord_components),
                "shape": list(self.grid_spec.shape),
                "coord_system": self.grid_spec.coord_system,
            },
            "metadata": self.metadata,
        }
