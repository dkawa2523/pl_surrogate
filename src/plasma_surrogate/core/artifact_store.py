"""Artifact persistence for json/csv/npz and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ArtifactStore:
    """Thin wrapper around output directory writes/reads."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, rel_path: str | Path) -> Path:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, rel_path: str | Path, payload: Any) -> Path:
        path = self._path(rel_path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def load_json(self, rel_path: str | Path) -> Any:
        with (self.root / rel_path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_csv(self, rel_path: str | Path, header: list[str], rows: list[list[Any]]) -> Path:
        path = self._path(rel_path)
        with path.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")
        return path

    def save_npz(self, rel_path: str | Path, **arrays: np.ndarray) -> Path:
        path = self._path(rel_path)
        np.savez_compressed(path, **arrays)
        return path

    def load_npz(self, rel_path: str | Path) -> dict[str, np.ndarray]:
        data = np.load(self.root / rel_path)
        return {k: data[k] for k in data.files}
