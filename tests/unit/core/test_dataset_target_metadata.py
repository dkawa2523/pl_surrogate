from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from plasma_surrogate.core.dataset_io import load_csv_npz_dataset


def test_csv_npz_dataset_preserves_target_role_metadata(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "geometry").mkdir()
    np.save(root / "geometry" / "mask_plasma.npy", np.ones((2, 2), dtype=np.float32))
    np.savez(
        root / "case_a.npz",
        den=np.ones((2, 2), dtype=np.float32),
        temp=np.ones((2, 2), dtype=np.float32) * 2.0,
        pot=np.zeros((2, 2), dtype=np.float32),
    )
    with (root / "index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "axis", "p", "fields_npz"])
        writer.writeheader()
        writer.writerow({"case_id": "a", "axis": "0.0", "p": "1.0", "fields_npz": "case_a.npz"})

    dataset = load_csv_npz_dataset(
        {
            "root": str(root),
            "cond_columns": ["p"],
            "targets": [
                {
                    "id": "electron_density",
                    "source_key": "den",
                    "role": "density_electron",
                    "positive": True,
                    "field_family": "density",
                    "default_region": "plasma_only",
                    "units": "m^-3",
                    "dtype": "float32",
                },
                {
                    "id": "electron_temperature",
                    "source_key": "temp",
                    "role": "temperature_electron",
                    "positive": True,
                    "field_family": "temperature",
                },
                {
                    "id": "plasma_potential",
                    "source_key": "pot",
                    "role": "potential",
                    "positive": False,
                    "field_family": "electrostatic",
                },
            ],
        },
        run_dir=tmp_path,
    )

    assert [target["id"] for target in dataset.target_metadata] == [
        "electron_density",
        "electron_temperature",
        "plasma_potential",
    ]
    assert dataset.target_metadata[0]["role"] == "density_electron"
    assert dataset.target_metadata[0]["positive"] is True
    assert dataset.target_metadata[0]["default_region"] == "plasma_only"
    assert set(dataset.cases[0]["y"]) == {"electron_density", "electron_temperature", "plasma_potential"}
