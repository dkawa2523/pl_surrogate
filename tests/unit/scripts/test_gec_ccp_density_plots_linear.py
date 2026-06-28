from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_method_triplet_density_display_uses_linear_physical_values() -> None:
    module = _load_script(ROOT / "scripts" / "plot_gec_ccp_method_spatial_triplets.py", "plot_triplets")
    arr = np.asarray([[1.0e14, 1.0e16]], dtype=np.float32)

    display, label = module._as_display("ne", arr)

    assert label == "ne"
    np.testing.assert_allclose(display, arr)


def test_truth_prediction_density_display_uses_linear_physical_values() -> None:
    module = _load_script(
        ROOT / "experiments" / "gec_ccp" / "scripts" / "plot_gec_ccp_spatial_truth_pred.py",
        "plot_truth_pred",
    )
    truth = np.asarray([[1.0e14, 1.0e16]], dtype=np.float64)
    pred = np.asarray([[2.0e14, 0.5e16]], dtype=np.float64)

    truth_display, pred_display, error_display, label = module._display_arrays("ni", truth, pred)

    assert label == "ni"
    np.testing.assert_allclose(truth_display, truth)
    np.testing.assert_allclose(pred_display, pred)
    np.testing.assert_allclose(error_display, pred - truth)
