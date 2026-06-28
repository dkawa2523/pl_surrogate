"""Batch and axis-window aggregation for inference results."""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def aggregate_inference_results(results: list[Any], mode: str) -> Any:
    if len(results) == 0:
        raise ValueError("No results to aggregate")
    result_cls = type(results[0])
    if mode == "window_max":
        best = max(results, key=lambda r: float(r.qoi.get("uniformity", 0.0)))
        return result_cls(
            fields_model={k: np.asarray(v, dtype=np.float32) for k, v in best.fields_model.items()},
            fields_phys={k: np.asarray(v, dtype=np.float32) for k, v in best.fields_phys.items()},
            derived={k: np.asarray(v, dtype=np.float32) for k, v in best.derived.items()},
            qoi={k: (float(v) if _is_numeric_scalar(v) else v) for k, v in best.qoi.items()},
            diagnostics={k: float(v) if _is_numeric_scalar(v) else v for k, v in best.diagnostics.items()},
            case_key=str(best.case_key),
        )

    mean_fields_model = {
        k: np.mean(np.stack([np.asarray(r.fields_model[k], dtype=np.float32) for r in results], axis=0), axis=0)
        for k in results[0].fields_model.keys()
    }
    mean_fields_phys = {
        k: np.mean(np.stack([np.asarray(r.fields_phys[k], dtype=np.float32) for r in results], axis=0), axis=0)
        for k in results[0].fields_phys.keys()
    }
    mean_derived = {
        k: np.mean(np.stack([np.asarray(r.derived[k], dtype=np.float32) for r in results], axis=0), axis=0)
        for k in results[0].derived.keys()
    }
    mean_qoi: dict[str, Any] = {}
    for key in results[0].qoi.keys():
        first = results[0].qoi[key]
        if _is_numeric_scalar(first):
            mean_qoi[key] = float(np.mean([float(r.qoi[key]) for r in results]))
        else:
            mean_qoi[key] = first
    mean_diag = {
        k: float(np.mean([float(r.diagnostics[k]) for r in results]))
        for k in results[0].diagnostics.keys()
        if _is_numeric_scalar(results[0].diagnostics[k])
    }
    return result_cls(
        fields_model=mean_fields_model,
        fields_phys=mean_fields_phys,
        derived=mean_derived,
        qoi=mean_qoi,
        diagnostics=mean_diag,
        case_key="",
    )


__all__ = ["aggregate_inference_results"]
