from __future__ import annotations

import numpy as np
import pytest

from plasma_surrogate.train.deeponet_contracts import (
    resolve_pod_descriptor_and_latent_contract,
    validate_pod_deeponet_experimental_contract,
)
from plasma_surrogate.train.model_dispatch import _resolve_pod_descriptor_splits


def _descriptor_pack() -> dict[str, np.ndarray]:
    rows = np.arange(24, dtype=np.float32).reshape(6, 4)
    return {
        "vector": np.zeros((4,), dtype=np.float32),
        "vectors": rows,
        "feature_names": np.asarray(["a", "b", "c", "d"]),
        "case_ids": np.asarray([f"case_{i}" for i in range(6)]),
    }


def test_pod_descriptor_contract_loads_case_matrix() -> None:
    descriptor_input, meta = resolve_pod_descriptor_and_latent_contract(
        input_mode="table_plus_structure",
        adapter_mode="descriptor_branch",
        descriptor_profile="struct_desc_v2",
        latent_profile="none",
        descriptor_pack=_descriptor_pack(),
        latent_pack=None,
    )

    assert descriptor_input is not None
    assert descriptor_input.shape == (6, 4)
    assert meta["deeponet_pod_descriptor_dim_effective"] == 4
    assert meta["deeponet_pod_descriptor_scope_effective"] == "case_specific"


def test_pod_descriptor_splits_follow_dataset_indices_and_case_ids() -> None:
    pack = _descriptor_pack()
    case_structure_pack = {
        "data": np.arange(6 * 2, dtype=np.float32).reshape(6, 2, 1, 1),
        "case_ids": np.asarray([f"case_{i}" for i in range(6)]),
    }

    train, val, test = _resolve_pod_descriptor_splits(
        descriptor_input=pack["vectors"],
        descriptor_pack=pack,
        case_structure_pack=case_structure_pack,
        n_cases=6,
        tr=np.asarray([4, 1, 0]),
        va=np.asarray([5]),
        te=np.asarray([3, 2]),
        case_ids=[f"case_{i}" for i in range(6)],
    )

    assert np.array_equal(train, pack["vectors"][[4, 1, 0]])
    assert np.array_equal(val, pack["vectors"][[5]])
    assert np.array_equal(test, pack["vectors"][[3, 2]])


def test_pod_descriptor_rejects_case_id_misalignment() -> None:
    pack = _descriptor_pack()
    with pytest.raises(ValueError, match="not aligned"):
        _resolve_pod_descriptor_splits(
            descriptor_input=pack["vectors"],
            descriptor_pack=pack,
            case_structure_pack={
                "data": np.zeros((6, 1, 1, 1), dtype=np.float32),
                "case_ids": np.asarray(["case_1", "case_0", "case_2", "case_3", "case_4", "case_5"]),
            },
            n_cases=6,
            tr=np.asarray([0, 1, 2]),
            va=np.asarray([3]),
            te=np.asarray([4, 5]),
        )


def test_pod_descriptor_rejects_runtime_dataset_case_id_misalignment() -> None:
    pack = _descriptor_pack()
    with pytest.raises(ValueError, match="runtime dataset case_ids"):
        _resolve_pod_descriptor_splits(
            descriptor_input=pack["vectors"],
            descriptor_pack=pack,
            case_structure_pack=None,
            n_cases=6,
            tr=np.asarray([0, 1, 2]),
            va=np.asarray([3]),
            te=np.asarray([4, 5]),
            case_ids=["case_1", "case_0", "case_2", "case_3", "case_4", "case_5"],
        )


def test_pod_descriptor_rejects_repeated_static_vector_for_varying_cases() -> None:
    with pytest.raises(ValueError, match="refuses to repeat one structure descriptor"):
        _resolve_pod_descriptor_splits(
            descriptor_input=np.asarray([1.0, 2.0], dtype=np.float32),
            descriptor_pack={"vector": np.asarray([1.0, 2.0], dtype=np.float32)},
            case_structure_pack={
                "data": np.asarray([[[[0.0]]], [[[1.0]]], [[[2.0]]]], dtype=np.float32),
                "case_ids": np.asarray(["a", "b", "c"]),
            },
            n_cases=3,
            tr=np.asarray([0]),
            va=np.asarray([1]),
            te=np.asarray([2]),
        )


def test_pod_deeponet_contract_accepts_spatial_selection() -> None:
    resolved = validate_pod_deeponet_experimental_contract(
        model_name="deeponet_pod",
        y_vars=["density", "potential"],
        target_family="allvars",
        target_vars=["density", "potential"],
        model_cfg={},
        selection_cfg={
            "mode": "best_val_spatial_objective",
            "spatial": {"boundary_weight": 0.0},
        },
    )

    assert resolved["fit_scope"] == "train_only"
