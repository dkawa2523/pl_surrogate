from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plasma_surrogate.infer.contracts import validate_output_head_metadata_contract
from plasma_surrogate.models.checkpoint import build_model_from_name, load_checkpoint, save_checkpoint
from tests._runtime_requirements import require_torch_runtime


pytestmark = pytest.mark.torch_runtime


def _target_role_schema() -> dict:
    return {
        "targets": [
            {"id": "temperature_main", "field_family": "temperature", "role": "temperature_electron"},
            {"id": "density_b", "field_family": "density", "role": "density_ion"},
            {"id": "potential", "field_family": "electrostatic", "role": "potential"},
            {"id": "density_a", "field_family": "density", "role": "density_electron"},
        ],
        "positive_targets": ["temperature_main", "density_b", "density_a"],
    }


def _logical_output_keys() -> list[str]:
    return ["electron_temperature", "electron_density", "ion_density", "plasma_potential"]


def _logical_target_role_schema() -> dict:
    return {
        "targets": [
            {"id": "electron_temperature", "field_family": "temperature", "role": "temperature_electron"},
            {"id": "electron_density", "field_family": "density", "role": "density_electron"},
            {"id": "ion_density", "field_family": "density", "role": "density_ion"},
            {"id": "plasma_potential", "field_family": "electrostatic", "role": "potential"},
        ],
        "positive_targets": ["electron_temperature", "electron_density", "ion_density"],
    }


def _model_cfg(model_name: str) -> dict:
    base = {
        "backend": "torch",
        "output_heads": {"mode": "role_grouped"},
        "target_role_schema": _target_role_schema(),
    }
    if model_name in {"fno", "ffno"}:
        base.update({"n_modes": 2, "spectral_cfg": {"width": 8, "n_layers": 1, "dropout": 0.0}})
    elif model_name == "unet":
        base.update({"conv_cfg": {"base_channels": 4, "depth": 1, "upsample_mode": "deconv"}})
    elif model_name in {"unetpp", "unetpp_attn"}:
        conv_cfg = {"base_channels": 4, "depth": 2, "upsample_mode": "bilinear"}
        if model_name == "unetpp_attn":
            conv_cfg["attention_cfg"] = {"enabled": True, "reduction": 2, "gate_activation": "sigmoid"}
        base.update({"conv_cfg": conv_cfg})
    elif model_name == "u_no":
        base.update({"n_modes": 2, "uno_cfg": {"width": 8, "n_layers": 1, "dropout": 0.0}})
    elif model_name == "cno":
        base.update({"cno_cfg": {"width": 8, "n_layers": 1, "dropout": 0.0, "kernel_size": 3}})
    else:
        raise AssertionError(f"missing role_grouped test cfg for {model_name}")
    return base


def _custom_groups_output_heads(
    *,
    strict: bool = True,
    groups: dict | None = None,
) -> dict:
    return {
        "mode": "custom_groups",
        "strict": bool(strict),
        "groups": groups
        or {
            "density": {"targets": ["ion_density", "electron_density"]},
            "thermal": {"targets": ["electron_temperature"]},
            "electrostatic": {"targets": ["plasma_potential"]},
        },
    }


def _build_role_grouped_model(model_name: str):
    require_torch_runtime()
    output_keys = ["temperature_main", "density_b", "potential", "density_a"]
    return build_model_from_name(
        model_name=model_name,
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg=_model_cfg(model_name),
        out_channels=len(output_keys),
        output_keys=output_keys,
        unet_feature_channels=["x", "y"],
    )


def _build_custom_groups_model(*, output_heads: dict | None = None):
    require_torch_runtime()
    output_keys = _logical_output_keys()
    cfg = _model_cfg("fno")
    cfg["output_heads"] = dict(output_heads or _custom_groups_output_heads())
    cfg["target_role_schema"] = _logical_target_role_schema()
    return build_model_from_name(
        model_name="fno",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg=cfg,
        out_channels=len(output_keys),
        output_keys=output_keys,
        unet_feature_channels=["x", "y"],
    )


@pytest.mark.parametrize("model_name", ["fno", "ffno", "unet", "unetpp", "unetpp_attn", "u_no", "cno"])
def test_role_grouped_output_head_preserves_shape_and_output_order(model_name: str) -> None:
    model = _build_role_grouped_model(model_name)
    cond = np.zeros((2, 3), dtype=np.float32)
    spatial = np.zeros((8, 8, 2), dtype=np.float32)
    pred = model.forward(cond, spatial_features=spatial)

    assert pred.shape == (2, 4, 8, 8)
    assert model.output_heads_mode == "role_grouped"
    assert model.output_keys == ["temperature_main", "density_b", "potential", "density_a"]
    assert model.target_groups["density"].targets == ("density_b", "density_a")
    assert model.target_groups["temperature"].targets == ("temperature_main",)
    assert model.target_groups["electrostatic"].targets == ("potential",)
    assert [group["name"] for group in model.target_groups_metadata] == ["temperature", "density", "electrostatic"]


def test_custom_groups_output_head_preserves_output_order() -> None:
    model = _build_custom_groups_model()
    pred = model.forward(np.zeros((2, 3), dtype=np.float32))

    assert pred.shape == (2, 4, 8, 8)
    assert model.output_heads_mode == "custom_groups"
    assert model.output_keys == _logical_output_keys()
    assert list(model.target_groups) == ["density", "thermal", "electrostatic"]
    assert model.target_groups["density"].targets == ("electron_density", "ion_density")
    assert model.target_groups["thermal"].targets == ("electron_temperature",)
    assert model.target_groups["electrostatic"].targets == ("plasma_potential",)
    assert model.target_groups["density"].source == "custom"


def test_custom_groups_spatial_refine_head_preserves_output_order(tmp_path: Path) -> None:
    model = _build_custom_groups_model(
        output_heads=_custom_groups_output_heads(
            groups={
                "electron_density": {"targets": ["electron_density"]},
                "ion_density": {"targets": ["ion_density"]},
                "thermal": {"targets": ["electron_temperature"]},
                "electrostatic": {"targets": ["plasma_potential"]},
            },
        )
        | {
            "group_options": {
                "electron_density": {"head": "spatial_refine"},
                "ion_density": {"head": "spatial_refine"},
            }
        }
    )
    pred = model.forward(np.zeros((2, 3), dtype=np.float32))
    ckpt = save_checkpoint(model, tmp_path / "custom_groups_spatial_refine")
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))

    assert pred.shape == (2, 4, 8, 8)
    assert model.output_keys == _logical_output_keys()
    assert model.output_head_group_options == {
        "electron_density": {"head": "spatial_refine"},
        "ion_density": {"head": "spatial_refine"},
    }
    assert meta["output_heads"]["group_options"] == model.output_head_group_options


def test_custom_groups_output_head_rejects_duplicate_target() -> None:
    with pytest.raises(ValueError, match="already assigned"):
        _build_custom_groups_model(
            output_heads=_custom_groups_output_heads(
                groups={
                    "density": {"targets": ["electron_density"]},
                    "again": {"targets": ["electron_density"]},
                    "thermal": {"targets": ["electron_temperature"]},
                    "electrostatic": {"targets": ["plasma_potential"]},
                }
            )
        )


def test_custom_groups_output_head_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="unknown targets"):
        _build_custom_groups_model(
            output_heads=_custom_groups_output_heads(
                groups={
                    "density": {"targets": ["electron_density", "missing_density"]},
                    "thermal": {"targets": ["electron_temperature"]},
                    "electrostatic": {"targets": ["plasma_potential"]},
                }
            )
        )


def test_custom_groups_output_head_rejects_missing_target_when_strict() -> None:
    with pytest.raises(ValueError, match="missing output_vars"):
        _build_custom_groups_model(
            output_heads=_custom_groups_output_heads(
                groups={
                    "density": {"targets": ["electron_density", "ion_density"]},
                    "thermal": {"targets": ["electron_temperature"]},
                }
            )
        )


def test_custom_groups_output_head_strict_false_defaults_missing_targets() -> None:
    model = _build_custom_groups_model(
        output_heads=_custom_groups_output_heads(
            strict=False,
            groups={
                "density": {"targets": ["electron_density", "ion_density"]},
                "thermal": {"targets": ["electron_temperature"]},
            },
        )
    )

    assert list(model.target_groups) == ["density", "thermal", "default"]
    assert model.target_groups["default"].targets == ("plasma_potential",)
    assert model.target_groups["default"].field_family is None


def test_custom_groups_output_head_rejects_empty_group() -> None:
    with pytest.raises(ValueError, match="must contain target ids"):
        _build_custom_groups_model(
            output_heads=_custom_groups_output_heads(
                groups={
                    "density": {"targets": ["electron_density", "ion_density"]},
                    "empty": {"targets": []},
                    "thermal": {"targets": ["electron_temperature"]},
                    "electrostatic": {"targets": ["plasma_potential"]},
                }
            )
        )


def test_role_grouped_output_head_rejects_unknown_custom_target() -> None:
    require_torch_runtime()
    cfg = _model_cfg("fno")
    cfg["output_heads"] = {"mode": "role_grouped", "custom_groups": {"bad": ["missing"]}}
    with pytest.raises(ValueError, match="unknown targets"):
        build_model_from_name(
            model_name="fno",
            input_dim=3,
            grid_shape=(8, 8),
            model_cfg=cfg,
            out_channels=4,
            output_keys=["temperature_main", "density_b", "potential", "density_a"],
            unet_feature_channels=["x", "y"],
        )


def test_role_grouped_group_options_default_is_metadata_only(tmp_path: Path) -> None:
    require_torch_runtime()
    cfg = _model_cfg("fno")
    cfg["output_heads"] = {
        "mode": "role_grouped",
        "group_options": {"electrostatic": {"head": "default"}},
    }
    output_keys = ["temperature_main", "density_b", "potential", "density_a"]
    model = build_model_from_name(
        model_name="fno",
        input_dim=3,
        grid_shape=(8, 8),
        model_cfg=cfg,
        out_channels=4,
        output_keys=output_keys,
        unet_feature_channels=["x", "y"],
    )

    pred = model.forward(np.zeros((1, 3), dtype=np.float32))
    ckpt = save_checkpoint(model, tmp_path / "role_grouped_default_group_options")
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))

    assert pred.shape == (1, 4, 8, 8)
    assert model.output_head_group_options == {"electrostatic": {"head": "default"}}
    assert meta["output_heads"]["group_options"] == {"electrostatic": {"head": "default"}}


def test_role_grouped_group_options_reject_poisson_hybrid_until_implemented() -> None:
    require_torch_runtime()
    cfg = _model_cfg("fno")
    cfg["output_heads"] = {
        "mode": "role_grouped",
        "group_options": {"electrostatic": {"head": "poisson_hybrid"}},
    }
    with pytest.raises(ValueError, match="not implemented yet"):
        build_model_from_name(
            model_name="fno",
            input_dim=3,
            grid_shape=(8, 8),
            model_cfg=cfg,
            out_channels=4,
            output_keys=["temperature_main", "density_b", "potential", "density_a"],
            unet_feature_channels=["x", "y"],
        )


def test_role_grouped_group_options_reject_unknown_group() -> None:
    require_torch_runtime()
    cfg = _model_cfg("fno")
    cfg["output_heads"] = {
        "mode": "role_grouped",
        "group_options": {"not_a_group": {"head": "default"}},
    }
    with pytest.raises(ValueError, match="does not match a resolved target group"):
        build_model_from_name(
            model_name="fno",
            input_dim=3,
            grid_shape=(8, 8),
            model_cfg=cfg,
            out_channels=4,
            output_keys=["temperature_main", "density_b", "potential", "density_a"],
            unet_feature_channels=["x", "y"],
        )


def test_role_grouped_group_options_reject_unsupported_keys() -> None:
    require_torch_runtime()
    cfg = _model_cfg("fno")
    cfg["output_heads"] = {
        "mode": "role_grouped",
        "group_options": {"electrostatic": {"head": "default", "alpha": 0.5}},
    }
    with pytest.raises(ValueError, match="unsupported keys"):
        build_model_from_name(
            model_name="fno",
            input_dim=3,
            grid_shape=(8, 8),
            model_cfg=cfg,
            out_channels=4,
            output_keys=["temperature_main", "density_b", "potential", "density_a"],
            unet_feature_channels=["x", "y"],
        )


def test_shared_output_head_rejects_group_options() -> None:
    require_torch_runtime()
    cfg = _model_cfg("fno")
    cfg["output_heads"] = {
        "mode": "shared",
        "group_options": {"electrostatic": {"head": "default"}},
    }
    with pytest.raises(ValueError, match="group_options requires"):
        build_model_from_name(
            model_name="fno",
            input_dim=3,
            grid_shape=(8, 8),
            model_cfg=cfg,
            out_channels=4,
            output_keys=["temperature_main", "density_b", "potential", "density_a"],
            unet_feature_channels=["x", "y"],
        )


def test_role_grouped_output_head_rejects_unsupported_model_family() -> None:
    with pytest.raises(ValueError, match="supported only"):
        build_model_from_name(
            model_name="global_mlp",
            input_dim=3,
            grid_shape=(8, 8),
            model_cfg={"output_heads": {"mode": "role_grouped"}},
            out_channels=4,
            output_keys=["temperature_main", "density_b", "potential", "density_a"],
        )


def test_role_grouped_checkpoint_preserves_head_metadata(tmp_path: Path) -> None:
    model = _build_role_grouped_model("fno")
    ckpt = save_checkpoint(model, tmp_path / "role_grouped_fno")
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))

    assert meta["output_heads_mode_effective"] == "role_grouped"
    assert meta["output_heads"]["mode"] == "role_grouped"
    assert "target_groups_hash" not in meta
    assert [group["name"] for group in meta["target_groups"]] == ["temperature", "density", "electrostatic"]

    loaded = load_checkpoint(ckpt)
    pred = loaded.forward(np.zeros((1, 3), dtype=np.float32))
    assert pred.shape == (1, 4, 8, 8)
    assert loaded.output_heads_mode == "role_grouped"
    assert loaded.target_groups_metadata == model.target_groups_metadata


def test_custom_groups_checkpoint_preserves_effective_config(tmp_path: Path) -> None:
    model = _build_custom_groups_model()
    ckpt = save_checkpoint(model, tmp_path / "custom_groups_fno")
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))

    assert meta["output_heads_mode_effective"] == "custom_groups"
    assert meta["output_heads"]["mode"] == "custom_groups"
    assert meta["output_heads"]["strict"] is True
    assert meta["output_heads"]["groups"]["density"]["targets"] == ["ion_density", "electron_density"]
    assert "target_groups_hash" not in meta
    assert [group["name"] for group in meta["target_groups"]] == ["density", "thermal", "electrostatic"]

    loaded = load_checkpoint(ckpt)
    pred = loaded.forward(np.zeros((1, 3), dtype=np.float32))
    assert pred.shape == (1, 4, 8, 8)
    assert loaded.output_heads_mode == "custom_groups"
    assert loaded.target_groups["density"].targets == ("electron_density", "ion_density")


def test_inference_custom_groups_metadata_rejects_runtime_schema_mismatch(tmp_path: Path) -> None:
    model = _build_custom_groups_model()
    ckpt = save_checkpoint(model, tmp_path / "custom_groups_fno")
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))

    validate_output_head_metadata_contract(
        model=model,
        checkpoint_meta=meta,
        target_role_schema=_logical_target_role_schema(),
    )

    bad_schema = _logical_target_role_schema()
    bad_schema["targets"] = [
        {**entry, "role": "density_total"} if entry["id"] == "electron_density" else entry
        for entry in bad_schema["targets"]
    ]
    with pytest.raises(ValueError, match="target_groups differ"):
        validate_output_head_metadata_contract(
            model=model,
            checkpoint_meta=meta,
            target_role_schema=bad_schema,
        )


def test_inference_output_head_metadata_uses_checkpoint_groups_without_runtime_schema(tmp_path: Path) -> None:
    model = _build_role_grouped_model("fno")
    ckpt = save_checkpoint(model, tmp_path / "role_grouped_fno")
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))

    validate_output_head_metadata_contract(
        model=model,
        checkpoint_meta=meta,
        target_role_schema={},
    )


def test_inference_output_head_metadata_rejects_runtime_schema_mismatch(tmp_path: Path) -> None:
    model = _build_role_grouped_model("fno")
    ckpt = save_checkpoint(model, tmp_path / "role_grouped_fno")
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))

    validate_output_head_metadata_contract(
        model=model,
        checkpoint_meta=meta,
        target_role_schema=_target_role_schema(),
    )

    bad_schema = _target_role_schema()
    bad_schema["targets"] = [
        {**entry, "field_family": "temperature"} if entry["id"] == "density_a" else entry
        for entry in bad_schema["targets"]
    ]
    with pytest.raises(ValueError, match="target_groups differ"):
        validate_output_head_metadata_contract(
            model=model,
            checkpoint_meta=meta,
            target_role_schema=bad_schema,
        )
