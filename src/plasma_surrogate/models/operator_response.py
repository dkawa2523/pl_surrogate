"""Shared axisymmetric response-scale adapter for grid neural operators."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_operator_response_cfg(raw: dict[str, Any] | None, *, cfg_prefix: str) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    imbalance = float(cfg.get("density_imbalance_fraction", 0.05))
    radial_padding = str(cfg.get("radial_padding_mode", "axisymmetric_neumann")).strip().lower()
    native_scale = int(cfg.get("native_scale", 4))
    pad_cells = int(cfg.get("padding_cells", 2))
    if not np.isfinite(imbalance) or not 0.0 <= imbalance < 1.0:
        raise ValueError(f"{cfg_prefix}.operator_response_cfg.density_imbalance_fraction must be in [0,1)")
    if radial_padding not in {"zero", "axisymmetric_neumann"}:
        raise ValueError(f"{cfg_prefix}.operator_response_cfg.radial_padding_mode must be axisymmetric_neumann or zero")
    if native_scale not in {1, 2, 4}:
        raise ValueError(f"{cfg_prefix}.operator_response_cfg.native_scale must be one of 1, 2, 4")
    if pad_cells < 0:
        raise ValueError(f"{cfg_prefix}.operator_response_cfg.padding_cells must be >= 0")
    affine = dict(cfg.get("target_affine", {}) or {})
    if enabled:
        for name in ("ne", "ni"):
            item = dict(affine.get(name, {}) or {})
            mean, scale = float(item.get("mean", np.nan)), float(item.get("scale", np.nan))
            if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"{cfg_prefix}.operator_response_cfg.target_affine.{name} is invalid")
    return {
        "enabled": enabled,
        "density_imbalance_fraction": imbalance,
        "radial_padding_mode": radial_padding,
        "native_scale": native_scale,
        "padding_cells": pad_cells,
        "target_affine": affine,
    }


def apply_operator_response_adapter(model: Any, cfg: dict[str, Any] | None) -> None:
    """Wrap ``model.net`` with boundary handling and the physical density head."""
    normalized = normalize_operator_response_cfg(cfg, cfg_prefix=str(model._config_prefix))
    model.operator_response_cfg = normalized
    if not normalized["enabled"]:
        return
    if model.with_rho_eff_head or set(model.output_keys) != {"ne", "ni", "Te", "phi"}:
        raise ValueError("operator response-scale requires exactly ne, ni, Te, phi and no rho_eff head")
    if str(model.output_heads_mode) != "shared":
        raise ValueError("operator response-scale requires output_heads.mode=shared")
    if "mask_plasma" not in model.input_feature_channels:
        raise ValueError("operator response-scale requires mask_plasma input feature")

    torch, nn = model.torch, model.torch.nn
    backbone = model.net
    output_keys = list(model.output_keys)
    mask_index = model.input_dim + model.input_feature_channels.index("mask_plasma")
    affine = normalized["target_affine"]
    means = [float(affine[k]["mean"]) for k in ("ne", "ni")]
    scales = [float(affine[k]["scale"]) for k in ("ne", "ni")]

    class _OperatorResponseAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.imbalance_fraction = float(normalized["density_imbalance_fraction"])
            self.native_scale = int(normalized["native_scale"])
            self.pad_cells = int(normalized["padding_cells"])
            self.radial_padding_mode = str(normalized["radial_padding_mode"])
            self.register_buffer("density_means", torch.tensor(means).reshape(1, 2, 1, 1))
            self.register_buffer("density_scales", torch.tensor(scales).reshape(1, 2, 1, 1))
            self.register_buffer("density_reference_scale", torch.tensor(float(np.mean(scales))))

        def _pad(self, value):
            p = self.pad_cells
            if p == 0:
                return value
            left = value[..., :p].flip(-1) if self.radial_padding_mode == "axisymmetric_neumann" else value.new_zeros((*value.shape[:-1], p))
            value = torch.cat((left, value), dim=-1)
            return nn.functional.pad(value, (0, p, p, p), mode="constant", value=0.0)

        def forward(self, value):
            full_size = value.shape[-2:]
            if self.native_scale > 1:
                native_size = (
                    max(1, (int(full_size[0]) + self.native_scale - 1) // self.native_scale),
                    max(1, (int(full_size[1]) + self.native_scale - 1) // self.native_scale),
                )
                # Area resampling is an anti-aliased cell average.  It preserves
                # scalar conditioning, signed-distance trends, and occupancy
                # fractions while matching the adopted U-Net's coarse response grid.
                operator_input = nn.functional.interpolate(value, size=native_size, mode="area")
            else:
                operator_input = value
            operator_size = operator_input.shape[-2:]
            raw = self.backbone(self._pad(operator_input))
            p = self.pad_cells
            if p:
                raw = raw[..., p : p + operator_size[0], p : p + operator_size[1]]
            mask = operator_input[:, mask_index : mask_index + 1].clamp(0, 1)
            shape_logits = raw[:, output_keys.index("ne") : output_keys.index("ne") + 1]
            imbalance_logits = raw[:, output_keys.index("ni") : output_keys.index("ni") + 1]
            positive_shape = nn.functional.softplus(shape_logits) + 1.0e-6
            denom = torch.sum(mask, dim=(2, 3), keepdim=True).clamp_min(1.0)
            normalizer = torch.sum(positive_shape * mask, dim=(2, 3), keepdim=True) / denom
            shape = positive_shape / normalizer.clamp_min(1.0e-6)
            amplitude = nn.functional.softplus(torch.sum(shape_logits * mask, dim=(2, 3), keepdim=True) / denom) + 1.0e-6
            common = self.density_reference_scale * amplitude * shape
            delta = self.imbalance_fraction * torch.tanh(imbalance_logits)
            physical = torch.cat((common * (1 - delta), common * (1 + delta)), dim=1)
            standard = (physical - self.density_means) / self.density_scales
            out = raw.clone()
            out[:, output_keys.index("ne") : output_keys.index("ne") + 1] = standard[:, :1]
            out[:, output_keys.index("ni") : output_keys.index("ni") + 1] = standard[:, 1:2]
            if out.shape[-2:] != full_size:
                out = nn.functional.interpolate(out, size=full_size, mode="bilinear", align_corners=False)
            return out

        def step_reference(self):
            return self.backbone.in_proj.weight, self.backbone.post[-1].weight

    model.net = _OperatorResponseAdapter()
    model.head_arch_version = "operator_response_scale_quasineutral_v1"
