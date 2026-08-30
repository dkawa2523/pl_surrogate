"""Isolated experimental UNO variants for the ICP structure/EM v46 study.

This module intentionally lives outside ``src/``.  The launcher monkey-patches
the model factory only for the lifetime of an experimental process, so the
adopted conference implementation and its checkpoints remain byte-for-byte
untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_surrogate.models.operator_response import apply_operator_response_adapter
from plasma_surrogate.models.uno.simple_uno import (
    UNOBaseline as _FormalUNOBaseline,
    normalize_uno_cfg as _normalize_formal_uno_cfg,
)


def normalize_experimental_uno_cfg(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the formal fields and retain only the isolated v46 controls."""

    raw = dict(raw_cfg or {})
    out = dict(_normalize_formal_uno_cfg(raw))
    out.update(
        {
            "experimental_variant": str(raw.get("experimental_variant", "A_multires")),
            "multiresolution": bool(raw.get("multiresolution", True)),
            "separate_lifting": bool(raw.get("separate_lifting", False)),
            "adaptive_local_global": bool(raw.get("adaptive_local_global", False)),
            "permutation_invariant_coils": bool(raw.get("permutation_invariant_coils", False)),
            "em_reparameterization": bool(raw.get("em_reparameterization", False)),
            "structure_delta_weight": float(raw.get("structure_delta_weight", 0.0)),
            "sobolev_weight": float(raw.get("sobolev_weight", 0.0)),
            "multiscale_weight": float(raw.get("multiscale_weight", 0.0)),
            "em_reconstruction_weight": float(raw.get("em_reconstruction_weight", 0.0)),
            "target_specific_decoders": bool(raw.get("target_specific_decoders", False)),
            "em_asinh_scale": float(raw.get("em_asinh_scale", 1.0)),
        }
    )
    for name in ("structure_delta_weight", "sobolev_weight", "multiscale_weight", "em_reconstruction_weight"):
        if not np.isfinite(out[name]) or float(out[name]) < 0.0:
            raise ValueError(f"experimental UNO {name} must be finite and >= 0")
    if not np.isfinite(out["em_asinh_scale"]) or float(out["em_asinh_scale"]) <= 0.0:
        raise ValueError("experimental UNO em_asinh_scale must be finite and > 0")
    return out


class ExperimentalUNOBaseline(_FormalUNOBaseline):
    """UNO-only test bed for generic structure and electromagnetic upgrades."""

    def __init__(
        self,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int = 3,
        output_keys: list[str] | None = None,
        seed: int = 0,
        with_rho_eff_head: bool = False,
        n_modes: int = 12,
        head_mlp: dict[str, object] | None = None,
        input_feature_channels: list[str] | None = None,
        uno_cfg: dict[str, Any] | None = None,
        backend: str = "torch",
        output_heads: dict[str, Any] | None = None,
        target_role_schema: dict[str, Any] | None = None,
        operator_response_cfg: dict[str, Any] | None = None,
    ) -> None:
        exp_cfg = normalize_experimental_uno_cfg(uno_cfg)
        # Build the adopted object first so all public training/checkpoint
        # contracts stay identical.  Its network is immediately replaced.
        super().__init__(
            input_dim=input_dim,
            grid_shape=grid_shape,
            out_channels=out_channels,
            output_keys=output_keys,
            seed=seed,
            with_rho_eff_head=with_rho_eff_head,
            n_modes=n_modes,
            head_mlp=head_mlp,
            input_feature_channels=input_feature_channels,
            uno_cfg=exp_cfg,
            backend=backend,
            output_heads=output_heads,
            target_role_schema=target_role_schema,
            operator_response_cfg=operator_response_cfg,
        )
        self.uno_cfg = exp_cfg
        self.experimental_variant = str(exp_cfg["experimental_variant"])
        self.fno_impl_version = "isolated_icp_uno_structure_em_v46"

        torch = self.torch
        nn = torch.nn
        functional = torch.nn.functional
        width = int(exp_cfg["width"])
        n_layers = max(int(exp_cfg["n_layers"]), 3)
        dropout = float(exp_cfg["dropout"])
        n_modes_i = int(self.n_modes)
        input_labels = [f"condition_{idx}" for idx in range(int(self.input_dim))] + list(
            self.input_feature_channels
        )
        cond_indices = list(range(int(self.input_dim)))
        spatial_offset = int(self.input_dim)

        def _indices(names: set[str]) -> list[int]:
            return [spatial_offset + idx for idx, name in enumerate(self.input_feature_channels) if name in names]

        static_names = {"x", "y", "mask_plasma", "distance_signed", "distance_any"}
        em_names = {"vacuum_aphi_unit", "vacuum_br_unit", "vacuum_bz_unit", "vacuum_bmag_unit"}
        per_coil_names = {f"sdf_coil_{slot:02d}" for slot in range(1, 7)}
        static_indices = _indices(static_names)
        em_indices = _indices(em_names)
        per_coil_indices = _indices(per_coil_names)
        geometry_indices = [
            spatial_offset + idx
            for idx, name in enumerate(self.input_feature_channels)
            if name not in static_names and name not in em_names and name not in per_coil_names
        ]

        class _LowFreqMix2d(nn.Module):
            def __init__(self, channels: int, modes: int) -> None:
                super().__init__()
                self.channels = int(channels)
                self.modes = int(modes)
                scale = 1.0 / max(self.channels, 1)
                self.weight_pos = nn.Parameter(
                    scale * torch.randn(self.channels, self.channels, self.modes, self.modes, 2)
                )
                self.weight_neg = nn.Parameter(
                    scale * torch.randn(self.channels, self.channels, self.modes, self.modes, 2)
                )

            @staticmethod
            def _multiply(x_ft, weight):
                return torch.einsum("bixy,ioxy->boxy", x_ft, weight)

            def forward(self, value):
                batch, _, height, width_px = value.shape
                value_ft = torch.fft.rfft2(value, norm="ortho")
                out_ft = torch.zeros(
                    (batch, self.channels, height, (width_px // 2) + 1),
                    dtype=torch.cfloat,
                    device=value.device,
                )
                mode_h = min(self.modes, height)
                mode_w = min(self.modes, (width_px // 2) + 1)
                pos = torch.view_as_complex(self.weight_pos[:, :, :mode_h, :mode_w].contiguous())
                neg = torch.view_as_complex(self.weight_neg[:, :, :mode_h, :mode_w].contiguous())
                out_ft[:, :, :mode_h, :mode_w] = self._multiply(value_ft[:, :, :mode_h, :mode_w], pos)
                out_ft[:, :, -mode_h:, :mode_w] = self._multiply(value_ft[:, :, -mode_h:, :mode_w], neg)
                return torch.fft.irfft2(out_ft, s=(height, width_px), norm="ortho")

        class _HybridUNOBlock(nn.Module):
            def __init__(self, channels: int, modes: int, use_gate: bool) -> None:
                super().__init__()
                self.global_mix = _LowFreqMix2d(channels, modes)
                self.local_mix = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                )
                self.skip = nn.Conv2d(channels, channels, 1)
                self.norm = nn.GroupNorm(1, channels)
                self.drop = nn.Dropout(dropout)
                self.gate = nn.Conv2d(channels * 3, channels, 1) if use_gate else None

            def forward(self, value):
                global_value = self.global_mix(value)
                local_value = self.local_mix(value)
                if self.gate is None:
                    mixed = global_value + local_value
                else:
                    gate = torch.sigmoid(self.gate(torch.cat((value, global_value, local_value), dim=1)))
                    mixed = gate * global_value + (1.0 - gate) * local_value
                return functional.gelu(self.drop(self.norm(mixed)) + self.skip(value))

        class _SharedCoilEncoder(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, channels, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                )

            def forward(self, value):
                batch, slots, height, width_px = value.shape
                flat = value.reshape(batch * slots, 1, height, width_px)
                encoded = self.net(flat).reshape(batch, slots, -1, height, width_px)
                # Empty slots have no negative signed-distance interior.  Mean
                # aggregation makes the representation permutation invariant
                # while avoiding a magnitude jump solely from coil count.
                active = (value.amin(dim=(-2, -1), keepdim=True) < 0.0).to(value.dtype)
                denom = active.sum(dim=1, keepdim=True).clamp_min(1.0)
                return (encoded * active.unsqueeze(2)).sum(dim=1) / denom.squeeze(1).unsqueeze(1)

        class _ExperimentalBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.multiresolution = bool(exp_cfg["multiresolution"])
                self.separate_lifting = bool(exp_cfg["separate_lifting"])
                self.use_per_coil = bool(exp_cfg["permutation_invariant_coils"] and per_coil_indices)
                self.use_em_reparam = bool(exp_cfg["em_reparameterization"] and em_indices)
                self.use_target_decoders = bool(exp_cfg["target_specific_decoders"])
                self.em_scale = float(exp_cfg["em_asinh_scale"])
                self.last_aux_pred = None
                self.last_aux_target = None
                self.last_condition = None

                if self.separate_lifting:
                    group_specs: list[tuple[str, list[int], int]] = []
                    if cond_indices:
                        group_specs.append(("condition", cond_indices, len(cond_indices)))
                    if static_indices:
                        group_specs.append(("static", static_indices, len(static_indices)))
                    if geometry_indices:
                        group_specs.append(("geometry", geometry_indices, len(geometry_indices)))
                    if em_indices:
                        em_in = len(em_indices) + (1 if self.use_em_reparam else 0)
                        group_specs.append(("electromagnetic", em_indices, em_in))
                    if per_coil_indices and not self.use_per_coil:
                        group_specs.append(("coil_slots", per_coil_indices, len(per_coil_indices)))
                    self.group_specs = group_specs
                    self.group_lifts = nn.ModuleDict(
                        {name: nn.Conv2d(in_count, width, 1) for name, _, in_count in group_specs}
                    )
                    self.coil_encoder = _SharedCoilEncoder(width) if self.use_per_coil else None
                    fused_groups = len(group_specs) + (1 if self.use_per_coil else 0)
                    self.in_proj = nn.Conv2d(width * fused_groups, width, 1)
                    self.condition_film = (
                        nn.ModuleList([nn.Linear(len(cond_indices), 2 * width) for _ in range(3)])
                        if cond_indices
                        else None
                    )
                else:
                    self.group_specs = []
                    self.group_lifts = nn.ModuleDict()
                    self.coil_encoder = None
                    self.in_proj = nn.Conv2d(len(input_labels), width, 1)
                    self.condition_film = None

                use_gate = bool(exp_cfg["adaptive_local_global"])
                if self.multiresolution:
                    enc_count = max(1, n_layers // 3)
                    low_count = max(1, n_layers // 3)
                    bottleneck_count = max(1, n_layers - enc_count - low_count)
                    self.enc_blocks = nn.ModuleList(
                        [_HybridUNOBlock(width, max(4, n_modes_i), use_gate) for _ in range(enc_count)]
                    )
                    self.low_blocks = nn.ModuleList(
                        [_HybridUNOBlock(width, max(4, n_modes_i), use_gate) for _ in range(low_count)]
                    )
                    self.bottleneck_blocks = nn.ModuleList(
                        [_HybridUNOBlock(width, max(4, n_modes_i), use_gate) for _ in range(bottleneck_count)]
                    )
                    self.up_low = nn.Conv2d(2 * width, width, 1)
                    self.up_high = nn.Conv2d(2 * width, width, 1)
                    self.decode_low = _HybridUNOBlock(width, max(4, n_modes_i), use_gate)
                    self.decode_high = _HybridUNOBlock(width, max(4, n_modes_i), use_gate)
                    self.blocks = nn.ModuleList()
                else:
                    self.blocks = nn.ModuleList(
                        [_HybridUNOBlock(width, max(4, n_modes_i), use_gate) for _ in range(n_layers)]
                    )
                    self.enc_blocks = nn.ModuleList()
                    self.low_blocks = nn.ModuleList()
                    self.bottleneck_blocks = nn.ModuleList()

                if self.use_target_decoders:
                    self.post = nn.Sequential(
                        nn.Conv2d(width, width, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv2d(width, width, 1)
                    )
                    self.target_heads = nn.ModuleDict(
                        {
                            "density": nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, 2, 1)),
                            "temperature": nn.Sequential(nn.Conv2d(width, width // 2, 3, padding=1), nn.GELU(), nn.Conv2d(width // 2, 1, 1)),
                            "potential": nn.Sequential(nn.Conv2d(width, width // 2, 3, padding=1), nn.GELU(), nn.Conv2d(width // 2, 1, 1)),
                        }
                    )
                else:
                    self.post = nn.Sequential(
                        nn.Conv2d(width, width, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv2d(width, out_channels, 1)
                    )
                    self.target_heads = nn.ModuleDict()

                self.aux_em_head = (
                    nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, len(em_indices), 1))
                    if float(exp_cfg["em_reconstruction_weight"]) > 0.0 and em_indices
                    else None
                )

            @staticmethod
            def _select(value, indices: list[int]):
                return value[:, indices]

            def _em_features(self, value):
                em = self._select(value, em_indices)
                if not self.use_em_reparam:
                    return em
                amplitude = torch.sqrt(torch.mean(em.square(), dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-8))
                shape = torch.asinh(em / (self.em_scale * amplitude + 1.0e-4))
                amplitude_map = torch.log1p(amplitude).expand(-1, 1, em.shape[-2], em.shape[-1])
                return torch.cat((shape, amplitude_map), dim=1)

            def _lift(self, value):
                self.last_condition = value[:, cond_indices, 0, 0] if cond_indices else None
                if not self.separate_lifting:
                    return self.in_proj(value), None
                lifted = []
                geometry_latent = None
                for name, indices, _ in self.group_specs:
                    group_value = self._em_features(value) if name == "electromagnetic" else self._select(value, indices)
                    encoded = self.group_lifts[name](group_value)
                    lifted.append(encoded)
                    if name == "geometry":
                        geometry_latent = encoded
                if self.use_per_coil and self.coil_encoder is not None:
                    coil_latent = self.coil_encoder(self._select(value, per_coil_indices))
                    lifted.append(coil_latent)
                    geometry_latent = coil_latent if geometry_latent is None else geometry_latent + coil_latent
                return self.in_proj(torch.cat(lifted, dim=1)), geometry_latent

            def _film(self, value, level: int):
                if self.condition_film is None or self.last_condition is None:
                    return value
                gamma, beta = self.condition_film[level](self.last_condition).chunk(2, dim=1)
                return value * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None]) + 0.1 * beta[:, :, None, None]

            @staticmethod
            def _run_blocks(value, blocks):
                for block in blocks:
                    value = block(value)
                return value

            def _decode_targets(self, feature):
                feature = self.post(feature)
                if not self.use_target_decoders:
                    return feature
                density = self.target_heads["density"](feature)
                temperature = self.target_heads["temperature"](feature)
                potential = self.target_heads["potential"](feature)
                by_name = {"ne": density[:, :1], "ni": density[:, 1:2], "Te": temperature, "phi": potential}
                return torch.cat([by_name[name] for name in self_output_keys], dim=1)

            def forward(self, value):
                high, geometry_latent = self._lift(value)
                if self.multiresolution:
                    high = self._run_blocks(self._film(high, 0), self.enc_blocks)
                    low = functional.avg_pool2d(high, 2, 2, ceil_mode=True)
                    low = self._run_blocks(self._film(low, 1), self.low_blocks)
                    bottleneck = functional.avg_pool2d(low, 2, 2, ceil_mode=True)
                    bottleneck = self._run_blocks(self._film(bottleneck, 2), self.bottleneck_blocks)
                    decoded_low = functional.interpolate(bottleneck, size=low.shape[-2:], mode="bilinear", align_corners=False)
                    decoded_low = self.decode_low(self.up_low(torch.cat((decoded_low, low), dim=1)))
                    decoded_high = functional.interpolate(decoded_low, size=high.shape[-2:], mode="bilinear", align_corners=False)
                    feature = self.decode_high(self.up_high(torch.cat((decoded_high, high), dim=1)))
                else:
                    feature = self._run_blocks(high, self.blocks)
                if self.aux_em_head is not None and geometry_latent is not None:
                    self.last_aux_pred = self.aux_em_head(geometry_latent)
                    self.last_aux_target = self._select(value, em_indices)
                else:
                    self.last_aux_pred = None
                    self.last_aux_target = None
                return self._decode_targets(feature)

        self_output_keys = list(self.output_keys)
        self.net = _ExperimentalBackbone()
        # Reapply the same adopted response-scale density contract to the new
        # UNO backbone.  This preserves the target interpretation and makes the
        # architecture/loss comparison controlled.
        apply_operator_response_adapter(self, operator_response_cfg)
        self.head_arch_version = (
            "operator_response_scale_quasineutral_v1+target_specific_v46"
            if bool(exp_cfg["target_specific_decoders"])
            else "operator_response_scale_quasineutral_v1+experimental_uno_v46"
        )
        if (
            float(exp_cfg["structure_delta_weight"]) <= 0.0
            and float(exp_cfg["sobolev_weight"]) <= 0.0
            and float(exp_cfg["multiscale_weight"]) <= 0.0
            and float(exp_cfg["em_reconstruction_weight"]) <= 0.0
        ):
            # The trainer uses method presence as the auxiliary-validation gate.
            self.evaluate_auxiliary_losses = None  # type: ignore[assignment]

    def _experimental_backbone(self):
        return self.net.backbone if hasattr(self.net, "backbone") else self.net

    def _auxiliary_tensor(
        self,
        target_raw: np.ndarray | None,
        supervised_mask: np.ndarray | None,
    ):
        if target_raw is None or self._torch_last_out is None:
            return None, {}
        torch = self.torch
        backbone = self._experimental_backbone()
        total = None
        diagnostics: dict[str, float] = {}
        delta_weight = float(self.uno_cfg.get("structure_delta_weight", 0.0))
        if delta_weight > 0.0 and int(self._torch_last_out.shape[0]) > 1:
            cond = getattr(backbone, "last_condition", None)
            if cond is not None:
                with torch.no_grad():
                    distances = torch.cdist(cond, cond)
                    distances.fill_diagonal_(float("inf"))
                    partner = torch.argmin(distances, dim=1)
                target = torch.nan_to_num(
                    torch.as_tensor(np.asarray(target_raw, dtype=np.float32), device=self.device)
                )
                pred_delta = self._torch_last_out - self._torch_last_out[partner]
                target_delta = target - target[partner]
                element = torch.nn.functional.smooth_l1_loss(pred_delta, target_delta, reduction="none", beta=1.0)
                if supervised_mask is not None:
                    mask = torch.as_tensor(np.asarray(supervised_mask, dtype=np.float32), device=self.device)
                    if mask.ndim == 3:
                        mask = mask[:, None]
                    pair_mask = (mask * mask[partner]).expand_as(element)
                    delta_loss = torch.sum(element * pair_mask) / pair_mask.sum().clamp_min(1.0)
                else:
                    delta_loss = element.mean()
                weighted = delta_weight * delta_loss
                total = weighted if total is None else total + weighted
                diagnostics["loss_aux_structure_delta"] = float(weighted.detach().cpu())

        target = torch.nan_to_num(torch.as_tensor(np.asarray(target_raw, dtype=np.float32), device=self.device))
        prediction = self._torch_last_out
        mask_tensor = None
        if supervised_mask is not None:
            mask_tensor = torch.as_tensor(np.asarray(supervised_mask, dtype=np.float32), device=self.device)
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor[:, None]
        sobolev_weight = float(self.uno_cfg.get("sobolev_weight", 0.0))
        if sobolev_weight > 0.0:
            residual = prediction - target
            grad_r = residual[..., 1:] - residual[..., :-1]
            grad_z = residual[..., 1:, :] - residual[..., :-1, :]
            if mask_tensor is not None:
                mask_r = (mask_tensor[..., 1:] * mask_tensor[..., :-1]).expand_as(grad_r)
                mask_z = (mask_tensor[..., 1:, :] * mask_tensor[..., :-1, :]).expand_as(grad_z)
                grad_loss = (
                    torch.sum(torch.nn.functional.smooth_l1_loss(grad_r, torch.zeros_like(grad_r), reduction="none") * mask_r)
                    / mask_r.sum().clamp_min(1.0)
                    + torch.sum(torch.nn.functional.smooth_l1_loss(grad_z, torch.zeros_like(grad_z), reduction="none") * mask_z)
                    / mask_z.sum().clamp_min(1.0)
                ) * 0.5
            else:
                grad_loss = 0.5 * (
                    torch.nn.functional.smooth_l1_loss(grad_r, torch.zeros_like(grad_r))
                    + torch.nn.functional.smooth_l1_loss(grad_z, torch.zeros_like(grad_z))
                )
            weighted = sobolev_weight * grad_loss
            total = weighted if total is None else total + weighted
            diagnostics["loss_aux_sobolev"] = float(weighted.detach().cpu())

        multiscale_weight = float(self.uno_cfg.get("multiscale_weight", 0.0))
        if multiscale_weight > 0.0:
            scale_loss = prediction.new_zeros(())
            for scale in (2, 4):
                pred_s = torch.nn.functional.avg_pool2d(prediction, scale, scale)
                target_s = torch.nn.functional.avg_pool2d(target, scale, scale)
                element = torch.nn.functional.smooth_l1_loss(pred_s, target_s, reduction="none")
                if mask_tensor is not None:
                    mask_s = torch.nn.functional.avg_pool2d(mask_tensor, scale, scale).expand_as(element)
                    scale_loss = scale_loss + torch.sum(element * mask_s) / mask_s.sum().clamp_min(1.0)
                else:
                    scale_loss = scale_loss + element.mean()
            scale_loss = scale_loss / 2.0
            weighted = multiscale_weight * scale_loss
            total = weighted if total is None else total + weighted
            diagnostics["loss_aux_multiscale"] = float(weighted.detach().cpu())

        em_weight = float(self.uno_cfg.get("em_reconstruction_weight", 0.0))
        aux_pred = getattr(backbone, "last_aux_pred", None)
        aux_target = getattr(backbone, "last_aux_target", None)
        if em_weight > 0.0 and aux_pred is not None and aux_target is not None:
            em_loss = torch.nn.functional.smooth_l1_loss(aux_pred, aux_target, reduction="mean", beta=1.0)
            weighted = em_weight * em_loss
            total = weighted if total is None else total + weighted
            diagnostics["loss_aux_em_reconstruction"] = float(weighted.detach().cpu())
        return total, diagnostics

    def backward_raw(
        self,
        grad_raw: np.ndarray,
        *,
        lr: float,
        weight_decay: float = 0.0,
        apply_step: bool = True,
        target_raw: np.ndarray | None = None,
        loss_cfg: dict[str, Any] | None = None,
        supervised_mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        del loss_cfg, weight_decay
        if self._torch_last_out is None:
            raise RuntimeError("experimental UNO backward called without a training forward")
        torch = self.torch
        params = [parameter for parameter in self.net.parameters() if parameter.requires_grad]
        hidden_w, out_w = self._torch_step_reference()
        with torch.no_grad():
            hidden_prev = hidden_w.detach().clone() if hidden_w is not None else None
            output_prev = out_w.detach().clone() if out_w is not None else None
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.zero_()
        aux_tensor, diagnostics = self._auxiliary_tensor(target_raw, supervised_mask)
        grad_tensor = torch.as_tensor(np.asarray(grad_raw, dtype=np.float32), device=self.device)
        self._torch_last_out.backward(grad_tensor, retain_graph=aux_tensor is not None)
        if aux_tensor is not None:
            aux_tensor.backward()
        step_hidden = 0.0
        step_output = 0.0
        if apply_step:
            with torch.no_grad():
                for parameter in params:
                    if parameter.grad is not None:
                        parameter -= float(lr) * parameter.grad
                hidden_cur, output_cur = self._torch_step_reference()
                if hidden_prev is not None and hidden_cur is not None:
                    step_hidden = float(torch.linalg.norm(hidden_cur - hidden_prev) / torch.linalg.norm(hidden_prev).clamp_min(1.0e-12))
                if output_prev is not None and output_cur is not None:
                    step_output = float(torch.linalg.norm(output_cur - output_prev) / torch.linalg.norm(output_prev).clamp_min(1.0e-12))
        diagnostics["loss_aux_total"] = float(aux_tensor.detach().cpu()) if aux_tensor is not None else 0.0
        diagnostics["step_rel_hidden_mean"] = step_hidden
        diagnostics["step_rel_output"] = step_output
        self._torch_last_in = None
        self._torch_last_out = None
        return diagnostics

    def evaluate_auxiliary_losses(
        self,
        cond: np.ndarray,
        target_raw: np.ndarray,
        *,
        spatial_features: np.ndarray | None = None,
        supervised_mask: np.ndarray | None = None,
        loss_cfg: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        del loss_cfg
        torch = self.torch
        fmap = self._feature_map(cond, spatial_features=spatial_features)
        value = torch.from_numpy(np.moveaxis(fmap, -1, 1).astype(np.float32)).to(self.device)
        self._ensure_net_device()
        self.net.eval()
        with torch.no_grad():
            prediction = self.net(value)
            cached = self._torch_last_out
            self._torch_last_out = prediction
            total, diagnostics = self._auxiliary_tensor(target_raw, supervised_mask)
            self._torch_last_out = cached
        diagnostics["loss_aux_total"] = float(total.detach().cpu()) if total is not None else 0.0
        return diagnostics
