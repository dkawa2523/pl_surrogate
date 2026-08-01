"""Small shared helpers for model-owned auxiliary-loss diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plasma_surrogate.core.target_groups import resolve_target_weight_multipliers


def coefficient_aux_loss_diagnostics(
    *,
    coeff_loss: float,
    coeff_loss_weight: float,
    coeff_loss_by_target: Mapping[str, float] | None = None,
    coeff_loss_by_group: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return the common flat diagnostics contract for a coefficient loss.

    ``coeff_loss`` is retained as a compatibility alias for existing callers.
    ``loss_aux_total`` is the weighted contribution added to the optimization
    objective; it is intentionally separate from the unweighted diagnostic.
    """

    raw = float(coeff_loss)
    weighted = float(coeff_loss_weight) * raw
    out = {
        "coeff_loss": raw,
        "loss_aux_coeff": raw,
        "loss_aux_coeff_weighted": weighted,
        "loss_aux_total": weighted,
    }
    for name, value in dict(coeff_loss_by_target or {}).items():
        out[f"loss_aux_coeff_{name}"] = float(value)
    for name, value in dict(coeff_loss_by_group or {}).items():
        out[f"loss_aux_coeff_group_{name}"] = float(value)
    return out


def coefficient_aux_loss_tensor(
    coeff_pred: Any,
    coeff_target: Any,
    *,
    basis_keys: list[str],
    coeff_slices: Mapping[str, tuple[int, int]],
    loss_cfg: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Aggregate coefficient MSE without rank- or family-count bias.

    Legacy ``group_weighting.mode=none`` preserves the historical flat
    coefficient mean.  Explicit target/group-balanced modes first average the
    coefficients within each target and then reuse the field-loss target
    multipliers, so a larger POD rank cannot silently dominate optimization.
    """

    if tuple(coeff_pred.shape) != tuple(coeff_target.shape) or len(tuple(coeff_pred.shape)) != 2:
        raise ValueError(
            "coefficient auxiliary loss expects matching [B,K] tensors; "
            f"got={tuple(coeff_pred.shape)}/{tuple(coeff_target.shape)}"
        )
    cfg = dict(loss_cfg or {})
    group_cfg = dict(cfg.get("group_weighting", {}) or {})
    mode = str(group_cfg.get("mode", "none")).strip().lower()
    multipliers, groups = resolve_target_weight_multipliers(
        output_vars=[str(name) for name in basis_keys],
        target_role_schema=dict(cfg.get("target_role_schema", {}) or {}),
        mode=mode,
        group_weights=dict(group_cfg.get("weights", {}) or {}),
        context="train.loss.group_weighting",
    )
    by_target: dict[str, Any] = {}
    for name in basis_keys:
        start, stop = coeff_slices[str(name)]
        if int(stop) <= int(start):
            raise ValueError(f"coefficient slice for {name} must be non-empty")
        err = coeff_pred[:, int(start) : int(stop)] - coeff_target[:, int(start) : int(stop)]
        by_target[str(name)] = (err * err).mean()
    if mode == "none":
        error = coeff_pred - coeff_target
        total = (error * error).mean()
    else:
        total = sum(
            by_target[str(name)] * float(multipliers[str(name)])
            for name in basis_keys
        )
    by_group = {
        str(group_name): sum(by_target[str(name)] for name in group.targets) / float(len(group.targets))
        for group_name, group in groups.items()
    }
    return total, by_target, by_group


def project_masked_pod_coefficients_torch(
    centered_fields: Any,
    basis: Any,
    *,
    active_mask: Any | None,
    ridge: float = 1.0e-6,
) -> Any:
    """Project [B,P] fields onto [R,P] POD modes over each case's active region."""

    from plasma_surrogate.core.torch_backend import require_torch

    torch = require_torch()
    if active_mask is None:
        return centered_fields @ basis.transpose(0, 1)
    if tuple(active_mask.shape) != tuple(centered_fields.shape):
        raise ValueError(
            "masked POD projection expects mask [B,P] aligned with fields; "
            f"got={tuple(active_mask.shape)}/{tuple(centered_fields.shape)}"
        )
    rank = int(basis.shape[0])
    eye = torch.eye(rank, dtype=basis.dtype, device=basis.device) * float(ridge)
    chunks = []
    for case_idx in range(int(centered_fields.shape[0])):
        active = active_mask[case_idx] > 0
        if int(active.sum().detach().cpu().item()) <= 0:
            raise ValueError(f"masked POD projection has an empty active mask: case={case_idx}")
        design = basis[:, active].transpose(0, 1)
        values = centered_fields[case_idx, active]
        gram = design.transpose(0, 1) @ design + eye
        rhs = design.transpose(0, 1) @ values
        chunks.append(torch.linalg.solve(gram, rhs))
    return centered_fields.new_zeros((0, rank)) if not chunks else torch.stack(chunks, dim=0)


__all__ = [
    "coefficient_aux_loss_diagnostics",
    "coefficient_aux_loss_tensor",
    "project_masked_pod_coefficients_torch",
]
