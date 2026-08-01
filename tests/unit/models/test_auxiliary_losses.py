from __future__ import annotations

import pytest

from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.models._auxiliary_losses import coefficient_aux_loss_tensor
from tests._runtime_requirements import require_torch_runtime


@pytest.mark.torch_runtime
def test_coefficient_aux_loss_balances_targets_and_field_families_before_pod_rank() -> None:
    require_torch_runtime(enable_backend=True)
    torch = require_torch()
    pred = torch.zeros((1, 5), dtype=torch.float32)
    target = torch.tensor([[1.0, 1.0, 1.0, 1.0, 3.0]], dtype=torch.float32)
    cfg = {
        "group_weighting": {"mode": "uniform_by_group"},
        "target_role_schema": {
            "targets": [
                {"id": "ne", "field_family": "density"},
                {"id": "ni", "field_family": "density"},
                {"id": "Te", "field_family": "temperature"},
            ]
        },
    }

    loss, by_target, by_group = coefficient_aux_loss_tensor(
        pred,
        target,
        basis_keys=["ne", "ni", "Te"],
        coeff_slices={"ne": (0, 1), "ni": (1, 4), "Te": (4, 5)},
        loss_cfg=cfg,
    )

    assert float(by_target["ne"].item()) == pytest.approx(1.0)
    assert float(by_target["ni"].item()) == pytest.approx(1.0)
    assert float(by_target["Te"].item()) == pytest.approx(9.0)
    assert float(by_group["density"].item()) == pytest.approx(1.0)
    assert float(by_group["temperature"].item()) == pytest.approx(9.0)
    assert float(loss.item()) == pytest.approx(5.0)


@pytest.mark.torch_runtime
def test_coefficient_aux_loss_preserves_flat_rank_weighting_only_in_legacy_none_mode() -> None:
    require_torch_runtime(enable_backend=True)
    torch = require_torch()
    pred = torch.zeros((1, 5), dtype=torch.float32)
    target = torch.tensor([[1.0, 1.0, 1.0, 1.0, 3.0]], dtype=torch.float32)

    loss, _by_target, by_group = coefficient_aux_loss_tensor(
        pred,
        target,
        basis_keys=["ne", "ni", "Te"],
        coeff_slices={"ne": (0, 1), "ni": (1, 4), "Te": (4, 5)},
        loss_cfg={"group_weighting": {"mode": "none"}},
    )

    assert by_group == {}
    assert float(loss.item()) == pytest.approx(13.0 / 5.0)
