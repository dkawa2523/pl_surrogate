"""DeepONet training-stage configuration helpers."""

from __future__ import annotations

from typing import Any


def resolve_deeponet_stages(
    deeponet_cfg: dict[str, Any],
    *,
    default_epochs: int,
    default_lr: float,
) -> list[dict[str, Any]]:
    cfg = dict(deeponet_cfg or {})
    has_s1 = "stage1" in cfg
    has_s2 = "stage2" in cfg
    s1 = dict(cfg.get("stage1", {}))
    s2 = dict(cfg.get("stage2", {}))

    if not has_s1 and not has_s2:
        return [
            {
                "name": "stage1",
                "epochs": int(max(1, int(default_epochs))),
                "lr": float(default_lr),
                "freeze_poisson_head": False,
                "freeze_boundary_operator": False,
                "refine_iters": 0,
            }
        ]

    stages: list[dict[str, Any]] = [
        {
            "name": "stage1",
            "epochs": int(s1.get("epochs", max(1, int(default_epochs)))),
            "lr": float(s1.get("lr", float(default_lr))),
            "freeze_poisson_head": bool(s1.get("freeze_poisson_head", True)),
            "freeze_boundary_operator": bool(s1.get("freeze_boundary_operator", True)),
            "refine_iters": int(s1.get("refine_iters", 0)),
        }
    ]
    if has_s2:
        stages.append(
            {
                "name": "stage2",
                "epochs": int(s2.get("epochs", max(1, int(default_epochs)))),
                "lr": float(s2.get("lr", max(float(default_lr) * 0.5, 1e-4))),
                "freeze_poisson_head": bool(s2.get("freeze_poisson_head", False)),
                "freeze_boundary_operator": bool(s2.get("freeze_boundary_operator", False)),
                "refine_iters": int(s2.get("refine_iters", 1)),
            }
        )
    return stages
