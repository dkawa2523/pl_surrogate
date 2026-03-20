"""Torch trainer for DeepONet PDE coupled workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_surrogate.core.artifact_store import ArtifactStore
from plasma_surrogate.core.deeponet_contract import allvars_plasma_balance_score
from plasma_surrogate.core.physics_contract import resolve_epoch_scaled_physics
from plasma_surrogate.core.torch_backend import require_torch
from plasma_surrogate.train.loss_composer import compose_supervised_torch, compose_torch


@dataclass
class TorchTrainOutput:
    history: list[dict[str, float]]
    model: Any


class TorchTrainer:
    def __init__(self, output_dir: str | Path):
        self.store = ArtifactStore(output_dir)

    def _save_physics_terms(self, history: list[dict[str, float]]) -> None:
        if not history:
            return
        header = [
            "epoch",
            "stage",
            "train_phys_loss",
            "train_poisson_loss",
            "train_boundary_operator_loss",
            "train_boundary_loss",
            "train_rho_loss",
        ]
        rows = [
            [
                row.get("epoch", 0.0),
                row.get("stage", 0.0),
                row.get("train_phys_loss", 0.0),
                row.get("train_poisson_loss", 0.0),
                row.get("train_boundary_operator_loss", 0.0),
                row.get("train_boundary_loss", 0.0),
                row.get("train_rho_loss", 0.0),
            ]
            for row in history
        ]
        self.store.save_csv("scalars/physics_terms.csv", header, rows)

    def _save_resolved_physics(self, physics_cfg: dict[str, Any] | None) -> None:
        cfg = dict(physics_cfg or {})
        self.store.save_json(
            "resolved_physics.json",
            {
                "enabled": bool(cfg.get("enabled", False)),
                "resolved_terms": list(cfg.get("resolved_terms", [])),
                "lambda_poisson": float(cfg.get("lambda_poisson", 0.0)),
                "boundary_operator": {
                    "enabled": bool(cfg.get("boundary_operator", {}).get("enabled", False)),
                    "lambda": float(cfg.get("boundary_operator", {}).get("lambda", 0.0)),
                    "mode": str(cfg.get("boundary_operator", {}).get("mode", "operator_prior")),
                    "primary_qoi_key": str(cfg.get("boundary_operator", {}).get("primary_qoi_key", "Gamma_i")),
                },
            },
        )

    def _save_optimization_diagnostics(self, rows: list[dict[str, float]]) -> None:
        if not rows:
            return
        base_header = [
            "epoch",
            "grad_l2_total",
            "active_weight_ratio",
            "step_rel_hidden_mean",
            "step_rel_output",
            "step_rel_output_to_hidden",
            "grad_scale_applied",
            "grad_norm_pre_scale",
            "grad_norm_post_scale",
            "grad_norm_post_clip",
            "clip_ratio",
            "effective_clip_flag",
            "stagnation_flag",
            "head_refresh_applied",
            "head_design_cond",
        ]
        dynamic_grad = sorted(
            {
                str(k)
                for row in rows
                for k in row.keys()
                if str(k).startswith("grad_l2_") and str(k) != "grad_l2_total"
            }
        )
        header = ["epoch", "grad_l2_total"] + dynamic_grad + [k for k in base_header if k not in {"epoch", "grad_l2_total"}]
        data = [[r.get(k, 0.0) for k in header] for r in rows]
        self.store.save_csv("scalars/optimization_diagnostics.csv", header, data)

    @staticmethod
    def _clone_model_state_numpy(model: Any) -> dict[str, np.ndarray]:
        if not hasattr(model, "state_dict_numpy"):
            raise TypeError("deeponet model must support state_dict_numpy")
        state = dict(model.state_dict_numpy())
        return {str(k): np.asarray(v, dtype=np.float32).copy() for k, v in state.items()}

    @staticmethod
    def _restore_model_state_numpy(model: Any, state: dict[str, np.ndarray]) -> None:
        if not hasattr(model, "load_state_dict_numpy"):
            raise TypeError("deeponet model must support load_state_dict_numpy")
        model.load_state_dict_numpy({str(k): np.asarray(v, dtype=np.float32).copy() for k, v in state.items()})

    @staticmethod
    def _align_mask_bhw(mask: Any | None, *, batch_size: int, h: int, w: int) -> np.ndarray:
        if mask is None:
            return np.ones((batch_size, h, w), dtype=bool)
        arr = np.asarray(mask, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.broadcast_to(arr[None, ...], (batch_size, h, w))
        elif arr.ndim == 3:
            if arr.shape[0] == 1 and batch_size > 1:
                arr = np.broadcast_to(arr, (batch_size, h, w))
            elif arr.shape[0] != batch_size:
                raise ValueError(f"supervised_mask batch mismatch: expected {batch_size}, got {arr.shape[0]}")
        elif arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            raise ValueError(f"unsupported supervised_mask shape for deeponet: {arr.shape}")
        if arr.shape[-2:] != (h, w):
            raise ValueError(f"supervised_mask spatial mismatch for deeponet: expected {(h, w)}, got {arr.shape[-2:]}")
        return arr > 0.5

    @staticmethod
    def _set_requires_grad(module: Any | None, flag: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad_(bool(flag))

    @staticmethod
    def _collect_trainable_params(model: Any) -> list[dict[str, Any]]:
        params: list[dict[str, Any]] = []
        seen: set[int] = set()

        def _append(module: Any, prefix: str) -> None:
            if module is None or not hasattr(module, "named_parameters"):
                return
            for name, p in module.named_parameters():
                if p is None:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                if bool(getattr(p, "requires_grad", False)):
                    params.append(
                        {
                            "name": f"{prefix}{name}",
                            "shape": [int(v) for v in p.shape],
                            "n_params": int(np.prod(tuple(p.shape))),
                        }
                    )

        _append(getattr(model, "sensor_encoder", None), "sensor_encoder.")
        _append(getattr(model, "branch", None), "branch.")
        _append(getattr(model, "fused_head", None), "fused_head.")
        _append(getattr(model, "global_head", None), "global_head.")
        _append(getattr(model, "branch_latent_norm", None), "branch_latent_norm.")
        _append(getattr(model, "trunk", None), "trunk.")
        _append(getattr(model, "trunk_latent_norm", None), "trunk_latent_norm.")
        _append(getattr(model, "trunk_film", None), "trunk_film.")
        _append(getattr(model, "residual_cond", None), "residual_cond.")
        _append(getattr(model, "residual_head", None), "residual_head.")
        out_bias = getattr(model, "out_bias", None)
        if out_bias is not None and bool(getattr(out_bias, "requires_grad", False)):
            params.append(
                {
                    "name": "out_bias",
                    "shape": [int(v) for v in out_bias.shape],
                    "n_params": int(np.prod(tuple(out_bias.shape))),
                }
            )
        residual_scale = getattr(model, "residual_scale", None)
        if residual_scale is not None and bool(getattr(residual_scale, "requires_grad", False)):
            params.append(
                {
                    "name": "residual_scale",
                    "shape": [int(v) for v in residual_scale.shape],
                    "n_params": int(np.prod(tuple(residual_scale.shape))),
                }
            )
        ph = getattr(model, "poisson_head", None)
        if ph is not None:
            _append(getattr(ph, "net", None), "poisson_head.net.")
        bo = getattr(model, "boundary_operator", None)
        if bo is not None:
            for name in ["w_log_ne", "w_te", "w_en", "bias"]:
                p = getattr(bo, name, None)
                if p is None:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                if bool(getattr(p, "requires_grad", False)):
                    params.append(
                        {
                            "name": f"boundary_operator.{name}",
                            "shape": [int(v) for v in p.shape],
                            "n_params": int(np.prod(tuple(p.shape))),
                        }
                    )
        return params

    @staticmethod
    def _resolve_deeponet_optimizer_cfg(raw_cfg: dict[str, Any] | None, *, default_lr: float) -> dict[str, Any]:
        cfg = dict(raw_cfg or {})
        opt_type = str(cfg.get("type", "adamw")).strip().lower()
        if opt_type != "adamw":
            raise ValueError("train.deeponet_plasma.optimizer.type must be: adamw")
        lr = float(cfg.get("lr", default_lr))
        if lr <= 0.0:
            raise ValueError("train.deeponet_plasma.optimizer.lr must be > 0")
        weight_decay = float(cfg.get("weight_decay", 0.0))
        if weight_decay < 0.0:
            raise ValueError("train.deeponet_plasma.optimizer.weight_decay must be >= 0")
        betas_raw = cfg.get("betas", [0.9, 0.999])
        if not isinstance(betas_raw, list) or len(betas_raw) != 2:
            raise ValueError("train.deeponet_plasma.optimizer.betas must be [beta1, beta2]")
        beta1 = float(betas_raw[0])
        beta2 = float(betas_raw[1])
        eps = float(cfg.get("eps", 1.0e-8))
        if eps <= 0.0:
            raise ValueError("train.deeponet_plasma.optimizer.eps must be > 0")
        schedule = str(cfg.get("schedule", "none")).strip().lower()
        if schedule not in {"none", "cosine"}:
            raise ValueError("train.deeponet_plasma.optimizer.schedule must be one of: none, cosine")
        warmup_epochs = int(max(int(cfg.get("warmup_epochs", 0)), 0))
        return {
            "type": opt_type,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": (beta1, beta2),
            "eps": eps,
            "schedule": schedule,
            "warmup_epochs": warmup_epochs,
        }

    @staticmethod
    def _build_deeponet_param_groups(model: Any, *, base_weight_decay: float) -> list[dict[str, Any]]:
        named_params: list[tuple[str, Any]] = []
        if hasattr(model, "named_parameters"):
            named_params = [
                (str(name), p)
                for name, p in model.named_parameters()
                if p is not None and bool(getattr(p, "requires_grad", False))
            ]
        if not named_params:
            params = [p for p in model.parameters() if bool(getattr(p, "requires_grad", False))]
            return [{"params": params, "weight_decay": float(base_weight_decay)}]
        decay: list[Any] = []
        no_decay: list[Any] = []
        for name, param in named_params:
            lname = str(name).lower()
            is_no_decay = (
                lname.endswith(".bias")
                or lname == "out_bias"
                or "norm" in lname
                or lname.endswith("residual_scale")
            )
            if is_no_decay:
                no_decay.append(param)
            else:
                decay.append(param)
        groups: list[dict[str, Any]] = []
        if decay:
            groups.append({"params": decay, "weight_decay": float(base_weight_decay)})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    @staticmethod
    def _set_optimizer_epoch_lr(
        optimizer: Any,
        *,
        base_lr: float,
        epoch: int,
        epochs: int,
        schedule: str,
        warmup_epochs: int,
    ) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            scale = float(epoch + 1) / float(max(warmup_epochs, 1))
        elif schedule == "cosine":
            tail = max(int(epochs) - int(warmup_epochs), 1)
            t = float(max(epoch - warmup_epochs, 0)) / float(max(tail - 1, 1))
            scale = 0.5 * (1.0 + float(np.cos(np.pi * t)))
        else:
            scale = 1.0
        lr_now = float(base_lr) * float(scale)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        return lr_now

    @staticmethod
    def _iter_case_batches(
        n_cases: int,
        *,
        batch_size_cases: int,
        shuffle: bool,
        seed: int,
        epoch: int,
    ) -> list[np.ndarray]:
        n = int(max(int(n_cases), 0))
        if n == 0:
            return []
        bs = int(max(int(batch_size_cases), 0))
        if bs <= 0 or bs >= n:
            return [np.arange(n, dtype=np.int64)]
        idx = np.arange(n, dtype=np.int64)
        if bool(shuffle):
            rng = np.random.default_rng(int(seed) + int(epoch))
            rng.shuffle(idx)
        return [idx[start : start + bs] for start in range(0, n, bs)]

    def run_deeponet(
        self,
        model: Any,
        cond_train: np.ndarray,
        y_train: np.ndarray,
        cond_val: np.ndarray,
        y_val: np.ndarray,
        geom_ctx: Any,
        y_vars: list[str],
        stages: list[dict[str, Any]] | None = None,
        physics_cfg: dict[str, Any] | None = None,
        supervised_targets: dict[str, Any] | None = None,
        loss_cfg: dict[str, Any] | None = None,
        curriculum_cfg: dict[str, Any] | None = None,
        supervised_mask: Any | None = None,
        supervised_distance: Any | None = None,
        optimizer_contract: dict[str, Any] | None = None,
        selection_cfg: dict[str, Any] | None = None,
        batch_size_cases: int = 0,
        shuffle_cases: bool = True,
        seed: int = 0,
    ) -> TorchTrainOutput:
        torch = require_torch()
        self._save_resolved_physics(physics_cfg)
        resolved_terms = list((physics_cfg or {}).get("resolved_terms", []))
        stages_cfg = list(stages or [{"name": "stage1", "epochs": 10, "lr": 1e-3, "freeze_poisson_head": True, "freeze_boundary_operator": True}])
        c_tr = torch.as_tensor(cond_train, dtype=torch.float32)
        y_tr = torch.as_tensor(y_train, dtype=torch.float32)
        c_va = torch.as_tensor(cond_val, dtype=torch.float32)
        y_va = torch.as_tensor(y_val, dtype=torch.float32)

        history: list[dict[str, float]] = []
        diagnostics_rows: list[dict[str, float]] = []
        epoch_offset = 0
        poisson_head = getattr(model, "poisson_head", None)
        boundary_operator = getattr(model, "boundary_operator", None)
        stage_audit: list[dict[str, Any]] = []

        opt_contract = dict(optimizer_contract or {})
        selection_cfg = dict(selection_cfg or {})
        selection_mode = str(selection_cfg.get("mode", "last")).strip().lower()
        if selection_mode not in {"last", "best_val_allvars_balance"}:
            raise ValueError("train.deeponet_plasma.selection.mode must be one of: last, best_val_allvars_balance")
        selection_eval_every = int(max(int(selection_cfg.get("eval_every_n_epochs", 2)), 1))
        selection_warmup = int(max(int(selection_cfg.get("warmup_epochs", 5)), 0))
        raw_weights = dict(selection_cfg.get("weights", {}))
        if len(raw_weights) == 0:
            uniform = 1.0 / float(max(len(y_vars), 1))
            selection_weights = {str(name): float(uniform) for name in y_vars}
        else:
            unknown = sorted(set(str(k) for k in raw_weights.keys()) - set(str(v) for v in y_vars))
            if unknown:
                raise ValueError(
                    "train.deeponet_plasma.selection.weights contains unknown vars: "
                    f"{unknown}; expected={list(y_vars)}"
                )
            selection_weights = {str(name): float(raw_weights.get(name, 0.0)) for name in y_vars}
        grad_clip_cfg = dict(opt_contract.get("grad_clip", {}))
        diag_cfg = dict(opt_contract.get("diagnostics", {}))
        alert_cfg = dict(diag_cfg.get("alert", {}))
        alert_enabled = bool(alert_cfg.get("enabled", False))
        min_effective_clip_ratio = float(alert_cfg.get("min_effective_clip_ratio", 0.99))
        clip_effective_direction = str(alert_cfg.get("clip_effective_direction", "lt")).strip().lower()
        if clip_effective_direction not in {"lt", "gt"}:
            raise ValueError(
                "train.optimizer_contract.diagnostics.alert.clip_effective_direction must be one of: lt, gt"
            )
        min_grad_l2_total = float(alert_cfg.get("min_grad_l2_total", 1.0e-5))
        best_state: dict[str, np.ndarray] | None = None
        best_score = float("-inf")
        best_epoch = -1
        best_score_parts: dict[str, float] = {}
        selection_valid = selection_mode == "last"
        total_epochs = int(sum(max(int(stage.get("epochs", 0)), 0) for stage in stages_cfg))

        for stage in stages_cfg:
            n_epochs = int(stage.get("epochs", 0))
            if n_epochs <= 0:
                continue
            lr = float(stage.get("lr", 1e-3))
            freeze_head = bool(stage.get("freeze_poisson_head", True))
            freeze_bo = bool(stage.get("freeze_boundary_operator", True))
            self._set_requires_grad(model, True)
            self._set_requires_grad(poisson_head, not freeze_head)
            self._set_requires_grad(boundary_operator, not freeze_bo)
            trainable = self._collect_trainable_params(model)
            stage_name = str(stage.get("name", f"stage_{len(stage_audit)+1}"))
            stage_info = {
                "stage_name": stage_name,
                "epochs": n_epochs,
                "lr": lr,
                "freeze_poisson_head": freeze_head,
                "freeze_boundary_operator": freeze_bo,
                "trainable_params": trainable,
                "n_trainable_params": int(sum(int(p["n_params"]) for p in trainable)),
            }
            stage_audit.append(stage_info)
            self.store.save_json(f"trainable_params_{stage_name}.json", stage_info)
            params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
            if len(params) == 0:
                raise RuntimeError("No trainable parameters in deeponet_plasma stage")
            deeponet_opt_cfg = self._resolve_deeponet_optimizer_cfg(
                dict(opt_contract.get("deeponet_optimizer", {})),
                default_lr=lr,
            )
            param_groups = self._build_deeponet_param_groups(
                model,
                base_weight_decay=float(deeponet_opt_cfg["weight_decay"]),
            )
            optim = torch.optim.AdamW(
                param_groups,
                lr=float(deeponet_opt_cfg["lr"]),
                betas=tuple(deeponet_opt_cfg["betas"]),
                eps=float(deeponet_opt_cfg["eps"]),
            )

            for e in range(n_epochs):
                epoch_abs = epoch_offset + e
                lr_now = self._set_optimizer_epoch_lr(
                    optim,
                    base_lr=float(deeponet_opt_cfg["lr"]),
                    epoch=int(epoch_abs),
                    epochs=max(int(total_epochs), 1),
                    schedule=str(deeponet_opt_cfg.get("schedule", "none")),
                    warmup_epochs=int(deeponet_opt_cfg.get("warmup_epochs", 0)),
                )
                epoch_physics_cfg = resolve_epoch_scaled_physics(
                    physics_cfg,
                    epoch=epoch_abs,
                    curriculum_cfg=curriculum_cfg,
                )
                resolved_terms = list(epoch_physics_cfg.get("resolved_terms", []))
                model.train()
                train_batches = self._iter_case_batches(
                    int(c_tr.shape[0]),
                    batch_size_cases=int(batch_size_cases),
                    shuffle=bool(shuffle_cases),
                    seed=int(seed),
                    epoch=int(epoch_abs),
                )
                train_loss_num = 0.0
                train_data_num = 0.0
                train_phys_num = 0.0
                terms_acc = {"poisson": 0.0, "boundary": 0.0, "boundary_operator": 0.0, "rho": 0.0}
                grad_l2_by_var = {str(name): 0.0 for name in y_vars}
                grad_norm_pre = 0.0
                grad_norm_post = 0.0
                for batch_idx in train_batches:
                    c_tr_b = c_tr[batch_idx]
                    y_tr_b = y_tr[batch_idx]
                    pred = model.predict_fields_torch(c_tr_b, geom_ctx=geom_ctx)
                    if poisson_head is not None and "rho_eff" in pred:
                        pred["phi"] = poisson_head.predict_phi(
                            pred["rho_eff"],
                            c_tr_b,
                            geom_ctx=geom_ctx,
                            refine_iters=int(stage.get("refine_iters", 0)),
                        )
                    data_loss, _ = compose_supervised_torch(
                        pred_fields=pred,
                        target_fields=y_tr_b,
                        y_order=y_vars,
                        loss_cfg=loss_cfg,
                        mask=supervised_mask,
                        distance_any=supervised_distance,
                    )
                    phys_loss, terms = compose_torch(
                        pred_fields=pred,
                        cond_vec=c_tr_b,
                        geom_ctx=geom_ctx,
                        physics_cfg=epoch_physics_cfg,
                        boundary_operator_model=boundary_operator,
                        supervised_targets=supervised_targets,
                        resolved_terms=resolved_terms,
                    )
                    total = data_loss + phys_loss
                    optim.zero_grad()
                    total.backward()
                    bsz_case = int(c_tr_b.shape[0])
                    train_loss_num += float(total.detach().cpu().item()) * float(bsz_case)
                    train_data_num += float(data_loss.detach().cpu().item()) * float(bsz_case)
                    train_phys_num += float(phys_loss.detach().cpu().item()) * float(bsz_case)
                    for key in terms_acc:
                        terms_acc[key] += float(terms.get(key, 0.0)) * float(bsz_case)
                    out_bias = getattr(model, "out_bias", None)
                    if out_bias is not None and getattr(out_bias, "grad", None) is not None:
                        g = np.asarray(out_bias.grad.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                        for idx, name in enumerate(y_vars):
                            if idx < g.shape[0]:
                                grad_l2_by_var[str(name)] = float(abs(float(g[idx])))
                    grad_norm_pre = float(
                        np.sqrt(
                            sum(
                                float((p.grad.detach() ** 2).sum().cpu().item())
                                for p in params
                                if getattr(p, "grad", None) is not None
                            )
                        )
                    )
                    clip_mode = str(grad_clip_cfg.get("mode", "off")).strip().lower()
                    clip_norm = float(grad_clip_cfg.get("norm", 0.0))
                    if clip_mode != "off" and clip_norm > 0.0:
                        scale = 1.0
                        if bool(grad_clip_cfg.get("adaptive_by_dim", False)):
                            n_dim = float(
                                sum(float(np.prod(tuple(p.shape))) for p in params if getattr(p, "grad", None) is not None)
                            )
                            scale = float(np.sqrt(max(n_dim, 1.0)))
                        torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip_norm) * scale)
                    grad_norm_post = float(
                        np.sqrt(
                            sum(
                                float((p.grad.detach() ** 2).sum().cpu().item())
                                for p in params
                                if getattr(p, "grad", None) is not None
                            )
                        )
                    )
                    optim.step()
                denom_tr = float(max(int(c_tr.shape[0]), 1))
                train_loss = float(train_loss_num / denom_tr)
                train_data_loss = float(train_data_num / denom_tr)
                train_phys_loss = float(train_phys_num / denom_tr)
                terms_mean = {k: float(v / denom_tr) for k, v in terms_acc.items()}

                should_eval_selection = (
                    selection_mode == "best_val_allvars_balance"
                    and epoch_abs >= selection_warmup
                    and ((epoch_abs - selection_warmup) % selection_eval_every == 0)
                )
                model.eval()
                with torch.no_grad():
                    val_batches = self._iter_case_batches(
                        int(c_va.shape[0]),
                        batch_size_cases=int(batch_size_cases),
                        shuffle=False,
                        seed=int(seed),
                        epoch=int(epoch_abs),
                    )
                    val_total_num = 0.0
                    val_data_num = 0.0
                    val_phys_num = 0.0
                    pred_stack_chunks: list[np.ndarray] = []
                    target_stack_chunks: list[np.ndarray] = []
                    for batch_idx in val_batches:
                        c_va_b = c_va[batch_idx]
                        y_va_b = y_va[batch_idx]
                        pred_va_b = model.predict_fields_torch(c_va_b, geom_ctx=geom_ctx)
                        if poisson_head is not None and "rho_eff" in pred_va_b:
                            pred_va_b["phi"] = poisson_head.predict_phi(
                                pred_va_b["rho_eff"],
                                c_va_b,
                                geom_ctx=geom_ctx,
                                refine_iters=int(stage.get("refine_iters", 0)),
                            )
                        val_data_b, _ = compose_supervised_torch(
                            pred_fields=pred_va_b,
                            target_fields=y_va_b,
                            y_order=y_vars,
                            loss_cfg=loss_cfg,
                            mask=supervised_mask,
                            distance_any=supervised_distance,
                        )
                        val_phys_b, _ = compose_torch(
                            pred_fields=pred_va_b,
                            cond_vec=c_va_b,
                            geom_ctx=geom_ctx,
                            physics_cfg=epoch_physics_cfg,
                            boundary_operator_model=boundary_operator,
                            supervised_targets=supervised_targets,
                            resolved_terms=resolved_terms,
                        )
                        bsz_case = int(c_va_b.shape[0])
                        val_data_num += float(val_data_b.detach().cpu().item()) * float(bsz_case)
                        val_phys_num += float(val_phys_b.detach().cpu().item()) * float(bsz_case)
                        val_total_num += float((val_data_b + val_phys_b).detach().cpu().item()) * float(bsz_case)
                        if should_eval_selection:
                            pred_stack_chunks.append(
                                np.concatenate(
                                    [np.asarray(pred_va_b[name].detach().cpu().numpy(), dtype=np.float32) for name in y_vars],
                                    axis=1,
                                )
                            )
                            target_stack_chunks.append(np.asarray(y_va_b.detach().cpu().numpy(), dtype=np.float32))
                    denom_va = float(max(int(c_va.shape[0]), 1))
                    val_data = float(val_data_num / denom_va)
                    val_phys = float(val_phys_num / denom_va)
                    val_total = float(val_total_num / denom_va)
                val_balance_score = float("nan")
                if should_eval_selection:
                    pred_stack = (
                        np.concatenate(pred_stack_chunks, axis=0)
                        if pred_stack_chunks
                        else np.zeros((0, len(y_vars), int(y_va.shape[-2]), int(y_va.shape[-1])), dtype=np.float32)
                    )
                    target_stack = (
                        np.concatenate(target_stack_chunks, axis=0)
                        if target_stack_chunks
                        else np.zeros((0, len(y_vars), int(y_va.shape[-2]), int(y_va.shape[-1])), dtype=np.float32)
                    )
                    bsz, _c, hh, ww = pred_stack.shape
                    plasma_mask = self._align_mask_bhw(supervised_mask, batch_size=bsz, h=hh, w=ww)
                    val_balance_score, parts = allvars_plasma_balance_score(
                        pred=pred_stack,
                        target=target_stack,
                        y_vars=list(y_vars),
                        weights=selection_weights,
                        plasma_mask=plasma_mask,
                    )
                    if np.isfinite(val_balance_score) and (
                        best_state is None or float(val_balance_score) > float(best_score)
                    ):
                        best_state = self._clone_model_state_numpy(model)
                        best_score = float(val_balance_score)
                        best_epoch = int(epoch_abs)
                        best_score_parts = dict(parts)
                        selection_valid = True

                history.append(
                    {
                        "epoch": float(epoch_offset + e),
                        "stage": float(len(history)),
                        "train_loss": float(train_loss),
                        "train_data_loss": float(train_data_loss),
                        "train_phys_loss": float(train_phys_loss),
                        "train_poisson_loss": float(terms_mean.get("poisson", 0.0)),
                        "train_boundary_loss": float(terms_mean.get("boundary", 0.0)),
                        "train_boundary_operator_loss": float(terms_mean.get("boundary_operator", 0.0)),
                        "train_rho_loss": float(terms_mean.get("rho", 0.0)),
                        "train_physics_scale": float(epoch_physics_cfg.get("physics_ramp_scale", 1.0)),
                        "train_lr": float(lr_now),
                        "val_loss": float(val_total),
                        "val_data_loss": float(val_data),
                        "val_phys_loss": float(val_phys),
                        "val_balance_score": float(val_balance_score if np.isfinite(val_balance_score) else 0.0),
                        "selection_valid_flag": 1.0 if selection_valid else 0.0,
                        "selected_epoch_flag": 0.0,
                        "selected_epoch_score": 0.0,
                        **{f"selection_score_{name}": 0.0 for name in y_vars},
                    }
                )
                diagnostics_rows.append(
                    {
                        "epoch": float(epoch_offset + e),
                        "grad_l2_total": float(grad_norm_post),
                        **{f"grad_l2_{name}": float(grad_l2_by_var.get(name, 0.0)) for name in y_vars},
                        "active_weight_ratio": 0.0,
                        "step_rel_hidden_mean": 0.0,
                        "step_rel_output": 0.0,
                        "step_rel_output_to_hidden": 0.0,
                        "grad_scale_applied": 1.0,
                        "grad_norm_pre_scale": float(grad_norm_pre),
                        "grad_norm_post_scale": float(grad_norm_pre),
                        "grad_norm_post_clip": float(grad_norm_post),
                        "clip_ratio": float(grad_norm_post / max(grad_norm_pre, 1e-12)),
                        "effective_clip_flag": float(
                            1.0
                            if (
                                alert_enabled
                                and clip_mode != "off"
                                and (
                                    (clip_effective_direction == "lt" and (grad_norm_post / max(grad_norm_pre, 1e-12)) < min_effective_clip_ratio)
                                    or (clip_effective_direction == "gt" and (grad_norm_post / max(grad_norm_pre, 1e-12)) > min_effective_clip_ratio)
                                )
                            )
                            else 0.0
                        ),
                        "stagnation_flag": float(1.0 if (alert_enabled and grad_norm_post <= min_grad_l2_total) else 0.0),
                        "head_refresh_applied": 0.0,
                        "head_design_cond": 0.0,
                    }
                )
            epoch_offset += n_epochs

        selected_epoch_effective = int(best_epoch if best_epoch >= 0 else (len(history) - 1))
        if selection_mode == "best_val_allvars_balance" and best_state is not None:
            self._restore_model_state_numpy(model, best_state)
        if selection_mode == "best_val_allvars_balance" and best_state is None:
            selection_valid = False
        for row in history:
            row["selected_epoch_flag"] = 1.0 if int(row.get("epoch", -1)) == selected_epoch_effective else 0.0
            row["selection_valid_flag"] = 1.0 if selection_valid else 0.0
            row["selected_epoch_score"] = float(best_score if best_epoch >= 0 else 0.0)
            if int(row.get("epoch", -1)) == selected_epoch_effective and best_epoch >= 0:
                for name in y_vars:
                    key = f"selection_score_{name}"
                    row[key] = float(best_score_parts.get(f"r2_{name}_plasma", 0.0))

        header = list(history[0].keys()) if history else ["epoch", "train_loss", "val_loss"]
        rows = [[r[k] for k in header] for r in history]
        self.store.save_csv("scalars/metrics.csv", header, rows)
        self._save_physics_terms(history)
        self._save_optimization_diagnostics(diagnostics_rows)
        self.store.save_json(
            "trainable_params.json",
            {
                "n_params": int(sum(int(np.prod(tuple(p.shape))) for p in model.parameters())),
                "has_poisson_head": bool(poisson_head is not None),
                "has_boundary_operator": bool(boundary_operator is not None),
                "stages": stage_audit,
            },
        )
        return TorchTrainOutput(history=history, model=model)
