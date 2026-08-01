"""Resolved benchmark model contract projection helpers."""

from __future__ import annotations

from typing import Any


class BenchmarkContractEmitter:
    def __init__(self, *, unet_contract_optional_scopes: set[str]):
        self.unet_contract_optional_scopes = set(unet_contract_optional_scopes)

    def emit(
        self,
        *,
        resolved: dict[str, Any],
        train_cfg: dict[str, Any],
        y_vars: list[str],
        active_unet_model: str | None,
        eval_protocol_scope: str,
        contract_samples: Any,
    ) -> None:
        resolved["model_contracts"] = dict(resolved.get("model_contracts", {}) or {})
        emit_unet_contract = bool(contract_samples.unet) or (
            active_unet_model is not None
            and eval_protocol_scope not in self.unet_contract_optional_scopes
        )
        if emit_unet_contract:
            unet_contract_effective = self._aggregate_unet_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                model_key=str(active_unet_model or "unet"),
                unet_contract_samples=contract_samples.unet,
            )
            self._record_model_contract(resolved=resolved, model_name="unet", contract=unet_contract_effective)
            self._emit_display_projection(
                resolved=resolved,
                prefix="unet",
                contract=unet_contract_effective,
                default_target_vars=y_vars,
                backend_key="unet_backend_effective",
                selection_key="unet_selection_mode_effective",
            )
        if contract_samples.fno:
            fno_contract_effective = self._aggregate_fno_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                fno_contract_samples=contract_samples.fno,
            )
            self._emit_spectral_contract_resolved(
                resolved=resolved,
                prefix="fno",
                contract_effective=fno_contract_effective,
                default_target_vars=y_vars,
            )
        if contract_samples.ffno:
            ffno_contract_effective = self._aggregate_ffno_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                ffno_contract_samples=contract_samples.ffno,
            )
            self._emit_spectral_contract_resolved(
                resolved=resolved,
                prefix="ffno",
                contract_effective=ffno_contract_effective,
                default_target_vars=y_vars,
            )
        if contract_samples.coord_mlp:
            coord_mlp_contract_effective = self._aggregate_coord_mlp_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                coord_mlp_contract_samples=contract_samples.coord_mlp,
            )
            self._record_model_contract(
                resolved=resolved,
                model_name="coord_mlp",
                contract=coord_mlp_contract_effective,
            )
            self._emit_display_projection(
                resolved=resolved,
                prefix="coord_mlp",
                contract=coord_mlp_contract_effective,
                default_target_vars=y_vars,
                backend_key=None,
                selection_key=None,
            )
        if contract_samples.deeponet_pod:
            deeponet_pod_contract_effective = self._aggregate_deeponet_pod_contract_effective(
                train_cfg=train_cfg,
                y_vars=y_vars,
                deeponet_pod_contract_samples=contract_samples.deeponet_pod,
            )
            self._record_model_contract(
                resolved=resolved,
                model_name="deeponet_pod",
                contract=deeponet_pod_contract_effective,
            )
            self._emit_display_projection(
                resolved=resolved,
                prefix="deeponet_pod",
                contract=deeponet_pod_contract_effective,
                default_target_vars=y_vars,
                backend_key=None,
                selection_key=None,
            )
        if not contract_samples.deeponet:
            return

        deeponet_contract_effective = self._aggregate_deeponet_contract_effective(
            train_cfg=train_cfg,
            y_vars=y_vars,
            deeponet_contract_samples=contract_samples.deeponet,
        )
        self._record_model_contract(
            resolved=resolved,
            model_name="deeponet",
            contract=deeponet_contract_effective,
        )
        self._emit_display_projection(
            resolved=resolved,
            prefix="deeponet",
            contract=deeponet_contract_effective,
            default_target_vars=y_vars,
            backend_key=None,
            selection_key="selection_mode_effective",
        )

    @staticmethod
    def _record_model_contract(
        *,
        resolved: dict[str, Any],
        model_name: str,
        contract: dict[str, Any],
    ) -> None:
        model_contracts = dict(resolved.get("model_contracts", {}) or {})
        model_contracts[str(model_name)] = dict(contract)
        resolved["model_contracts"] = model_contracts

    @staticmethod
    def _emit_display_projection(
        *,
        resolved: dict[str, Any],
        prefix: str,
        contract: dict[str, Any],
        default_target_vars: list[str],
        backend_key: str | None,
        selection_key: str | None,
    ) -> None:
        resolved[f"{prefix}_target_family_effective"] = str(contract.get("target_family_effective", "allvars"))
        resolved[f"{prefix}_target_vars_effective"] = list(contract.get("target_vars_effective", default_target_vars))
        if backend_key is not None:
            resolved[f"{prefix}_backend_effective"] = str(contract.get(backend_key, "torch"))
        if selection_key is not None:
            resolved[f"{prefix}_selection_mode_effective"] = str(contract.get(selection_key, "last"))

    @staticmethod
    def _first_str(samples: list[dict[str, Any]], key: str) -> str | None:
        for sample in samples:
            val = str(sample.get(key, ""))
            if val:
                return val
        return None

    @staticmethod
    def _first_dict(samples: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        for sample in samples:
            val = sample.get(key)
            if isinstance(val, dict) and len(val) > 0:
                return dict(val)
        return None

    @staticmethod
    def _first_list(samples: list[dict[str, Any]], key: str) -> list[Any] | None:
        for sample in samples:
            val = sample.get(key)
            if isinstance(val, list) and len(val) > 0:
                return list(val)
        return None

    @staticmethod
    def _first_contract_sample(
        samples: list[dict[str, Any]],
        *,
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(defaults)
        sample = samples[0] if samples else {}
        if isinstance(sample, dict):
            out.update(dict(sample))
        return out

    def _aggregate_unet_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        model_key: str,
        unet_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        unet_cfg = dict(train_cfg.get(str(model_key), {}))
        unet_family = str(unet_cfg.get("target_family", "allvars")).strip().lower()
        target_vars_default = list(y_vars)
        unet_input_cfg = dict(unet_cfg.get("input_features", {}))
        raw_feat = unet_input_cfg.get("features", ["x", "y"])
        feat_list = [str(v) for v in raw_feat] if isinstance(raw_feat, list) and raw_feat else ["x", "y"]
        unet_selection_cfg = dict(unet_cfg.get("selection", {}))
        unet_optimizer_cfg = dict(unet_cfg.get("optimizer", {}))
        unet_model_cfg = dict(unet_cfg.get("model_cfg", {}))
        unet_operator_cfg = dict(unet_model_cfg.get("unet_operator_v2_cfg", {}))
        out = {
            "target_vars_effective": target_vars_default,
            "target_family_effective": unet_family,
            "merge_role": "single",
            "unet_backend_effective": str(unet_model_cfg.get("backend", "numpy")).strip().lower(),
            "unet_input_channels_effective": feat_list,
            "unet_selection_mode_effective": str(unet_selection_cfg.get("mode", "last")).strip().lower(),
            "selection_weights_effective": dict(unet_selection_cfg.get("weights", {})),
            "boundary_bonus_weight_effective": float(unet_selection_cfg.get("boundary_bonus_weight", 0.0)),
            "unet_optimizer_effective": {
                "type": str(unet_optimizer_cfg.get("type", "adamw")).strip().lower(),
                "lr": float(unet_optimizer_cfg.get("lr", unet_cfg.get("lr", train_cfg.get("lr", 1e-3)))),
                "weight_decay": float(unet_optimizer_cfg.get("weight_decay", 0.0)),
                "schedule": str(unet_optimizer_cfg.get("schedule", "none")).strip().lower(),
                "warmup_epochs": int(max(int(unet_optimizer_cfg.get("warmup_epochs", 0)), 0)),
            },
            "unet_output_heads_mode_effective": str(
                dict(unet_model_cfg.get("output_heads", {})).get(
                    "mode",
                    unet_operator_cfg.get("head_mode", "shared"),
                )
            ).strip().lower(),
            "feature": {
                "input_features_mode": str(unet_input_cfg.get("mode", "geom_feature_pack")).strip().lower(),
                "input_feature_channels": feat_list,
                "upsample_mode": str(
                    dict(unet_model_cfg.get("conv_cfg", {})).get(
                        "upsample_mode",
                        unet_operator_cfg.get("upsample", unet_model_cfg.get("upsample_mode", "deconv")),
                    )
                ).strip().lower(),
                "distance_transform_mode": str(
                    dict(unet_input_cfg.get("distance_transform", {})).get("mode", "raw")
                ).strip().lower(),
            },
            "operator_v2": {
                "enabled": str(model_key) == "unet_operator_v2",
                "depth": int(unet_operator_cfg.get("depth", 0)),
                "width": int(unet_operator_cfg.get("width", 0)),
                "blocks_per_level": int(unet_operator_cfg.get("blocks_per_level", 0)),
                "downsample": str(unet_operator_cfg.get("downsample", "")).strip().lower(),
                "use_film": bool(unet_operator_cfg.get("use_film", False)),
                "head_mode": str(unet_operator_cfg.get("head_mode", "shared")).strip().lower(),
            },
        }
        v = self._first_str(unet_contract_samples, "target_family_effective")
        if v is not None:
            out["target_family_effective"] = v
        v = self._first_str(unet_contract_samples, "merge_role")
        if v is not None:
            out["merge_role"] = v
        v = self._first_str(unet_contract_samples, "unet_backend_effective")
        if v is not None:
            out["unet_backend_effective"] = v
        values = self._first_list(unet_contract_samples, "unet_input_channels_effective")
        if values is not None:
            out["unet_input_channels_effective"] = [str(x) for x in values]
        values = self._first_list(unet_contract_samples, "target_vars_effective")
        if values is not None:
            out["target_vars_effective"] = [str(x) for x in values]
        v = self._first_str(unet_contract_samples, "unet_selection_mode_effective")
        if v is not None:
            out["unet_selection_mode_effective"] = v
        d = self._first_dict(unet_contract_samples, "selection_weights_effective")
        if d is not None:
            out["selection_weights_effective"] = d
        for sample in unet_contract_samples:
            if isinstance(sample, dict) and "boundary_bonus_weight_effective" in sample:
                out["boundary_bonus_weight_effective"] = float(sample["boundary_bonus_weight_effective"])
                break
        d = self._first_dict(unet_contract_samples, "unet_optimizer_effective")
        if d is not None:
            merged_optimizer = dict(out.get("unet_optimizer_effective", {}))
            merged_optimizer.update(dict(d))
            out["unet_optimizer_effective"] = merged_optimizer
        if isinstance(out.get("unet_optimizer_effective"), dict):
            if "weight_decay" not in out["unet_optimizer_effective"]:
                out["unet_optimizer_effective"]["weight_decay"] = float(
                    dict(unet_cfg.get("optimizer", {})).get("weight_decay", 0.0)
                )
        v = self._first_str(unet_contract_samples, "unet_output_heads_mode_effective")
        if v is not None:
            out["unet_output_heads_mode_effective"] = v
        d = self._first_dict(unet_contract_samples, "feature")
        if d is not None:
            out["feature"] = d
        d = self._first_dict(unet_contract_samples, "operator_v2")
        if d is not None:
            out["operator_v2"] = d
        # Keep resolved_benchmark focused on effective contracts only.
        for key in ["operator_v2"]:
            payload = out.get(key)
            if isinstance(payload, dict) and not bool(payload.get("enabled", False)):
                out.pop(key, None)
        return out

    def _aggregate_spectral_contract(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        model_key: str,
        prefix: str,
        contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        spectral_cfg = dict(train_cfg.get(model_key, {}))
        spectral_family = str(spectral_cfg.get("target_family", "allvars")).strip().lower()
        target_vars_default = list(y_vars)
        input_cfg = dict(spectral_cfg.get("input_features", {}))
        raw_feat = input_cfg.get("features", ["x", "y"])
        feat_list = [str(v) for v in raw_feat] if isinstance(raw_feat, list) and raw_feat else ["x", "y"]
        model_cfg = dict(spectral_cfg.get("model_cfg", {}))
        factorized_cfg = dict(dict(model_cfg.get("spectral_cfg", {})).get("factorized_cfg", {}))
        spectral_model_cfg = dict(model_cfg.get("spectral_cfg", {}))
        selection_cfg = dict(spectral_cfg.get("selection", {}))
        out = {
            "target_vars_effective": target_vars_default,
            "target_family_effective": spectral_family,
            "input_features_mode": str(input_cfg.get("mode", "geom_feature_pack")).strip().lower(),
            "input_feature_channels": feat_list,
            f"{prefix}_backend_effective": "torch",
            f"{prefix}_input_channels_effective": feat_list,
            f"{prefix}_input_features_mode_effective": str(input_cfg.get("mode", "geom_feature_pack")).strip().lower(),
            f"{prefix}_selection_mode_effective": str(selection_cfg.get("mode", "last")).strip().lower(),
            f"{prefix}_optimizer_effective": dict(spectral_cfg.get("optimizer", {})),
            "feature": {
                "input_features_mode": str(input_cfg.get("mode", "geom_feature_pack")).strip().lower(),
                "input_feature_channels": feat_list,
                "distance_transform_mode": str(
                    dict(input_cfg.get("distance_transform", {})).get("mode", "raw")
                ).strip().lower(),
            },
            "spectral": {
                "n_modes": int(model_cfg.get("n_modes", model_cfg.get("fno_n_modes", 2))),
                "dealias_ratio": float(spectral_model_cfg.get("dealias_ratio", 1.0)),
                "taper_alpha": float(spectral_model_cfg.get("taper_alpha", 0.0)),
                "skip_filter": str(spectral_model_cfg.get("skip_filter", "none")).strip().lower(),
            },
        }
        if prefix == "ffno":
            out["spectral"]["factorized_cfg"] = {
                "enabled": bool(factorized_cfg.get("enabled", True)),
                "mode": str(factorized_cfg.get("mode", "separable_1d")).strip().lower(),
                "share_weights": bool(factorized_cfg.get("share_weights", False)),
            }
            local_skip_cfg = dict(spectral_model_cfg.get("local_skip_cfg", {}))
            out["spectral"]["local_skip_cfg"] = {
                "enabled": bool(local_skip_cfg.get("enabled", False)),
                "init_scale": float(local_skip_cfg.get("init_scale", 0.0)),
            }
            axis_mix_cfg = dict(spectral_model_cfg.get("axis_mix_cfg", {}))
            out["spectral"]["axis_mix_cfg"] = {
                "enabled": bool(axis_mix_cfg.get("enabled", False)),
                "init_h": float(axis_mix_cfg.get("init_h", 1.0)),
                "init_w": float(axis_mix_cfg.get("init_w", 1.0)),
            }
        values = self._first_list(contract_samples, "target_vars_effective")
        if values is not None:
            out["target_vars_effective"] = [str(x) for x in values]
        v = self._first_str(contract_samples, "target_family_effective")
        if v is not None:
            out["target_family_effective"] = v
        v = self._first_str(contract_samples, "input_features_mode")
        if v is not None:
            out["input_features_mode"] = v
        values = self._first_list(contract_samples, "input_feature_channels")
        if values is not None:
            out["input_feature_channels"] = [str(x) for x in values]
        v = self._first_str(contract_samples, f"{prefix}_backend_effective")
        if v is not None:
            out[f"{prefix}_backend_effective"] = v
        values = self._first_list(contract_samples, f"{prefix}_input_channels_effective")
        if values is not None:
            out[f"{prefix}_input_channels_effective"] = [str(x) for x in values]
        v = self._first_str(contract_samples, f"{prefix}_input_features_mode_effective")
        if v is not None:
            out[f"{prefix}_input_features_mode_effective"] = v
        v = self._first_str(contract_samples, f"{prefix}_selection_mode_effective")
        if v is not None:
            out[f"{prefix}_selection_mode_effective"] = v
        d = self._first_dict(contract_samples, f"{prefix}_optimizer_effective")
        if d is not None:
            out[f"{prefix}_optimizer_effective"] = d
        d = self._first_dict(contract_samples, "feature")
        if d is not None:
            out["feature"] = d
        d = self._first_dict(contract_samples, "spectral")
        if d is not None:
            out["spectral"] = d
        return out

    def _aggregate_fno_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        fno_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._aggregate_spectral_contract(
            train_cfg=train_cfg,
            y_vars=y_vars,
            model_key="fno",
            prefix="fno",
            contract_samples=fno_contract_samples,
        )

    def _aggregate_ffno_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        ffno_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._aggregate_spectral_contract(
            train_cfg=train_cfg,
            y_vars=y_vars,
            model_key="ffno",
            prefix="ffno",
            contract_samples=ffno_contract_samples,
        )

    def _emit_spectral_contract_resolved(
        self,
        *,
        resolved: dict[str, Any],
        prefix: str,
        contract_effective: dict[str, Any],
        default_target_vars: list[str],
    ) -> None:
        self._record_model_contract(
            resolved=resolved,
            model_name=prefix,
            contract=contract_effective,
        )
        self._emit_display_projection(
            resolved=resolved,
            prefix=prefix,
            contract=contract_effective,
            default_target_vars=default_target_vars,
            backend_key=f"{prefix}_backend_effective",
            selection_key=f"{prefix}_selection_mode_effective",
        )

    def _aggregate_coord_mlp_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        coord_mlp_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        _ = train_cfg
        return self._first_contract_sample(
            coord_mlp_contract_samples,
            defaults={
                "model_type_effective": "coord_mlp",
                "backend_effective": "torch",
                "target_family_effective": "allvars",
                "target_vars_effective": list(y_vars),
                "input_features_mode": "geom_feature_pack",
                "input_feature_channels": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
                "selection_mode_effective": "last",
                "selection_weights_effective": {},
            },
        )

    def _aggregate_deeponet_pod_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        deeponet_pod_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        _ = train_cfg
        return self._first_contract_sample(
            deeponet_pod_contract_samples,
            defaults={
                "model_type_effective": "deeponet_pod",
                "target_family_effective": "allvars",
                "target_vars_effective": list(y_vars),
                "basis_rank_by_var": {},
                "basis_fit_scope_effective": "train_only",
                "basis_center_effective": True,
                "basis_per_var_effective": True,
                "selection_mode_effective": "last",
                "selection_weights_effective": {},
                "model_cfg_effective": {},
            },
        )

    def _aggregate_deeponet_contract_effective(
        self,
        *,
        train_cfg: dict[str, Any],
        y_vars: list[str],
        deeponet_contract_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        _ = train_cfg
        return self._first_contract_sample(
            deeponet_contract_samples,
            defaults={
                "target_family_effective": "allvars",
                "target_vars_effective": list(y_vars),
                "selection_mode_effective": "last",
                "selection_weights_effective": {},
                "operator_mode_effective": "plain",
                "feature": {},
                "optimizer_effective": {},
                "distance_transform_effective": {},
                "strict_mainline_effective": False,
            },
        )



__all__ = ["BenchmarkContractEmitter"]
