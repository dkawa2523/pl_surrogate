"""Simple scalers and fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class BaseScaler:
    def fit(self, x: np.ndarray) -> "BaseScaler":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def to_dict(self) -> dict[str, Any]:
        return {"type": "none"}


@dataclass
class StandardScaler(BaseScaler):
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        arr = np.asarray(x, dtype=np.float64)
        self.mean = np.mean(arr, axis=0)
        self.std = np.std(arr, axis=0)
        self.std = np.where(self.std < 1e-12, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")
        out = np.asarray(x, dtype=np.float64) * self.std + self.mean
        if not np.all(np.isfinite(out)):
            raise ValueError("StandardScaler.inverse_transform produced non-finite values")
        return out

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.std is None:
            return {"type": "zscore", "mean": None, "std": None}
        return {"type": "zscore", "mean": np.asarray(self.mean).tolist(), "std": np.asarray(self.std).tolist()}


@dataclass
class RobustScaler(BaseScaler):
    median: np.ndarray | None = None
    iqr: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "RobustScaler":
        arr = np.asarray(x, dtype=np.float64)
        self.median = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        self.iqr = np.where((q75 - q25) < 1e-12, 1.0, q75 - q25)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.median is None or self.iqr is None:
            raise RuntimeError("Scaler not fitted")
        return (np.asarray(x, dtype=np.float64) - self.median) / self.iqr

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.median is None or self.iqr is None:
            raise RuntimeError("Scaler not fitted")
        out = np.asarray(x, dtype=np.float64) * self.iqr + self.median
        if not np.all(np.isfinite(out)):
            raise ValueError("RobustScaler.inverse_transform produced non-finite values")
        return out

    def to_dict(self) -> dict[str, Any]:
        if self.median is None or self.iqr is None:
            return {"type": "robust", "median": None, "iqr": None}
        return {
            "type": "robust",
            "median": np.asarray(self.median).tolist(),
            "iqr": np.asarray(self.iqr).tolist(),
        }


@dataclass
class MinMaxScaler(BaseScaler):
    min_: np.ndarray | None = None
    max_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        arr = np.asarray(x, dtype=np.float64)
        self.min_ = np.min(arr, axis=0)
        self.max_ = np.max(arr, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler not fitted")
        denom = np.where((self.max_ - self.min_) < 1e-12, 1.0, self.max_ - self.min_)
        return (np.asarray(x, dtype=np.float64) - self.min_) / denom

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler not fitted")
        denom = np.where((self.max_ - self.min_) < 1e-12, 1.0, self.max_ - self.min_)
        out = np.asarray(x, dtype=np.float64) * denom + self.min_
        if not np.all(np.isfinite(out)):
            raise ValueError("MinMaxScaler.inverse_transform produced non-finite values")
        return out

    def to_dict(self) -> dict[str, Any]:
        if self.min_ is None or self.max_ is None:
            return {"type": "minmax", "min": None, "max": None}
        return {"type": "minmax", "min": np.asarray(self.min_).tolist(), "max": np.asarray(self.max_).tolist()}


class IdentityScaler(BaseScaler):
    def to_dict(self) -> dict[str, Any]:
        return {"type": "none"}


class ScalerFactory:
    @staticmethod
    def create(kind: str) -> BaseScaler:
        kind = str(kind).strip().lower()
        if kind == "zscore":
            return StandardScaler()
        if kind == "robust":
            return RobustScaler()
        if kind == "minmax":
            return MinMaxScaler()
        if kind == "none":
            return IdentityScaler()
        raise ValueError(f"Unknown scaler type: {kind}")

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> BaseScaler:
        kind = str(raw.get("type", "none")).strip().lower()
        if kind == "zscore":
            sc = StandardScaler()
            if raw.get("mean") is not None:
                sc.mean = np.array(raw["mean"], dtype=np.float64)
            if raw.get("std") is not None:
                sc.std = np.array(raw["std"], dtype=np.float64)
            return sc
        if kind == "robust":
            sc = RobustScaler()
            if raw.get("median") is not None:
                sc.median = np.array(raw["median"], dtype=np.float64)
            if raw.get("iqr") is not None:
                sc.iqr = np.array(raw["iqr"], dtype=np.float64)
            return sc
        if kind == "minmax":
            sc = MinMaxScaler()
            if raw.get("min") is not None:
                sc.min_ = np.array(raw["min"], dtype=np.float64)
            if raw.get("max") is not None:
                sc.max_ = np.array(raw["max"], dtype=np.float64)
            return sc
        return IdentityScaler()


_ALLOWED_VALUE_TRANSFORMS = {"identity", "log10", "log10_floor", "log1p", "signed_log1p"}
_ALLOWED_SCALERS = {"none", "zscore", "minmax", "robust"}
_ALLOWED_FIT_SCOPES = {"all", "plasma_only"}
_ALLOWED_CLIP_MODES = {"none", "quantile"}


def _to_float32_checked(values: np.ndarray, *, label: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.asarray(values, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{label} produced non-finite float32 values")
    return out


def _default_target_transform(var: str, *, default_scaler: str, default_fit_scope: str) -> dict[str, Any]:
    return {
        "value_transform": "identity",
        "scaler": str(default_scaler),
        "fit_scope": str(default_fit_scope),
        "clip": {"mode": "none"},
        "floor": 0.0,
    }


def _normalize_target_transform_spec(
    var: str,
    raw: dict[str, Any] | None,
    *,
    default_scaler: str,
    default_fit_scope: str,
) -> dict[str, Any]:
    spec = _default_target_transform(var, default_scaler=default_scaler, default_fit_scope=default_fit_scope)
    if raw is not None:
        spec.update(dict(raw))
    value_transform = str(spec.get("value_transform", "identity")).strip().lower()
    if value_transform not in _ALLOWED_VALUE_TRANSFORMS:
        raise ValueError(f"target_transforms.{var}.value_transform must be one of: {sorted(_ALLOWED_VALUE_TRANSFORMS)}")
    scaler_kind = str(spec.get("scaler", default_scaler)).strip().lower()
    if scaler_kind not in _ALLOWED_SCALERS:
        raise ValueError(f"target_transforms.{var}.scaler must be one of: {sorted(_ALLOWED_SCALERS)}")
    fit_scope = str(spec.get("fit_scope", "all")).strip().lower()
    if fit_scope not in _ALLOWED_FIT_SCOPES:
        raise ValueError(f"target_transforms.{var}.fit_scope must be one of: {sorted(_ALLOWED_FIT_SCOPES)}")
    clip_raw = dict(spec.get("clip", {}))
    clip_mode = str(clip_raw.get("mode", "none")).strip().lower()
    if clip_mode not in _ALLOWED_CLIP_MODES:
        raise ValueError(f"target_transforms.{var}.clip.mode must be one of: {sorted(_ALLOWED_CLIP_MODES)}")
    clip_spec: dict[str, Any] = {"mode": clip_mode}
    if clip_mode == "quantile":
        q_low = float(clip_raw.get("q_low", 0.01))
        q_high = float(clip_raw.get("q_high", 0.99))
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError(f"target_transforms.{var}.clip requires 0 <= q_low < q_high <= 1")
        clip_spec["q_low"] = q_low
        clip_spec["q_high"] = q_high
    floor_raw = spec.get("floor", 0.0)
    floor = 0.0 if floor_raw is None else float(floor_raw)
    return {
        "value_transform": value_transform,
        "scaler": scaler_kind,
        "fit_scope": fit_scope,
        "clip": clip_spec,
        "floor": floor,
    }


def _apply_value_transform(
    values: np.ndarray,
    *,
    mode: str,
    var: str,
    floor: float = 0.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if mode == "identity":
        return arr
    if mode == "log10":
        if np.any(arr <= 0.0):
            raise ValueError(f"target_transforms.{var}.value_transform=log10 requires strictly positive values")
        return np.log10(arr.astype(np.float64)).astype(np.float32)
    if mode == "log10_floor":
        f = max(float(floor), 1.0e-30)
        return np.log10(np.maximum(arr.astype(np.float64), f)).astype(np.float32)
    if mode == "log1p":
        x = arr.astype(np.float64)
        x = np.maximum(x, float(floor))
        if np.any(x <= -1.0):
            raise ValueError(f"target_transforms.{var}.value_transform=log1p requires values > -1")
        return np.log1p(x).astype(np.float32)
    if mode == "signed_log1p":
        x = arr.astype(np.float64)
        return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)
    raise ValueError(f"Unsupported value_transform: {mode}")


def _apply_inverse_value_transform(
    values: np.ndarray,
    *,
    mode: str,
    floor: float = 0.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if mode == "identity":
        return arr
    if mode in {"log10", "log10_floor"}:
        with np.errstate(over="ignore", invalid="ignore"):
            out = np.power(10.0, arr)
        floor_eff = max(float(floor), 1.0e-30) if mode == "log10_floor" else float(floor)
        out = np.maximum(out, floor_eff)
        if not np.all(np.isfinite(out)):
            raise ValueError(f"target inverse value_transform={mode} produced non-finite values")
        return out
    if mode == "log1p":
        with np.errstate(over="ignore", invalid="ignore"):
            out = np.expm1(arr)
        out = np.maximum(out, float(floor))
        return out
    if mode == "signed_log1p":
        with np.errstate(over="ignore", invalid="ignore"):
            return np.sign(arr) * np.expm1(np.abs(arr))
    raise ValueError(f"Unsupported value_transform: {mode}")


@dataclass
class TransformBundle:
    """Holds fitted cond/y scalers and provides consistent apply/inverse paths."""

    cond_scaler: BaseScaler
    y_scalers: dict[str, BaseScaler]
    y_order: list[str] = field(default_factory=list)
    cond_dim: int | None = None
    fit_policy: str = "all"
    mask_applied: bool = False
    target_transforms: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _resolve_cond_dim(self) -> int | None:
        return None if self.cond_dim is None else int(self.cond_dim)

    def _validate_cond_matrix(self, cond_matrix: np.ndarray, *, op: str) -> np.ndarray:
        arr = np.asarray(cond_matrix, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{op} expects cond_matrix with shape [N,D], got ndim={arr.ndim}")
        expected = self._resolve_cond_dim()
        if expected is not None and int(arr.shape[1]) != int(expected):
            raise ValueError(f"{op} cond feature dim mismatch: expected {int(expected)}, got {int(arr.shape[1])}")
        return arr

    def transform_cond(self, cond_matrix: np.ndarray) -> np.ndarray:
        arr = self._validate_cond_matrix(cond_matrix, op="transform_cond")
        return self.cond_scaler.transform(arr).astype(np.float32)

    def inverse_cond(self, cond_matrix: np.ndarray) -> np.ndarray:
        arr = self._validate_cond_matrix(cond_matrix, op="inverse_cond")
        return self.cond_scaler.inverse_transform(arr).astype(np.float32)

    def _apply_var(self, values: np.ndarray, var: str, inverse: bool) -> np.ndarray:
        scaler = self.y_scalers[var]
        spec = dict(
            self.target_transforms.get(
                var,
                _default_target_transform(var, default_scaler="zscore", default_fit_scope="all"),
            )
        )
        value_mode = str(spec.get("value_transform", "identity")).strip().lower()
        floor_raw = spec.get("floor", 0.0)
        floor = 0.0 if floor_raw is None else float(floor_raw)
        clip_spec = dict(spec.get("clip", {}))
        flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if inverse:
            out = scaler.inverse_transform(flat)
            out = _apply_inverse_value_transform(out, mode=value_mode, floor=floor)
        else:
            out = _apply_value_transform(flat, mode=value_mode, var=var, floor=floor)
            if str(clip_spec.get("mode", "none")).strip().lower() == "quantile":
                lo = float(clip_spec.get("clip_low", clip_spec.get("q_low_value", 0.0)))
                hi = float(clip_spec.get("clip_high", clip_spec.get("q_high_value", 0.0)))
                out = np.clip(out, lo, hi).astype(np.float32)
            out = scaler.transform(out)
        return _to_float32_checked(out.reshape(values.shape), label=f"target '{var}' transform")

    def transform_field_dict(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for var, val in fields.items():
            if var in self.y_scalers:
                out[var] = self._apply_var(val, var, inverse=False)
            else:
                out[var] = np.asarray(val, dtype=np.float32)
        return out

    def inverse_field_dict(self, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for var, val in fields.items():
            if var in self.y_scalers:
                out[var] = self._apply_var(val, var, inverse=True)
            else:
                out[var] = np.asarray(val, dtype=np.float32)
        return out

    def transform_fields(self, fields: np.ndarray) -> np.ndarray:
        arr = np.asarray(fields, dtype=np.float32).copy()
        for i, var in enumerate(self.y_order):
            arr[:, i, ...] = self._apply_var(arr[:, i, ...], var, inverse=False)
        return arr

    def inverse_fields(self, fields: np.ndarray) -> np.ndarray:
        arr = np.asarray(fields, dtype=np.float32).copy()
        for i, var in enumerate(self.y_order):
            arr[:, i, ...] = self._apply_var(arr[:, i, ...], var, inverse=True)
        return arr

    def transform_point_targets(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).copy()
        for i, var in enumerate(self.y_order):
            arr[:, i] = self._apply_var(arr[:, i], var, inverse=False).reshape(-1)
        return arr

    def inverse_point_targets(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).copy()
        for i, var in enumerate(self.y_order):
            arr[:, i] = self._apply_var(arr[:, i], var, inverse=True).reshape(-1)
        return arr

    def to_dict(self) -> dict[str, Any]:
        return {
            "cond_scaler": self.cond_scaler.to_dict(),
            "y_scalers": {k: v.to_dict() for k, v in self.y_scalers.items()},
            "y_order": list(self.y_order),
            "cond_dim": self._resolve_cond_dim(),
            "fit_policy": str(self.fit_policy),
            "mask_applied": bool(self.mask_applied),
            "target_transforms": dict(self.target_transforms),
        }

    @classmethod
    def from_dict(
        cls,
        cond_scaler: dict[str, Any],
        y_scalers: dict[str, dict[str, Any]],
        y_order: list[str] | None = None,
        cond_dim: int | None = None,
        fit_policy: str | None = None,
        mask_applied: bool | None = None,
        target_transforms: dict[str, dict[str, Any]] | None = None,
    ) -> "TransformBundle":
        order = list(y_order) if y_order is not None else list(y_scalers.keys())
        inferred_dim = cond_dim
        if inferred_dim is None:
            inferred_dim = cond_scaler.get("cond_dim")
        if inferred_dim is None:
            raise ValueError("TransformBundle.from_dict requires cond_dim")
        return cls(
            cond_scaler=ScalerFactory.from_dict(cond_scaler),
            y_scalers={k: ScalerFactory.from_dict(v) for k, v in y_scalers.items()},
            y_order=order,
            cond_dim=None if inferred_dim is None else int(inferred_dim),
            fit_policy=str(fit_policy or cond_scaler.get("fit_policy", "all")),
            mask_applied=bool(mask_applied if mask_applied is not None else cond_scaler.get("mask_applied", False)),
            target_transforms=dict(
                target_transforms
                if target_transforms is not None
                else cond_scaler.get("target_transforms", {})
            ),
        )


def fit_scalers_train_only(
    cond_matrix: np.ndarray,
    y_by_var: dict[str, np.ndarray],
    train_indices: np.ndarray,
    mask_plasma: np.ndarray | None = None,
    scaler_fit_policy: str = "all",
    cond_scaler_type: str = "zscore",
    y_scaler_type: str = "zscore",
    target_transforms: dict[str, Any] | None = None,
) -> TransformBundle:
    """Fit cond/y scalers on train split only."""

    tr = np.asarray(train_indices, dtype=np.int64)
    if tr.size == 0:
        raise ValueError("train_indices must not be empty")

    cond_scaler = ScalerFactory.create(cond_scaler_type).fit(np.asarray(cond_matrix, dtype=np.float32)[tr])

    fit_policy = str(scaler_fit_policy).strip().lower()
    if fit_policy not in {"all", "plasma_only"}:
        raise ValueError(f"Unknown scaler_fit_policy: {scaler_fit_policy}")
    mask_applied = False
    mask2d: np.ndarray | None = None
    if fit_policy == "plasma_only":
        if mask_plasma is None:
            raise ValueError("scaler_fit_policy='plasma_only' requires mask_plasma")
        mask_arr = np.asarray(mask_plasma, dtype=np.float32)
        if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
            mask_arr = mask_arr[0]
        if mask_arr.ndim != 2:
            raise ValueError(f"mask_plasma must be [H,W] (or [1,H,W]), got {mask_arr.shape}")
        mask2d = (mask_arr > 0.5)
        if not np.any(mask2d):
            raise ValueError("mask_plasma has no active cells")

    y_scalers: dict[str, BaseScaler] = {}
    resolved_target_transforms: dict[str, dict[str, Any]] = {}
    raw_transforms = dict(target_transforms or {})
    for var, values in y_by_var.items():
        spec = _normalize_target_transform_spec(
            var,
            dict(raw_transforms.get(var, {})) if var in raw_transforms else None,
            default_scaler=y_scaler_type,
            default_fit_scope=fit_policy,
        )
        sampled_all = np.asarray(values, dtype=np.float32)[tr]
        fit_scope = str(spec.get("fit_scope", fit_policy)).strip().lower()
        if fit_scope == "plasma_only" and mask2d is not None and sampled_all.ndim >= 3:
            sampled = sampled_all[..., mask2d].reshape(-1, 1)
            if sampled.size > 0:
                mask_applied = True
            else:
                sampled = sampled_all.reshape(-1, 1)
        else:
            sampled = sampled_all.reshape(-1, 1)
        sampled_fit = _apply_value_transform(
            sampled,
            mode=str(spec["value_transform"]),
            var=var,
            floor=float(spec.get("floor", 0.0)),
        )
        clip_spec = dict(spec.get("clip", {}))
        if str(clip_spec.get("mode", "none")).strip().lower() == "quantile":
            flat = sampled_fit.reshape(-1).astype(np.float64)
            q_lo = float(clip_spec.get("q_low", 0.01))
            q_hi = float(clip_spec.get("q_high", 0.99))
            lo = float(np.quantile(flat, q_lo))
            hi = float(np.quantile(flat, q_hi))
            sampled_fit = np.clip(sampled_fit, lo, hi).astype(np.float32)
            clip_spec["clip_low"] = float(lo)
            clip_spec["clip_high"] = float(hi)
        spec["clip"] = clip_spec
        scaler_kind = str(spec.get("scaler", y_scaler_type)).strip().lower()
        y_scalers[var] = ScalerFactory.create(scaler_kind).fit(sampled_fit)
        resolved_target_transforms[var] = spec

    return TransformBundle(
        cond_scaler=cond_scaler,
        y_scalers=y_scalers,
        y_order=list(y_by_var.keys()),
        cond_dim=int(np.asarray(cond_matrix, dtype=np.float32).shape[1]),
        fit_policy=fit_policy,
        mask_applied=mask_applied,
        target_transforms=resolved_target_transforms,
    )
