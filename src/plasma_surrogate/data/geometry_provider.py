"""Geometry providers for fixed and parametric runtime modes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from plasma_surrogate.data.geometry_context import (
    GeometryContext,
    build_coord_grid,
    build_distance_fields,
    build_signed_distance_fields,
)


PROVIDER_MODE_FIXED = "fixed"
PROVIDER_MODE_PARAMETRIC_PARTS = "parametric_parts"
PROVIDER_MODES: tuple[str, str] = (PROVIDER_MODE_FIXED, PROVIDER_MODE_PARAMETRIC_PARTS)

_PART_PARAM_RE = re.compile(r"^part\.(?P<part_id>[A-Za-z0-9_\-]+)\.(?P<name>tx|ty|scale_x|scale_y|rotation_deg|fillet)$")
_GAP_PARAM_RE = re.compile(r"^gap\.(?P<name>[A-Za-z0-9_\-]+)$")
_OFFSET_PARAM_RE = re.compile(r"^offset\.(?P<name>x|y|tx|ty)$")


class GeometryProviderLike(Protocol):
    provider_mode: str
    supports_geom_param: bool

    def get(self, geom_ref: dict[str, object] | None = None) -> GeometryContext:
        ...


def normalize_provider_mode(provider_mode: Any) -> str:
    mode = str(provider_mode if provider_mode is not None else PROVIDER_MODE_FIXED).strip().lower()
    if mode not in set(PROVIDER_MODES):
        raise ValueError(f"provider_mode must be one of {list(PROVIDER_MODES)}; got={provider_mode!r}")
    return mode


def _normalize_geom_ref(geom_ref: dict[str, object] | None) -> dict[str, Any]:
    if geom_ref is None:
        return {"geom_id": "default"}
    if not isinstance(geom_ref, dict):
        raise TypeError("geom_ref must be dict when provided")
    out: dict[str, Any] = {"geom_id": str(geom_ref.get("geom_id", "default"))}
    if "geom_param" in geom_ref:
        raw = geom_ref["geom_param"]
        if not isinstance(raw, dict):
            raise TypeError("geom_param must be dict")
        params: dict[str, float] = {}
        for key, value in raw.items():
            name = str(key).strip()
            if not name:
                raise ValueError("geom_param keys must be non-empty strings")
            fval = float(value)
            if not np.isfinite(fval):
                raise ValueError(f"geom_param[{name!r}] must be finite; got={value!r}")
            params[name] = fval
        out["geom_param"] = {k: params[k] for k in sorted(params.keys())}
    return out


class FixedGeometryProvider:
    """v1 fixed geometry provider with SDF-less fallback distance generation."""

    provider_mode: str = PROVIDER_MODE_FIXED
    supports_geom_param: bool = False

    def __init__(self, dataset_root: str | Path, *, coord_grid_source: str = "coord_grid"):
        self.dataset_root = Path(dataset_root)
        self.geometry_root = self.dataset_root / "geometry"
        self.coord_grid_source = str(coord_grid_source).strip().lower()
        if self.coord_grid_source not in {"coord_grid", "rz_linear", "normalized_fallback"}:
            raise ValueError(
                "coord_grid_source must be one of: coord_grid, rz_linear, normalized_fallback"
            )
        self._ctx: GeometryContext | None = None

    def _load_array(self, rel_path: str, default: np.ndarray | None = None) -> np.ndarray | None:
        path = self.geometry_root / rel_path
        if path.exists():
            return np.load(path)
        return default

    def _build(self) -> GeometryContext:
        mask = self._load_array("mask_plasma.npy")
        if mask is None:
            raise FileNotFoundError("geometry/mask_plasma.npy is required")
        mask = mask.astype(np.float32)

        eps = self._load_array("eps.npy", default=np.ones_like(mask, dtype=np.float32))
        if eps is None:
            eps = np.ones_like(mask, dtype=np.float32)

        bc_dir_mask = self._load_array("bc_dir_mask.npy", default=np.zeros_like(mask, dtype=np.float32))
        if bc_dir_mask is None:
            bc_dir_mask = np.zeros_like(mask, dtype=np.float32)

        bc_dir_value = self._load_array("bc_dir_value.npy", default=np.zeros_like(mask, dtype=np.float32))
        if bc_dir_value is None:
            bc_dir_value = np.zeros_like(mask, dtype=np.float32)

        distance_any = self._load_array("distance_any.npy")
        dist0 = self._load_array("dist0.npy")
        distance_signed = self._load_array("distance_signed.npy")

        if distance_signed is None:
            if distance_any is not None:
                distance_signed = np.where(mask > 0.5, distance_any, -distance_any).astype(np.float32)
            else:
                distance_signed = build_signed_distance_fields(mask).astype(np.float32)
        else:
            distance_signed = np.asarray(distance_signed, dtype=np.float32)
            if distance_signed.shape != mask.shape:
                raise ValueError(
                    f"geometry/distance_signed.npy shape mismatch: expected {tuple(mask.shape)}, got {tuple(distance_signed.shape)}"
                )

        if distance_any is None:
            distance_any = np.abs(distance_signed).astype(np.float32)
        else:
            distance_any = np.asarray(distance_any, dtype=np.float32)
            if distance_any.shape != mask.shape:
                raise ValueError(
                    f"geometry/distance_any.npy shape mismatch: expected {tuple(mask.shape)}, got {tuple(distance_any.shape)}"
                )

        if dist0 is None:
            _, dist0 = build_distance_fields(mask, bc_dir_mask)
        else:
            dist0 = np.asarray(dist0, dtype=np.float32)
            if dist0.shape != mask.shape:
                raise ValueError(
                    f"geometry/dist0.npy shape mismatch: expected {tuple(mask.shape)}, got {tuple(dist0.shape)}"
                )

        coord_source = "normalized_fallback"
        coord = None
        if self.coord_grid_source == "coord_grid":
            coord = self._load_array("coord_grid.npy")
            if coord is not None:
                coord_source = "coord_grid"
        if coord is None and self.coord_grid_source in {"coord_grid", "rz_linear"}:
            r_coords = self._load_array("r_coords.npy")
            z_coords = self._load_array("z_coords.npy")
            if r_coords is not None and z_coords is not None:
                r = np.asarray(r_coords, dtype=np.float32).reshape(-1)
                z = np.asarray(z_coords, dtype=np.float32).reshape(-1)
                if int(r.shape[0]) == int(mask.shape[1]) and int(z.shape[0]) == int(mask.shape[0]):
                    rr, zz = np.meshgrid(r, z, indexing="xy")
                    coord = np.stack([rr, zz], axis=0).astype(np.float32)
                    coord_source = "rz_linear"
        if coord is None:
            coord = build_coord_grid(mask.shape)
            coord_source = "normalized_fallback"

        regions = {}
        wafer = self._load_array("wafer_mask.npy")
        if wafer is not None:
            regions["wafer_mask"] = wafer.astype(np.float32)

        return GeometryContext(
            mask_plasma=mask,
            distance_any=distance_any.astype(np.float32),
            dist0=dist0.astype(np.float32),
            eps=eps.astype(np.float32),
            coord_grid=coord.astype(np.float32),
            distance_signed=distance_signed.astype(np.float32),
            regions=regions,
            ops={"hx": 1.0, "hy": 1.0},
            bc_dir_mask=bc_dir_mask.astype(np.float32),
            bc_dir_value=bc_dir_value.astype(np.float32),
            coord_source=coord_source,
        )

    def get(self, geom_ref: dict[str, object] | None = None) -> GeometryContext:
        norm = _normalize_geom_ref(geom_ref)
        if "geom_param" in norm:
            raise ValueError("provider_mode=fixed does not allow geom_ref.geom_param")
        if self._ctx is None:
            self._ctx = self._build()
        return self._ctx


class ParametricPartsGeometryProvider:
    """Manifest-strict provider that rebuilds geometry context from part parameters."""

    provider_mode: str = PROVIDER_MODE_PARAMETRIC_PARTS
    supports_geom_param: bool = True

    def __init__(self, dataset_root: str | Path, *, coord_grid_source: str = "coord_grid"):
        self.dataset_root = Path(dataset_root)
        self.geometry_root = self.dataset_root / "geometry"
        self._base_provider = FixedGeometryProvider(dataset_root, coord_grid_source=coord_grid_source)
        self._manifest_path = self.geometry_root / "parts_manifest.json"
        self._parts_pack_path = self.geometry_root / "parts_pack.npz"
        self._manifest: dict[str, Any] | None = None
        self._part_ids: list[str] | None = None
        self._mask_stack: np.ndarray | None = None
        self._param_specs: dict[str, dict[str, Any]] | None = None
        self._ctx_cache: dict[str, GeometryContext] = {}

    @staticmethod
    def _iter_neighbors(arr: np.ndarray) -> list[np.ndarray]:
        p = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
        return [
            p[:-2, :-2],
            p[:-2, 1:-1],
            p[:-2, 2:],
            p[1:-1, :-2],
            p[1:-1, 1:-1],
            p[1:-1, 2:],
            p[2:, :-2],
            p[2:, 1:-1],
            p[2:, 2:],
        ]

    @classmethod
    def _dilate_binary(cls, mask: np.ndarray, steps: int) -> np.ndarray:
        out = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
        for _ in range(max(0, int(steps))):
            neigh = cls._iter_neighbors(out)
            out = np.maximum.reduce(neigh).astype(np.float32)
        return out

    @classmethod
    def _erode_binary(cls, mask: np.ndarray, steps: int) -> np.ndarray:
        out = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
        for _ in range(max(0, int(steps))):
            neigh = cls._iter_neighbors(out)
            out = np.minimum.reduce(neigh).astype(np.float32)
        return out

    @classmethod
    def _morph_binary(cls, mask: np.ndarray, delta: float, scale_ref: int) -> np.ndarray:
        steps = int(round(abs(float(delta)) * float(max(1, int(scale_ref)))))
        if steps <= 0:
            return (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
        if float(delta) > 0.0:
            return cls._dilate_binary(mask, steps)
        return cls._erode_binary(mask, steps)

    @staticmethod
    def _center_of_mask(mask: np.ndarray) -> tuple[float, float]:
        yy, xx = np.nonzero(mask > 0.5)
        if len(xx) == 0:
            return 0.5, 0.5
        h, w = mask.shape
        cx = float(np.mean(xx)) / float(max(w - 1, 1))
        cy = float(np.mean(yy)) / float(max(h - 1, 1))
        return cx, cy

    @staticmethod
    def _warp_mask_affine(
        mask: np.ndarray,
        *,
        tx: float,
        ty: float,
        scale_x: float,
        scale_y: float,
        rotation_deg: float,
    ) -> np.ndarray:
        src = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
        h, w = src.shape
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yv, xv = np.meshgrid(yy, xx, indexing="ij")
        cx, cy = ParametricPartsGeometryProvider._center_of_mask(src)
        sx = float(scale_x)
        sy = float(scale_y)
        if sx <= 0.0 or sy <= 0.0:
            raise ValueError("part scale_x/scale_y must be > 0")
        theta = np.deg2rad(float(rotation_deg))
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        x_rel = xv - np.float32(cx) - np.float32(tx)
        y_rel = yv - np.float32(cy) - np.float32(ty)
        x_rot = c * x_rel + s * y_rel
        y_rot = -s * x_rel + c * y_rel
        x_src = x_rot / np.float32(sx) + np.float32(cx)
        y_src = y_rot / np.float32(sy) + np.float32(cy)
        ix = np.clip(np.rint(x_src * np.float32(max(w - 1, 1))).astype(np.int64), 0, max(w - 1, 0))
        iy = np.clip(np.rint(y_src * np.float32(max(h - 1, 1))).astype(np.int64), 0, max(h - 1, 0))
        return src[iy, ix].astype(np.float32)

    def _load_manifest_and_pack(self, base_ctx: GeometryContext) -> None:
        if self._manifest is not None and self._part_ids is not None and self._mask_stack is not None and self._param_specs is not None:
            return
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"provider_mode=parametric_parts requires {self._manifest_path}")
        if not self._parts_pack_path.exists():
            raise FileNotFoundError(f"provider_mode=parametric_parts requires {self._parts_pack_path}")
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("parts_manifest.json must be a JSON object")
        param_specs_raw = manifest.get("param_specs")
        if not isinstance(param_specs_raw, dict) or len(param_specs_raw) == 0:
            raise ValueError("parts_manifest.json must include non-empty param_specs object")
        manifest_part_ids_raw = manifest.get("part_ids")
        manifest_part_ids: list[str] | None = None
        if manifest_part_ids_raw is not None:
            if not isinstance(manifest_part_ids_raw, list):
                raise ValueError("parts_manifest.json part_ids must be a list when provided")
            manifest_part_ids = [str(v).strip() for v in manifest_part_ids_raw]
            if any(not v for v in manifest_part_ids):
                raise ValueError("parts_manifest.json part_ids must not contain empty strings")
            if len(set(manifest_part_ids)) != len(manifest_part_ids):
                raise ValueError("parts_manifest.json part_ids must be unique")
        with np.load(self._parts_pack_path, allow_pickle=True) as pack:
            if "mask_stack" not in pack:
                raise ValueError("parts_pack.npz must include mask_stack")
            mask_stack = np.asarray(pack["mask_stack"], dtype=np.float32)
            if mask_stack.ndim != 3:
                raise ValueError("parts_pack.npz mask_stack must be [P,H,W]")
            if not np.all(np.isfinite(mask_stack)):
                raise ValueError("parts_pack.npz mask_stack must contain only finite values")
            if tuple(mask_stack.shape[1:]) != tuple(base_ctx.mask_plasma.shape):
                raise ValueError(
                    "parts_pack.npz mask_stack spatial shape mismatch: "
                    f"expected={tuple(base_ctx.mask_plasma.shape)} got={tuple(mask_stack.shape[1:])}"
                )
            if "part_ids" in pack:
                part_ids = [str(v).strip() for v in np.asarray(pack["part_ids"]).reshape(-1).tolist()]
            else:
                part_ids = [str(v).strip() for v in list(manifest.get("part_ids", []))]
            if any(not v for v in part_ids):
                raise ValueError("parts_pack part_ids must not contain empty strings")
            if len(set(part_ids)) != len(part_ids):
                raise ValueError("parts_pack part_ids must be unique")
            if manifest_part_ids is not None and part_ids and part_ids != manifest_part_ids:
                raise ValueError(
                    "parts_manifest.json part_ids and parts_pack.npz part_ids must match exactly; "
                    f"manifest={manifest_part_ids}, pack={part_ids}"
                )
            if len(part_ids) != int(mask_stack.shape[0]):
                raise ValueError(
                    "parts_pack.npz part_ids length mismatch with mask_stack[0]: "
                    f"{len(part_ids)} != {int(mask_stack.shape[0])}"
                )
        part_set = set(part_ids)
        specs: dict[str, dict[str, Any]] = {}
        for key, raw_spec in param_specs_raw.items():
            name = str(key).strip()
            if not name:
                raise ValueError("parts_manifest param_specs keys must be non-empty strings")
            spec = dict(raw_spec) if isinstance(raw_spec, dict) else {"default": raw_spec}
            part_match = _PART_PARAM_RE.match(name)
            if part_match is not None:
                part_id = str(part_match.group("part_id"))
                if part_id not in part_set:
                    raise ValueError(
                        f"parts_manifest param key references unknown part id: {name!r}; known={sorted(part_set)}"
                    )
            elif _GAP_PARAM_RE.match(name) is not None:
                pass
            elif _OFFSET_PARAM_RE.match(name) is not None:
                pass
            else:
                raise ValueError(
                    "parts_manifest param key is unsupported. "
                    "Use part.<id>.(tx|ty|scale_x|scale_y|rotation_deg|fillet), gap.<name>, or offset.(x|y|tx|ty). "
                    f"got={name!r}"
                )
            spec_min = None
            spec_max = None
            if "min" in spec:
                spec_min = float(spec["min"])
                if not np.isfinite(spec_min):
                    raise ValueError(f"parts_manifest param_specs[{name!r}].min must be finite")
            if "max" in spec:
                spec_max = float(spec["max"])
                if not np.isfinite(spec_max):
                    raise ValueError(f"parts_manifest param_specs[{name!r}].max must be finite")
            if spec_min is not None and spec_max is not None and spec_min > spec_max:
                raise ValueError(
                    f"parts_manifest param_specs[{name!r}] requires min <= max; got min={spec_min}, max={spec_max}"
                )
            if "default" in spec:
                spec_default = float(spec["default"])
                if not np.isfinite(spec_default):
                    raise ValueError(f"parts_manifest param_specs[{name!r}].default must be finite")
                if spec_min is not None and spec_default < spec_min:
                    raise ValueError(
                        f"parts_manifest param_specs[{name!r}].default must be >= min ({spec_min}); got {spec_default}"
                    )
                if spec_max is not None and spec_default > spec_max:
                    raise ValueError(
                        f"parts_manifest param_specs[{name!r}].default must be <= max ({spec_max}); got {spec_default}"
                    )
            specs[name] = spec
        self._manifest = manifest
        self._part_ids = part_ids
        self._mask_stack = mask_stack
        self._param_specs = specs

    def _resolve_effective_params(self, requested: dict[str, float]) -> dict[str, float]:
        if self._param_specs is None:
            raise RuntimeError("param specs are not loaded")
        out: dict[str, float] = {}
        for key, spec in self._param_specs.items():
            part_match = _PART_PARAM_RE.match(key)
            default = spec.get("default")
            if default is None and part_match is not None:
                attr = str(part_match.group("name"))
                default = 1.0 if attr in {"scale_x", "scale_y"} else 0.0
            if default is None:
                default = 0.0
            out[key] = float(default)
        unknown = sorted(set(requested.keys()) - set(out.keys()))
        if unknown:
            raise ValueError(
                "geom_param contains keys not declared in parts_manifest.param_specs: "
                f"{unknown}"
            )
        for key, value in requested.items():
            out[key] = float(value)
        for key, value in out.items():
            if not np.isfinite(value):
                raise ValueError(f"geom_param[{key!r}] must be finite; got={value!r}")
            spec = self._param_specs[key]
            if "min" in spec and value < float(spec["min"]):
                raise ValueError(f"geom_param[{key!r}] must be >= {float(spec['min'])}; got={value}")
            if "max" in spec and value > float(spec["max"]):
                raise ValueError(f"geom_param[{key!r}] must be <= {float(spec['max'])}; got={value}")
        return out

    def _build_context_for_params(
        self,
        *,
        base_ctx: GeometryContext,
        params: dict[str, float],
        geom_id: str,
    ) -> GeometryContext:
        if self._part_ids is None or self._mask_stack is None:
            raise RuntimeError("parts data is not loaded")
        h, w = base_ctx.mask_plasma.shape
        global_tx = float(params.get("offset.x", params.get("offset.tx", 0.0)))
        global_ty = float(params.get("offset.y", params.get("offset.ty", 0.0)))
        per_part_masks: list[np.ndarray] = []
        for idx, part_id in enumerate(self._part_ids):
            prefix = f"part.{part_id}."
            tx = float(params.get(prefix + "tx", 0.0)) + global_tx
            ty = float(params.get(prefix + "ty", 0.0)) + global_ty
            sx = float(params.get(prefix + "scale_x", 1.0))
            sy = float(params.get(prefix + "scale_y", 1.0))
            rot = float(params.get(prefix + "rotation_deg", 0.0))
            fillet = float(params.get(prefix + "fillet", 0.0))
            warped = self._warp_mask_affine(
                self._mask_stack[idx],
                tx=tx,
                ty=ty,
                scale_x=sx,
                scale_y=sy,
                rotation_deg=rot,
            )
            warped = self._morph_binary(warped, delta=fillet, scale_ref=max(h, w))
            per_part_masks.append(warped.astype(np.float32))
        union_solid = np.maximum.reduce(per_part_masks).astype(np.float32) if per_part_masks else np.zeros((h, w), dtype=np.float32)
        gap_delta = float(sum(value for key, value in params.items() if _GAP_PARAM_RE.match(key) is not None))
        if gap_delta != 0.0:
            union_solid = self._morph_binary(union_solid, delta=gap_delta, scale_ref=max(h, w))
        mask_base = (np.asarray(base_ctx.mask_plasma, dtype=np.float32) > 0.5).astype(np.float32)
        mask_plasma = (mask_base * (1.0 - union_solid)).astype(np.float32)
        if float(np.sum(mask_plasma)) <= 0.0:
            raise ValueError("parametric_parts geom_param produced empty plasma region")
        distance_signed = build_signed_distance_fields(mask_plasma).astype(np.float32)
        bc_mask = (
            np.zeros_like(mask_plasma, dtype=np.float32)
            if base_ctx.bc_dir_mask is None
            else np.asarray(base_ctx.bc_dir_mask, dtype=np.float32)
        )
        distance_any, dist0 = build_distance_fields(mask_plasma, bc_mask)
        regions = dict(base_ctx.regions or {})
        regions["solid_union_mask"] = union_solid.astype(np.float32)
        regions["part_mask_stack"] = np.stack(per_part_masks, axis=0).astype(np.float32) if per_part_masks else np.zeros((0, h, w), dtype=np.float32)
        regions["geom_param_values"] = np.array([params[k] for k in sorted(params.keys())], dtype=np.float32)
        regions["geom_param_keys"] = np.array(sorted(params.keys()), dtype=object)
        return GeometryContext(
            mask_plasma=mask_plasma.astype(np.float32),
            distance_any=np.asarray(distance_any, dtype=np.float32),
            dist0=np.asarray(dist0, dtype=np.float32),
            eps=np.asarray(base_ctx.eps, dtype=np.float32),
            coord_grid=np.asarray(base_ctx.coord_grid, dtype=np.float32),
            distance_signed=np.asarray(distance_signed, dtype=np.float32),
            regions=regions,
            ops=dict(base_ctx.ops or {"hx": 1.0, "hy": 1.0}),
            bc_dir_mask=None if base_ctx.bc_dir_mask is None else np.asarray(base_ctx.bc_dir_mask, dtype=np.float32),
            bc_dir_value=None if base_ctx.bc_dir_value is None else np.asarray(base_ctx.bc_dir_value, dtype=np.float32),
            coord_source=str(base_ctx.coord_source),
        )

    @staticmethod
    def _cache_key(*, geom_id: str, params: dict[str, float]) -> str:
        payload = json.dumps(
            {"geom_id": str(geom_id), "geom_param": {k: float(params[k]) for k in sorted(params.keys())}},
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, geom_ref: dict[str, object] | None = None) -> GeometryContext:
        norm = _normalize_geom_ref(geom_ref)
        base_ctx = self._base_provider.get({"geom_id": str(norm.get("geom_id", "default"))})
        if "geom_param" not in norm:
            return base_ctx
        self._load_manifest_and_pack(base_ctx)
        params = self._resolve_effective_params(dict(norm.get("geom_param", {})))
        cache_key = self._cache_key(geom_id=str(norm.get("geom_id", "default")), params=params)
        if cache_key not in self._ctx_cache:
            self._ctx_cache[cache_key] = self._build_context_for_params(
                base_ctx=base_ctx,
                params=params,
                geom_id=str(norm.get("geom_id", "default")),
            )
        return self._ctx_cache[cache_key]


def build_geometry_provider(
    dataset_root: str | Path,
    *,
    provider_mode: Any = PROVIDER_MODE_FIXED,
    coord_grid_source: str = "coord_grid",
) -> GeometryProviderLike:
    mode = normalize_provider_mode(provider_mode)
    if mode == PROVIDER_MODE_FIXED:
        return FixedGeometryProvider(dataset_root, coord_grid_source=coord_grid_source)
    if mode == PROVIDER_MODE_PARAMETRIC_PARTS:
        return ParametricPartsGeometryProvider(dataset_root, coord_grid_source=coord_grid_source)
    raise ValueError(f"Unsupported provider_mode: {provider_mode!r}")


__all__ = [
    "GeometryProviderLike",
    "PROVIDER_MODE_FIXED",
    "PROVIDER_MODE_PARAMETRIC_PARTS",
    "PROVIDER_MODES",
    "FixedGeometryProvider",
    "ParametricPartsGeometryProvider",
    "build_geometry_provider",
    "normalize_provider_mode",
]
