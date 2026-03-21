"""Benchmark profile lock definitions for mainline runs."""

from __future__ import annotations

import copy
from typing import Any


BENCHMARK_PROFILE_LOCKS: dict[str, dict[str, Any]] = {
    "m7_global_frozen_ref": {
        "profile": "m7_global_frozen_ref",
        "dimension": "2d_steady",
        "models": ["global_mlp"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_periodic_common": {
        "profile": "m7_periodic_common",
        "dimension": "2d_steady",
        "models": ["global_mlp", "unet", "fno", "deeponet_plasma"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_unet_isolated": {
        "profile": "m7_unet_isolated",
        "dimension": "2d_steady",
        "models": ["unet"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_unetpp_isolated": {
        "profile": "m7_unetpp_isolated",
        "dimension": "2d_steady",
        "models": ["unetpp"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_unetpp_attn_isolated": {
        "profile": "m7_unetpp_attn_isolated",
        "dimension": "2d_steady",
        "models": ["unetpp_attn"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_fno_isolated": {
        "profile": "m7_fno_isolated",
        "dimension": "2d_steady",
        "models": ["fno"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_ffno_isolated": {
        "profile": "m7_ffno_isolated",
        "dimension": "2d_steady",
        "models": ["ffno"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_deeponet_isolated": {
        "profile": "m7_deeponet_isolated",
        "dimension": "2d_steady",
        "models": ["deeponet_plasma"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_coord_mlp_fourier_experimental": {
        "profile": "m7_coord_mlp_fourier_experimental",
        "dimension": "2d_steady",
        "models": ["coord_mlp_fourier"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_coord_mlp_siren_experimental": {
        "profile": "m7_coord_mlp_siren_experimental",
        "dimension": "2d_steady",
        "models": ["coord_mlp_siren"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
    "m7_deeponet_pod_experimental": {
        "profile": "m7_deeponet_pod_experimental",
        "dimension": "2d_steady",
        "models": ["deeponet_pod"],
        "phi_mode": "direct",
        "axis_mode": "steady",
        "poisson_refine_iters": 0,
        "sensor_query_seed": None,
        "sensor_query_spec_hash": "",
    },
}

PROFILE_ALIASES = {
    "m7_common": "m7_periodic_common",
    "m7_unet": "m7_unet_isolated",
    "m7_unetpp": "m7_unetpp_isolated",
    "m7_unetpp_attn": "m7_unetpp_attn_isolated",
    "m7_fno": "m7_fno_isolated",
    "m7_ffno": "m7_ffno_isolated",
    "m7_deeponet": "m7_deeponet_isolated",
}


def resolve_profile_lock(profile: str) -> dict[str, Any]:
    key = PROFILE_ALIASES.get(str(profile), str(profile))
    raw = BENCHMARK_PROFILE_LOCKS.get(key)
    if raw is None:
        raise ValueError(f"Unsupported benchmark profile: {profile}")
    return copy.deepcopy(raw)


__all__ = ["BENCHMARK_PROFILE_LOCKS", "resolve_profile_lock"]
