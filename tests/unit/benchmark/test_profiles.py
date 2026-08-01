from __future__ import annotations

import pytest

from plasma_surrogate.benchmark.profiles import resolve_profile_lock


@pytest.mark.parametrize(
    "name",
    [
        "m7_global_frozen_ref",
        "m7_unetpp_attn_isolated",
        "m7_fno_isolated",
        "m7_ffno_isolated",
        "m7_coord_mlp_siren_isolated",
        "m7_deeponet_isolated",
    ],
)
def test_resolve_profile_lock_supported(name: str):
    profile = resolve_profile_lock(name)
    assert profile["profile"]
    assert isinstance(profile["models"], list)
    assert profile["phi_mode"] in {"direct", "poisson_hybrid", "deeponet_poisson"}


def test_resolve_profile_lock_alias_and_copy():
    a = resolve_profile_lock("m7_fno")
    b = resolve_profile_lock("m7_fno")
    assert a["profile"] == "m7_fno_isolated"
    a["models"].append("x")
    assert "x" not in b["models"]


def test_resolve_profile_lock_ffno_alias():
    profile = resolve_profile_lock("m7_ffno")
    assert profile["profile"] == "m7_ffno_isolated"
    assert profile["models"] == ["ffno"]


def test_resolve_profile_lock_coord_mlp_siren_alias():
    profile = resolve_profile_lock("m7_coord_mlp_siren")
    assert profile["profile"] == "m7_coord_mlp_siren_isolated"
    assert profile["models"] == ["coord_mlp_siren"]


def test_resolve_profile_lock_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported benchmark profile"):
        resolve_profile_lock("unknown_profile")
