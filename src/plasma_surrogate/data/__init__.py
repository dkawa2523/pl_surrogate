"""Data layer contracts and providers."""

from plasma_surrogate.data.case_record import CaseRecord
from plasma_surrogate.data.geometry_context import GeometryContext, build_distance_fields, build_signed_distance_fields
from plasma_surrogate.data.geometry_provider import (
    FixedGeometryProvider,
    ParametricPartsGeometryProvider,
    build_geometry_provider,
)

__all__ = [
    "CaseRecord",
    "GeometryContext",
    "FixedGeometryProvider",
    "ParametricPartsGeometryProvider",
    "build_geometry_provider",
    "build_distance_fields",
    "build_signed_distance_fields",
]
