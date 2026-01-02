from functools import singledispatch
from typing import Any

from geometry import Cartesian
from sources.geometry import Point


@singledispatch
def transform(a: Any, b: Any) -> float:
    raise NotImplementedError(f"Transform not implemented for types {type(a)} and {type(b)}")

# Point - Point
@transform.register
def _(point_local: Point, coordinate_system: Cartesian) -> Point:
    point_global = coordinate_system.to_array() @ point_local.to_array_homogeneous()
    return Point(point_global[:3])
