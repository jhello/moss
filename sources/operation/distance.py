from functools import singledispatch
from typing import Any

from data.topo import Vertex
from sources.geometry import Point


@singledispatch
def distance(a: Any, b: Any) -> float:
    raise NotImplementedError(f"Distance not implemented for types {type(a)} and {type(b)}")

# Point - Point
@distance.register
def _(a: Point, b: Point) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return (dx**2 + dy**2 + dz**2) ** 0.5

# Vertex - Vertex
@distance.register
def _(a: Vertex, b: Vertex) -> float:
    return distance(a.location(), b.location())
