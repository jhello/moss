import pytest

from data.topo import Vertex
from sources.geometry import Point
from sources.operation import distance


def test_point_point():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 2, 2)
    assert distance(p1, p2) == pytest.approx(3.0)

def test_vertex_vertex():
    v1 = Vertex()
    v2 = Vertex()
    assert distance(v1, v2) == pytest.approx(0.0)