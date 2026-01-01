import pytest

from sources.geometry import Point
from sources.operation import distance


def test_point_point():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 2, 2)
    assert distance(p1, p2) == pytest.approx(3.0)