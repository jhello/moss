import pytest

from sources.geometry import Point


def test_constructor():
    p = Point()
    assert p.x == pytest.approx(0)
    assert p.x == pytest.approx(0)
    assert p.x == pytest.approx(0)


def test_constructor_1():
    p = Point(1,2,3)
    assert p.x == pytest.approx(1)
    assert p.y == pytest.approx(2)
    assert p.z == pytest.approx(3)

def test_distance_to():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 2, 2)
    assert p1.distance_to(p2) == pytest.approx(3.0)