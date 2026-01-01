from sources.solver.Darwin.geometry.ReproduciblePoint import ReproduciblePoint


def test_reproduce():

    parent = ReproduciblePoint(1.0, 2.0, 3.0)
    offspring = parent.reproduce(n_offspring=5, max_deviation=0.1)

    assert len(offspring) == 5
    assert offspring[0] != offspring[1]
    for child in offspring:
        assert isinstance(child, ReproduciblePoint)
        assert abs(child.x - parent.x) <= 0.1
        assert abs(child.y - parent.y) <= 0.1
        assert abs(child.z - parent.z) <= 0.1
