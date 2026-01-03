from typing import Final

from data import Case
from data.constraint import VertexVertexDistance
from data.topo import Vertex, Solid
from geometry import Point, Cartesian
from operation import distance
from solver import Status
from solver.Darwin import DarwinSolver

def test_darwin_solver_basic():
    case = Case('TestCase001', 'A test case for DarwinSolver')

    SQRT2: Final[float] = 2 ** 0.5

    origin_a = Point(SQRT2, 0, 0)
    x_axis_point_a = Point(1.5 * SQRT2, 0, 0.5 * SQRT2)
    z_axis_point_a = Point(0.5 * SQRT2, 0, 0.5 * SQRT2)
    cartesian_a = Cartesian(origin_a, x_axis_point_a, z_axis_point_a)
    solid_a = Solid(cartesian_a)
    vertex_a = Vertex()
    solid_a.add_vertex(vertex_a)
    case.add_solid(solid_a)

    origin_b = Point(0, SQRT2, 0)
    x_axis_point_b = Point(1, SQRT2, 0)
    z_axis_point_b = Point(0, 0.5 * SQRT2, 0.5 * SQRT2)
    cartesian_b = Cartesian(origin_b, x_axis_point_b, z_axis_point_b)
    solid_b = Solid(cartesian_b)
    vertex_b = Vertex()
    solid_b.add_vertex(vertex_b)
    case.add_solid(solid_b)

    vertices_distance_constraint = VertexVertexDistance(vertex_a, vertex_b, 5.0)
    case.add_constraint(vertices_distance_constraint)

    solver = DarwinSolver(case)
    solver.solve()

    assert solver.get_status() == Status.SUCCESS
    assert distance(vertex_a, vertex_b) == 5.0

