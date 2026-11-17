from typing import Final
import math

from sources.data import Case
from sources.data.constraint import VertexVertexDistance
from sources.data.topo import Vertex, Solid
from sources.geometry.Point import Point
from sources.geometry.Cartesian import Cartesian
from sources.operation import distance, transform
from sources.solver import Status
from sources.solver.Darwin import DarwinSolver

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
    
    # 优化器更新的是 solid 的坐标系，vertex 的局部坐标不变
    # 需要通过 transform 获取全局坐标后计算距离
    solid_a = case.solids[0]
    solid_b = case.solids[1]
    global_point_a = transform(vertex_a.point, solid_a.coordinates_system)
    global_point_b = transform(vertex_b.point, solid_b.coordinates_system)
    actual_distance = distance(global_point_a, global_point_b)
    assert math.isclose(actual_distance, 5.0, abs_tol=1e-3), f"Expected distance ~5.0, got {actual_distance}"

