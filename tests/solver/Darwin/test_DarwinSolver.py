from data import Case
from data.constraint import VertexVertexDistance
from operation import distance
from solver import Status
from solver.Darwin import DarwinSolver

def test_darwin_solver_basic():
    case = Case(case_id="TestCase001", description="A test case for DarwinSolver")
    vertex_a = case.add_vertex()
    vertex_b = case.add_vertex()
    vertices_distance_constraint = VertexVertexDistance(vertex1_id=vertex_a, vertex2_id=vertex_b, distance=5.0)
    case.add_constraint(vertices_distance_constraint)
    solver = DarwinSolver(case)
    solver.solve()
    assert solver.get_status() == Status.SUCCESS
    assert distance(vertex_a, vertex_b) == 5.0

