import unittest
from typing import Final

from sources.data import Case
from sources.data.constraint import VertexVertexDistance
from sources.data.topo import Vertex, Solid
from sources.geometry.Point import Point
from sources.geometry.Cartesian import Cartesian
from sources.operation import distance
from sources.solver import Status
from sources.solver.Darwin import DarwinSolver
from sources.solver.Darwin.optimizer import DarwinOptimizer, _detect_device


class TestDetectDevice(unittest.TestCase):
    """Test device detection logic."""

    def test_detect_device_returns_string(self):
        device = _detect_device()
        self.assertIn(device, ('cpu', 'cuda'))

    def test_detect_device_returns_valid_device(self):
        """Device detection should return a valid device string."""
        device = _detect_device()
        self.assertIn(device, ('cpu', 'cuda'))


class TestDarwinOptimizer(unittest.TestCase):
    """Test DarwinOptimizer core functionality."""

    def _create_test_case(self):
        """Create a simple test case with two solids and a vertex-vertex distance constraint."""
        case = Case('TestOptimizer', 'Test case for DarwinOptimizer')

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

        return case, vertex_a, vertex_b

    def test_optimizer_initialization(self):
        """Test that optimizer initializes correctly with CPU device."""
        optimizer = DarwinOptimizer(device='cpu')
        self.assertEqual(optimizer.device, 'cpu')
        self.assertFalse(optimizer._use_gpu)

    def test_optimizer_constants(self):
        """Test that optimizer constants are set correctly."""
        self.assertEqual(DarwinOptimizer.CONVERGENCE_THRESHOLD, 1e-6)
        self.assertEqual(DarwinOptimizer.MAX_ITERATIONS, 10000)
        self.assertEqual(DarwinOptimizer.OFFSPRING_PER_GENERATION, 100)
        self.assertEqual(DarwinOptimizer.MAX_DEVIATION, 1.0)

    def test_convert_to_reproductive(self):
        """Test converting a regular Cartesian to ReproductiveCartesian."""
        from sources.solver.Darwin.geometry import ReproductiveCartesian

        cartesian = Cartesian()
        reproductive = DarwinOptimizer._convert_to_reproductive(cartesian)
        self.assertIsInstance(reproductive, ReproductiveCartesian)

    def test_clone_cartesian(self):
        """Test cloning a ReproductiveCartesian."""
        from sources.solver.Darwin.geometry import ReproductiveCartesian

        cartesian = Cartesian()
        reproductive = DarwinOptimizer._convert_to_reproductive(cartesian)
        cloned = DarwinOptimizer._clone_cartesian(reproductive)
        self.assertIsInstance(cloned, ReproductiveCartesian)

    def test_cpu_evaluate_offspring(self):
        """Test CPU-based offspring evaluation returns valid indices."""
        case, vertex_a, vertex_b = self._create_test_case()
        constraint = case.constraints[0]

        from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator
        from sources.solver.Darwin.geometry import ReproductiveCartesian, ReproduciblePoint

        evaluator = VertexVertexDistanceEvaluator(constraint, case.solids)
        optimizer = DarwinOptimizer(device='cpu')

        # Convert to reproductive with ReproduciblePoint origins
        solid_a = case.solids[0]
        solid_b = case.solids[1]

        # Create ReproductiveCartesian with ReproduciblePoint components
        origin_a = ReproduciblePoint(
            solid_a.coordinates_system.origin[0],
            solid_a.coordinates_system.origin[1],
            solid_a.coordinates_system.origin[2]
        )
        x_point_a = ReproduciblePoint(
            solid_a.coordinates_system.origin[0] + solid_a.coordinates_system.x_axis[0],
            solid_a.coordinates_system.origin[1] + solid_a.coordinates_system.x_axis[1],
            solid_a.coordinates_system.origin[2] + solid_a.coordinates_system.x_axis[2]
        )
        xz_point_a = ReproduciblePoint(
            solid_a.coordinates_system.origin[0] + solid_a.coordinates_system.z_axis[0],
            solid_a.coordinates_system.origin[1] + solid_a.coordinates_system.z_axis[1],
            solid_a.coordinates_system.origin[2] + solid_a.coordinates_system.z_axis[2]
        )
        solid_a.coordinates_system = ReproductiveCartesian(origin=origin_a, x_point=x_point_a, xz_plane_point=xz_point_a)

        origin_b = ReproduciblePoint(
            solid_b.coordinates_system.origin[0],
            solid_b.coordinates_system.origin[1],
            solid_b.coordinates_system.origin[2]
        )
        x_point_b = ReproduciblePoint(
            solid_b.coordinates_system.origin[0] + solid_b.coordinates_system.x_axis[0],
            solid_b.coordinates_system.origin[1] + solid_b.coordinates_system.x_axis[1],
            solid_b.coordinates_system.origin[2] + solid_b.coordinates_system.x_axis[2]
        )
        xz_point_b = ReproduciblePoint(
            solid_b.coordinates_system.origin[0] + solid_b.coordinates_system.z_axis[0],
            solid_b.coordinates_system.origin[1] + solid_b.coordinates_system.z_axis[1],
            solid_b.coordinates_system.origin[2] + solid_b.coordinates_system.z_axis[2]
        )
        solid_b.coordinates_system = ReproductiveCartesian(origin=origin_b, x_point=x_point_b, xz_plane_point=xz_point_b)

        offspring1 = solid_a.coordinates_system.reproduce(10, 1.0)
        offspring2 = solid_b.coordinates_system.reproduce(10, 1.0)

        score, idx1, idx2 = optimizer._cpu_evaluate_offspring(
            evaluator, case.solids, offspring1, offspring2,
            float('inf'), solid_a, solid_b
        )

        self.assertIsInstance(score, float)
        self.assertIsInstance(idx1, int)
        self.assertIsInstance(idx2, int)
        self.assertGreaterEqual(idx1, 0)
        self.assertLess(idx1, 10)
        self.assertGreaterEqual(idx2, 0)
        self.assertLess(idx2, 10)
        self.assertGreaterEqual(score, 0)

    def test_optimize_returns_tuple(self):
        """Test that optimize returns a tuple of (score, iterations)."""
        case, vertex_a, vertex_b = self._create_test_case()
        constraint = case.constraints[0]

        from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator

        evaluator = VertexVertexDistanceEvaluator(constraint, case.solids)
        optimizer = DarwinOptimizer(device='cpu')

        # Temporarily reduce iterations for fast testing
        original_max = DarwinOptimizer.MAX_ITERATIONS
        DarwinOptimizer.MAX_ITERATIONS = 5

        try:
            result = optimizer.optimize(
                evaluator=evaluator,
                solids=case.solids,
                vertex1=vertex_a,
                vertex2=vertex_b
            )
        finally:
            DarwinOptimizer.MAX_ITERATIONS = original_max

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], int)

    def test_optimize_iterations_limit(self):
        """Test that optimization stops at max iterations."""
        case, vertex_a, vertex_b = self._create_test_case()
        constraint = case.constraints[0]

        from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator

        evaluator = VertexVertexDistanceEvaluator(constraint, case.solids)
        optimizer = DarwinOptimizer(device='cpu')

        original_max = DarwinOptimizer.MAX_ITERATIONS
        DarwinOptimizer.MAX_ITERATIONS = 3

        try:
            score, iterations = optimizer.optimize(
                evaluator=evaluator,
                solids=case.solids,
                vertex1=vertex_a,
                vertex2=vertex_b
            )
        finally:
            DarwinOptimizer.MAX_ITERATIONS = original_max

        self.assertLessEqual(iterations, 3)


class TestDarwinSolverIntegration(unittest.TestCase):
    """Integration tests for DarwinSolver with optimizer."""

    def test_solver_with_cpu_device(self):
        """Test solver runs with explicit CPU device."""
        case = Case('TestSolver', 'Integration test')

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

        # Reduce iterations for fast testing
        original_max = DarwinOptimizer.MAX_ITERATIONS
        DarwinOptimizer.MAX_ITERATIONS = 5

        try:
            solver = DarwinSolver(case, device='cpu')
            solver.solve()
            # Solver should reach a terminal status
            self.assertTrue(solver.get_status().is_terminal())
        finally:
            DarwinOptimizer.MAX_ITERATIONS = original_max

    def test_solver_device_parameter(self):
        """Test that solver accepts device parameter."""
        case = Case('TestDevice', 'Device parameter test')
        solver = DarwinSolver(case, device='cpu')
        self.assertEqual(solver.device, 'cpu')


if __name__ == '__main__':
    unittest.main()
