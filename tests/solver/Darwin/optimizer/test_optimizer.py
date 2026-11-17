import math
import time
import unittest
from typing import Final

import numpy as np

from sources.data import Case
from sources.data.constraint import VertexVertexDistance
from sources.data.topo import Vertex, Solid
from sources.geometry.Point import Point
from sources.geometry.Cartesian import Cartesian
from sources.operation import distance, transform
from sources.solver import Status
from sources.solver.Darwin import DarwinSolver
from sources.solver.Darwin.optimizer import DarwinOptimizer, _detect_device


def make_test_case():
    """Two solids and a vertex-vertex distance constraint of length 5."""
    case = Case('TestOptimizer', 'Test case for DarwinOptimizer')

    SQRT2: Final[float] = 2 ** 0.5

    cartesian_a = Cartesian(
        Point(SQRT2, 0, 0),
        Point(1.5 * SQRT2, 0, 0.5 * SQRT2),
        Point(0.5 * SQRT2, 0, 0.5 * SQRT2)
    )
    solid_a = Solid(cartesian_a)
    vertex_a = Vertex()
    solid_a.add_vertex(vertex_a)
    case.add_solid(solid_a)

    cartesian_b = Cartesian(
        Point(0, SQRT2, 0),
        Point(1, SQRT2, 0),
        Point(0, 0.5 * SQRT2, 0.5 * SQRT2)
    )
    solid_b = Solid(cartesian_b)
    vertex_b = Vertex()
    solid_b.add_vertex(vertex_b)
    case.add_solid(solid_b)

    case.add_constraint(VertexVertexDistance(vertex_a, vertex_b, 5.0))
    return case, vertex_a, vertex_b


# Parameters shared by the fast unit tests (the production defaults are large:
# 10000 individuals, 1e-6 threshold).
FAST_PARAMS = dict(
    max_population=20,
    offspring_per_individual=3,
    max_deviation=1.0,
    convergence_threshold=1e-4,
    max_iterations=500,
    seed=42
)


class TestDetectDevice(unittest.TestCase):
    """Test device detection logic."""

    def test_detect_device_returns_string(self):
        self.assertIn(_detect_device(), ('cpu', 'cuda'))

    def test_detect_device_returns_valid_device(self):
        device = _detect_device()
        self.assertIn(device, ('cpu', 'cuda'))


class TestDarwinOptimizer(unittest.TestCase):
    """Test DarwinOptimizer core functionality."""

    def test_optimizer_initialization(self):
        optimizer = DarwinOptimizer(device='cpu')
        self.assertEqual(optimizer.device, 'cpu')
        self.assertFalse(optimizer._use_gpu)
        self.assertEqual(optimizer.max_population, DarwinOptimizer.MAX_POPULATION)
        self.assertEqual(optimizer.convergence_threshold, 1e-6)

    def test_parameters_are_configurable(self):
        """All four tunables can be set directly on the optimizer (also by tests)."""
        optimizer = DarwinOptimizer(
            device='cpu',
            max_population=50,
            offspring_per_individual=7,
            max_deviation=0.25,
            convergence_threshold=1e-3,
            max_iterations=12,
            seed=7
        )
        self.assertEqual(optimizer.max_population, 50)
        self.assertEqual(optimizer.offspring_per_individual, 7)
        self.assertEqual(optimizer.max_deviation, 0.25)
        self.assertEqual(optimizer.convergence_threshold, 1e-3)
        self.assertEqual(optimizer.max_iterations, 12)

    def test_points_roundtrip(self):
        """Packing/unpacking a candidate must preserve its coordinate systems."""
        from sources.solver.Darwin.geometry import ReproductiveCartesian, ReproduciblePoint

        cs1 = ReproductiveCartesian(
            ReproduciblePoint(1, 2, 3),
            ReproduciblePoint(2, 2, 3),
            ReproduciblePoint(1, 2, 4)
        )
        cs2 = ReproductiveCartesian()
        points = DarwinOptimizer._points_from_cartesian(cs1, cs2)
        self.assertEqual(points.shape, (1, 6, 3))

        new1, new2 = DarwinOptimizer._cartesian_from_points(points[0])
        np.testing.assert_allclose(new1.to_array(), cs1.to_array(), atol=1e-12)
        np.testing.assert_allclose(new2.to_array(), cs2.to_array(), atol=1e-12)

    def test_reproduce_offspring_count_and_deviation(self):
        optimizer = DarwinOptimizer(device='cpu', max_deviation=0.1, seed=1)
        parents = np.zeros((5, 6, 3))
        offspring = optimizer._reproduce(parents, 7)
        self.assertEqual(offspring.shape, (35, 6, 3))
        self.assertLessEqual(np.abs(offspring).max(), 0.1)
        self.assertLessEqual(np.abs(offspring[:7] - parents[0]).max(), 0.1)

    def test_select_survivors_enforces_population_cap(self):
        """Only the best `max_population` individuals survive reproduction."""
        optimizer = DarwinOptimizer(device='cpu', max_population=4)
        population = np.repeat(np.arange(10, dtype=float).reshape(10, 1, 1), 18, axis=2)
        scores = np.array([9., 0., 5., 2., 7., 1., 8., 3., 6., 4.])

        survivors, survivor_scores = optimizer._select_survivors(population, scores)
        self.assertEqual(survivors.shape[0], 4)
        np.testing.assert_array_equal(survivor_scores, np.array([0., 1., 2., 3.]))
        np.testing.assert_array_equal(
            survivors[:, 0, 0], np.array([1., 5., 3., 7.])
        )

    def test_cpu_evaluate_scores_matches_evaluator(self):
        case, vertex_a, vertex_b = make_test_case()
        constraint = case.constraints[0]

        from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator

        evaluator = VertexVertexDistanceEvaluator(constraint, case.solids)
        optimizer = DarwinOptimizer(device='cpu')
        population = DarwinOptimizer._points_from_cartesian(
            case.solids[0].coordinates_system, case.solids[1].coordinates_system
        )
        p1 = np.array([vertex_a.point.x, vertex_a.point.y, vertex_a.point.z, 1.0])
        p2 = np.array([vertex_b.point.x, vertex_b.point.y, vertex_b.point.z, 1.0])

        scores = optimizer._evaluate_scores(population, p1, p2, evaluator.distance)
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(math.isclose(scores[0], evaluator.evaluate(), abs_tol=1e-9))

    def test_population_cap_applied_every_generation(self):
        """Generation size never exceeds the configured total population cap."""
        case, vertex_a, vertex_b = make_test_case()
        from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator

        evaluator = VertexVertexDistanceEvaluator(case.constraints[0], case.solids)
        optimizer = DarwinOptimizer(device='cpu', max_population=20,
                                    offspring_per_individual=3, max_iterations=10, seed=1)
        optimizer.optimize(evaluator, case.solids, vertex_a, vertex_b)

        self.assertEqual(len(optimizer.generation_stats), 10)
        self.assertTrue(all(s['population'] <= 20 for s in optimizer.generation_stats))

    def test_eliminated_individuals_never_breed_again(self):
        """Every breeding parent must come from the previous generation's survivors."""
        case, vertex_a, vertex_b = make_test_case()
        from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator

        evaluator = VertexVertexDistanceEvaluator(case.constraints[0], case.solids)
        optimizer = DarwinOptimizer(device='cpu', max_population=5,
                                    offspring_per_individual=2, max_iterations=4, seed=3)

        parents_seen = []
        survivors_seen = []
        original_reproduce = optimizer._reproduce
        original_select = optimizer._select_survivors

        def spy_reproduce(population, n_offspring):
            parents_seen.append(population.copy())
            return original_reproduce(population, n_offspring)

        def spy_select(population, scores):
            survivors, survivor_scores = original_select(population, scores)
            survivors_seen.append(survivors.copy())
            return survivors, survivor_scores

        optimizer._reproduce = spy_reproduce
        optimizer._select_survivors = spy_select
        optimizer.optimize(evaluator, case.solids, vertex_a, vertex_b)

        self.assertEqual(len(parents_seen), 4)
        for parents, previous_survivors in zip(parents_seen[1:], survivors_seen[:-1]):
            self.assertTrue(np.array_equal(parents, previous_survivors))

    def test_optimize_returns_tuple_and_honours_iteration_limit(self):
        case, vertex_a, vertex_b = make_test_case()
        from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator

        evaluator = VertexVertexDistanceEvaluator(case.constraints[0], case.solids)
        optimizer = DarwinOptimizer(device='cpu', max_population=20,
                                    offspring_per_individual=3, max_iterations=3, seed=2)
        result = optimizer.optimize(evaluator, case.solids, vertex_a, vertex_b)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], int)
        self.assertLessEqual(result[1], 3)

    def test_optimize_converges_with_test_parameters(self):
        case, vertex_a, vertex_b = make_test_case()
        from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator

        evaluator = VertexVertexDistanceEvaluator(case.constraints[0], case.solids)
        optimizer = DarwinOptimizer(device='cpu', max_population=300,
                                    offspring_per_individual=2, max_deviation=1.0,
                                    convergence_threshold=1e-4,
                                    max_iterations=2000, seed=17)
        score, generations = optimizer.optimize(
            evaluator, case.solids, vertex_a, vertex_b
        )
        self.assertLess(score, 1e-4)
        self.assertLess(generations, 2000)

        global_a = transform(vertex_a.point, case.solids[0].coordinates_system)
        global_b = transform(vertex_b.point, case.solids[1].coordinates_system)
        self.assertTrue(math.isclose(distance(global_a, global_b), 5.0, abs_tol=1e-3))


class TestDarwinSolverIntegration(unittest.TestCase):
    """Integration tests for DarwinSolver with optimizer."""

    def test_solver_with_cpu_device(self):
        case, _, _ = make_test_case()
        solver = DarwinSolver(case, device='cpu', **FAST_PARAMS)
        solver.solve()

        self.assertEqual(solver.get_status(), Status.SUCCESS)
        self.assertLess(solver.final_score, FAST_PARAMS['convergence_threshold'])
        self.assertTrue(all(
            s['population'] <= FAST_PARAMS['max_population']
            for s in solver.optimizer.generation_stats
        ))

    def test_solver_device_parameter(self):
        case = Case('TestDevice', 'Device parameter test')
        solver = DarwinSolver(case, device='cpu', **FAST_PARAMS)
        self.assertEqual(solver.device, 'cpu')
        self.assertFalse(solver.optimizer._use_gpu)
        self.assertEqual(solver.optimizer.max_population, FAST_PARAMS['max_population'])


class TestCpuGpuComparison(unittest.TestCase):
    """Compare convergence speed of the same search on CPU and GPU."""

    @unittest.skipUnless(_detect_device() == 'cuda', 'CUDA device not available')
    def test_convergence_steps_cpu_vs_gpu(self):
        params = dict(
            max_population=10000,
            offspring_per_individual=2,
            max_deviation=1.0,
            convergence_threshold=1e-6,
            max_iterations=2000,
            seed=42
        )
        results = {}
        for device in ('cpu', 'cuda'):
            case, _, _ = make_test_case()
            solver = DarwinSolver(case, device=device, **params)
            start = time.perf_counter()
            solver.solve()
            elapsed = time.perf_counter() - start
            results[device] = (solver.iterations, solver.final_score, elapsed,
                               solver.get_status())

        print('\nCPU / GPU convergence comparison (identical seed and parameters):')
        print('| Device | Generations to converge | Final score | Wall time (s) | Status |')
        print('| ------ | ----------------------- | ----------- | ------------- | ------ |')
        for device, (gens, score, elapsed, status) in results.items():
            print(f'| {device:6s} | {gens:23d} | {score:.2e} | {elapsed:13.3f} | {status.name} |')

        for device, (gens, score, _, status) in results.items():
            self.assertEqual(status, Status.SUCCESS, device)
            self.assertLess(score, params['convergence_threshold'], device)
            self.assertLess(gens, params['max_iterations'], device)


if __name__ == '__main__':
    unittest.main()
