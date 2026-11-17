import numpy as np
from typing import List, Optional, Tuple

from sources.data.topo import Solid, Vertex
from sources.geometry import Cartesian
from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator
from sources.solver.Darwin.geometry import ReproductiveCartesian, ReproduciblePoint


def _detect_device() -> str:
    """Detect available compute device. Returns 'cuda' if GPU is available, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'


class DarwinOptimizer:
    """
    Evolutionary optimizer that uses reproductive Cartesian coordinate systems
    to minimize constraint scores.

    A population of candidate solutions is maintained. Every candidate holds the
    two coordinate systems involved in the constraint; its score is the
    constraint score of that pair. In each generation every surviving candidate
    breeds ``offspring_per_individual`` children, the children are scored and
    sorted, and only the best ``max_population`` candidates survive. All others
    are eliminated and cannot reproduce again.

    Supports both GPU (PyTorch CUDA) and CPU (NumPy) computation. All tunable
    parameters are exposed on the constructor so tests can override them.
    """

    # Defaults (may be overridden per instance, see __init__).
    CONVERGENCE_THRESHOLD = 1e-6
    MAX_ITERATIONS = 10000
    MAX_POPULATION = 10000
    OFFSPRING_PER_INDIVIDUAL = 2
    MAX_DEVIATION = 1.0

    # Backwards-compatible alias for the previous constant name.
    OFFSPRING_PER_GENERATION = OFFSPRING_PER_INDIVIDUAL

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        max_population: Optional[int] = None,
        offspring_per_individual: Optional[int] = None,
        max_deviation: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        max_iterations: Optional[int] = None,
        seed: Optional[int] = None
    ):
        self.max_population = (
            self.MAX_POPULATION if max_population is None else int(max_population)
        )
        self.offspring_per_individual = (
            self.OFFSPRING_PER_INDIVIDUAL
            if offspring_per_individual is None
            else int(offspring_per_individual)
        )
        self.max_deviation = (
            self.MAX_DEVIATION if max_deviation is None else float(max_deviation)
        )
        self.convergence_threshold = (
            self.CONVERGENCE_THRESHOLD
            if convergence_threshold is None
            else float(convergence_threshold)
        )
        self.max_iterations = (
            self.MAX_ITERATIONS if max_iterations is None else int(max_iterations)
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if self.max_population < 1 or self.offspring_per_individual < 1:
            raise ValueError("max_population and offspring_per_individual must be >= 1")
        if self.max_deviation < 0:
            raise ValueError("max_deviation must be >= 0")

        # Per-generation diagnostics: {"generation", "population", "best", "mean"}
        self.generation_stats: List[dict] = []

        self.device = device or _detect_device()
        self._torch = None
        self._use_gpu = False
        self._setup_compute_backend()

    def _setup_compute_backend(self):
        """Initialize the compute backend (PyTorch if available, otherwise numpy)."""
        if self.device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
                self._torch = torch
                self._use_gpu = True
                self.device = str(torch.device('cuda'))
                print(f"[DarwinOptimizer] Using GPU (CUDA) on device: {torch.cuda.get_device_name(0)}")
            except (ImportError, RuntimeError) as exc:
                print(f"[DarwinOptimizer] {exc}, falling back to CPU (numpy)")
                self.device = 'cpu'
                self._torch = None
                self._use_gpu = False
        else:
            print("[DarwinOptimizer] Using CPU (numpy)")

    def optimize(
        self,
        evaluator: VertexVertexDistanceEvaluator,
        solids: List[Solid],
        vertex1: Vertex,
        vertex2: Vertex
    ) -> Tuple[float, int]:
        """
        Run evolutionary optimization to minimize the constraint score.

        Args:
            evaluator: The constraint evaluator
            solids: List of solids whose coordinate systems may be updated
            vertex1: First vertex of the constraint
            vertex2: Second vertex of the constraint

        Returns:
            Tuple of (final_score, generations) where ``generations`` is the
            number of generations that were bred.
        """
        solid1, solid2 = self._find_solids(solids, vertex1, vertex2)
        if solid1 is None or solid2 is None:
            print("[DarwinOptimizer] Could not find both solids for constraint, skipping")
            return evaluator.evaluate(), 0

        # Make sure both coordinate systems can be described by reproducible points.
        if not isinstance(solid1.coordinates_system, ReproductiveCartesian):
            solid1.coordinates_system = self._convert_to_reproductive(solid1.coordinates_system)
        if not isinstance(solid2.coordinates_system, ReproductiveCartesian):
            solid2.coordinates_system = self._convert_to_reproductive(solid2.coordinates_system)

        p1 = np.array([
            evaluator.vertex1.point.x, evaluator.vertex1.point.y,
            evaluator.vertex1.point.z, 1.0
        ], dtype=np.float64)
        p2 = np.array([
            evaluator.vertex2.point.x, evaluator.vertex2.point.y,
            evaluator.vertex2.point.z, 1.0
        ], dtype=np.float64)
        target = float(evaluator.distance)

        # Generation 1: fill the population from the seed pair.
        seed = self._points_from_cartesian(
            solid1.coordinates_system, solid2.coordinates_system
        )
        population = self._reproduce(seed, self.max_population)
        scores = self._evaluate_scores(population, p1, p2, target)
        population, scores = self._select_survivors(population, scores)
        best_points = population[0].copy()
        best_score = float(scores[0])

        self.generation_stats = []
        self._record_generation(1, population, scores)

        generations = 1
        while (
            best_score >= self.convergence_threshold
            and generations < self.max_iterations
        ):
            generations += 1
            offspring = self._reproduce(population, self.offspring_per_individual)
            scores = self._evaluate_scores(offspring, p1, p2, target)
            population, scores = self._select_survivors(offspring, scores)

            if float(scores[0]) < best_score:
                best_score = float(scores[0])
                best_points = population[0].copy()

            self._record_generation(generations, population, scores)

        # Apply the best candidate ever found (the elite never breeds, but it is
        # always the one written back to the solids).
        solid1.coordinates_system, solid2.coordinates_system = \
            self._cartesian_from_points(best_points)

        if best_score < self.convergence_threshold:
            print(f"[DarwinOptimizer] Converged at generation {generations} "
                  f"with score {best_score:.2e}")
        else:
            print(f"[DarwinOptimizer] Max generations ({self.max_iterations}) "
                  f"reached. Final score: {best_score:.2e}")

        return best_score, generations

    @staticmethod
    def _find_solids(solids, vertex1, vertex2):
        solid1 = solid2 = None
        for solid in solids:
            if vertex1 in solid.vertices:
                solid1 = solid
            if vertex2 in solid.vertices:
                solid2 = solid
            if solid1 is not None and solid2 is not None:
                break
        return solid1, solid2

    def _record_generation(self, generation: int, population: np.ndarray,
                           scores: np.ndarray) -> None:
        self.generation_stats.append({
            'generation': generation,
            'population': int(population.shape[0]),
            'best': float(scores[0]),
            'mean': float(np.mean(scores)),
        })
        if generation % 100 == 0:
            print(f"[DarwinOptimizer] Generation {generation}: "
                  f"best_score = {float(scores[0]):.2e}")

    # ------------------------------------------------------------------
    # Population representation
    # ------------------------------------------------------------------
    @staticmethod
    def _defining_points(cartesian: Cartesian) -> np.ndarray:
        """Return the 3 points defining a coordinate system as an (3, 3) array."""
        if isinstance(cartesian, ReproductiveCartesian):
            return np.array([
                cartesian._origin_point.to_array(),
                cartesian._x_point.to_array(),
                cartesian._xz_plane_point.to_array(),
            ], dtype=np.float64)
        # Plain Cartesian stores numpy arrays: origin plus unit axis vectors.
        origin = np.asarray(cartesian.origin, dtype=np.float64)
        return np.array([
            origin,
            origin + cartesian.x_axis,
            origin + cartesian.z_axis,
        ], dtype=np.float64)

    @classmethod
    def _points_from_cartesian(cls, cs1: Cartesian, cs2: Cartesian) -> np.ndarray:
        """Pack a pair of coordinate systems into an (1, 6, 3) array.

        The six points are [origin_1, x_point_1, xz_point_1,
        origin_2, x_point_2, xz_point_2].
        """
        return np.stack([
            np.concatenate([cls._defining_points(cs1), cls._defining_points(cs2)])
        ]).astype(np.float64)

    @staticmethod
    def _cartesian_from_points(points: np.ndarray) -> Tuple[ReproductiveCartesian, ReproductiveCartesian]:
        """Rebuild the two reproductive coordinate systems from a candidate."""
        o1, x1, z1, o2, x2, z2 = points
        cs1 = ReproductiveCartesian(
            origin=ReproduciblePoint(*o1),
            x_point=ReproduciblePoint(*x1),
            xz_plane_point=ReproduciblePoint(*z1)
        )
        cs2 = ReproductiveCartesian(
            origin=ReproduciblePoint(*o2),
            x_point=ReproduciblePoint(*x2),
            xz_plane_point=ReproduciblePoint(*z2)
        )
        return cs1, cs2

    def _reproduce(self, population: np.ndarray, n_offspring: int) -> np.ndarray:
        """Breed ``n_offspring`` children from every individual.

        Returns an array of shape (population_size * n_offspring, 6, 3).
        The same RNG is used for both CPU and GPU runs, which makes the
        convergence comparison reproducible.
        """
        total = population.shape[0] * n_offspring
        noise = self.rng.uniform(
            -self.max_deviation, self.max_deviation, size=(total, 6, 3)
        )
        return np.repeat(population, n_offspring, axis=0) + noise

    def _select_survivors(self, population: np.ndarray, scores: np.ndarray):
        """Keep only the best ``max_population`` candidates (truncation selection)."""
        keep = min(self.max_population, population.shape[0])
        order = np.argsort(scores)[:keep]
        return population[order], scores[order]

    # ------------------------------------------------------------------
    # Batched scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _numpy_matrices(points: np.ndarray):
        """Vectorized equivalent of Cartesian.to_array() for (N, 3) point triples."""
        origin = points[:, 0, :]
        x_point = points[:, 1, :]
        xz_point = points[:, 2, :]

        x_vec = x_point - origin
        x_norm = np.linalg.norm(x_vec, axis=1, keepdims=True)
        x_axis = x_vec / np.maximum(x_norm, 1e-15)

        y_vec = np.cross(xz_point - origin, x_axis)
        y_norm = np.linalg.norm(y_vec, axis=1, keepdims=True)
        y_axis = y_vec / np.maximum(y_norm, 1e-15)
        z_axis = np.cross(x_axis, y_axis)

        matrices = np.zeros((points.shape[0], 4, 4), dtype=np.float64)
        matrices[:, :3, 0] = x_axis
        matrices[:, :3, 1] = y_axis
        matrices[:, :3, 2] = z_axis
        matrices[:, :3, 3] = origin
        matrices[:, 3, 3] = 1.0

        valid = (x_norm[:, 0] > 1e-9) & (y_norm[:, 0] > 1e-9)
        return matrices, valid

    @staticmethod
    def _torch_matrices(torch, points):
        origin = points[:, 0, :]
        x_point = points[:, 1, :]
        xz_point = points[:, 2, :]

        x_vec = x_point - origin
        x_norm = torch.linalg.norm(x_vec, dim=1, keepdim=True)
        x_axis = x_vec / torch.clamp(x_norm, min=1e-15)

        y_vec = torch.cross(xz_point - origin, x_axis, dim=1)
        y_norm = torch.linalg.norm(y_vec, dim=1, keepdim=True)
        y_axis = y_vec / torch.clamp(y_norm, min=1e-15)
        z_axis = torch.cross(x_axis, y_axis, dim=1)

        matrices = torch.zeros(
            (points.shape[0], 4, 4), dtype=points.dtype, device=points.device
        )
        matrices[:, :3, 0] = x_axis
        matrices[:, :3, 1] = y_axis
        matrices[:, :3, 2] = z_axis
        matrices[:, :3, 3] = origin
        matrices[:, 3, 3] = 1.0

        valid = (x_norm[:, 0] > 1e-9) & (y_norm[:, 0] > 1e-9)
        return matrices, valid

    def _evaluate_scores(
        self, population: np.ndarray, p1: np.ndarray, p2: np.ndarray, target: float
    ) -> np.ndarray:
        """Score every candidate. Returns a numpy array of shape (N,)."""
        if self._use_gpu:
            torch = self._torch
            device = torch.device(self.device)
            points = torch.from_numpy(population).to(device=device, dtype=torch.float32)
            local1 = torch.from_numpy(p1).to(device=device, dtype=torch.float32)
            local2 = torch.from_numpy(p2).to(device=device, dtype=torch.float32)

            m1, valid1 = self._torch_matrices(torch, points[:, :3, :])
            m2, valid2 = self._torch_matrices(torch, points[:, 3:, :])

            g1 = torch.einsum('nij,j->ni', m1, local1)
            g2 = torch.einsum('nij,j->ni', m2, local2)
            diff = g1[:, :3] - g2[:, :3]
            scores = torch.abs(torch.linalg.norm(diff, dim=1) - target)
            scores[~(valid1 & valid2)] = float('inf')
            return scores.cpu().numpy()

        m1, valid1 = self._numpy_matrices(population[:, :3, :])
        m2, valid2 = self._numpy_matrices(population[:, 3:, :])

        g1 = np.einsum('nij,j->ni', m1, p1)
        g2 = np.einsum('nij,j->ni', m2, p2)
        diff = g1[:, :3] - g2[:, :3]
        scores = np.abs(np.linalg.norm(diff, axis=1) - target)
        scores[~(valid1 & valid2)] = float('inf')
        return scores

    # ------------------------------------------------------------------
    # Helpers kept from the original implementation
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_to_reproductive(cartesian: Cartesian) -> ReproductiveCartesian:
        """Convert a regular Cartesian to a ReproductiveCartesian."""
        origin = np.asarray(cartesian.origin, dtype=float)
        return ReproductiveCartesian(
            origin=ReproduciblePoint(*origin),
            x_point=ReproduciblePoint(*(origin + cartesian.x_axis)),
            xz_plane_point=ReproduciblePoint(*(origin + cartesian.z_axis))
        )

    @classmethod
    def _clone_cartesian(cls, cartesian: Cartesian) -> ReproductiveCartesian:
        """Create a clone of a ReproductiveCartesian."""
        return cls._convert_to_reproductive(cartesian)
