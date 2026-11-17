import numpy as np
from typing import List, Tuple, Optional

from sources.data.topo import Solid, Vertex
from sources.geometry import Cartesian, Point
from sources.solver.Darwin.geometry import ReproductiveCartesian, ReproduciblePoint
from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator


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

    Supports both GPU (PyTorch CUDA) and CPU computation.
    """

    CONVERGENCE_THRESHOLD = 1e-6
    MAX_ITERATIONS = 10000
    OFFSPRING_PER_GENERATION = 100
    MAX_DEVIATION = 1.0

    def __init__(self, device: Optional[str] = None):
        self.device = device or _detect_device()
        self._torch = None
        self._use_gpu = False
        self._setup_compute_backend()

    def _setup_compute_backend(self):
        """Initialize the compute backend (PyTorch if available, otherwise numpy)."""
        if self.device == 'cuda':
            try:
                import torch
                self._torch = torch
                self._use_gpu = True
                print(f"[DarwinOptimizer] Using GPU (CUDA) on device: {torch.cuda.get_device_name(0)}")
            except ImportError:
                print("[DarwinOptimizer] PyTorch not available, falling back to CPU (numpy)")
                self._device = 'cpu'
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
            Tuple of (final_score, iterations)
        """
        # Find the two solids involved in this constraint
        solid1 = None
        solid2 = None
        for solid in solids:
            if vertex1 in solid.vertices:
                solid1 = solid
            if vertex2 in solid.vertices:
                solid2 = solid
            if solid1 is not None and solid2 is not None:
                break

        if solid1 is None or solid2 is None:
            print("[DarwinOptimizer] Could not find both solids for constraint, skipping")
            return evaluator.evaluate(), 0

        # Ensure coordinate systems are reproductive
        if not isinstance(solid1.coordinates_system, ReproductiveCartesian):
            solid1.coordinates_system = self._convert_to_reproductive(solid1.coordinates_system)
        if not isinstance(solid2.coordinates_system, ReproductiveCartesian):
            solid2.coordinates_system = self._convert_to_reproductive(solid2.coordinates_system)

        best_score = evaluator.evaluate()
        best_cs1 = self._clone_cartesian(solid1.coordinates_system)
        best_cs2 = self._clone_cartesian(solid2.coordinates_system)

        iterations = 0

        for iteration in range(self.MAX_ITERATIONS):
            iterations += 1

            # Check convergence
            if best_score < self.CONVERGENCE_THRESHOLD:
                print(f"[DarwinOptimizer] Converged at iteration {iterations} with score {best_score:.2e}")
                solid1.coordinates_system = best_cs1
                solid2.coordinates_system = best_cs2
                return best_score, iterations

            # Generate offspring for both coordinate systems
            offspring1 = solid1.coordinates_system.reproduce(
                self.OFFSPRING_PER_GENERATION, self.MAX_DEVIATION
            )
            offspring2 = solid2.coordinates_system.reproduce(
                self.OFFSPRING_PER_GENERATION, self.MAX_DEVIATION
            )

            # Evaluate all offspring combinations and find the best
            generation_best_score = best_score
            generation_best_idx1 = 0
            generation_best_idx2 = 0

            if self._use_gpu:
                generation_best_score, generation_best_idx1, generation_best_idx2 = \
                    self._gpu_evaluate_offspring(
                        evaluator, solids, offspring1, offspring2,
                        best_score, solid1, solid2
                    )
            else:
                generation_best_score, generation_best_idx1, generation_best_idx2 = \
                    self._cpu_evaluate_offspring(
                        evaluator, solids, offspring1, offspring2,
                        best_score, solid1, solid2
                    )

            # Update best if we found a better solution
            if generation_best_score < best_score:
                best_score = generation_best_score
                best_cs1 = self._clone_cartesian(offspring1[generation_best_idx1])
                best_cs2 = self._clone_cartesian(offspring2[generation_best_idx2])
                solid1.coordinates_system = offspring1[generation_best_idx1]
                solid2.coordinates_system = offspring2[generation_best_idx2]

            if iteration % 1000 == 0:
                print(f"[DarwinOptimizer] Iteration {iteration}: best_score = {best_score:.2e}")

        # Max iterations reached
        print(f"[DarwinOptimizer] Max iterations ({self.MAX_ITERATIONS}) reached. Final score: {best_score:.2e}")
        solid1.coordinates_system = best_cs1
        solid2.coordinates_system = best_cs2
        return best_score, iterations

    def _gpu_evaluate_offspring(
        self,
        evaluator,
        solids: List[Solid],
        offspring1: List[ReproductiveCartesian],
        offspring2: List[ReproductiveCartesian],
        best_score: float,
        solid1: Solid,
        solid2: Solid
    ) -> Tuple[float, int, int]:
        """GPU-accelerated offspring evaluation using PyTorch batched matrix operations."""
        torch = self._torch

        # Build batched transform matrices on GPU
        n = len(offspring1)
        matrices1 = torch.stack([
            torch.tensor(off.to_array(), dtype=torch.float32)
            for off in offspring1
        ]).cuda()

        matrices2 = torch.stack([
            torch.tensor(off.to_array(), dtype=torch.float32)
            for off in offspring2
        ]).cuda()

        # Local points (from evaluator's constraint vertices)
        p1_local = torch.tensor(
            [evaluator.vertex1.point.x, evaluator.vertex1.point.y, evaluator.vertex1.point.z, 1.0],
            dtype=torch.float32
        ).cuda()

        p2_local = torch.tensor(
            [evaluator.vertex2.point.x, evaluator.vertex2.point.y, evaluator.vertex2.point.z, 1.0],
            dtype=torch.float32
        ).cuda()

        # Batched transform: (N, 4, 4) @ (4, 1) -> (N, 4, 1)
        global1 = (matrices1 @ p1_local.unsqueeze(1)).squeeze(2)  # (N, 4)
        global2 = (matrices2 @ p2_local.unsqueeze(1)).squeeze(2)  # (N, 4)

        # Extract xyz
        xyz1 = global1[:, :3]
        xyz2 = global2[:, :3]

        # We need to evaluate all N x N combinations
        # xyz1: (N, 3), xyz2: (N, 3)
        # diff[i,j] = xyz1[i] - xyz2[j]
        diff = xyz1.unsqueeze(1) - xyz2.unsqueeze(0)  # (N, N, 3)
        distances = torch.norm(diff, dim=2)  # (N, N)
        scores = torch.abs(distances - evaluator.distance)  # (N, N)

        min_score = scores.min().item()
        min_idx = scores.argmin().item()
        idx1 = min_idx // n
        idx2 = min_idx % n

        return min_score, idx1, idx2

    def _cpu_evaluate_offspring(
        self,
        evaluator,
        solids: List[Solid],
        offspring1: List[ReproductiveCartesian],
        offspring2: List[ReproductiveCartesian],
        best_score: float,
        solid1: Solid,
        solid2: Solid
    ) -> Tuple[float, int, int]:
        """CPU offspring evaluation using vectorized numpy operations."""
        n = len(offspring1)

        # Build batched transform matrices
        matrices1 = np.array([off.to_array() for off in offspring1])  # (N, 4, 4)
        matrices2 = np.array([off.to_array() for off in offspring2])  # (N, 4, 4)

        # Local homogeneous points
        p1_local = np.array([
            evaluator.vertex1.point.x,
            evaluator.vertex1.point.y,
            evaluator.vertex1.point.z,
            1.0
        ])
        p2_local = np.array([
            evaluator.vertex2.point.x,
            evaluator.vertex2.point.y,
            evaluator.vertex2.point.z,
            1.0
        ])

        # Batched transform
        global1 = (matrices1 @ p1_local)  # (N, 4)
        global2 = (matrices2 @ p2_local)  # (N, 4)

        xyz1 = global1[:, :3]  # (N, 3)
        xyz2 = global2[:, :3]  # (N, 3)

        # All pairs: (N, N, 3)
        diff = xyz1[:, np.newaxis, :] - xyz2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
        scores = np.abs(distances - evaluator.distance)  # (N, N)

        min_score = float(scores.min())
        min_idx = int(scores.argmin())
        idx1 = min_idx // n
        idx2 = min_idx % n

        return min_score, idx1, idx2

    @staticmethod
    def _convert_to_reproductive(cartesian: Cartesian) -> ReproductiveCartesian:
        """Convert a regular Cartesian to a ReproductiveCartesian."""
        origin = ReproduciblePoint(cartesian.origin[0], cartesian.origin[1], cartesian.origin[2])
        x_point = ReproduciblePoint(
            cartesian.origin[0] + cartesian.x_axis[0],
            cartesian.origin[1] + cartesian.x_axis[1],
            cartesian.origin[2] + cartesian.x_axis[2]
        )
        xz_plane_point = ReproduciblePoint(
            cartesian.origin[0] + cartesian.z_axis[0],
            cartesian.origin[1] + cartesian.z_axis[1],
            cartesian.origin[2] + cartesian.z_axis[2]
        )
        return ReproductiveCartesian(origin=origin, x_point=x_point, xz_plane_point=xz_plane_point)

    @staticmethod
    def _clone_cartesian(cartesian: Cartesian) -> ReproductiveCartesian:
        """Create a clone of a ReproductiveCartesian."""
        if isinstance(cartesian, ReproductiveCartesian):
            return DarwinOptimizer._convert_to_reproductive(cartesian)
        return DarwinOptimizer._convert_to_reproductive(cartesian)
