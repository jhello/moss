from typing import Optional

from sources.data import Case
from sources.data.constraint import VertexVertexDistance
from sources.solver import Status
from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator
from sources.solver.Darwin.optimizer import DarwinOptimizer


class DarwinSolver:
    def __init__(
        self,
        case: Case,
        device: str = None,
        *,
        max_population: Optional[int] = None,
        offspring_per_individual: Optional[int] = None,
        max_deviation: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        max_iterations: Optional[int] = None,
        seed: Optional[int] = None
    ):
        self.case = case
        self.status = Status.NOT_STARTED
        self.device = device
        self.iterations = 0
        self.final_score = None
        self.optimizer = DarwinOptimizer(
            device=device,
            max_population=max_population,
            offspring_per_individual=offspring_per_individual,
            max_deviation=max_deviation,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
            seed=seed
        )

    def solve(self):
        print("Solving using DarwinSolver")
        optimizer = self.optimizer

        for constraint in self.case.constraints:
            if isinstance(constraint, VertexVertexDistance):
                evaluator = VertexVertexDistanceEvaluator(constraint, self.case.solids)
                score = evaluator.evaluate()
                print(f"Evaluating VertexVertexDistance constraint: Score = {score}")

                # Optimization: iteratively minimize score using reproductive
                # Cartesian coordinate systems with a capped population.
                final_score, iterations = optimizer.optimize(
                    evaluator=evaluator,
                    solids=self.case.solids,
                    vertex1=constraint.vertex1,
                    vertex2=constraint.vertex2
                )
                self.iterations += iterations
                self.final_score = final_score
                print(f"Optimization finished after {iterations} generations. "
                      f"Final score: {final_score:.2e}")

                if final_score >= optimizer.convergence_threshold:
                    self.status = Status.FAILURE
                    print(f"Optimization failed to converge. Final score: {final_score:.2e}")
                    return
            else:
                print(f"Unknown constraint type: {type(constraint)}")

        self.status = Status.SUCCESS

    def get_status(self):
        return self.status
