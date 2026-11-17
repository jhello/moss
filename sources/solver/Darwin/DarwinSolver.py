from sources.data import Case
from sources.data.constraint import VertexVertexDistance
from sources.solver import Status
from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator
from sources.solver.Darwin.optimizer import DarwinOptimizer


class DarwinSolver:
    def __init__(self, case: Case, device: str = None):
        self.case = case
        self.status = Status.NOT_STARTED
        self.device = device

    def solve(self):
        print("Solving using DarwinSolver")
        optimizer = DarwinOptimizer(device=self.device)

        for constraint in self.case.constraints:
            if isinstance(constraint, VertexVertexDistance):
                evaluator = VertexVertexDistanceEvaluator(constraint, self.case.solids)
                score = evaluator.evaluate()
                print(f"Evaluating VertexVertexDistance constraint: Score = {score}")

                # Optimization: iteratively minimize score using reproductive Cartesian coordinate systems
                final_score, iterations = optimizer.optimize(
                    evaluator=evaluator,
                    solids=self.case.solids,
                    vertex1=constraint.vertex1,
                    vertex2=constraint.vertex2
                )
                print(f"Optimization finished after {iterations} iterations. Final score: {final_score:.2e}")

                if final_score >= DarwinOptimizer.CONVERGENCE_THRESHOLD:
                    self.status = Status.FAILURE
                    print(f"Optimization failed to converge. Final score: {final_score:.2e}")
                    return
            else:
                print(f"Unknown constraint type: {type(constraint)}")

        self.status = Status.SUCCESS

    def get_status(self):
        return self.status