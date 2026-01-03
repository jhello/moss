from data import Case
from data.constraint import VertexVertexDistance
from solver import Status
from solver.Darwin.evaluator import VertexVertexDistanceEvaluator


class DarwinSolver:
    def __init__(self, case: Case):
        self.case = case
        self.status = Status.NOT_STARTED

    def solve(self):
        print("Solving using DarwinSolver")
        for constraint in self.case.constraints:
            if isinstance(constraint, VertexVertexDistance):
                evaluator = VertexVertexDistanceEvaluator(constraint, self.case.solids)
                score = evaluator.evaluate()
                print(f"Evaluating VertexVertexDistance constraint: Score = {score}")
                #TODO: Implement optimization logic to adjust solids based on the constraint evaluation
            else:
                print(f"Unknown constraint type: {type(constraint)}")


        self.status = Status.SUCCESS

    def get_status(self):
        return self.status