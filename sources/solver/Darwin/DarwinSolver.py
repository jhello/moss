from data import Case
from data.constraint import VertexVertexDistance
from operation.transform import transform
from solver import Status


class DarwinSolver:
    def __init__(self, case: Case):
        self.case = case
        self.status = Status.NOT_STARTED

    def solve(self):
        print("Solving using DarwinSolver")
        for constraint in self.case.constraints:
            print(f"Applying constraint: {constraint}")
            if isinstance(constraint, VertexVertexDistance):
                v1 = constraint.vertex1_id
                v2 = constraint.vertex2_id
                distance = constraint.distance
                for solid in self.case.solids:
                    if v1 in solid.vertices:
                        p1_global = transform(v1.point, solid.coordinates_system)

                        #TODO: Find the solid containing v2
            else:
                print(f"Unknown constraint type: {type(constraint)}")


        self.status = Status.SUCCESS

    def get_status(self):
        return self.status