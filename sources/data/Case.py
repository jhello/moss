from typing import Optional

from data.constraint import Constraint
from data.topo import Vertex, Solid


class Case:
    def __init__(self, case_id, description):
        self.case_id = case_id
        self.description = description

        self.fixed_vertices = []
        self.constraints = []
        self.solids = []

    def __repr__(self):
        return f"Case(case_id={self.case_id}, description={self.description}, status={self.status})"

    def fix_vertex(self, vertex: Vertex) -> None:
        self.fixed_vertices.append(vertex)

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def add_solid(self, solid: Solid) -> None:
        self.solids.append(solid)
