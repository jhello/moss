from typing import Optional

from data.constraint import Constraint
from data.topo import Vertex


class Case:
    def __init__(self, case_id, description):
        self.case_id = case_id
        self.description = description

        self.vertices = []
        self.fixed_vertices = []
        self.constraints = []

    def __repr__(self):
        return f"Case(case_id={self.case_id}, description={self.description}, status={self.status})"

    def add_vertex(self, vertex: Optional[Vertex] = None) -> Optional[Vertex]:
        if vertex is None:
            vertex = Vertex()
            self.vertices.append(vertex)
            return vertex
        self.vertices.append(vertex)
        return None

    def fix_vertex(self, vertex: Vertex) -> None:
        self.fixed_vertices.append(vertex)

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)
