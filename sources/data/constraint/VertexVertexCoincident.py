from data.constraint import Constraint
from data.topo import Vertex


class VertexVertexCoincident(Constraint):
    def __init__(self, vertex1_id: Vertex, vertex2_id: Vertex):
        super().__init__()
        self.vertex1_id = vertex1_id
        self.vertex2_id = vertex2_id

    def __repr__(self):
        return f"VertexVertexCoincident(vertex1_id={self.vertex1_id}, vertex2_id={self.vertex2_id})"
