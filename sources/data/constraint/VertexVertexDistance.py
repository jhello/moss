from data.constraint import Constraint
from data.topo import Vertex


class VertexVertexDistance(Constraint):
    def __init__(self, vertex1: Vertex, vertex2: Vertex, distance: float):
        super().__init__()
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.distance = distance

    def __repr__(self):
        return (f"VertexVertexDistance(vertex1_id={self.vertex1}, "
                f"vertex2_id={self.vertex2}, distance={self.distance})")