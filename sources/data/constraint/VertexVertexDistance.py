from data.constraint import Constraint
from data.topo import Vertex


class VertexVertexDistance(Constraint):
    def __init__(self, vertex1_id: Vertex, vertex2_id: Vertex, distance: float):
        super().__init__()
        self.vertex1_id = vertex1_id
        self.vertex2_id = vertex2_id
        self.distance = distance

    def __repr__(self):
        return (f"VertexVertexDistance(vertex1_id={self.vertex1_id}, "
                f"vertex2_id={self.vertex2_id}, distance={self.distance})")