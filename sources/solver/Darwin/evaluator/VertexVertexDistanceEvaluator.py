from data.constraint import VertexVertexDistance
from data.topo import Solid
from geometry import Point
from operation import transform, distance


class VertexVertexDistanceEvaluator:
    def __init__(self, constraint: VertexVertexDistance, solids: list[Solid]):
        self.vertex1 = constraint.vertex1
        self.vertex2 = constraint.vertex2
        self.distance = constraint.distance
        self.solids = solids

    def evaluate(self) -> float:
        point1 = None
        point2 = None
        for solid in self.solids:
            if self.vertex1 in solid.vertices:
                point1 = transform(self.vertex1.point, solid.coordinates_system)
            if self.vertex2 in solid.vertices:
                point2 = transform(self.vertex2.point, solid.coordinates_system)
            if point1 and point2:
                break
        return abs(distance(point1, point2) - self.distance)

