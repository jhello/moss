from data.topo import Vertex
from geometry import Cartesian


class Solid:
    def __init__(self, coordinates_system: Cartesian=None):
        self.coordinates_system = coordinates_system or Cartesian()
        self.vertices = set()

    def add_vertex(self, vertex: Vertex):
        self.vertices.add(vertex)
