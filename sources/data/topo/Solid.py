from sources.data.topo import Vertex
from sources.geometry import Cartesian


class Solid:
    def __init__(self, coordinates_system: Cartesian=None):
        self.coordinates_system = coordinates_system or Cartesian()
        self.vertices = set()

    def add_vertex(self, vertex: Vertex):
        self.vertices.add(vertex)
