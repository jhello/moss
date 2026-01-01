from sources.geometry import Point


class Vertex:
    def __init__(self, point: Point = None):
        if point is None:
            point = Point()
        self.point = point

    def __repr__(self):
        return f"Vertex(x={self.point.x}, y={self.point.y}, z={self.point.z})"

    def location(self) -> Point:
        return self.point
