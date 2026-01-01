import math


class Point:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """
        Constructor for a 3D point. Default coordinates are (0.0, 0.0, 0.0)

        Args:
            x (float): X coordinate, default is 0.0
            y (float): Y coordinate, default is 0.0
            z (float): Z coordinate, default is 0.0
        """
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def __repr__(self) -> str:
        """Return string representation of the point"""
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

    def __eq__(self, other: "Point") -> bool:
        """Check if two points are equal based on their coordinates"""
        return (
                math.isclose(self.x, other.x, rel_tol=1e-9, abs_tol=1e-12)
                and math.isclose(self.y, other.y, rel_tol=1e-9, abs_tol=1e-12)
                and math.isclose(self.z, other.z, rel_tol=1e-9, abs_tol=1e-12)
        )

    def distance_to(self, other: "Point") -> float:
        """
        Calculate the Euclidean distance from this point to another 3D point

        Args:
            other (Point3D): Another 3D point

        Returns:
            float: Distance between the two points
        """
        dx: float = self.x - other.x
        dy: float = self.y - other.y
        dz: float = self.z - other.z
        return (dx**2 + dy**2 + dz**2) ** 0.5
