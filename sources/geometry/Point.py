from numbers import Real
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from geometry import Vector


class Point:
    """3D point representation using NumPy array internally"""

    def __init__(
            self,
            x: Union[ArrayLike, Real] = 0.0,
            y: Real = None,
            z: Real = None
    ) -> None:
        """
        Constructor for a 3D point

        Args:
            x: Either x-coordinate or array-like [x, y, z]. Default is 0.0
            y: y-coordinate (ignored if x is array-like). Default is 0.0
            z: z-coordinate (ignored if x is array-like). Default is 0.0

        Examples:
            Point(1, 2, 3)
            Point([1, 2, 3])
            Point(x=1.0)  # Creates Point(1, 0, 0)
        """
        # If x is array-like, use it directly
        if y is None and z is None and hasattr(x, "__iter__"):
            arr = np.asarray(x, dtype=float)
            if arr.shape != (3,):
                raise ValueError("Array-like input must have exactly 3 elements")
            self._coords = arr.copy()
        else:
            # Scalar initialization with defaults
            self._coords = np.array([
                float(x),
                0.0 if y is None else float(y),
                0.0 if z is None else float(z)
            ], dtype=float)

    def __repr__(self) -> str:
        """Return string representation of the point"""
        return f"Point({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        """Return human-readable string representation"""
        return f"({self.x}, {self.y}, {self.z})"

    def __eq__(self, other: object) -> bool:
        """Check if two points are equal with tolerance"""
        if not isinstance(other, Point):
            return NotImplemented
        return np.allclose(self._coords, other._coords, rtol=1e-9, atol=1e-12)

    def __add__(self, other: 'Vector') -> 'Point':
        """Add a vector to the point component-wise"""
        if not isinstance(other, Vector):
            return NotImplemented
        result = Point()
        result._coords = self._coords + other._coords
        return result

    def __sub__(self, other: Union['Point', 'Vector']) -> Union['Vector', 'Point']:
        """
        Subtract another point or vector

        Point - Point = Vector (displacement)
        Point - Vector = Point (translation)
        """
        if isinstance(other, Point):
            # Point - Point = Vector
            result = Vector()
            result._coords = self._coords - other._coords
            return result
        elif isinstance(other, Vector):
            # Point - Vector = Point
            result = Point()
            result._coords = self._coords - other._coords
            return result
        return NotImplemented

    @property
    def x(self) -> float:
        """X coordinate of the point"""
        return float(self._coords[0])

    @property
    def y(self) -> float:
        """Y coordinate of the point"""
        return float(self._coords[1])

    @property
    def z(self) -> float:
        """Z coordinate of the point"""
        return float(self._coords[2])

    @x.setter
    def x(self, value: float) -> None:
        """Set X coordinate"""
        self._coords[0] = float(value)

    @y.setter
    def y(self, value: float) -> None:
        """Set Y coordinate"""
        self._coords[1] = float(value)

    @z.setter
    def z(self, value: float) -> None:
        """Set Z coordinate"""
        self._coords[2] = float(value)

    def distance_to(self, other: 'Point') -> float:
        """
        Calculate Euclidean distance to another point

        Args:
            other: Another 3D point

        Returns:
            Distance between the two points
        """
        if not isinstance(other, Point):
            raise TypeError("distance_to requires a Point")
        return float(np.linalg.norm(self._coords - other._coords))

    def distance_squared_to(self, other: 'Point') -> float:
        """
        Calculate squared distance (faster, avoids sqrt)

        Args:
            other: Another 3D point

        Returns:
            Squared distance between the two points
        """
        if not isinstance(other, Point):
            raise TypeError("distance_squared_to requires a Point")
        diff = self._coords - other._coords
        return float(np.dot(diff, diff))

    def to_tuple(self) -> tuple[float, float, float]:
        """
        Convert point to tuple

        Returns:
            Tuple (x, y, z) of coordinates
        """
        return float(self._coords[0]), float(self._coords[1]), float(self._coords[2])
