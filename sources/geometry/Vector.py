from typing import Union
import numpy as np
from numbers import Real
from numpy.typing import ArrayLike


class Vector:
    """3D vector class with numpy backend"""

    def __init__(
            self,
            x: Union[ArrayLike, Real] = 1.0,
            y: Real = None,
            z: Real = None
    ):
        """
        Create a 3D vector

        Args:
            x: Either x-coordinate or array-like [x, y, z]
            y: y-coordinate (ignored if x is array-like)
            z: z-coordinate (ignored if x is array-like)

        Examples:
            Vector(1, 2, 3)
            Vector([1, 2, 3])
            Vector(x=1.0)  # Creates Vector(1, 0, 0)
        """
        # If x is array-like, use it directly
        if y is None and z is None and hasattr(x, "__iter__"):
            arr = np.asarray(x, dtype=float)
            if arr.shape != (3,):
                raise ValueError("Array-like input must have exactly 3 elements")
            self._v = arr.copy()
        else:
            # Scalar initialization with defaults
            y = 0.0 if y is None else y
            z = 0.0 if z is None else z
            self._v = np.array([float(x), float(y), float(z)], dtype=float)

    @property
    def x(self) -> float:
        """X component of the vector"""
        return float(self._v[0])

    @property
    def y(self) -> float:
        """Y component of the vector"""
        return float(self._v[1])

    @property
    def z(self) -> float:
        """Z component of the vector"""
        return float(self._v[2])


    def __add__(self, other: 'Vector') -> 'Vector':
        """Vector addition"""
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self._v + other._v)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """Vector subtraction"""
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self._v - other._v)

    def __mul__(self, scalar: Real) -> 'Vector':
        """Scalar multiplication"""
        if not isinstance(scalar, Real):
            return NotImplemented
        return Vector(self._v * scalar)

    def __rmul__(self, scalar: Real) -> 'Vector':
        """Right scalar multiplication"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Real) -> 'Vector':
        """Scalar division"""
        if not isinstance(scalar, Real):
            return NotImplemented
        if scalar == 0:
            raise ValueError("Cannot divide vector by zero")
        return Vector(self._v / scalar)

    def __neg__(self) -> 'Vector':
        """Vector negation"""
        return Vector(-self._v)

    def __eq__(self, other: object) -> bool:
        """Vector equality with tolerance"""
        if not isinstance(other, Vector):
            return NotImplemented
        return np.allclose(self._v, other._v)

    def dot(self, other: 'Vector') -> float:
        """
        Dot product with another vector

        Args:
            other: Another Vector

        Returns:
            Scalar dot product
        """
        if not isinstance(other, Vector):
            raise TypeError("dot requires a Vector")
        return float(np.dot(self._v, other._v))

    def cross(self, other: 'Vector') -> 'Vector':
        """
        Cross product with another vector

        Args:
            other: Another Vector

        Returns:
            Vector perpendicular to both input vectors
        """
        if not isinstance(other, Vector):
            raise TypeError("cross requires a Vector")
        return Vector(np.cross(self._v, other._v))

    def magnitude(self) -> float:
        """Calculate the magnitude (length) of the vector"""
        return float(np.linalg.norm(self._v))

    def normalized(self) -> 'Vector':
        """
        Return a normalized (unit) vector

        Returns:
            Unit vector in the same direction, or zero vector if magnitude is 0
        """
        mag = self.magnitude()
        if mag == 0.0:
            return Vector(0.0, 0.0, 0.0)
        return Vector(self._v / mag)

    def to_array(self) -> np.ndarray:
        """Return a copy of the internal numpy array"""
        return self._v.copy()

    def to_list(self) -> list[float]:
        """Return vector as a list [x, y, z]"""
        return self._v.tolist()

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"