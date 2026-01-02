import numpy as np
from typing import Optional

from geometry import Point


class Cartesian:
    """3D Cartesian coordinate system"""

    def __init__(
            self,
            origin: Point = None,
            x_point: Point = None,
            xz_plane_point: Point = None
    ):
        """
        Construct a Cartesian coordinate system

        Args:
            origin: Origin point coordinates (3D point). Defaults to (0, 0, 0)
            x_point: A point on the x-axis (defines positive x-axis direction).
                    Defaults to (1, 0, 0)
            xz_plane_point: A point in the first or second quadrant of xOz plane
                          (excludes x-axis but includes positive z-axis).
                          Defaults to (0, 0, 1)

        Raises:
            ValueError: If x_point coincides with origin
            ValueError: If xz_plane_point lies on the x-axis
        """
        # Set default values
        origin = origin or Point()
        x_point = x_point or Point(1.0, 0.0, 0.0)
        xz_plane_point = xz_plane_point or Point(0.0, 0.0, 1.0)

        # Convert to numpy arrays
        self.origin = origin.to_array()
        x_point_arr = x_point.to_array()
        xz_plane_point_arr = xz_plane_point.to_array()

        # Calculate x-axis unit vector
        x_vec = x_point_arr - self.origin
        x_vec_norm = np.linalg.norm(x_vec)

        if x_vec_norm == 0:
            raise ValueError("x_point cannot coincide with origin")

        self.x_axis = x_vec / x_vec_norm

        # Calculate y-axis using cross product (right-hand rule: y = (xz_vec) × x)
        xz_vec = xz_plane_point_arr - self.origin
        y_vec = np.cross(xz_vec, self.x_axis)

        if np.allclose(y_vec, 0):
            raise ValueError(
                "xz_plane_point cannot lie on the x-axis defined by origin and x_point"
            )

        self.y_axis = y_vec / np.linalg.norm(y_vec)

        # Calculate z-axis (ensures right-handed coordinate system: z = x × y)
        self.z_axis = np.cross(self.x_axis, self.y_axis)

    def __repr__(self) -> str:
        return (
            f"Cartesian(\n"
            f"  origin={self.origin},\n"
            f"  x_axis={self.x_axis},\n"
            f"  y_axis={self.y_axis},\n"
            f"  z_axis={self.z_axis}\n"
            f")"
        )

    def to_array(self) -> np.ndarray:
        transform_matrix = np.eye(4)

        # Rotation part: place x, y, z as column vectors
        transform_matrix[0:3, 0] = self.x_axis
        transform_matrix[0:3, 1] = self.y_axis
        transform_matrix[0:3, 2] = self.z_axis

        # Translation part: origin position
        transform_matrix[0:3, 3] = self.origin

        return transform_matrix