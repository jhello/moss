from typing import List

from sources.geometry import Cartesian, Point, Vector

from .ReproduciblePoint import ReproduciblePoint


class ReproductiveCartesian(Cartesian):
    def __init__(
            self,
            origin: Point = None,
            x_point: Point = None,
            xz_plane_point: Point = None
    ):
        # Store the original reproductive points before super().__init__ converts them to arrays
        self._origin_point = origin or ReproduciblePoint()
        self._x_point = x_point or ReproduciblePoint(1.0, 0.0, 0.0)
        self._xz_plane_point = xz_plane_point or ReproduciblePoint(0.0, 0.0, 1.0)
        super().__init__(self._origin_point, self._x_point, self._xz_plane_point)

    def reproduce(self, n_offspring: int, max_deviation: float) -> List["ReproductiveCartesian"]:
        """
        Generate offspring Cartesian coordinate systems that are also ReproductiveCartesian.
        """
        offspring_origins = self._origin_point.reproduce(n_offspring, max_deviation)

        # Compute x_point and xz_plane_point as ReproduciblePoint so they can also reproduce
        x_point = ReproduciblePoint(
            self._origin_point.x + self.x_axis[0],
            self._origin_point.y + self.x_axis[1],
            self._origin_point.z + self.x_axis[2]
        )
        offspring_x_points = x_point.reproduce(n_offspring, max_deviation)

        xz_plane_point = ReproduciblePoint(
            self._origin_point.x + self.z_axis[0],
            self._origin_point.y + self.z_axis[1],
            self._origin_point.z + self.z_axis[2]
        )
        offspring_xz_plane_points = xz_plane_point.reproduce(n_offspring, max_deviation)

        return [
            ReproductiveCartesian(origin=o, x_point=xp, xz_plane_point=xzp)
            for o, xp, xzp in zip(offspring_origins, offspring_x_points, offspring_xz_plane_points)
        ]
