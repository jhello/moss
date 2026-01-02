from typing import List

from geometry import Cartesian, Point


class ReproductiveCartesian(Cartesian):
    def __init__(
            self,
            origin: Point = None,
            x_point: Point = None,
            xz_plane_point: Point = None
    ):
        super().__init__(origin, x_point, xz_plane_point)

    def reproduce(self, n_offspring: int, max_deviation: float) -> List["ReproductiveCartesian"]:
        """
        Generate offspring Cartesian coordinate systems that are also ReproductiveCartesian.
        """
        offspring_origins = self.origin.reproduce(n_offspring, max_deviation)

        x_point = self.origin + self.x_axis
        offspring_x_points = x_point.reproduce(n_offspring, max_deviation)

        xz_plane_point = self.origin + self.z_axis
        offspring_xz_plane_points = xz_plane_point.reproduce(n_offspring, max_deviation)

        return [
            ReproductiveCartesian(origin=o, x_point=xp, xz_plane_point=xzp)
            for o, xp, xzp in zip(offspring_origins, offspring_x_points, offspring_xz_plane_points)
        ]
