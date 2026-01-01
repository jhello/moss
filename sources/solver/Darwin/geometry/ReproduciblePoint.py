from typing import List

import numpy as np

from geometry import Point


class ReproduciblePoint(Point):
    def reproduce(self, n_offspring: int, max_deviation: float) -> List["ReproduciblePoint"]:
        """
        Generate offspring points that are also ReproduciblePoint.
        """
        deviations = np.random.uniform(-max_deviation, max_deviation, size=(n_offspring, 3))

        offspring_coords = deviations + np.array([self.x, self.y, self.z])

        return [ReproduciblePoint(x, y, z) for x, y, z in offspring_coords]
