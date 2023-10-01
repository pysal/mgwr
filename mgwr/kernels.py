# GWR kernel function specifications

__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
# adaptive specifications should be parameterized with nn-1 to match original gwr
# implementation. That is, pysal counts self neighbors with knn automatically.

# Soft dependency of numba's njit
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func


@njit
def local_cdist(coords_i: np.array, coords: np.array, spherical: bool) -> float:
    """
    Compute Haversine (spherical=True) or Euclidean (spherical=False) distance for a local kernel.

    Args:
        coords_i (np.array): the coordinates for the ith observation
        coords (np.array): the coordinates for all observations
        spherical (bool): whether to use Haversine or Euclidean distance

    Returns:
        float: the distance between coords_i and coords in Haversine (spherical=True) or Euclidean (spherical=False)

    Examples:
        >>> import numpy as np
        >>> from mgwr.kernels import local_cdist
        >>> coords_i = np.array([0, 0])
        >>> coords = np.array([[1, 1], [2, 2], [3, 3]])
        >>> local_cdist(coords_i, coords, spherical=False)
        array([1.41421356, 2.82842712, 4.24264069])
        >>> local_cdist(coords_i, coords, spherical=True)
        array([157.24938127, 314.49876253, 471.7481438 ])
    """

    if not spherical:
        return np.sqrt(np.sum((coords_i - coords)**2, axis=1))

    dLat = np.radians(coords[:, 1] - coords_i[1])
    dLon = np.radians(coords[:, 0] - coords_i[0])
    lat1 = np.radians(coords[:, 1])
    lat2 = np.radians(coords_i[1])
    a = np.sin(dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


class Kernel(object):
    """
    GWR kernel function specifications.
    """

    def __init__(self, i: int, data: np.array,
                 bw: float = None, fixed: bool = True,
                 function: str = 'triangular', eps: float = 1.0000001,
                 ids=None, points: np.array = None, spherical=False):

        if points is None:
            self.dvec = local_cdist(data[i], data, spherical).reshape(-1)
        else:
            self.dvec = local_cdist(points[i], data, spherical).reshape(-1)

        self.function = function.lower()

        if fixed:
            self.bandwidth = bw
        else:
            # partial sort in O(n) Time
            self.bandwidth = np.partition(self.dvec, int(bw) - 1)[int(bw) - 1] * eps

        self.kernel = self._kernel_funcs(self.dvec / self.bandwidth)

        # Truncate for bisquare
        if self.function == "bisquare":
            self.kernel[(self.dvec >= self.bandwidth)] = 0

    def _kernel_funcs(self, zs: float) -> float:
        # functions follow Anselin and Rey (2010) table 5.4
        func_dict = {
            "triangular": 1 - zs,
            "uniform": np.ones(zs.shape) * 0.5,
            "quadratic": (3. / 4) * (1 - zs**2),
            "quartic": (15. / 16) * (1 - zs**2)**2,
            "gaussian": np.exp(-0.5 * (zs)**2),
            "bisquare": (1 - (zs)**2)**2,
            "exponential": np.exp(-zs)
        }

        return func_dict.get(self.function, "Unsupported kernel function")
