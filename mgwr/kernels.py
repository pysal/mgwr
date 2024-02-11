# GWR kernel function specifications

__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
#adaptive specifications should be parameterized with nn-1 to match original gwr
#implementation. That is, pysal counts self neighbors with knn automatically.

#Soft dependency of numba's njit
try:
    from numba import njit
except ImportError:

    def njit(func):
        return func


@njit
def local_cdist(coords_i, coords, spherical):
    """
    Compute Haversine (spherical=True) or Euclidean (spherical=False) distance for a local kernel.
    """
    if spherical:
        dLat = np.radians(coords[:, 1] - coords_i[1])
        dLon = np.radians(coords[:, 0] - coords_i[0])
        lat1 = np.radians(coords[:, 1])
        lat2 = np.radians(coords_i[1])
        a = np.sin(
            dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0
        return R * c
    else:
        return np.sqrt(np.sum((coords_i - coords)**2, axis=1))


class Kernel(object):
    """
    GWR kernel function specifications.

    """

    def __init__(self, i: int,
                 data: np.array,
                 bw: float = None,
                 fixed: bool = True,
                 function: str = 'triangular',
                 eps: float = 1.0000001,
                 points: np.array = None,
                 spherical=False) -> None:
        """Kernel function specifications

        Args:
            i (int): index of focal point
            data (np.array): n-by-2 array of coordinates for n observations
            bw (float, optional): the bandwidth for the kernel function. Defaults to None.
            fixed (bool, optional): whether to implement fixed bandwidth or not. Defaults to True.
            function (str, optional): specify kernel function, 7 candidate kernel functions. Defaults to 'triangular'.
            eps (float, optional): exploration value. Defaults to 1.0000001.
            points (np.array, optional): whether to use points to calculate bandwidth. Defaults to None.
            spherical (bool, optional): whether to use to calculate bandwidth spatially. Defaults to False.
        """

        if points is None:
            self.dvec = local_cdist(data[i], data, spherical).reshape(-1)
        else:
            self.dvec = local_cdist(points[i], data, spherical).reshape(-1)

        self.function = function.lower()

        if fixed:
            self.bandwidth = float(bw)
        else:
            self.bandwidth = np.partition(
                self.dvec,
                int(bw) - 1)[int(bw) - 1] * eps  #partial sort in O(n) Time

        self.kernel = self._kernel_funcs(self.dvec / self.bandwidth)

        if self.function == "bisquare":  #Truncate for bisquare
            self.kernel[(self.dvec >= self.bandwidth)] = 0

    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zs.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs**2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs**2)**2
        elif self.function == 'gaussian':
            return np.exp(-0.5 * (zs)**2)
        elif self.function == 'bisquare':
            return (1 - (zs)**2)**2
        elif self.function == 'exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)
