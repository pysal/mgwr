# GWR kernel function specifications

__author__ = "Taylor Oshan tayoshan@gmail.com"

import scipy
from scipy.spatial.kdtree import KDTree
import numpy as np
from scipy.spatial.distance import cdist as cdist_scipy
from math import radians, sin, cos, sqrt, asin

#adaptive specifications should be parameterized with nn-1 to match original gwr
#implementation. That is, pysal counts self neighbors with knn automatically.

def fix_gauss(i, coords, bw, points=None, spherical=False):
    """
    Fixed Gaussian kernel.
    """
    w = _Kernel(i, coords, function='gwr_gaussian', bandwidth=bw, truncate=False, points=points, spherical=spherical)
    return w.kernel

def adapt_gauss(i, coords, nn, points=None,spherical=False):
    """
    Spatially adaptive Gaussian kernel.
    """
    w = _Kernel(i, coords, fixed=False, k=nn-1, function='gwr_gaussian', truncate=False, points=points, spherical=spherical)
    return w.kernel

def fix_bisquare(i, coords, bw, points=None, spherical=False):
    """
    Fixed bisquare kernel.
    """
    w = _Kernel(i, coords, function='bisquare', bandwidth=bw, points=points, spherical=spherical)
    return w.kernel

def adapt_bisquare(i, coords, nn, points=None, spherical=False):
    """
    Spatially adaptive bisquare kernel.
    """
    w = _Kernel(i, coords, fixed=False, k=nn-1, function='bisquare', points=points, spherical=spherical)
    return w.kernel

def fix_exp(i, coords, bw, points=None, spherical=False):
    """
    Fixed exponential kernel.
    """
    w = _Kernel(i, coords, function='exponential', bandwidth=bw, truncate=False, points=points, spherical=spherical)
    return w.kernel

def adapt_exp(i, coords, nn, points=None, spherical=False):
    """
    Spatially adaptive exponential kernel.
    """
    w = _Kernel(i, coords, fixed=False, k=nn-1, function='exponential', truncate=False, points=points, spherical=spherical)
    return w.kernel


#Customized Kernel class user for GWR because the default PySAL kernel class
#favors memory optimization over speed optimizations and GWR often needs the 
#speed optimization since it is not always assume points far awary
# are truncated #to zero

def cdist(coords1,coords2,spherical):
    def _haversine(lon1, lat1, lon2, lat2):
        R = 6371.0 # Earth radius in kilometers
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        return R * c
    n = len(coords1)
    m = len(coords2)
    dmat = np.zeros((n,m))

    if spherical:
        for i in range(n) :
            for j in range(m):
                dmat[i,j] = _haversine(coords1[i][0], coords1[i][1], coords2[j][0], coords2[j][1])
    else:
        dmat = cdist_scipy(coords1,coords2)

    return dmat

class _Kernel(object):
    """
    GWR kernel function specifications.

    """
    def __init__(self, i, data, bandwidth=None, fixed=True, k=None,
                 function='triangular', eps=1.0000001, ids=None, truncate=True,
                 points=None, spherical=False): #Added truncate flag
        
        if issubclass(type(data), scipy.spatial.KDTree):
            self.data = data.data
            data = self.data
        else:
            self.data = data
        if k is not None:
            self.k = int(k) + 1
        else:
            self.k = k
        
        self.spherical = spherical
        
        if points is None:
            self.dvec = cdist([self.data[i]], self.data, self.spherical).reshape(-1)
        else:
            self.points = points
            self.dvec = cdist([self.points[i]], self.data, self.spherical).reshape(-1)

        self.function = function.lower()
        self.fixed = fixed
        self.eps = eps
        self.trunc = truncate
        if bandwidth:
            self.bandwidth = float(bandwidth)
        else:
            self.bandwidth = np.partition(self.dvec, int(self.k)-1)[int(self.k)-1] * self.eps #partial sort in O(n)

        self.kernel = self._kernel_funcs(self.dvec/self.bandwidth)

        if self.trunc:
            self.kernel[(self.dvec >= self.bandwidth)] = 0
    '''
    def _set_bw(self):
        if self.searching:
            if self.k is not None:
                dmat = self.sorted_dmat[:,:self.k]
            else:
                dmat = self.dmat
        else:
            if self.k is not None:
                dmat = np.sort(self.dmat)[:,:self.k]
            else:
                dmat = self.dmat
        
        if self.fixed:
            # use max knn distance as bandwidth
            bandwidth = dmat.max() * self.eps
            n = len(self.data)
            self.bandwidth = np.ones((n, 1), 'float') * bandwidth
        else:
            # use local max knn distance
            self.bandwidth = dmat.max(axis=1) * self.eps
            self.bandwidth.shape = (self.bandwidth.size, 1)
    '''
    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zi.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs ** 2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs ** 2) ** 2
        elif self.function == 'gaussian':
            c = np.pi * 2
            c = c ** (-0.5)
            return c * np.exp(-(zs ** 2) / 2.)
        elif self.function == 'gwr_gaussian':
            return np.exp(-0.5*(zs)**2)
        elif self.function == 'bisquare':
            return (1-(zs)**2)**2
        elif self.function =='exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)
