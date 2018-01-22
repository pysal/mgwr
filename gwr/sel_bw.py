"""
GWR Bandwidth selection class
"""

#x_glob parameter does not yet do anything; it is for semiparametric

__author__ = "Taylor Oshan Tayoshan@gmail.com"

import numpy as np
from scipy.spatial.distance import cdist,pdist,squareform
#from pysal.common import KDTree
#import pysal.spreg.user_output as USER
from spglm.family import Gaussian, Poisson, Binomial
from spglm.iwls import iwls
from .kernels import *
from .search import golden_section, equal_interval

#kernel types where fk = fixed kernels and ak = adaptive kernels
fk = {'gaussian': fix_gauss, 'bisquare': fix_bisquare, 'exponential': fix_exp}
ak = {'gaussian': adapt_gauss, 'bisquare': adapt_bisquare, 'exponential': adapt_exp}

class Sel_BW(object):
    """
    Select bandwidth for kernel

    Methods: p211 - p213, bandwidth selection
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Parameters
    ----------
    y              : array
                     n*1, dependent variable.
    X_glob         : array
                     n*k1, fixed independent variable.
    X_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    offset         : array
                     n*1, offset variable for Poisson model
    kernel         : string
                     kernel function: 'gaussian', 'bisquare', 'exponetial'
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.

    Attributes
    ----------
    y              : array
                     n*1, dependent variable.
    X_glob         : array
                     n*k1, fixed independent variable.
    X_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    kernel         : string
                     type of kernel used and wether fixed or adaptive
    criterion      : string
                     bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
    search         : string
                     bw search method: 'golden', 'interval'
    bw_min         : float
                     min value used in bandwidth search
    bw_max         : float
                     max value used in bandwidth search
    interval       : float
                     interval increment used in interval search
    tol            : float
                     tolerance used to determine convergence
    max_iter       : integer
                     max interations if no convergence to tol
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    Examples
    ________

    >>> import libpysal
    >>> from gwr.sel_bw import Sel_BW
    >>> data = libpysal.open(libpysal.examples.get_path('GData_utm.csv'))
    >>> coords = zip(data.bycol('X'), data.by_col('Y')) 
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([rural, pov, african_amer])
    
    #Golden section search AICc - adaptive bisquare
    >>> bw = Sel_BW(coords, y, X).search(criterion='AICc')
    >>> print bw
    93.0

    #Golden section search AIC - adaptive Gaussian
    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='AIC')
    >>> print bw
    50.0

    #Golden section search BIC - adaptive Gaussian
    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='BIC')
    >>> print bw
    62.0

    #Golden section search CV - adaptive Gaussian
    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='CV')
    >>> print bw
    68.0

    #Interval AICc - fixed bisquare
    >>>  sel = Sel_BW(coords, y, X, fixed=True).
    >>>  bw = sel.search(search='interval', bw_min=211001.0, bw_max=211035.0, interval=2) 
    >>> print bw
    211025.0

    """
    def __init__(self, coords, y, X_loc, X_glob=None, family=Gaussian(),
            offset=None, kernel='bisquare', fixed=False, constant=True):
        self.coords = coords
        self.y = y
        self.X_loc = X_loc
        if X_glob is not None:
            self.X_glob = X_glob
        else:
            self.X_glob = []
        self.family=family
        self.fixed = fixed
        self.kernel = kernel
        if offset is None:
            self.offset = np.ones((len(y), 1))
        else:
            self.offset = offset * 1.0
        
        self.constant = constant
        self._build_dMat()

    def search(self, search='golden_section', criterion='AICc', bw_min=0.0,
            bw_max=0.0, interval=0.0, tol=1.0e-6, max_iter=200):
        """
        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        search         : string
                         bw search method: 'golden', 'interval'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        interval       : float
                         interval increment used in interval search
        tol            : float
                         tolerance used to determine convergence
        max_iter       : integer
                         max iterations if no convergence to tol

        Returns
        -------
        bw             : scalar or array
                         optimal bandwidth value
        """
        self.search = search
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter

        if self.fixed:
            if self.kernel == 'gaussian':
                ktype = 1
            elif self.kernel == 'bisquare':
                ktype = 3
            elif self.kernel == 'exponential':
                ktype = 5
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)
        else:
            if self.kernel == 'gaussian':
                ktype = 2
            elif self.kernel == 'bisquare':
                ktype = 4
            elif self.kernel == 'exponential':
                ktype = 6
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)

        if ktype % 2 == 0:
            int_score = True
        else:
            int_score = False
        self.int_score = int_score

        self._bw()

        return self.bw[0]
            
    #hold it for allowing lat-lons in next PR
    #_haversine formula to calculate distance
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371400 # Earth radius in meters
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        return R * c
    
    #return distance matrix NxM from lat-lons
    def _cdist_sph(coords1,coords2):
        n = len(coords1)
        m = len(coords2)
        mat = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                mat[i][j] = _haversine(coords1[i][1], coords1[i][0], coords2[j][1], coords2[j][0])
        return mat
    
    
    def _build_dMat(self):
        if self.fixed:
            self.dmat = cdist(self.coords,self.coords)
            self.sorted_dmat = None
        else:
            self.dmat = cdist(self.coords,self.coords)
            self.sorted_dmat = np.sort(self.dmat)

    #return the spatial kernel
    def _build_kernel_fast(self, bw, points=None):
        if self.fixed:
            try:
                W = fk[self.kernel](self.coords, bw, points, self.dmat,self.sorted_dmat)
            except:
                raise TypeError('Unsupported kernel function  ', self.kernel)
        else:
            try:
                W = ak[self.kernel](self.coords, bw, points, self.dmat,self.sorted_dmat)
            except:
                raise TypeError('Unsupported kernel function  ', self.kernel)

        return W

    def _fast_fit(self, bw,ini_params=None, tol=1.0e-5, max_iter=20,constant=True):
        W = self._build_kernel_fast(bw)
        trS = 0 #trace of S
        RSS = 0
        dev = 0
        CV_score = 0
        n = self.y.shape[0]
        for i in range(n):
            nonzero_i = np.nonzero(W[i]) #local neighborhood
            wi = W[i,nonzero_i].reshape((-1,1))
            X_new = self.X_loc[nonzero_i]
            Y_new = self.y[nonzero_i]
            current_i = np.where(nonzero_i[0]==i)[0][0]
            if constant:
                ones = np.ones(X_new.shape[0]).reshape((-1,1))
                X_new = np.hstack([ones,X_new])
            #Using OLS for Gaussian
            if isinstance(self.family, Gaussian):
                X_new = X_new * np.sqrt(wi)
                Y_new = Y_new * np.sqrt(wi)
                inv_xtx_xt = np.dot(np.linalg.inv(np.dot(X_new.T,X_new)),X_new.T)
                hat = np.dot(X_new[current_i],inv_xtx_xt[:,current_i])
                yhat = np.sum(np.dot(X_new,inv_xtx_xt[:,current_i]).reshape(-1,1)*Y_new)
                err = Y_new[current_i][0]-yhat
                RSS += err*err
                trS += hat
                CV_score += (err/(1-hat))**2
            #Using IWLS for GLMs
            elif isinstance(self.family, (Poisson, Binomial)):
                rslt = iwls(Y_new, X_new, self.family, self.offset[nonzero_i], None, ini_params, tol, max_iter, wi=wi)
                xtx_inv_xt = rslt[5]
                current_i = np.where(wi==1)[0]
                hat = np.dot(X_new[current_i],xtx_inv_xt[:,current_i])[0][0]*rslt[3][current_i][0][0]
                yhat = rslt[1][current_i][0][0]
                err = Y_new[current_i][0][0]-yhat
                trS += hat
                dev += self.family.resid_dev(Y_new[current_i][0][0], yhat)**2
        
        if isinstance(self.family, Gaussian):
            ll = -np.log(RSS)*n/2 - (1+np.log(np.pi/n*2))*n/2 #log likelihood
            aic = -2*ll + 2.0 * (trS + 1)
            aicc = -2.0*ll + 2.0*n*(trS + 1.0)/(n - trS - 2.0)
            bic = -2*ll + (trS+1) * np.log(n)
            cv = CV_score/n
        elif isinstance(self.family, (Poisson, Binomial)):
            aic = dev + 2.0 * trS
            aicc = aic + 2.0 * trS * (trS + 1.0)/(n - trS - 1.0)
            bic = dev + trS * np.log(n)
            cv = None

        return {'AICc': aicc,'AIC':aic, 'BIC': bic,'CV': cv}


    def _bw(self):
        gwr_func = lambda bw: self._fast_fit(bw,constant=self.constant)[self.criterion]
        if self.search == 'golden_section':
            a,c = self._init_section(self.X_glob, self.X_loc, self.coords,self.constant)
            delta = 0.38197 #1 - (np.sqrt(5.0)-1.0)/2.0
            self.bw = golden_section(a, c, delta, gwr_func, self.tol,self.max_iter, self.int_score)
        elif self.search == 'interval':
            self.bw = equal_interval(self.bw_min, self.bw_max, self.interval,gwr_func, self.int_score)
        else:
            raise TypeError('Unsupported computational search method ', search)

    def _init_section(self, X_glob, X_loc, coords, constant):
        if len(X_glob) > 0:
            n_glob = X_glob.shape[1]
        else:
            n_glob = 0
        if len(X_loc) > 0:
            n_loc = X_loc.shape[1]
        else:
            n_loc = 0
        if constant:
            n_vars = n_glob + n_loc + 1
        else:
            n_vars = n_glob + n_loc
        n = np.array(coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            nn = 40 + 2 * n_vars
            sq_dists = squareform(pdist(coords))
            sort_dists = np.sort(sq_dists, axis=1)
            min_dists = sort_dists[:,nn-1]
            max_dists = sort_dists[:,-1]
            a = np.min(min_dists)/2.0
            c = np.max(max_dists)/2.0

        if a < self.bw_min:
            a = self.bw_min
        if c > self.bw_max and self.bw_max > 0:
            c = self.bw_max
        return a, c
