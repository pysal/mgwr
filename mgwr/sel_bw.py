# GWR Bandwidth selection class

# x_glob parameter does not yet do anything; it is for semi-parametric

__author__ = "Taylor Oshan Tayoshan@gmail.com"

import spreg.user_output as USER
import numpy as np
import multiprocessing as mp
from scipy.spatial.distance import pdist
from scipy.optimize import minimize_scalar
from spglm.family import Gaussian, Poisson, Binomial
from .kernels import Kernel, local_cdist
from .gwr import GWR
from .search import golden_section, equal_interval, multi_bw
from .diagnostics import get_AICc, get_AIC, get_BIC, get_CV

getDiag = {'AICc': get_AICc, 'AIC': get_AIC, 'BIC': get_BIC, 'CV': get_CV}


class Sel_BW(object):
    """Select bandwidth for kernel

    Methods: p211 - p213, bandwidth selection

    :cite:`fotheringham_geographically_2002`: Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Args:
        coords (list[tuple]): (x,y) of points used in bandwidth selection
        y (np.array): n*1, dependent variable
        X_loc (np.array): n*k2, local independent variable, including intercept
        X_glob (np.array, optional): n*k1, fixed independent variable. Defaults to None.
        family (Gaussian, optional): function object/instance, underlying probability model:
                                    Gaussian(), Poisson(), Binomial(). Defaults to Gaussian().
        offset (np.array, optional): n*1, the offset variable at the ith location. For Poisson
                                    model this term is often the size of the population at risk or the expected size of the outcome in spatial epidemiology. Default is None where Ni becomes 1.0 for all locations
        kernel (str, optional): kernel function: 'gaussian', 'bisquare', 'exponential'.
                                Defaults to 'bisquare'.
        fixed (bool, optional): True for fixed bandwidth and False for adaptive (NN).
                                Defaults to False.
        multi (bool, optional): True for multiple (covaraite-specific) bandwidths. False for a
                                traditional (same for  all covariates) bandwdith.
                                Defaults to False.
        constant (bool, optional): True to include intercept (default) in model and
                                    False to exclude intercept. Defaults to True.
        spherical (bool, optional): True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).
                    Defaults to False.
    Examples:
        >>> import libpysal as ps
        >>> from mgwr.sel_bw import Sel_BW
        >>> data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
        >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
        >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
        >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
        >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
        >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
        >>> X = np.hstack([rural, pov, african_amer])

        Golden section search AICc - adaptive bisquare

        >>> bw = Sel_BW(coords, y, X).search(criterion='AICc')
        >>> print(bw)
        93.0

        Golden section search AIC - adaptive Gaussian

        >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='AIC')
        >>> print(bw)
        50.0

        Golden section search BIC - adaptive Gaussian

        >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='BIC')
        >>> print(bw)
        62.0

        Golden section search CV - adaptive Gaussian

        >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='CV')
        >>> print(bw)
        68.0

        Interval AICc - fixed bisquare

        >>> sel = Sel_BW(coords, y, X, fixed=True)
        >>> bw = sel.search(search_method='interval', bw_min=211001.0, bw_max=211035.0, interval=2)
        >>> print(bw)
        211025.0
    """

    def __init__(self,
                 coords: list[tuple],
                 y: np.array,
                 X_loc: np.array,
                 X_glob: np.array = None,
                 family: Gaussian = Gaussian(),
                 offset: np.array = None,
                 kernel: str = 'bisquare',
                 fixed: bool = False,
                 multi: bool = False,
                 constant: bool = True,
                 spherical: bool = False) -> None:

        self.coords = np.array(coords)
        self.y = y
        self.X_loc = X_loc
        self.X_glob = X_glob if X_glob is not None else []
        self.family = family
        self.fixed = fixed
        self.kernel = kernel
        self.offset = np.ones((len(y), 1)) if offset is None else offset * 1.0
        self.multi = multi
        self._functions = []
        self.constant = constant
        self.spherical = spherical
        self.search_params = {}

    def search(self,
               search_method: str = 'golden_section',
               criterion: str = 'AICc',
               bw_min: float = None,
               bw_max: float = None,
               interval: float = 0.0,
               tol: float = 1.0e-6,
               max_iter: int = 200,
               init_multi: float = None,
               tol_multi: float = 1.0e-5,
               rss_score: bool = False,
               max_iter_multi: int = 200,
               multi_bw_min: list = [None],
               multi_bw_max: list = [None],
               bws_same_times: int = 5,
               pool: mp.Pool = None,
               verbose: bool = False) -> np.array | float:
        """Method to select one unique bandwidth for a gwr model or a bandwidth vector for a mgwr model.

        Args:
            search_method (str, optional): bandwidth search method: 'golden_selection', 'interval', 'scipy'.
                                           Defaults to 'golden_section'.
            criterion (str, optional): bandwidth selection criterion: 'AICc', 'AIC', 'BIC', 'CV'. Defaults to 'AICc'.
            bw_min (float, optional): minimum value used in bandwidth search. Defaults to None.
            bw_max (float, optional): maximum value used in bandwidth search. Defaults to None.
            interval (float, optional): interval increment used in interval search. Defaults to 0.0.
            tol (float, optional): tolerance used to determine convergence. Defaults to 1.0e-6.
            max_iter (int, optional): max iterations if no convergence to tol. Defaults to 200.
            init_multi (float, optional): None (default) to initialize MGWR with a bandwidth
                         derived from GWR. Otherwise this option will choose the
                         bandwidth to initialize MGWR with.. Defaults to None.
            tol_multi (float, optional): convergence tolerence for the multiple bandwidth
                         backfitting algorithm; a larger tolerance may stop the
                         algorith faster though it may result in a less optimal
                         model. Defaults to 1.0e-5.
            rss_score (bool, optional): True to use the residual sum of sqaures to evaluate
                         each iteration of the multiple bandwidth backfitting
                         routine and False to use a smooth function. Defaults to False.
            max_iter_multi (int, optional): max iterations if no convergence to tol for multiple
                         bandwidth backfitting algorithm. Defaults to 200.
            multi_bw_min (list, optional): min values used for each covariate in mgwr bandwidth search.
                         Must be either a single value or have one value for
                         each covariate including the intercept. Defaults to [None].
            multi_bw_max (list, optional): max values used for each covariate in mgwr bandwidth
                         search. Must be either a single value or have one value
                         for each covariate including the intercept. Defaults to [None].
            bws_same_times (int, optional): If bandwidths keep the same between iterations for
                         bws_same_times (default 5) in backfitting, then use the
                         current set of bandwidths as final bandwidths. Defaults to 5.
            pool (mp.Pool, optional): A multiprocessing Pool object to enbale parallel fitting. Defaults to None.
            verbose (bool, optional): If true, bandwidth searching history is printed out. Defaults to False.

        Raises:
            AttributeError: if multi_bw_min is not a list of length 1 or k, where k is the number of covariates,
                            the attribute error is raised. "multi_bw_min must be either a list containing
                            a single entry or a list containing an entry for each of k covariates including the intercept"
            AttributeError: if multi_bw_max is not a list of length 1 or k, where k is the number of covariates,
                            the attribute error is raised. "multi_bw_max must be either a list containing
                            a single entry or a list containing an entry for each of k covariates including the intercept"

        Returns:
            np.array | float: optimal bandwidth value or values; returns scalar for
                         multi=False and array for multi=True; ordering of bandwidths
                         matches the ordering of the covariates (columns) of the
                         designs matrix, X
        """

        k = self.X_loc.shape[1]
        if self.constant:  # k is the number of covariates
            k += 1
        self.search_method = search_method
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.bws_same_times = bws_same_times
        self.pool = pool
        self.verbose = verbose

        if len(multi_bw_min) == k:
            self.multi_bw_min = multi_bw_min
        elif len(multi_bw_min) == 1:
            self.multi_bw_min = multi_bw_min * k
        else:
            raise AttributeError(
                "multi_bw_min must be either a list containing"
                " a single entry or a list containing an entry for each of k"
                " covariates including the intercept")

        if len(multi_bw_max) == k:
            self.multi_bw_max = multi_bw_max
        elif len(multi_bw_max) == 1:
            self.multi_bw_max = multi_bw_max * k
        else:
            raise AttributeError(
                "multi_bw_max must be either a list containing"
                " a single entry or a list containing an entry for each of k"
                " covariates including the intercept")

        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter
        self.init_multi = init_multi
        self.tol_multi = tol_multi
        self.rss_score = rss_score
        self.max_iter_multi = max_iter_multi
        self.search_params['search_method'] = search_method
        self.search_params['criterion'] = criterion
        self.search_params['bw_min'] = bw_min
        self.search_params['bw_max'] = bw_max
        self.search_params['interval'] = interval
        self.search_params['tol'] = tol
        self.search_params['max_iter'] = max_iter
        # self._check_min_max()

        self.int_score = not self.fixed

        if self.multi:
            self._mbw()
            self.params = self.bw[3]  # params n by k
            self.sel_hist = self.bw[-2]  # bw searching history
            self.bw_init = self.bw[-1]  # scalar, optimal bw from initial gwr model
        else:
            self._bw()
            self.sel_hist = self.bw[-1]

        self.pool = None
        return self.bw[0]

    def _bw(self):
        # define objective function for optimizer
        gwr_func = lambda bw: getDiag[self.criterion](GWR(
            self.coords, self.y, self.X_loc, bw, family=self.family, kernel=
            self.kernel, fixed=self.fixed, constant=self.constant, offset=self.
            offset, spherical=self.spherical).fit(lite=True, pool=self.pool))

        self._optimized_function = gwr_func

        if self.search_method == 'golden_section':
            a, c = self._init_section(self.X_glob, self.X_loc, self.coords,
                                      self.constant)
            delta = 0.38197  # 1 - (np.sqrt(5.0)-1.0)/2.0
            self.bw = golden_section(a, c, delta, gwr_func, self.tol,
                                     self.max_iter, self.bw_max, self.int_score,
                                     self.verbose)
        elif self.search_method == 'interval':
            self.bw = equal_interval(self.bw_min, self.bw_max, self.interval,
                                     gwr_func, self.int_score, self.verbose)
        elif self.search_method == 'scipy':
            self.bw_min, self.bw_max = self._init_section(
                self.X_glob, self.X_loc, self.coords, self.constant)
            if self.bw_min == self.bw_max:
                raise Exception(
                    'Maximum bandwidth and minimum bandwidth must be distinct for scipy optimizer.'
                )
            self._optimize_result = minimize_scalar(
                gwr_func, bounds=(self.bw_min, self.bw_max), method='bounded')
            self.bw = [self._optimize_result.x, self._optimize_result.fun, []]
        else:
            raise TypeError('Unsupported computational search method ',
                            self.search_method)

    def _mbw(self):
        if self.constant:
            X, keep_x, warn = USER.check_constant(self.X_loc)
        else:
            X = self.X_loc

        y = self.y
        n, k = X.shape

        def gwr_func(y, X, bw):
            return GWR(self.coords,
                       y,
                       X,
                       bw,
                       family=self.family,
                       kernel=self.kernel,
                       fixed=self.fixed,
                       offset=self.offset,
                       constant=False,
                       spherical=self.spherical,
                       hat_matrix=False).fit(lite=True, pool=self.pool)

        def bw_func(y, X):
            selector = Sel_BW(self.coords,
                              y,
                              X,
                              X_glob=[],
                              family=self.family,
                              kernel=self.kernel,
                              fixed=self.fixed,
                              offset=self.offset,
                              constant=False, spherical=self.spherical)
            return selector

        def sel_func(bw_func, bw_min=None, bw_max=None):
            return bw_func.search(
                search_method=self.search_method,
                criterion=self.criterion,
                bw_min=self.bw_min,
                bw_max=self.bw_max,
                interval=self.interval,
                tol=self.tol,
                max_iter=self.max_iter,
                pool=self.pool, verbose=False)

        self.bw = multi_bw(self.init_multi,
                           y,
                           X,
                           n,
                           k,
                           self.family,
                           self.tol_multi,
                           self.max_iter_multi,
                           self.rss_score,
                           gwr_func,
                           bw_func,
                           sel_func,
                           self.multi_bw_min,
                           self.multi_bw_max,
                           self.bws_same_times,
                           self.verbose)

    def _init_section(self, X_glob, X_loc, coords, constant):

        n_glob = X_glob.shape[1] if len(X_glob) > 0 else 0

        n_loc = X_loc.shape[1] if len(X_loc) > 0 else 0

        n_vars = n_glob + n_loc + 1 if constant else n_glob + n_loc

        n = np.array(coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            min_dist = np.min(np.array(
                [np.min(np.delete(local_cdist(coords[i], coords, spherical=self.spherical), i)) for i in range(n)]))

            max_dist = np.max(np.array([np.max(
                local_cdist(coords[i], coords, spherical=self.spherical)) for i in range(n)]))

            a = min_dist / 2.0
            c = max_dist * 2.0

        if self.bw_min is not None:
            a = self.bw_min

        if self.bw_max is not None and self.bw_max is not np.inf:
            c = self.bw_max

        return a, c
