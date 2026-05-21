"""
Numba-accelerated kernels for GWR batch fitting.

Public API
----------
HAS_NUMBA : bool
compute_all_kernel_weights(D, bw, fixed, kernel) -> W, bandwidths
gwr_fit_lite(X, y, W) -> params, influ, predy, resid
gwr_fit_full(X, y, W) -> params, influ, predy, resid, tr_STS_per_obs, CCT
euclidean_dist_matrix(coords) -> D

Design notes
------------
* Single factorization: ``np.linalg.inv(XtWX)`` is computed once per location
  and used for both beta estimation and influence calculation.  The previous
  approach called ``np.linalg.solve`` then ``np.linalg.inv`` separately —
  two full Cholesky decompositions of the same matrix.
* Parallelism via ``prange`` over observations inside ``@njit(parallel=True)``;
  no joblib dispatch overhead.
* Falls back gracefully when Numba is unavailable.
"""

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # noqa: E301
        def decorator(func):
            return func
        return decorator

    prange = range


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def euclidean_dist_matrix(coords):
    """Full Euclidean distance matrix.

    Parameters
    ----------
    coords : (n, 2) float64

    Returns
    -------
    D : (n, n) float64
    """
    n = len(coords)
    D = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            D[i, j] = d
            D[j, i] = d
    return D


# ---------------------------------------------------------------------------
# Haversine distance (numpy fallback — scipy.cdist dropped this metric)
# ---------------------------------------------------------------------------

def haversine_dist_matrix_numpy(coords):
    """Haversine (great-circle) distance matrix in kilometres.

    Parameters
    ----------
    coords : (n, 2) float64  — columns are (longitude, latitude) in degrees.

    Returns
    -------
    D : (n, n) float64
    """
    R = 6371.0
    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Kernel weight matrix
# ---------------------------------------------------------------------------

def compute_all_kernel_weights(D, bw, fixed=False, kernel='bisquare'):
    """Compute the full (n, n) kernel weight matrix from a pre-computed
    distance matrix.

    Parameters
    ----------
    D : (n, n) float64
        Pairwise distance matrix.
    bw : float or int
        Bandwidth — distance threshold when ``fixed=True``, number of
        nearest neighbours when ``fixed=False``.
    fixed : bool
    kernel : {'bisquare', 'gaussian', 'exponential'}

    Returns
    -------
    W : (n, n) float64
        W[i, :] contains the kernel weights for location i.
    bandwidths : (n,) float64
        Effective per-location bandwidth (identical for all i when fixed).
    """
    n = D.shape[0]

    # bw may arrive as a 0-d or 1-element array from scipy optimizers
    bw = float(np.asarray(bw).flat[0])

    if fixed:
        bandwidths = np.full(n, bw)
    else:
        k = int(bw)
        # k-th nearest-neighbour distance (row-wise partial sort, O(n) per row)
        bandwidths = np.partition(D, k - 1, axis=1)[:, k - 1] * 1.0000001

    Z = D / bandwidths.reshape(-1, 1)

    if kernel == 'bisquare':
        W = (1.0 - Z ** 2) ** 2
        W[Z >= 1.0] = 0.0
    elif kernel == 'gaussian':
        W = np.exp(-0.5 * Z ** 2)
    elif kernel == 'exponential':
        W = np.exp(-Z)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. "
                         "Choose from 'bisquare', 'gaussian', 'exponential'.")

    return W, bandwidths


# ---------------------------------------------------------------------------
# Batch GWR fit — lite (bandwidth selection)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True, fastmath=True)
def _gwr_fit_lite_numba(X, y, W):
    n, k = X.shape
    params = np.empty((n, k), dtype=np.float64)
    influ  = np.empty(n, dtype=np.float64)
    predy  = np.empty(n, dtype=np.float64)
    resid  = np.empty(n, dtype=np.float64)

    for i in prange(n):
        wi = W[i, :]

        # Build X'WX  (k×k) and X'Wy  (k,)
        XtWX = np.zeros((k, k), dtype=np.float64)
        XtWy = np.zeros(k, dtype=np.float64)
        for m in range(n):
            w_m = wi[m]
            y_m = y[m]
            for j in range(k):
                XtWy[j] += X[m, j] * w_m * y_m
                for l in range(k):
                    XtWX[j, l] += X[m, j] * w_m * X[m, l]

        # Single factorisation — used for both betas and influence
        XtWX_inv = np.linalg.inv(XtWX)

        # beta = XtWX_inv @ XtWy
        beta = np.zeros(k, dtype=np.float64)
        for j in range(k):
            for l in range(k):
                beta[j] += XtWX_inv[j, l] * XtWy[l]
        params[i, :] = beta

        pred = 0.0
        for j in range(k):
            pred += X[i, j] * beta[j]
        predy[i] = pred
        resid[i] = y[i] - pred

        # influence: xi @ XtWX_inv @ xi
        xi = X[i, :]
        tmp = np.zeros(k, dtype=np.float64)
        for j in range(k):
            for l in range(k):
                tmp[j] += XtWX_inv[j, l] * xi[l]
        infl = 0.0
        for j in range(k):
            infl += xi[j] * tmp[j]
        influ[i] = infl

    return params, influ, predy, resid


# ---------------------------------------------------------------------------
# Batch GWR fit — full diagnostics
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True, fastmath=True)
def _gwr_fit_full_numba(X, y, W):
    """Batch GWR fit returning full diagnostics.

    Returns
    -------
    params   : (n, k)
    influ    : (n,)
    predy    : (n,)
    resid    : (n,)
    tr_STS   : (n,)   per-observation tr(S'S) contribution; sum to get scalar
    CCT      : (n, k) diagonal of inv_xtx_xt @ inv_xtx_xt.T per location
    """
    n, k = X.shape
    params  = np.empty((n, k), dtype=np.float64)
    influ   = np.empty(n, dtype=np.float64)
    predy   = np.empty(n, dtype=np.float64)
    resid   = np.empty(n, dtype=np.float64)
    tr_STS  = np.empty(n, dtype=np.float64)
    CCT     = np.empty((n, k), dtype=np.float64)

    for i in prange(n):
        wi = W[i, :]

        # Build X'WX, X'Wy, and X'W (k×n)
        XtWX = np.zeros((k, k), dtype=np.float64)
        XtWy = np.zeros(k, dtype=np.float64)
        XtW  = np.zeros((k, n), dtype=np.float64)
        for m in range(n):
            w_m = wi[m]
            y_m = y[m]
            for j in range(k):
                XtW[j, m]   = X[m, j] * w_m
                XtWy[j]    += X[m, j] * w_m * y_m
                for l in range(k):
                    XtWX[j, l] += X[m, j] * w_m * X[m, l]

        XtWX_inv = np.linalg.inv(XtWX)

        # beta
        beta = np.zeros(k, dtype=np.float64)
        for j in range(k):
            for l in range(k):
                beta[j] += XtWX_inv[j, l] * XtWy[l]
        params[i, :] = beta

        pred = 0.0
        for j in range(k):
            pred += X[i, j] * beta[j]
        predy[i] = pred
        resid[i] = y[i] - pred

        # inv_xtx_xt = XtWX_inv @ XtW  (k×n)
        inv_xtx_xt = np.zeros((k, n), dtype=np.float64)
        for j in range(k):
            for m in range(n):
                for l in range(k):
                    inv_xtx_xt[j, m] += XtWX_inv[j, l] * XtW[l, m]

        # influence: xi @ inv_xtx_xt[:, i]
        xi = X[i, :]
        infl = 0.0
        for j in range(k):
            infl += xi[j] * inv_xtx_xt[j, i]
        influ[i] = infl

        # Si = xi @ inv_xtx_xt  (n,)  — hat matrix row
        Si = np.zeros(n, dtype=np.float64)
        for m in range(n):
            for j in range(k):
                Si[m] += xi[j] * inv_xtx_xt[j, m]
        tr_STS[i] = np.dot(Si, Si)

        # CCT[i, :] = diag(inv_xtx_xt @ inv_xtx_xt.T)
        for j in range(k):
            CCT[i, j] = np.dot(inv_xtx_xt[j, :], inv_xtx_xt[j, :])

    return params, influ, predy, resid, tr_STS, CCT


# ---------------------------------------------------------------------------
# Public wrappers (handle fallback when Numba is absent)
# ---------------------------------------------------------------------------

def gwr_fit_lite(X, y, W):
    """Lite GWR fit — returns only the quantities needed for bandwidth selection.

    Falls back to a NumPy loop when Numba is unavailable.
    """
    if HAS_NUMBA:
        return _gwr_fit_lite_numba(X, y.ravel(), W)
    return _gwr_fit_lite_numpy(X, y.ravel(), W)


def gwr_fit_full(X, y, W):
    """Full GWR fit — returns params, influ, predy, resid, tr_STS_per_obs, CCT.

    Falls back to a NumPy loop when Numba is unavailable.
    """
    if HAS_NUMBA:
        return _gwr_fit_full_numba(X, y.ravel(), W)
    return _gwr_fit_full_numpy(X, y.ravel(), W)


# ---------------------------------------------------------------------------
# NumPy fallbacks
# ---------------------------------------------------------------------------

def _gwr_fit_lite_numpy(X, y, W):
    n, k = X.shape
    params = np.empty((n, k))
    influ  = np.empty(n)
    predy  = np.empty(n)
    resid  = np.empty(n)

    for i in range(n):
        wi = W[i, :].reshape(-1, 1)
        XtWX = X.T @ (X * wi)
        XtWy = X.T @ (wi * y.reshape(-1, 1))
        XtWX_inv = np.linalg.inv(XtWX)
        beta = (XtWX_inv @ XtWy).ravel()
        params[i] = beta
        predy[i]  = X[i] @ beta
        resid[i]  = y[i] - predy[i]
        influ[i]  = float(X[i] @ XtWX_inv @ X[i])

    return params, influ, predy, resid


def _gwr_fit_full_numpy(X, y, W):
    n, k = X.shape
    params  = np.empty((n, k))
    influ   = np.empty(n)
    predy   = np.empty(n)
    resid   = np.empty(n)
    tr_STS  = np.empty(n)
    CCT     = np.empty((n, k))

    for i in range(n):
        wi = W[i, :].reshape(-1, 1)
        XtW       = (X * wi).T                        # (k, n)
        XtWX      = XtW @ X                           # (k, k)
        XtWy      = XtW @ y.reshape(-1, 1)            # (k, 1)
        XtWX_inv  = np.linalg.inv(XtWX)
        beta      = (XtWX_inv @ XtWy).ravel()
        params[i] = beta
        predy[i]  = X[i] @ beta
        resid[i]  = y[i] - predy[i]

        inv_xtx_xt = XtWX_inv @ XtW                  # (k, n)
        influ[i]   = float(X[i] @ inv_xtx_xt[:, i])
        Si         = X[i] @ inv_xtx_xt               # (n,)
        tr_STS[i]  = float(Si @ Si)
        CCT[i]     = np.einsum('jm,jm->j', inv_xtx_xt, inv_xtx_xt)

    return params, influ, predy, resid, tr_STS, CCT
